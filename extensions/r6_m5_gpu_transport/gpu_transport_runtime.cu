// R6-M5: job-level GPU scheduler -> packer -> real MSCCL++ MemoryChannel.
// The M4 source is not modified. This runtime consumes the same common IR and
// preserves its deterministic action semantics, then stops after remote decode.

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "rlccl/scheduler/common/scheduler_ir.h"
#include "rlccl/transport/cuda/gpu_packer.cuh"
#include "rlccl/transport/cuda/gpu_transport_ir.h"

namespace {

using namespace rlccl::scheduler;
using namespace rlccl::transport::cuda_backend;

thread_local std::string last_error;
constexpr uint64_t kChecksumModulus = (1ULL << 63) - 25;
__device__ __constant__ uint64_t kChecksumCoefficients[8] = {
    1000003, 1000033, 1000037, 1000081,
    1000099, 1000117, 1000121, 1000133};

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status_)); \
  } while (0)

struct DeviceRevealQueue {
  RevealRecord* records;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

struct DeviceActionQueue {
  CommittedAction* actions;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

struct DevicePipelineState {
  uint64_t* revealed_count;
  uint64_t* committed_count;
  uint32_t* descriptor_epoch;
  uint32_t last_reveal_epoch;
  uint32_t has_reveal_epoch;
  uint64_t next_action_id;
  uint32_t scheduler_done;
  uint32_t transport_done;
};

struct DeviceTraceState {
  TransportTrace* traces;
  uint64_t* wait_times;
  uint32_t trace_capacity;
  uint32_t trace_count;
};

__device__ __forceinline__ uint64_t global_time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint64_t add_mod(uint64_t left, uint64_t right) {
  return left >= kChecksumModulus - right ? left - (kChecksumModulus - right) : left + right;
}

__device__ uint64_t multiply_mod(uint64_t value, uint64_t coefficient) {
  uint64_t output = 0;
  value %= kChecksumModulus;
  while (coefficient) {
    if (coefficient & 1) output = add_mod(output, value);
    value = add_mod(value, value);
    coefficient >>= 1;
  }
  return output;
}

__device__ uint64_t metadata_checksum(const int64_t* prefix) {
  uint64_t output = 0;
  for (int index = 0; index < 8; ++index) {
    uint64_t value = static_cast<uint64_t>(prefix[index]) % kChecksumModulus;
    output = add_mod(output, multiply_mod(value, kChecksumCoefficients[index]));
  }
  return output;
}

__device__ uint64_t plan_checksum(const uint8_t* bytes, uint64_t size) {
  uint64_t value = 0xCBF29CE484222325ULL;
  for (uint64_t index = 0; index < size; ++index) {
    uint8_t byte = (index >= 24 && index < 32) ? 0 : bytes[index];
    value ^= byte;
    value *= 0x100000001B3ULL;
  }
  return value;
}

__device__ __forceinline__ const CompiledRouteTemplate* routes(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const CompiledRouteTemplate*>(blob + header->route_templates_offset);
}

__device__ __forceinline__ const int32_t* pair_routes(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const int32_t*>(blob + header->rank_pair_to_route_offset);
}

__device__ void router_publish_role(
    const int64_t* input_records,
    uint32_t record_count,
    const int32_t* destination_ranks,
    const int64_t* expert_ids,
    const int64_t* token_ids,
    const int64_t* feature_digests,
    int64_t* metadata,
    uint32_t total_assignments,
    uint32_t source_rank,
    uint32_t world_size,
    DeviceRevealQueue* queue,
    RevealRecord* records_by_descriptor,
    PipelineTiming* timing,
    uint64_t* final_router_ns,
    uint64_t delay_cycles,
    TransportCounters* counters) {
  if (threadIdx.x != 0) return;
  for (uint32_t input = 0; input < record_count; ++input) {
    if (input && delay_cycles) {
      uint64_t start = clock64();
      while (clock64() - start < delay_cycles) __nanosleep(64);
    }
    const int64_t* row = input_records + static_cast<uint64_t>(input) * 8;
    RevealRecord reveal{
        static_cast<uint32_t>(row[0]), static_cast<uint32_t>(row[1]),
        static_cast<uint32_t>(row[2]), static_cast<uint32_t>(row[3]),
        static_cast<uint32_t>(row[4]), static_cast<uint32_t>(row[5]),
        static_cast<uint32_t>(row[6]), static_cast<uint32_t>(row[7])};
    if (static_cast<uint64_t>(reveal.assignment_begin) + reveal.assignment_count > total_assignments) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&counters->errors), 1ULL);
      return;
    }
    for (uint32_t local = 0; local < reveal.assignment_count; ++local) {
      uint32_t assignment = reveal.assignment_begin + local;
      int64_t* output = metadata + static_cast<uint64_t>(assignment) * 9;
      output[0] = token_ids[assignment];
      output[1] = source_rank;
      output[2] = destination_ranks[assignment];
      output[3] = expert_ids[assignment];
      output[4] = reveal.chunk_id;
      output[5] = local;
      output[6] = reveal.token_begin + local;
      output[7] = feature_digests[assignment];
      output[8] = static_cast<int64_t>(metadata_checksum(output));
      if (destination_ranks[assignment] != expert_ids[assignment] % static_cast<int64_t>(world_size)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&counters->errors), 1ULL);
        return;
      }
    }
    uint64_t reveal_time = global_time_ns();
    timing[input].t0_router_reveal = reveal_time;
    timing[input].chunk_id = reveal.chunk_id;
    timing[input].descriptor_id = reveal.descriptor_id;
    records_by_descriptor[reveal.descriptor_id] = reveal;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> head_ref(queue->head);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_ref(queue->tail);
    uint32_t tail;
    while (true) {
      tail = tail_ref.load(cuda::memory_order_relaxed);
      uint32_t head = head_ref.load(cuda::memory_order_acquire);
      if (tail - head < queue->capacity) break;
      __nanosleep(64);
    }
    queue->records[tail % queue->capacity] = reveal;
    tail_ref.store(tail + 1, cuda::memory_order_release);
    if (input + 1 == record_count) {
      *final_router_ns = reveal_time;
      __threadfence();
    }
  }
}

__device__ void frozen_scheduler_progress_role(
    const uint8_t* blob,
    const int32_t* destination_ranks,
    uint32_t total_assignments,
    uint32_t source_rank,
    uint32_t expected_reveals,
    DeviceRevealQueue* reveal_queue,
    DeviceActionQueue* action_queue,
    DevicePipelineState* state,
    PipelineTiming* timing,
    TransportCounters* counters) {
  if (threadIdx.x != 0) return;
  const auto* header = reinterpret_cast<const CompiledPlanBlobHeader*>(blob);
  if (header->magic != kCompiledPlanMagic || header->version != kCompiledPlanVersion ||
      plan_checksum(blob, header->total_bytes) != header->checksum) {
    counters->errors += 1;
    state->scheduler_done = 1;
    __threadfence();
    return;
  }
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_head(reveal_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_tail(reveal_queue->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> action_head(action_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> action_tail(action_queue->tail);
  uint32_t processed = 0;
  uint64_t last_progress = global_time_ns();
  while (processed < expected_reveals) {
    uint32_t head = reveal_head.load(cuda::memory_order_relaxed);
    uint32_t tail = reveal_tail.load(cuda::memory_order_acquire);
    if (head == tail) {
      if (global_time_ns() - last_progress > 5'000'000'000ULL) {
        counters->errors += 1; break;
      }
      __nanosleep(64); continue;
    }
    RevealRecord reveal = reveal_queue->records[head % reveal_queue->capacity];
    reveal_head.store(head + 1, cuda::memory_order_release);
    ++processed;
    last_progress = global_time_ns();
    if (reveal.descriptor_id >= header->max_descriptors || reveal.chunk_id >= header->max_chunks ||
        (state->has_reveal_epoch && reveal.reveal_epoch <= state->last_reveal_epoch) ||
        state->descriptor_epoch[reveal.descriptor_id] != UINT32_MAX ||
        static_cast<uint64_t>(reveal.assignment_begin) + reveal.assignment_count > total_assignments) {
      if (state->has_reveal_epoch && reveal.reveal_epoch <= state->last_reveal_epoch)
        counters->stale_action += 1;
      counters->errors += 1; continue;
    }
    uint32_t counts[kMaxWorldSize]{};
    bool invalid = false;
    for (uint32_t offset = 0; offset < reveal.assignment_count; ++offset) {
      int32_t dst = destination_ranks[reveal.assignment_begin + offset];
      if (dst < 0 || static_cast<uint32_t>(dst) >= header->world_size) {
        invalid = true; break;
      }
      ++counts[dst];
    }
    if (invalid) { counters->errors += 1; continue; }
    uint32_t candidates = 0;
    for (uint32_t dst = 0; dst < header->world_size; ++dst) candidates += counts[dst] != 0;
    uint32_t queue_head = action_head.load(cuda::memory_order_acquire);
    uint32_t queue_tail = action_tail.load(cuda::memory_order_relaxed);
    while (queue_tail - queue_head + candidates > action_queue->capacity) {
      queue_head = action_head.load(cuda::memory_order_acquire);
      __nanosleep(64);
    }
    uint32_t built = 0;
    for (uint32_t dst = 0; dst < header->world_size; ++dst) {
      uint32_t count = counts[dst];
      if (!count) continue;
      uint32_t pair = source_rank * header->world_size + dst;
      int32_t route_index = pair_routes(blob, header)[pair];
      if (route_index < 0 || static_cast<uint32_t>(route_index) >= header->route_count) {
        invalid = true; break;
      }
      const auto& route = routes(blob, header)[route_index];
      if (route.src_rank != static_cast<int32_t>(source_rank) ||
          route.dst_rank != static_cast<int32_t>(dst) || count > header->max_tokens_per_peer) {
        invalid = true; break;
      }
      CommittedAction action{
          state->next_action_id + built, reveal.descriptor_id, reveal.chunk_id,
          reveal.reveal_epoch, static_cast<int32_t>(source_rank), static_cast<int32_t>(dst),
          static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride + route.send_region_base,
          static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride + route.recv_region_base,
          count, 0, static_cast<uint64_t>(count) * header->record_bytes,
          static_cast<uint32_t>(route.route_id), route.flags};
      action_queue->actions[queue_tail % action_queue->capacity] = action;
      action_tail.store(++queue_tail, cuda::memory_order_release);
      state->revealed_count[pair] += count;
      state->committed_count[pair] += count;
      ++built;
    }
    if (invalid) { counters->errors += 1; continue; }
    state->next_action_id += built;
    state->descriptor_epoch[reveal.descriptor_id] = reveal.reveal_epoch;
    state->last_reveal_epoch = reveal.reveal_epoch;
    state->has_reveal_epoch = 1;
    counters->scheduler_actions += built;
    timing[reveal.descriptor_id].t1_scheduler_commit = global_time_ns();
  }
  __threadfence();
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done_ref(state->scheduler_done);
  done_ref.store(1, cuda::memory_order_release);
}

__device__ TransportErrorCode map_physical_action(
    const CommittedAction& action,
    const CompiledPlanBlobHeader* header,
    uint32_t source_rank,
    uint32_t* slot_claims,
    PhysicalTransportAction* physical) {
  if (action.src_rank != static_cast<int32_t>(source_rank) || action.dst_rank < 0 ||
      static_cast<uint32_t>(action.dst_rank) >= header->world_size) return kTransportInvalidRank;
  if (action.descriptor_id >= header->max_descriptors) return kTransportInvalidDescriptor;
  if (action.token_count == 0 || action.bytes == 0) return kTransportZeroSize;
  if (action.bytes != static_cast<uint64_t>(action.token_count) * header->record_bytes)
    return kTransportBytesMismatch;
  uint64_t peer_stride = header->descriptor_stride / header->world_size;
  uint64_t expected_src = static_cast<uint64_t>(action.descriptor_id) * header->descriptor_stride +
                          static_cast<uint64_t>(action.dst_rank) * peer_stride;
  uint64_t expected_dst = header->region_bytes +
                          static_cast<uint64_t>(action.descriptor_id) * header->descriptor_stride +
                          static_cast<uint64_t>(source_rank) * peer_stride;
  if (action.src_offset != expected_src || action.dst_offset != expected_dst)
    return kTransportOffsetMismatch;
  uint64_t physical_bytes = 8 + action.bytes;
  if (expected_src + physical_bytes > header->region_bytes ||
      expected_dst + physical_bytes > 2 * header->region_bytes ||
      expected_src % 8 != expected_dst % 8) return kTransportOffsetOverflow;
  uint32_t slot = action.descriptor_id * header->world_size + action.dst_rank;
  if (atomicCAS(slot_claims + slot, 0U, 1U) != 0U) return kTransportSlotReplay;
  *physical = PhysicalTransportAction{
      action.action_id, action.descriptor_id, action.src_rank, action.dst_rank,
      action.src_offset, action.dst_offset, expected_src, expected_dst,
      action.bytes, physical_bytes, action.token_count, action.route_id};
  return kTransportOk;
}

__device__ void gpu_transport_progress_role(
    mscclpp::MemoryChannelDeviceHandle handle,
    const uint8_t* blob,
    uint32_t source_rank,
    DeviceActionQueue* action_queue,
    DevicePipelineState* state,
    const RevealRecord* records_by_descriptor,
    DevicePackingInput packing,
    uint32_t* slot_claims,
    DeviceTraceState* trace_state,
    TransportCounters* counters) {
  __shared__ CommittedAction action;
  __shared__ PhysicalTransportAction physical;
  __shared__ uint32_t has_action;
  __shared__ uint32_t should_exit;
  __shared__ uint32_t error_code;
  __shared__ uint32_t trace_index;
  const auto* header = reinterpret_cast<const CompiledPlanBlobHeader*>(blob);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> head_ref(action_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_ref(action_queue->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done_ref(state->scheduler_done);
  while (true) {
    if (threadIdx.x == 0) {
      uint32_t head = head_ref.load(cuda::memory_order_relaxed);
      uint32_t tail = tail_ref.load(cuda::memory_order_acquire);
      has_action = head != tail;
      should_exit = !has_action && done_ref.load(cuda::memory_order_acquire);
      if (has_action) {
        action = action_queue->actions[head % action_queue->capacity];
        head_ref.store(head + 1, cuda::memory_order_release);
        trace_index = atomicAdd(&trace_state->trace_count, 1U);
        error_code = map_physical_action(action, header, source_rank, slot_claims, &physical);
        if (error_code == kTransportSlotReplay) counters->slot_replays += 1;
      }
    }
    __syncthreads();
    if (should_exit) break;
    if (!has_action) { __nanosleep(64); __syncthreads(); continue; }
    TransportTrace* trace = trace_index < trace_state->trace_capacity
        ? &trace_state->traces[trace_index] : nullptr;
    if (threadIdx.x == 0 && trace) {
      trace->action = physical;
      trace->t2_action_consumed = global_time_ns();
      trace->t3_pack_start = global_time_ns();
      trace->error_code = error_code;
      trace->is_remote = action.dst_rank != action.src_rank;
    }
    __syncthreads();
    if (error_code == kTransportOk) {
      auto pack_error = pack_committed_action(
          action, records_by_descriptor[action.descriptor_id], packing);
      if (threadIdx.x == 0) error_code = pack_error;
    }
    __syncthreads();
    if (threadIdx.x == 0 && trace) trace->t4_pack_end = global_time_ns();
    if (error_code == kTransportOk) {
      if (action.dst_rank != action.src_rank) {
        if (threadIdx.x == 0 && trace) trace->t5_put_start = global_time_ns();
        __syncthreads();
        handle.put<8>(physical.physical_dst_offset, physical.physical_src_offset,
                      physical.physical_bytes, threadIdx.x, blockDim.x);
        __syncthreads();
        if (threadIdx.x == 0) {
          handle.signal();
          if (trace) trace->t6_put_end = global_time_ns();
          counters->mscclpp_put_calls += 1;
          counters->mscclpp_bytes_transferred += physical.physical_bytes;
          counters->mscclpp_signals += 1;
        }
      } else {
        const uint64_t* source = reinterpret_cast<const uint64_t*>(
            packing.registered_buffer + physical.physical_src_offset);
        uint64_t* target = reinterpret_cast<uint64_t*>(
            packing.registered_buffer + physical.physical_dst_offset);
        uint64_t words = physical.physical_bytes / 8;
        for (uint64_t word = threadIdx.x; word < words; word += blockDim.x)
          target[word] = source[word];
        __syncthreads();
        __threadfence();
      }
    }
    if (threadIdx.x == 0) {
      if (trace) trace->error_code = error_code;
      counters->transport_actions += 1;
      counters->pack_calls += error_code == kTransportOk;
      counters->errors += error_code != kTransportOk;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    __threadfence();
    state->transport_done = 1;
  }
}

__device__ void remote_wait_progress_role(
    mscclpp::MemoryChannelDeviceHandle handle,
    uint32_t expected_remote_actions,
    uint64_t* wait_times,
    TransportCounters* counters) {
  if (threadIdx.x != 0) return;
  for (uint32_t index = 0; index < expected_remote_actions; ++index) {
    handle.wait(/*maxSpinCount=*/-1);
    wait_times[index] = global_time_ns();
    counters->mscclpp_waits += 1;
  }
}

// The transport role remains available as a standalone persistent kernel for
// integration users.  The formal two-rank gate below uses the fused job kernel
// so the four persistent roles do not depend on cross-stream launch fairness.
__global__ void gpu_transport_progress_kernel(
    mscclpp::MemoryChannelDeviceHandle handle,
    const uint8_t* blob,
    uint32_t source_rank,
    DeviceActionQueue* action_queue,
    DevicePipelineState* state,
    const RevealRecord* records_by_descriptor,
    DevicePackingInput packing,
    uint32_t* slot_claims,
    DeviceTraceState* trace_state,
    TransportCounters* counters) {
  gpu_transport_progress_role(
      handle, blob, source_rank, action_queue, state, records_by_descriptor,
      packing, slot_claims, trace_state, counters);
}

__global__ void gpu_transport_pipeline_kernel(
    const int64_t* input_records,
    uint32_t record_count,
    const int32_t* destination_ranks,
    const int64_t* expert_ids,
    const int64_t* token_ids,
    const int64_t* feature_digests,
    int64_t* metadata,
    uint32_t total_assignments,
    uint32_t source_rank,
    DeviceRevealQueue* reveal_queue,
    RevealRecord* records_by_descriptor,
    PipelineTiming* timing,
    uint64_t* final_router_ns,
    uint64_t delay_cycles,
    mscclpp::MemoryChannelDeviceHandle handle,
    const uint8_t* blob,
    DeviceActionQueue* action_queue,
    DevicePipelineState* state,
    DevicePackingInput packing,
    uint32_t* slot_claims,
    DeviceTraceState* trace_state,
    TransportCounters* counters,
    uint32_t expected_remote_actions,
    uint64_t* wait_times) {
  if (blockIdx.x == 0) {
    router_publish_role(
        input_records, record_count, destination_ranks, expert_ids, token_ids,
        feature_digests, metadata, total_assignments, source_rank, 2,
        reveal_queue, records_by_descriptor, timing, final_router_ns,
        delay_cycles, counters);
  } else if (blockIdx.x == 1) {
    frozen_scheduler_progress_role(
        blob, destination_ranks, total_assignments, source_rank, record_count,
        reveal_queue, action_queue, state, timing, counters);
  } else if (blockIdx.x == 2) {
    gpu_transport_progress_role(
        handle, blob, source_rank, action_queue, state, records_by_descriptor,
        packing, slot_claims, trace_state, counters);
  } else if (blockIdx.x == 3) {
    remote_wait_progress_role(handle, expected_remote_actions, wait_times, counters);
  }
}

template <typename Fn>
int protect(Fn&& fn) {
  try {
    fn(); last_error.clear(); return 0;
  } catch (const std::exception& error) {
    last_error = error.what(); return -1;
  } catch (...) {
    last_error = "unknown native exception"; return -1;
  }
}

template <typename T>
T* allocate_device(size_t count = 1) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, sizeof(T) * count));
  return pointer;
}

struct Runtime {
  int rank;
  int remote_rank;
  int device;
  CompiledPlanBlobHeader plan_header{};
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::unique_ptr<mscclpp::Communicator> communicator;
  mscclpp::RegisteredMemory local_memory;
  mscclpp::RegisteredMemory remote_memory;
  mscclpp::Connection connection;
  mscclpp::Semaphore semaphore;
  std::unique_ptr<mscclpp::MemoryChannel> channel;
  mscclpp::MemoryChannelDeviceHandle handle;
  uint8_t* registered_buffer;
  uint64_t registered_bytes;
  uint8_t* d_plan = nullptr;
  RevealRecord* d_reveal_records = nullptr;
  RevealRecord* d_records_by_descriptor = nullptr;
  CommittedAction* d_actions = nullptr;
  DeviceRevealQueue* d_reveal_queue = nullptr;
  DeviceActionQueue* d_action_queue = nullptr;
  DevicePipelineState* d_state = nullptr;
  DeviceTraceState* d_trace_state = nullptr;
  PipelineTiming* d_timing = nullptr;
  TransportTrace* d_traces = nullptr;
  TransportCounters* d_counters = nullptr;
  uint64_t* d_revealed_count = nullptr;
  uint64_t* d_committed_count = nullptr;
  uint32_t* d_descriptor_epoch = nullptr;
  uint32_t* d_slot_claims = nullptr;
  uint64_t* d_wait_times = nullptr;
  uint64_t* d_final_router_ns = nullptr;
  cudaStream_t producer_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  std::vector<CommittedAction> actions;
  std::vector<TransportTrace> traces;
  std::vector<PipelineTiming> timings;
  TransportCounters counters{};

  Runtime(int rank_, int device_, void* buffer, size_t bytes, const char* endpoint,
          const void* plan_blob, size_t plan_bytes)
      : rank(rank_), remote_rank(rank_ ^ 1), device(device_),
        registered_buffer(static_cast<uint8_t*>(buffer)), registered_bytes(bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    if (plan_bytes < sizeof(CompiledPlanBlobHeader)) throw std::runtime_error("M5 plan header truncated");
    std::memcpy(&plan_header, plan_blob, sizeof(plan_header));
    if (plan_header.magic != kCompiledPlanMagic || plan_header.world_size != 2 ||
        bytes != 2 * plan_header.region_bytes) throw std::runtime_error("M5 plan/buffer layout mismatch");
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, 2);
    bootstrap->initialize(endpoint);
    communicator = std::make_unique<mscclpp::Communicator>(bootstrap);
    constexpr auto transport = mscclpp::Transport::CudaIpc;
    connection = communicator->connect(
        {transport, {mscclpp::DeviceType::GPU, device}}, remote_rank).get();
    semaphore = communicator->buildSemaphore(connection, remote_rank).get();
    local_memory = communicator->registerMemory(buffer, bytes, transport);
    communicator->sendMemory(local_memory, remote_rank);
    remote_memory = communicator->recvMemory(remote_rank).get();
    channel = std::make_unique<mscclpp::MemoryChannel>(semaphore, remote_memory, local_memory);
    handle = channel->deviceHandle();

    uint32_t reveal_capacity = std::max<uint32_t>(plan_header.max_chunks, 8);
    uint32_t action_capacity = std::max<uint32_t>(plan_header.max_descriptors * 2, 16);
    uint32_t pairs = plan_header.world_size * plan_header.world_size;
    d_plan = allocate_device<uint8_t>(plan_bytes);
    d_reveal_records = allocate_device<RevealRecord>(reveal_capacity);
    d_records_by_descriptor = allocate_device<RevealRecord>(plan_header.max_descriptors);
    d_actions = allocate_device<CommittedAction>(action_capacity);
    d_reveal_queue = allocate_device<DeviceRevealQueue>();
    d_action_queue = allocate_device<DeviceActionQueue>();
    d_state = allocate_device<DevicePipelineState>();
    d_trace_state = allocate_device<DeviceTraceState>();
    d_timing = allocate_device<PipelineTiming>(plan_header.max_descriptors);
    d_traces = allocate_device<TransportTrace>(action_capacity);
    d_counters = allocate_device<TransportCounters>();
    d_revealed_count = allocate_device<uint64_t>(pairs);
    d_committed_count = allocate_device<uint64_t>(pairs);
    d_descriptor_epoch = allocate_device<uint32_t>(plan_header.max_descriptors);
    d_slot_claims = allocate_device<uint32_t>(plan_header.max_descriptors * plan_header.world_size);
    d_wait_times = allocate_device<uint64_t>(plan_header.max_descriptors);
    d_final_router_ns = allocate_device<uint64_t>();
    CUDA_CHECK(cudaMemcpy(d_plan, plan_blob, plan_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_reveal_records, 0, sizeof(RevealRecord) * reveal_capacity));
    CUDA_CHECK(cudaMemset(d_records_by_descriptor, 0, sizeof(RevealRecord) * plan_header.max_descriptors));
    CUDA_CHECK(cudaMemset(d_actions, 0, sizeof(CommittedAction) * action_capacity));
    CUDA_CHECK(cudaMemset(d_timing, 0, sizeof(PipelineTiming) * plan_header.max_descriptors));
    CUDA_CHECK(cudaMemset(d_traces, 0, sizeof(TransportTrace) * action_capacity));
    CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(TransportCounters)));
    CUDA_CHECK(cudaMemset(d_revealed_count, 0, sizeof(uint64_t) * pairs));
    CUDA_CHECK(cudaMemset(d_committed_count, 0, sizeof(uint64_t) * pairs));
    CUDA_CHECK(cudaMemset(d_descriptor_epoch, 0xFF, sizeof(uint32_t) * plan_header.max_descriptors));
    CUDA_CHECK(cudaMemset(d_slot_claims, 0, sizeof(uint32_t) * plan_header.max_descriptors * plan_header.world_size));
    CUDA_CHECK(cudaMemset(d_wait_times, 0, sizeof(uint64_t) * plan_header.max_descriptors));
    CUDA_CHECK(cudaMemset(d_final_router_ns, 0, sizeof(uint64_t)));
    DeviceRevealQueue reveal_queue{d_reveal_records, reveal_capacity, 0, 0};
    DeviceActionQueue action_queue{d_actions, action_capacity, 0, 0};
    DevicePipelineState state{
        d_revealed_count, d_committed_count, d_descriptor_epoch,
        0, 0, 0, 0, 0};
    DeviceTraceState trace_state{d_traces, d_wait_times, action_capacity, 0};
    CUDA_CHECK(cudaMemcpy(d_reveal_queue, &reveal_queue, sizeof(reveal_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_action_queue, &action_queue, sizeof(action_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state, &state, sizeof(state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_trace_state, &trace_state, sizeof(trace_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
  }

  ~Runtime() {
    if (input_ready) cudaEventDestroy(input_ready);
    if (producer_stream) cudaStreamDestroy(producer_stream);
    cudaFree(d_plan); cudaFree(d_reveal_records); cudaFree(d_records_by_descriptor);
    cudaFree(d_actions); cudaFree(d_reveal_queue); cudaFree(d_action_queue);
    cudaFree(d_state); cudaFree(d_trace_state); cudaFree(d_timing); cudaFree(d_traces);
    cudaFree(d_counters); cudaFree(d_revealed_count); cudaFree(d_committed_count);
    cudaFree(d_descriptor_epoch); cudaFree(d_slot_claims); cudaFree(d_wait_times);
    cudaFree(d_final_router_ns);
  }

  void run(
      const int64_t* reveal_records, uint32_t record_count,
      const int32_t* destination_ranks, const int64_t* expert_ids,
      const int64_t* token_ids, const int64_t* feature_digests,
      const float* features, int64_t* metadata, uint32_t total_assignments,
      uint32_t feature_width, uint32_t expected_remote_actions,
      uint64_t delay_cycles, cudaStream_t router_stream) {
    DevicePackingInput packing{
        destination_ranks, metadata, features, registered_buffer,
        total_assignments, 9, feature_width, plan_header.record_bytes};
    // Bridge the caller's input-production stream once at job scope, then use a
    // dedicated non-blocking producer stream.  This avoids legacy-default-stream
    // serialization without adding any per-descriptor host operation.
    CUDA_CHECK(cudaEventRecord(input_ready, router_stream));
    CUDA_CHECK(cudaStreamWaitEvent(producer_stream, input_ready, 0));
    gpu_transport_pipeline_kernel<<<4, 256, 0, producer_stream>>>(
        reveal_records, record_count, destination_ranks, expert_ids, token_ids,
        feature_digests, metadata, total_assignments, rank, d_reveal_queue,
        d_records_by_descriptor, d_timing, d_final_router_ns, delay_cycles,
        handle, d_plan, d_action_queue, d_state, packing, d_slot_claims,
        d_trace_state, d_counters, expected_remote_actions, d_wait_times);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(producer_stream));
    CUDA_CHECK(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
    DeviceActionQueue action_queue{};
    DeviceTraceState trace_state{};
    CUDA_CHECK(cudaMemcpy(&action_queue, d_action_queue, sizeof(action_queue), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&trace_state, d_trace_state, sizeof(trace_state), cudaMemcpyDeviceToHost));
    actions.resize(action_queue.tail);
    traces.resize(trace_state.trace_count);
    timings.resize(record_count);
    if (!actions.empty()) CUDA_CHECK(cudaMemcpy(actions.data(), d_actions, actions.size() * sizeof(CommittedAction), cudaMemcpyDeviceToHost));
    if (!traces.empty()) CUDA_CHECK(cudaMemcpy(traces.data(), d_traces, traces.size() * sizeof(TransportTrace), cudaMemcpyDeviceToHost));
    if (!timings.empty()) CUDA_CHECK(cudaMemcpy(timings.data(), d_timing, timings.size() * sizeof(PipelineTiming), cudaMemcpyDeviceToHost));
    std::vector<uint64_t> wait_times(expected_remote_actions);
    uint64_t final_router = 0;
    if (!wait_times.empty()) CUDA_CHECK(cudaMemcpy(wait_times.data(), d_wait_times, wait_times.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&final_router, d_final_router_ns, sizeof(final_router), cudaMemcpyDeviceToHost));
    uint32_t remote_index = 0;
    for (const auto& trace : traces) {
      if (trace.is_remote && remote_index < wait_times.size()) {
        timings[trace.action.descriptor_id].t7_remote_completion = wait_times[remote_index++];
      }
    }
    for (auto& value : timings) value.t8_final_router_completion = final_router;
  }
};

}  // namespace

extern "C" {

const char* r6_m5_last_error() { return last_error.c_str(); }

void* r6_m5_create(int rank, int device, void* buffer, size_t bytes,
                   const char* endpoint, const void* plan_blob, size_t plan_bytes) {
  Runtime* runtime = nullptr;
  if (protect([&] { runtime = new Runtime(rank, device, buffer, bytes, endpoint, plan_blob, plan_bytes); }) != 0)
    return nullptr;
  return runtime;
}

int r6_m5_run(void* opaque, const int64_t* reveal_records, uint32_t record_count,
              const int32_t* destination_ranks, const int64_t* expert_ids,
              const int64_t* token_ids, const int64_t* feature_digests,
              const float* features, int64_t* metadata, uint32_t total_assignments,
              uint32_t feature_width, uint32_t expected_remote_actions,
              uint64_t delay_cycles, uintptr_t router_stream) {
  return protect([&] {
    static_cast<Runtime*>(opaque)->run(
        reveal_records, record_count, destination_ranks, expert_ids, token_ids,
        feature_digests, features, metadata, total_assignments, feature_width,
        expected_remote_actions, delay_cycles, reinterpret_cast<cudaStream_t>(router_stream));
  });
}

uint64_t r6_m5_counter(void* opaque, int index) {
  const auto& value = static_cast<Runtime*>(opaque)->counters;
  const uint64_t* begin = &value.scheduler_actions;
  return index >= 0 && index < 12 ? begin[index] : 0;
}

size_t r6_m5_action_count(void* opaque) { return static_cast<Runtime*>(opaque)->actions.size(); }
size_t r6_m5_trace_count(void* opaque) { return static_cast<Runtime*>(opaque)->traces.size(); }
size_t r6_m5_timing_count(void* opaque) { return static_cast<Runtime*>(opaque)->timings.size(); }

int r6_m5_copy_actions(void* opaque, int64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->actions;
    if (rows < values.size()) throw std::runtime_error("action output capacity too small");
    for (size_t index = 0; index < values.size(); ++index) {
      const auto& a = values[index];
      int64_t* row = output + index * 12;
      row[0] = a.action_id; row[1] = a.descriptor_id; row[2] = a.chunk_id;
      row[3] = a.reveal_epoch; row[4] = a.src_rank; row[5] = a.dst_rank;
      row[6] = a.src_offset; row[7] = a.dst_offset; row[8] = a.token_count;
      row[9] = a.bytes; row[10] = a.route_id; row[11] = a.flags;
    }
  });
}

int r6_m5_copy_traces(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->traces;
    if (rows < values.size()) throw std::runtime_error("trace output capacity too small");
    for (size_t index = 0; index < values.size(); ++index) {
      const auto& t = values[index];
      uint64_t* row = output + index * 19;
      row[0] = t.action.action_id; row[1] = t.action.descriptor_id;
      row[2] = static_cast<uint64_t>(t.action.src_rank); row[3] = static_cast<uint64_t>(t.action.dst_rank);
      row[4] = t.action.logical_src_offset; row[5] = t.action.logical_dst_offset;
      row[6] = t.action.physical_src_offset; row[7] = t.action.physical_dst_offset;
      row[8] = t.action.payload_bytes; row[9] = t.action.physical_bytes;
      row[10] = t.action.token_count; row[11] = t.action.route_id;
      row[12] = t.t2_action_consumed; row[13] = t.t3_pack_start;
      row[14] = t.t4_pack_end; row[15] = t.t5_put_start; row[16] = t.t6_put_end;
      row[17] = t.error_code; row[18] = t.is_remote;
    }
  });
}

int r6_m5_copy_timings(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->timings;
    if (rows < values.size()) throw std::runtime_error("timing output capacity too small");
    for (size_t index = 0; index < values.size(); ++index) {
      const auto& t = values[index];
      uint64_t* row = output + index * 6;
      row[0] = t.chunk_id; row[1] = t.descriptor_id;
      row[2] = t.t0_router_reveal; row[3] = t.t1_scheduler_commit;
      row[4] = t.t7_remote_completion; row[5] = t.t8_final_router_completion;
    }
  });
}

void r6_m5_destroy(void* opaque) { delete static_cast<Runtime*>(opaque); }

}  // extern "C"
