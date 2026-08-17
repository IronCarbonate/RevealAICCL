// R6-M9: revealed-only DescriptorCommit scheduler -> fused token-centric
// dispatch -> direct NCCL LSA stores -> expert-contiguous receive layout.

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <nccl.h>
#include <nccl_device.h>

#include "rlccl/ep/common/dispatch_handle.h"
#include "rlccl/ep/common/combine_ir.h"
#include "rlccl/ep/common/progressive_ep_ir.h"
#include "rlccl/ep/cuda/combine_epilogue.cuh"
#include "rlccl/ep/cuda/combine_layout.cuh"
#include "rlccl/ep/cuda/dispatch_epilogue.cuh"
#include "rlccl/ep/cuda/progressive_combine.cuh"
#include "rlccl/ep/cuda/progressive_dispatch.cuh"
#include "rlccl/ep/cuda/staged_gin_combine.cuh"
#include "rlccl/ep/cuda/staged_gin_dispatch.cuh"
#include "rlccl/scheduler/common/scheduler_ir.h"

// Keep the reusable data-plane implementation in rlccl/ep.  The include makes
// this standalone extension buildable with one nvcc command on the validation
// host, while the symbols remain in the EP namespace.
#include "rlccl/ep/cuda/progressive_dispatch.cu"
#include "rlccl/ep/cuda/progressive_combine.cu"

namespace rlccl::ep::m9_runtime {

using namespace rlccl::scheduler;
using namespace rlccl::ep;
using namespace rlccl::ep::cuda_backend;

#include "extensions/r6_m9_e2e_perf/m9_progressive_dispatch.cuh"

thread_local std::string last_error;
constexpr uint32_t kRuntimeMaxWorldSize = 16;
constexpr uint32_t kMaxBenchmarkRuns = 640;

#define CUDA_CHECK(call) do {                                                   \
  cudaError_t status_ = (call);                                                 \
  if (status_ != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status_)); \
} while (0)

#define NCCL_CHECK(call) do {                                                   \
  ncclResult_t status_ = (call);                                                \
  if (status_ != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status_)); \
} while (0)

struct DeviceRevealQueue {
  RevealRecord* records;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

enum BenchmarkArm : uint32_t {
  kProgressiveArm = 0,
  kDelayedArm = 1,
};

struct GateTiming {
  uint64_t t_gate_start;
  uint64_t t_final_router_seen;
  uint64_t t_first_commit_forwarded;
  uint64_t t_gate_complete;
};

struct E2EStageTiming {
  uint64_t router_start;
  uint64_t expert_start;
  uint64_t expert_complete;
  uint64_t combine_start;
  uint64_t combine_complete;
  uint64_t final_output_ready;
};

struct M7SchedulerState {
  uint64_t* revealed_count;
  uint64_t* committed_count;
  uint32_t* descriptor_epoch;
  uint32_t last_reveal_epoch;
  uint32_t has_reveal_epoch;
  uint64_t next_commit_id;
  uint64_t next_action_id;
};

__device__ __forceinline__ uint64_t device_time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
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
  return reinterpret_cast<const CompiledRouteTemplate*>(
      blob + header->route_templates_offset);
}

__device__ __forceinline__ const int32_t* pair_routes(
    const uint8_t* blob, const CompiledPlanBlobHeader* header) {
  return reinterpret_cast<const int32_t*>(
      blob + header->rank_pair_to_route_offset);
}

__device__ void router_reveal_role(
    const int64_t* input_records, uint32_t record_count,
    DeviceRevealQueue* queue, RevealRecord* records_by_descriptor,
    DispatchTiming* timings, uint64_t* final_router_ns,
    uint64_t delay_cycles, M7Counters* counters) {
  if (threadIdx.x != 0) return;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> head_ref(queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_ref(queue->tail);
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
    uint64_t now = device_time_ns();
    timings[reveal.descriptor_id].t_router_reveal = now;
    timings[reveal.descriptor_id].chunk_id = reveal.chunk_id;
    timings[reveal.descriptor_id].descriptor_id = reveal.descriptor_id;
    records_by_descriptor[reveal.descriptor_id] = reveal;
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
      *final_router_ns = now;
      __threadfence();
    }
  }
}

__device__ void descriptor_commit_scheduler_role(
    const uint8_t* blob, const int64_t* topk_idx,
    uint32_t num_tokens, uint32_t num_topk, uint32_t experts_per_rank,
    uint32_t num_experts, uint32_t source_rank, uint32_t expected_reveals,
    DeviceRevealQueue* reveal_queue, DeviceCommitQueue* commit_queue,
    CommitPeerPlan* peer_plans, CommittedAction* shadow_actions,
    uint32_t shadow_capacity, uint32_t* shadow_count,
    M7SchedulerState* scheduler_state, DeviceCommitState* commit_state,
    DispatchLayout layout, DispatchTiming* timings, M7Counters* counters) {
  if (threadIdx.x != 0) return;
  const auto* header = reinterpret_cast<const CompiledPlanBlobHeader*>(blob);
  if (header->magic != kCompiledPlanMagic ||
      header->version != kCompiledPlanVersion ||
      header->world_size != layout.world_size ||
      header->world_size > kRuntimeMaxWorldSize ||
      plan_checksum(blob, header->total_bytes) != header->checksum) {
    counters->errors += 1;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done(commit_state->scheduler_done);
    done.store(1, cuda::memory_order_release);
    return;
  }
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_head(reveal_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> reveal_tail(reveal_queue->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> commit_head(commit_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> commit_tail(commit_queue->tail);
  uint32_t processed = 0;
  uint64_t last_progress = device_time_ns();
  while (processed < expected_reveals) {
    uint32_t head = reveal_head.load(cuda::memory_order_relaxed);
    uint32_t tail = reveal_tail.load(cuda::memory_order_acquire);
    if (head == tail) {
      if (device_time_ns() - last_progress > 5'000'000'000ULL) {
        counters->errors += 1;
        break;
      }
      __nanosleep(64);
      continue;
    }
    RevealRecord reveal = reveal_queue->records[head % reveal_queue->capacity];
    reveal_head.store(head + 1, cuda::memory_order_release);
    ++processed;
    last_progress = device_time_ns();

    bool invalid = reveal.descriptor_id >= header->max_descriptors ||
        reveal.chunk_id >= header->max_chunks || num_topk == 0 ||
        reveal.assignment_count != reveal.token_count * num_topk ||
        reveal.assignment_begin != reveal.token_begin * num_topk ||
        static_cast<uint64_t>(reveal.token_begin) + reveal.token_count > num_tokens ||
        (scheduler_state->has_reveal_epoch &&
         reveal.reveal_epoch <= scheduler_state->last_reveal_epoch) ||
        scheduler_state->descriptor_epoch[reveal.descriptor_id] != UINT32_MAX;
    if (invalid) {
      if (scheduler_state->has_reveal_epoch &&
          reveal.reveal_epoch <= scheduler_state->last_reveal_epoch)
        counters->stale_action += 1;
      counters->errors += 1;
      continue;
    }

    uint32_t counts[kRuntimeMaxWorldSize]{};
    for (uint32_t local = 0; local < reveal.assignment_count; ++local) {
      uint32_t token = reveal.token_begin + local / num_topk;
      uint32_t slot = local % num_topk;
      int64_t expert = topk_idx[static_cast<uint64_t>(token) * num_topk + slot];
      if (expert < 0 || static_cast<uint64_t>(expert) >= num_experts) {
        invalid = true;
        break;
      }
      uint32_t dst = static_cast<uint32_t>(expert) / experts_per_rank;
      if (dst >= header->world_size) {
        invalid = true;
        break;
      }
      ++counts[dst];
    }
    if (invalid) {
      counters->errors += 1;
      continue;
    }

    uint64_t authorized_mask = 0;
    uint32_t built = 0;
    uint32_t shadow_begin = *shadow_count;
    for (uint32_t dst = 0; dst < header->world_size; ++dst) {
      CommitPeerPlan& output = peer_plans[
          reveal.descriptor_id * header->world_size + dst];
      output = CommitPeerPlan{};
      uint32_t count = counts[dst];
      if (!count) continue;
      uint32_t pair = source_rank * header->world_size + dst;
      int32_t route_index = pair_routes(blob, header)[pair];
      if (route_index < 0 || static_cast<uint32_t>(route_index) >= header->route_count ||
          count > layout.max_assignments_per_peer) {
        invalid = true;
        break;
      }
      const auto& route = routes(blob, header)[route_index];
      if (route.src_rank != static_cast<int32_t>(source_rank) ||
          route.dst_rank != static_cast<int32_t>(dst) ||
          shadow_begin + built >= shadow_capacity) {
        invalid = true;
        break;
      }
      output = CommitPeerPlan{
          count, static_cast<uint32_t>(route.route_id),
          layout.staging_offset(reveal.descriptor_id, dst),
          layout.recv_offset(reveal.descriptor_id, source_rank),
          kCommitLsaDirect, 0};
      // The old action is a shadow projection only.  Its frozen M6 logical
      // offsets remain byte-for-byte comparable to the reference scheduler.
      shadow_actions[shadow_begin + built] = CommittedAction{
          scheduler_state->next_action_id + built,
          reveal.descriptor_id, reveal.chunk_id, reveal.reveal_epoch,
          static_cast<int32_t>(source_rank), static_cast<int32_t>(dst),
          static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride +
              route.send_region_base,
          static_cast<uint64_t>(reveal.descriptor_id) * header->descriptor_stride +
              route.recv_region_base,
          count, 0, static_cast<uint64_t>(count) * header->record_bytes,
          static_cast<uint32_t>(route.route_id), route.flags};
      authorized_mask |= uint64_t(1) << dst;
      scheduler_state->revealed_count[pair] += count;
      scheduler_state->committed_count[pair] += count;
      ++built;
    }
    if (invalid) {
      counters->errors += 1;
      continue;
    }
    uint32_t queue_tail = commit_tail.load(cuda::memory_order_relaxed);
    while (queue_tail - commit_head.load(cuda::memory_order_acquire) >=
           commit_queue->capacity) {
      __nanosleep(64);
    }
    DescriptorCommit commit{
        scheduler_state->next_commit_id++, reveal.descriptor_id,
        reveal.chunk_id, reveal.reveal_epoch, reveal.token_begin,
        reveal.token_count, reveal.assignment_begin, reveal.assignment_count,
        authorized_mask, kCommitLsaDirect, 0};
    commit_queue->commits[queue_tail % commit_queue->capacity] = commit;
    commit_tail.store(queue_tail + 1, cuda::memory_order_release);
    *shadow_count = shadow_begin + built;
    scheduler_state->next_action_id += built;
    scheduler_state->descriptor_epoch[reveal.descriptor_id] = reveal.reveal_epoch;
    scheduler_state->last_reveal_epoch = reveal.reveal_epoch;
    scheduler_state->has_reveal_epoch = 1;
    timings[reveal.descriptor_id].t_scheduler_commit = device_time_ns();
    counters->descriptor_commits += 1;
    counters->shadow_actions += built;
  }
  __threadfence();
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done(commit_state->scheduler_done);
  done.store(1, cuda::memory_order_release);
}

__device__ void commit_gate_role(
    DeviceCommitQueue* producer, DeviceCommitState* producer_state,
    DeviceCommitQueue* consumer, DeviceCommitState* consumer_state,
    const uint64_t* final_router_ns, uint32_t expected_commits,
    BenchmarkArm arm, GateTiming* timing);

__global__ void m9_pipeline_kernel(
    const int64_t* reveal_records, uint32_t record_count,
    const uint8_t* plan_blob, const float* x, const int64_t* topk_idx,
    const float* topk_weights, uint32_t num_tokens, uint32_t feature_width,
    uint32_t num_topk, uint32_t experts_per_rank, uint32_t num_experts,
    uint32_t rank, DeviceRevealQueue* reveal_queue,
    DeviceCommitQueue* producer_commit_queue,
    DeviceCommitQueue* consumer_commit_queue,
    RevealRecord* records_by_descriptor,
    CommitPeerPlan* peer_plans, CommittedAction* shadow_actions,
    uint32_t shadow_capacity, uint32_t* shadow_count,
    M7SchedulerState* scheduler_state,
    DeviceCommitState* producer_commit_state,
    DeviceCommitState* consumer_commit_state,
    DispatchLayout layout, ncclDevComm dev_comm, ncclWindow_t window,
    uint8_t* registered_buffer, uint32_t* dst_cursors,
    DispatchTrace* traces, uint32_t trace_capacity, uint32_t* trace_count,
    DispatchTiming* timings, M7Counters* counters,
    uint64_t* final_router_ns, uint64_t delay_cycles,
    BenchmarkArm arm, GateTiming* gate_timing, uint32_t completion_base) {
  if (blockIdx.x == 0) {
    router_reveal_role(reveal_records, record_count, reveal_queue,
                       records_by_descriptor, timings, final_router_ns,
                       delay_cycles, counters);
  } else if (blockIdx.x == 1) {
    descriptor_commit_scheduler_role(
        plan_blob, topk_idx, num_tokens, num_topk, experts_per_rank,
        num_experts, rank, record_count, reveal_queue, producer_commit_queue,
        peer_plans, shadow_actions, shadow_capacity, shadow_count,
        scheduler_state, producer_commit_state, layout, timings, counters);
  } else if (blockIdx.x == 2) {
    ProgressiveDispatchInput input{
        x, topk_idx, topk_weights, num_tokens, feature_width, num_topk,
        experts_per_rank, num_experts};
    m9_progressive_dispatch_progress_role(
        consumer_commit_queue, peer_plans, records_by_descriptor,
        consumer_commit_state, input,
        layout, rank, dev_comm, window, registered_buffer, dst_cursors,
        traces, trace_capacity, trace_count, timings, counters,
        completion_base);
  } else if (blockIdx.x == 3) {
    m9_progressive_dispatch_wait_role(
        dev_comm, record_count, timings, counters, completion_base);
  } else if (blockIdx.x == 4) {
    commit_gate_role(
        producer_commit_queue, producer_commit_state,
        consumer_commit_queue, consumer_commit_state,
        final_router_ns, record_count, arm, gate_timing);
  }
}

__device__ void commit_gate_role(
    DeviceCommitQueue* producer, DeviceCommitState* producer_state,
    DeviceCommitQueue* consumer, DeviceCommitState* consumer_state,
    const uint64_t* final_router_ns, uint32_t expected_commits,
    BenchmarkArm arm, GateTiming* timing) {
  if (threadIdx.x != 0) return;
  timing->t_gate_start = device_time_ns();
  if (arm == kDelayedArm) {
    const volatile uint64_t* final_value = final_router_ns;
    while (*final_value == 0) __nanosleep(64);
    timing->t_final_router_seen = device_time_ns();
  }
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> producer_head(producer->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> producer_tail(producer->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> producer_done(
      producer_state->scheduler_done);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumer_head(consumer->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumer_tail(consumer->tail);
  uint32_t forwarded = 0;
  while (forwarded < expected_commits) {
    uint32_t source_head = producer_head.load(cuda::memory_order_relaxed);
    uint32_t source_tail = producer_tail.load(cuda::memory_order_acquire);
    if (source_head == source_tail) {
      if (producer_done.load(cuda::memory_order_acquire)) break;
      __nanosleep(64);
      continue;
    }
    uint32_t target_tail = consumer_tail.load(cuda::memory_order_relaxed);
    while (target_tail - consumer_head.load(cuda::memory_order_acquire) >=
           consumer->capacity) __nanosleep(64);
    consumer->commits[target_tail % consumer->capacity] =
        producer->commits[source_head % producer->capacity];
    consumer_tail.store(target_tail + 1, cuda::memory_order_release);
    producer_head.store(source_head + 1, cuda::memory_order_release);
    if (forwarded++ == 0) timing->t_first_commit_forwarded = device_time_ns();
  }
  timing->t_gate_complete = device_time_ns();
  __threadfence();
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> consumer_done(
      consumer_state->scheduler_done);
  consumer_done.store(1, cuda::memory_order_release);
}

__global__ void mark_stage_time_kernel(uint64_t* value) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *value = device_time_ns();
}

__global__ void preflight_lsa_barrier_kernel(
    ncclDevComm dev_comm, uint32_t barrier_index) {
  ncclLsaBarrierSession<ncclCoopCta> session{
      ncclCoopCta(), dev_comm, ncclTeamTagLsa(), barrier_index};
  session.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

template <typename Fn>
int protect(Fn&& fn) {
  try {
    fn();
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  } catch (...) {
    last_error = "unknown native exception";
    return -1;
  }
}

template <typename T>
T* allocate_device(size_t count = 1) {
  T* pointer = nullptr;
  CUDA_CHECK(cudaMalloc(&pointer, sizeof(T) * count));
  return pointer;
}

__global__ void expert_gemm_kernel(
    const ProgressiveEPHandle* handle, const float* recv_x,
    const float* expert_weights, uint32_t hidden, float* expert_output) {
  uint32_t row = blockIdx.x;
  if (row >= handle->num_recv_tokens) return;
  uint32_t local_expert = UINT32_MAX;
  for (uint32_t expert = 0; expert < handle->num_local_experts; ++expert) {
    if (row >= handle->expert_offsets[expert] &&
        row < handle->expert_offsets[expert + 1]) {
      local_expert = expert;
      break;
    }
  }
  if (local_expert == UINT32_MAX) return;
  for (uint32_t output = threadIdx.x; output < hidden; output += blockDim.x) {
    float sum = 0.0f;
    const float* weight = expert_weights +
        (static_cast<uint64_t>(local_expert) * hidden + output) * hidden;
    const float* input = recv_x + static_cast<uint64_t>(row) * hidden;
    for (uint32_t inner = 0; inner < hidden; ++inner)
      sum += weight[inner] * input[inner];
    expert_output[static_cast<uint64_t>(row) * hidden + output] = sum;
  }
}

__global__ void convert_topk_idx_kernel(
    const int64_t* source, int32_t* destination, uint64_t count) {
  for (uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    destination[index] = static_cast<int32_t>(source[index]);
}

struct Runtime {
  int rank;
  int device;
  CompiledPlanBlobHeader plan_header{};
  DispatchLayout layout{};
  CombineLayout combine_layout{};
  ncclComm_t comm = nullptr;
  int nccl_version = 0;
  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclDevComm dev_comm{};
  ncclWindow_t window = nullptr;
  bool dev_comm_created = false;
  uint8_t* registered_buffer = nullptr;
  size_t registered_bytes = 0;
  uint8_t* d_plan = nullptr;
  RevealRecord* d_reveals = nullptr;
  RevealRecord* d_records_by_descriptor = nullptr;
  DescriptorCommit* d_commits = nullptr;
  DescriptorCommit* d_dispatch_commits = nullptr;
  CommitPeerPlan* d_peer_plans = nullptr;
  CommittedAction* d_shadow_actions = nullptr;
  DeviceRevealQueue* d_reveal_queue = nullptr;
  DeviceCommitQueue* d_commit_queue = nullptr;
  DeviceCommitQueue* d_dispatch_queue = nullptr;
  M7SchedulerState* d_scheduler_state = nullptr;
  DeviceCommitState* d_commit_state = nullptr;
  DeviceCommitState* d_dispatch_state = nullptr;
  uint64_t* d_revealed_count = nullptr;
  uint64_t* d_committed_count = nullptr;
  uint32_t* d_descriptor_epoch = nullptr;
  uint32_t* d_shadow_count = nullptr;
  uint32_t* d_dst_cursors = nullptr;
  DispatchTrace* d_traces = nullptr;
  uint32_t* d_trace_count = nullptr;
  DispatchTiming* d_timings = nullptr;
  M7Counters* d_counters = nullptr;
  uint64_t* d_final_router_ns = nullptr;
  uint32_t* d_expert_counts = nullptr;
  uint32_t* d_expert_offsets = nullptr;
  uint32_t* d_expert_cursors = nullptr;
  float* d_recv_x = nullptr;
  DispatchTokenMeta* d_recv_metadata = nullptr;
  ProgressiveEPHandle* d_handle = nullptr;
  CombineRange* d_combine_ranges = nullptr;
  float* d_expert_output = nullptr;
  ReturnTrace* d_return_traces = nullptr;
  uint32_t* d_return_trace_count = nullptr;
  M8CombineCounters* d_combine_counters = nullptr;
  uint64_t* d_combine_completion_time = nullptr;
  float* d_final_output = nullptr;
  int32_t* d_source_topk_idx = nullptr;
  GateTiming* d_gate_timing = nullptr;
  E2EStageTiming* d_stage_timing = nullptr;
  cudaStream_t pipeline_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t e2e_start = nullptr;
  cudaEvent_t e2e_end = nullptr;
  uint32_t queue_capacity = 0;
  uint32_t shadow_capacity = 0;
  uint32_t max_recv_records = 0;
  uint32_t num_local_experts = 0;
  std::vector<DescriptorCommit> commits;
  std::vector<CommitPeerPlan> peer_plans;
  std::vector<CommittedAction> shadow_actions;
  std::vector<DispatchTrace> traces;
  std::vector<DispatchTiming> timings;
  std::vector<uint32_t> expert_counts;
  std::vector<uint32_t> expert_offsets;
  ProgressiveEPHandle handle{};
  M7Counters counters{};
  std::vector<ReturnTrace> return_traces;
  std::vector<float> final_output;
  M8CombineCounters combine_counters{};
  GateTiming gate_timing{};
  E2EStageTiming stage_timing{};
  float last_e2e_ms = 0.0f;
  uint32_t run_epoch = 0;

  Runtime(int rank_, int device_, const ncclUniqueId* unique_id,
          const void* plan_blob, size_t plan_bytes, DispatchLayout layout_,
          CombineLayout combine_layout_)
      : rank(rank_), device(device_), layout(layout_),
        combine_layout(combine_layout_) {
    CUDA_CHECK(cudaSetDevice(device));
    if (!unique_id) throw std::runtime_error("M7 NCCL unique ID is null");
    if (plan_bytes < sizeof(CompiledPlanBlobHeader))
      throw std::runtime_error("M7 plan header truncated");
    std::memcpy(&plan_header, plan_blob, sizeof(plan_header));
    registered_bytes = layout.capacity_bytes + combine_layout.capacity_bytes;
    if (combine_layout.base_offset != layout.capacity_bytes)
      throw std::runtime_error("M8 combine region must follow frozen M7 window");
    if (plan_header.magic != kCompiledPlanMagic ||
        plan_header.world_size != layout.world_size ||
        plan_header.max_descriptors != layout.max_descriptors ||
        layout.world_size > kRuntimeMaxWorldSize)
      throw std::runtime_error("M7 plan/layout mismatch");
    NCCL_CHECK(ncclGetVersion(&nccl_version));
    NCCL_CHECK(ncclCommInitRank(&comm, layout.world_size, *unique_id, rank));
    NCCL_CHECK(ncclCommQueryProperties(comm, &properties));
    if (!properties.deviceApiSupport)
      throw std::runtime_error("NCCL_DEVICE_API_NOT_AVAILABLE: deviceApiSupport=false");
    NCCL_CHECK(ncclMemAlloc(reinterpret_cast<void**>(&registered_buffer), registered_bytes));
    CUDA_CHECK(cudaMemset(registered_buffer, 0, registered_bytes));
    NCCL_CHECK(ncclCommWindowRegister(
        comm, registered_buffer, registered_bytes, &window,
        NCCL_WIN_COLL_SYMMETRIC));
    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.lsaBarrierCount =
        (layout.max_descriptors + 2) * kMaxBenchmarkRuns;
    NCCL_CHECK(ncclDevCommCreate(comm, &requirements, &dev_comm));
    dev_comm_created = true;
    if (dev_comm.lsaSize < layout.world_size)
      throw std::runtime_error("NCCL_DEVICE_API_NOT_AVAILABLE: incomplete LSA team");

    queue_capacity = std::max<uint32_t>(layout.max_descriptors, 8);
    shadow_capacity = std::max<uint32_t>(layout.max_descriptors * layout.world_size, 16);
    max_recv_records = layout.max_descriptors * layout.world_size *
                       layout.max_assignments_per_peer;
    uint32_t pairs = layout.world_size * layout.world_size;
    d_plan = allocate_device<uint8_t>(plan_bytes);
    d_reveals = allocate_device<RevealRecord>(queue_capacity);
    d_records_by_descriptor = allocate_device<RevealRecord>(layout.max_descriptors);
    d_commits = allocate_device<DescriptorCommit>(queue_capacity);
    d_dispatch_commits = allocate_device<DescriptorCommit>(queue_capacity);
    d_peer_plans = allocate_device<CommitPeerPlan>(layout.max_descriptors * layout.world_size);
    d_shadow_actions = allocate_device<CommittedAction>(shadow_capacity);
    d_reveal_queue = allocate_device<DeviceRevealQueue>();
    d_commit_queue = allocate_device<DeviceCommitQueue>();
    d_dispatch_queue = allocate_device<DeviceCommitQueue>();
    d_scheduler_state = allocate_device<M7SchedulerState>();
    d_commit_state = allocate_device<DeviceCommitState>();
    d_dispatch_state = allocate_device<DeviceCommitState>();
    d_revealed_count = allocate_device<uint64_t>(pairs);
    d_committed_count = allocate_device<uint64_t>(pairs);
    d_descriptor_epoch = allocate_device<uint32_t>(layout.max_descriptors);
    d_shadow_count = allocate_device<uint32_t>();
    d_dst_cursors = allocate_device<uint32_t>(layout.max_descriptors * layout.world_size);
    d_traces = allocate_device<DispatchTrace>(layout.max_descriptors * layout.world_size);
    d_trace_count = allocate_device<uint32_t>();
    d_timings = allocate_device<DispatchTiming>(layout.max_descriptors);
    d_counters = allocate_device<M7Counters>();
    d_final_router_ns = allocate_device<uint64_t>();
    d_expert_output = allocate_device<float>(
        static_cast<size_t>(max_recv_records) * layout.feature_width);
    d_return_traces = allocate_device<ReturnTrace>(max_recv_records);
    d_return_trace_count = allocate_device<uint32_t>();
    d_combine_counters = allocate_device<M8CombineCounters>();
    d_combine_completion_time = allocate_device<uint64_t>();
    d_final_output = allocate_device<float>(
        static_cast<size_t>(combine_layout.num_source_tokens) * layout.feature_width);
    d_source_topk_idx = allocate_device<int32_t>(
        static_cast<size_t>(combine_layout.num_source_tokens) * combine_layout.num_topk);
    d_gate_timing = allocate_device<GateTiming>();
    d_stage_timing = allocate_device<E2EStageTiming>();
    CUDA_CHECK(cudaMemcpy(d_plan, plan_blob, plan_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&pipeline_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&e2e_start));
    CUDA_CHECK(cudaEventCreate(&e2e_end));
  }

  ~Runtime() {
    if (input_ready) cudaEventDestroy(input_ready);
    if (e2e_start) cudaEventDestroy(e2e_start);
    if (e2e_end) cudaEventDestroy(e2e_end);
    if (pipeline_stream) cudaStreamDestroy(pipeline_stream);
    cudaFree(d_plan); cudaFree(d_reveals); cudaFree(d_records_by_descriptor);
    cudaFree(d_commits); cudaFree(d_dispatch_commits); cudaFree(d_peer_plans);
    cudaFree(d_shadow_actions); cudaFree(d_reveal_queue); cudaFree(d_commit_queue);
    cudaFree(d_dispatch_queue); cudaFree(d_scheduler_state);
    cudaFree(d_commit_state); cudaFree(d_dispatch_state);
    cudaFree(d_revealed_count); cudaFree(d_committed_count);
    cudaFree(d_descriptor_epoch); cudaFree(d_shadow_count); cudaFree(d_dst_cursors);
    cudaFree(d_traces); cudaFree(d_trace_count); cudaFree(d_timings);
    cudaFree(d_counters); cudaFree(d_final_router_ns); cudaFree(d_expert_counts);
    cudaFree(d_expert_offsets); cudaFree(d_expert_cursors); cudaFree(d_recv_x);
    cudaFree(d_recv_metadata); cudaFree(d_handle);
    cudaFree(d_combine_ranges); cudaFree(d_expert_output);
    cudaFree(d_return_traces); cudaFree(d_return_trace_count);
    cudaFree(d_combine_counters); cudaFree(d_combine_completion_time);
    cudaFree(d_final_output);
    cudaFree(d_source_topk_idx);
    cudaFree(d_gate_timing); cudaFree(d_stage_timing);
    if (dev_comm_created) ncclDevCommDestroy(comm, &dev_comm);
    if (window) ncclCommWindowDeregister(comm, window);
    if (registered_buffer) ncclMemFree(registered_buffer);
    if (comm) ncclCommDestroy(comm);
  }

  void run(const int64_t* reveal_records, uint32_t record_count,
           const float* x, const int64_t* topk_idx, const float* topk_weights,
           const float* expert_weights,
           uint32_t num_tokens, uint32_t feature_width, uint32_t num_topk,
           uint32_t experts_per_rank, uint32_t num_experts,
           uint64_t delay_cycles, BenchmarkArm arm,
           cudaStream_t router_stream) {
    if (feature_width != layout.feature_width || num_topk == 0 ||
        num_tokens != combine_layout.num_source_tokens ||
        num_topk != combine_layout.num_topk ||
        num_experts != experts_per_rank * layout.world_size ||
        record_count > layout.max_descriptors)
      throw std::runtime_error("M7 run shape/layout mismatch");
    if (run_epoch >= kMaxBenchmarkRuns)
      throw std::runtime_error("M9 benchmark run capacity exceeded");
    uint32_t completion_base =
        run_epoch * (layout.max_descriptors + 2);
    num_local_experts = experts_per_rank;
    if (!d_handle) {
      d_combine_ranges = allocate_device<CombineRange>(num_local_experts);
      d_expert_counts = allocate_device<uint32_t>(num_local_experts);
      d_expert_offsets = allocate_device<uint32_t>(num_local_experts + 1);
      d_expert_cursors = allocate_device<uint32_t>(num_local_experts);
      d_recv_x = allocate_device<float>(static_cast<size_t>(max_recv_records) * feature_width);
      d_recv_metadata = allocate_device<DispatchTokenMeta>(max_recv_records);
      d_handle = allocate_device<ProgressiveEPHandle>();
    }

    CUDA_CHECK(cudaMemset(registered_buffer, 0, registered_bytes));
    CUDA_CHECK(cudaMemset(d_reveals, 0, sizeof(RevealRecord) * queue_capacity));
    CUDA_CHECK(cudaMemset(d_records_by_descriptor, 0, sizeof(RevealRecord) * layout.max_descriptors));
    CUDA_CHECK(cudaMemset(d_commits, 0, sizeof(DescriptorCommit) * queue_capacity));
    CUDA_CHECK(cudaMemset(d_dispatch_commits, 0, sizeof(DescriptorCommit) * queue_capacity));
    CUDA_CHECK(cudaMemset(d_peer_plans, 0, sizeof(CommitPeerPlan) * layout.max_descriptors * layout.world_size));
    CUDA_CHECK(cudaMemset(d_shadow_actions, 0, sizeof(CommittedAction) * shadow_capacity));
    CUDA_CHECK(cudaMemset(d_revealed_count, 0, sizeof(uint64_t) * layout.world_size * layout.world_size));
    CUDA_CHECK(cudaMemset(d_committed_count, 0, sizeof(uint64_t) * layout.world_size * layout.world_size));
    CUDA_CHECK(cudaMemset(d_descriptor_epoch, 0xFF, sizeof(uint32_t) * layout.max_descriptors));
    CUDA_CHECK(cudaMemset(d_shadow_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_dst_cursors, 0, sizeof(uint32_t) * layout.max_descriptors * layout.world_size));
    CUDA_CHECK(cudaMemset(d_traces, 0, sizeof(DispatchTrace) * layout.max_descriptors * layout.world_size));
    CUDA_CHECK(cudaMemset(d_trace_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_timings, 0, sizeof(DispatchTiming) * layout.max_descriptors));
    CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(M7Counters)));
    CUDA_CHECK(cudaMemset(d_final_router_ns, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_expert_counts, 0, sizeof(uint32_t) * num_local_experts));
    CUDA_CHECK(cudaMemset(d_expert_offsets, 0, sizeof(uint32_t) * (num_local_experts + 1)));
    CUDA_CHECK(cudaMemset(d_expert_cursors, 0, sizeof(uint32_t) * num_local_experts));
    CUDA_CHECK(cudaMemset(d_recv_x, 0, sizeof(float) * static_cast<size_t>(max_recv_records) * feature_width));
    CUDA_CHECK(cudaMemset(d_recv_metadata, 0, sizeof(DispatchTokenMeta) * max_recv_records));
    CUDA_CHECK(cudaMemset(d_combine_ranges, 0, sizeof(CombineRange) * num_local_experts));
    CUDA_CHECK(cudaMemset(d_expert_output, 0, sizeof(float) * static_cast<size_t>(max_recv_records) * feature_width));
    CUDA_CHECK(cudaMemset(d_return_traces, 0, sizeof(ReturnTrace) * max_recv_records));
    CUDA_CHECK(cudaMemset(d_return_trace_count, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_combine_counters, 0, sizeof(M8CombineCounters)));
    CUDA_CHECK(cudaMemset(d_combine_completion_time, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_final_output, 0, sizeof(float) * static_cast<size_t>(num_tokens) * feature_width));
    CUDA_CHECK(cudaMemset(d_gate_timing, 0, sizeof(GateTiming)));
    CUDA_CHECK(cudaMemset(d_stage_timing, 0, sizeof(E2EStageTiming)));
    DeviceRevealQueue reveal_queue{d_reveals, queue_capacity, 0, 0};
    DeviceCommitQueue commit_queue{d_commits, queue_capacity, 0, 0};
    DeviceCommitQueue dispatch_queue{d_dispatch_commits, queue_capacity, 0, 0};
    M7SchedulerState scheduler_state{
        d_revealed_count, d_committed_count, d_descriptor_epoch, 0, 0, 0, 0};
    DeviceCommitState commit_state{0, 0};
    DeviceCommitState dispatch_state{0, 0};
    ProgressiveEPHandle initial_handle{};
    initial_handle.num_recv_tokens = 0;
    initial_handle.num_local_experts = num_local_experts;
    initial_handle.num_topk = num_topk;
    initial_handle.expert_counts = d_expert_counts;
    initial_handle.expert_offsets = d_expert_offsets;
    initial_handle.recv_src_metadata = d_recv_metadata;
    initial_handle.generation = 1;
    initial_handle.num_source_tokens = num_tokens;
    initial_handle.source_topk_idx = d_source_topk_idx;
    initial_handle.source_topk_weights = const_cast<float*>(topk_weights);
    CUDA_CHECK(cudaMemcpy(d_reveal_queue, &reveal_queue, sizeof(reveal_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_commit_queue, &commit_queue, sizeof(commit_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dispatch_queue, &dispatch_queue, sizeof(dispatch_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scheduler_state, &scheduler_state, sizeof(scheduler_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_commit_state, &commit_state, sizeof(commit_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dispatch_state, &dispatch_state, sizeof(dispatch_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_handle, &initial_handle, sizeof(initial_handle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(input_ready, router_stream));
    CUDA_CHECK(cudaStreamWaitEvent(pipeline_stream, input_ready, 0));
    preflight_lsa_barrier_kernel<<<1, 32, 0, pipeline_stream>>>(
        dev_comm, completion_base + layout.max_descriptors + 1);
    convert_topk_idx_kernel<<<32, 256, 0, pipeline_stream>>>(
        topk_idx, d_source_topk_idx,
        static_cast<uint64_t>(num_tokens) * num_topk);
    CUDA_CHECK(cudaEventRecord(e2e_start, pipeline_stream));
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->router_start);
    m9_pipeline_kernel<<<5, 256, 0, pipeline_stream>>>(
        reveal_records, record_count, d_plan, x, topk_idx, topk_weights,
        num_tokens, feature_width, num_topk, experts_per_rank, num_experts,
        rank, d_reveal_queue, d_commit_queue, d_dispatch_queue,
        d_records_by_descriptor,
        d_peer_plans, d_shadow_actions, shadow_capacity, d_shadow_count,
        d_scheduler_state, d_commit_state, d_dispatch_state,
        layout, dev_comm, window,
        registered_buffer, d_dst_cursors, d_traces,
        layout.max_descriptors * layout.world_size, d_trace_count,
        d_timings, d_counters, d_final_router_ns, delay_cycles,
        arm, d_gate_timing, completion_base);
    CUDA_CHECK(cudaGetLastError());
    dispatch_count_experts_kernel<<<
        layout.max_descriptors * layout.world_size, 256, 0, pipeline_stream>>>(
        registered_buffer, layout, rank, experts_per_rank, d_expert_counts,
        d_counters);
    dispatch_exclusive_scan_kernel<<<1, 1, 0, pipeline_stream>>>(
        d_expert_counts, d_expert_offsets, d_expert_cursors,
        num_local_experts, d_handle, 1);
    dispatch_scatter_experts_kernel<<<
        layout.max_descriptors * layout.world_size, 256, 0, pipeline_stream>>>(
        registered_buffer, layout, rank, experts_per_rank, d_expert_cursors,
        d_recv_x, d_recv_metadata, d_counters);
    build_combine_ranges_kernel<<<1, 32, 0, pipeline_stream>>>(
        d_handle, rank, d_combine_ranges);
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->expert_start);
    expert_gemm_kernel<<<max_recv_records, 32, 0, pipeline_stream>>>(
        d_handle, d_recv_x, expert_weights, feature_width, d_expert_output);
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->expert_complete);
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->combine_start);
    progressive_combine_kernel<<<1, 256, 0, pipeline_stream>>>(
        d_handle, d_combine_ranges, num_local_experts, d_expert_output,
        combine_layout, rank, layout.world_size, 1, dev_comm, window,
        registered_buffer, completion_base + layout.max_descriptors,
        d_return_traces,
        max_recv_records, d_return_trace_count, d_combine_counters);
    progressive_combine_wait_kernel<<<1, 256, 0, pipeline_stream>>>(
        dev_comm, completion_base + layout.max_descriptors,
        d_combine_completion_time,
        d_combine_counters);
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->combine_complete);
    combine_reduce_epilogue<<<num_tokens, 32, 0, pipeline_stream>>>(
        registered_buffer, combine_layout, d_handle, 1, d_final_output,
        d_combine_counters);
    mark_stage_time_kernel<<<1, 1, 0, pipeline_stream>>>(
        &d_stage_timing->final_output_ready);
    CUDA_CHECK(cudaEventRecord(e2e_end, pipeline_stream));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(pipeline_stream));
    ++run_epoch;
    CUDA_CHECK(cudaEventElapsedTime(&last_e2e_ms, e2e_start, e2e_end));

    DeviceCommitQueue commit_queue_out{};
    uint32_t shadow_count_out = 0, trace_count_out = 0;
    uint64_t final_router = 0;
    uint64_t combine_completion = 0;
    uint32_t return_trace_count_out = 0;
    CUDA_CHECK(cudaMemcpy(&commit_queue_out, d_commit_queue, sizeof(commit_queue_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&shadow_count_out, d_shadow_count, sizeof(shadow_count_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&trace_count_out, d_trace_count, sizeof(trace_count_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&handle, d_handle, sizeof(handle), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&final_router, d_final_router_ns, sizeof(final_router), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&combine_completion, d_combine_completion_time, sizeof(combine_completion), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&return_trace_count_out, d_return_trace_count, sizeof(return_trace_count_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&combine_counters, d_combine_counters, sizeof(combine_counters), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&gate_timing, d_gate_timing, sizeof(gate_timing), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stage_timing, d_stage_timing, sizeof(stage_timing), cudaMemcpyDeviceToHost));
    commits.resize(commit_queue_out.tail);
    peer_plans.resize(layout.max_descriptors * layout.world_size);
    shadow_actions.resize(shadow_count_out);
    traces.resize(trace_count_out);
    timings.resize(record_count);
    expert_counts.resize(num_local_experts);
    expert_offsets.resize(num_local_experts + 1);
    return_traces.resize(return_trace_count_out);
    final_output.resize(static_cast<size_t>(num_tokens) * feature_width);
    if (!commits.empty()) CUDA_CHECK(cudaMemcpy(commits.data(), d_commits, commits.size() * sizeof(DescriptorCommit), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(peer_plans.data(), d_peer_plans, peer_plans.size() * sizeof(CommitPeerPlan), cudaMemcpyDeviceToHost));
    if (!shadow_actions.empty()) CUDA_CHECK(cudaMemcpy(shadow_actions.data(), d_shadow_actions, shadow_actions.size() * sizeof(CommittedAction), cudaMemcpyDeviceToHost));
    if (!traces.empty()) CUDA_CHECK(cudaMemcpy(traces.data(), d_traces, traces.size() * sizeof(DispatchTrace), cudaMemcpyDeviceToHost));
    if (!timings.empty()) CUDA_CHECK(cudaMemcpy(timings.data(), d_timings, timings.size() * sizeof(DispatchTiming), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_counts.data(), d_expert_counts, expert_counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_offsets.data(), d_expert_offsets, expert_offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (!return_traces.empty()) CUDA_CHECK(cudaMemcpy(return_traces.data(), d_return_traces, return_traces.size() * sizeof(ReturnTrace), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(final_output.data(), d_final_output, final_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto& timing : timings) timing.t_final_router_completion = final_router;
    for (auto& trace : traces)
      trace.t_remote_completion = timings[trace.descriptor_id].t_remote_completion;
    for (auto& trace : return_traces)
      trace.t_remote_completion = combine_completion;
  }
};

}  // namespace rlccl::ep::m9_runtime

using namespace rlccl::ep::m9_runtime;

extern "C" {

const char* r6_m9_last_error() { return last_error.c_str(); }
size_t r6_m9_unique_id_size() { return sizeof(ncclUniqueId); }

int r6_m9_get_unique_id(void* output, size_t bytes) {
  return protect([&] {
    if (!output || bytes != sizeof(ncclUniqueId))
      throw std::runtime_error("NCCL unique ID output size mismatch");
    NCCL_CHECK(ncclGetUniqueId(static_cast<ncclUniqueId*>(output)));
  });
}

void* r6_m9_create(
    int rank, int device, const void* unique_id, size_t unique_id_bytes,
    const void* plan_blob, size_t plan_bytes, uint32_t world_size,
    uint32_t max_descriptors, uint32_t max_assignments_per_peer,
    uint32_t feature_width, uint64_t peer_stride,
    uint64_t descriptor_stride, uint64_t region_bytes,
    uint32_t num_source_tokens, uint32_t num_topk) {
  Runtime* runtime = nullptr;
  if (unique_id_bytes != sizeof(ncclUniqueId)) {
    last_error = "NCCL unique ID size mismatch";
    return nullptr;
  }
  if (protect([&] {
        DispatchLayout layout{world_size, max_descriptors,
            max_assignments_per_peer, feature_width,
            32 + feature_width * 4, peer_stride, descriptor_stride,
            region_bytes, 2 * region_bytes};
        CombineLayout combine_layout = make_combine_layout(
            num_source_tokens, num_topk, feature_width, layout.capacity_bytes);
        runtime = new Runtime(
            rank, device, static_cast<const ncclUniqueId*>(unique_id),
            plan_blob, plan_bytes, layout, combine_layout);
      }) != 0) return nullptr;
  return runtime;
}

int r6_m9_run(
    void* opaque, const int64_t* reveal_records, uint32_t record_count,
    const float* x, const int64_t* topk_idx, const float* topk_weights,
    const float* expert_weights,
    uint32_t num_tokens, uint32_t feature_width, uint32_t num_topk,
    uint32_t experts_per_rank, uint32_t num_experts,
    uint32_t arm, uint64_t delay_cycles, uintptr_t router_stream) {
  return protect([&] {
    if (arm > static_cast<uint32_t>(kDelayedArm))
      throw std::runtime_error("invalid M9 benchmark arm");
    static_cast<Runtime*>(opaque)->run(
        reveal_records, record_count, x, topk_idx, topk_weights, expert_weights,
        num_tokens, feature_width, num_topk, experts_per_rank, num_experts,
        delay_cycles, static_cast<BenchmarkArm>(arm),
        reinterpret_cast<cudaStream_t>(router_stream));
  });
}

float r6_m9_last_e2e_ms(void* opaque) {
  return static_cast<Runtime*>(opaque)->last_e2e_ms;
}

uint64_t r6_m9_combine_counter(void* opaque, int index) {
  const auto& value = static_cast<Runtime*>(opaque)->combine_counters;
  const uint64_t* begin = &value.rows_mapped;
  return index >= 0 && index < 17 ? begin[index] : 0;
}

uint64_t r6_m9_counter(void* opaque, int index) {
  const auto& value = static_cast<Runtime*>(opaque)->counters;
  const uint64_t* begin = &value.descriptor_commits;
  return index >= 0 && index < 15 ? begin[index] : 0;
}

uint64_t r6_m9_capability(void* opaque, int index) {
  const auto* runtime = static_cast<Runtime*>(opaque);
  switch (index) {
    case 0: return runtime->nccl_version;
    case 1: return runtime->properties.deviceApiSupport;
    case 2: return runtime->properties.multimemSupport;
    case 3: return runtime->properties.ginType;
    case 4: return runtime->properties.nLsaTeams;
    case 5: return runtime->dev_comm.lsaSize;
    case 6: return runtime->window != nullptr;
    default: return 0;
  }
}

size_t r6_m9_commit_count(void* opaque) { return static_cast<Runtime*>(opaque)->commits.size(); }
size_t r6_m9_peer_plan_count(void* opaque) { return static_cast<Runtime*>(opaque)->peer_plans.size(); }
size_t r6_m9_shadow_action_count(void* opaque) { return static_cast<Runtime*>(opaque)->shadow_actions.size(); }
size_t r6_m9_trace_count(void* opaque) { return static_cast<Runtime*>(opaque)->traces.size(); }
size_t r6_m9_timing_count(void* opaque) { return static_cast<Runtime*>(opaque)->timings.size(); }
size_t r6_m9_num_recv_tokens(void* opaque) { return static_cast<Runtime*>(opaque)->handle.num_recv_tokens; }
size_t r6_m9_num_local_experts(void* opaque) { return static_cast<Runtime*>(opaque)->num_local_experts; }
size_t r6_m9_return_trace_count(void* opaque) { return static_cast<Runtime*>(opaque)->return_traces.size(); }
size_t r6_m9_num_source_tokens(void* opaque) { return static_cast<Runtime*>(opaque)->combine_layout.num_source_tokens; }

int r6_m9_copy_commits(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->commits;
    if (rows < values.size()) throw std::runtime_error("commit output too small");
    for (size_t i = 0; i < values.size(); ++i) {
      const auto& c = values[i]; uint64_t* r = output + i * 10;
      r[0]=c.commit_id; r[1]=c.descriptor_id; r[2]=c.chunk_id;
      r[3]=c.reveal_epoch; r[4]=c.token_begin; r[5]=c.token_count;
      r[6]=c.assignment_begin; r[7]=c.assignment_count;
      r[8]=c.authorized_dst_mask; r[9]=c.flags;
    }
  });
}

int r6_m9_copy_peer_plans(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto* runtime = static_cast<Runtime*>(opaque);
    if (rows < runtime->peer_plans.size()) throw std::runtime_error("peer-plan output too small");
    for (size_t i = 0; i < runtime->peer_plans.size(); ++i) {
      const auto& p = runtime->peer_plans[i]; uint64_t* r = output + i * 7;
      r[0]=i/runtime->layout.world_size; r[1]=i%runtime->layout.world_size;
      r[2]=p.token_count; r[3]=p.route_id; r[4]=p.src_base_offset;
      r[5]=p.dst_base_offset; r[6]=p.flags;
    }
  });
}

int r6_m9_copy_shadow_actions(void* opaque, int64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->shadow_actions;
    if (rows < values.size()) throw std::runtime_error("shadow output too small");
    for (size_t i = 0; i < values.size(); ++i) {
      const auto& a = values[i]; int64_t* r = output + i * 12;
      r[0]=a.action_id; r[1]=a.descriptor_id; r[2]=a.chunk_id;
      r[3]=a.reveal_epoch; r[4]=a.src_rank; r[5]=a.dst_rank;
      r[6]=a.src_offset; r[7]=a.dst_offset; r[8]=a.token_count;
      r[9]=a.bytes; r[10]=a.route_id; r[11]=a.flags;
    }
  });
}

int r6_m9_copy_traces(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->traces;
    if (rows < values.size()) throw std::runtime_error("trace output too small");
    for (size_t i = 0; i < values.size(); ++i) {
      const auto& t = values[i]; uint64_t* r = output + i * 11;
      r[0]=t.commit_id; r[1]=t.descriptor_id; r[2]=t.peer;
      r[3]=t.token_count; r[4]=t.is_remote; r[5]=t.bytes;
      r[6]=t.t_commit_consumed; r[7]=t.t_dispatch_start;
      r[8]=t.t_dispatch_end; r[9]=t.t_remote_completion; r[10]=t.error_code;
    }
  });
}

int r6_m9_copy_timings(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->timings;
    if (rows < values.size()) throw std::runtime_error("timing output too small");
    for (size_t i = 0; i < values.size(); ++i) {
      const auto& t = values[i]; uint64_t* r = output + i * 8;
      r[0]=t.chunk_id; r[1]=t.descriptor_id; r[2]=t.t_router_reveal;
      r[3]=t.t_scheduler_commit; r[4]=t.t_dispatch_start;
      r[5]=t.t_dispatch_end; r[6]=t.t_remote_completion;
      r[7]=t.t_final_router_completion;
    }
  });
}

int r6_m9_copy_expert_counts(void* opaque, uint32_t* output, size_t values) {
  return protect([&] {
    const auto& source = static_cast<Runtime*>(opaque)->expert_counts;
    if (values < source.size()) throw std::runtime_error("expert-count output too small");
    std::memcpy(output, source.data(), source.size() * sizeof(uint32_t));
  });
}

int r6_m9_copy_expert_offsets(void* opaque, uint32_t* output, size_t values) {
  return protect([&] {
    const auto& source = static_cast<Runtime*>(opaque)->expert_offsets;
    if (values < source.size()) throw std::runtime_error("expert-offset output too small");
    std::memcpy(output, source.data(), source.size() * sizeof(uint32_t));
  });
}

int r6_m9_copy_recv_x(void* opaque, float* output, size_t elements) {
  return protect([&] {
    const auto* runtime = static_cast<Runtime*>(opaque);
    size_t required = static_cast<size_t>(runtime->handle.num_recv_tokens) * runtime->layout.feature_width;
    if (elements < required) throw std::runtime_error("recv_x output too small");
    if (required) CUDA_CHECK(cudaMemcpy(output, runtime->d_recv_x, required * sizeof(float), cudaMemcpyDeviceToHost));
  });
}

int r6_m9_copy_recv_metadata(void* opaque, void* output, size_t rows) {
  return protect([&] {
    const auto* runtime = static_cast<Runtime*>(opaque);
    if (rows < runtime->handle.num_recv_tokens) throw std::runtime_error("metadata output too small");
    if (runtime->handle.num_recv_tokens) CUDA_CHECK(cudaMemcpy(
        output, runtime->d_recv_metadata,
        runtime->handle.num_recv_tokens * sizeof(DispatchTokenMeta),
        cudaMemcpyDeviceToHost));
  });
}

int r6_m9_copy_return_traces(void* opaque, uint64_t* output, size_t rows) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->return_traces;
    if (rows < values.size()) throw std::runtime_error("return-trace output too small");
    for (size_t i = 0; i < values.size(); ++i) {
      const auto& t = values[i]; uint64_t* r = output + i * 11;
      r[0]=t.row; r[1]=t.src_rank; r[2]=t.src_token_idx;
      r[3]=t.topk_slot; r[4]=t.expert_id; r[5]=t.is_remote;
      r[6]=t.bytes; r[7]=t.t_return_start; r[8]=t.t_return_end;
      r[9]=t.t_remote_completion; r[10]=t.error_code;
    }
  });
}

int r6_m9_copy_final_output(void* opaque, float* output, size_t elements) {
  return protect([&] {
    const auto& values = static_cast<Runtime*>(opaque)->final_output;
    if (elements < values.size()) throw std::runtime_error("final-output buffer too small");
    if (!values.empty()) std::memcpy(output, values.data(), values.size() * sizeof(float));
  });
}

int r6_m9_copy_gate_timing(void* opaque, uint64_t* output, size_t values) {
  return protect([&] {
    if (values < 4) throw std::runtime_error("gate-timing output too small");
    const auto& t = static_cast<Runtime*>(opaque)->gate_timing;
    output[0]=t.t_gate_start; output[1]=t.t_final_router_seen;
    output[2]=t.t_first_commit_forwarded; output[3]=t.t_gate_complete;
  });
}

int r6_m9_copy_stage_timing(void* opaque, uint64_t* output, size_t values) {
  return protect([&] {
    if (values < 6) throw std::runtime_error("stage-timing output too small");
    const auto& t = static_cast<Runtime*>(opaque)->stage_timing;
    output[0]=t.router_start; output[1]=t.expert_start;
    output[2]=t.expert_complete; output[3]=t.combine_start;
    output[4]=t.combine_complete; output[5]=t.final_output_ready;
  });
}

void r6_m9_destroy(void* opaque) { delete static_cast<Runtime*>(opaque); }

}  // extern "C"
