// R6-M7: revealed-only DescriptorCommit scheduler -> fused token-centric
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
#include "rlccl/ep/common/progressive_ep_ir.h"
#include "rlccl/ep/cuda/dispatch_epilogue.cuh"
#include "rlccl/ep/cuda/progressive_dispatch.cuh"
#include "rlccl/ep/cuda/staged_gin_dispatch.cuh"
#include "rlccl/scheduler/common/scheduler_ir.h"

// Keep the reusable data-plane implementation in rlccl/ep.  The include makes
// this standalone extension buildable with one nvcc command on the validation
// host, while the symbols remain in the EP namespace.
#include "rlccl/ep/cuda/progressive_dispatch.cu"

namespace rlccl::ep::m7_runtime {

using namespace rlccl::scheduler;
using namespace rlccl::ep;
using namespace rlccl::ep::cuda_backend;

thread_local std::string last_error;
constexpr uint32_t kRuntimeMaxWorldSize = 16;

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

__global__ void m7_pipeline_kernel(
    const int64_t* reveal_records, uint32_t record_count,
    const uint8_t* plan_blob, const float* x, const int64_t* topk_idx,
    const float* topk_weights, uint32_t num_tokens, uint32_t feature_width,
    uint32_t num_topk, uint32_t experts_per_rank, uint32_t num_experts,
    uint32_t rank, DeviceRevealQueue* reveal_queue,
    DeviceCommitQueue* commit_queue, RevealRecord* records_by_descriptor,
    CommitPeerPlan* peer_plans, CommittedAction* shadow_actions,
    uint32_t shadow_capacity, uint32_t* shadow_count,
    M7SchedulerState* scheduler_state, DeviceCommitState* commit_state,
    DispatchLayout layout, ncclDevComm dev_comm, ncclWindow_t window,
    uint8_t* registered_buffer, uint32_t* dst_cursors,
    DispatchTrace* traces, uint32_t trace_capacity, uint32_t* trace_count,
    DispatchTiming* timings, M7Counters* counters,
    uint64_t* final_router_ns, uint64_t delay_cycles) {
  if (blockIdx.x == 0) {
    router_reveal_role(reveal_records, record_count, reveal_queue,
                       records_by_descriptor, timings, final_router_ns,
                       delay_cycles, counters);
  } else if (blockIdx.x == 1) {
    descriptor_commit_scheduler_role(
        plan_blob, topk_idx, num_tokens, num_topk, experts_per_rank,
        num_experts, rank, record_count, reveal_queue, commit_queue,
        peer_plans, shadow_actions, shadow_capacity, shadow_count,
        scheduler_state, commit_state, layout, timings, counters);
  } else if (blockIdx.x == 2) {
    ProgressiveDispatchInput input{
        x, topk_idx, topk_weights, num_tokens, feature_width, num_topk,
        experts_per_rank, num_experts};
    progressive_dispatch_progress_role(
        commit_queue, peer_plans, records_by_descriptor, commit_state, input,
        layout, rank, dev_comm, window, registered_buffer, dst_cursors,
        traces, trace_capacity, trace_count, timings, counters);
  } else if (blockIdx.x == 3) {
    progressive_dispatch_wait_role(dev_comm, record_count, timings, counters);
  }
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

struct Runtime {
  int rank;
  int device;
  CompiledPlanBlobHeader plan_header{};
  DispatchLayout layout{};
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
  CommitPeerPlan* d_peer_plans = nullptr;
  CommittedAction* d_shadow_actions = nullptr;
  DeviceRevealQueue* d_reveal_queue = nullptr;
  DeviceCommitQueue* d_commit_queue = nullptr;
  M7SchedulerState* d_scheduler_state = nullptr;
  DeviceCommitState* d_commit_state = nullptr;
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
  cudaStream_t pipeline_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
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

  Runtime(int rank_, int device_, const ncclUniqueId* unique_id,
          const void* plan_blob, size_t plan_bytes, DispatchLayout layout_)
      : rank(rank_), device(device_), layout(layout_) {
    CUDA_CHECK(cudaSetDevice(device));
    if (!unique_id) throw std::runtime_error("M7 NCCL unique ID is null");
    if (plan_bytes < sizeof(CompiledPlanBlobHeader))
      throw std::runtime_error("M7 plan header truncated");
    std::memcpy(&plan_header, plan_blob, sizeof(plan_header));
    registered_bytes = layout.capacity_bytes;
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
    requirements.lsaBarrierCount = layout.max_descriptors;
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
    d_peer_plans = allocate_device<CommitPeerPlan>(layout.max_descriptors * layout.world_size);
    d_shadow_actions = allocate_device<CommittedAction>(shadow_capacity);
    d_reveal_queue = allocate_device<DeviceRevealQueue>();
    d_commit_queue = allocate_device<DeviceCommitQueue>();
    d_scheduler_state = allocate_device<M7SchedulerState>();
    d_commit_state = allocate_device<DeviceCommitState>();
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
    CUDA_CHECK(cudaMemcpy(d_plan, plan_blob, plan_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreateWithFlags(&pipeline_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
  }

  ~Runtime() {
    if (input_ready) cudaEventDestroy(input_ready);
    if (pipeline_stream) cudaStreamDestroy(pipeline_stream);
    cudaFree(d_plan); cudaFree(d_reveals); cudaFree(d_records_by_descriptor);
    cudaFree(d_commits); cudaFree(d_peer_plans); cudaFree(d_shadow_actions);
    cudaFree(d_reveal_queue); cudaFree(d_commit_queue); cudaFree(d_scheduler_state);
    cudaFree(d_commit_state); cudaFree(d_revealed_count); cudaFree(d_committed_count);
    cudaFree(d_descriptor_epoch); cudaFree(d_shadow_count); cudaFree(d_dst_cursors);
    cudaFree(d_traces); cudaFree(d_trace_count); cudaFree(d_timings);
    cudaFree(d_counters); cudaFree(d_final_router_ns); cudaFree(d_expert_counts);
    cudaFree(d_expert_offsets); cudaFree(d_expert_cursors); cudaFree(d_recv_x);
    cudaFree(d_recv_metadata); cudaFree(d_handle);
    if (dev_comm_created) ncclDevCommDestroy(comm, &dev_comm);
    if (window) ncclCommWindowDeregister(comm, window);
    if (registered_buffer) ncclMemFree(registered_buffer);
    if (comm) ncclCommDestroy(comm);
  }

  void run(const int64_t* reveal_records, uint32_t record_count,
           const float* x, const int64_t* topk_idx, const float* topk_weights,
           uint32_t num_tokens, uint32_t feature_width, uint32_t num_topk,
           uint32_t experts_per_rank, uint32_t num_experts,
           uint64_t delay_cycles, cudaStream_t router_stream) {
    if (feature_width != layout.feature_width || num_topk == 0 ||
        num_experts != experts_per_rank * layout.world_size ||
        record_count > layout.max_descriptors)
      throw std::runtime_error("M7 run shape/layout mismatch");
    num_local_experts = experts_per_rank;
    d_expert_counts = allocate_device<uint32_t>(num_local_experts);
    d_expert_offsets = allocate_device<uint32_t>(num_local_experts + 1);
    d_expert_cursors = allocate_device<uint32_t>(num_local_experts);
    d_recv_x = allocate_device<float>(static_cast<size_t>(max_recv_records) * feature_width);
    d_recv_metadata = allocate_device<DispatchTokenMeta>(max_recv_records);
    d_handle = allocate_device<ProgressiveEPHandle>();

    CUDA_CHECK(cudaMemset(registered_buffer, 0, registered_bytes));
    CUDA_CHECK(cudaMemset(d_reveals, 0, sizeof(RevealRecord) * queue_capacity));
    CUDA_CHECK(cudaMemset(d_records_by_descriptor, 0, sizeof(RevealRecord) * layout.max_descriptors));
    CUDA_CHECK(cudaMemset(d_commits, 0, sizeof(DescriptorCommit) * queue_capacity));
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
    DeviceRevealQueue reveal_queue{d_reveals, queue_capacity, 0, 0};
    DeviceCommitQueue commit_queue{d_commits, queue_capacity, 0, 0};
    M7SchedulerState scheduler_state{
        d_revealed_count, d_committed_count, d_descriptor_epoch, 0, 0, 0, 0};
    DeviceCommitState commit_state{0, 0};
    ProgressiveEPHandle initial_handle{
        0, num_local_experts, num_topk, d_expert_counts, d_expert_offsets,
        d_recv_metadata, 1};
    CUDA_CHECK(cudaMemcpy(d_reveal_queue, &reveal_queue, sizeof(reveal_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_commit_queue, &commit_queue, sizeof(commit_queue), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scheduler_state, &scheduler_state, sizeof(scheduler_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_commit_state, &commit_state, sizeof(commit_state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_handle, &initial_handle, sizeof(initial_handle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(input_ready, router_stream));
    CUDA_CHECK(cudaStreamWaitEvent(pipeline_stream, input_ready, 0));
    m7_pipeline_kernel<<<4, 256, 0, pipeline_stream>>>(
        reveal_records, record_count, d_plan, x, topk_idx, topk_weights,
        num_tokens, feature_width, num_topk, experts_per_rank, num_experts,
        rank, d_reveal_queue, d_commit_queue, d_records_by_descriptor,
        d_peer_plans, d_shadow_actions, shadow_capacity, d_shadow_count,
        d_scheduler_state, d_commit_state, layout, dev_comm, window,
        registered_buffer, d_dst_cursors, d_traces,
        layout.max_descriptors * layout.world_size, d_trace_count,
        d_timings, d_counters, d_final_router_ns, delay_cycles);
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(pipeline_stream));

    DeviceCommitQueue commit_queue_out{};
    uint32_t shadow_count_out = 0, trace_count_out = 0;
    uint64_t final_router = 0;
    CUDA_CHECK(cudaMemcpy(&commit_queue_out, d_commit_queue, sizeof(commit_queue_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&shadow_count_out, d_shadow_count, sizeof(shadow_count_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&trace_count_out, d_trace_count, sizeof(trace_count_out), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&handle, d_handle, sizeof(handle), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&counters, d_counters, sizeof(counters), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&final_router, d_final_router_ns, sizeof(final_router), cudaMemcpyDeviceToHost));
    commits.resize(commit_queue_out.tail);
    peer_plans.resize(layout.max_descriptors * layout.world_size);
    shadow_actions.resize(shadow_count_out);
    traces.resize(trace_count_out);
    timings.resize(record_count);
    expert_counts.resize(num_local_experts);
    expert_offsets.resize(num_local_experts + 1);
    if (!commits.empty()) CUDA_CHECK(cudaMemcpy(commits.data(), d_commits, commits.size() * sizeof(DescriptorCommit), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(peer_plans.data(), d_peer_plans, peer_plans.size() * sizeof(CommitPeerPlan), cudaMemcpyDeviceToHost));
    if (!shadow_actions.empty()) CUDA_CHECK(cudaMemcpy(shadow_actions.data(), d_shadow_actions, shadow_actions.size() * sizeof(CommittedAction), cudaMemcpyDeviceToHost));
    if (!traces.empty()) CUDA_CHECK(cudaMemcpy(traces.data(), d_traces, traces.size() * sizeof(DispatchTrace), cudaMemcpyDeviceToHost));
    if (!timings.empty()) CUDA_CHECK(cudaMemcpy(timings.data(), d_timings, timings.size() * sizeof(DispatchTiming), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_counts.data(), d_expert_counts, expert_counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_offsets.data(), d_expert_offsets, expert_offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (auto& timing : timings) timing.t_final_router_completion = final_router;
    for (auto& trace : traces)
      trace.t_remote_completion = timings[trace.descriptor_id].t_remote_completion;
  }
};

}  // namespace rlccl::ep::m7_runtime

using namespace rlccl::ep::m7_runtime;

extern "C" {

const char* r6_m7_last_error() { return last_error.c_str(); }
size_t r6_m7_unique_id_size() { return sizeof(ncclUniqueId); }

int r6_m7_get_unique_id(void* output, size_t bytes) {
  return protect([&] {
    if (!output || bytes != sizeof(ncclUniqueId))
      throw std::runtime_error("NCCL unique ID output size mismatch");
    NCCL_CHECK(ncclGetUniqueId(static_cast<ncclUniqueId*>(output)));
  });
}

void* r6_m7_create(
    int rank, int device, const void* unique_id, size_t unique_id_bytes,
    const void* plan_blob, size_t plan_bytes, uint32_t world_size,
    uint32_t max_descriptors, uint32_t max_assignments_per_peer,
    uint32_t feature_width, uint64_t peer_stride,
    uint64_t descriptor_stride, uint64_t region_bytes) {
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
        runtime = new Runtime(
            rank, device, static_cast<const ncclUniqueId*>(unique_id),
            plan_blob, plan_bytes, layout);
      }) != 0) return nullptr;
  return runtime;
}

int r6_m7_run(
    void* opaque, const int64_t* reveal_records, uint32_t record_count,
    const float* x, const int64_t* topk_idx, const float* topk_weights,
    uint32_t num_tokens, uint32_t feature_width, uint32_t num_topk,
    uint32_t experts_per_rank, uint32_t num_experts,
    uint64_t delay_cycles, uintptr_t router_stream) {
  return protect([&] {
    static_cast<Runtime*>(opaque)->run(
        reveal_records, record_count, x, topk_idx, topk_weights,
        num_tokens, feature_width, num_topk, experts_per_rank, num_experts,
        delay_cycles, reinterpret_cast<cudaStream_t>(router_stream));
  });
}

uint64_t r6_m7_counter(void* opaque, int index) {
  const auto& value = static_cast<Runtime*>(opaque)->counters;
  const uint64_t* begin = &value.descriptor_commits;
  return index >= 0 && index < 15 ? begin[index] : 0;
}

uint64_t r6_m7_capability(void* opaque, int index) {
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

size_t r6_m7_commit_count(void* opaque) { return static_cast<Runtime*>(opaque)->commits.size(); }
size_t r6_m7_peer_plan_count(void* opaque) { return static_cast<Runtime*>(opaque)->peer_plans.size(); }
size_t r6_m7_shadow_action_count(void* opaque) { return static_cast<Runtime*>(opaque)->shadow_actions.size(); }
size_t r6_m7_trace_count(void* opaque) { return static_cast<Runtime*>(opaque)->traces.size(); }
size_t r6_m7_timing_count(void* opaque) { return static_cast<Runtime*>(opaque)->timings.size(); }
size_t r6_m7_num_recv_tokens(void* opaque) { return static_cast<Runtime*>(opaque)->handle.num_recv_tokens; }
size_t r6_m7_num_local_experts(void* opaque) { return static_cast<Runtime*>(opaque)->num_local_experts; }

int r6_m7_copy_commits(void* opaque, uint64_t* output, size_t rows) {
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

int r6_m7_copy_peer_plans(void* opaque, uint64_t* output, size_t rows) {
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

int r6_m7_copy_shadow_actions(void* opaque, int64_t* output, size_t rows) {
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

int r6_m7_copy_traces(void* opaque, uint64_t* output, size_t rows) {
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

int r6_m7_copy_timings(void* opaque, uint64_t* output, size_t rows) {
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

int r6_m7_copy_expert_counts(void* opaque, uint32_t* output, size_t values) {
  return protect([&] {
    const auto& source = static_cast<Runtime*>(opaque)->expert_counts;
    if (values < source.size()) throw std::runtime_error("expert-count output too small");
    std::memcpy(output, source.data(), source.size() * sizeof(uint32_t));
  });
}

int r6_m7_copy_expert_offsets(void* opaque, uint32_t* output, size_t values) {
  return protect([&] {
    const auto& source = static_cast<Runtime*>(opaque)->expert_offsets;
    if (values < source.size()) throw std::runtime_error("expert-offset output too small");
    std::memcpy(output, source.data(), source.size() * sizeof(uint32_t));
  });
}

int r6_m7_copy_recv_x(void* opaque, float* output, size_t elements) {
  return protect([&] {
    const auto* runtime = static_cast<Runtime*>(opaque);
    size_t required = static_cast<size_t>(runtime->handle.num_recv_tokens) * runtime->layout.feature_width;
    if (elements < required) throw std::runtime_error("recv_x output too small");
    if (required) CUDA_CHECK(cudaMemcpy(output, runtime->d_recv_x, required * sizeof(float), cudaMemcpyDeviceToHost));
  });
}

int r6_m7_copy_recv_metadata(void* opaque, void* output, size_t rows) {
  return protect([&] {
    const auto* runtime = static_cast<Runtime*>(opaque);
    if (rows < runtime->handle.num_recv_tokens) throw std::runtime_error("metadata output too small");
    if (runtime->handle.num_recv_tokens) CUDA_CHECK(cudaMemcpy(
        output, runtime->d_recv_metadata,
        runtime->handle.num_recv_tokens * sizeof(DispatchTokenMeta),
        cudaMemcpyDeviceToHost));
  });
}

void r6_m7_destroy(void* opaque) { delete static_cast<Runtime*>(opaque); }

}  // extern "C"
