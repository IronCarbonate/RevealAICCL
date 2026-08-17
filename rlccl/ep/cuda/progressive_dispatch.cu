#include "progressive_dispatch.cuh"

#include <cuda/atomic>

#include "arch_copy.cuh"
#include "dispatch_epilogue.cuh"

namespace rlccl::ep::cuda_backend {

namespace {

constexpr uint32_t kMaxWorldSize = 16;

__device__ __forceinline__ uint64_t global_time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ uint8_t* direct_peer_base(
    uint32_t peer, uint32_t rank, ncclWindow_t window,
    uint8_t* registered_buffer, uint64_t offset) {
  if (peer == rank) return registered_buffer + offset;
  return static_cast<uint8_t*>(ncclGetPeerPointer(window, offset, peer));
}

}  // namespace

__device__ void progressive_dispatch_progress_role(
    DeviceCommitQueue* commit_queue,
    const rlccl::ep::CommitPeerPlan* peer_plans,
    const rlccl::scheduler::RevealRecord* records_by_descriptor,
    DeviceCommitState* state,
    ProgressiveDispatchInput input,
    DispatchLayout layout,
    uint32_t rank,
    ncclDevComm dev_comm,
    ncclWindow_t window,
    uint8_t* registered_buffer,
    uint32_t* dst_cursors,
    rlccl::ep::DispatchTrace* traces,
    uint32_t trace_capacity,
    uint32_t* trace_count,
    rlccl::ep::DispatchTiming* timings,
    rlccl::ep::M7Counters* counters) {
  __shared__ rlccl::ep::DescriptorCommit commit;
  __shared__ uint8_t* remote_bases[kMaxWorldSize];
  __shared__ uint32_t peer_counts[kMaxWorldSize];
  __shared__ uint32_t has_commit;
  __shared__ uint32_t should_exit;
  __shared__ uint32_t commit_error;
  __shared__ uint64_t dispatch_start;

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> head_ref(commit_queue->head);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tail_ref(commit_queue->tail);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> done_ref(state->scheduler_done);

  while (true) {
    if (threadIdx.x == 0) {
      uint32_t head = head_ref.load(cuda::memory_order_relaxed);
      uint32_t tail = tail_ref.load(cuda::memory_order_acquire);
      has_commit = head != tail;
      should_exit = !has_commit && done_ref.load(cuda::memory_order_acquire);
      commit_error = rlccl::ep::kEpOk;
      if (has_commit) {
        commit = commit_queue->commits[head % commit_queue->capacity];
        head_ref.store(head + 1, cuda::memory_order_release);
        dispatch_start = global_time_ns();
        timings[commit.descriptor_id].t_dispatch_start = dispatch_start;
      }
    }
    __syncthreads();
    if (should_exit) break;
    if (!has_commit) {
      __nanosleep(64);
      __syncthreads();
      continue;
    }

    if (layout.world_size > kMaxWorldSize || input.num_topk == 0 ||
        commit.descriptor_id >= layout.max_descriptors ||
        commit.assignment_count != commit.token_count * input.num_topk ||
        commit.token_begin + commit.token_count > input.num_tokens) {
      if (threadIdx.x == 0) commit_error = rlccl::ep::kEpCommitInvalid;
    }
    __syncthreads();

    for (uint32_t peer = threadIdx.x; peer < layout.world_size; peer += blockDim.x) {
      const auto& plan = peer_plans[commit.descriptor_id * layout.world_size + peer];
      peer_counts[peer] = plan.token_count;
      remote_bases[peer] = nullptr;
      if (plan.token_count) {
        uint8_t* base = direct_peer_base(
            peer, rank, window, registered_buffer, plan.dst_base_offset);
        remote_bases[peer] = base;
        if (!base) {
          atomicCAS(&commit_error, rlccl::ep::kEpOk, rlccl::ep::kEpPeerNotDirect);
        } else {
          if (threadIdx.x == peer) {
            *reinterpret_cast<uint64_t*>(base) = plan.token_count;
            *reinterpret_cast<uint64_t*>(base + 8) = commit.commit_id;
          }
        }
      }
    }
    __syncthreads();

    if (commit_error == rlccl::ep::kEpOk) {
      for (uint32_t local = threadIdx.x; local < commit.assignment_count;
           local += blockDim.x) {
        uint32_t token_delta = local / input.num_topk;
        uint32_t topk_slot = local % input.num_topk;
        uint32_t token = commit.token_begin + token_delta;
        uint64_t topk_offset = static_cast<uint64_t>(token) * input.num_topk + topk_slot;
        int64_t expert64 = input.topk_idx[topk_offset];
        if (expert64 < 0 || static_cast<uint64_t>(expert64) >= input.num_experts) {
          atomicCAS(&commit_error, rlccl::ep::kEpOk, rlccl::ep::kEpExpertRange);
          continue;
        }
        uint32_t expert = static_cast<uint32_t>(expert64);
        uint32_t destination = expert / input.experts_per_rank;
        if (destination >= layout.world_size ||
            !(commit.authorized_dst_mask & (uint64_t(1) << destination))) {
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &counters->unauthorized_destination), 1ULL);
          atomicCAS(&commit_error, rlccl::ep::kEpOk,
                    rlccl::ep::kEpUnauthorizedDestination);
          continue;
        }
        uint32_t cursor_index = commit.descriptor_id * layout.world_size + destination;
        uint32_t slot = atomicAdd(dst_cursors + cursor_index, 1U);
        if (slot >= peer_counts[destination] || !remote_bases[destination]) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&counters->cursor_overflow), 1ULL);
          atomicCAS(&commit_error, rlccl::ep::kEpOk, rlccl::ep::kEpCursorOverflow);
          continue;
        }
        uint8_t* record = remote_bases[destination] + kDispatchSlotHeaderBytes +
                          static_cast<uint64_t>(slot) * layout.record_bytes;
        rlccl::ep::DispatchTokenMeta meta{
            rank, token, expert, topk_slot, commit.descriptor_id,
            commit.reveal_epoch, input.topk_weights[topk_offset]};
        *reinterpret_cast<rlccl::ep::DispatchTokenMeta*>(record) = meta;
        *reinterpret_cast<uint32_t*>(record + sizeof(meta)) = 0;
        copy_feature_vectorized(
            record + kDispatchMetaStorageBytes,
            input.x + static_cast<uint64_t>(token) * input.feature_width,
            input.feature_width * sizeof(float));
        atomicAdd(reinterpret_cast<unsigned long long*>(&counters->assignments_scanned), 1ULL);
        if (destination == rank) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&counters->local_records), 1ULL);
        } else {
          atomicAdd(reinterpret_cast<unsigned long long*>(&counters->direct_remote_records), 1ULL);
          atomicAdd(reinterpret_cast<unsigned long long*>(&counters->direct_remote_bytes),
                    static_cast<unsigned long long>(layout.record_bytes));
        }
      }
    }
    __syncthreads();
    __threadfence_system();

    ncclLsaBarrierSession<ncclCoopCta> completion{
        ncclCoopCta(), dev_comm, ncclTeamTagLsa(), commit.descriptor_id};
    completion.arrive(ncclCoopCta(), cuda::memory_order_release);

    if (threadIdx.x == 0) {
      uint64_t dispatch_end = global_time_ns();
      timings[commit.descriptor_id].t_dispatch_end = dispatch_end;
      counters->lsa_arrives += 1;
      counters->errors += commit_error != rlccl::ep::kEpOk;
      for (uint32_t peer = 0; peer < layout.world_size; ++peer) {
        if (!peer_counts[peer]) continue;
        uint32_t index = atomicAdd(trace_count, 1U);
        if (index < trace_capacity) {
          traces[index] = rlccl::ep::DispatchTrace{
              commit.commit_id, commit.descriptor_id, peer, peer_counts[peer],
              peer != rank,
              kDispatchSlotHeaderBytes +
                  static_cast<uint64_t>(peer_counts[peer]) * layout.record_bytes,
              dispatch_start, dispatch_start, dispatch_end, 0, commit_error, 0};
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    __threadfence();
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> dispatch_done(state->dispatch_done);
    dispatch_done.store(1, cuda::memory_order_release);
  }
}

__device__ void progressive_dispatch_wait_role(
    ncclDevComm dev_comm, uint32_t descriptor_count,
    rlccl::ep::DispatchTiming* timings, rlccl::ep::M7Counters* counters) {
  for (uint32_t descriptor = 0; descriptor < descriptor_count; ++descriptor) {
    ncclLsaBarrierSession<ncclCoopCta> completion{
        ncclCoopCta(), dev_comm, ncclTeamTagLsa(), descriptor};
    completion.wait(ncclCoopCta(), cuda::memory_order_acquire);
    if (threadIdx.x == 0) {
      timings[descriptor].t_remote_completion = global_time_ns();
      counters->lsa_waits += 1;
    }
  }
}

__global__ void progressive_dispatch_progress_kernel(
    DeviceCommitQueue* commit_queue,
    const rlccl::ep::CommitPeerPlan* peer_plans,
    const rlccl::scheduler::RevealRecord* records_by_descriptor,
    DeviceCommitState* state,
    ProgressiveDispatchInput input,
    DispatchLayout layout,
    uint32_t rank,
    ncclDevComm dev_comm,
    ncclWindow_t window,
    uint8_t* registered_buffer,
    uint32_t* dst_cursors,
    rlccl::ep::DispatchTrace* traces,
    uint32_t trace_capacity,
    uint32_t* trace_count,
    rlccl::ep::DispatchTiming* timings,
    rlccl::ep::M7Counters* counters) {
  progressive_dispatch_progress_role(
      commit_queue, peer_plans, records_by_descriptor, state, input, layout,
      rank, dev_comm, window, registered_buffer, dst_cursors, traces,
      trace_capacity, trace_count, timings, counters);
}

__global__ void progressive_dispatch_wait_kernel(
    ncclDevComm dev_comm, uint32_t descriptor_count,
    rlccl::ep::DispatchTiming* timings, rlccl::ep::M7Counters* counters) {
  progressive_dispatch_wait_role(dev_comm, descriptor_count, timings, counters);
}

__global__ void dispatch_count_experts_kernel(
    const uint8_t* registered_buffer, DispatchLayout layout,
    uint32_t rank, uint32_t experts_per_rank, uint32_t* expert_counts,
    rlccl::ep::M7Counters* counters) {
  uint32_t slot_index = blockIdx.x;
  uint32_t total_slots = layout.max_descriptors * layout.world_size;
  if (slot_index >= total_slots) return;
  uint32_t descriptor = slot_index / layout.world_size;
  uint32_t source = slot_index % layout.world_size;
  const uint8_t* slot = registered_buffer + layout.recv_offset(descriptor, source);
  uint64_t count = *reinterpret_cast<const uint64_t*>(slot);
  if (count > layout.max_assignments_per_peer) return;
  for (uint32_t record_index = threadIdx.x; record_index < count;
       record_index += blockDim.x) {
    const uint8_t* record = slot + kDispatchSlotHeaderBytes +
                            static_cast<uint64_t>(record_index) * layout.record_bytes;
    const auto* meta = reinterpret_cast<const rlccl::ep::DispatchTokenMeta*>(record);
    uint32_t first_expert = rank * experts_per_rank;
    if (meta->expert_id < first_expert ||
        meta->expert_id >= first_expert + experts_per_rank) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&counters->errors), 1ULL);
      continue;
    }
    atomicAdd(expert_counts + (meta->expert_id - first_expert), 1U);
  }
}

__global__ void dispatch_exclusive_scan_kernel(
    uint32_t* expert_counts, uint32_t* expert_offsets,
    uint32_t* expert_cursors, uint32_t num_local_experts,
    rlccl::ep::ProgressiveEPHandle* handle, uint32_t generation) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  uint32_t running = 0;
  for (uint32_t expert = 0; expert < num_local_experts; ++expert) {
    expert_offsets[expert] = running;
    expert_cursors[expert] = running;
    running += expert_counts[expert];
  }
  expert_offsets[num_local_experts] = running;
  handle->num_recv_tokens = running;
  handle->num_local_experts = num_local_experts;
  handle->expert_counts = expert_counts;
  handle->expert_offsets = expert_offsets;
  handle->generation = generation;
}

__global__ void dispatch_scatter_experts_kernel(
    const uint8_t* registered_buffer, DispatchLayout layout,
    uint32_t rank, uint32_t experts_per_rank, uint32_t* expert_cursors,
    float* recv_x, rlccl::ep::DispatchTokenMeta* recv_metadata,
    rlccl::ep::M7Counters* counters) {
  uint32_t slot_index = blockIdx.x;
  uint32_t total_slots = layout.max_descriptors * layout.world_size;
  if (slot_index >= total_slots) return;
  uint32_t descriptor = slot_index / layout.world_size;
  uint32_t source = slot_index % layout.world_size;
  const uint8_t* slot = registered_buffer + layout.recv_offset(descriptor, source);
  uint64_t count = *reinterpret_cast<const uint64_t*>(slot);
  if (count > layout.max_assignments_per_peer) return;
  for (uint32_t record_index = threadIdx.x; record_index < count;
       record_index += blockDim.x) {
    const uint8_t* record = slot + kDispatchSlotHeaderBytes +
                            static_cast<uint64_t>(record_index) * layout.record_bytes;
    const auto* meta = reinterpret_cast<const rlccl::ep::DispatchTokenMeta*>(record);
    uint32_t first_expert = rank * experts_per_rank;
    if (meta->expert_id < first_expert ||
        meta->expert_id >= first_expert + experts_per_rank) continue;
    uint32_t local_expert = meta->expert_id - first_expert;
    uint32_t output = atomicAdd(expert_cursors + local_expert, 1U);
    recv_metadata[output] = *meta;
    copy_feature_vectorized(
        recv_x + static_cast<uint64_t>(output) * layout.feature_width,
        record + kDispatchMetaStorageBytes,
        layout.feature_width * sizeof(float));
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->epilogue_records), 1ULL);
  }
}

}  // namespace rlccl::ep::cuda_backend
