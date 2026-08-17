#include "progressive_combine.cuh"

#include <cuda/atomic>

#include "../../transport/cuda/device_transport.cuh"
#include "arch_copy.cuh"
#include "combine_epilogue.cuh"

namespace rlccl::ep::cuda_backend {

namespace {

__device__ __forceinline__ uint64_t combine_time_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ void increment(uint64_t* value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(value), 1ULL);
}

struct NcclLsaDirectTransport {
  uint32_t rank;
  uint32_t world_size;
  ncclWindow_t window;
  uint8_t* local_buffer;

  __device__ bool is_direct(int32_t peer) const {
    return peer >= 0 && static_cast<uint32_t>(peer) < world_size &&
           (static_cast<uint32_t>(peer) == rank ||
            ncclGetPeerPointer(window, 0, peer) != nullptr);
  }

  __device__ uint8_t* get_remote_ptr(
      int32_t peer, uint64_t dst_offset) const {
    if (static_cast<uint32_t>(peer) == rank) return local_buffer + dst_offset;
    return static_cast<uint8_t*>(ncclGetPeerPointer(window, dst_offset, peer));
  }
};

}  // namespace

__global__ void build_combine_ranges_kernel(
    const rlccl::ep::ProgressiveEPHandle* handle, uint32_t rank,
    rlccl::ep::CombineRange* ranges) {
  uint32_t local_expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_expert >= handle->num_local_experts) return;
  uint32_t begin = handle->expert_offsets[local_expert];
  uint32_t end = handle->expert_offsets[local_expert + 1];
  ranges[local_expert] = rlccl::ep::CombineRange{
      begin, end - begin,
      rank * handle->num_local_experts + local_expert,
      handle->generation, rlccl::ep::kCombineLsaDirect};
}

__global__ void progressive_combine_kernel(
    const rlccl::ep::ProgressiveEPHandle* handle,
    const rlccl::ep::CombineRange* ranges, uint32_t range_count,
    const float* expert_output, CombineLayout layout,
    uint32_t rank, uint32_t world_size, uint32_t expected_generation,
    ncclDevComm dev_comm, ncclWindow_t window,
    uint8_t* registered_buffer, uint32_t completion_id,
    rlccl::ep::ReturnTrace* traces, uint32_t trace_capacity,
    uint32_t* trace_count, rlccl::ep::M8CombineCounters* counters) {
  NcclLsaDirectTransport transport{rank, world_size, window, registered_buffer};
  bool handle_valid = handle->generation == expected_generation &&
      handle->num_source_tokens == layout.num_source_tokens &&
      handle->num_topk == layout.num_topk;
  if (!handle_valid && threadIdx.x == 0) {
    increment(&counters->stale_handle);
    increment(&counters->errors);
  }
  __syncthreads();

  if (handle_valid) {
    for (uint32_t range_index = 0; range_index < range_count; ++range_index) {
      rlccl::ep::CombineRange range = ranges[range_index];
      bool range_valid = range.generation == expected_generation &&
          static_cast<uint64_t>(range.row_begin) + range.row_count <=
              handle->num_recv_tokens;
      if (!range_valid) {
        if (threadIdx.x == 0) {
          increment(&counters->range_bounds);
          increment(&counters->errors);
        }
        __syncthreads();
        continue;
      }
      for (uint32_t local = threadIdx.x; local < range.row_count;
           local += blockDim.x) {
        uint32_t row = range.row_begin + local;
        const auto meta = handle->recv_src_metadata[row];
        uint32_t error = rlccl::ep::kCombineOk;
        if (meta.src_rank >= world_size) {
          error = rlccl::ep::kCombineSourceRank;
          increment(&counters->wrong_source_rank);
        } else if (meta.src_token_idx >= layout.num_source_tokens) {
          error = rlccl::ep::kCombineSourceToken;
          increment(&counters->wrong_token);
        } else if (meta.topk_slot >= layout.num_topk) {
          error = rlccl::ep::kCombineTopkSlot;
          increment(&counters->wrong_topk_slot);
        } else if (meta.expert_id != range.expert_id) {
          error = rlccl::ep::kCombineExpertMismatch;
          increment(&counters->wrong_expert);
        }
        uint64_t start = combine_time_ns();
        uint64_t offset = 0;
        uint8_t* target = nullptr;
        if (error == rlccl::ep::kCombineOk) {
          offset = layout.return_offset(meta.src_token_idx, meta.topk_slot);
          if (offset + layout.record_bytes >
              layout.base_offset + layout.region_bytes) {
            error = rlccl::ep::kCombineCapacity;
          } else if (!rlccl::transport::cuda_backend::device_transport_is_direct(
                         transport, static_cast<int32_t>(meta.src_rank))) {
            error = rlccl::ep::kCombinePeerNotDirect;
          } else {
            target = rlccl::transport::cuda_backend::device_transport_get_remote_ptr(
                transport, static_cast<int32_t>(meta.src_rank), offset);
            if (!target) error = rlccl::ep::kCombinePeerNotDirect;
          }
        }
        if (error == rlccl::ep::kCombineOk) {
          auto* slot_meta = reinterpret_cast<rlccl::ep::ReturnSlotMeta*>(target);
          uint32_t old = atomicCAS(&slot_meta->generation, 0U, expected_generation);
          if (old != 0U) {
            error = rlccl::ep::kCombineSlotCollision;
            increment(&counters->slot_collision);
          } else {
            slot_meta->expert_id = meta.expert_id;
            slot_meta->src_token_idx = meta.src_token_idx;
            slot_meta->topk_slot = meta.topk_slot;
            copy_feature_vectorized(
                target + kReturnMetaBytes,
                expert_output + static_cast<uint64_t>(row) * layout.hidden,
                layout.hidden * sizeof(float));
            increment(&counters->rows_mapped);
            if (meta.src_rank == rank) {
              increment(&counters->local_returns);
            } else {
              increment(&counters->remote_returns);
              atomicAdd(reinterpret_cast<unsigned long long*>(&counters->remote_bytes),
                        static_cast<unsigned long long>(layout.record_bytes));
            }
          }
        }
        if (error != rlccl::ep::kCombineOk) increment(&counters->errors);
        uint32_t trace_index = atomicAdd(trace_count, 1U);
        if (trace_index < trace_capacity) {
          traces[trace_index] = rlccl::ep::ReturnTrace{
              row, meta.src_rank, meta.src_token_idx, meta.topk_slot,
              meta.expert_id, meta.src_rank != rank, layout.record_bytes,
              start, combine_time_ns(), 0, error, 0};
        }
      }
      __syncthreads();
    }
  }
  __threadfence_system();
  ncclLsaBarrierSession<ncclCoopCta> completion{
      ncclCoopCta(), dev_comm, ncclTeamTagLsa(), completion_id};
  completion.arrive(ncclCoopCta(), cuda::memory_order_release);
  if (threadIdx.x == 0) increment(&counters->lsa_arrives);
}

__global__ void progressive_combine_wait_kernel(
    ncclDevComm dev_comm, uint32_t completion_id,
    uint64_t* completion_time, rlccl::ep::M8CombineCounters* counters) {
  ncclLsaBarrierSession<ncclCoopCta> completion{
      ncclCoopCta(), dev_comm, ncclTeamTagLsa(), completion_id};
  completion.wait(ncclCoopCta(), cuda::memory_order_acquire);
  if (threadIdx.x == 0) {
    *completion_time = combine_time_ns();
    increment(&counters->lsa_waits);
  }
}

__global__ void combine_reduce_epilogue(
    const uint8_t* registered_buffer, CombineLayout layout,
    const rlccl::ep::ProgressiveEPHandle* handle,
    uint32_t expected_generation, float* output,
    rlccl::ep::M8CombineCounters* counters) {
  uint32_t token = blockIdx.x;
  if (token >= layout.num_source_tokens) return;
  bool handle_valid = handle->generation == expected_generation &&
      handle->num_source_tokens == layout.num_source_tokens &&
      handle->num_topk == layout.num_topk;
  if (!handle_valid) return;
  for (uint32_t hidden = threadIdx.x; hidden < layout.hidden;
       hidden += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < layout.num_topk; ++slot) {
      uint64_t index = static_cast<uint64_t>(token) * layout.num_topk + slot;
      int32_t expected_expert = handle->source_topk_idx[index];
      if (expected_expert < 0) continue;
      const uint8_t* record = registered_buffer +
          layout.return_offset(token, slot);
      const auto* meta = reinterpret_cast<const rlccl::ep::ReturnSlotMeta*>(record);
      bool valid = meta->generation == expected_generation &&
          meta->src_token_idx == token && meta->topk_slot == slot &&
          meta->expert_id == static_cast<uint32_t>(expected_expert);
      if (!valid) continue;
      const float* contribution = reinterpret_cast<const float*>(
          record + kReturnMetaBytes);
      sum += handle->source_topk_weights[index] * contribution[hidden];
    }
    output[static_cast<uint64_t>(token) * layout.hidden + hidden] = sum;
  }
  if (threadIdx.x == 0) {
    for (uint32_t slot = 0; slot < layout.num_topk; ++slot) {
      uint64_t index = static_cast<uint64_t>(token) * layout.num_topk + slot;
      int32_t expected_expert = handle->source_topk_idx[index];
      if (expected_expert < 0) continue;
      const auto* meta = reinterpret_cast<const rlccl::ep::ReturnSlotMeta*>(
          registered_buffer + layout.return_offset(token, slot));
      if (meta->generation != expected_generation) {
        increment(&counters->missing_return);
        increment(&counters->errors);
      } else if (meta->src_token_idx != token) {
        increment(&counters->wrong_token);
        increment(&counters->corruption);
        increment(&counters->errors);
      } else if (meta->topk_slot != slot) {
        increment(&counters->wrong_topk_slot);
        increment(&counters->corruption);
        increment(&counters->errors);
      } else if (meta->expert_id != static_cast<uint32_t>(expected_expert)) {
        increment(&counters->wrong_expert);
        increment(&counters->corruption);
        increment(&counters->errors);
      } else {
        increment(&counters->contributions_reduced);
      }
    }
  }
}

}  // namespace rlccl::ep::cuda_backend
