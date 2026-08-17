#pragma once

#include <cstdint>

namespace rlccl::ep::cuda_backend {

constexpr uint32_t kReturnMetaBytes = 16;

struct CombineLayout {
  uint32_t num_source_tokens;
  uint32_t num_topk;
  uint32_t hidden;
  uint32_t record_bytes;
  uint64_t base_offset;
  uint64_t region_bytes;
  uint64_t capacity_bytes;

  __host__ __device__ uint64_t slot_id(
      uint32_t token, uint32_t topk_slot) const {
    return static_cast<uint64_t>(token) * num_topk + topk_slot;
  }

  __host__ __device__ uint64_t return_offset(
      uint32_t token, uint32_t topk_slot) const {
    return base_offset + slot_id(token, topk_slot) * record_bytes;
  }

  __host__ __device__ uint64_t staging_offset(
      uint32_t token, uint32_t topk_slot) const {
    return base_offset + region_bytes + slot_id(token, topk_slot) * record_bytes;
  }
};

inline CombineLayout make_combine_layout(
    uint32_t num_source_tokens, uint32_t num_topk, uint32_t hidden,
    uint64_t base_offset) {
  CombineLayout layout{};
  layout.num_source_tokens = num_source_tokens;
  layout.num_topk = num_topk;
  layout.hidden = hidden;
  layout.record_bytes = kReturnMetaBytes + hidden * sizeof(float);
  layout.base_offset = base_offset;
  layout.region_bytes = static_cast<uint64_t>(num_source_tokens) * num_topk *
                        layout.record_bytes;
  layout.capacity_bytes = 2 * layout.region_bytes;
  return layout;
}

}  // namespace rlccl::ep::cuda_backend
