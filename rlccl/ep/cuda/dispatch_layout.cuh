#pragma once

#include <cstdint>

namespace rlccl::ep::cuda_backend {

constexpr uint32_t kDispatchMetaStorageBytes = 32;
constexpr uint32_t kDispatchSlotHeaderBytes = 16;

struct DispatchLayout {
  uint32_t world_size;
  uint32_t max_descriptors;
  uint32_t max_assignments_per_peer;
  uint32_t feature_width;
  uint32_t record_bytes;
  uint64_t peer_stride;
  uint64_t descriptor_stride;
  uint64_t region_bytes;
  uint64_t capacity_bytes;

  __host__ __device__ uint64_t recv_offset(
      uint32_t descriptor, uint32_t source) const {
    return static_cast<uint64_t>(descriptor) * descriptor_stride +
           static_cast<uint64_t>(source) * peer_stride;
  }

  __host__ __device__ uint64_t staging_offset(
      uint32_t descriptor, uint32_t destination) const {
    return region_bytes + static_cast<uint64_t>(descriptor) * descriptor_stride +
           static_cast<uint64_t>(destination) * peer_stride;
  }
};

inline DispatchLayout make_dispatch_layout(
    uint32_t world_size, uint32_t max_descriptors,
    uint32_t max_assignments_per_peer, uint32_t feature_width) {
  DispatchLayout layout{};
  layout.world_size = world_size;
  layout.max_descriptors = max_descriptors;
  layout.max_assignments_per_peer = max_assignments_per_peer;
  layout.feature_width = feature_width;
  layout.record_bytes = kDispatchMetaStorageBytes + feature_width * sizeof(float);
  uint64_t raw = kDispatchSlotHeaderBytes +
                 static_cast<uint64_t>(max_assignments_per_peer) * layout.record_bytes;
  layout.peer_stride = (raw + 15) & ~uint64_t(15);
  layout.descriptor_stride = world_size * layout.peer_stride;
  layout.region_bytes = max_descriptors * layout.descriptor_stride;
  layout.capacity_bytes = 2 * layout.region_bytes;
  return layout;
}

}  // namespace rlccl::ep::cuda_backend
