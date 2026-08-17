#pragma once

// R6-M7 GIN integration surface.  GIN is not a direct-address transport: the
// fused scan writes each peer's records to its local registered staging slot,
// then one device-side put publishes that completed slot.  Signal completion
// protects receiver visibility and counter completion protects staging reuse.

#include "../../transport/cuda/nccl_gin_transport.cuh"
#include "dispatch_layout.cuh"

namespace rlccl::ep::cuda_backend {

__device__ inline rlccl::transport::cuda_backend::DeviceTransportStatus
publish_gin_staging_slot(
    const rlccl::transport::cuda_backend::NcclGinDeviceTransport& transport,
    const DispatchLayout& layout, uint32_t descriptor, uint32_t peer,
    uint32_t source_rank, uint32_t records, uint64_t completion_id) {
  using namespace rlccl::transport::cuda_backend;
  DeviceTransportRequest request{
      static_cast<int32_t>(peer),
      layout.recv_offset(descriptor, source_rank),
      layout.staging_offset(descriptor, peer),
      kDispatchSlotHeaderBytes + static_cast<uint64_t>(records) * layout.record_bytes,
      completion_id};
  return device_transport_put(transport, request, threadIdx.x, blockDim.x);
}

}  // namespace rlccl::ep::cuda_backend
