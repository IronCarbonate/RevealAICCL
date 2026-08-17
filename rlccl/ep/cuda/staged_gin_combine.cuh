#pragma once

#include "../../transport/cuda/nccl_gin_transport.cuh"
#include "combine_layout.cuh"

namespace rlccl::ep::cuda_backend {

__device__ inline rlccl::transport::cuda_backend::DeviceTransportStatus
publish_gin_return_slot(
    const rlccl::transport::cuda_backend::NcclGinDeviceTransport& transport,
    const CombineLayout& layout, uint32_t source_rank, uint32_t source_token,
    uint32_t topk_slot, uint64_t completion_id) {
  using namespace rlccl::transport::cuda_backend;
  DeviceTransportRequest request{
      static_cast<int32_t>(source_rank),
      layout.return_offset(source_token, topk_slot),
      layout.staging_offset(source_token, topk_slot),
      layout.record_bytes, completion_id};
  return device_transport_put(transport, request, threadIdx.x, blockDim.x);
}

}  // namespace rlccl::ep::cuda_backend
