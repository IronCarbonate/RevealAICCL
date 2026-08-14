#pragma once

#include <cstddef>
#include <cstdint>

#include "rlccl/transport/cuda/gpu_transport_ir.h"

namespace rlccl::transport::cuda_backend {

enum class DeviceTransportKind : uint32_t {
  kMscclpp = 0,
  kNcclLsa = 1,
  kNcclGin = 2,
};

enum class DeviceTransportStatus : uint32_t {
  kOk = 0,
  kInvalidPeer = 1,
  kInvalidOffset = 2,
  kInvalidBytes = 3,
  kCompletionUnavailable = 4,
  kBackendUnavailable = 5,
};

struct DeviceTransportRequest {
  int32_t peer;
  uint64_t dst_offset;
  uint64_t src_offset;
  uint64_t bytes;
  uint64_t completion_id;
};

__host__ __device__ inline DeviceTransportRequest make_transport_request(
    const PhysicalTransportAction& action, uint64_t completion_id) {
  return DeviceTransportRequest{
      action.dst_rank,
      action.physical_dst_offset,
      action.physical_src_offset,
      action.physical_bytes,
      completion_id,
  };
}

template <typename Backend>
__device__ inline DeviceTransportStatus device_transport_put(
    Backend& backend, const DeviceTransportRequest& request,
    uint32_t thread_id, uint32_t thread_count) {
  return backend.put(request, thread_id, thread_count);
}

template <typename Backend>
__device__ inline bool device_transport_test(
    Backend& backend, uint64_t completion_id) {
  return backend.test_completion(completion_id);
}

template <typename Backend>
__device__ inline DeviceTransportStatus device_transport_wait(
    Backend& backend, uint64_t completion_id) {
  return backend.wait_completion(completion_id);
}

}  // namespace rlccl::transport::cuda_backend
