#pragma once

// Compile-checked R6-M6 GIN backend surface. Runtime creation is deliberately
// gated by ncclCommQueryProperties().ginType and a real RDMA device.

#include <nccl_device.h>

#include "rlccl/transport/cuda/device_transport.cuh"

namespace rlccl::transport::cuda_backend {

inline ncclDevCommRequirements_t gin_requirements(int completion_count) {
  ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginForceEnable = true;
  requirements.ginContextCount = 1;
  requirements.ginSignalCount = completion_count;
  requirements.ginCounterCount = completion_count;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  return requirements;
}

struct NcclGinDeviceTransport {
  ncclDevComm dev_comm;
  ncclWindow_t send_window;
  ncclWindow_t recv_window;
  int context_index;

  __device__ DeviceTransportStatus put(
      const DeviceTransportRequest& request,
      uint32_t thread_id, uint32_t /*thread_count*/) const {
    if (thread_id != 0) return DeviceTransportStatus::kOk;
    ncclGin gin(dev_comm, context_index);
    gin.put(
        ncclTeamWorld(dev_comm), request.peer,
        recv_window, request.dst_offset,
        send_window, request.src_offset, request.bytes,
        ncclGin_SignalInc{static_cast<ncclGinSignal_t>(request.completion_id)},
        ncclGin_CounterInc{static_cast<ncclGinCounter_t>(request.completion_id)});
    return DeviceTransportStatus::kOk;
  }

  // The receiver-side signal establishes remote payload visibility. The local
  // counter independently protects sender source-slot reuse.
  __device__ bool test_completion(uint64_t completion_id) const {
    ncclGin gin(dev_comm, context_index);
    return gin.readSignal(completion_id, 64, cuda::memory_order_acquire) >= 1;
  }

  __device__ DeviceTransportStatus wait_completion(uint64_t completion_id) const {
    ncclGin gin(dev_comm, context_index);
    gin.waitSignal(ncclCoopThread{}, completion_id, 1, 64,
                   cuda::memory_order_acquire);
    return DeviceTransportStatus::kOk;
  }

  __device__ void wait_source_reusable(uint32_t completion_id) const {
    ncclGin gin(dev_comm, context_index);
    gin.waitCounter(ncclCoopThread{}, completion_id, 1, 56,
                    cuda::memory_order_acquire);
  }
};

}  // namespace rlccl::transport::cuda_backend
