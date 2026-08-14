#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "gpu_transport_ir.h"
#include "../../scheduler/common/scheduler_ir.h"

namespace rlccl::transport::cuda_backend {

struct DevicePackingInput {
  const int32_t* destination_ranks;
  const int64_t* metadata;
  const float* features;
  uint8_t* registered_buffer;
  uint32_t total_assignments;
  uint32_t metadata_fields;
  uint32_t feature_width;
  uint32_t record_bytes;
};

__device__ inline TransportErrorCode pack_committed_action(
    const rlccl::scheduler::CommittedAction& action,
    const rlccl::scheduler::RevealRecord& reveal,
    const DevicePackingInput& input) {
  if (static_cast<uint64_t>(reveal.assignment_begin) + reveal.assignment_count >
      input.total_assignments) return kTransportPackCountMismatch;
  if (threadIdx.x == 0) {
    *reinterpret_cast<uint64_t*>(input.registered_buffer + action.src_offset) = action.token_count;
  }
  __syncthreads();
  uint32_t packed_index = 0;
  for (uint32_t local = 0; local < reveal.assignment_count; ++local) {
    uint32_t assignment = reveal.assignment_begin + local;
    if (input.destination_ranks[assignment] != action.dst_rank) continue;
    uint8_t* target = input.registered_buffer + action.src_offset + 8 +
                      static_cast<uint64_t>(packed_index) * input.record_bytes;
    const uint8_t* meta = reinterpret_cast<const uint8_t*>(
        input.metadata + static_cast<uint64_t>(assignment) * input.metadata_fields);
    const uint8_t* feature = reinterpret_cast<const uint8_t*>(
        input.features + static_cast<uint64_t>(assignment) * input.feature_width);
    const uint32_t metadata_bytes = input.metadata_fields * sizeof(int64_t);
    const uint32_t feature_bytes = input.feature_width * sizeof(float);
    for (uint32_t byte = threadIdx.x; byte < metadata_bytes; byte += blockDim.x)
      target[byte] = meta[byte];
    for (uint32_t byte = threadIdx.x; byte < feature_bytes; byte += blockDim.x)
      target[metadata_bytes + byte] = feature[byte];
    __syncthreads();
    ++packed_index;
  }
  if (packed_index != action.token_count) return kTransportPackCountMismatch;
  __threadfence();
  return kTransportOk;
}

}  // namespace rlccl::transport::cuda_backend
