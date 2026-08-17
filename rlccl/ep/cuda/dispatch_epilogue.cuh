#pragma once

#include <cuda_runtime.h>

#include "../common/dispatch_handle.h"
#include "dispatch_layout.cuh"

namespace rlccl::ep::cuda_backend {

__global__ void dispatch_count_experts_kernel(
    const uint8_t* registered_buffer, DispatchLayout layout,
    uint32_t rank, uint32_t experts_per_rank, uint32_t* expert_counts,
    rlccl::ep::M7Counters* counters);

__global__ void dispatch_exclusive_scan_kernel(
    uint32_t* expert_counts, uint32_t* expert_offsets,
    uint32_t* expert_cursors, uint32_t num_local_experts,
    rlccl::ep::ProgressiveEPHandle* handle, uint32_t generation);

__global__ void dispatch_scatter_experts_kernel(
    const uint8_t* registered_buffer, DispatchLayout layout,
    uint32_t rank, uint32_t experts_per_rank, uint32_t* expert_cursors,
    float* recv_x, rlccl::ep::DispatchTokenMeta* recv_metadata,
    rlccl::ep::M7Counters* counters);

}  // namespace rlccl::ep::cuda_backend
