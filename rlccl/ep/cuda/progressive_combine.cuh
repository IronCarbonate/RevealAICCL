#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "../common/combine_ir.h"
#include "../common/dispatch_handle.h"
#include "combine_layout.cuh"

namespace rlccl::ep::cuda_backend {

__global__ void build_combine_ranges_kernel(
    const rlccl::ep::ProgressiveEPHandle* handle, uint32_t rank,
    rlccl::ep::CombineRange* ranges);

__global__ void progressive_combine_kernel(
    const rlccl::ep::ProgressiveEPHandle* handle,
    const rlccl::ep::CombineRange* ranges, uint32_t range_count,
    const float* expert_output, CombineLayout layout,
    uint32_t rank, uint32_t world_size, uint32_t expected_generation,
    ncclDevComm dev_comm, ncclWindow_t window,
    uint8_t* registered_buffer, uint32_t completion_id,
    rlccl::ep::ReturnTrace* traces, uint32_t trace_capacity,
    uint32_t* trace_count, rlccl::ep::M8CombineCounters* counters);

__global__ void progressive_combine_wait_kernel(
    ncclDevComm dev_comm, uint32_t completion_id,
    uint64_t* completion_time, rlccl::ep::M8CombineCounters* counters);

}  // namespace rlccl::ep::cuda_backend
