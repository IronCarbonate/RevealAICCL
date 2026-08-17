#pragma once

#include <cuda_runtime.h>

#include "../common/combine_ir.h"
#include "../common/dispatch_handle.h"
#include "combine_layout.cuh"

namespace rlccl::ep::cuda_backend {

__global__ void combine_reduce_epilogue(
    const uint8_t* registered_buffer, CombineLayout layout,
    const rlccl::ep::ProgressiveEPHandle* handle,
    uint32_t expected_generation, float* output,
    rlccl::ep::M8CombineCounters* counters);

}  // namespace rlccl::ep::cuda_backend
