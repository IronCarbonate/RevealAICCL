#pragma once

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>

#include "../../scheduler/common/scheduler_ir.h"
#include "../common/dispatch_handle.h"
#include "dispatch_layout.cuh"

namespace rlccl::ep::cuda_backend {

struct DeviceCommitQueue {
  rlccl::ep::DescriptorCommit* commits;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

struct DeviceCommitState {
  uint32_t scheduler_done;
  uint32_t dispatch_done;
};

struct ProgressiveDispatchInput {
  const float* x;
  const int64_t* topk_idx;
  const float* topk_weights;
  uint32_t num_tokens;
  uint32_t feature_width;
  uint32_t num_topk;
  uint32_t experts_per_rank;
  uint32_t num_experts;
};

__device__ void progressive_dispatch_progress_role(
    DeviceCommitQueue* commit_queue,
    const rlccl::ep::CommitPeerPlan* peer_plans,
    const rlccl::scheduler::RevealRecord* records_by_descriptor,
    DeviceCommitState* state,
    ProgressiveDispatchInput input,
    DispatchLayout layout,
    uint32_t rank,
    ncclDevComm dev_comm,
    ncclWindow_t window,
    uint8_t* registered_buffer,
    uint32_t* dst_cursors,
    rlccl::ep::DispatchTrace* traces,
    uint32_t trace_capacity,
    uint32_t* trace_count,
    rlccl::ep::DispatchTiming* timings,
    rlccl::ep::M7Counters* counters);

__device__ void progressive_dispatch_wait_role(
    ncclDevComm dev_comm, uint32_t descriptor_count,
    rlccl::ep::DispatchTiming* timings, rlccl::ep::M7Counters* counters);

__global__ void progressive_dispatch_progress_kernel(
    DeviceCommitQueue* commit_queue,
    const rlccl::ep::CommitPeerPlan* peer_plans,
    const rlccl::scheduler::RevealRecord* records_by_descriptor,
    DeviceCommitState* state,
    ProgressiveDispatchInput input,
    DispatchLayout layout,
    uint32_t rank,
    ncclDevComm dev_comm,
    ncclWindow_t window,
    uint8_t* registered_buffer,
    uint32_t* dst_cursors,
    rlccl::ep::DispatchTrace* traces,
    uint32_t trace_capacity,
    uint32_t* trace_count,
    rlccl::ep::DispatchTiming* timings,
    rlccl::ep::M7Counters* counters);

__global__ void progressive_dispatch_wait_kernel(
    ncclDevComm dev_comm, uint32_t descriptor_count,
    rlccl::ep::DispatchTiming* timings, rlccl::ep::M7Counters* counters);

}  // namespace rlccl::ep::cuda_backend
