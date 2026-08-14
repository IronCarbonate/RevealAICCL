#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "../common/scheduler_ir.h"

namespace rlccl::scheduler::cuda_backend {

struct DeviceRevealQueue {
  RevealRecord* records;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

struct DeviceActionQueue {
  CommittedAction* actions;
  uint32_t capacity;
  uint32_t head;
  uint32_t tail;
};

struct DeviceIncrementalState {
  uint64_t* revealed_count;
  uint64_t* committed_count;
  uint64_t* next_send_offset;
  uint64_t* next_recv_offset;
  uint32_t* descriptor_epoch;
  uint32_t last_reveal_epoch;
  uint32_t has_reveal_epoch;
  uint64_t next_action_id;
};

struct DeviceErrorLog {
  DeviceSchedulerError* errors;
  uint32_t capacity;
  uint32_t count;
};

struct DeviceSchedulerCounters {
  uint32_t processed_reveals;
  uint32_t committed_actions;
  uint32_t rejected_reveals;
  uint32_t producer_published;
  uint64_t uploaded_plan_checksum;
};

std::vector<torch::Tensor> run_gpu_scheduler_cuda(
    torch::Tensor plan_blob,
    torch::Tensor reveal_records,
    torch::Tensor dst_ranks,
    int64_t source_rank,
    int64_t reveal_queue_capacity,
    int64_t action_queue_capacity,
    int64_t producer_delay_cycles);

}  // namespace rlccl::scheduler::cuda_backend
