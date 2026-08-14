#pragma once

#include <cstdint>

namespace rlccl::transport::cuda_backend {

enum TransportErrorCode : uint32_t {
  kTransportOk = 0,
  kTransportInvalidRank = 1,
  kTransportInvalidDescriptor = 2,
  kTransportZeroSize = 3,
  kTransportBytesMismatch = 4,
  kTransportOffsetMismatch = 5,
  kTransportOffsetOverflow = 6,
  kTransportSlotReplay = 7,
  kTransportPackCountMismatch = 8,
  kTransportSchedulerError = 9,
  kTransportTimeout = 10,
};

struct PhysicalTransportAction {
  uint64_t action_id;
  uint32_t descriptor_id;
  int32_t src_rank;
  int32_t dst_rank;
  uint64_t logical_src_offset;
  uint64_t logical_dst_offset;
  uint64_t physical_src_offset;
  uint64_t physical_dst_offset;
  uint64_t payload_bytes;
  uint64_t physical_bytes;
  uint32_t token_count;
  uint32_t route_id;
};

struct TransportTrace {
  PhysicalTransportAction action;
  uint64_t t2_action_consumed;
  uint64_t t3_pack_start;
  uint64_t t4_pack_end;
  uint64_t t5_put_start;
  uint64_t t6_put_end;
  uint32_t error_code;
  uint32_t is_remote;
};

struct PipelineTiming {
  uint64_t t0_router_reveal;
  uint64_t t1_scheduler_commit;
  uint64_t t7_remote_completion;
  uint64_t t8_final_router_completion;
  uint32_t chunk_id;
  uint32_t descriptor_id;
};

struct TransportCounters {
  uint64_t scheduler_actions;
  uint64_t transport_actions;
  uint64_t pack_calls;
  uint64_t mscclpp_put_calls;
  uint64_t mscclpp_bytes_transferred;
  uint64_t mscclpp_signals;
  uint64_t mscclpp_waits;
  uint64_t errors;
  uint64_t slot_replays;
  uint64_t future_access;
  uint64_t unrevealed_access;
  uint64_t stale_action;
};

}  // namespace rlccl::transport::cuda_backend
