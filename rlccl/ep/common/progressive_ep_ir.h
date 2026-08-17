#pragma once

#include <cstdint>
#include <type_traits>

namespace rlccl::ep {

enum ProgressiveEPError : uint32_t {
  kEpOk = 0,
  kEpPlanInvalid = 1,
  kEpCommitQueueOverflow = 2,
  kEpCommitInvalid = 3,
  kEpUnauthorizedDestination = 4,
  kEpCursorOverflow = 5,
  kEpPeerNotDirect = 6,
  kEpExpertRange = 7,
  kEpTimeout = 8,
};

enum DescriptorCommitFlags : uint32_t {
  kCommitNone = 0,
  kCommitLsaDirect = 1U << 0,
  kCommitGinStaged = 1U << 1,
};

struct DescriptorCommit {
  uint64_t commit_id;
  uint32_t descriptor_id;
  uint32_t chunk_id;
  uint32_t reveal_epoch;
  uint32_t token_begin;
  uint32_t token_count;
  uint32_t assignment_begin;
  uint32_t assignment_count;
  uint64_t authorized_dst_mask;
  uint32_t flags;
  uint32_t reserved;
};

struct CommitPeerPlan {
  uint32_t token_count;
  uint32_t route_id;
  uint64_t src_base_offset;
  uint64_t dst_base_offset;
  uint32_t flags;
  uint32_t reserved;
};

struct DispatchTokenMeta {
  uint32_t src_rank;
  uint32_t src_token_idx;
  uint32_t expert_id;
  uint32_t topk_slot;
  uint32_t descriptor_id;
  uint32_t reveal_epoch;
  float topk_weight;
};

struct DispatchTrace {
  uint64_t commit_id;
  uint32_t descriptor_id;
  uint32_t peer;
  uint32_t token_count;
  uint32_t is_remote;
  uint64_t bytes;
  uint64_t t_commit_consumed;
  uint64_t t_dispatch_start;
  uint64_t t_dispatch_end;
  uint64_t t_remote_completion;
  uint32_t error_code;
  uint32_t reserved;
};

struct DispatchTiming {
  uint64_t t_router_reveal;
  uint64_t t_scheduler_commit;
  uint64_t t_dispatch_start;
  uint64_t t_dispatch_end;
  uint64_t t_remote_completion;
  uint64_t t_final_router_completion;
  uint32_t chunk_id;
  uint32_t descriptor_id;
};

struct M7Counters {
  uint64_t descriptor_commits;
  uint64_t shadow_actions;
  uint64_t assignments_scanned;
  uint64_t direct_remote_records;
  uint64_t direct_remote_bytes;
  uint64_t local_records;
  uint64_t lsa_arrives;
  uint64_t lsa_waits;
  uint64_t epilogue_records;
  uint64_t errors;
  uint64_t unauthorized_destination;
  uint64_t cursor_overflow;
  uint64_t future_access;
  uint64_t unrevealed_access;
  uint64_t stale_action;
};

static_assert(std::is_trivially_copyable_v<DescriptorCommit>);
static_assert(std::is_trivially_copyable_v<CommitPeerPlan>);
static_assert(std::is_trivially_copyable_v<DispatchTokenMeta>);
static_assert(sizeof(DispatchTokenMeta) == 28);

}  // namespace rlccl::ep
