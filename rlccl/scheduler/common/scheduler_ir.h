#pragma once

#include <cstdint>
#include <type_traits>

namespace rlccl::scheduler {

constexpr uint64_t kCompiledPlanMagic = 0x524C43434C4D3401ULL;
constexpr uint32_t kCompiledPlanVersion = 1;
constexpr uint32_t kMaxWorldSize = 64;

enum SchedulerErrorCode : uint32_t {
  kNoError = 0,
  kPlanChecksum = 1,
  kRevealQueueOverflow = 2,
  kActionQueueOverflow = 3,
  kStaleReveal = 4,
  kDuplicateDescriptor = 5,
  kInvalidSourceRank = 6,
  kInvalidDestinationRank = 7,
  kZeroTokenAction = 8,
  kFutureDemand = 9,
  kUnrevealedDemand = 10,
  kOffsetOverflow = 11,
  kBytesOverflow = 12,
  kInvalidRoute = 13,
  kDescriptorRange = 14,
  kChunkRange = 15,
  kAssignmentRange = 16,
  kInternal = 17,
};

enum RevealFlags : uint32_t {
  kRevealNone = 0,
  kInjectFuture = 1U << 0,
  kInjectUnrevealed = 1U << 1,
  kInjectStaleAction = 1U << 2,
  kInjectDuplicateAction = 1U << 3,
  kInjectOffsetOverflow = 1U << 4,
  kInjectInvalidRank = 1U << 5,
  kInjectInvalidRoute = 1U << 6,
  kInjectZeroTokenAction = 1U << 7,
  kInjectBytesOverflow = 1U << 8,
};

enum ActionFlags : uint32_t {
  kActionNone = 0,
  kLogicalOffsets = 1U << 0,
  kSelfAction = 1U << 1,
};

struct RevealRecord {
  uint32_t chunk_id;
  uint32_t reveal_epoch;
  uint32_t token_begin;
  uint32_t token_count;
  uint32_t assignment_begin;
  uint32_t assignment_count;
  uint32_t descriptor_id;
  uint32_t flags;
};

struct CompiledRouteTemplate {
  int32_t src_rank;
  int32_t dst_rank;
  int32_t route_id;
  int32_t channel_id;
  uint64_t send_region_base;
  uint64_t recv_region_base;
  uint32_t flags;
  uint32_t reserved;
};

struct CommittedAction {
  uint64_t action_id;
  uint32_t descriptor_id;
  uint32_t chunk_id;
  uint32_t reveal_epoch;
  int32_t src_rank;
  int32_t dst_rank;
  uint64_t src_offset;
  uint64_t dst_offset;
  uint32_t token_count;
  uint32_t reserved;
  uint64_t bytes;
  uint32_t route_id;
  uint32_t flags;
};

struct DeviceSchedulerError {
  uint32_t error_code;
  uint32_t reserved;
  uint64_t action_id;
  uint32_t descriptor_id;
  uint32_t reveal_epoch;
};

struct SchedulerTiming {
  uint64_t t0_router_complete;
  uint64_t t1_reveal_published;
  uint64_t t2_reveal_consumed;
  uint64_t t3_binder_complete;
  uint64_t t4_guard_complete;
  uint64_t t5_action_published;
  uint32_t chunk_id;
  uint32_t descriptor_id;
};

#pragma pack(push, 1)
struct CompiledPlanBlobHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t header_bytes;
  uint64_t total_bytes;
  uint64_t checksum;
  uint32_t world_size;
  uint32_t route_count;
  uint32_t max_descriptors;
  uint32_t max_chunks;
  uint32_t record_bytes;
  uint32_t max_tokens_per_peer;
  uint64_t descriptor_stride;
  uint64_t region_bytes;
  uint64_t route_templates_offset;
  uint64_t rank_pair_to_route_offset;
  uint64_t capacity_table_offset;
  uint64_t legality_flags_offset;
};
#pragma pack(pop)

static_assert(sizeof(CompiledPlanBlobHeader) == 104);
static_assert(sizeof(CompiledRouteTemplate) == 40);
static_assert(std::is_trivially_copyable_v<RevealRecord>);
static_assert(std::is_trivially_copyable_v<CommittedAction>);

}  // namespace rlccl::scheduler
