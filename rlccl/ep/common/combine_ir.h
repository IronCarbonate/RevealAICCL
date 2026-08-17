#pragma once

#include <cstdint>
#include <type_traits>

namespace rlccl::ep {

enum CombineFlags : uint32_t {
  kCombineNone = 0,
  kCombineLsaDirect = 1U << 0,
  kCombineGinStaged = 1U << 1,
};

enum CombineError : uint32_t {
  kCombineOk = 0,
  kCombineStaleHandle = 1,
  kCombineRangeBounds = 2,
  kCombineSourceRank = 3,
  kCombineSourceToken = 4,
  kCombineTopkSlot = 5,
  kCombineExpertMismatch = 6,
  kCombineSourceTopkMismatch = 7,
  kCombineSlotCollision = 8,
  kCombineCapacity = 9,
  kCombinePeerNotDirect = 10,
  kCombineMissingReturn = 11,
};

struct CombineRange {
  uint32_t row_begin;
  uint32_t row_count;
  uint32_t expert_id;
  uint32_t generation;
  uint32_t flags;
};

struct ReturnSlotMeta {
  uint32_t generation;
  uint32_t expert_id;
  uint32_t src_token_idx;
  uint32_t topk_slot;
};

struct ReturnTrace {
  uint32_t row;
  uint32_t src_rank;
  uint32_t src_token_idx;
  uint32_t topk_slot;
  uint32_t expert_id;
  uint32_t is_remote;
  uint64_t bytes;
  uint64_t t_return_start;
  uint64_t t_return_end;
  uint64_t t_remote_completion;
  uint32_t error_code;
  uint32_t reserved;
};

struct M8CombineCounters {
  uint64_t rows_mapped;
  uint64_t local_returns;
  uint64_t remote_returns;
  uint64_t remote_bytes;
  uint64_t lsa_arrives;
  uint64_t lsa_waits;
  uint64_t contributions_reduced;
  uint64_t errors;
  uint64_t stale_handle;
  uint64_t range_bounds;
  uint64_t wrong_source_rank;
  uint64_t wrong_token;
  uint64_t wrong_topk_slot;
  uint64_t wrong_expert;
  uint64_t slot_collision;
  uint64_t missing_return;
  uint64_t corruption;
};

static_assert(sizeof(ReturnSlotMeta) == 16);
static_assert(std::is_trivially_copyable_v<CombineRange>);
static_assert(std::is_trivially_copyable_v<ReturnSlotMeta>);

}  // namespace rlccl::ep
