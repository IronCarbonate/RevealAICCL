#pragma once

#include <cstdint>

#include "progressive_ep_ir.h"

namespace rlccl::ep {

struct ProgressiveEPHandle {
  uint32_t num_recv_tokens;
  uint32_t num_local_experts;
  uint32_t num_topk;
  uint32_t* expert_counts;
  uint32_t* expert_offsets;
  DispatchTokenMeta* recv_src_metadata;
  uint32_t generation;

  // M8 appends source-side combine state after the frozen M7 ABI prefix.
  // Existing M7 aggregate initializers therefore keep their original meaning.
  uint32_t num_source_tokens;
  int32_t* source_topk_idx;
  float* source_topk_weights;
};

}  // namespace rlccl::ep
