"""R6-M8 handle-driven combine contracts that do not require a local GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlccl.ep.combine import ReturnLayout, reference_moe_output


@pytest.mark.parametrize("topk", [1, 2, 3])
def test_return_slot_is_unique_for_every_token_topk_pair(topk: int) -> None:
    layout = ReturnLayout(7, topk, 16, 4096)
    offsets = [layout.return_offset(token, slot)
               for token in range(7) for slot in range(topk)]
    assert len(offsets) == len(set(offsets)) == 7 * topk
    assert min(offsets) == layout.base_offset
    assert max(offsets) + layout.record_bytes == layout.base_offset + layout.region_bytes
    assert layout.staging_offset(0, 0) == layout.base_offset + layout.region_bytes


@pytest.mark.parametrize("shape", ["balanced", "skewed", "all_to_one_like"])
@pytest.mark.parametrize("topk", [1, 2, 3])
def test_reference_combine_uses_fixed_k_order(shape: str, topk: int) -> None:
    tokens, hidden, experts = 8, 4, 4
    x = np.arange(tokens * hidden, dtype=np.float32).reshape(tokens, hidden) / 17
    if shape == "balanced":
        idx = np.asarray([[(token + slot) % experts for slot in range(topk)]
                          for token in range(tokens)], dtype=np.int64)
    elif shape == "skewed":
        idx = np.asarray([[0, 1, 2][:topk] for _ in range(tokens)], dtype=np.int64)
    else:
        idx = np.asarray([[0, 1, 3][:topk] for _ in range(tokens)], dtype=np.int64)
    weights = np.asarray([[3.0 - slot * 0.25 for slot in range(topk)]
                          for _ in range(tokens)], dtype=np.float32)
    matrices = np.zeros((experts, hidden, hidden), dtype=np.float32)
    for expert in range(experts):
        matrices[expert] = np.eye(hidden, dtype=np.float32) * np.float32(1 + expert / 8)
    first = reference_moe_output(x, idx, weights, matrices)
    second = reference_moe_output(x, idx, weights, matrices)
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()


def test_handle_extension_preserves_m7_prefix_and_adds_source_state() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "common" / "dispatch_handle.h").read_text()
    prefix = (
        "uint32_t num_recv_tokens", "uint32_t num_local_experts",
        "uint32_t num_topk", "uint32_t* expert_counts",
        "uint32_t* expert_offsets", "DispatchTokenMeta* recv_src_metadata",
        "uint32_t generation",
    )
    positions = [source.index(item) for item in prefix]
    assert positions == sorted(positions)
    assert source.index("uint32_t num_source_tokens") > positions[-1]
    assert "int32_t* source_topk_idx" in source
    assert "float* source_topk_weights" in source


def test_combine_kernel_is_handle_driven_and_has_no_scheduler_or_router() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "cuda" /
              "progressive_combine.cu").read_text()
    for required in (
        "handle->recv_src_metadata", "meta.src_rank", "meta.src_token_idx",
        "meta.topk_slot", "meta.expert_id", "device_transport_get_remote_ptr",
        "atomicCAS", "copy_feature_vectorized", "completion.arrive",
        "completion.wait", "combine_reduce_epilogue",
        "handle->source_topk_idx", "handle->source_topk_weights",
    ):
        assert required in source
    for forbidden in (
        "CommittedAction", "DescriptorCommit", "DeviceCommitQueue",
        "FastBinder", "router_topk", "ncclSend", "ncclRecv", "ncclAllToAll",
        "mscclpp::",
    ):
        assert forbidden not in source


def test_gin_return_is_staged_and_not_claimed_direct() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "cuda" /
              "staged_gin_combine.cuh").read_text()
    assert "layout.staging_offset" in source
    assert "layout.return_offset" in source
    assert "publish_gin_return_slot" in source
    gin = (root / "rlccl" / "transport" / "cuda" /
           "nccl_gin_transport.cuh").read_text()
    assert "__device__ bool is_direct" in gin and "return false" in gin
    assert "waitSignal" in gin and "waitCounter" in gin


def test_m8_runtime_is_separate_from_frozen_m7_extension() -> None:
    root = Path(__file__).resolve().parents[1]
    m8 = root / "extensions" / "r6_m8_handle_combine" / "handle_combine_runtime.cu"
    assert m8.is_file()
    source = m8.read_text()
    assert "progressive_combine_kernel<<<1, 256" in source
    assert "combine_reduce_epilogue<<<num_tokens" in source
    assert "r6_m8_run" in source
