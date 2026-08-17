"""R6-M7 commit/data-plane contracts that do not require a local GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rlccl.ep.layout import ProgressiveDispatchLayout
from rlccl.ep.reference import build_commit_reference
from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan
from rlccl.scheduler.common.scheduler_schema import RevealRecord, SchedulerConfig


def _layout() -> ProgressiveDispatchLayout:
    return ProgressiveDispatchLayout(2, 2, 4, 16)


def _plan():
    layout = _layout()
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=2, source_rank=0, record_bytes=136,
        max_descriptors=2, max_chunks=2, max_tokens_per_peer=4,
        reveal_queue_capacity=8, action_queue_capacity=8, block_size=32,
    ))


def test_dispatch_layout_is_aligned_and_has_disjoint_recv_staging_regions() -> None:
    layout = _layout()
    assert layout.record_bytes == 96
    assert layout.peer_stride % 16 == 0
    assert layout.recv_offset(1, 1) < layout.region_bytes
    assert layout.staging_offset(0, 0) == layout.region_bytes
    assert layout.capacity_bytes == 2 * layout.region_bytes


def test_one_commit_per_reveal_expands_to_frozen_action_shadow() -> None:
    records = (
        RevealRecord(0, 1, 0, 4, 0, 4, 0),
        RevealRecord(1, 2, 4, 4, 4, 4, 1),
    )
    topk = np.asarray([[0], [1], [2], [3], [1], [1], [2], [2]], dtype=np.int64)
    reference = build_commit_reference(_plan(), _layout(), records, topk, 2)
    assert len(reference.commits) == len(records)
    assert [commit.authorized_dst_mask for commit in reference.commits] == [3, 3]
    assert [action.token_count for action in reference.shadow_actions] == [2, 2, 2, 2]
    active = [peer for peer in reference.peer_plans if peer.token_count]
    assert [(peer.descriptor_id, peer.destination, peer.token_count)
            for peer in active] == [(0, 0, 2), (0, 1, 2), (1, 0, 2), (1, 1, 2)]


def test_native_source_contains_persistent_commit_and_direct_lsa_path() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime = (root / "extensions" / "r6_m7_deepep_style" /
               "deepep_dispatch_runtime.cu").read_text(encoding="utf-8")
    dispatch = (root / "rlccl" / "ep" / "cuda" /
                "progressive_dispatch.cu").read_text(encoding="utf-8")
    for required in (
        "DescriptorCommit", "CommitPeerPlan", "DeviceCommitQueue",
        "cuda::memory_order_release", "cuda::memory_order_acquire",
        "descriptor_commit_scheduler_role", "m7_pipeline_kernel<<<4",
    ):
        assert required in runtime
    for required in (
        "progressive_dispatch_progress_kernel", "ncclGetPeerPointer",
        "topk_idx", "topk_weights", "copy_feature_vectorized",
        "completion.arrive", "completion.wait",
        "dispatch_count_experts_kernel", "dispatch_scatter_experts_kernel",
    ):
        assert required in dispatch
    for forbidden in ("ncclSend", "ncclRecv", "ncclAllToAll", "mscclpp::"):
        assert forbidden not in runtime + dispatch


def test_copy_traits_are_volta_ampere_hopper_capability_specific() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "cuda" / "arch_copy.cuh").read_text()
    for arch in ("DispatchCopyTraits<700>", "DispatchCopyTraits<800>",
                 "DispatchCopyTraits<900>"):
        assert arch in source
    assert "kSupportsCpAsync = false" in source
    assert "kSupportsTma = true" in source


def test_gin_is_staged_and_keeps_signal_counter_completion_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "cuda" /
              "staged_gin_dispatch.cuh").read_text()
    assert "layout.staging_offset" in source
    assert "layout.recv_offset" in source
    assert "publish_gin_staging_slot" in source
    gin = (root / "rlccl" / "transport" / "cuda" /
           "nccl_gin_transport.cuh").read_text()
    assert "return false" in gin
    assert "waitSignal" in gin and "waitCounter" in gin
