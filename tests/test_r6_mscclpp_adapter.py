"""R6-M1 legality and offset contracts independent of CUDA availability."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlccl.transport.mscclpp_backend import (
    COUNT_HEADER_BYTES,
    RECORD_BYTES,
    MscclppCommittedAdapter,
    RegisteredBufferLayout,
)
from rlccl.transport.reference_a2av import PackedShard


def _packed(*, chunks=(0,), counts=(2, 1)) -> PackedShard:
    total = sum(counts)
    return PackedShard(
        source_rank=0,
        chunk_ids=tuple(chunks),
        sendcounts_tokens=tuple(counts),
        offsets_tokens=(0, counts[0]),
        records=np.zeros((total, 8), dtype=np.int64),
    )


def _pass(applied=1):
    return SimpleNamespace(accepted=True, applied_actions=applied, state_version=1)


def test_guarded_descriptor_maps_to_fixed_registered_offsets() -> None:
    layout = RegisteredBufferLayout(
        world_size=2, max_descriptors=8, max_tokens_per_peer_descriptor=16,
    )
    adapter = MscclppCommittedAdapter(rank=0, layout=layout)
    actions = adapter.commit_descriptor(
        _packed(), descriptor_id=3, guard_decision=_pass(),
        completed_chunks={0}, revealed_chunks={0},
    )
    assert [action.dst_rank for action in actions] == [0, 1]
    assert actions[0].src_offset == layout.send_offset(3, 0)
    assert actions[1].src_offset == layout.send_offset(3, 1)
    assert actions[1].dst_offset == layout.receive_offset(3, 0)
    assert actions[0].bytes == 2 * RECORD_BYTES
    assert actions[0].physical_bytes == COUNT_HEADER_BYTES + 2 * RECORD_BYTES
    assert adapter.counters()["committed_payload_bytes"] == 3 * RECORD_BYTES


def test_zero_destination_does_not_create_zero_size_action() -> None:
    layout = RegisteredBufferLayout(
        world_size=2, max_descriptors=2, max_tokens_per_peer_descriptor=4,
    )
    actions = MscclppCommittedAdapter(rank=0, layout=layout).commit_descriptor(
        _packed(counts=(0, 1)), descriptor_id=0, guard_decision=_pass(),
        completed_chunks={0}, revealed_chunks={0},
    )
    assert len(actions) == 1 and actions[0].dst_rank == 1


def test_zero_remote_payload_transfers_only_nonzero_count_header() -> None:
    layout = RegisteredBufferLayout(
        world_size=2, max_descriptors=2, max_tokens_per_peer_descriptor=4,
    )
    actions = MscclppCommittedAdapter(rank=0, layout=layout).commit_descriptor(
        _packed(counts=(1, 0)), descriptor_id=0, guard_decision=_pass(),
        completed_chunks={0}, revealed_chunks={0},
    )
    remote = next(action for action in actions if action.dst_rank == 1)
    assert remote.token_count == 0 and remote.bytes == 0
    assert remote.physical_bytes == COUNT_HEADER_BYTES


def test_guard_failure_future_unrevealed_and_replay_fail_closed() -> None:
    layout = RegisteredBufferLayout(
        world_size=2, max_descriptors=2, max_tokens_per_peer_descriptor=4,
    )
    adapter = MscclppCommittedAdapter(rank=0, layout=layout)
    with pytest.raises(ValueError, match="DynamicGuard"):
        adapter.commit_descriptor(
            _packed(), descriptor_id=0,
            guard_decision=SimpleNamespace(accepted=False, applied_actions=0),
            completed_chunks={0}, revealed_chunks={0},
        )
    with pytest.raises(ValueError, match="future"):
        adapter.commit_descriptor(
            _packed(chunks=(1,)), descriptor_id=0, guard_decision=_pass(),
            completed_chunks={0}, revealed_chunks={0},
        )
    with pytest.raises(ValueError, match="unrevealed"):
        adapter.commit_descriptor(
            _packed(chunks=(1,)), descriptor_id=0, guard_decision=_pass(),
            completed_chunks={0, 1}, revealed_chunks={0},
        )
    adapter.commit_descriptor(
        _packed(), descriptor_id=0, guard_decision=_pass(),
        completed_chunks={0}, revealed_chunks={0},
    )
    with pytest.raises(ValueError, match="stale"):
        adapter.commit_descriptor(
            _packed(), descriptor_id=0, guard_decision=_pass(),
            completed_chunks={0}, revealed_chunks={0},
        )
    assert adapter.counters() == {
        "committed_actions": 2,
        "committed_payload_bytes": 3 * RECORD_BYTES,
        "planned_physical_bytes": 2 * COUNT_HEADER_BYTES + 3 * RECORD_BYTES,
        "future_access": 1,
        "unrevealed_access": 1,
        "stale_action": 1,
    }


def test_capacity_and_world_size_are_frozen_fail_closed() -> None:
    with pytest.raises(ValueError, match="world_size=2"):
        RegisteredBufferLayout(
            world_size=4, max_descriptors=1, max_tokens_per_peer_descriptor=1,
        )
    layout = RegisteredBufferLayout(
        world_size=2, max_descriptors=1, max_tokens_per_peer_descriptor=1,
    )
    with pytest.raises(ValueError, match="capacity"):
        MscclppCommittedAdapter(rank=0, layout=layout).commit_descriptor(
            _packed(counts=(2, 1)), descriptor_id=0, guard_decision=_pass(),
            completed_chunks={0}, revealed_chunks={0},
        )
