from __future__ import annotations

import numpy as np
import pytest

from rlccl.transport.reference_a2av import (
    PAYLOAD_FIELDS,
    ProgressivePackingState,
    RouterAssignment,
    build_destination_layout,
    pack_destination_layout,
    verify_received_records,
)


def _assignment(token: int, destination: int, chunk: int = 0) -> RouterAssignment:
    return RouterAssignment(
        token_id=token,
        source_rank=0,
        destination_rank=destination,
        expert_id=destination,
        chunk_id=chunk,
        chunk_offset=token,
        payload_word=token * 17 + 3,
    )


def test_router_assignments_derive_variable_counts_offsets_and_contiguous_payload() -> None:
    values = (_assignment(4, 1), _assignment(2, 0), _assignment(1, 0))
    layout = build_destination_layout(values, world_size=2, source_rank=0, chunk_ids=(0,))
    assert layout.sendcounts_tokens == (2, 1)
    assert layout.offsets_tokens == (0, 2)
    packed = pack_destination_layout(layout)
    assert packed.records.flags.c_contiguous
    assert packed.records.shape == (3, PAYLOAD_FIELDS)
    assert packed.sendcounts_elements == (2 * PAYLOAD_FIELDS, PAYLOAD_FIELDS)
    assert packed.records[:, 0].tolist() == [1, 2, 4]


def test_zero_sized_pair_and_empty_shard_are_explicit() -> None:
    layout = build_destination_layout(
        (_assignment(1, 1), _assignment(2, 1)),
        world_size=2,
        source_rank=0,
        chunk_ids=(0,),
    )
    assert layout.sendcounts_tokens == (0, 2)
    empty = pack_destination_layout(build_destination_layout(
        (), world_size=2, source_rank=0, chunk_ids=(1,),
    ))
    assert empty.sendcounts_tokens == (0, 0)
    assert empty.records.shape == (0, PAYLOAD_FIELDS)


def test_progressive_state_rejects_future_unrevealed_duplicate_and_stale_dispatch() -> None:
    state = ProgressivePackingState(world_size=2, source_rank=0, max_chunks=3)
    with pytest.raises(ValueError, match="future"):
        state.build_delta((0,))
    assert state.future_access_attempts == 1
    state.mark_completed(0, (_assignment(1, 0, 0),))
    with pytest.raises(ValueError, match="unrevealed"):
        state.build_delta((0,))
    assert state.unrevealed_execution == 1
    state.reveal(0)
    _, packed = state.build_delta((0,))
    assert packed.total_tokens == 1
    with pytest.raises(ValueError, match="stale"):
        state.build_delta((0,))
    assert state.stale_dispatch == 1

    state.mark_completed(1, (_assignment(1, 1, 1),))
    state.reveal(1)
    with pytest.raises(ValueError, match="duplicate"):
        state.build_delta((1,))
    assert state.duplicate_dispatch == 1


def test_receive_verifier_detects_corruption_wrong_destination_and_loss() -> None:
    rows = np.asarray([_assignment(7, 1).record()], dtype=np.int64)
    expected = {7: tuple(int(value) for value in rows[0])}
    assert verify_received_records(rows, destination_rank=1, expected_by_token=expected)["pass"]

    corrupt = rows.copy()
    corrupt[0, 6] += 1
    result = verify_received_records(corrupt, destination_rank=0, expected_by_token=expected)
    assert not result["pass"]
    assert result["wrong_destination"] == 1
    assert result["corruption"] >= 1

    missing = np.empty((0, PAYLOAD_FIELDS), dtype=np.int64)
    result = verify_received_records(missing, destination_rank=1, expected_by_token=expected)
    assert result["lost"] == 1
