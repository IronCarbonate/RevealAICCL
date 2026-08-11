from __future__ import annotations

import numpy as np

from rlccl.transport.reference_a2av import RouterAssignment, build_destination_layout
from rlccl.transport.reference_full_moe import (
    pack_forward_payload, pack_return_payload, verify_forward_payload,
    verify_return_and_combine,
)


def test_forward_feature_identity_and_return_combine_roundtrip() -> None:
    assignments = (
        RouterAssignment(10, 0, 0, 0, 0, 0, 1),
        RouterAssignment(11, 0, 1, 1, 0, 1, 2),
    )
    layout = build_destination_layout(assignments, world_size=2, source_rank=0, chunk_ids=(0,))
    features = {10: np.asarray([1.0, 2.0], np.float32), 11: np.asarray([3.0, 4.0], np.float32)}
    payload = pack_forward_payload(
        layout, features_by_token=features, original_position_by_token={10: 0, 11: 1},
    )
    assert payload.sendcounts_tokens == (1, 1)
    first_meta, first_features = payload.metadata[:1], payload.features[:1]
    assert verify_forward_payload(first_meta, first_features, destination_rank=0, world_size=2)["pass"]

    expert_outputs = np.asarray([[5.0, 6.0]], dtype=np.float32)
    returned = pack_return_payload(first_meta, expert_outputs, expert_rank=0, world_size=2)
    combined, result = verify_return_and_combine(
        returned.metadata, returned.outputs, origin_rank=0,
        recvcounts_tokens=(1, 0), expected_expert_by_token={10: 0},
        expected_position_by_token={10: 0}, expected_output_by_token={10: expert_outputs[0]},
        total_tokens=1,
    )
    assert result["pass"]
    assert np.array_equal(combined, expert_outputs)


def test_forward_and_return_corruption_fail_closed() -> None:
    assignment = RouterAssignment(20, 0, 0, 0, 0, 0, 1)
    layout = build_destination_layout((assignment,), world_size=2, source_rank=0, chunk_ids=(0,))
    payload = pack_forward_payload(
        layout, features_by_token={20: np.asarray([1.0, 2.0], np.float32)},
        original_position_by_token={20: 0},
    )
    bad_features = payload.features.copy(); bad_features[0, 0] += 1
    assert not verify_forward_payload(
        payload.metadata, bad_features, destination_rank=0, world_size=2,
    )["pass"]

    returned = pack_return_payload(
        payload.metadata, np.asarray([[7.0]], np.float32), expert_rank=0, world_size=2,
    )
    bad_meta = returned.metadata.copy(); bad_meta[0, 7] += 1
    _, result = verify_return_and_combine(
        bad_meta, returned.outputs, origin_rank=0, recvcounts_tokens=(1, 0),
        expected_expert_by_token={20: 0}, expected_position_by_token={20: 0},
        expected_output_by_token={20: np.asarray([7.0], np.float32)}, total_tokens=1,
    )
    assert not result["pass"]
    assert result["corruption"] == 1
