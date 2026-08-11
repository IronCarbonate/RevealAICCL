from dataclasses import replace

import numpy as np
import pytest

from rlccl.traffic.matrix_utils import validate_traffic_matrix
from rlccl.traffic.moment_validation import validate_sequence_moment_bounds
from rlccl.traffic.process_generator import (
    TrafficProcessConfig,
    generate_traffic_sequence,
)
from rlccl.traffic.types import TrafficSequence


def _config(family, seed=42):
    return TrafficProcessConfig(
        num_nodes=4,
        sequence_length=32,
        window_size=16,
        mean_level=2.0,
        std_level=1.0,
        max_entry=8,
        epsilon_mean=0.20,
        epsilon_var=0.30,
        family=family,
        seed=seed,
        max_generation_attempts=2,
        topology_name="Rear4GPU",
    )


@pytest.mark.parametrize("family", TrafficProcessConfig.FAMILIES)
def test_every_family_is_valid_reproducible_and_bounded(family):
    first = generate_traffic_sequence(_config(family))
    second = generate_traffic_sequence(_config(family))
    assert first.to_dict() == second.to_dict()
    for matrix in first.matrices:
        validate_traffic_matrix(matrix)
        assert np.issubdtype(matrix.dtype, np.integer)
        assert matrix.max() <= 8
    diagnostics = validate_sequence_moment_bounds(first)
    assert diagnostics["passed"], diagnostics


@pytest.mark.parametrize("family", TrafficProcessConfig.FAMILIES)
def test_different_seed_usually_changes_sequence(family):
    first = generate_traffic_sequence(_config(family, seed=1))
    second = generate_traffic_sequence(_config(family, seed=2))
    assert any(not np.array_equal(a, b) for a, b in zip(first.matrices, second.matrices))


def test_same_reference_moments_support_distinct_families():
    smooth = generate_traffic_sequence(_config("smooth_ar"))
    burst = generate_traffic_sequence(
        replace(
            _config("alternating_burst"),
            mean_ref=smooth.mean_ref,
            var_ref=smooth.var_ref,
        )
    )
    np.testing.assert_array_equal(smooth.mean_ref, burst.mean_ref)
    np.testing.assert_array_equal(smooth.var_ref, burst.var_ref)
    assert not np.array_equal(np.stack(smooth.matrices), np.stack(burst.matrices))


def test_sparse_switching_is_sparse_when_requested_moments_are_compatible():
    config = TrafficProcessConfig(
        num_nodes=4,
        sequence_length=32,
        window_size=16,
        mean_level=0.75,
        std_level=1.3,
        max_entry=8,
        epsilon_mean=0.30,
        epsilon_var=0.40,
        family="sparse_switching",
        seed=5,
        max_generation_attempts=1,
    )
    sequence = generate_traffic_sequence(config)
    stacked = np.stack(sequence.matrices)
    assert np.mean(stacked == 0) > 0.75
    active_masks = [matrix > 0 for matrix in sequence.matrices[: config.window_size]]
    assert any(not np.array_equal(active_masks[0], mask) for mask in active_masks[1:])


def test_moving_hotspot_changes_peak_destination():
    sequence = generate_traffic_sequence(_config("moving_hotspot"))
    peak_destinations = [int(matrix.sum(axis=0).argmax()) for matrix in sequence.matrices[:16]]
    assert len(set(peak_destinations)) > 1


def test_impossible_tight_bounds_raise_with_diagnostics():
    config = TrafficProcessConfig(
        num_nodes=3,
        sequence_length=8,
        window_size=8,
        mean_level=0.3,
        std_level=0.0,
        max_entry=1,
        epsilon_mean=0.0,
        epsilon_var=0.0,
        family="bimodal",
        seed=1,
        max_generation_attempts=1,
    )
    with pytest.raises(RuntimeError, match="max_mean_error"):
        generate_traffic_sequence(config)


def test_validation_rejects_a_corrupted_window():
    sequence = generate_traffic_sequence(_config("bimodal"))
    matrices = [matrix.copy() for matrix in sequence.matrices]
    for matrix in matrices[:16]:
        matrix[:] = 0
    corrupted = TrafficSequence(
        sequence.sequence_id,
        sequence.topology_name,
        sequence.family,
        sequence.seed,
        matrices,
        sequence.mean_ref,
        sequence.var_ref,
        sequence.bounds,
        sequence.metadata,
    )
    assert not validate_sequence_moment_bounds(corrupted)["passed"]
