import numpy as np
import pytest

from rlccl.envs.problem import TopologyInfo
from rlccl.envs.sequence_env import TrafficSequenceRunner
from rlccl.evaluation.traffic_audit import autocorrelation
from rlccl.traffic.long_horizon_generator import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    SPATIAL_MODES,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
    generate_same_moment_group,
)
from rlccl.traffic.matrix_utils import validate_traffic_matrix


def _config(family, seed=42, length=1024, **overrides):
    values = dict(
        num_nodes=4,
        sequence_length=length,
        family=family,
        seed=seed,
        calibration_candidates=3,
        topology_name="Rear4GPU",
    )
    values.update(overrides)
    return LongHorizonTrafficConfig(**values)


@pytest.mark.parametrize("family", LONG_HORIZON_FAMILIES)
def test_long_families_are_reproducible_legal_and_seeded(family):
    first = generate_long_horizon_sequence(_config(family))
    second = generate_long_horizon_sequence(_config(family))
    different = generate_long_horizon_sequence(_config(family, seed=43))
    assert first.to_dict() == second.to_dict()
    assert any(not np.array_equal(a, b) for a, b in zip(first.matrices, different.matrices))
    for matrix in first.matrices:
        validate_traffic_matrix(matrix)
        assert np.issubdtype(matrix.dtype, np.integer)
        assert matrix.min() >= 0
        assert matrix.max() <= first.metadata["generator_config"]["max_entry"]
        assert np.all(np.diag(matrix) == 0)
    assert first.metadata["generator_kind"] == "long-horizon generator"
    assert first.metadata["metadata_usage"].startswith("audit/evaluation only")
    assert first.metadata["pre_clip_statistics"]
    assert first.metadata["post_clip_statistics"]
    spatial_validation = first.metadata["spatial_distribution_validation"]
    assert spatial_validation["minimum_probability"] >= 0.0
    assert spatial_validation["maximum_sum_error"] < 1e-12
    assert spatial_validation["diagonal_nonzero_count"] == 0


@pytest.mark.parametrize("family", LONG_HORIZON_FAMILIES)
def test_long_constraints_pass_without_short_window_hard_rejection(family):
    sequence = generate_long_horizon_sequence(_config(family, seed=7))
    constraints = sequence.metadata["multi_scale_constraints"]
    matrix_constraints = sequence.metadata["matrix_multi_scale_constraints"]
    assert constraints["short"]["hard_or_soft_bound_applied"] is False
    assert constraints["short"]["violation_fraction"] is None
    assert constraints["medium"]["passed"], constraints["medium"]
    assert constraints["long"]["passed"], constraints["long"]
    assert matrix_constraints["short"]["hard_or_soft_bound_applied"] is False
    assert matrix_constraints["medium"]["passed"], matrix_constraints["medium"]
    assert matrix_constraints["long"]["passed"], matrix_constraints["long"]
    assert sequence.metadata["constraint_status"] == "passed"


def test_short_windows_can_deviate_while_long_window_recovers():
    sequence = generate_long_horizon_sequence(_config("rare_shock_recovery", seed=42))
    constraints = sequence.metadata["multi_scale_constraints"]
    assert constraints["short"]["mean_error_max"] > 0.50
    assert constraints["short"]["variance_error_max"] > 2.0
    assert constraints["long"]["passed"]
    assert constraints["long"]["violation_fraction"] <= 0.05


def test_rare_shock_has_duration_flags_and_recovery():
    sequence = generate_long_horizon_sequence(_config("rare_shock_recovery", seed=5))
    metadata = sequence.metadata
    assert sum(metadata["shock_flags"]) > 0
    assert metadata["shock_records"]
    for record in metadata["shock_records"]:
        assert 1 <= record["duration"] <= 16
        assert record["recovery_end"] >= record["end"]
    total = np.asarray(metadata["total_traffic_pre_integer"])
    record = metadata["shock_records"][0]
    assert total[record["start"] : record["end"]].max() > np.median(total)


def test_regime_dwell_is_random_bounded_and_not_fixed():
    sequence = generate_long_horizon_sequence(_config("regime_switching_long", seed=9, length=4096))
    records = sequence.metadata["regime_dwell_records"]
    complete_lengths = [record["length"] for record in records[:-1]]
    assert complete_lengths
    assert all(32 <= length <= 512 for length in complete_lengths)
    assert len(set(complete_lengths)) > 1
    assert {record["state"] for record in records} == {"low", "normal", "high"}


def test_hotspot_random_walk_has_random_dwell_and_no_fixed_cycle():
    sequence = generate_long_horizon_sequence(_config("hotspot_random_walk", seed=13, length=2048))
    records = sequence.metadata["hotspot_dwell_records"]
    destinations = np.asarray(sequence.metadata["hotspot_destination"])
    dwell = [record["length"] for record in records[:-1]]
    assert len(set(dwell)) > 1
    assert len(set(destinations.tolist())) > 2
    assert not np.array_equal(destinations[:1024], destinations[1024:])
    transition_steps = np.flatnonzero(destinations[1:] != destinations[:-1]) + 1
    assert len(set(np.diff(transition_steps).tolist())) > 1


def test_all_spatial_modes_are_supported_and_materially_distinct():
    mean_source_share = {}
    mean_destination_share = {}
    cross_group_share = {}
    groups = np.array([0, 0, 1, 1])
    cross = groups[:, None] != groups[None, :]
    for mode in SPATIAL_MODES:
        sequence = generate_long_horizon_sequence(
            _config("hotspot_random_walk", seed=31, length=512, spatial_mode=mode)
        )
        stack = np.stack(sequence.matrices).astype(float)
        total = np.maximum(stack.sum(axis=(1, 2)), 1.0)
        mean_source_share[mode] = float(np.mean(stack.sum(axis=2).max(axis=1) / total))
        mean_destination_share[mode] = float(np.mean(stack.sum(axis=1).max(axis=1) / total))
        cross_group_share[mode] = float(np.mean(stack[:, cross].sum(axis=1) / total))
    assert mean_source_share["single_source_hotspot"] > mean_source_share["balanced"]
    assert mean_destination_share["single_destination_hotspot"] > mean_destination_share["balanced"]
    assert mean_source_share["dual_hotspot"] > mean_source_share["balanced"]
    assert mean_destination_share["dual_hotspot"] > mean_destination_share["balanced"]
    assert cross_group_share["cross_group_concentration"] > cross_group_share["balanced"]


def test_same_moments_group_matches_moments_but_not_dynamics():
    group = generate_same_moment_group(
        _config("same_moments_different_dynamics", seed=77, length=2048)
    )
    assert set(group) == set(SAME_MOMENT_VARIANTS)
    means, variances, lag8, maxima = {}, {}, {}, {}
    for variant, sequence in group.items():
        total = np.stack(sequence.matrices).sum(axis=(1, 2)).astype(float)
        means[variant] = float(total.mean())
        variances[variant] = float(total.var(ddof=0))
        lag8[variant] = autocorrelation(total, [8])["values"]["8"]
        maxima[variant] = float(total.max())
    assert max(means.values()) - min(means.values()) < 0.10
    assert max(variances.values()) / min(variances.values()) < 1.02
    assert max(lag8.values()) - min(lag8.values()) > 0.50
    assert maxima["shock_recovery"] > max(
        maxima["smooth"], maxima["random_switching"], maxima["long_regime"]
    )
    assert sum(group["shock_recovery"].metadata["shock_flags"]) > 0


def test_long_sequences_keep_estimator_state_isolated_and_history_only():
    first = generate_long_horizon_sequence(_config("stochastic_volatility", seed=101, length=64))
    second = generate_long_horizon_sequence(_config("stochastic_volatility", seed=202, length=64))
    edges = np.asarray([(src, dst) for src in range(4) for dst in range(4) if src != dst])
    topology = TopologyInfo(4, len(edges), edges, np.ones(len(edges)), [], name="complete4")
    first_rows = iter(TrafficSequenceRunner(first, topology, min_history=2))
    second_rows = iter(TrafficSequenceRunner(second, topology, min_history=2))
    _, first_context_0, _ = next(first_rows)
    _, second_context_0, _ = next(second_rows)
    assert first_context_0.history_length == second_context_0.history_length == 0
    _, first_context_1, _ = next(first_rows)
    assert first_context_1.history_length == 1
    np.testing.assert_allclose(first_context_1.mean_matrix, first.matrices[0])
    assert second_context_0.history_length == 0
