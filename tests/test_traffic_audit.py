import numpy as np

from rlccl.evaluation.traffic_audit import (
    audit_sequence,
    autocorrelation,
    run_lengths,
    summarize_audits,
)
from rlccl.traffic.process_generator import TrafficProcessConfig, generate_traffic_sequence


def _sequence(length=64, family="smooth_ar", seed=42):
    return generate_traffic_sequence(
        TrafficProcessConfig(
            num_nodes=4,
            sequence_length=length,
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
    )


def test_run_lengths_handles_edges_and_empty_runs():
    assert run_lengths([True, True, False, True, False, True, True, True]) == [2, 1, 3]
    assert run_lengths([False, False]) == []


def test_constant_acf_is_explicitly_undefined():
    result = autocorrelation(np.ones(32), [1, 2, 4])
    assert not result["defined"]
    assert result["effective_sample_size"] is None
    assert all(value is None for value in result["values"].values())


def test_existing_generator_is_detected_as_exactly_periodic():
    result = audit_sequence(_sequence(), short_window=16, medium_window=32, long_window=64)
    assert result["temporal"]["periodicity"]["detected_exact_period"] == 16
    assert result["temporal"]["exact_duplicate_ratio"] >= 0.75
    assert result["generation"]["intermediate_instrumentation_available"] is False
    assert result["generation"]["pre_clip"] is None


def test_multi_window_moments_use_all_overlapping_windows():
    result = audit_sequence(_sequence(length=64, family="bimodal"), short_window=16, medium_window=32, long_window=128)
    windows = result["multi_window_moments"]
    assert windows["16"]["num_windows"] == 49
    assert windows["32"]["num_windows"] == 33
    assert windows["16"]["any_violation_fraction"] == 0.0
    assert not windows["128"]["available"]


def test_spatial_group_metric_is_not_invented_without_routing():
    result = audit_sequence(_sequence())
    assert set(result["spatial"]["source_load_by_node"]) == {"0", "1", "2", "3"}
    assert set(result["spatial"]["destination_load_by_node"]) == {"0", "1", "2", "3"}
    group = result["spatial"]["bandwidth_group_concentration"]
    assert group["available"] is False
    assert "routing" in group["reason"]


def test_summary_preserves_family_length_and_base_seed_groups():
    audited = audit_sequence(_sequence(), generation_seconds=0.1)
    records = []
    for seed_base in (42, 142):
        records.append(
            {
                **audited,
                "status": "success",
                "seed_base": seed_base,
                "sequence_index": 0,
                "actual_seed": seed_base,
            }
        )
    summary = summarize_audits(records)
    assert len(summary["groups"]) == 2
    assert {group["seed_base"] for group in summary["groups"]} == {42, 142}
