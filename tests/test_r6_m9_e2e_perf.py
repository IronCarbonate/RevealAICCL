"""R6-M9 fairness-gate and paired-statistics contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rlccl.ep.perf_stats import interval_overlap, paired_bootstrap


def test_paired_bootstrap_classifies_positive_negative_and_crossing_zero() -> None:
    passed = paired_bootstrap([1.0] * 20, [1.2] * 20, samples=1000)
    failed = paired_bootstrap([1.2] * 20, [1.0] * 20, samples=1000)
    inconclusive = paired_bootstrap(
        [1.0] * 20, [0.9, 1.1] * 10, samples=2000,
    )
    assert passed["performance"] == "PASS" and passed["ci95_lower_ms"] > 0
    assert failed["performance"] == "FAIL" and failed["ci95_upper_ms"] < 0
    assert inconclusive["performance"] == "INCONCLUSIVE"


def test_bootstrap_is_paired_and_deterministic() -> None:
    progressive = np.linspace(1.0, 2.0, 30)
    delayed = progressive + np.linspace(0.01, 0.03, 30)
    first = paired_bootstrap(progressive, delayed, samples=500, seed=7)
    second = paired_bootstrap(progressive, delayed, samples=500, seed=7)
    assert first == second
    assert first["pairs"] == 30


def test_interval_overlap_handles_disjoint_nested_and_partial_intervals() -> None:
    assert interval_overlap(0, 10, 12, 20) == 0
    assert interval_overlap(0, 10, 2, 7) == 5
    assert interval_overlap(0, 10, 8, 12) == 2


def test_m9_uses_one_external_gate_and_same_frozen_fast_path_for_both_arms() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "r6_m9_e2e_perf" /
              "e2e_perf_runtime.cu").read_text()
    assert "enum BenchmarkArm" in source
    assert "commit_gate_role(" in source
    assert "arm == kDelayedArm" in source
    assert "m9_pipeline_kernel<<<5, 256" in source
    assert "progressive_dispatch_progress_role" in source
    assert "progressive_combine_kernel<<<1, 256" in source
    assert "combine_reduce_epilogue<<<num_tokens, 32" in source
    assert source.count("arm == kDelayedArm") == 1


def test_delayed_gate_waits_for_final_router_before_forwarding() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "r6_m9_e2e_perf" /
              "e2e_perf_runtime.cu").read_text()
    delayed = source.index("if (arm == kDelayedArm)")
    wait = source.index("while (*final_value == 0)", delayed)
    publish = source.index("consumer_tail.store", delayed)
    assert delayed < wait < publish


def test_repeated_trials_use_disjoint_lsa_completion_ranges() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "r6_m9_e2e_perf" /
              "e2e_perf_runtime.cu").read_text()
    assert "kMaxBenchmarkRuns = 640" in source
    assert "run_epoch * (layout.max_descriptors + 2)" in source
    assert "completion_base + layout.max_descriptors" in source


def test_preflight_window_barrier_is_outside_measured_interval() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "r6_m9_e2e_perf" /
              "e2e_perf_runtime.cu").read_text()
    preflight = source.index("preflight_lsa_barrier_kernel<<<")
    start = source.index("cudaEventRecord(e2e_start", preflight)
    assert preflight < start


def test_m7_and_m8_extensions_are_not_referenced_as_alternate_arms() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "ep" / "gpu_e2e_perf.py").read_text()
    assert "arm_value" in source
    assert "progressive" in source and "delayed" in source
    assert "libdeepep_dispatch_runtime" not in source
    assert "libhandle_combine_runtime" not in source
