"""RED contracts for the frozen Phase 3B experiment and safe runner.

Production imports are deliberately delayed so this module can be collected
before the experiment implementation and runner exist.  These tests exercise
only synthetic rows/specifications; they never generate the 75 formal traffic
sequences and never write the formal output directory.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pytest


FAMILIES = (
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
)
BASE_SEEDS = (342, 442, 542)
SPLITS = ("fit", "fit", "validation", "calibration", "test")
CHECKPOINTS = tuple(range(32, 993, 64))
REVEAL_MODES = (
    "random_entries",
    "source_totals_first",
    "source_destination_totals_first",
    "partial_shards",
    "time_based_arrival",
)
REVEAL_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
UNKNOWN_RATIOS = REVEAL_RATIOS[:-1]
REQUESTED_K = (1, 4, 8, 16)
ORDINARY_METHODS = (
    "random_empirical",
    "worst_recent_cases",
    "boundary_scenarios",
    "minimax_subset",
)
ALL_METHODS = ORDINARY_METHODS + ("oracle_support_upper_bound",)
TIE_ORDER = (
    "minimax_subset",
    "boundary_scenarios",
    "worst_recent_cases",
    "random_empirical",
)
FORMAL_OUTPUT = Path("outputs/phase3b_ambiguity")
ARTIFACT_NAMES = (
    "manifest.json",
    "raw_case_metrics.csv",
    "raw_sequence_metrics.csv",
    "summary.json",
)
FORBIDDEN_IMPORTS = {
    "torch",
    "rlccl.uncertainty.execution",
    "rlccl.envs.decoder",
    "rlccl.scheduling",
}
FORBIDDEN_API_FRAGMENTS = {
    "proposal",
    "transferaction",
    "truthtokenid",
    "commit",
    "prefix",
    "horizon",
    "robustscore",
    "repair",
    "recourse",
}


def _resolved_imports(source: str, module_name: str) -> set[str]:
    """Resolve relative ImportFrom nodes and possible imported submodules."""

    tree = ast.parse(source)
    imported: set[str] = set()
    package_parts = module_name.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = max(0, len(package_parts) - (node.level - 1))
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            imported.add(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            imported.add(".".join(part for part in (base, alias.name) if part))
    return imported


def _normalized_public_name(name: Any) -> str:
    return "".join(character for character in str(name).casefold() if character != "_")


def _forbidden_public_names(names: Iterable[Any]) -> set[str]:
    return {
        str(name)
        for name in names
        if any(
            fragment in _normalized_public_name(name)
            for fragment in FORBIDDEN_API_FRAGMENTS
        )
    }


def _api() -> Any:
    return importlib.import_module("rlccl.uncertainty.ambiguity_experiment")


def _runner() -> ModuleType:
    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_phase3b_ambiguity.py"
    assert runner_path.is_file(), "Phase 3B runner has not been implemented"
    spec = importlib.util.spec_from_file_location("_phase3b_runner_contract", runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item[name]
    return getattr(item, name)


def _canonical(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "data": item.tolist(),
            }
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [normalize(element) for element in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for seed_index, base_seed in enumerate(BASE_SEEDS):
            sequence_id = f"cal-{family_index}-{seed_index}"
            # A distinct, deterministic 320-case universe per sequence.
            scores = np.arange(320, dtype=np.float64) / 1000.0
            scores += family_index / 100.0 + seed_index / 10000.0
            for case_index, score in enumerate(scores):
                checkpoint_index, remainder = divmod(case_index, 20)
                mode_index, stage_index = divmod(remainder, 4)
                rows.append(
                    {
                        "family": family,
                        "base_seed": base_seed,
                        "sequence_id": sequence_id,
                        "checkpoint": CHECKPOINTS[checkpoint_index],
                        "reveal_mode": REVEAL_MODES[mode_index],
                        "reveal_ratio": UNKNOWN_RATIOS[stage_index],
                        "score": float(score),
                    }
                )
    return rows


def _validation_rows(method_means: Mapping[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence_index in range(15):
        for method, mean_value in method_means.items():
            # Two values with the requested sequence mean, to expose accidental
            # raw-case weighting while keeping the toy input compact.
            rows.extend(
                [
                    {
                        "sequence_id": f"validation-{sequence_index}",
                        "method": method,
                        "requested_k": 8,
                        "reveal_ratio": 0.0,
                        "nearest_rms_distance": mean_value - 0.01,
                    },
                    {
                        "sequence_id": f"validation-{sequence_index}",
                        "method": method,
                        "requested_k": 8,
                        "reveal_ratio": 0.75,
                        "nearest_rms_distance": mean_value + 0.01,
                    },
                ]
            )
    return rows


def _passing_evidence() -> dict[str, Any]:
    return {
        "selected_joint_coverage": 0.90,
        "selected_joint_coverage_by_family": {family: 0.86 for family in FAMILIES},
        "paired_delta_ci95": [0.01, 0.09],
        "paired_delta_by_base_seed": {str(seed): 0.03 for seed in BASE_SEEDS},
        "paired_delta_by_family": {family: 0.03 for family in FAMILIES},
        "lofo_aggregate_delta": 0.02,
        "lofo_delta_by_family": {family: 0.01 for family in FAMILIES},
        "lofo_relative_degradation_by_family": {family: 0.0 for family in FAMILIES},
        "total_tail_hits": 9,
        "total_tail_events": 12,
        "group_tail_hits": 18,
        "group_tail_events": 24,
        "hotspot_hits": 250,
        "hotspot_events": 320,
        "selected_mean_physical_normalized_width": 0.70,
        "ordinary_invalid_or_empty_rate": {
            method: 0.0 for method in ORDINARY_METHODS
        },
        "ratio1_singleton_coverage": 1.0,
        "all_timings_finite": True,
        "integrity_checks_complete": True,
        "integrity_checks_passed": True,
    }


def _metric_fixture() -> dict[str, Any]:
    truth_descriptor = np.asarray([2.0, 2.0, 1.0], dtype=np.float64)
    support_descriptors = np.asarray(
        [[1.0, 2.0, 1.0], [3.0, 1.0, 1.0]], dtype=np.float64
    )
    pool_descriptors = np.asarray(
        [[0.0, 2.0, 1.0], [1.0, 2.0, 1.0], [3.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    truth_matrix = np.asarray([[0, 2], [0, 0]], dtype=np.int64)
    support_matrices = np.asarray(
        [[[0, 1], [0, 0]], [[0, 3], [0, 0]]], dtype=np.int64
    )
    return {
        "truth_descriptor": truth_descriptor,
        "truth_matrix": truth_matrix,
        "support_descriptors": support_descriptors,
        "support_matrices": support_matrices,
        "pool_descriptors": pool_descriptors,
        "fit_scale": np.ones(3, dtype=np.float64),
        "lower": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "upper": np.asarray([3.0, 3.0, 1.0], dtype=np.float64),
        "physical_low": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        "physical_high": np.asarray([4.0, 4.0, 1.0], dtype=np.float64),
        "truth_satisfies_observation": True,
        "total_index": 0,
        "group_indices": (2,),
        "total_tail_threshold": 1.5,
        "group_tail_thresholds": np.asarray([0.5], dtype=np.float64),
        "truth_hotspot_destination": 1,
        "support_hotspot_destinations": (0, 1),
        "requested_k": 2,
    }


def test_phase3b_experiment_is_a_delayed_red_dependency() -> None:
    assert importlib.util.find_spec("rlccl.uncertainty.ambiguity_experiment") is not None
    api = _api()
    required = {
        "build_formal_sequence_specs",
        "validate_sequence_records",
        "reveal_seed",
        "replicate_seed",
        "calibration_exceedance_score",
        "calibrate_envelope_radius",
        "select_validation_method",
        "build_lofo_fold",
        "compute_case_metrics",
        "aggregate_random_replicates",
        "aggregate_sequence_metrics",
        "family_stratified_sequence_bootstrap",
        "evaluate_phase3b_gate",
        "validate_raw_case_rows",
        "recompute_artifacts",
        "write_experiment_artifacts",
        "run_toy_experiment",
    }
    assert required.issubset(set(vars(api)))


def test_frozen_constants_and_75_fresh_sequence_specs_without_generation() -> None:
    api = _api()
    assert tuple(api.FORMAL_FAMILIES) == FAMILIES
    assert tuple(api.FORMAL_BASE_SEEDS) == BASE_SEEDS
    assert tuple(api.FORMAL_SPLITS) == SPLITS
    assert tuple(api.CHECKPOINTS) == CHECKPOINTS
    assert tuple(api.REVEAL_MODES) == REVEAL_MODES
    assert tuple(api.REVEAL_RATIOS) == REVEAL_RATIOS
    assert tuple(api.REQUESTED_K) == REQUESTED_K
    assert tuple(api.ORDINARY_METHODS) == ORDINARY_METHODS
    assert tuple(api.ALL_METHODS) == ALL_METHODS
    assert api.RANDOM_REPLICATES == 8
    assert api.HISTORY_WINDOW == 32
    assert api.FORMAL_SEQUENCE_LENGTH == 1024
    assert api.FORMAL_MAX_ENTRY == 8

    specs = tuple(api.build_formal_sequence_specs())
    assert len(specs) == 75
    assert Counter(_value(spec, "split") for spec in specs) == {
        "fit": 30,
        "validation": 15,
        "calibration": 15,
        "test": 15,
    }
    assert len({_value(spec, "sequence_id") for spec in specs}) == 75
    expected_variants = tuple(getattr(api, "SAME_MOMENT_VARIANTS"))
    assert len(expected_variants) == 4

    for record_index, spec in enumerate(specs):
        family_index, remainder = divmod(record_index, 15)
        seed_index, sequence_index = divmod(remainder, 5)
        assert _value(spec, "record_index") == record_index
        assert _value(spec, "family") == FAMILIES[family_index]
        assert _value(spec, "base_seed") == BASE_SEEDS[seed_index]
        assert _value(spec, "sequence_index") == sequence_index
        assert _value(spec, "split") == SPLITS[sequence_index]
        assert _value(spec, "actual_seed") == (
            BASE_SEEDS[seed_index] + family_index * 1_000_000 + sequence_index * 10_000
        )
        assert _value(spec, "sequence_length") == 1024
        config = _value(spec, "generator_config")
        assert _value(config, "mean_level") == 2.0
        assert _value(config, "std_level") == 1.5
        assert _value(config, "max_entry") == 8
        assert _value(config, "calibration_candidates") == 1
        if FAMILIES[family_index] == "same_moments_different_dynamics":
            assert _value(spec, "same_moments_variant") == expected_variants[
                (seed_index + sequence_index) % 4
            ]


def test_fresh_corpus_digest_and_split_overlap_are_hard_errors() -> None:
    api = _api()
    specs = tuple(api.build_formal_sequence_specs())
    records = [
        {
            "sequence_id": _value(spec, "sequence_id"),
            "split": _value(spec, "split"),
            "sequence_digest": _sha(f"fresh-{index}"),
            "record_index": index,
        }
        for index, spec in enumerate(specs)
    ]
    validated = api.validate_sequence_records(records, h1_sequence_digests={_sha("h1")})
    assert len(validated) == 75

    duplicate_id = [dict(row) for row in records]
    duplicate_id[-1]["sequence_id"] = duplicate_id[0]["sequence_id"]
    with pytest.raises(ValueError, match="sequence.*(duplicate|overlap|unique)"):
        api.validate_sequence_records(duplicate_id, h1_sequence_digests=set())

    duplicate_digest = [dict(row) for row in records]
    duplicate_digest[-1]["sequence_digest"] = duplicate_digest[0]["sequence_digest"]
    with pytest.raises(ValueError, match="digest.*(duplicate|overlap|unique)"):
        api.validate_sequence_records(duplicate_digest, h1_sequence_digests=set())

    h1_overlap = [dict(row) for row in records]
    with pytest.raises(ValueError, match="H1|h1|digest.*overlap"):
        api.validate_sequence_records(
            h1_overlap, h1_sequence_digests={h1_overlap[7]["sequence_digest"]}
        )


def test_checkpoint_case_registry_and_seed_formulas_are_canonical() -> None:
    api = _api()
    assert CHECKPOINTS == (
        32, 96, 160, 224, 288, 352, 416, 480,
        544, 608, 672, 736, 800, 864, 928, 992,
    )
    for record_index in (0, 17, 74):
        for checkpoint in (32, 992):
            for mode_index in range(5):
                assert api.reveal_seed(record_index, checkpoint, mode_index) == (
                    31_000_000 + record_index * 100_000 + checkpoint * 10 + mode_index
                )
    for case_index in (0, 1, 31_999):
        seeds = [api.replicate_seed(case_index, replicate) for replicate in range(8)]
        assert seeds == [41_000_000 + case_index * 100 + replicate for replicate in range(8)]
        assert len(set(seeds)) == 8

    specs = tuple(api.build_formal_sequence_specs())
    registry = tuple(api.build_case_registry(specs))
    evaluated_specs = tuple(spec for spec in specs if _value(spec, "split") != "fit")
    fit_record_indices = {
        _value(spec, "record_index") for spec in specs if _value(spec, "split") == "fit"
    }
    evaluated_record_indices = {
        _value(spec, "record_index") for spec in evaluated_specs
    }
    assert len(evaluated_specs) == 45
    assert len(fit_record_indices) == 30
    expected_count = 45 * 16 * 5 * 5 * 4
    assert len(registry) == expected_count == 72_000
    assert tuple(_value(row, "case_index") for row in registry) == tuple(
        range(expected_count)
    )
    assert all(_value(row, "split") in {"validation", "calibration", "test"} for row in registry)
    assert {_value(row, "record_index") for row in registry} == evaluated_record_indices
    assert not ({_value(row, "record_index") for row in registry} & fit_record_indices)
    identities = [
        (
            _value(row, "split"), _value(row, "record_index"),
            _value(row, "checkpoint"), _value(row, "mode_index"),
            _value(row, "stage_index"), _value(row, "requested_k"),
        )
        for row in registry
    ]
    expected_identities = {
        (
            _value(spec, "split"), _value(spec, "record_index"), checkpoint,
            mode_index, stage_index, requested_k,
        )
        for spec in evaluated_specs
        for checkpoint in CHECKPOINTS
        for mode_index in range(len(REVEAL_MODES))
        for stage_index in range(len(REVEAL_RATIOS))
        for requested_k in REQUESTED_K
    }
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities)) == len(expected_identities)
    assert set(identities) == expected_identities
    assert all(_value(row, "reveal_ratio") in REVEAL_RATIOS for row in registry)


def test_calibration_joint_exceedance_score_matches_hand_calculation() -> None:
    api = _api()
    raw_lower = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    raw_upper = np.asarray([3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    fit_scale = np.asarray([2.0, 0.5, 1.0, 4.0], dtype=np.float64)
    truth = np.asarray([0.0, 7.0, 4.0, 4.0], dtype=np.float64)
    lower_violation = (raw_lower - truth) / fit_scale
    upper_violation = (truth - raw_upper) / fit_scale
    expected = float(
        np.max(np.maximum(np.maximum(lower_violation, upper_violation), 0.0))
    )
    assert expected == 6.0
    assert api.calibration_exceedance_score(
        raw_lower, raw_upper, truth, fit_scale
    ) == pytest.approx(expected)

    inside = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    assert api.calibration_exceedance_score(
        raw_lower, raw_upper, inside, fit_scale
    ) == 0.0
    lower_dominant = np.asarray([-3.0, 3.0, 4.0, 5.0], dtype=np.float64)
    assert api.calibration_exceedance_score(
        raw_lower, raw_upper, lower_dominant, fit_scale
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="scale|positive|finite"):
        api.calibration_exceedance_score(
            raw_lower, raw_upper, truth, np.asarray([2.0, 0.0, 1.0, 4.0])
        )


def test_calibration_is_exact_two_level_higher_quantile_and_k_selector_free() -> None:
    api = _api()
    rows = _calibration_rows()
    assert len(rows) == 15 * 320
    radius = api.calibrate_envelope_radius(rows)
    sequence_quantiles = []
    for sequence_id in sorted({row["sequence_id"] for row in rows}):
        values = [row["score"] for row in rows if row["sequence_id"] == sequence_id]
        assert len(values) == 320
        sequence_quantiles.append(float(np.quantile(values, 0.9, method="higher")))
    expected = float(np.quantile(sequence_quantiles, 0.9, method="higher"))
    assert radius == expected

    polluted = [dict(row, method="minimax_subset", requested_k=16) for row in rows]
    assert api.calibrate_envelope_radius(polluted) == expected
    with pytest.raises(ValueError, match="320|universe|calibration"):
        api.calibrate_envelope_radius(rows[:-1])
    with pytest.raises(ValueError, match="ratio|unknown|calibration"):
        api.calibrate_envelope_radius([*rows, dict(rows[0], reveal_ratio=1.0)])
    equal_length_identity_corruption = [dict(row) for row in rows]
    equal_length_identity_corruption[-1] = dict(equal_length_identity_corruption[-2])
    assert len(equal_length_identity_corruption) == len(rows)
    with pytest.raises(
        ValueError, match="duplicate|missing|identity|universe|calibration"
    ):
        api.calibrate_envelope_radius(equal_length_identity_corruption)


@pytest.mark.parametrize("held_out_family", FAMILIES)
def test_lofo_calibration_excludes_held_family_and_uses_exact_12_sequences(
    held_out_family: str,
) -> None:
    api = _api()
    rows = _calibration_rows()
    radius = api.calibrate_envelope_radius(rows, held_out_family=held_out_family)
    seen = [row for row in rows if row["family"] != held_out_family]
    assert len({row["sequence_id"] for row in seen}) == 12
    expected_sequence_quantiles = []
    for sequence_id in sorted({row["sequence_id"] for row in seen}):
        values = [row["score"] for row in seen if row["sequence_id"] == sequence_id]
        expected_sequence_quantiles.append(
            float(np.quantile(values, 0.9, method="higher"))
        )
    assert radius == float(np.quantile(expected_sequence_quantiles, 0.9, method="higher"))

    poisoned = [dict(row) for row in rows]
    for row in poisoned:
        if row["family"] == held_out_family:
            row["score"] = 1e12
    assert api.calibrate_envelope_radius(
        poisoned, held_out_family=held_out_family
    ) == radius


def test_lofo_fold_completely_excludes_held_family_from_fit_validation_calibration() -> None:
    api = _api()
    specs = tuple(api.build_formal_sequence_specs())
    for held_out_family in FAMILIES:
        fold = api.build_lofo_fold(specs, held_out_family=held_out_family)
        for split_name in ("fit", "validation", "calibration"):
            items = tuple(_value(fold, split_name))
            assert items
            assert all(_value(item, "family") != held_out_family for item in items)
        test_items = tuple(_value(fold, "test"))
        assert len(test_items) == 3
        assert {_value(item, "family") for item in test_items} == {held_out_family}


def test_validation_selection_is_sequence_equal_k8_unknown_only_with_frozen_ties() -> None:
    api = _api()
    rows = _validation_rows(
        {
            "random_empirical": 2.0,
            "worst_recent_cases": 1.5,
            "boundary_scenarios": 1.0,
            "minimax_subset": 0.5,
        }
    )
    selected = api.select_validation_method(rows)
    assert selected.method == "minimax_subset"
    assert selected.requested_k == 8
    assert selected.sequence_count == 15
    assert tuple(selected.tie_order) == TIE_ORDER

    for expected in TIE_ORDER:
        tied = {method: 1.0 for method in ORDINARY_METHODS}
        # Earlier tie-order choices are made worse; remaining methods tie.
        for earlier in TIE_ORDER[: TIE_ORDER.index(expected)]:
            tied[earlier] = 1.0 + 2e-12
        result = api.select_validation_method(_validation_rows(tied))
        assert result.method == expected

    contaminated = [*rows, dict(rows[0], requested_k=16, nearest_rms_distance=-1e9)]
    contaminated.append(dict(rows[0], reveal_ratio=1.0, nearest_rms_distance=-1e9))
    assert api.select_validation_method(contaminated).method == "minimax_subset"


def test_metric_formulas_match_hand_calculation_and_tail_semantics() -> None:
    api = _api()
    metrics = api.compute_case_metrics(**_metric_fixture())
    assert metrics.nearest_rms_distance == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics.nearest_matrix_l1_distance == 1
    assert metrics.covering_radius == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics.mean_pairwise_diversity == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics.duplicate_fraction == 0.0
    assert metrics.actual_k == 2
    assert metrics.requested_k == 2
    assert metrics.component_coverage == 1.0
    assert metrics.joint_coverage == 1
    # The third component has zero physical range and is excluded.
    assert metrics.physical_normalized_mean_width == pytest.approx((3 / 4 + 2 / 4) / 2)
    assert metrics.zero_physical_range_components == 1
    assert metrics.total_tail_event == 1
    assert metrics.total_tail_hit == 1
    assert metrics.group_tail_events == 1
    assert metrics.group_tail_hits == 1
    assert metrics.hotspot_event == 1
    assert metrics.hotspot_hit == 1

    duplicate = _metric_fixture()
    duplicate["support_matrices"] = np.repeat(
        duplicate["support_matrices"][:1], repeats=2, axis=0
    )
    duplicate["support_descriptors"] = np.repeat(
        duplicate["support_descriptors"][:1], repeats=2, axis=0
    )
    assert api.compute_case_metrics(**duplicate).duplicate_fraction == 0.5


def test_joint_coverage_requires_hard_observation_and_tolerance_is_exact() -> None:
    api = _api()
    values = _metric_fixture()
    values["truth_descriptor"] = values["upper"].copy()
    values["truth_descriptor"][0] += 1e-10
    assert api.compute_case_metrics(**values).joint_coverage == 1
    values["truth_descriptor"][0] += 1e-11
    assert api.compute_case_metrics(**values).joint_coverage == 0
    values = _metric_fixture()
    values["truth_satisfies_observation"] = False
    metrics = api.compute_case_metrics(**values)
    assert metrics.component_coverage == 1.0
    assert metrics.joint_coverage == 0


def test_random_eight_replicates_are_case_averaged_without_envelope_reweighting() -> None:
    api = _api()
    rows = [
        {
            "case_id": "paired-case",
            "method": "random_empirical",
            "replicate_index": replicate,
            "nearest_rms_distance": float(replicate),
            "joint_coverage": 1,
            "lower": [0.0, 1.0],
            "upper": [2.0, 3.0],
        }
        for replicate in range(8)
    ]
    result = api.aggregate_random_replicates(rows)
    assert result.replicate_count == 8
    assert result.nearest_rms_distance == pytest.approx(3.5)
    assert result.joint_coverage == 1
    assert result.envelope_count == 1
    assert result.lower == [0.0, 1.0]
    assert result.upper == [2.0, 3.0]
    with pytest.raises(ValueError, match="8|replicate"):
        api.aggregate_random_replicates(rows[:-1])
    with pytest.raises(ValueError, match="replicate|duplicate"):
        api.aggregate_random_replicates([*rows[:-1], rows[0]])
    inconsistent_envelope = [dict(row) for row in rows]
    inconsistent_envelope[-1]["upper"] = [2.0, 3.5]
    with pytest.raises(ValueError, match="envelope|lower|upper|consistent"):
        api.aggregate_random_replicates(inconsistent_envelope)


def test_sequence_aggregation_requires_exact_320_equal_weight_cases() -> None:
    api = _api()
    rows = []
    for case_index in range(320):
        checkpoint_index, remainder = divmod(case_index, 20)
        mode_index, ratio_index = divmod(remainder, 4)
        rows.append(
            {
                "sequence_id": "test-sequence",
                "method": "minimax_subset",
                "requested_k": 8,
                "checkpoint": CHECKPOINTS[checkpoint_index],
                "reveal_mode": REVEAL_MODES[mode_index],
                "reveal_ratio": UNKNOWN_RATIOS[ratio_index],
                "nearest_rms_distance": float(case_index),
                "total_tail_events": int(case_index in {1, 9}),
                "total_tail_hits": int(case_index == 9),
                "group_tail_events": int(case_index in {2, 10, 18}),
                "group_tail_hits": int(case_index in {10, 18}),
                "hotspot_event": 1,
                "hotspot_hit": int(case_index % 2 == 0),
            }
        )
    result = api.aggregate_sequence_metrics(rows)
    assert result.raw_case_count == 320
    assert result.nearest_rms_distance == pytest.approx(159.5)
    assert result.total_tail_events == 2
    assert result.total_tail_hits == 1
    assert result.total_tail_recall == 0.5
    assert result.group_tail_events == 3
    assert result.group_tail_hits == 2
    assert result.group_tail_recall == pytest.approx(2 / 3)
    assert result.hotspot_events == 320
    assert result.hotspot_hits == 160
    with pytest.raises(ValueError, match="320|case universe"):
        api.aggregate_sequence_metrics(rows[:-1])
    equal_length_identity_corruption = [dict(row) for row in rows]
    equal_length_identity_corruption[-1] = dict(equal_length_identity_corruption[-2])
    assert len(equal_length_identity_corruption) == 320
    with pytest.raises(ValueError, match="duplicate|missing|identity|case universe"):
        api.aggregate_sequence_metrics(equal_length_identity_corruption)


def test_family_stratified_bootstrap_is_sequence_level_exact_and_deterministic() -> None:
    api = _api()
    sequence_rows = []
    for family_index, family in enumerate(FAMILIES):
        for seed_index, base_seed in enumerate(BASE_SEEDS):
            sequence_rows.append(
                {
                    "family": family,
                    "base_seed": base_seed,
                    "sequence_id": f"test-{family_index}-{seed_index}",
                    "paired_delta": 0.01 * (1 + family_index + seed_index),
                }
            )
    first = api.family_stratified_sequence_bootstrap(
        sequence_rows, replicates=10_000, seed=20260731
    )
    second = api.family_stratified_sequence_bootstrap(
        list(reversed(sequence_rows)), replicates=10_000, seed=20260731
    )
    assert _canonical(first) == _canonical(second)
    assert first.sequence_count == 15
    assert first.family_count == 5
    assert first.replicates == 10_000
    assert first.seed == 20260731
    assert np.isfinite(first.ci_lower) and np.isfinite(first.ci_upper)
    with pytest.raises(ValueError, match="3|15|sequence"):
        api.family_stratified_sequence_bootstrap(
            sequence_rows[:-1], replicates=10_000, seed=20260731
        )


def test_gate_pass_fail_hold_and_tail_insufficiency_masking() -> None:
    api = _api()
    passing = _passing_evidence()
    decision = api.evaluate_phase3b_gate(passing)
    assert decision.data_status == "PASS"
    assert decision.gate_status == "PENDING_SUPERVISOR"
    assert tuple(decision.conditions) == (1, 2, 3, 4, 5, 6)

    insufficient = dict(passing)
    insufficient["total_tail_events"] = 9
    assert api.evaluate_phase3b_gate(insufficient).data_status == "HOLD"

    failed_and_insufficient = dict(insufficient)
    failed_and_insufficient["selected_joint_coverage"] = 0.84
    failed = api.evaluate_phase3b_gate(failed_and_insufficient)
    assert failed.data_status == "FAIL"
    assert 1 in failed.failed_conditions
    assert 4 in failed.insufficient_conditions

    random_selected = dict(passing)
    random_selected["selected_method"] = "random_empirical"
    random_selected["paired_delta_ci95"] = [0.0, 0.0]
    random_decision = api.evaluate_phase3b_gate(random_selected)
    assert random_decision.data_status == "FAIL"
    assert 2 in random_decision.failed_conditions


@pytest.mark.parametrize(
    ("mutation", "condition"),
    (
        pytest.param({"selected_joint_coverage": 0.849999}, 1, id="overall-coverage"),
        pytest.param(
            {
                "selected_joint_coverage_by_family": {
                    **{family: 0.86 for family in FAMILIES},
                    FAMILIES[0]: 0.799999,
                }
            },
            1,
            id="family-coverage",
        ),
        pytest.param({"paired_delta_ci95": [0.0, 0.1]}, 2, id="ci-lower"),
        pytest.param(
            {
                "paired_delta_by_base_seed": {
                    str(BASE_SEEDS[0]): 0.0,
                    str(BASE_SEEDS[1]): 0.03,
                    str(BASE_SEEDS[2]): 0.03,
                }
            },
            2,
            id="base-seed-nonpositive",
        ),
        pytest.param(
            {
                "paired_delta_by_family": {
                    family: (0.03 if index < 3 else 0.0)
                    for index, family in enumerate(FAMILIES)
                }
            },
            2,
            id="fewer-than-four-families-positive",
        ),
        pytest.param({"lofo_aggregate_delta": -1e-12}, 3, id="lofo-aggregate"),
        pytest.param(
            {
                "lofo_delta_by_family": {
                    family: (0.01 if index < 2 else 0.0)
                    for index, family in enumerate(FAMILIES)
                }
            },
            3,
            id="lofo-fewer-than-three-positive",
        ),
        pytest.param(
            {
                "lofo_relative_degradation_by_family": {
                    family: (0.100001 if index < 2 else 0.0)
                    for index, family in enumerate(FAMILIES)
                }
            },
            3,
            id="lofo-two-families-over-ten-percent",
        ),
        pytest.param(
            {"total_tail_hits": 8, "total_tail_events": 12},
            4,
            id="total-tail-recall",
        ),
        pytest.param(
            {"group_tail_hits": 16, "group_tail_events": 24},
            4,
            id="group-tail-recall",
        ),
        pytest.param(
            {"hotspot_hits": 223, "hotspot_events": 320},
            4,
            id="hotspot-recall",
        ),
        pytest.param(
            {"selected_mean_physical_normalized_width": 0.750001},
            5,
            id="width",
        ),
        pytest.param(
            {
                "ordinary_invalid_or_empty_rate": {
                    **{method: 0.0 for method in ORDINARY_METHODS},
                    "boundary_scenarios": 1e-12,
                }
            },
            5,
            id="ordinary-invalid-rate",
        ),
        pytest.param({"ratio1_singleton_coverage": 0.999999}, 5, id="ratio1-coverage"),
        pytest.param({"all_timings_finite": False}, 5, id="nonfinite-timing"),
        pytest.param(
            {"integrity_checks_complete": False}, 6, id="integrity-incomplete"
        ),
        pytest.param(
            {"integrity_checks_passed": False}, 6, id="integrity-failed"
        ),
    ),
)
def test_each_data_gate_condition_fails_independently(
    mutation: Mapping[str, Any], condition: int,
) -> None:
    api = _api()
    evidence = _passing_evidence()
    evidence.update(mutation)
    decision = api.evaluate_phase3b_gate(evidence)
    assert decision.data_status == "FAIL"
    assert condition in decision.failed_conditions
    assert decision.gate_status == "PENDING_SUPERVISOR"


def test_raw_validator_rejects_missing_duplicate_nonfinite_and_oracle_primary_rows() -> None:
    api = _api()
    row = {
        "case_id": "case-0",
        "sequence_id": "test-0",
        "split": "test",
        "family": FAMILIES[0],
        "base_seed": BASE_SEEDS[0],
        "checkpoint": 32,
        "reveal_mode": REVEAL_MODES[0],
        "reveal_ratio": 0.0,
        "requested_k": 8,
        "construction_seed": 123,
        "method": "minimax_subset",
        "nearest_rms_distance": 0.5,
        "joint_coverage": 1,
        "uses_oracle": False,
        "upper_bound_only": False,
        "sequence_digest": _sha("sequence"),
        "observation_digest": _sha("observation"),
        "ambiguity_digest": _sha("ambiguity"),
        "support_digest": _sha("support"),
    }
    assert api.validate_raw_case_rows([row], allow_incomplete_universe=True)

    missing = dict(row)
    del missing["observation_digest"]
    with pytest.raises(ValueError, match="observation_digest|missing"):
        api.validate_raw_case_rows([missing], allow_incomplete_universe=True)
    with pytest.raises(ValueError, match="duplicate|case_id"):
        api.validate_raw_case_rows([row, dict(row)], allow_incomplete_universe=True)
    semantic_duplicate = dict(row, case_id="different-label-same-semantic-case")
    with pytest.raises(ValueError, match="duplicate|semantic|identity|case"):
        api.validate_raw_case_rows(
            [row, semantic_duplicate], allow_incomplete_universe=True
        )
    nonfinite = dict(row, nearest_rms_distance=float("nan"))
    with pytest.raises(ValueError, match="finite|NaN"):
        api.validate_raw_case_rows([nonfinite], allow_incomplete_universe=True)

    oracle = dict(
        row,
        case_id="case-oracle",
        method="oracle_support_upper_bound",
        uses_oracle=True,
        upper_bound_only=True,
    )
    with pytest.raises(ValueError, match="oracle|primary|validation"):
        api.select_validation_method([oracle])


def test_recompute_detects_raw_corruption_and_is_byte_deterministic() -> None:
    api = _api()
    manifest = {
        "protocol_sha256": "FD3A6A9701956BCF137B0B869994F3153460FD8B5AC45CD58AAA14E2D51E9467",
        "selected_method": "minimax_subset",
        "selected_k": 8,
        "sequence_count": 15,
    }
    raw_case_rows = [
        {
            "case_id": "case-0",
            "sequence_id": "test-0",
            "method": "minimax_subset",
            "requested_k": 8,
            "nearest_rms_distance": 0.4,
            "joint_coverage": 1,
            "uses_oracle": False,
            "upper_bound_only": False,
        }
    ]
    raw_sequence_rows = [
        {
            "sequence_id": "test-0",
            "method": "minimax_subset",
            "requested_k": 8,
            "nearest_rms_distance": 0.4,
            "joint_coverage": 1.0,
            "raw_case_count": 1,
        }
    ]
    first = api.recompute_artifacts(
        manifest, raw_case_rows, raw_sequence_rows, allow_incomplete_universe=True
    )
    second = api.recompute_artifacts(
        json.loads(json.dumps(manifest)),
        list(reversed(raw_case_rows)),
        list(reversed(raw_sequence_rows)),
        allow_incomplete_universe=True,
    )
    assert _canonical(first) == _canonical(second)
    assert first.summary["gate_status"] == "PENDING_SUPERVISOR"

    corrupted = [dict(raw_case_rows[0], nearest_rms_distance=0.6)]
    with pytest.raises(ValueError, match="recompute|mismatch|corrupt"):
        api.recompute_artifacts(
            manifest,
            corrupted,
            raw_sequence_rows,
            expected_summary=first.summary,
            allow_incomplete_universe=True,
        )


def test_summary_is_pending_supervisor_and_has_no_phase4_fields_or_claims() -> None:
    api = _api()
    summary = api.build_summary(_passing_evidence(), selected_method="minimax_subset")
    assert summary["gate_status"] == "PENDING_SUPERVISOR"
    assert summary["data_status"] == "PASS"

    forbidden_fragments = {
        "completion",
        "oracle_regret",
        "legality",
        "wasted_prefix",
        "proposal",
        "transfer_action",
        "truth_token_id",
        "commit",
        "prefix",
        "horizon",
        "robust_score",
        "repair",
        "recourse",
    }
    encoded = json.dumps(summary, sort_keys=True).lower()
    assert all(fragment not in encoded for fragment in forbidden_fragments)
    forbidden_claims = (
        "scheduling gain",
        "scheduling benefit",
        "aiccl gain",
        "proves planning",
        "调度收益",
        "提前规划收益",
    )
    assert all(claim not in encoded for claim in forbidden_claims)
    assert "supervisor_no_veto" not in encoded


def test_toy_artifacts_have_exact_names_validate_and_are_deterministic(tmp_path: Path) -> None:
    api = _api()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = api.run_toy_experiment(first_dir)
    second = api.run_toy_experiment(second_dir)
    assert set(path.name for path in first_dir.iterdir()) == set(ARTIFACT_NAMES)
    assert set(path.name for path in second_dir.iterdir()) == set(ARTIFACT_NAMES)
    for name in ARTIFACT_NAMES:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    assert first["summary"]["gate_status"] == "PENDING_SUPERVISOR"
    assert _canonical(first) == _canonical(second)

    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((first_dir / "summary.json").read_text(encoding="utf-8"))
    with (first_dir / "raw_case_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        raw_cases = list(csv.DictReader(handle))
    with (first_dir / "raw_sequence_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        raw_sequences = list(csv.DictReader(handle))
    checked = api.recompute_artifacts(
        manifest,
        raw_cases,
        raw_sequences,
        expected_summary=summary,
        allow_incomplete_universe=True,
    )
    assert checked.summary == summary


def test_runner_import_is_side_effect_free_and_cli_is_frozen_safe() -> None:
    runner_path = Path(__file__).resolve().parents[1] / "scripts" / "run_phase3b_ambiguity.py"
    before = {
        path: path.stat().st_mtime_ns
        for path in (Path(__file__).resolve().parents[1] / FORMAL_OUTPUT).glob("*")
        if path.is_file()
    }
    runner = _runner()
    after = {
        path: path.stat().st_mtime_ns
        for path in (Path(__file__).resolve().parents[1] / FORMAL_OUTPUT).glob("*")
        if path.is_file()
    }
    assert before == after
    parser = runner.build_parser()
    args = parser.parse_args([])
    assert Path(args.output_dir).as_posix().endswith("outputs/phase3b_ambiguity")
    assert hasattr(args, "toy") and args.toy is False
    assert hasattr(args, "formal") and args.formal is False
    option_strings = {
        option
        for action in parser._actions
        for option in getattr(action, "option_strings", ())
    }
    forbidden_options = {
        "--family", "--families", "--base-seed", "--base-seeds",
        "--sequence-length", "--length", "--max-entry", "--checkpoint",
        "--reveal-ratio", "--selector", "--k",
    }
    assert option_strings.isdisjoint(forbidden_options)
    with pytest.raises(SystemExit):
        runner.main([])

    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
    ]
    assert main_guards
    assert hasattr(runner, "run_formal_experiment")


def test_experiment_and_runner_have_no_forbidden_imports_or_phase4_api() -> None:
    api = _api()
    runner = _runner()
    for module in (api, runner):
        source_path = Path(inspect.getsourcefile(module) or "")
        imported = _resolved_imports(
            source_path.read_text(encoding="utf-8"), module.__name__
        )
        assert not any(
            name == forbidden or name.startswith(forbidden + ".")
            for name in imported for forbidden in FORBIDDEN_IMPORTS
        )
        assert not _forbidden_public_names(getattr(module, "__all__", ()))

    assert "rlccl.uncertainty.execution" in _resolved_imports(
        "from . import execution", "rlccl.uncertainty.ambiguity_experiment"
    )
    assert "rlccl.uncertainty.execution" in _resolved_imports(
        "from .execution import hidden", "rlccl.uncertainty.ambiguity_experiment"
    )
    assert _forbidden_public_names(
        ("build_prefix", "ROBUST_SCORE_report", "Truth_Token_IdFactory")
    ) == {"build_prefix", "ROBUST_SCORE_report", "Truth_Token_IdFactory"}

    script = """
import json, sys
import rlccl.uncertainty
before = set(sys.modules)
import rlccl.uncertainty.ambiguity_experiment
new = set(sys.modules) - before
forbidden = ('torch', 'rlccl.uncertainty.execution', 'rlccl.envs.decoder', 'rlccl.scheduling')
print(json.dumps(sorted(name for name in new if name in forbidden or name.startswith(tuple(x + '.' for x in forbidden)))))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_formal_output_is_not_created_by_collection_or_import() -> None:
    root = Path(__file__).resolve().parents[1]
    formal_dir = root / FORMAL_OUTPUT
    before = {
        name: (formal_dir / name).read_bytes()
        for name in ARTIFACT_NAMES
        if (formal_dir / name).is_file()
    }
    _api()
    _runner()
    after = {
        name: (formal_dir / name).read_bytes()
        for name in ARTIFACT_NAMES
        if (formal_dir / name).is_file()
    }
    assert after == before


# Schema-v2 references are deliberately test-owned.  They do not call any
# production canonicalization helper to construct expected bytes or digests.
SCHEMA_V2_PROTOCOL_SHA256 = "7E01108E362973461B5E676CF163A491D7E90E5D30D40AE0356CA83D6680D7A3"
SCHEMA_V2_ARTIFACTS = (
    "manifest.json", "raw_calibration_scores.csv",
    "raw_validation_metrics.csv", "raw_case_metrics.csv",
    "raw_sequence_metrics.csv", "raw_lofo_calibration_scores.csv",
    "raw_lofo_validation_metrics.csv", "raw_lofo_test_metrics.csv",
    "raw_dependence_metrics.csv", "summary.json",
)
SCHEMA_V2_SOURCE_KEYS = {
    "rlccl/uncertainty/ambiguity.py",
    "rlccl/uncertainty/ambiguity_experiment.py",
    "scripts/run_phase3b_ambiguity.py",
}
SCHEMA_V2_ENV_KEYS = {"python", "python_executable", "numpy", "platform"}
SCHEMA_V2_TABLE_SCHEMAS = {
    "raw_calibration_scores": (
        ("case_id","s"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),("record_index","i"),
        ("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("construction_seed","i"),("score","f"),("sequence_digest","s"),
        ("generator_config_digest","s"),("topology_digest","s"),("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),
    ),
    "raw_validation_metrics": (
        ("case_id","s"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),("record_index","i"),
        ("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("requested_k","i"),("construction_seed","i"),("method","s"),("replicate_count","i"),
        ("nearest_rms_distance","f"),("uses_oracle","b"),("upper_bound_only","b"),("sequence_digest","s"),
        ("generator_config_digest","s"),("topology_digest","s"),("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),("support_digest","s"),
    ),
    "raw_case_metrics": (
        ("case_id","s"),("case_index","i"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),
        ("record_index","i"),("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("requested_k","i"),("construction_seed","i"),("method","s"),("replicate_count","i"),
        ("nearest_rms_distance","f"),("nearest_matrix_l1_distance","f"),("covering_radius","f"),("mean_pairwise_diversity","f"),
        ("duplicate_fraction","f"),("actual_k","i"),("component_coverage","f"),("joint_coverage","f"),
        ("physical_normalized_mean_width","f"),("zero_physical_range_components","i"),("total_tail_events","f"),("total_tail_hits","f"),
        ("group_tail_events","f"),("group_tail_hits","f"),("hotspot_events","f"),("hotspot_hits","f"),("invalid_or_empty","f"),
        ("construction_seconds","f"),("selector_seconds","f"),("uses_oracle","b"),("upper_bound_only","b"),("sequence_digest","s"),
        ("generator_config_digest","s"),("topology_digest","s"),("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),("support_digest","s"),
    ),
    "raw_sequence_metrics": (
        ("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),("record_index","i"),("method","s"),("requested_k","i"),
        ("raw_case_count","i"),("nearest_rms_distance","f"),("total_tail_events","f"),("total_tail_hits","f"),("total_tail_recall","f?"),
        ("group_tail_events","f"),("group_tail_hits","f"),("group_tail_recall","f?"),("hotspot_events","f"),("hotspot_hits","f"),("hotspot_recall","f?"),
        ("sequence_digest","s"),("generator_config_digest","s"),("topology_digest","s"),("normalizer_digest","s"),
    ),
    "raw_lofo_calibration_scores": (
        ("case_id","s"),("fold_id","s"),("held_out_family","s"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),
        ("record_index","i"),("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("construction_seed","i"),("score","f"),("sequence_digest","s"),("generator_config_digest","s"),
        ("topology_digest","s"),("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),
    ),
    "raw_lofo_validation_metrics": (
        ("case_id","s"),("fold_id","s"),("held_out_family","s"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),
        ("record_index","i"),("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("requested_k","i"),("construction_seed","i"),("method","s"),("replicate_count","i"),("nearest_rms_distance","f"),
        ("uses_oracle","b"),("upper_bound_only","b"),("sequence_digest","s"),("generator_config_digest","s"),("topology_digest","s"),
        ("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),("support_digest","s"),
    ),
    "raw_lofo_test_metrics": (
        ("case_id","s"),("fold_id","s"),("held_out_family","s"),("role","s"),("sequence_id","s"),("split","s"),("family","s"),
        ("base_seed","i"),("record_index","i"),("checkpoint","i"),("mode_index","i"),("reveal_mode","s"),("stage_index","i"),("reveal_ratio","f"),
        ("actual_entry_fraction","f"),("requested_k","i"),("construction_seed","i"),("method","s"),("replicate_count","i"),("nearest_rms_distance","f"),
        ("uses_oracle","b"),("upper_bound_only","b"),("sequence_digest","s"),("generator_config_digest","s"),("topology_digest","s"),
        ("normalizer_digest","s"),("observation_digest","s"),("ambiguity_digest","s"),("support_digest","s"),
    ),
    "raw_dependence_metrics": (
        ("case_id","s"),("sequence_id","s"),("split","s"),("family","s"),("base_seed","i"),("record_index","i"),
        ("checkpoint","i"),("total_traffic","f"),("sequence_digest","s"),("generator_config_digest","s"),("topology_digest","s"),
    ),
}
SCHEMA_V2_IDENTITIES = {
    "raw_calibration_scores": ("record_index","checkpoint","mode_index","stage_index"),
    "raw_validation_metrics": ("record_index","checkpoint","mode_index","stage_index","method"),
    "raw_case_metrics": ("case_index","method"),
    "raw_sequence_metrics": ("record_index","method","requested_k"),
    "raw_lofo_calibration_scores": ("fold_id","record_index","checkpoint","mode_index","stage_index"),
    "raw_lofo_validation_metrics": ("fold_id","record_index","checkpoint","mode_index","stage_index","method"),
    "raw_lofo_test_metrics": ("fold_id","record_index","checkpoint","mode_index","stage_index","role"),
    "raw_dependence_metrics": ("record_index","checkpoint"),
}
SCHEMA_V2_MANIFEST_KEYS = {
    "schema_version","protocol_sha256","artifact_names","artifact_logical_sha256","artifact_scientific_sha256",
    "combined_scientific_evidence_sha256","authorized_source_sha256","families","base_seeds","splits","sequence_length",
    "max_entry","history_window","checkpoints","reveal_modes","reveal_ratios","requested_k","random_replicates","sequence_specs",
    "sequence_records","h1_exclusion_manifest_sha256","h1_excluded_sequence_digests","topology","normalizer_digest",
    "group_coefficients_digest","lofo_fold_normalizer_digests","calibration_radius","selected_method","selected_k",
    "validation_method_means","lofo_fold_evidence","test_total_traffic_dependence","gate_evidence","data_status","summary_sha256","environment",
}
SCHEMA_V2_SUMMARY_KEYS = {
    "schema_version","protocol_sha256","selected_method","selected_k","calibration_radius","validation_method_means","gate_evidence",
    "test_total_traffic_dependence","raw_row_counts","data_status","gate_status","conditions_evaluated","failed_conditions",
    "insufficient_conditions","combined_scientific_evidence_sha256",
}


def _v2_scalar(value: Any) -> Any:
    if value is None:
        return ["n", None]
    if type(value) is bool:
        return ["b", "true" if value else "false"]
    if type(value) is int:
        return ["i", str(value)]
    if type(value) is float:
        number = 0.0 if value == 0.0 else value
        assert np.isfinite(number)
        return ["f", number.hex().lower()]
    if type(value) is str:
        return ["s", value]
    if type(value) is list:
        return ["l", [_v2_scalar(item) for item in value]]
    if isinstance(value, Mapping):
        assert all(type(key) is str for key in value)
        return ["m", [[key, _v2_scalar(value[key])] for key in sorted(value)]]
    raise TypeError(type(value))


def _v2_digest(value: Any) -> str:
    payload = json.dumps(
        _v2_scalar(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _v2_validate_typed(value: Any, kind: str) -> None:
    if kind == "s" and type(value) is str:
        return
    if kind == "b" and type(value) is bool:
        return
    if kind == "i" and type(value) is int:
        return
    if kind == "f" and type(value) is float and np.isfinite(value):
        return
    if kind == "f?" and (value is None or (type(value) is float and np.isfinite(value))):
        return
    raise ValueError(f"invalid strict {kind} value")


def _v2_table_digest(
    table_name: str,
    header: tuple[str, ...],
    rows: Iterable[Mapping[str, Any]],
    *,
    scientific: bool = False,
) -> str:
    schema = SCHEMA_V2_TABLE_SCHEMAS[table_name]
    expected_header = tuple(name for name, _ in schema)
    if header != expected_header:
        raise ValueError("unknown, missing, or reordered columns")
    identities = SCHEMA_V2_IDENTITIES[table_name]
    typed_rows = []
    seen = set()
    for source in rows:
        if set(source) != set(expected_header):
            raise ValueError("row schema mismatch")
        for name, kind in schema:
            _v2_validate_typed(source[name], kind)
        identity = tuple(source[name] for name in identities)
        if identity in seen:
            raise ValueError("duplicate identity")
        seen.add(identity)
        typed_rows.append(dict(source))
    typed_rows.sort(key=lambda row: tuple(row[name] for name in identities))
    columns = list(expected_header)
    if scientific and table_name == "raw_case_metrics":
        columns = [
            name for name in columns
            if name not in {"construction_seconds", "selector_seconds"}
        ]
    payload = [
        "table-v1", table_name, columns,
        [[_v2_scalar(row[name]) for name in columns] for row in typed_rows],
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _v2_read_csv(path: Path, table_name: str) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    schema = SCHEMA_V2_TABLE_SCHEMAS[table_name]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        raw_rows = list(reader)
    expected = tuple(name for name, _ in schema)
    if header != expected:
        raise ValueError("CSV header mismatch")
    rows = []
    for raw in raw_rows:
        row: dict[str, Any] = {}
        for name, kind in schema:
            value = raw[name]
            if kind == "s":
                row[name] = value
            elif kind == "b":
                if value not in {"true", "false"}:
                    raise ValueError("strict bool lexical form")
                row[name] = value == "true"
            elif kind == "i":
                if value != "0" and (not value or value.startswith(("+", "0"))):
                    raise ValueError("strict integer lexical form")
                row[name] = int(value)
            elif kind == "f?" and value == "":
                row[name] = None
            else:
                row[name] = float(value)
                if not np.isfinite(row[name]):
                    raise ValueError("finite float required")
        rows.append(row)
    return header, rows


def _v2_write_csv(path: Path, header: tuple[str, ...], rows: Sequence[Mapping[str, Any]]) -> None:
    def lexical(value: Any) -> str:
        if value is None:
            return ""
        if type(value) is bool:
            return "true" if value else "false"
        return str(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name:lexical(row[name]) for name in header})


def _v2_combined_scientific_digest(digests: Mapping[str, str]) -> str:
    ordered = [[name, digests[name]] for name in SCHEMA_V2_ARTIFACTS[1:-1]]
    payload = ["phase3b-scientific-v1", ordered]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _v2_minimal_row(table_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, kind in SCHEMA_V2_TABLE_SCHEMAS[table_name]:
        if kind == "s":
            row[name] = "x"
        elif kind == "i":
            row[name] = 0
        elif kind == "f":
            row[name] = 0.0
        elif kind == "b":
            row[name] = False
        else:
            row[name] = None
    defaults = {
        "case_id":"case-0", "sequence_id":"sequence-0", "split":"test",
        "family":FAMILIES[0], "reveal_mode":REVEAL_MODES[0], "method":ORDINARY_METHODS[0],
        "fold_id":f"lofo-0-{FAMILIES[0]}", "held_out_family":FAMILIES[0], "role":"selected",
    }
    for name, value in defaults.items():
        if name in row:
            row[name] = value
    for name in row:
        if name.endswith("_digest"):
            row[name] = _sha(name)
    return row


def _v2_valid_raw_case_reference(*, case_index: int, requested_k: int) -> dict[str, Any]:
    row = _v2_minimal_row("raw_case_metrics")
    family = FAMILIES[0]
    sequence_id = f"{family}-base342-sequence4-seed40342"
    row.update({
        "case_id":f"case-{case_index}-minimax_subset", "case_index":case_index,
        "sequence_id":sequence_id, "split":"test", "family":family,
        "base_seed":342, "record_index":4, "checkpoint":32,
        "mode_index":0, "reveal_mode":REVEAL_MODES[0], "stage_index":0,
        "reveal_ratio":0.0, "actual_entry_fraction":0.0,
        "requested_k":requested_k, "construction_seed":31_400_320,
        "method":"minimax_subset", "replicate_count":1,
        "nearest_rms_distance":0.4, "nearest_matrix_l1_distance":1.0,
        "covering_radius":0.5, "mean_pairwise_diversity":0.2,
        "duplicate_fraction":0.0, "actual_k":requested_k,
        "component_coverage":1.0, "joint_coverage":1.0,
        "physical_normalized_mean_width":0.5, "zero_physical_range_components":0,
        "total_tail_events":0.0, "total_tail_hits":0.0,
        "group_tail_events":0.0, "group_tail_hits":0.0,
        "hotspot_events":1.0, "hotspot_hits":1.0, "invalid_or_empty":0.0,
        "construction_seconds":0.001, "selector_seconds":0.001,
        "uses_oracle":False, "upper_bound_only":False,
    })
    return row


def test_schema_v2_protocol_ten_artifacts_and_exact_source_environment_sets() -> None:
    api = _api()
    protocol = Path(__file__).resolve().parents[1] / "docs/uncertainty_aiccl/PHASE3B_AMBIGUITY_PROTOCOL.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest().upper() == SCHEMA_V2_PROTOCOL_SHA256
    assert api.PROTOCOL_SHA256 == SCHEMA_V2_PROTOCOL_SHA256
    assert tuple(api.ARTIFACT_NAMES) == SCHEMA_V2_ARTIFACTS
    assert dict(getattr(api, "RAW_TABLE_SCHEMAS")) == SCHEMA_V2_TABLE_SCHEMAS
    assert dict(getattr(api, "RAW_TABLE_IDENTITIES")) == SCHEMA_V2_IDENTITIES
    manifest_keys = set(getattr(api, "SCHEMA_V2_MANIFEST_KEYS"))
    summary_keys = set(getattr(api, "SCHEMA_V2_SUMMARY_KEYS"))
    assert manifest_keys == SCHEMA_V2_MANIFEST_KEYS
    assert summary_keys == SCHEMA_V2_SUMMARY_KEYS
    assert set(getattr(api, "AUTHORIZED_SOURCE_KEYS")) == SCHEMA_V2_SOURCE_KEYS
    assert set(getattr(api, "ENVIRONMENT_KEYS")) == SCHEMA_V2_ENV_KEYS


def test_schema_v2_reference_scalar_container_object_and_exclusions() -> None:
    assert _v2_scalar(-0.0) == ["f", "0x0.0p+0"]
    assert _v2_scalar(True) == ["b", "true"]
    assert _v2_scalar({"β": [None, 7]}) == [
        "m", [["β", ["l", [["n", None], ["i", "7"]]]]]
    ]
    base = {"case_id": "x", "construction_seconds": 1.0, "selector_seconds": 2.0}
    changed = {**base, "construction_seconds": 9.0, "selector_seconds": 8.0}
    assert _v2_digest(base) != _v2_digest(changed)
    scientific_base = {k: v for k, v in base.items() if not k.endswith("_seconds")}
    scientific_changed = {k: v for k, v in changed.items() if not k.endswith("_seconds")}
    assert _v2_digest(scientific_base) == _v2_digest(scientific_changed)
    api = _api()
    assert api.canonical_object_sha256({"β": [None, 7]}) == _v2_digest({"β": [None, 7]})


def test_schema_v2_independent_table_reference_sort_types_columns_and_combined_order() -> None:
    table = "raw_case_metrics"
    header = tuple(name for name, _ in SCHEMA_V2_TABLE_SCHEMAS[table])
    first = _v2_valid_raw_case_reference(case_index=24_003, requested_k=16)
    second = _v2_valid_raw_case_reference(case_index=24_002, requested_k=8)
    logical = _v2_table_digest(table, header, [first, second])
    assert logical == _v2_table_digest(table, header, [second, first])
    with pytest.raises(ValueError, match="column"):
        _v2_table_digest(table, tuple(reversed(header)), [first])
    with pytest.raises(ValueError, match="schema"):
        _v2_table_digest(table, header, [{**first, "unknown": 1}])
    with pytest.raises(ValueError, match="strict i"):
        _v2_table_digest(table, header, [{**first, "case_index": True}])
    changed_timing = dict(first, construction_seconds=9.0, selector_seconds=8.0)
    assert _v2_table_digest(table, header, [first]) != _v2_table_digest(table, header, [changed_timing])
    assert _v2_table_digest(table, header, [first], scientific=True) == _v2_table_digest(table, header, [changed_timing], scientific=True)
    ordered = [[name, _sha(name)] for name in SCHEMA_V2_ARTIFACTS[1:-1]]
    forward = hashlib.sha256(json.dumps(["phase3b-scientific-v1", ordered], separators=(",", ":")).encode()).hexdigest()
    reverse = hashlib.sha256(json.dumps(["phase3b-scientific-v1", list(reversed(ordered))], separators=(",", ":")).encode()).hexdigest()
    assert forward != reverse
    api = _api()
    assert api.canonical_table_sha256(table, header, [second, first], scientific=False) == logical


def test_schema_v2_lofo_dependence_gate_summary_and_atomic_readback_contract() -> None:
    api = _api()
    required = {
        "validate_all_raw_tables", "validate_test_raw_aggregates",
        "validate_schema_v2_table", "canonical_table_sha256",
        "combined_scientific_evidence_sha256",
        "materialize_provisional_toy_artifacts", "read_back_artifacts",
        "finalize_staged_artifacts", "publish_artifacts_atomically",
    }
    assert required.issubset(vars(api))
    assert api.SCHEMA_VERSION == 2
    assert api.RAW_ROW_COUNTS == {
        "raw_calibration_scores.csv": 4800,
        "raw_validation_metrics.csv": 19200,
        "raw_case_metrics.csv": 120000,
        "raw_sequence_metrics.csv": 300,
        "raw_lofo_calibration_scores.csv": 19200,
        "raw_lofo_validation_metrics.csv": 76800,
        "raw_lofo_test_metrics.csv": 9600,
        "raw_dependence_metrics.csv": 240,
    }


@pytest.fixture(scope="session")
def schema_v2_full_120k_fixture() -> dict[str, Any]:
    """Exact 120k/300 coordinate universe using the pre-v2 callable surface."""
    api = _api()
    specs = tuple(api.build_formal_sequence_specs())
    test_specs = {spec.record_index: spec for spec in specs if spec.split == "test"}
    registry = [row for row in api.build_case_registry(specs) if row.split == "test"]
    topology_digest = _sha("topology")
    normalizer_digest = _sha("normalizer")
    seq_digest = {index: _sha(f"sequence-{index}") for index in test_specs}
    config_digest = {index: _sha(f"config-{index}") for index in test_specs}
    raw: list[dict[str, Any]] = []
    for entry in registry:
        spec = test_specs[entry.record_index]
        coordinate = f"{entry.record_index}-{entry.checkpoint}-{entry.mode_index}-{entry.stage_index}"
        observation_digest = _sha("observation-" + coordinate)
        ambiguity_digest = _sha("ambiguity-" + coordinate)
        for method in ALL_METHODS:
            oracle = method == "oracle_support_upper_bound"
            ratio1 = entry.reveal_ratio == 1.0
            raw.append({
                "case_id": f"case-{entry.case_index}-{method}", "case_index": entry.case_index,
                "sequence_id": spec.sequence_id, "split": "test", "family": spec.family,
                "base_seed": spec.base_seed, "record_index": spec.record_index,
                "checkpoint": entry.checkpoint, "mode_index": entry.mode_index,
                "reveal_mode": entry.reveal_mode, "stage_index": entry.stage_index,
                "reveal_ratio": entry.reveal_ratio, "actual_entry_fraction": entry.reveal_ratio,
                "requested_k": entry.requested_k, "construction_seed": entry.construction_seed,
                "method": method, "replicate_count": 1 if method != "random_empirical" or ratio1 else 8,
                "nearest_rms_distance": 0.4, "nearest_matrix_l1_distance": 1.0,
                "covering_radius": 0.5, "mean_pairwise_diversity": 0.2,
                "duplicate_fraction": 0.0, "actual_k": 1 if ratio1 else entry.requested_k,
                "component_coverage": 1.0, "joint_coverage": 1.0,
                "physical_normalized_mean_width": 0.5, "zero_physical_range_components": 0,
                "total_tail_events": 1.0, "total_tail_hits": 1.0,
                "group_tail_events": 1.0, "group_tail_hits": 1.0,
                "hotspot_events": 1.0, "hotspot_hits": 1.0, "invalid_or_empty": 0.0,
                "construction_seconds": 0.001, "selector_seconds": 0.001,
                "uses_oracle": oracle, "upper_bound_only": oracle,
                "sequence_digest": seq_digest[spec.record_index],
                "generator_config_digest": config_digest[spec.record_index],
                "topology_digest": topology_digest, "normalizer_digest": normalizer_digest,
                "observation_digest": observation_digest, "ambiguity_digest": ambiguity_digest,
                "support_digest": _sha(f"support-{entry.case_index}-{method}"),
            })
    sequence_rows = []
    for spec in test_specs.values():
        for method in ALL_METHODS:
            for requested_k in REQUESTED_K:
                sequence_rows.append({
                    "sequence_id": spec.sequence_id, "split": "test", "family": spec.family,
                    "base_seed": spec.base_seed, "record_index": spec.record_index,
                    "method": method, "requested_k": requested_k, "raw_case_count": 320,
                    "nearest_rms_distance": 0.4, "total_tail_events": 320.0,
                    "total_tail_hits": 320.0, "total_tail_recall": 1.0,
                    "group_tail_events": 320.0, "group_tail_hits": 320.0,
                    "group_tail_recall": 1.0, "hotspot_events": 320.0,
                    "hotspot_hits": 320.0, "hotspot_recall": 1.0,
                    "sequence_digest": seq_digest[spec.record_index],
                    "generator_config_digest": config_digest[spec.record_index],
                    "topology_digest": topology_digest, "normalizer_digest": normalizer_digest,
                })
    manifest = {
        "schema_version": 2, "protocol_sha256": SCHEMA_V2_PROTOCOL_SHA256,
        "selected_method": "minimax_subset", "selected_k": 8, "data_status": "PASS",
    }
    expected_provenance = {
        "protocol_sha256": SCHEMA_V2_PROTOCOL_SHA256,
        "topology_digest": topology_digest,
        "normalizer_digest": normalizer_digest,
        "sequence_records": {
            spec.sequence_id: {
                "split": "test", "family": spec.family,
                "base_seed": spec.base_seed, "record_index": spec.record_index,
                "sequence_digest": seq_digest[spec.record_index],
                "generator_config_digest": config_digest[spec.record_index],
            }
            for spec in test_specs.values()
        },
    }
    expected_derived = {
        "selected_method":"minimax_subset", "selected_k":8,
        "selected_joint_coverage":1.0, "selected_component_coverage":1.0,
        "selected_mean_physical_normalized_width":0.5,
        "selected_total_tail_events":4_800.0, "selected_total_tail_hits":4_800.0,
        "selected_group_tail_events":4_800.0, "selected_group_tail_hits":4_800.0,
        "selected_hotspot_events":4_800.0, "selected_hotspot_hits":4_800.0,
        "ordinary_invalid_or_empty_rate":{method:0.0 for method in ORDINARY_METHODS},
        "ratio1_singleton_coverage":1.0, "ratio1_actual_k":1,
        "all_timings_finite_nonnegative":True,
        "paired_sequence_delta":{spec.sequence_id:0.0 for spec in test_specs.values()},
    }
    legacy_baseline = None
    if not hasattr(api, "validate_test_raw_aggregates"):
        legacy_baseline = api.recompute_artifacts(
            manifest, raw, sequence_rows, allow_incomplete_universe=False
        )
    assert len(raw) == 120_000 and len(sequence_rows) == 300
    return {
        "manifest": manifest, "raw": raw, "sequences": sequence_rows,
        "legacy_baseline": legacy_baseline,
        "expected_provenance": expected_provenance,
        "expected_derived": expected_derived,
    }


def _validate_full_test_tables(api: Any, fixture: Mapping[str, Any]) -> Any:
    if hasattr(api, "validate_test_raw_aggregates"):
        return api.validate_test_raw_aggregates(
            fixture["raw"], fixture["sequences"],
            expected_provenance=fixture["expected_provenance"],
            expected_derived=fixture["expected_derived"], exact=True,
        )
    return api.recompute_artifacts(
        fixture["manifest"], fixture["raw"], fixture["sequences"],
        expected_summary=fixture["legacy_baseline"].summary,
        allow_incomplete_universe=False,
    )


def test_schema_v2_full_120k_300_subvalidator_accepts_baseline(
    schema_v2_full_120k_fixture: Mapping[str, Any],
) -> None:
    assert _validate_full_test_tables(_api(), schema_v2_full_120k_fixture) is not None


@pytest.fixture
def schema_v2_gate_raw_fixture() -> dict[str, Any]:
    selected = _v2_valid_raw_case_reference(case_index=24_002, requested_k=8)
    selected.update(
        method="minimax_subset", case_id="case-24002-minimax_subset",
        nearest_rms_distance=0.3, support_digest=_sha("gate-selected-support"),
    )
    random_row = dict(
        selected, method="random_empirical", case_id="case-24002-random_empirical",
        replicate_count=8, nearest_rms_distance=0.5,
        support_digest=_sha("gate-random-support"),
    )
    ratio1 = _v2_valid_raw_case_reference(case_index=24_018, requested_k=8)
    ratio1.update(
        case_id="case-24018-minimax_subset", stage_index=4, reveal_ratio=1.0,
        actual_entry_fraction=1.0, actual_k=1, nearest_rms_distance=0.0,
        observation_digest=_sha("gate-ratio1-observation"),
        ambiguity_digest=_sha("gate-ratio1-ambiguity"),
        support_digest=_sha("gate-ratio1-support"),
    )
    for row in (selected, random_row, ratio1):
        row.update(
            component_coverage=1.0, joint_coverage=1.0,
            physical_normalized_mean_width=0.5,
            total_tail_events=1.0, total_tail_hits=1.0,
            group_tail_events=1.0, group_tail_hits=1.0,
            hotspot_events=1.0, hotspot_hits=1.0,
            invalid_or_empty=0.0, construction_seconds=0.01, selector_seconds=0.02,
        )
    expected_provenance = {
        "topology_digest":selected["topology_digest"],
        "normalizer_digest":selected["normalizer_digest"],
        "sequence_records": {
            selected["sequence_id"]: {
                "split":"test", "family":selected["family"],
                "base_seed":selected["base_seed"], "record_index":selected["record_index"],
                "sequence_digest":selected["sequence_digest"],
                "generator_config_digest":selected["generator_config_digest"],
            }
        },
        "support_digests": {
            row["case_id"]:row["support_digest"] for row in (selected, random_row, ratio1)
        },
        "support_canonical_bytes": {
            selected["case_id"]:b"gate-selected-support",
            random_row["case_id"]:b"gate-random-support",
            ratio1["case_id"]:b"gate-ratio1-support",
        },
    }
    expected_derived = {
        "selected_method":"minimax_subset", "selected_k":8,
        "selected_nearest_rms_distance":0.3,
        "random_nearest_rms_distance":0.5, "paired_delta":0.2,
        "bootstrap_mean_delta":0.2, "component_coverage":1.0,
        "joint_coverage":1.0, "physical_normalized_mean_width":0.5,
        "total_tail_events":1.0, "total_tail_hits":1.0,
        "group_tail_events":1.0, "group_tail_hits":1.0,
        "hotspot_events":1.0, "hotspot_hits":1.0,
        "invalid_or_empty_rate":0.0, "ratio1_joint_coverage":1.0,
        "ratio1_actual_k":1, "all_timings_finite_nonnegative":True,
    }
    return {
        "rows":[selected, random_row, ratio1],
        "expected_provenance":expected_provenance,
        "expected_derived":expected_derived,
    }


def _validate_gate_raw(api: Any, fixture: Mapping[str, Any]) -> Any:
    if hasattr(api, "validate_gate_raw_evidence"):
        return api.validate_gate_raw_evidence(
            fixture["rows"], expected_provenance=fixture["expected_provenance"],
            expected_derived=fixture["expected_derived"], allow_incomplete_universe=True,
        )
    return api.validate_raw_case_rows(
        fixture["rows"], allow_incomplete_universe=True
    )


def test_schema_v2_gate_raw_validator_accepts_complete_baseline(
    schema_v2_gate_raw_fixture: Mapping[str, Any],
) -> None:
    assert _validate_gate_raw(_api(), schema_v2_gate_raw_fixture) is not None


@pytest.mark.parametrize(
    "row_index,field,value",
    (
        (0,"nearest_rms_distance",0.31), (1,"nearest_rms_distance",0.51),
        (0,"component_coverage",0.9), (0,"joint_coverage",0.0),
        (0,"physical_normalized_mean_width",0.6),
        (0,"total_tail_events",2.0), (0,"total_tail_hits",0.0),
        (0,"group_tail_events",2.0), (0,"group_tail_hits",0.0),
        (0,"hotspot_events",2.0), (0,"hotspot_hits",0.0),
        (0,"invalid_or_empty",1.0), (2,"joint_coverage",0.0),
        (2,"actual_k",2), (0,"construction_seconds",-0.01),
        (0,"selector_seconds",float("inf")),
    ),
)
def test_schema_v2_gate_raw_each_field_tamper_is_rejected_after_baseline(
    schema_v2_gate_raw_fixture: Mapping[str, Any],
    row_index: int, field: str, value: Any,
) -> None:
    api = _api()
    _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    row = schema_v2_gate_raw_fixture["rows"][row_index]
    original = row[field]
    row[field] = value
    try:
        with pytest.raises(ValueError, match="nearest|paired|bootstrap|coverage|width|tail|group|hotspot|invalid|ratio1|actual|timing|finite|domain"):
            _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    finally:
        row[field] = original


def test_schema_v2_full_120k_300_valid_domain_raw_corruption_fails_closed(
    schema_v2_full_120k_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    raw = schema_v2_full_120k_fixture["raw"]
    _validate_full_test_tables(api, schema_v2_full_120k_fixture)
    target = next(
        row for row in raw
        if row["method"] == "minimax_subset"
        and row["requested_k"] == 8
        and row["reveal_ratio"] < 1.0
    )
    assert target["method"] == "minimax_subset"
    assert target["requested_k"] == 8 and target["reveal_ratio"] < 1.0
    original = target["joint_coverage"]
    target["joint_coverage"] = 0.0 if original != 0.0 else 1.0
    try:
        with pytest.raises(ValueError, match="coverage|raw|summary|recompute|corrupt"):
            _validate_full_test_tables(api, schema_v2_full_120k_fixture)
    finally:
        target["joint_coverage"] = original


@pytest.mark.parametrize(
    "field,value",
    (
        ("raw_case_count", 319), ("nearest_rms_distance", 0.41),
        ("total_tail_events", 319.0),
        ("total_tail_hits", 319.0), ("total_tail_recall", 319.0 / 320.0),
        ("group_tail_events", 319.0),
        ("group_tail_hits", 319.0), ("group_tail_recall", 319.0 / 320.0),
        ("hotspot_events", 319.0),
        ("hotspot_hits", 319.0), ("hotspot_recall", 319.0 / 320.0),
    ),
)
def test_schema_v2_full_sequence_non_distance_aggregate_tamper_is_rejected(
    schema_v2_full_120k_fixture: Mapping[str, Any], field: str, value: float,
) -> None:
    api = _api()
    row = schema_v2_full_120k_fixture["sequences"][0]
    original = row[field]
    row[field] = value
    try:
        with pytest.raises(ValueError, match="tail|group|hotspot|recall|aggregate|recompute"):
            _validate_full_test_tables(api, schema_v2_full_120k_fixture)
    finally:
        row[field] = original


def _validate_single_test_case(
    api: Any, row: Mapping[str, Any], fixture: Mapping[str, Any]
) -> Any:
    if hasattr(api, "validate_schema_v2_table"):
        header = tuple(name for name, _ in SCHEMA_V2_TABLE_SCHEMAS["raw_case_metrics"])
        expected_provenance = dict(fixture["expected_provenance"])
        support_material = (
            f"support-{row['case_index']}-{row['method']}".encode("utf-8")
        )
        expected_provenance["support_digests"] = {
            row["case_id"]:hashlib.sha256(support_material).hexdigest()
        }
        expected_provenance["support_canonical_bytes"] = {
            row["case_id"]:support_material
        }
        return api.validate_schema_v2_table(
            "raw_case_metrics", header, [row],
            expected_provenance=expected_provenance,
            allow_incomplete_universe=True,
        )
    return api.validate_raw_case_rows([row], allow_incomplete_universe=True)


def test_schema_v2_all_same_lower_hex_digests_are_not_accepted_as_provenance(
    schema_v2_gate_raw_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    fields = (
        "sequence_digest","generator_config_digest","topology_digest","normalizer_digest",
        "observation_digest","ambiguity_digest","support_digest",
    )
    originals = [
        {field:row[field] for field in fields}
        for row in schema_v2_gate_raw_fixture["rows"]
    ]
    for row in schema_v2_gate_raw_fixture["rows"]:
        for field in fields:
            row[field] = "a" * 64
    try:
        with pytest.raises(ValueError, match="digest|provenance|manifest|config|topology|coordinate|support"):
            _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    finally:
        for row, original in zip(schema_v2_gate_raw_fixture["rows"], originals):
            row.update(original)


@pytest.fixture
def schema_v2_provisional_toy_staging(tmp_path: Path) -> dict[str, Any]:
    api = _api()
    staging = tmp_path / ".phase3b-staging-provisional-toy"
    api.materialize_provisional_toy_artifacts(staging)
    assert {path.name for path in staging.iterdir()} == set(SCHEMA_V2_ARTIFACTS)
    baseline = api.read_back_artifacts(
        staging, integrity_expected=False, allow_incomplete_universe=True
    )
    evidence = baseline["manifest"]["gate_evidence"]
    assert evidence["integrity_checks_complete"] is False
    assert evidence["integrity_checks_passed"] is False
    return {"staging": staging, "baseline": baseline}


@pytest.mark.parametrize(
    "field,value",
    (
        ("protocol_sha256", "b" * 64), ("selected_method", "boundary_scenarios"),
        ("selected_k", 4), ("data_status", "FAIL"),
        ("authorized_source_sha256", {name: "b" * 64 for name in SCHEMA_V2_SOURCE_KEYS}),
        ("topology", {"name": "Rear4GPU", "sha256": "b" * 64}),
        ("normalizer_digest", "b" * 64),
    ),
)
def test_schema_v2_manifest_derived_or_provenance_tamper_is_rejected(
    schema_v2_provisional_toy_staging: Mapping[str, Any], field: str, value: Any,
) -> None:
    api = _api()
    staging = schema_v2_provisional_toy_staging["staging"]
    manifest_path = staging / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original.decode("utf-8"))
    manifest[field] = value
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest|protocol|selected|status|source|topology|normalizer|recompute"):
        api.read_back_artifacts(
            staging, integrity_expected=False, allow_incomplete_universe=True
        )
    manifest_path.write_bytes(original)


@pytest.mark.parametrize(
    "field",
    (
        "sequence_digest","generator_config_digest","topology_digest","normalizer_digest",
        "support_digest",
    ),
)
def test_schema_v2_row_provenance_digest_tamper_is_rejected_without_missing_fields(
    schema_v2_full_120k_fixture: Mapping[str, Any], field: str,
) -> None:
    api = _api()
    baseline = dict(schema_v2_full_120k_fixture["raw"][0])
    _validate_single_test_case(api, baseline, schema_v2_full_120k_fixture)
    row = dict(baseline, **{field: "b" * 64})
    with pytest.raises(ValueError, match="digest|provenance|manifest|coordinate"):
        _validate_single_test_case(api, row, schema_v2_full_120k_fixture)


@pytest.mark.parametrize("field", ("observation_digest", "ambiguity_digest"))
def test_schema_v2_same_coordinate_digest_must_match_across_methods(
    schema_v2_gate_raw_fixture: Mapping[str, Any], field: str,
) -> None:
    api = _api()
    _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    row = schema_v2_gate_raw_fixture["rows"][1]
    original = row[field]
    row[field] = "b" * 64
    try:
        with pytest.raises(ValueError, match="observation|ambiguity|coordinate|digest|method"):
            _validate_gate_raw(api, schema_v2_gate_raw_fixture)
    finally:
        row[field] = original


def test_schema_v2_coordinated_manifest_gate_and_summary_tamper_cannot_override_raw(
    schema_v2_provisional_toy_staging: Mapping[str, Any],
) -> None:
    api = _api()
    staging = schema_v2_provisional_toy_staging["staging"]
    manifest_path = staging / "manifest.json"
    summary_path = staging / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest["gate_evidence"]["selected_joint_coverage"] = 0.1
    manifest["data_status"] = "FAIL"
    summary["gate_evidence"]["selected_joint_coverage"] = 0.1
    summary["data_status"] = "FAIL"
    coordinated_summary_digest = _v2_digest(summary)
    manifest["summary_sha256"] = coordinated_summary_digest
    manifest["artifact_logical_sha256"]["summary.json"] = coordinated_summary_digest
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gate|summary|raw|coverage|status"):
        api.read_back_artifacts(
            staging, integrity_expected=False, allow_incomplete_universe=True,
        )


def _v2_nested_value(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = root
    for key in path:
        assert isinstance(value, Mapping) and key in value
        value = value[key]
    return value


def _v2_set_nested_value(root: dict[str, Any], path: Sequence[str], value: Any) -> None:
    target: dict[str, Any] = root
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    assert path[-1] in target
    target[path[-1]] = value


def _v2_derived_tamper_path(manifest: Mapping[str, Any], mutation: str) -> tuple[str, ...]:
    if mutation == "calibration_radius":
        return ("calibration_radius",)
    if mutation == "validation_method_mean":
        means = manifest["validation_method_means"]
        assert isinstance(means, Mapping) and "minimax_subset" in means
        return ("validation_method_means", "minimax_subset")
    if mutation.startswith("lofo_"):
        evidence = manifest["lofo_fold_evidence"]
        assert isinstance(evidence, Mapping)
        if mutation == "lofo_aggregate_delta":
            assert "aggregate_delta" in evidence
            return ("lofo_fold_evidence", "aggregate_delta")
        field = {
            "lofo_family_delta":"family_delta",
            "lofo_relative_degradation":"relative_degradation",
        }[mutation]
        fold_id = next(
            key for key, value in evidence.items()
            if isinstance(value, Mapping) and field in value
        )
        return ("lofo_fold_evidence", str(fold_id), field)
    dependence = manifest["test_total_traffic_dependence"]
    assert isinstance(dependence, Mapping) and "aggregate" in dependence
    if mutation.startswith("dependence_aggregate_"):
        field = mutation.removeprefix("dependence_aggregate_")
        aggregate = dependence["aggregate"]
        assert isinstance(aggregate, Mapping) and field in aggregate
        return ("test_total_traffic_dependence", "aggregate", field)
    field = mutation.removeprefix("dependence_sequence_")
    sequence_id = next(
        key for key, value in dependence.items()
        if key != "aggregate"
        and isinstance(value, Mapping)
        and field in value
        and value[field] is not None
    )
    return ("test_total_traffic_dependence", str(sequence_id), field)


@pytest.mark.parametrize(
    "mutation",
    (
        "calibration_radius", "validation_method_mean",
        "lofo_family_delta", "lofo_aggregate_delta",
        "lofo_relative_degradation",
        "dependence_sequence_lag1_acf",
        "dependence_sequence_positive_sequence_ess",
        "dependence_aggregate_mean_lag1_acf",
        "dependence_aggregate_mean_positive_sequence_ess",
    ),
)
def test_schema_v2_readback_recomputes_each_linked_derived_field_from_unchanged_raw(
    schema_v2_provisional_toy_staging: Mapping[str, Any], mutation: str,
) -> None:
    api = _api()
    staging = schema_v2_provisional_toy_staging["staging"]
    manifest_path = staging / "manifest.json"
    summary_path = staging / "summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_before = {
        name:(staging / name).read_bytes() for name in SCHEMA_V2_ARTIFACTS[1:-1]
    }
    path = _v2_derived_tamper_path(manifest, mutation)
    original = _v2_nested_value(manifest, path)
    assert type(original) in {int, float} and np.isfinite(float(original))
    changed = float(original) + 0.125
    _v2_set_nested_value(manifest, path, changed)
    if path[0] in summary:
        assert _v2_nested_value(summary, path) == original
        _v2_set_nested_value(summary, path, changed)
    summary_digest = _v2_digest(summary)
    manifest["summary_sha256"] = summary_digest
    manifest["artifact_logical_sha256"]["summary.json"] = summary_digest
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assert {
        name:(staging / name).read_bytes() for name in SCHEMA_V2_ARTIFACTS[1:-1]
    } == raw_before
    with pytest.raises(
        ValueError,
        match="raw|derived|recompute|calibration|validation|LOFO|lofo|dependence|ACF|acf|ESS|ess",
    ):
        api.read_back_artifacts(
            staging, integrity_expected=False, allow_incomplete_universe=True,
        )


def test_schema_v2_linked_raw_digest_attack_cannot_preserve_stale_derived_evidence(
    schema_v2_provisional_toy_staging: Mapping[str, Any],
) -> None:
    api = _api()
    staging = schema_v2_provisional_toy_staging["staging"]
    manifest_path = staging / "manifest.json"
    summary_path = staging / "summary.json"
    raw_path = staging / "raw_case_metrics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_gate = json.loads(json.dumps(manifest["gate_evidence"]))
    old_summary = summary_path.read_bytes()
    header, rows = _v2_read_csv(raw_path, "raw_case_metrics")
    selected = manifest["selected_method"]
    target = next(
        row for row in rows
        if row["method"] == selected
        and row["requested_k"] == 8
        and row["reveal_ratio"] < 1.0
        and row["joint_coverage"] == 1.0
    )
    target["joint_coverage"] = 0.0
    _v2_write_csv(raw_path, header, rows)
    logical = _v2_table_digest("raw_case_metrics", header, rows)
    scientific = _v2_table_digest("raw_case_metrics", header, rows, scientific=True)
    manifest["artifact_logical_sha256"]["raw_case_metrics.csv"] = logical
    manifest["artifact_scientific_sha256"]["raw_case_metrics.csv"] = scientific
    manifest["combined_scientific_evidence_sha256"] = _v2_combined_scientific_digest(
        manifest["artifact_scientific_sha256"]
    )
    assert manifest["gate_evidence"] == old_gate
    assert summary_path.read_bytes() == old_summary
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="raw|gate|coverage|derived|recompute|summary|scientific",
    ):
        api.read_back_artifacts(
            staging, integrity_expected=False, allow_incomplete_universe=True,
        )


def _v2_small_valid_table_case(table_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    row = _v2_minimal_row(table_name)
    if table_name in {"raw_calibration_scores", "raw_lofo_calibration_scores"}:
        split, family, record_index, sequence_index = "calibration", FAMILIES[1], 18, 3
    elif table_name in {"raw_validation_metrics", "raw_lofo_validation_metrics"}:
        split, family, record_index, sequence_index = "validation", FAMILIES[1], 17, 2
    elif table_name == "raw_lofo_test_metrics":
        split, family, record_index, sequence_index = "test", FAMILIES[0], 4, 4
    else:
        split, family, record_index, sequence_index = "test", FAMILIES[0], 4, 4
    family_index = FAMILIES.index(family)
    actual_seed = BASE_SEEDS[0] + family_index * 1_000_000 + sequence_index * 10_000
    sequence_id = f"{family}-base{BASE_SEEDS[0]}-sequence{sequence_index}-seed{actual_seed}"
    values = {
        "sequence_id":sequence_id, "split":split, "family":family,
        "base_seed":BASE_SEEDS[0], "record_index":record_index,
        "checkpoint":32, "mode_index":0, "reveal_mode":REVEAL_MODES[0],
        "stage_index":0, "reveal_ratio":0.0, "actual_entry_fraction":0.0,
        "requested_k":8, "construction_seed":31_000_000 + record_index * 100_000 + 320,
        "method":"minimax_subset", "replicate_count":1, "nearest_rms_distance":0.4,
        "uses_oracle":False, "upper_bound_only":False,
        "fold_id":f"lofo-0-{FAMILIES[0]}", "held_out_family":FAMILIES[0],
        "role":"selected", "score":0.1, "total_traffic":1.0,
        "sequence_digest":_sha("small-sequence"), "generator_config_digest":_sha("small-config"),
        "topology_digest":_sha("small-topology"), "normalizer_digest":_sha("small-normalizer"),
        "observation_digest":_sha("small-observation"), "ambiguity_digest":_sha("small-ambiguity"),
        "support_digest":_sha("small-support"),
    }
    for name, value in values.items():
        if name in row:
            row[name] = value
    identity = SCHEMA_V2_IDENTITIES[table_name]
    row["case_id"] = table_name + ":" + ":".join(str(row[name]) for name in identity)
    expected = {
        "protocol_sha256":SCHEMA_V2_PROTOCOL_SHA256,
        "topology_digest":row["topology_digest"],
        "normalizer_digest":row.get("normalizer_digest"),
        "sequence_records": {
            sequence_id: {
                "split":split, "family":family, "base_seed":BASE_SEEDS[0],
                "record_index":record_index, "sequence_digest":row["sequence_digest"],
                "generator_config_digest":row["generator_config_digest"],
            }
        },
    }
    if table_name in {
        "raw_lofo_calibration_scores",
        "raw_lofo_validation_metrics",
        "raw_lofo_test_metrics",
    }:
        expected["lofo_fold_normalizer_digests"] = {
            row["fold_id"]:row["normalizer_digest"]
        }
    return row, expected


def _ref_two_level_higher(rows: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence_id"]), []).append(float(row["score"]))
    sequence_values = [
        float(np.quantile(values, 0.9, method="higher"))
        for _, values in sorted(grouped.items())
    ]
    return float(np.quantile(sequence_values, 0.9, method="higher"))


def _ref_positive_sequence(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean()
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        return {"lag1_acf":None, "positive_sequence_ess":None, "defined":False}
    correlations = [
        float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        for lag in range(1, min(64, len(array) - 1) + 1)
    ]
    positive = []
    for value in correlations:
        if value <= 0.0:
            break
        positive.append(value)
    ess = len(array) / (1.0 + 2.0 * sum(positive))
    return {
        "lag1_acf":correlations[0],
        "positive_sequence_ess":float(np.clip(ess, 1.0, len(array))),
        "defined":True,
    }


def _set_case_id(table_name: str, row: dict[str, Any]) -> None:
    row["case_id"] = table_name + ":" + ":".join(
        str(row[name]) for name in SCHEMA_V2_IDENTITIES[table_name]
    )


@pytest.fixture(scope="session")
def schema_v2_derived_fixture() -> dict[str, Any]:
    api = _api()
    specs = tuple(api.build_formal_sequence_specs())
    topology_digest = _sha("derived-topology")
    normalizer_digest = _sha("derived-normalizer")
    sequence_records = {
        spec.sequence_id: {
            "split":spec.split, "family":spec.family, "base_seed":spec.base_seed,
            "record_index":spec.record_index,
            "sequence_digest":_sha(f"derived-sequence-{spec.record_index}"),
            "generator_config_digest":_sha(f"derived-config-{spec.record_index}"),
        }
        for spec in specs
    }

    def make_row(
        table_name: str, spec: Any, *, checkpoint: int = 32,
        mode_index: int = 0, stage_index: int = 0,
        method: str = "minimax_subset", fold_id: str | None = None,
        held_out_family: str | None = None, role: str = "selected",
        fold_normalizer: str | None = None,
    ) -> dict[str, Any]:
        row = _v2_minimal_row(table_name)
        values = {
            "sequence_id":spec.sequence_id, "split":spec.split, "family":spec.family,
            "base_seed":spec.base_seed, "record_index":spec.record_index,
            "checkpoint":checkpoint, "mode_index":mode_index,
            "reveal_mode":REVEAL_MODES[mode_index], "stage_index":stage_index,
            "reveal_ratio":UNKNOWN_RATIOS[stage_index],
            "actual_entry_fraction":UNKNOWN_RATIOS[stage_index],
            "requested_k":8,
            "construction_seed":31_000_000 + spec.record_index * 100_000 + checkpoint * 10 + mode_index,
            "method":method, "replicate_count":8 if method == "random_empirical" else 1,
            "nearest_rms_distance":0.0, "uses_oracle":False, "upper_bound_only":False,
            "fold_id":fold_id, "held_out_family":held_out_family, "role":role,
            "score":0.0, "total_traffic":0.0,
            "sequence_digest":sequence_records[spec.sequence_id]["sequence_digest"],
            "generator_config_digest":sequence_records[spec.sequence_id]["generator_config_digest"],
            "topology_digest":topology_digest,
            "normalizer_digest":fold_normalizer or normalizer_digest,
            "observation_digest":_sha(f"derived-observation-{spec.record_index}-{checkpoint}-{mode_index}-{stage_index}-{fold_id}"),
            "ambiguity_digest":_sha(f"derived-ambiguity-{spec.record_index}-{checkpoint}-{mode_index}-{stage_index}-{fold_id}"),
            "support_digest":_sha(f"derived-support-{spec.record_index}-{checkpoint}-{mode_index}-{stage_index}-{method}-{role}-{fold_id}"),
        }
        for name, value in values.items():
            if name in row:
                row[name] = value
        _set_case_id(table_name, row)
        return row

    calibration: list[dict[str, Any]] = []
    calibration_specs = [spec for spec in specs if spec.split == "calibration"]
    for sequence_index, spec in enumerate(calibration_specs):
        case_ordinal = 0
        for checkpoint in CHECKPOINTS:
            for mode_index in range(len(REVEAL_MODES)):
                for stage_index in range(len(UNKNOWN_RATIOS)):
                    row = make_row(
                        "raw_calibration_scores", spec, checkpoint=checkpoint,
                        mode_index=mode_index, stage_index=stage_index,
                    )
                    row["score"] = sequence_index * 10.0 + case_ordinal / 319.0
                    calibration.append(row)
                    case_ordinal += 1

    method_offsets = {
        "minimax_subset":0.0, "boundary_scenarios":5e-13,
        "worst_recent_cases":1.0, "random_empirical":2.0,
    }
    validation: list[dict[str, Any]] = []
    validation_specs = [spec for spec in specs if spec.split == "validation"]
    for sequence_index, spec in enumerate(validation_specs):
        repeats = 10 if sequence_index == 0 else 1
        for case_index in range(repeats):
            checkpoint = CHECKPOINTS[case_index]
            for method in ORDINARY_METHODS:
                row = make_row(
                    "raw_validation_metrics", spec, checkpoint=checkpoint,
                    method=method,
                )
                row["nearest_rms_distance"] = method_offsets[method] + (
                    0.0 if sequence_index == 0 else 1.0
                )
                validation.append(row)
    validation_means = {
        method:method_offsets[method] + 14.0 / 15.0 for method in ORDINARY_METHODS
    }

    lofo_calibration: list[dict[str, Any]] = []
    lofo_validation: list[dict[str, Any]] = []
    lofo_test: list[dict[str, Any]] = []
    fold_normalizers: dict[str, str] = {}
    lofo_evidence: dict[str, Any] = {}
    for held_index, held_family in enumerate(FAMILIES):
        fold_id = f"lofo-{held_index}-{held_family}"
        fold_digest = _sha(f"derived-lofo-normalizer-{held_index}")
        fold_normalizers[fold_id] = fold_digest
        seen_cal = [spec for spec in calibration_specs if spec.family != held_family]
        seen_val = [spec for spec in validation_specs if spec.family != held_family]
        held_test = [spec for spec in specs if spec.split == "test" and spec.family == held_family]
        for index, spec in enumerate(seen_cal):
            row = make_row(
                "raw_lofo_calibration_scores", spec, fold_id=fold_id,
                held_out_family=held_family, fold_normalizer=fold_digest,
            )
            row["score"] = index / 10.0
            lofo_calibration.append(row)
        for spec in seen_val:
            for method in ORDINARY_METHODS:
                row = make_row(
                    "raw_lofo_validation_metrics", spec, method=method,
                    fold_id=fold_id, held_out_family=held_family,
                    fold_normalizer=fold_digest,
                )
                row["nearest_rms_distance"] = method_offsets[method]
                lofo_validation.append(row)
        selected_values = []
        random_values = []
        for spec in held_test:
            selected_distance = 1.0 + held_index / 10.0
            random_distance = 2.0 + held_index / 10.0
            for role, method, distance in (
                ("selected","minimax_subset",selected_distance),
                ("random_comparator","random_empirical",random_distance),
            ):
                row = make_row(
                    "raw_lofo_test_metrics", spec, method=method,
                    fold_id=fold_id, held_out_family=held_family, role=role,
                    fold_normalizer=fold_digest,
                )
                row["nearest_rms_distance"] = distance
                lofo_test.append(row)
            selected_values.append(selected_distance)
            random_values.append(random_distance)
        selected_mean = float(np.mean(selected_values))
        random_mean = float(np.mean(random_values))
        lofo_evidence[fold_id] = {
            "held_out_family":held_family,
            "calibration_radius":_ref_two_level_higher([
                row for row in lofo_calibration if row["fold_id"] == fold_id
            ]),
            "selected_method":"minimax_subset",
            "family_delta":random_mean - selected_mean,
            "relative_degradation":(selected_mean - random_mean) / random_mean,
        }
    lofo_evidence["aggregate_delta"] = float(np.mean([
        lofo_evidence[f"lofo-{index}-{family}"]["family_delta"]
        for index, family in enumerate(FAMILIES)
    ]))

    dependence: list[dict[str, Any]] = []
    dependence_expected: dict[str, Any] = {}
    test_specs = [spec for spec in specs if spec.split == "test"]
    for sequence_index, spec in enumerate(test_specs):
        totals = []
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            total = float(
                20.0 + sequence_index + 3.0 * math.sin(checkpoint_index / 2.0)
                + checkpoint_index / 10.0
            )
            row = make_row("raw_dependence_metrics", spec, checkpoint=checkpoint)
            row["total_traffic"] = total
            dependence.append(row)
            totals.append(total)
        dependence_expected[spec.sequence_id] = _ref_positive_sequence(totals)
    defined_dependence = list(dependence_expected.values())
    dependence_expected["aggregate"] = {
        "mean_lag1_acf":float(np.mean([row["lag1_acf"] for row in defined_dependence])),
        "mean_positive_sequence_ess":float(np.mean([row["positive_sequence_ess"] for row in defined_dependence])),
        "sum_positive_sequence_ess":float(np.sum([row["positive_sequence_ess"] for row in defined_dependence])),
    }

    tables = {
        "raw_calibration_scores":calibration,
        "raw_validation_metrics":validation,
        "raw_lofo_calibration_scores":lofo_calibration,
        "raw_lofo_validation_metrics":lofo_validation,
        "raw_lofo_test_metrics":lofo_test,
        "raw_dependence_metrics":dependence,
    }
    expected_provenance = {
        "topology_digest":topology_digest, "normalizer_digest":normalizer_digest,
        "lofo_fold_normalizer_digests":fold_normalizers,
        "sequence_records":sequence_records,
    }
    expected_derived = {
        "calibration_radius":_ref_two_level_higher(calibration),
        "validation_method_means":validation_means,
        "selected_method":"minimax_subset", "selected_k":8,
        "lofo_fold_evidence":lofo_evidence,
        "test_total_traffic_dependence":dependence_expected,
    }
    return {
        "tables":tables, "expected_provenance":expected_provenance,
        "expected_derived":expected_derived,
    }


def _validate_derived_tables(api: Any, fixture: Mapping[str, Any]) -> Any:
    validator = getattr(api, "validate_schema_v2_derived_evidence", None)
    assert callable(validator), "schema-v2 derived-evidence validator contract missing"
    return validator(
        fixture["tables"], expected_provenance=fixture["expected_provenance"],
        expected_derived=fixture["expected_derived"], allow_incomplete_universe=True,
    )


def test_schema_v2_derived_tables_validator_accepts_baseline(
    schema_v2_derived_fixture: Mapping[str, Any],
) -> None:
    assert _validate_derived_tables(_api(), schema_v2_derived_fixture) is not None


def test_schema_v2_calibration_valid_score_change_recomputes_two_level_radius(
    schema_v2_derived_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    row = schema_v2_derived_fixture["tables"]["raw_calibration_scores"][13 * 320 + 288]
    original = row["score"]
    row["score"] = original + 0.25
    try:
        with pytest.raises(ValueError, match="calibration|radius|score|higher|derived"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        row["score"] = original


def test_schema_v2_manifest_calibration_radius_tamper_is_rejected_from_raw(
    schema_v2_derived_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    expected = schema_v2_derived_fixture["expected_derived"]
    original = expected["calibration_radius"]
    expected["calibration_radius"] = original + 0.1
    try:
        with pytest.raises(ValueError, match="calibration|radius|derived|manifest"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        expected["calibration_radius"] = original


@pytest.mark.parametrize(
    "mutation",
    ("nearest", "k", "method_means", "tie_selected_method"),
)
def test_schema_v2_validation_raw_and_derived_tamper_sequence_equal_and_tie_locked(
    schema_v2_derived_fixture: Mapping[str, Any], mutation: str,
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    rows = schema_v2_derived_fixture["tables"]["raw_validation_metrics"]
    expected = schema_v2_derived_fixture["expected_derived"]
    if mutation == "nearest":
        row, field, value = rows[0], "nearest_rms_distance", rows[0]["nearest_rms_distance"] + 0.2
    elif mutation == "k":
        row, field, value = rows[0], "requested_k", 4
    elif mutation == "method_means":
        row, field = expected["validation_method_means"], "minimax_subset"
        value = row[field] + 0.1
    else:
        row, field, value = expected, "selected_method", "boundary_scenarios"
    original = row[field]
    row[field] = value
    try:
        with pytest.raises(ValueError, match="validation|nearest|K|method|mean|tie|selected|derived"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        row[field] = original


@pytest.mark.parametrize("role", ("selected", "random_comparator"))
def test_schema_v2_lofo_valid_role_nearest_change_is_rejected(
    schema_v2_derived_fixture: Mapping[str, Any], role: str,
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    row = next(
        row for row in schema_v2_derived_fixture["tables"]["raw_lofo_test_metrics"]
        if row["fold_id"] == f"lofo-0-{FAMILIES[0]}" and row["role"] == role
    )
    original = row["nearest_rms_distance"]
    row["nearest_rms_distance"] = original + 0.1
    try:
        with pytest.raises(ValueError, match="LOFO|lofo|selected|random|delta|degradation|derived"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        row["nearest_rms_distance"] = original


@pytest.mark.parametrize("field", ("family_delta", "relative_degradation", "aggregate_delta"))
def test_schema_v2_lofo_fold_evidence_tamper_is_rejected(
    schema_v2_derived_fixture: Mapping[str, Any], field: str,
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    evidence = schema_v2_derived_fixture["expected_derived"]["lofo_fold_evidence"]
    target = evidence if field == "aggregate_delta" else evidence[f"lofo-0-{FAMILIES[0]}"]
    original = target[field]
    target[field] = original + 0.1
    try:
        with pytest.raises(ValueError, match="LOFO|lofo|family|aggregate|delta|degradation|derived"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        target[field] = original


def test_schema_v2_lofo_fold_normalizer_digest_is_unique_and_bound_per_fold(
    schema_v2_derived_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    digests = schema_v2_derived_fixture["expected_provenance"]["lofo_fold_normalizer_digests"]
    assert len(digests) == 5 and len(set(digests.values())) == 5
    keys = sorted(digests)
    original = digests[keys[1]]
    digests[keys[1]] = digests[keys[0]]
    try:
        with pytest.raises(ValueError, match="LOFO|lofo|normalizer|digest|fold|provenance"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        digests[keys[1]] = original


def test_schema_v2_dependence_valid_checkpoint_total_change_recomputes_acf_ess(
    schema_v2_derived_fixture: Mapping[str, Any],
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    row = schema_v2_derived_fixture["tables"]["raw_dependence_metrics"][3]
    original = row["total_traffic"]
    row["total_traffic"] = original + 0.5
    try:
        with pytest.raises(ValueError, match="dependence|traffic|ACF|acf|ESS|ess|derived"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        row["total_traffic"] = original


@pytest.mark.parametrize("scope,field", (("sequence","lag1_acf"),("sequence","positive_sequence_ess"),("aggregate","mean_lag1_acf"),("aggregate","mean_positive_sequence_ess")))
def test_schema_v2_manifest_dependence_acf_ess_tamper_is_rejected(
    schema_v2_derived_fixture: Mapping[str, Any], scope: str, field: str,
) -> None:
    api = _api()
    _validate_derived_tables(api, schema_v2_derived_fixture)
    dependence = schema_v2_derived_fixture["expected_derived"]["test_total_traffic_dependence"]
    target = dependence["aggregate"] if scope == "aggregate" else dependence[
        schema_v2_derived_fixture["tables"]["raw_dependence_metrics"][0]["sequence_id"]
    ]
    original = target[field]
    target[field] = original + 0.1
    try:
        with pytest.raises(ValueError, match="dependence|ACF|acf|ESS|ess|aggregate|derived|manifest"):
            _validate_derived_tables(api, schema_v2_derived_fixture)
    finally:
        target[field] = original


@pytest.mark.parametrize(
    "table_name,mutation",
    (
        ("raw_calibration_scores", {"score": -0.1}),
        ("raw_validation_metrics", {"method": "oracle_support_upper_bound"}),
        ("raw_lofo_calibration_scores", {"family": FAMILIES[0], "held_out_family": FAMILIES[0]}),
        ("raw_lofo_validation_metrics", {"family": FAMILIES[1], "held_out_family": FAMILIES[1]}),
        ("raw_lofo_test_metrics", {"role": "unknown-role"}),
        ("raw_dependence_metrics", {"total_traffic": -1.0}),
    ),
)
def test_schema_v2_small_typed_cal_validation_lofo_dependence_semantic_tamper(
    table_name: str, mutation: Mapping[str, Any],
) -> None:
    api = _api()
    row, expected_provenance = _v2_small_valid_table_case(table_name)
    header = tuple(name for name, _ in SCHEMA_V2_TABLE_SCHEMAS[table_name])
    assert len(_v2_table_digest(table_name, header, [row])) == 64
    validator = getattr(api, "validate_schema_v2_table", None)
    assert callable(validator), "schema-v2 table validator contract missing"
    validator(
        table_name, header, [row], allow_incomplete_universe=True,
        expected_provenance=expected_provenance,
    )
    corrupted = dict(row)
    corrupted.update(mutation)
    with pytest.raises(ValueError, match="score|oracle|held|fold|role|traffic|domain|derived"):
        validator(
            table_name, header, [corrupted], allow_incomplete_universe=True,
            expected_provenance=expected_provenance,
            expected_derived={"radius": 0.1, "selected_method": "minimax_subset"},
        )


def test_schema_v2_lofo_four_fold_identity_is_not_collapsed() -> None:
    rows = []
    for held_index, held in enumerate(FAMILIES[1:]):
        row, _ = _v2_small_valid_table_case("raw_lofo_calibration_scores")
        actual_seed = BASE_SEEDS[0] + 30_000
        sequence_id = f"{FAMILIES[0]}-base{BASE_SEEDS[0]}-sequence3-seed{actual_seed}"
        row.update(
            fold_id=f"lofo-{held_index + 1}-{held}", held_out_family=held,
            family=FAMILIES[0], sequence_id=sequence_id, split="calibration",
            record_index=3, checkpoint=32, mode_index=0, stage_index=0,
            construction_seed=31_000_000 + 3 * 100_000 + 320,
            normalizer_digest=_sha(f"lofo-fourfold-normalizer-{held_index + 1}"),
        )
        identity = SCHEMA_V2_IDENTITIES["raw_lofo_calibration_scores"]
        row["case_id"] = "raw_lofo_calibration_scores:" + ":".join(
            str(row[name]) for name in identity
        )
        rows.append(row)
    header = tuple(name for name, _ in SCHEMA_V2_TABLE_SCHEMAS["raw_lofo_calibration_scores"])
    digest = _v2_table_digest("raw_lofo_calibration_scores", header, rows)
    assert len(digest) == 64 and len({row["fold_id"] for row in rows}) == 4
    api = _api()
    expected = {
        "protocol_sha256":SCHEMA_V2_PROTOCOL_SHA256,
        "topology_digest":rows[0]["topology_digest"],
        "normalizer_digest":_sha("small-normalizer"),
        "lofo_fold_normalizer_digests": {
            row["fold_id"]: row["normalizer_digest"] for row in rows
        },
        "sequence_records": {
            rows[0]["sequence_id"]: {
                "split":"calibration", "family":FAMILIES[0], "base_seed":BASE_SEEDS[0],
                "record_index":3, "sequence_digest":rows[0]["sequence_digest"],
                "generator_config_digest":rows[0]["generator_config_digest"],
            }
        },
    }
    validator = getattr(api, "validate_schema_v2_table", None)
    assert callable(validator), "schema-v2 table validator contract missing"
    validator(
        "raw_lofo_calibration_scores", header, rows,
        expected_provenance=expected, allow_incomplete_universe=True,
    )
    collapsed = [dict(row, fold_id=rows[0]["fold_id"]) for row in rows]
    with pytest.raises(ValueError, match="fold|duplicate|identity"):
        validator(
            "raw_lofo_calibration_scores", header, collapsed,
            expected_provenance=expected, allow_incomplete_universe=True,
        )


def test_schema_v2_formal_test_converter_preserves_requested_k_for_all_methods_and_ratio_classes() -> None:
    api = _api()
    specs = tuple(api.build_formal_sequence_specs())
    spec = next(item for item in specs if item.split == "test")
    specs_by_id = {item.sequence_id:item for item in specs}
    _, records = api._v2_sequence_records()
    topology_digest = _sha("converter-topology")
    normalizer_digest = _sha("converter-normalizer")
    case_indices = api._case_index_lookup(specs)
    source_rows = []
    expected_coordinates = []
    for stage_index in (1, 4):
        ratio = REVEAL_RATIOS[stage_index]
        for requested_k in REQUESTED_K:
            case_index = case_indices[
                ("test", spec.record_index, CHECKPOINTS[0], 0, stage_index, requested_k)
            ]
            for method in ALL_METHODS:
                source = api._v2_make_row(
                    "raw_case_metrics", spec, records,
                    checkpoint=CHECKPOINTS[0], mode_index=0,
                    stage_index=stage_index, method=method,
                    normalizer_digest=normalizer_digest,
                    topology_digest=topology_digest, case_index=case_index,
                )
                source["requested_k"] = requested_k
                source["actual_k"] = 1 if ratio == 1.0 else requested_k
                source["uses_oracle"] = method == "oracle_support_upper_bound"
                source["upper_bound_only"] = method == "oracle_support_upper_bound"
                source_rows.append(source)
                expected_coordinates.append(
                    (stage_index, requested_k, method, case_index)
                )

    converted, sequence_rows = api._v2_convert_test_rows(
        source_rows, [], specs_by_id, records,
        topology_digest=topology_digest,
        normalizer_digest=normalizer_digest,
    )
    assert sequence_rows == []
    assert len(converted) == 2 * len(REQUESTED_K) * len(ALL_METHODS)
    assert [row["requested_k"] for row in converted] == [
        row["requested_k"] for row in source_rows
    ]
    for source, row, coordinate in zip(source_rows, converted, expected_coordinates):
        stage_index, requested_k, method, case_index = coordinate
        assert row["requested_k"] == requested_k
        assert row["actual_k"] == (1 if stage_index == 4 else requested_k)
        assert row["case_index"] == source["case_index"] == case_index
        assert row["case_id"] == source["case_id"] == f"case-{case_index}-{method}"

    unknown_rows = [row for row in converted if row["reveal_ratio"] < 1.0]
    header = tuple(name for name, _ in SCHEMA_V2_TABLE_SCHEMAS["raw_case_metrics"])
    api.validate_schema_v2_table(
        "raw_case_metrics", header, unknown_rows,
        allow_incomplete_universe=True,
        expected_provenance={
            "topology_digest":topology_digest,
            "normalizer_digest":normalizer_digest,
            "sequence_records":{spec.sequence_id:records[spec.sequence_id]},
        },
    )


def test_schema_v2_formal_observation_permanently_matches_real_reveal_process_all_25_cases() -> None:
    from rlccl.envs.problem import TopologyInfo
    from rlccl.uncertainty.ambiguity import (
        AmbiguityConstructionView, fit_descriptor_normalizer,
    )
    from rlccl.uncertainty.problem import UncertainProblemInstance
    from rlccl.uncertainty.reveal import DemandRevealProcess

    api = _api()
    edges = np.asarray(
        [(source, destination) for source in range(4) for destination in range(4) if source != destination],
        dtype=np.int64,
    )
    topology_info = TopologyInfo(4, len(edges), edges, np.ones(len(edges)), [], name="parity4")
    truth = np.asarray([[0,3,1,4],[2,0,5,1],[1,2,0,6],[4,1,3,0]], dtype=np.int64)
    history = tuple(truth.copy() for _ in range(32))
    public = api.PublicTopologyView.from_topology_info(topology_info)
    normalizer = fit_descriptor_normalizer(history, public)
    spec = api.SequenceSpec("parity","synthetic",0,0,0,0,0,"test",None,1024,{},0)
    for mode_index, mode in enumerate(REVEAL_MODES):
        seed = api.reveal_seed(0, 32, mode_index)
        world = UncertainProblemInstance.from_traffic_matrix(
            truth_matrix=truth, topology_info=topology_info, time_limit=1,
            sequence_id="parity", sequence_step=32, family="synthetic", generator_metadata={},
        )
        expected_states = tuple(DemandRevealProcess(
            problem=world, mode=mode, ratios=REVEAL_RATIOS, seed=seed,
        ))
        for expected, ratio in zip(expected_states, REVEAL_RATIOS):
            actual, _ = api._formal_observation(
                truth, spec=spec, checkpoint=32, topology=public,
                mode=mode, ratio=ratio, seed=seed,
            )
            np.testing.assert_array_equal(actual.entry_mask, expected.entry_mask)
            np.testing.assert_array_equal(actual.observed_matrix, expected.observed_matrix)
            expected_view = AmbiguityConstructionView.from_observation(
                history_matrices=history, history_offsets=tuple(range(-32,0)),
                observation=expected, construction_seed=seed, normalizer=normalizer,
            )
            actual_view = AmbiguityConstructionView.from_observation(
                history_matrices=history, history_offsets=tuple(range(-32,0)),
                observation=actual, construction_seed=seed, normalizer=normalizer,
            )
            assert api._canonical_bytes(expected_view) == api._canonical_bytes(actual_view)


def test_schema_v2_staging_double_readback_final_true_and_existing_destination_contract(tmp_path: Path) -> None:
    api = _api()
    staging = tmp_path / ".phase3b-staging-reference"
    api.materialize_provisional_toy_artifacts(staging)
    first = api.read_back_artifacts(
        staging, integrity_expected=False, allow_incomplete_universe=True
    )
    second = api.read_back_artifacts(
        staging, integrity_expected=False, allow_incomplete_universe=True
    )
    assert first is not second
    for provisional in (first, second):
        evidence = provisional["manifest"]["gate_evidence"]
        assert evidence["integrity_checks_complete"] is False
        assert evidence["integrity_checks_passed"] is False
    final = api.finalize_staged_artifacts(
        staging, allow_incomplete_universe=True
    )
    assert final["manifest"]["gate_evidence"]["integrity_checks_complete"] is True
    assert final["manifest"]["gate_evidence"]["integrity_checks_passed"] is True
    readback = api.read_back_artifacts(
        staging, integrity_expected=True, allow_incomplete_universe=True
    )
    assert readback == final
    assert readback["manifest"]["gate_evidence"]["integrity_checks_complete"] is True
    assert readback["manifest"]["gate_evidence"]["integrity_checks_passed"] is True
    destination = tmp_path / "phase3b"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="destination|exist|overwrite"):
        api.publish_artifacts_atomically(staging, destination)


def test_schema_v2_publish_uses_one_directory_rename_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api()
    staging = tmp_path / ".phase3b-staging-once"
    api.materialize_provisional_toy_artifacts(staging)
    api.finalize_staged_artifacts(staging, allow_incomplete_universe=True)
    destination = tmp_path / "phase3b"
    calls: list[tuple[str, Path, Path]] = []
    active = False
    real_os_replace = os.replace
    real_os_rename = os.rename
    real_path_replace = Path.replace
    real_path_rename = Path.rename

    def invoke_once(operation: str, source: Any, target: Any, callback: Any) -> Any:
        nonlocal active
        outermost = not active
        if outermost:
            active = True
            calls.append((operation, Path(source), Path(target)))
        try:
            return callback()
        finally:
            if outermost:
                active = False

    def tracked_os_replace(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke_once(
            "os.replace", source, target,
            lambda: real_os_replace(source, target, *args, **kwargs),
        )

    def tracked_os_rename(source: Any, target: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke_once(
            "os.rename", source, target,
            lambda: real_os_rename(source, target, *args, **kwargs),
        )

    def tracked_replace(source: Path, target: Path) -> Path:
        return invoke_once(
            "Path.replace", source, target,
            lambda: real_path_replace(source, target),
        )

    def tracked_rename(source: Path, target: Path) -> Path:
        return invoke_once(
            "Path.rename", source, target,
            lambda: real_path_rename(source, target),
        )

    monkeypatch.setattr(os, "replace", tracked_os_replace)
    monkeypatch.setattr(os, "rename", tracked_os_rename)
    monkeypatch.setattr(Path, "replace", tracked_replace)
    monkeypatch.setattr(Path, "rename", tracked_rename)
    api.publish_artifacts_atomically(staging, destination)
    assert staging.parent == destination.parent
    assert staging.name.startswith(".phase3b-staging-")
    assert len(calls) == 1
    assert calls[0][1:] == (staging, destination)
    assert calls[0][0] in {"os.replace", "os.rename", "Path.replace", "Path.rename"}
    assert not staging.exists()
    assert destination.is_dir()
    assert {path.name for path in destination.iterdir()} == set(SCHEMA_V2_ARTIFACTS)
