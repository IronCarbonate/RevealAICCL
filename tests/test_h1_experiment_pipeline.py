"""RED contracts for the auditable Gate H1 experiment pipeline.

Imports of the not-yet-authorized experiment module and CLI runner are delayed
until test execution.  The file must collect before either production entry
point exists.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pytest


FAMILIES = (
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
)
BASE_SEEDS = (42, 142, 242)
REGISTERED_METHODS = (
    "long_term_mean",
    "previous_value",
    "ewma",
    "moment_only",
    "recent_history_mlp",
    "causal_tcn",
    "quantile_scenario",
)
POINT_METHODS = (*REGISTERED_METHODS, "selected_recent")
CONTINUOUS_TARGETS = (
    "total_traffic",
    "source_load_vector",
    "destination_load_vector",
    "hotspot_strength",
    "sparsity",
    "bandwidth_group_offered_load_vector",
)
PRIMARY_TARGETS = CONTINUOUS_TARGETS[:3]
RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_h1_predictability.py"


def _api() -> SimpleNamespace:
    return SimpleNamespace(
        data=importlib.import_module("rlccl.prediction.data"),
        experiment=importlib.import_module("rlccl.prediction.experiment"),
    )


def _runner_module() -> ModuleType:
    if not RUNNER_PATH.is_file():
        raise AssertionError(f"H1 runner is missing: {RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("h1_runner_contract", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _formal_manifest(api: SimpleNamespace) -> dict[str, Any]:
    specs = api.data.build_formal_sequence_specs()
    records = [
        {
            "sequence_id": spec.sequence_id,
            "split": spec.split,
            "digest": _digest(spec.sequence_id),
        }
        for spec in specs
    ]
    return api.experiment.build_experiment_manifest(
        specs=specs,
        sequence_records=records,
        protocol={"path": "docs/uncertainty_aiccl/H1_PREDICTABILITY_PROTOCOL.md", "sha256": "a" * 64},
        source={"kind": "workspace_tree", "sha256": "b" * 64},
        command=["F:/AnaConda/python.exe", "-B", "scripts/run_h1_predictability.py"],
        environment={"python": "3.12.7", "numpy": "1.26.4", "sklearn": "1.5.1"},
        topology={"name": "Rear4GPU", "sha256": "c" * 64},
        group_coefficients_digest="d" * 64,
    )


def _identity(manifest: Mapping[str, Any], split: str) -> list[str]:
    return list(manifest["pooled_identity"][f"{split}_sequence_ids"])


def _validation_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {row["sequence_id"]: row for row in manifest["sequence_records"]}
    rows: list[dict[str, Any]] = []
    for sequence_id in _identity(manifest, "validation"):
        record = records[sequence_id]
        for method, rmse in (("recent_history_mlp", 2.0), ("causal_tcn", 1.0)):
            rows.append(
                {
                    "scope": "pooled",
                    "split": "validation",
                    "held_out_family": "",
                    "sequence_id": sequence_id,
                    "family": record["family"],
                    "base_seed": record["base_seed"],
                    "method": method,
                    "target": "total_traffic",
                    "rmse": rmse,
                }
            )
    for held_out in FAMILIES:
        identity = manifest["lofo_identities"][held_out]
        for sequence_id in identity["validation_sequence_ids"]:
            record = records[sequence_id]
            for method, rmse in (("recent_history_mlp", 2.0), ("causal_tcn", 1.0)):
                rows.append(
                    {
                        "scope": "lofo",
                        "split": "validation",
                        "held_out_family": held_out,
                        "sequence_id": sequence_id,
                        "family": record["family"],
                        "base_seed": record["base_seed"],
                        "method": method,
                        "target": "total_traffic",
                        "rmse": rmse,
                    }
                )
    return rows


def _previous_key(scope: str, held_out: str, sequence_id: str, target: str) -> str:
    return "::".join((scope, held_out, sequence_id, "previous_value", target))


def _point_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {row["sequence_id"]: row for row in manifest["sequence_records"]}
    rows = _validation_rows(manifest)
    scopes: list[tuple[str, str, Sequence[str]]] = [
        ("pooled", "", _identity(manifest, "test")),
        *(
            (
                "lofo",
                held_out,
                manifest["lofo_identities"][held_out]["test_sequence_ids"],
            )
            for held_out in FAMILIES
        ),
    ]
    for scope, held_out, sequence_ids in scopes:
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            for method in POINT_METHODS:
                rmse = (
                    2.0
                    if method == "previous_value"
                    else 1.0
                    if method in {"causal_tcn", "selected_recent", "quantile_scenario"}
                    else 1.5
                )
                for target in CONTINUOUS_TARGETS:
                    rows.append(
                        {
                            "scope": scope,
                            "split": "test",
                            "held_out_family": held_out,
                            "sequence_id": sequence_id,
                            "family": record["family"],
                            "base_seed": record["base_seed"],
                            "method": method,
                            "target": target,
                            "mae": rmse / 2.0,
                            "rmse": rmse,
                            "r2": 0.5,
                            "spearman": 0.5,
                            "delta_rmse": 2.0 - rmse,
                            "hotspot_accuracy": 0.75,
                            "acf_lag1": 0.20,
                            "ess": 500.0,
                            "raw_step_count": 1016,
                            "previous_row_key": _previous_key(
                                scope, held_out, sequence_id, target
                            ),
                        }
                    )
    return rows


def _probability_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {row["sequence_id"]: row for row in manifest["sequence_records"]}
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, Sequence[str]]] = [
        ("pooled", "", _identity(manifest, "test")),
        *(
            (
                "lofo",
                held_out,
                manifest["lofo_identities"][held_out]["test_sequence_ids"],
            )
            for held_out in FAMILIES
        ),
    ]
    for scope, held_out, sequence_ids in scopes:
        for sequence_id in sequence_ids:
            record = records[sequence_id]
            for target in CONTINUOUS_TARGETS:
                rows.append(
                    {
                        "scope": scope,
                        "held_out_family": held_out,
                        "sequence_id": sequence_id,
                        "family": record["family"],
                        "base_seed": record["base_seed"],
                        "target": target,
                        "coverage_numerator": 8,
                        "coverage_denominator": 10,
                        "interval_width_sum": 20.0,
                        "scenario_coverage_numerator": 8,
                        "scenario_coverage_denominator": 10,
                        "tail_event_count": 1 if target == "total_traffic" else 0,
                        "tail_true_positive_count": 1 if target == "total_traffic" else 0,
                    }
                )
    return rows


def test_experiment_module_and_cli_runner_are_delayed_red_dependencies() -> None:
    assert importlib.util.find_spec("rlccl.prediction.experiment") is not None
    assert RUNNER_PATH.is_file()


def test_runner_exposes_safe_main_without_executing_on_import() -> None:
    runner = _runner_module()
    assert callable(runner.build_parser)
    assert callable(runner.main)
    parser = runner.build_parser()
    assert parser.parse_args([]).output_dir == "outputs/h1_predictability"


def test_toy_end_to_end_respects_four_splits_selection_calibration_test_and_lofo() -> None:
    api = _api()
    bundle = api.experiment.run_toy_experiment(seed=20260731)
    manifest, summary = bundle["manifest"], bundle["summary"]
    pooled = manifest["pooled_identity"]
    split_sets = {
        split: set(pooled[f"{split}_sequence_ids"])
        for split in ("fit", "validation", "calibration", "test")
    }
    assert all(split_sets[left].isdisjoint(split_sets[right]) for left in split_sets for right in split_sets if left != right)
    assert summary["pipeline_trace"] == [
        "fit",
        "validation_selection",
        "calibration_residuals",
        "test_evaluation",
        "lofo_refits",
    ]
    assert summary["selected_backbone"] == api.experiment.recompute_validation_selection(
        bundle["point_rows"], manifest
    )
    assert set(summary["calibration_sequence_ids"]) == split_sets["calibration"]
    for held_out, identity in manifest["lofo_identities"].items():
        for split in ("fit", "validation", "calibration"):
            assert held_out not in {
                row["family"]
                for row in manifest["sequence_records"]
                if row["sequence_id"] in identity[f"{split}_sequence_ids"]
            }
        assert {row["family"] for row in manifest["sequence_records"] if row["sequence_id"] in identity["test_sequence_ids"]} == {held_out}


def test_formal_manifest_has_75_full_configs_digests_and_all_provenance() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    specs = {spec.sequence_id: spec for spec in api.data.build_formal_sequence_specs()}
    assert manifest["schema_version"] == 2
    assert len(manifest["sequence_records"]) == 75
    for record in manifest["sequence_records"]:
        spec = specs[record["sequence_id"]]
        assert record["digest"] == _digest(spec.sequence_id)
        assert record["generator_config"] == json.loads(json.dumps(spec.generator_config))
        assert record["actual_seed"] == spec.actual_seed
        assert record["base_seed"] == spec.base_seed
        assert record["family_index"] == spec.family_index
        assert record["seed_index"] == spec.seed_index
        assert record["sequence_index"] == spec.sequence_index
        assert record["dynamics_variant"] == spec.dynamics_variant
    assert manifest["protocol"]["path"] == "docs/uncertainty_aiccl/H1_PREDICTABILITY_PROTOCOL.md"
    assert manifest["protocol"]["sha256"] == "a" * 64
    assert manifest["source"]["kind"] == "workspace_tree"
    assert manifest["source"]["sha256"] == "b" * 64
    assert manifest["command"][-1] == "scripts/run_h1_predictability.py"
    assert {"python", "numpy", "sklearn"} <= set(manifest["environment"])
    assert manifest["topology"]["name"] == "Rear4GPU"
    assert manifest["topology"]["sha256"] == "c" * 64
    assert manifest["group_coefficients_digest"] == "d" * 64
    assert len(manifest["pooled_identity"]["calibration_sequence_ids"]) == 15
    assert set(manifest["lofo_identities"]) == set(FAMILIES)
    api.experiment.validate_experiment_manifest(manifest)


def test_validation_raw_rows_alone_recompute_pooled_and_lofo_selection() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _validation_rows(manifest)
    assert api.experiment.recompute_validation_selection(rows, manifest) == "causal_tcn"
    for held_out in FAMILIES:
        assert api.experiment.recompute_validation_selection(
            rows, manifest, held_out_family=held_out
        ) == "causal_tcn"
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["scope"] == "pooled" and row["method"] == "recent_history_mlp":
            row["rmse"] = 0.5
    assert api.experiment.recompute_validation_selection(changed, manifest) == "recent_history_mlp"


def test_pooled_and_lofo_point_raw_rows_are_exact_and_auditable() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _point_rows(manifest)
    api.experiment.validate_point_rows(rows, manifest)
    test_rows = [row for row in rows if row["split"] == "test"]
    assert {row["scope"] for row in test_rows} == {"pooled", "lofo"}
    assert {row["method"] for row in test_rows} == set(POINT_METHODS)
    assert {row["target"] for row in test_rows} == set(CONTINUOUS_TARGETS)
    assert all(0.0 <= row["hotspot_accuracy"] <= 1.0 for row in test_rows)
    assert all(np.isfinite(row["acf_lag1"]) and row["ess"] > 0 for row in test_rows)
    assert all(row["raw_step_count"] == 1016 for row in test_rows)
    assert all(row["previous_row_key"] for row in test_rows)
    by_identity = {
        (
            row["scope"],
            row["held_out_family"],
            row["sequence_id"],
            row["method"],
            row["target"],
        ): row
        for row in test_rows
    }
    alias_fields = (
        "mae",
        "rmse",
        "r2",
        "spearman",
        "delta_rmse",
        "hotspot_accuracy",
        "acf_lag1",
        "ess",
        "raw_step_count",
        "previous_row_key",
    )
    for row in test_rows:
        if row["method"] != "selected_recent":
            continue
        backbone = by_identity[
            (
                row["scope"],
                row["held_out_family"],
                row["sequence_id"],
                "causal_tcn",
                row["target"],
            )
        ]
        assert {field: row[field] for field in alias_fields} == {
            field: backbone[field] for field in alias_fields
        }


def test_point_validator_rejects_selected_alias_mismatch_from_validation_choice() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _point_rows(manifest)
    for row in rows:
        if (
            row["scope"] == "pooled"
            and row["split"] == "test"
            and row["method"] == "selected_recent"
            and row["target"] == "total_traffic"
        ):
            row["rmse"] += 0.25
            break
    with pytest.raises(ValueError, match="alias|selected|backbone|mismatch"):
        api.experiment.validate_point_rows(rows, manifest)


def test_probability_raw_rows_keep_recomputable_counts_width_and_tail_tp() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _probability_rows(manifest)
    api.experiment.validate_probability_rows(rows, manifest)
    assert {row["scope"] for row in rows} == {"pooled", "lofo"}
    for row in rows:
        assert 0 <= row["coverage_numerator"] <= row["coverage_denominator"]
        assert row["interval_width_sum"] >= 0.0
        assert 0 <= row["scenario_coverage_numerator"] <= row["scenario_coverage_denominator"]
        assert 0 <= row["tail_true_positive_count"] <= row["tail_event_count"]


def test_formal_summary_is_recomputed_from_raw_with_exact_identity_coverage() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    summary = api.experiment.recompute_formal_summary(
        _point_rows(manifest), _probability_rows(manifest), manifest
    )
    assert summary["identity_counts"] == {
        "families": 5,
        "base_seeds": 3,
        "pooled_test_sequences": 15,
        "lofo_test_sequences": 15,
        "point_methods": len(POINT_METHODS),
        "continuous_targets": len(CONTINUOUS_TARGETS),
    }
    assert set(summary["primary_delta_ci"]) == set(PRIMARY_TARGETS)
    assert all(summary["primary_delta_ci"][name]["lower"] > 0 for name in PRIMARY_TARGETS)
    probability = summary["probability"]["pooled"]
    assert probability["coverage"] == pytest.approx(0.80)
    assert probability["mean_interval_width"] == pytest.approx(2.0)
    assert probability["scenario_coverage"] == pytest.approx(0.80)
    assert probability["tail_event_count"] == 15
    assert probability["tail_true_positive_count"] == 15
    assert probability["tail_recall"] == pytest.approx(1.0)
    assert set(summary["lofo"]["family_mean_deltas"]) == set(FAMILIES)


def test_summary_contains_only_data_conditions_one_to_six_pending_supervisor() -> None:
    api = _api()
    manifest = _formal_manifest(api)
    summary = api.experiment.recompute_formal_summary(
        _point_rows(manifest), _probability_rows(manifest), manifest
    )
    assert summary["gate_status"] == "PENDING_SUPERVISOR"
    assert tuple(summary["conditions"]) == tuple(
        f"condition_{index}" for index in range(1, 7)
    )
    assert all(summary["conditions"].values())
    assert "condition_7" not in summary["conditions"]
    assert "supervisor_veto" not in summary
    assert "final_decision" not in summary


@pytest.mark.parametrize("corruption", ("missing", "duplicate"))
def test_point_raw_completeness_rejects_missing_or_duplicate_identity(corruption: str) -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _point_rows(manifest)
    corrupted = rows[:-1] if corruption == "missing" else [*rows, dict(rows[-1])]
    with pytest.raises(ValueError, match="missing|incomplete|duplicate|identity"):
        api.experiment.validate_point_rows(corrupted, manifest)


@pytest.mark.parametrize("corruption", ("missing", "duplicate"))
def test_probability_raw_completeness_rejects_missing_or_duplicate_identity(
    corruption: str,
) -> None:
    api = _api()
    manifest = _formal_manifest(api)
    rows = _probability_rows(manifest)
    corrupted = rows[:-1] if corruption == "missing" else [*rows, dict(rows[-1])]
    with pytest.raises(ValueError, match="missing|incomplete|duplicate|identity"):
        api.experiment.validate_probability_rows(corrupted, manifest)


@pytest.mark.parametrize(
    "corruption",
    (
        "generator_config",
        "base_seed",
        "actual_seed",
        "family_index",
        "seed_index",
        "sequence_index",
        "dynamics_variant",
        "digest",
        "duplicate_digest",
    ),
)
def test_manifest_validator_rejects_incomplete_records_and_duplicate_digests(
    corruption: str,
) -> None:
    api = _api()
    manifest = _formal_manifest(api)
    api.experiment.validate_experiment_manifest(manifest)
    corrupted = json.loads(json.dumps(manifest))
    if corruption == "duplicate_digest":
        corrupted["sequence_records"][1]["digest"] = corrupted["sequence_records"][0][
            "digest"
        ]
    else:
        del corrupted["sequence_records"][0][corruption]
    with pytest.raises(ValueError, match="manifest|record|missing|digest|duplicate"):
        api.experiment.validate_experiment_manifest(corrupted)


def test_two_toy_runs_write_byte_identical_auditable_artifacts(tmp_path: Path) -> None:
    api = _api()
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first = api.experiment.run_toy_experiment(output_dir=first_dir, seed=20260731)
    second = api.experiment.run_toy_experiment(output_dir=second_dir, seed=20260731)
    expected = {
        "manifest.json",
        "raw_sequence_metrics.csv",
        "raw_probability_metrics.csv",
        "summary.json",
    }
    assert {path.name for path in first_dir.iterdir()} == expected
    assert {path.name for path in second_dir.iterdir()} == expected
    assert first["summary"] == second["summary"]
    for name in expected:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
