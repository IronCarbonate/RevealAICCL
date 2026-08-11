"""Auditable end-to-end experiment pipeline for preregistered Gate H1.

The formal entry point is deliberately parameter-free with respect to the
frozen families, seeds, and sequence length.  Unit tests use the explicit toy
entry point, which follows the same split/selection/calibration/LOFO flow but
never generates a formal long-horizon sequence.
"""

from __future__ import annotations

import csv
from dataclasses import fields, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rlccl.models.traffic_predictor import deterministic_group_coefficients
from rlccl.traffic.long_horizon_generator import LongHorizonTrafficConfig

from .calibration import ResidualCalibrator
from .data import (
    FORMAL_BASE_SEEDS,
    FORMAL_FAMILIES,
    HistoryExamples,
    SequenceSpec,
    build_formal_sequence_specs,
    build_history_examples,
    generate_formal_sequence,
    sequence_digest,
)
from .models import HistoryPredictorSuite, METHOD_NAMES, select_recent_backbone
from .statistics import (
    PRIMARY_TARGETS,
    autocorrelation,
    family_stratified_paired_bootstrap,
    hotspot_from_destination_loads,
    positive_sequence_ess,
    sequence_target_metrics,
)


CONTINUOUS_TARGETS = (
    "total_traffic",
    "source_load_vector",
    "destination_load_vector",
    "hotspot_strength",
    "sparsity",
    "bandwidth_group_offered_load_vector",
)
POINT_METHODS = (*METHOD_NAMES, "selected_recent")
CANDIDATE_METHODS = ("recent_history_mlp", "causal_tcn")
PIPELINE_TRACE = (
    "fit",
    "validation_selection",
    "calibration_residuals",
    "test_evaluation",
    "lofo_refits",
)
_CONFIG_FIELDS = {item.name for item in fields(LongHorizonTrafficConfig)}
_HEX = set("0123456789abcdef")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _require_digest(value: Any, name: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in _HEX for character in digest):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return digest


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _expected_identities(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    pooled = {
        f"{split}_sequence_ids": [
            str(record["sequence_id"])
            for record in records
            if record["split"] == split
        ]
        for split in ("fit", "validation", "calibration", "test")
    }
    families = _ordered_unique(record["family"] for record in records)
    lofo: dict[str, dict[str, list[str]]] = {}
    for held_out in families:
        lofo[str(held_out)] = {
            f"{split}_sequence_ids": [
                str(record["sequence_id"])
                for record in records
                if record["split"] == split and record["family"] != held_out
            ]
            for split in ("fit", "validation", "calibration")
        }
        lofo[str(held_out)]["test_sequence_ids"] = [
            str(record["sequence_id"])
            for record in records
            if record["split"] == "test" and record["family"] == held_out
        ]
    return pooled, lofo


def build_experiment_manifest(
    *,
    specs: Sequence[SequenceSpec],
    sequence_records: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    command: Sequence[str],
    environment: Mapping[str, Any],
    topology: Mapping[str, Any],
    group_coefficients_digest: str,
) -> dict[str, Any]:
    """Build schema 2 with full per-sequence config and explicit identities."""

    spec_by_id = {spec.sequence_id: spec for spec in specs}
    if len(spec_by_id) != len(specs):
        raise ValueError("duplicate sequence identity in specs")
    supplied: dict[str, Mapping[str, Any]] = {}
    for record in sequence_records:
        sequence_id = str(record.get("sequence_id", ""))
        if not sequence_id or sequence_id in supplied:
            raise ValueError("duplicate or missing sequence identity in records")
        supplied[sequence_id] = record
    if set(supplied) != set(spec_by_id):
        raise ValueError("sequence record identities do not match specs")
    records: list[dict[str, Any]] = []
    for spec in specs:
        supplied_record = supplied[spec.sequence_id]
        if str(supplied_record.get("split")) != spec.split:
            raise ValueError(f"split mismatch for {spec.sequence_id}")
        records.append(
            {
                "sequence_id": spec.sequence_id,
                "family": spec.family,
                "family_index": spec.family_index,
                "base_seed": spec.base_seed,
                "seed_index": spec.seed_index,
                "sequence_index": spec.sequence_index,
                "actual_seed": spec.actual_seed,
                "split": spec.split,
                "dynamics_variant": spec.dynamics_variant,
                "sequence_length": spec.sequence_length,
                "generator_config": _json_value(spec.generator_config),
                "digest": str(supplied_record.get("digest", "")),
            }
        )
    pooled, lofo = _expected_identities(records)
    manifest = _json_value(
        {
            "schema_version": 2,
            "protocol": dict(protocol),
            "source": dict(source),
            "command": list(command),
            "environment": dict(environment),
            "topology": dict(topology),
            "group_coefficients_digest": group_coefficients_digest,
            "families": _ordered_unique(record["family"] for record in records),
            "base_seeds": _ordered_unique(record["base_seed"] for record in records),
            "methods": list(METHOD_NAMES),
            "point_methods": list(POINT_METHODS),
            "continuous_targets": list(CONTINUOUS_TARGETS),
            "random_seed": 20260731,
            "sequence_records": records,
            "pooled_identity": pooled,
            "lofo_identities": lofo,
        }
    )
    validate_experiment_manifest(manifest)
    return manifest


def validate_experiment_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol",
        "source",
        "command",
        "environment",
        "topology",
        "group_coefficients_digest",
        "families",
        "base_seeds",
        "methods",
        "point_methods",
        "continuous_targets",
        "sequence_records",
        "pooled_identity",
        "lofo_identities",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if int(manifest["schema_version"]) != 2:
        raise ValueError("manifest schema_version must be 2")
    for name in ("protocol", "source", "topology"):
        if not isinstance(manifest[name], Mapping):
            raise ValueError(f"manifest {name} must be a mapping")
    _require_digest(manifest["protocol"].get("sha256"), "protocol sha256")
    _require_digest(manifest["source"].get("sha256"), "source sha256")
    _require_digest(manifest["topology"].get("sha256"), "topology sha256")
    _require_digest(manifest["group_coefficients_digest"], "group coefficients digest")
    if not manifest["protocol"].get("path") or not manifest["source"].get("kind"):
        raise ValueError("manifest protocol/source provenance is incomplete")
    if not manifest["topology"].get("name") or not list(manifest["command"]):
        raise ValueError("manifest topology/command provenance is incomplete")
    if not {"python", "numpy", "sklearn"} <= set(manifest["environment"]):
        raise ValueError("manifest environment provenance is incomplete")
    if tuple(manifest["methods"]) != METHOD_NAMES:
        raise ValueError("manifest method registry mismatch")
    if tuple(manifest["point_methods"]) != POINT_METHODS:
        raise ValueError("manifest point method registry mismatch")
    if tuple(manifest["continuous_targets"]) != CONTINUOUS_TARGETS:
        raise ValueError("manifest target registry mismatch")

    records = list(manifest["sequence_records"])
    if not records:
        raise ValueError("manifest sequence records must not be empty")
    record_required = {
        "sequence_id",
        "family",
        "family_index",
        "base_seed",
        "seed_index",
        "sequence_index",
        "actual_seed",
        "split",
        "dynamics_variant",
        "sequence_length",
        "generator_config",
        "digest",
    }
    ids: set[str] = set()
    digests: set[str] = set()
    for index, record in enumerate(records):
        absent = record_required - set(record)
        if absent:
            raise ValueError(f"manifest record {index} missing fields: {sorted(absent)}")
        sequence_id = str(record["sequence_id"])
        digest = _require_digest(record["digest"], f"record {index} digest")
        if sequence_id in ids:
            raise ValueError(f"duplicate manifest sequence identity: {sequence_id}")
        if digest in digests:
            raise ValueError(f"duplicate manifest digest: {digest}")
        ids.add(sequence_id)
        digests.add(digest)
        family = str(record["family"])
        if family not in FORMAL_FAMILIES:
            raise ValueError(f"unknown manifest family: {family}")
        family_index = FORMAL_FAMILIES.index(family)
        base_seed = int(record["base_seed"])
        if base_seed not in FORMAL_BASE_SEEDS:
            raise ValueError(f"unknown manifest base_seed: {base_seed}")
        seed_index = FORMAL_BASE_SEEDS.index(base_seed)
        sequence_index = int(record["sequence_index"])
        if sequence_index not in range(5):
            raise ValueError("manifest sequence_index must be in [0, 4]")
        expected_split = ("fit", "fit", "validation", "calibration", "test")[
            sequence_index
        ]
        expected_seed = base_seed + family_index * 1_000_000 + sequence_index * 10_000
        expected_variant = (
            ("smooth", "random_switching", "long_regime", "shock_recovery")[
                (seed_index + sequence_index) % 4
            ]
            if family == "same_moments_different_dynamics"
            else None
        )
        if int(record["family_index"]) != family_index or int(record["seed_index"]) != seed_index:
            raise ValueError(f"manifest record index mismatch for {sequence_id}")
        if str(record["split"]) != expected_split:
            raise ValueError(f"manifest split/index mismatch for {sequence_id}")
        if int(record["actual_seed"]) != expected_seed:
            raise ValueError(f"manifest actual_seed mismatch for {sequence_id}")
        if record["dynamics_variant"] != expected_variant:
            raise ValueError(f"manifest dynamics variant mismatch for {sequence_id}")
        config = record["generator_config"]
        if not isinstance(config, Mapping) or set(config) != _CONFIG_FIELDS:
            raise ValueError(f"manifest record {sequence_id} lacks full generator_config")
        if (
            config["family"] != family
            or int(config["seed"]) != expected_seed
            or int(config["sequence_length"]) != int(record["sequence_length"])
            or config["dynamics_variant"] != expected_variant
        ):
            raise ValueError(f"manifest generator_config mismatch for {sequence_id}")
    if list(manifest["families"]) != _ordered_unique(record["family"] for record in records):
        raise ValueError("manifest family identity mismatch")
    if list(manifest["base_seeds"]) != _ordered_unique(record["base_seed"] for record in records):
        raise ValueError("manifest base_seed identity mismatch")
    expected_pooled, expected_lofo = _expected_identities(records)
    if manifest["pooled_identity"] != expected_pooled:
        raise ValueError("manifest pooled identity mismatch")
    if manifest["lofo_identities"] != expected_lofo:
        raise ValueError("manifest LOFO identity mismatch")


def _records_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sequence_id"]): row for row in manifest["sequence_records"]}


def recompute_validation_selection(
    point_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    held_out_family: str | None = None,
) -> str:
    """Recompute MLP-vs-TCN selection exclusively from validation raw rows."""

    validate_experiment_manifest(manifest)
    if held_out_family is None:
        scope, held_out = "pooled", ""
        expected_ids = set(manifest["pooled_identity"]["validation_sequence_ids"])
    else:
        if held_out_family not in manifest["lofo_identities"]:
            raise ValueError(f"unknown LOFO family: {held_out_family}")
        scope, held_out = "lofo", held_out_family
        expected_ids = set(
            manifest["lofo_identities"][held_out_family]["validation_sequence_ids"]
        )
    selected = [
        row
        for row in point_rows
        if row.get("scope") == scope
        and row.get("split") == "validation"
        and str(row.get("held_out_family", "")) == held_out
        and row.get("target") == "total_traffic"
        and row.get("method") in CANDIDATE_METHODS
    ]
    keyed: dict[tuple[str, str], float] = {}
    for row in selected:
        key = (str(row["sequence_id"]), str(row["method"]))
        if key in keyed:
            raise ValueError(f"duplicate validation raw identity: {key}")
        value = float(row["rmse"])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid validation RMSE for {key}")
        keyed[key] = value
    expected = {(sequence_id, method) for sequence_id in expected_ids for method in CANDIDATE_METHODS}
    if set(keyed) != expected:
        raise ValueError("validation raw identities are missing or incomplete")
    scores = {
        method: np.asarray([keyed[(sequence_id, method)] for sequence_id in sorted(expected_ids)])
        for method in CANDIDATE_METHODS
    }
    return select_recent_backbone(scores)


def _point_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("scope", "")),
        str(row.get("split", "")),
        str(row.get("held_out_family", "")),
        str(row.get("sequence_id", "")),
        str(row.get("method", "")),
        str(row.get("target", "")),
    )


def _previous_key(scope: str, held_out: str, sequence_id: str, target: str) -> str:
    return "::".join((scope, held_out, sequence_id, "previous_value", target))


def _expected_point_keys(manifest: Mapping[str, Any]) -> set[tuple[str, str, str, str, str, str]]:
    expected: set[tuple[str, str, str, str, str, str]] = set()
    for sequence_id in manifest["pooled_identity"]["validation_sequence_ids"]:
        for method in CANDIDATE_METHODS:
            expected.add(("pooled", "validation", "", sequence_id, method, "total_traffic"))
    for held_out, identity in manifest["lofo_identities"].items():
        for sequence_id in identity["validation_sequence_ids"]:
            for method in CANDIDATE_METHODS:
                expected.add(("lofo", "validation", held_out, sequence_id, method, "total_traffic"))
    for sequence_id in manifest["pooled_identity"]["test_sequence_ids"]:
        for method in POINT_METHODS:
            for target in CONTINUOUS_TARGETS:
                expected.add(("pooled", "test", "", sequence_id, method, target))
    for held_out, identity in manifest["lofo_identities"].items():
        for sequence_id in identity["test_sequence_ids"]:
            for method in POINT_METHODS:
                for target in CONTINUOUS_TARGETS:
                    expected.add(("lofo", "test", held_out, sequence_id, method, target))
    return expected


def validate_point_rows(
    point_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    validate_experiment_manifest(manifest)
    records = _records_by_id(manifest)
    keyed: dict[tuple[str, str, str, str, str, str], Mapping[str, Any]] = {}
    for row in point_rows:
        key = _point_key(row)
        if key in keyed:
            raise ValueError(f"duplicate point raw identity: {key}")
        keyed[key] = row
        sequence_id = key[3]
        if sequence_id not in records:
            raise ValueError(f"point raw has unknown sequence identity: {sequence_id}")
        record = records[sequence_id]
        if row.get("family") != record["family"] or int(row.get("base_seed")) != int(record["base_seed"]):
            raise ValueError(f"point raw provenance mismatch for {sequence_id}")
        rmse = float(row["rmse"])
        if not np.isfinite(rmse) or rmse < 0.0:
            raise ValueError(f"invalid point RMSE for {key}")
        if key[1] == "test":
            required = {
                "mae",
                "r2",
                "spearman",
                "delta_rmse",
                "hotspot_accuracy",
                "acf_lag1",
                "ess",
                "raw_step_count",
                "previous_row_key",
            }
            if required - set(row):
                raise ValueError(f"point test row missing metrics for {key}")
            for name in ("mae", "r2", "spearman", "delta_rmse", "hotspot_accuracy", "acf_lag1", "ess"):
                if not np.isfinite(float(row[name])):
                    raise ValueError(f"non-finite point metric {name} for {key}")
            if not 0.0 <= float(row["hotspot_accuracy"]) <= 1.0:
                raise ValueError(f"invalid hotspot accuracy for {key}")
            if float(row["ess"]) <= 0.0 or int(row["raw_step_count"]) <= 0:
                raise ValueError(f"invalid ESS/step count for {key}")
            expected_previous = _previous_key(key[0], key[2], key[3], key[5])
            if row["previous_row_key"] != expected_previous:
                raise ValueError(f"previous-value pairing identity mismatch for {key}")
    expected = _expected_point_keys(manifest)
    if set(keyed) != expected:
        missing = expected - set(keyed)
        extra = set(keyed) - expected
        raise ValueError(f"point raw identity incomplete; missing={len(missing)} extra={len(extra)}")

    for key, row in keyed.items():
        if key[1] != "test":
            continue
        previous = keyed[(key[0], "test", key[2], key[3], "previous_value", key[5])]
        expected_delta = float(previous["rmse"]) - float(row["rmse"])
        if not np.isclose(float(row["delta_rmse"]), expected_delta, rtol=0.0, atol=1e-12):
            raise ValueError(f"point delta mismatch for {key}")

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
    pooled_backbone = recompute_validation_selection(point_rows, manifest)
    backbones = {"": pooled_backbone}
    backbones.update(
        {
            held_out: recompute_validation_selection(
                point_rows, manifest, held_out_family=held_out
            )
            for held_out in manifest["lofo_identities"]
        }
    )
    for key, alias in keyed.items():
        if key[1] != "test" or key[4] != "selected_recent":
            continue
        backbone = keyed[(key[0], "test", key[2], key[3], backbones[key[2]], key[5])]
        if any(alias[name] != backbone[name] for name in alias_fields):
            raise ValueError(f"selected_recent alias/backbone mismatch for {key}")


def _probability_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scope", "")),
        str(row.get("held_out_family", "")),
        str(row.get("sequence_id", "")),
        str(row.get("target", "")),
    )


def _expected_probability_keys(manifest: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    expected = {
        ("pooled", "", sequence_id, target)
        for sequence_id in manifest["pooled_identity"]["test_sequence_ids"]
        for target in CONTINUOUS_TARGETS
    }
    for held_out, identity in manifest["lofo_identities"].items():
        expected.update(
            ("lofo", held_out, sequence_id, target)
            for sequence_id in identity["test_sequence_ids"]
            for target in CONTINUOUS_TARGETS
        )
    return expected


def validate_probability_rows(
    probability_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    validate_experiment_manifest(manifest)
    records = _records_by_id(manifest)
    keyed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    count_fields = (
        "coverage_numerator",
        "coverage_denominator",
        "scenario_coverage_numerator",
        "scenario_coverage_denominator",
        "tail_event_count",
        "tail_true_positive_count",
    )
    for row in probability_rows:
        key = _probability_key(row)
        if key in keyed:
            raise ValueError(f"duplicate probability raw identity: {key}")
        keyed[key] = row
        if key[2] not in records:
            raise ValueError(f"probability raw has unknown sequence identity: {key[2]}")
        record = records[key[2]]
        if row.get("family") != record["family"] or int(row.get("base_seed")) != int(record["base_seed"]):
            raise ValueError(f"probability raw provenance mismatch for {key}")
        try:
            counts = {name: int(row[name]) for name in count_fields}
            width = float(row["interval_width_sum"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"probability raw row missing/invalid counts for {key}") from error
        if any(value < 0 for value in counts.values()) or not np.isfinite(width) or width < 0.0:
            raise ValueError(f"negative/non-finite probability aggregate for {key}")
        if counts["coverage_denominator"] <= 0 or counts["scenario_coverage_denominator"] <= 0:
            raise ValueError(f"probability denominator must be positive for {key}")
        if counts["coverage_numerator"] > counts["coverage_denominator"]:
            raise ValueError(f"coverage numerator exceeds denominator for {key}")
        if counts["scenario_coverage_numerator"] > counts["scenario_coverage_denominator"]:
            raise ValueError(f"scenario numerator exceeds denominator for {key}")
        if counts["tail_true_positive_count"] > counts["tail_event_count"]:
            raise ValueError(f"tail true positives exceed events for {key}")
        if key[3] != "total_traffic" and (
            counts["tail_event_count"] or counts["tail_true_positive_count"]
        ):
            raise ValueError(f"tail counts are only valid for total_traffic: {key}")
    expected = _expected_probability_keys(manifest)
    if set(keyed) != expected:
        raise ValueError(
            f"probability raw identity incomplete; missing={len(expected - set(keyed))} "
            f"extra={len(set(keyed) - expected)}"
        )


def _probability_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage_denominator = sum(int(row["coverage_denominator"]) for row in rows)
    scenario_denominator = sum(int(row["scenario_coverage_denominator"]) for row in rows)
    events = sum(int(row["tail_event_count"]) for row in rows)
    true_positives = sum(int(row["tail_true_positive_count"]) for row in rows)
    coverage = sum(int(row["coverage_numerator"]) for row in rows) / coverage_denominator
    scenario = (
        sum(int(row["scenario_coverage_numerator"]) for row in rows)
        / scenario_denominator
    )
    result = {
        "coverage": float(coverage),
        "coverage_numerator": int(
            sum(int(row["coverage_numerator"]) for row in rows)
        ),
        "coverage_denominator": int(coverage_denominator),
        "mean_interval_width": float(
            sum(float(row["interval_width_sum"]) for row in rows)
            / coverage_denominator
        ),
        "scenario_coverage": float(scenario),
        "scenario_coverage_numerator": int(
            sum(int(row["scenario_coverage_numerator"]) for row in rows)
        ),
        "scenario_coverage_denominator": int(scenario_denominator),
        "tail_event_count": int(events),
        "tail_true_positive_count": int(true_positives),
        "tail_recall": float(true_positives / events) if events else None,
        "tail_status": "ok" if events >= 10 else "insufficient_events",
    }
    return result


def _recompute_summary(
    point_rows: Sequence[Mapping[str, Any]],
    probability_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    require_formal: bool,
) -> dict[str, Any]:
    validate_point_rows(point_rows, manifest)
    validate_probability_rows(probability_rows, manifest)
    records = list(manifest["sequence_records"])
    families = list(manifest["families"])
    base_seeds = [int(value) for value in manifest["base_seeds"]]
    if require_formal:
        split_counts = {
            split: sum(record["split"] == split for record in records)
            for split in ("fit", "validation", "calibration", "test")
        }
        if (
            len(records) != 75
            or families != list(FORMAL_FAMILIES)
            or base_seeds != list(FORMAL_BASE_SEEDS)
            or split_counts
            != {"fit": 30, "validation": 15, "calibration": 15, "test": 15}
        ):
            raise ValueError("formal summary requires exact 75-sequence identity coverage")

    pooled_selected = [
        row
        for row in point_rows
        if row["scope"] == "pooled"
        and row["split"] == "test"
        and row["method"] == "selected_recent"
    ]
    primary_ci: dict[str, dict[str, float]] = {}
    for target in PRIMARY_TARGETS:
        rows = [row for row in pooled_selected if row["target"] == target]
        bootstrap = family_stratified_paired_bootstrap(
            np.asarray([float(row["delta_rmse"]) for row in rows]),
            np.asarray([row["family"] for row in rows], dtype=object),
            samples=10_000,
            seed=20260731,
        )
        primary_ci[target] = {
            "mean": bootstrap.mean,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        }
    total_rows = [row for row in pooled_selected if row["target"] == "total_traffic"]
    seed_means = {
        str(seed): float(
            np.mean(
                [float(row["delta_rmse"]) for row in total_rows if int(row["base_seed"]) == seed]
            )
        )
        for seed in base_seeds
    }
    family_means = {
        family: float(
            np.mean(
                [float(row["delta_rmse"]) for row in total_rows if row["family"] == family]
            )
        )
        for family in families
    }
    lofo_family_means: dict[str, float] = {}
    lofo_relative: dict[str, float] = {}
    for family in families:
        selected = [
            row
            for row in point_rows
            if row["scope"] == "lofo"
            and row["held_out_family"] == family
            and row["split"] == "test"
            and row["method"] == "selected_recent"
            and row["target"] == "total_traffic"
        ]
        previous = [
            row
            for row in point_rows
            if row["scope"] == "lofo"
            and row["held_out_family"] == family
            and row["split"] == "test"
            and row["method"] == "previous_value"
            and row["target"] == "total_traffic"
        ]
        lofo_family_means[family] = float(
            np.mean([float(row["delta_rmse"]) for row in selected])
        )
        lofo_relative[family] = float(
            np.mean([float(row["rmse"]) for row in selected])
            / np.mean([float(row["rmse"]) for row in previous])
            - 1.0
        )

    pooled_probability_rows = [
        row
        for row in probability_rows
        if row["scope"] == "pooled" and row["target"] == "total_traffic"
    ]
    pooled_probability = _probability_summary(pooled_probability_rows)
    family_calibration_errors = {}
    for family in families:
        family_probability = _probability_summary(
            [row for row in pooled_probability_rows if row["family"] == family]
        )
        family_calibration_errors[family] = abs(
            float(family_probability["coverage"]) - 0.80
        )

    exact_families = set(families) == set(FORMAL_FAMILIES)
    exact_seeds = set(base_seeds) == set(FORMAL_BASE_SEEDS)
    conditions = {
        "condition_1": primary_ci["total_traffic"]["lower"] > 0.0,
        "condition_2": exact_seeds
        and all(value > 0.0 for value in seed_means.values())
        and exact_families
        and sum(value > 0.0 for value in family_means.values()) >= 4,
        "condition_3": all(primary_ci[target]["lower"] > 0.0 for target in PRIMARY_TARGETS),
        "condition_4": exact_families
        and float(np.mean(list(lofo_family_means.values()))) >= 0.0
        and sum(value > 0.0 for value in lofo_family_means.values()) >= 3
        and sum(value > 0.10 for value in lofo_relative.values()) <= 1,
        "condition_5": exact_families
        and abs(float(pooled_probability["coverage"]) - 0.80) <= 0.05
        and all(value <= 0.10 for value in family_calibration_errors.values())
        and abs(float(pooled_probability["scenario_coverage"]) - 0.80) <= 0.10,
        "condition_6": int(pooled_probability["tail_event_count"]) >= 10
        and pooled_probability["tail_recall"] is not None
        and float(pooled_probability["tail_recall"]) >= 0.70,
    }
    summary = {
        "schema_version": 2,
        "gate_status": "PENDING_SUPERVISOR",
        "pipeline_trace": list(PIPELINE_TRACE),
        "selected_backbone": recompute_validation_selection(point_rows, manifest),
        "lofo_selected_backbones": {
            family: recompute_validation_selection(
                point_rows, manifest, held_out_family=family
            )
            for family in families
        },
        "calibration_sequence_ids": list(
            manifest["pooled_identity"]["calibration_sequence_ids"]
        ),
        "identity_counts": {
            "families": len(families),
            "base_seeds": len(base_seeds),
            "pooled_test_sequences": len(
                manifest["pooled_identity"]["test_sequence_ids"]
            ),
            "lofo_test_sequences": sum(
                len(identity["test_sequence_ids"])
                for identity in manifest["lofo_identities"].values()
            ),
            "point_methods": len(POINT_METHODS),
            "continuous_targets": len(CONTINUOUS_TARGETS),
        },
        "primary_delta_ci": primary_ci,
        "seed_mean_deltas": seed_means,
        "family_mean_deltas": family_means,
        "lofo": {
            "aggregate_mean_delta": float(np.mean(list(lofo_family_means.values()))),
            "family_mean_deltas": lofo_family_means,
            "relative_rmse_changes": lofo_relative,
        },
        "probability": {
            "pooled": pooled_probability,
            "family_interval_calibration_errors": family_calibration_errors,
        },
        "conditions": {name: bool(value) for name, value in conditions.items()},
    }
    return _json_value(summary)


def recompute_formal_summary(
    point_rows: Sequence[Mapping[str, Any]],
    probability_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute all formal data conditions from the two raw tables + manifest."""

    return _recompute_summary(
        point_rows, probability_rows, manifest, require_formal=True
    )


def _stack_examples(
    examples: Mapping[str, HistoryExamples], sequence_ids: Sequence[str], field: str
) -> np.ndarray:
    if not sequence_ids:
        raise ValueError(f"cannot stack empty sequence identity for {field}")
    return np.concatenate(
        [np.asarray(getattr(examples[sequence_id], field)) for sequence_id in sequence_ids],
        axis=0,
    )


def _target_slices(num_nodes: int, group_count: int) -> dict[str, slice]:
    cursor = 0
    result = {"total_traffic": slice(cursor, cursor + 1)}
    cursor += 1
    result["source_load_vector"] = slice(cursor, cursor + num_nodes)
    cursor += num_nodes
    result["destination_load_vector"] = slice(cursor, cursor + num_nodes)
    cursor += num_nodes
    result["hotspot_strength"] = slice(cursor, cursor + 1)
    cursor += 1
    result["sparsity"] = slice(cursor, cursor + 1)
    cursor += 1
    result["bandwidth_group_offered_load_vector"] = slice(
        cursor, cursor + group_count
    )
    return result


def _blocks(values: np.ndarray, slices: Mapping[str, slice]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[:, target_slice]) for name, target_slice in slices.items()}


def _fit_suite(
    examples: Mapping[str, HistoryExamples], fit_ids: Sequence[str], seed: int
) -> HistoryPredictorSuite:
    recent = _stack_examples(examples, fit_ids, "recent_history")
    targets = _stack_examples(examples, fit_ids, "targets")
    suite = HistoryPredictorSuite(recent.shape[1], targets.shape[1], seed=seed)
    suite.fit(
        _stack_examples(examples, fit_ids, "moment_features"), recent, targets
    )
    return suite


def _suite_predictions(
    suite: HistoryPredictorSuite, example: HistoryExamples
) -> dict[str, np.ndarray]:
    return suite.predict(
        example.moment_features,
        example.recent_history,
        example.previous_targets,
        example.ewma_targets,
    )


def _run_scope(
    *,
    scope: str,
    held_out_family: str,
    identity: Mapping[str, Sequence[str]],
    records: Mapping[str, Mapping[str, Any]],
    examples: Mapping[str, HistoryExamples],
    target_slices: Mapping[str, slice],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    fit_ids = list(identity["fit_sequence_ids"])
    validation_ids = list(identity["validation_sequence_ids"])
    calibration_ids = list(identity["calibration_sequence_ids"])
    test_ids = list(identity["test_sequence_ids"])
    suite = _fit_suite(examples, fit_ids, seed)

    point_rows: list[dict[str, Any]] = []
    validation_scores: dict[str, list[float]] = {
        method: [] for method in CANDIDATE_METHODS
    }
    for sequence_id in validation_ids:
        example = examples[sequence_id]
        prediction = _suite_predictions(suite, example)
        record = records[sequence_id]
        for method in CANDIDATE_METHODS:
            rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            prediction[method][:, target_slices["total_traffic"]]
                            - example.targets[:, target_slices["total_traffic"]]
                        )
                        ** 2
                    )
                )
            )
            validation_scores[method].append(rmse)
            point_rows.append(
                {
                    "scope": scope,
                    "split": "validation",
                    "held_out_family": held_out_family,
                    "sequence_id": sequence_id,
                    "family": record["family"],
                    "base_seed": record["base_seed"],
                    "method": method,
                    "target": "total_traffic",
                    "rmse": rmse,
                }
            )
    selected_backbone = select_recent_backbone(
        {name: np.asarray(values) for name, values in validation_scores.items()}
    )

    calibration_points: list[np.ndarray] = []
    calibration_residuals: list[np.ndarray] = []
    for sequence_id in calibration_ids:
        example = examples[sequence_id]
        point = _suite_predictions(suite, example)[selected_backbone]
        calibration_points.append(point)
        calibration_residuals.append(example.targets - point)
    point_stack = np.concatenate(calibration_points, axis=0)
    residual_stack = np.concatenate(calibration_residuals, axis=0)
    calibrator = ResidualCalibrator(seed=seed, scenario_count=64).fit(
        point_stack, residual_stack
    )
    fit_total = _stack_examples(examples, fit_ids, "targets")[:, 0]
    fit_tail_threshold = float(np.quantile(fit_total, 0.90, method="linear"))

    probability_rows: list[dict[str, Any]] = []
    stable_cursor = 0
    for sequence_id in test_ids:
        example = examples[sequence_id]
        record = records[sequence_id]
        predictions = _suite_predictions(suite, example)
        selected_point = predictions[selected_backbone]
        stable_indices = np.arange(
            stable_cursor, stable_cursor + len(example.targets), dtype=np.int64
        )
        stable_cursor += len(example.targets)
        calibrated = calibrator.predict(
            selected_point, stable_example_indices=stable_indices
        )
        predictions["quantile_scenario"] = calibrated.q50
        # This is a literal derived alias, never an independently optimized row.
        predictions["selected_recent"] = predictions[selected_backbone]
        truth_blocks = _blocks(example.targets, target_slices)
        method_metrics: dict[str, dict[str, dict[str, float]]] = {}
        hotspot_accuracy: dict[str, float] = {}
        destination_slice = target_slices["destination_load_vector"]
        actual_hotspot = hotspot_from_destination_loads(
            example.targets[:, destination_slice]
        )
        for method in POINT_METHODS:
            predicted_blocks = _blocks(predictions[method], target_slices)
            method_metrics[method] = sequence_target_metrics(
                truth_blocks, predicted_blocks
            )
            predicted_hotspot = hotspot_from_destination_loads(
                predictions[method][:, destination_slice]
            )
            hotspot_accuracy[method] = float(
                np.mean(predicted_hotspot == actual_hotspot)
            )
        total_actual = truth_blocks["total_traffic"].reshape(-1)
        acf_lag1 = float(autocorrelation(total_actual, max_lag=1)[1])
        ess = float(positive_sequence_ess(total_actual, max_lag=64))
        for method in POINT_METHODS:
            for target in CONTINUOUS_TARGETS:
                metrics = method_metrics[method][target]
                previous_rmse = method_metrics["previous_value"][target]["rmse"]
                point_rows.append(
                    {
                        "scope": scope,
                        "split": "test",
                        "held_out_family": held_out_family,
                        "sequence_id": sequence_id,
                        "family": record["family"],
                        "base_seed": record["base_seed"],
                        "method": method,
                        "target": target,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "spearman": metrics["spearman"],
                        "delta_rmse": previous_rmse - metrics["rmse"],
                        "hotspot_accuracy": hotspot_accuracy[method],
                        "acf_lag1": acf_lag1,
                        "ess": ess,
                        "raw_step_count": len(example.targets),
                        "previous_row_key": _previous_key(
                            scope, held_out_family, sequence_id, target
                        ),
                    }
                )

        for target, target_slice in target_slices.items():
            truth = example.targets[:, target_slice]
            lower = calibrated.q10[:, target_slice]
            upper = calibrated.q90[:, target_slice]
            scenario_values = calibrated.scenarios[:, :, target_slice]
            scenario_lower = np.quantile(
                scenario_values, 0.10, axis=1, method="linear"
            )
            scenario_upper = np.quantile(
                scenario_values, 0.90, axis=1, method="linear"
            )
            events = np.zeros(truth.shape, dtype=bool)
            true_positives = np.zeros(truth.shape, dtype=bool)
            if target == "total_traffic":
                events = truth > fit_tail_threshold
                true_positives = events & (upper > fit_tail_threshold)
            probability_rows.append(
                {
                    "scope": scope,
                    "held_out_family": held_out_family,
                    "sequence_id": sequence_id,
                    "family": record["family"],
                    "base_seed": record["base_seed"],
                    "target": target,
                    "coverage_numerator": int(
                        np.sum((truth >= lower) & (truth <= upper))
                    ),
                    "coverage_denominator": int(truth.size),
                    "interval_width_sum": float(np.sum(upper - lower)),
                    "scenario_coverage_numerator": int(
                        np.sum(
                            (truth >= scenario_lower) & (truth <= scenario_upper)
                        )
                    ),
                    "scenario_coverage_denominator": int(truth.size),
                    "tail_event_count": int(events.sum()),
                    "tail_true_positive_count": int(true_positives.sum()),
                }
            )
    return point_rows, probability_rows, selected_backbone


def _execute_experiment(
    *,
    manifest: Mapping[str, Any],
    sequences: Mapping[str, Any],
    group_coefficients: np.ndarray,
    output_dir: str | Path | None,
    require_formal: bool,
) -> dict[str, Any]:
    records = _records_by_id(manifest)
    if set(sequences) != set(records):
        raise ValueError("generated sequence identities do not match manifest")
    examples = {
        sequence_id: build_history_examples(
            sequence, group_coefficients=group_coefficients
        )
        for sequence_id, sequence in sequences.items()
    }
    first_sequence = next(iter(sequences.values()))
    num_nodes = int(np.asarray(first_sequence.matrices[0]).shape[0])
    target_slices = _target_slices(num_nodes, int(group_coefficients.shape[0]))
    point_rows, probability_rows, _ = _run_scope(
        scope="pooled",
        held_out_family="",
        identity=manifest["pooled_identity"],
        records=records,
        examples=examples,
        target_slices=target_slices,
        seed=20260731,
    )
    for held_out, identity in manifest["lofo_identities"].items():
        fold_point, fold_probability, _ = _run_scope(
            scope="lofo",
            held_out_family=held_out,
            identity=identity,
            records=records,
            examples=examples,
            target_slices=target_slices,
            seed=20260731,
        )
        point_rows.extend(fold_point)
        probability_rows.extend(fold_probability)
    summary = _recompute_summary(
        point_rows,
        probability_rows,
        manifest,
        require_formal=require_formal,
    )
    result = {
        "manifest": _json_value(manifest),
        "point_rows": _json_value(point_rows),
        "probability_rows": _json_value(probability_rows),
        "summary": summary,
    }
    if output_dir is not None:
        write_experiment_artifacts(output_dir, **result)
    return result


_POINT_FIELDS = (
    "scope",
    "split",
    "held_out_family",
    "sequence_id",
    "family",
    "base_seed",
    "method",
    "target",
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
_PROBABILITY_FIELDS = (
    "scope",
    "held_out_family",
    "sequence_id",
    "family",
    "base_seed",
    "target",
    "coverage_numerator",
    "coverage_denominator",
    "interval_width_sum",
    "scenario_coverage_numerator",
    "scenario_coverage_denominator",
    "tail_event_count",
    "tail_true_positive_count",
)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_value(value),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields_: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields_), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_artifacts(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    point_rows: Sequence[Mapping[str, Any]],
    probability_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    validate_point_rows(point_rows, manifest)
    validate_probability_rows(probability_rows, manifest)
    require_formal = len(manifest["sequence_records"]) == 75
    recomputed = _recompute_summary(
        point_rows, probability_rows, manifest, require_formal=require_formal
    )
    if _json_value(summary) != recomputed:
        raise ValueError("summary does not equal raw+manifest recomputation")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "manifest.json", manifest)
    _write_csv(directory / "raw_sequence_metrics.csv", point_rows, _POINT_FIELDS)
    _write_csv(
        directory / "raw_probability_metrics.csv",
        probability_rows,
        _PROBABILITY_FIELDS,
    )
    _write_json(directory / "summary.json", recomputed)


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _toy_specs() -> list[SequenceSpec]:
    result: list[SequenceSpec] = []
    for spec in build_formal_sequence_specs():
        if spec.family not in FORMAL_FAMILIES[:2] or spec.base_seed != FORMAL_BASE_SEEDS[0]:
            continue
        config = dict(spec.generator_config)
        config["sequence_length"] = 32
        result.append(
            replace(
                spec,
                sequence_length=32,
                generator_config=config,
            )
        )
    return result


def _toy_sequence(spec: SequenceSpec) -> Any:
    rng = np.random.default_rng(spec.actual_seed)
    matrices: list[np.ndarray] = []
    state = int(spec.sequence_index + spec.family_index)
    for step in range(spec.sequence_length):
        state = (state + int(rng.integers(0, 2)) + step % 2) % 5
        matrix = np.zeros((4, 4), dtype=np.int64)
        for source in range(4):
            for destination in range(4):
                if source == destination:
                    continue
                lag = step - (2 if spec.family_index == 0 else 3)
                matrix[source, destination] = int(
                    (
                        state
                        + max(lag, 0)
                        + source * 2
                        + destination * 3
                        + spec.sequence_index
                    )
                    % 6
                )
        matrices.append(matrix)
    return SimpleNamespace(
        sequence_id=spec.sequence_id,
        family=spec.family,
        seed=spec.actual_seed,
        matrices=matrices,
        metadata={},
    )


def run_toy_experiment(
    *, output_dir: str | Path | None = None, seed: int = 20260731
) -> dict[str, Any]:
    """Run a real, small four-split + LOFO pipeline entirely in memory."""

    if int(seed) != 20260731:
        raise ValueError("toy H1 uses the same frozen seed 20260731")
    specs = _toy_specs()
    sequences = {spec.sequence_id: _toy_sequence(spec) for spec in specs}
    records = [
        {
            "sequence_id": spec.sequence_id,
            "split": spec.split,
            "digest": sequence_digest(sequences[spec.sequence_id].matrices),
        }
        for spec in specs
    ]
    # Exercise every frozen target block end to end, including bandwidth-group
    # offered load, while keeping the toy topology deterministic and minimal.
    group_coefficients = np.ones((1, 4, 4), dtype=np.float64)
    np.fill_diagonal(group_coefficients[0], 0.0)
    manifest = build_experiment_manifest(
        specs=specs,
        sequence_records=records,
        protocol={"path": "toy/H1_PREDICTABILITY_PROTOCOL", "sha256": "1" * 64},
        source={"kind": "deterministic_toy", "sha256": "2" * 64},
        command=["run_toy_experiment", "--seed", "20260731"],
        environment={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": importlib.metadata.version("scikit-learn"),
        },
        topology={"name": "ToyComplete4", "sha256": "3" * 64},
        group_coefficients_digest=_array_digest(group_coefficients),
    )
    return _execute_experiment(
        manifest=manifest,
        sequences=sequences,
        group_coefficients=group_coefficients,
        output_dir=output_dir,
        require_formal=False,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest(project_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((project_root / "rlccl" / "prediction").glob("*.py"))
    runner = project_root / "scripts" / "run_h1_predictability.py"
    if runner.exists():
        files.append(runner)
    for path in files:
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_formal_experiment(output_dir: str | Path = "outputs/h1_predictability") -> dict[str, Any]:
    """Run exactly the frozen 75-sequence formal design and write schema-2 evidence."""

    from rlccl.envs.evaluator import load_topology_info

    project_root = Path(__file__).resolve().parents[2]
    specs = build_formal_sequence_specs()
    if len(specs) != 75:
        raise AssertionError("formal H1 topology must contain exactly 75 specs")
    topology = load_topology_info("Rear4GPU")
    group_coefficients = deterministic_group_coefficients(topology)
    sequences: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for spec in specs:
        sequence = generate_formal_sequence(spec)
        sequences[spec.sequence_id] = sequence
        records.append(
            {
                "sequence_id": spec.sequence_id,
                "split": spec.split,
                "digest": sequence_digest(sequence.matrices),
            }
        )
    protocol_path = project_root / "docs" / "uncertainty_aiccl" / "H1_PREDICTABILITY_PROTOCOL.md"
    topology_path = project_root / "Data" / "Rear4GPU" / "Topology" / "pipeline_topology_no_switch.json"
    manifest = build_experiment_manifest(
        specs=specs,
        sequence_records=records,
        protocol={
            "path": protocol_path.relative_to(project_root).as_posix(),
            "sha256": _file_digest(protocol_path),
        },
        source={"kind": "workspace_tree", "sha256": _source_digest(project_root)},
        command=list(sys.argv),
        environment={
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "sklearn": importlib.metadata.version("scikit-learn"),
            "platform": platform.platform(),
        },
        topology={"name": "Rear4GPU", "sha256": _file_digest(topology_path)},
        group_coefficients_digest=_array_digest(group_coefficients),
    )
    return _execute_experiment(
        manifest=manifest,
        sequences=sequences,
        group_coefficients=group_coefficients,
        output_dir=output_dir,
        require_formal=True,
    )
