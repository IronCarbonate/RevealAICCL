"""Validated, reproducible artifact IO for Gate H1 evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import FORMAL_BASE_SEEDS, FORMAL_FAMILIES, SequenceSpec, validate_split_records
from .models import METHOD_NAMES


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


def validate_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple, np.ndarray)):
        for index, item in enumerate(value):
            validate_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"NaN/Inf non-finite value at {path}")


def build_manifest(
    *, specs: Sequence[SequenceSpec], sequence_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    split_counts = {
        split: sum(spec.split == split for spec in specs)
        for split in ("fit", "validation", "calibration", "test")
    }
    manifest = {
        "schema_version": 1,
        "protocol": "Gate H1 history predictability",
        "topology": "Rear4GPU",
        "sequence_length": 1024,
        "families": list(FORMAL_FAMILIES),
        "base_seeds": list(FORMAL_BASE_SEEDS),
        "split_counts": split_counts,
        "sequence_records": list(sequence_records),
        "generator_config": dict(specs[0].generator_config) if specs else {},
        "methods": list(METHOD_NAMES),
        "random_seeds": {"model_shuffle_scenario_bootstrap": 20260731},
        "model_hyperparameters": {
            "ewma": {"alpha": 0.30},
            "moment_only": {"alpha": 10.0, "maximum_window": 16},
            "recent_history_mlp": {
                "recent_steps": 8,
                "hidden_layer_sizes": [32],
                "activation": "tanh",
                "solver": "adam",
                "alpha": 1e-4,
                "batch_size": 256,
                "learning_rate_init": 1e-3,
                "max_iter": 80,
                "early_stopping": False,
            },
            "causal_tcn": {
                "recent_steps": 8,
                "kernel_size": 3,
                "hidden_channels": 8,
                "epochs": 40,
                "batch_size": 256,
                "learning_rate": 5e-3,
                "l2": 1e-4,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
            "quantile_scenario": {
                "quantiles": [0.10, 0.50, 0.90],
                "scenario_count": 64,
                "quantile_method": "linear",
            },
        },
    }
    return _json_value(manifest)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "split_counts",
        "sequence_records",
        "generator_config",
        "methods",
        "random_seeds",
        "model_hyperparameters",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing keys: {sorted(missing)}")
    if int(manifest["schema_version"]) != 1:
        raise ValueError("unsupported manifest schema_version")
    validate_split_records(manifest["sequence_records"])
    validate_finite_tree(manifest)


_RAW_REQUIRED = {
    "sequence_id",
    "family",
    "base_seed",
    "method",
    "target",
    "rmse",
    "mae",
    "r2",
    "spearman",
    "delta_rmse",
    "raw_step_count",
}
_FLOAT_FIELDS = ("rmse", "mae", "r2", "spearman", "delta_rmse")
_INT_FIELDS = ("base_seed", "raw_step_count")


def validate_raw_rows(raw_rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(raw_rows)
    if not rows:
        raise ValueError("raw rows must not be empty")
    identities: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        missing = _RAW_REQUIRED - set(row)
        if missing:
            raise ValueError(f"raw row {index} missing keys: {sorted(missing)}")
        for name in _FLOAT_FIELDS:
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"NaN/non-finite {name} in raw row {index}")
        if int(row["raw_step_count"]) <= 0:
            raise ValueError("raw_step_count must be positive")
        identity = (str(row["sequence_id"]), str(row["method"]), str(row["target"]))
        if identity in identities:
            raise ValueError(f"duplicate raw row: {identity}")
        identities.add(identity)


def recompute_summary(raw_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_raw_rows(raw_rows)
    rows = list(raw_rows)
    sequence_steps: dict[str, int] = {}
    for row in rows:
        sequence_id = str(row["sequence_id"])
        steps = int(row["raw_step_count"])
        if sequence_id in sequence_steps and sequence_steps[sequence_id] != steps:
            raise ValueError(f"inconsistent raw_step_count for {sequence_id}")
        sequence_steps[sequence_id] = steps
    methods: dict[str, dict[str, dict[str, float]]] = {}
    for method in dict.fromkeys(str(row["method"]) for row in rows):
        methods[method] = {}
        targets = dict.fromkeys(
            str(row["target"]) for row in rows if str(row["method"]) == method
        )
        for target in targets:
            selected = [
                row
                for row in rows
                if str(row["method"]) == method and str(row["target"]) == target
            ]
            methods[method][target] = {
                f"mean_{field}": float(np.mean([float(row[field]) for row in selected]))
                for field in (*_FLOAT_FIELDS,)
            }
    result = {
        "schema_version": 1,
        "independent_test_sequences": len(sequence_steps),
        "raw_test_steps": int(sum(sequence_steps.values())),
        "sequence_ids": sorted(sequence_steps),
        "methods": methods,
    }
    validate_finite_tree(result)
    return result


def write_artifact_bundle(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    validate_manifest(manifest)
    validate_raw_rows(raw_rows)
    recomputed = recompute_summary(raw_rows)
    if _json_value(summary) != recomputed:
        raise ValueError("summary does not match recomputation from raw rows")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_value(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
    fieldnames = list(raw_rows[0].keys())
    with (directory / "raw_sequence_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_json_value(raw_rows))
    with (directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(recomputed, handle, indent=2, sort_keys=True, allow_nan=False)


def read_artifact_bundle(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (directory / "raw_sequence_metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        raw_rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    for row in raw_rows:
        for field in _FLOAT_FIELDS:
            row[field] = float(row[field])
        for field in _INT_FIELDS:
            row[field] = int(row[field])
    with (directory / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    validate_manifest(manifest)
    validate_raw_rows(raw_rows)
    validate_finite_tree(summary)
    return {"manifest": manifest, "raw_rows": raw_rows, "summary": summary}
