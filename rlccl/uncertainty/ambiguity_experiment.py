"""Frozen Phase 3B experiment definitions, metrics, and artifact utilities.

Importing this module is side-effect free.  In particular, it never generates
the formal corpus and never writes the formal output directory.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence
import uuid

import numpy as np

from rlccl.traffic.long_horizon_generator import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
)

from rlccl.envs.problem import TopologyInfo

from .ambiguity import (
    AmbiguityConstructionView,
    build_empirical_ambiguity_set,
    fit_descriptor_normalizer,
    group_coefficients_digest,
    oracle_support_upper_bound,
    physical_descriptor_bounds,
    select_support,
    traffic_descriptor,
    truth_nearest_descriptor_distance,
)
from .observation import PartialObservationState, PublicTopologyView


PROTOCOL_SHA256 = "7E01108E362973461B5E676CF163A491D7E90E5D30D40AE0356CA83D6680D7A3"
FORMAL_FAMILIES = tuple(LONG_HORIZON_FAMILIES)
FORMAL_BASE_SEEDS = (342, 442, 542)
FORMAL_SPLITS = ("fit", "fit", "validation", "calibration", "test")
FORMAL_SEQUENCE_LENGTH = 1024
FORMAL_MAX_ENTRY = 8
HISTORY_WINDOW = 32
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
VALIDATION_TIE_ORDER = (
    "minimax_subset",
    "boundary_scenarios",
    "worst_recent_cases",
    "random_empirical",
)
RANDOM_REPLICATES = 8
FORMAL_OUTPUT_DIRECTORY = Path("outputs/phase3b_ambiguity")
ARTIFACT_NAMES = (
    "manifest.json",
    "raw_calibration_scores.csv",
    "raw_validation_metrics.csv",
    "raw_case_metrics.csv",
    "raw_sequence_metrics.csv",
    "raw_lofo_calibration_scores.csv",
    "raw_lofo_validation_metrics.csv",
    "raw_lofo_test_metrics.csv",
    "raw_dependence_metrics.csv",
    "summary.json",
)


def _row_value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    numeric = _finite_float(value, name)
    if numeric != math.floor(numeric):
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class _AttributeMapping(Mapping[str, Any]):
    """Dataclass mixin that remains JSON-normalizable as a Mapping."""

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    sequence_id: str
    family: str
    family_index: int
    base_seed: int
    seed_index: int
    sequence_index: int
    actual_seed: int
    split: str
    same_moments_variant: str | None
    sequence_length: int
    generator_config: dict[str, Any]
    record_index: int


def _generator_config(
    family: str,
    actual_seed: int,
    variant: str | None,
) -> dict[str, Any]:
    return asdict(
        LongHorizonTrafficConfig(
            num_nodes=4,
            sequence_length=FORMAL_SEQUENCE_LENGTH,
            family=family,
            seed=actual_seed,
            mean_level=2.0,
            std_level=1.5,
            max_entry=FORMAL_MAX_ENTRY,
            calibration_candidates=1,
            topology_name="Rear4GPU",
            dynamics_variant=variant,
        )
    )


def build_formal_sequence_specs() -> tuple[SequenceSpec, ...]:
    """Return all 75 immutable specifications without generating traffic."""

    specs: list[SequenceSpec] = []
    for family_index, family in enumerate(FORMAL_FAMILIES):
        for seed_index, base_seed in enumerate(FORMAL_BASE_SEEDS):
            for sequence_index, split in enumerate(FORMAL_SPLITS):
                actual_seed = (
                    base_seed + family_index * 1_000_000 + sequence_index * 10_000
                )
                variant = (
                    SAME_MOMENT_VARIANTS[(seed_index + sequence_index) % 4]
                    if family == "same_moments_different_dynamics"
                    else None
                )
                record_index = len(specs)
                specs.append(
                    SequenceSpec(
                        sequence_id=(
                            f"{family}-base{base_seed}-sequence{sequence_index}"
                            f"-seed{actual_seed}"
                        ),
                        family=family,
                        family_index=family_index,
                        base_seed=base_seed,
                        seed_index=seed_index,
                        sequence_index=sequence_index,
                        actual_seed=actual_seed,
                        split=split,
                        same_moments_variant=variant,
                        sequence_length=FORMAL_SEQUENCE_LENGTH,
                        generator_config=_generator_config(family, actual_seed, variant),
                        record_index=record_index,
                    )
                )
    return tuple(specs)


def validate_sequence_records(
    records: Sequence[Mapping[str, Any]],
    *,
    h1_sequence_digests: set[str] | frozenset[str],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(records)
    if len(rows) != 75:
        raise ValueError("formal sequence record universe must contain 75 records")
    identifiers = [str(row["sequence_id"]) for row in rows]
    digests = [str(row["sequence_digest"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sequence IDs must be unique; duplicate/overlap detected")
    if len(set(digests)) != len(digests):
        raise ValueError("sequence digest must be unique; duplicate/overlap detected")
    overlap = set(digests) & {str(value) for value in h1_sequence_digests}
    if overlap:
        raise ValueError("H1 sequence digest overlap is forbidden")
    split_ids: dict[str, set[str]] = defaultdict(set)
    split_digests: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        sequence_id = str(row["sequence_id"])
        digest = str(row["sequence_digest"])
        if any(sequence_id in values for key, values in split_ids.items() if key != split):
            raise ValueError("sequence split overlap detected")
        if any(digest in values for key, values in split_digests.items() if key != split):
            raise ValueError("digest split overlap detected")
        split_ids[split].add(sequence_id)
        split_digests[split].add(digest)
    return rows


def reveal_seed(record_index: int, checkpoint: int, mode_index: int) -> int:
    return 31_000_000 + int(record_index) * 100_000 + int(checkpoint) * 10 + int(mode_index)


def replicate_seed(case_index: int, replicate_index: int) -> int:
    replicate = int(replicate_index)
    if replicate < 0 or replicate >= RANDOM_REPLICATES:
        raise ValueError("replicate index must be in [0,7]")
    return 41_000_000 + int(case_index) * 100 + replicate


@dataclass(frozen=True, slots=True)
class CaseRegistryEntry:
    case_index: int
    split: str
    record_index: int
    checkpoint: int
    mode_index: int
    stage_index: int
    requested_k: int
    reveal_mode: str
    reveal_ratio: float
    reveal_seed: int
    construction_seed: int


def build_case_registry(specs: Sequence[Any]) -> tuple[CaseRegistryEntry, ...]:
    """Build the 72,000-case registry; fit specs never enter evaluation."""

    identities: list[tuple[str, int, int, int, int, int]] = []
    for spec in specs:
        split = str(_row_value(spec, "split"))
        if split == "fit":
            continue
        if split not in {"validation", "calibration", "test"}:
            raise ValueError(f"unsupported evaluated split: {split}")
        record_index = int(_row_value(spec, "record_index"))
        for checkpoint in CHECKPOINTS:
            for mode_index in range(len(REVEAL_MODES)):
                for stage_index in range(len(REVEAL_RATIOS)):
                    for requested_k in REQUESTED_K:
                        identities.append(
                            (
                                split,
                                record_index,
                                checkpoint,
                                mode_index,
                                stage_index,
                                requested_k,
                            )
                        )
    identities.sort()
    if len(identities) != 72_000 or len(set(identities)) != 72_000:
        raise ValueError("case registry must contain exactly 72,000 unique identities")
    return tuple(
        CaseRegistryEntry(
            case_index=case_index,
            split=identity[0],
            record_index=identity[1],
            checkpoint=identity[2],
            mode_index=identity[3],
            stage_index=identity[4],
            requested_k=identity[5],
            reveal_mode=REVEAL_MODES[identity[3]],
            reveal_ratio=REVEAL_RATIOS[identity[4]],
            reveal_seed=reveal_seed(identity[1], identity[2], identity[3]),
            construction_seed=reveal_seed(identity[1], identity[2], identity[3]),
        )
        for case_index, identity in enumerate(identities)
    )


def _case_index_lookup(
    specs: Sequence[Any],
) -> dict[tuple[str, int, int, int, int, int], int]:
    return {
        (
            row.split,
            row.record_index,
            row.checkpoint,
            row.mode_index,
            row.stage_index,
            row.requested_k,
        ): row.case_index
        for row in build_case_registry(specs)
    }


def calibration_exceedance_score(
    raw_lower: Any,
    raw_upper: Any,
    descriptor: Any,
    fit_scale: Any,
) -> float:
    lower = np.asarray(raw_lower, dtype=np.float64)
    upper = np.asarray(raw_upper, dtype=np.float64)
    values = np.asarray(descriptor, dtype=np.float64)
    scale = np.asarray(fit_scale, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape or values.shape != lower.shape:
        raise ValueError("calibration descriptor/bound shape mismatch")
    if scale.shape != lower.shape or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("calibration scale must be finite and positive")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or not np.isfinite(values).all():
        raise ValueError("calibration inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("calibration lower bound exceeds upper bound")
    lower_violation = (lower - values) / scale
    upper_violation = (values - upper) / scale
    return float(np.max(np.maximum(np.maximum(lower_violation, upper_violation), 0.0)))


def _calibration_identity(row: Mapping[str, Any]) -> tuple[int, str, float]:
    return (
        _integer(row["checkpoint"], "checkpoint"),
        str(row["reveal_mode"]),
        _finite_float(row["reveal_ratio"], "reveal_ratio"),
    )


def calibrate_envelope_radius(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out_family: str | None = None,
) -> float:
    selected = [
        row for row in rows
        if held_out_family is None or str(row["family"]) != str(held_out_family)
    ]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["sequence_id"])].append(row)
    expected_sequence_count = 15 if held_out_family is None else 12
    if len(grouped) != expected_sequence_count:
        raise ValueError(
            f"calibration universe requires {expected_sequence_count} sequences"
        )
    expected_identities = {
        (checkpoint, mode, ratio)
        for checkpoint in CHECKPOINTS
        for mode in REVEAL_MODES
        for ratio in UNKNOWN_RATIOS
    }
    sequence_quantiles: list[float] = []
    for sequence_id in sorted(grouped):
        sequence_rows = grouped[sequence_id]
        identities = [_calibration_identity(row) for row in sequence_rows]
        if len(sequence_rows) != 320 or len(set(identities)) != 320:
            raise ValueError("calibration 320-case universe has duplicate/missing identity")
        if set(identities) != expected_identities:
            raise ValueError("calibration universe has unknown ratio or missing identity")
        scores = np.asarray(
            [_finite_float(row["score"], "calibration score") for row in sequence_rows],
            dtype=np.float64,
        )
        if np.any(scores < 0.0):
            raise ValueError("calibration scores must be nonnegative")
        sequence_quantiles.append(float(np.quantile(scores, 0.9, method="higher")))
    return float(np.quantile(sequence_quantiles, 0.9, method="higher"))


@dataclass(frozen=True, slots=True)
class LofoFold:
    held_out_family: str
    fit: tuple[Any, ...]
    validation: tuple[Any, ...]
    calibration: tuple[Any, ...]
    test: tuple[Any, ...]


def build_lofo_fold(specs: Sequence[Any], *, held_out_family: str) -> LofoFold:
    family = str(held_out_family)
    if family not in FORMAL_FAMILIES:
        raise ValueError("unknown held-out family")
    return LofoFold(
        held_out_family=family,
        fit=tuple(
            spec for spec in specs
            if _row_value(spec, "split") == "fit" and _row_value(spec, "family") != family
        ),
        validation=tuple(
            spec for spec in specs
            if _row_value(spec, "split") == "validation"
            and _row_value(spec, "family") != family
        ),
        calibration=tuple(
            spec for spec in specs
            if _row_value(spec, "split") == "calibration"
            and _row_value(spec, "family") != family
        ),
        test=tuple(
            spec for spec in specs
            if _row_value(spec, "split") == "test" and _row_value(spec, "family") == family
        ),
    )


@dataclass(frozen=True, slots=True)
class ValidationSelection:
    method: str
    requested_k: int
    sequence_count: int
    tie_order: tuple[str, ...]
    method_means: Mapping[str, float]


def select_validation_method(rows: Sequence[Mapping[str, Any]]) -> ValidationSelection:
    if any(
        str(row.get("method", "")) == "oracle_support_upper_bound"
        or _bool_value(row.get("uses_oracle", False))
        for row in rows
    ):
        raise ValueError("oracle rows are forbidden from primary validation selection")
    filtered = [
        row for row in rows
        if str(row.get("method", "")) in ORDINARY_METHODS
        and _integer(row.get("requested_k", -1), "requested_k") == 8
        and _finite_float(row.get("reveal_ratio", 1.0), "reveal_ratio") < 1.0
    ]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in filtered:
        grouped[(str(row["sequence_id"]), str(row["method"]))].append(
            _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
        )
    sequence_ids = sorted({identity[0] for identity in grouped})
    if len(sequence_ids) != 15:
        raise ValueError("validation selection requires 15 complete sequences")
    method_means: dict[str, float] = {}
    for method in ORDINARY_METHODS:
        sequence_means = []
        for sequence_id in sequence_ids:
            values = grouped.get((sequence_id, method), [])
            if not values:
                raise ValueError("validation method/sequence universe is incomplete")
            sequence_means.append(float(np.mean(values)))
        method_means[method] = float(np.mean(sequence_means))
    minimum = min(method_means.values())
    tied = {
        method for method, value in method_means.items()
        if abs(value - minimum) <= 1e-12
    }
    selected = next(method for method in VALIDATION_TIE_ORDER if method in tied)
    return ValidationSelection(
        method=selected,
        requested_k=8,
        sequence_count=len(sequence_ids),
        tie_order=VALIDATION_TIE_ORDER,
        method_means=method_means,
    )


def _rms(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def _matrix_digest(matrix: np.ndarray) -> str:
    array = np.ascontiguousarray(matrix)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    nearest_rms_distance: float
    nearest_matrix_l1_distance: int
    covering_radius: float
    mean_pairwise_diversity: float
    duplicate_fraction: float
    actual_k: int
    requested_k: int
    component_coverage: float
    joint_coverage: int
    physical_normalized_mean_width: float
    zero_physical_range_components: int
    total_tail_event: int
    total_tail_hit: int
    group_tail_events: int
    group_tail_hits: int
    hotspot_event: int
    hotspot_hit: int


def compute_case_metrics(
    *,
    truth_descriptor: Any,
    truth_matrix: Any,
    support_descriptors: Any,
    support_matrices: Any,
    pool_descriptors: Any,
    fit_scale: Any,
    lower: Any,
    upper: Any,
    physical_low: Any,
    physical_high: Any,
    truth_satisfies_observation: bool,
    total_index: int,
    group_indices: Sequence[int],
    total_tail_threshold: float,
    group_tail_thresholds: Any,
    truth_hotspot_destination: int,
    support_hotspot_destinations: Sequence[int],
    requested_k: int,
) -> CaseMetrics:
    descriptor = np.asarray(truth_descriptor, dtype=np.float64)
    supports = np.asarray(support_descriptors, dtype=np.float64)
    pool = np.asarray(pool_descriptors, dtype=np.float64)
    scale = np.asarray(fit_scale, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    physical_min = np.asarray(physical_low, dtype=np.float64)
    physical_max = np.asarray(physical_high, dtype=np.float64)
    if descriptor.ndim != 1 or supports.ndim != 2 or supports.shape[1:] != descriptor.shape:
        raise ValueError("support descriptor shape mismatch")
    if pool.ndim != 2 or pool.shape[1:] != descriptor.shape or len(supports) == 0:
        raise ValueError("candidate pool/support cannot be empty")
    for name, array in (
        ("truth", descriptor), ("support", supports), ("pool", pool),
        ("scale", scale), ("lower", low), ("upper", high),
        ("physical low", physical_min), ("physical high", physical_max),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} descriptor values must be finite")
    if any(array.shape != descriptor.shape for array in (scale, low, high, physical_min, physical_max)):
        raise ValueError("descriptor vector shape mismatch")
    if np.any(scale <= 0.0) or np.any(low > high) or np.any(physical_min > physical_max):
        raise ValueError("invalid scale or descriptor bounds")

    nearest_rms = min(_rms(descriptor, candidate, scale) for candidate in supports)
    truth_values = np.asarray(truth_matrix)
    matrices = np.asarray(support_matrices)
    if truth_values.ndim != 2 or truth_values.shape[0] != truth_values.shape[1]:
        raise ValueError("truth matrix must be square")
    if matrices.shape != (len(supports), *truth_values.shape):
        raise ValueError("support matrix shape mismatch")
    off_diagonal = ~np.eye(truth_values.shape[0], dtype=bool)
    nearest_l1 = min(
        int(np.abs(truth_values[off_diagonal] - matrix[off_diagonal]).sum())
        for matrix in matrices
    )
    covering = max(
        min(_rms(candidate, chosen, scale) for chosen in supports)
        for candidate in pool
    )
    if len(supports) == 1:
        diversity = 0.0
    else:
        distances = [
            _rms(supports[left], supports[right], scale)
            for left in range(len(supports))
            for right in range(left + 1, len(supports))
        ]
        diversity = float(np.mean(distances))
    unique_count = len({_matrix_digest(np.asarray(matrix)) for matrix in matrices})
    duplicate_fraction = 1.0 - unique_count / len(matrices)
    tolerance = 1.000001e-10
    component_hits = (descriptor >= low - tolerance) & (descriptor <= high + tolerance)
    component_coverage = float(np.mean(component_hits))
    joint = int(bool(np.all(component_hits)) and bool(truth_satisfies_observation))
    physical_ranges = physical_max - physical_min
    positive = physical_ranges > 0.0
    zero_count = int(np.count_nonzero(~positive))
    width = float(np.mean((high[positive] - low[positive]) / physical_ranges[positive]))

    total = int(total_index)
    total_threshold = _finite_float(total_tail_threshold, "total tail threshold")
    total_event = int(descriptor[total] > total_threshold)
    total_hit = int(total_event and np.max(supports[:, total]) > total_threshold)
    indices = tuple(int(index) for index in group_indices)
    thresholds = np.asarray(group_tail_thresholds, dtype=np.float64)
    if thresholds.shape != (len(indices),):
        raise ValueError("group tail threshold shape mismatch")
    group_events = 0
    group_hits = 0
    for descriptor_index, threshold in zip(indices, thresholds):
        event = int(descriptor[descriptor_index] > threshold)
        hit = int(event and np.max(supports[:, descriptor_index]) > threshold)
        group_events += event
        group_hits += hit
    hotspot_event = 1
    hotspot_hit = int(
        int(truth_hotspot_destination)
        in {int(destination) for destination in support_hotspot_destinations}
    )
    return CaseMetrics(
        nearest_rms_distance=float(nearest_rms),
        nearest_matrix_l1_distance=int(nearest_l1),
        covering_radius=float(covering),
        mean_pairwise_diversity=float(diversity),
        duplicate_fraction=float(duplicate_fraction),
        actual_k=len(supports),
        requested_k=int(requested_k),
        component_coverage=component_coverage,
        joint_coverage=joint,
        physical_normalized_mean_width=width,
        zero_physical_range_components=zero_count,
        total_tail_event=total_event,
        total_tail_hit=total_hit,
        group_tail_events=group_events,
        group_tail_hits=group_hits,
        hotspot_event=hotspot_event,
        hotspot_hit=hotspot_hit,
    )


@dataclass(frozen=True, slots=True)
class RandomReplicateAggregate:
    replicate_count: int
    nearest_rms_distance: float
    joint_coverage: int
    envelope_count: int
    lower: list[float]
    upper: list[float]


def aggregate_random_replicates(
    rows: Sequence[Mapping[str, Any]],
) -> RandomReplicateAggregate:
    values = list(rows)
    if len(values) != RANDOM_REPLICATES:
        raise ValueError("random aggregation requires exactly 8 replicates")
    indices = [_integer(row["replicate_index"], "replicate index") for row in values]
    if set(indices) != set(range(RANDOM_REPLICATES)):
        raise ValueError("replicate indices are duplicate or incomplete")
    if len({str(row["case_id"]) for row in values}) != 1 or len(
        {str(row["method"]) for row in values}
    ) != 1:
        raise ValueError("replicate case identity mismatch")
    lower = list(values[0]["lower"])
    upper = list(values[0]["upper"])
    if any(list(row["lower"]) != lower or list(row["upper"]) != upper for row in values[1:]):
        raise ValueError("replicate lower/upper envelope must be consistent")
    coverage = _integer(values[0]["joint_coverage"], "joint coverage")
    if any(_integer(row["joint_coverage"], "joint coverage") != coverage for row in values[1:]):
        raise ValueError("replicate envelope coverage must be consistent")
    return RandomReplicateAggregate(
        replicate_count=RANDOM_REPLICATES,
        nearest_rms_distance=float(
            np.mean(
                [
                    _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
                    for row in values
                ]
            )
        ),
        joint_coverage=coverage,
        envelope_count=1,
        lower=lower,
        upper=upper,
    )


@dataclass(frozen=True, slots=True)
class SequenceMetricAggregate:
    sequence_id: str
    method: str
    requested_k: int
    raw_case_count: int
    nearest_rms_distance: float
    total_tail_events: float
    total_tail_hits: float
    total_tail_recall: float | None
    group_tail_events: float
    group_tail_hits: float
    group_tail_recall: float | None
    hotspot_events: float
    hotspot_hits: float
    hotspot_recall: float | None


def aggregate_sequence_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> SequenceMetricAggregate:
    values = list(rows)
    if len(values) != 320:
        raise ValueError("sequence aggregation requires exact 320-case universe")
    expected = {
        (checkpoint, mode, ratio)
        for checkpoint in CHECKPOINTS
        for mode in REVEAL_MODES
        for ratio in UNKNOWN_RATIOS
    }
    identities = [
        (
            _integer(row["checkpoint"], "checkpoint"),
            str(row["reveal_mode"]),
            _finite_float(row["reveal_ratio"], "reveal ratio"),
        )
        for row in values
    ]
    if len(set(identities)) != 320 or set(identities) != expected:
        raise ValueError("sequence case universe has duplicate/missing identity")
    sequence_ids = {str(row["sequence_id"]) for row in values}
    methods = {str(row["method"]) for row in values}
    requested = {_integer(row["requested_k"], "requested K") for row in values}
    if len(sequence_ids) != 1 or len(methods) != 1 or len(requested) != 1:
        raise ValueError("sequence aggregation identity mismatch")
    total_events = sum(_finite_float(row["total_tail_events"], "total events") for row in values)
    total_hits = sum(_finite_float(row["total_tail_hits"], "total hits") for row in values)
    group_events = sum(_finite_float(row["group_tail_events"], "group events") for row in values)
    group_hits = sum(_finite_float(row["group_tail_hits"], "group hits") for row in values)
    hotspot_events = sum(_finite_float(row["hotspot_event"], "hotspot event") for row in values)
    hotspot_hits = sum(_finite_float(row["hotspot_hit"], "hotspot hit") for row in values)
    return SequenceMetricAggregate(
        sequence_id=next(iter(sequence_ids)),
        method=next(iter(methods)),
        requested_k=next(iter(requested)),
        raw_case_count=320,
        nearest_rms_distance=float(
            np.mean(
                [_finite_float(row["nearest_rms_distance"], "nearest RMS distance") for row in values]
            )
        ),
        total_tail_events=total_events,
        total_tail_hits=total_hits,
        total_tail_recall=(None if total_events == 0 else total_hits / total_events),
        group_tail_events=group_events,
        group_tail_hits=group_hits,
        group_tail_recall=(None if group_events == 0 else group_hits / group_events),
        hotspot_events=hotspot_events,
        hotspot_hits=hotspot_hits,
        hotspot_recall=(None if hotspot_events == 0 else hotspot_hits / hotspot_events),
    )


@dataclass(frozen=True, slots=True)
class BootstrapResult(_AttributeMapping):
    sequence_count: int
    family_count: int
    replicates: int
    seed: int
    mean_delta: float
    ci_lower: float
    ci_upper: float


def family_stratified_sequence_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
    seed: int = 20260731,
) -> BootstrapResult:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["family"]),
            _integer(row["base_seed"], "base seed"),
            str(row["sequence_id"]),
        ),
    )
    if len(ordered) != 15 or len({str(row["sequence_id"]) for row in ordered}) != 15:
        raise ValueError("bootstrap requires 15 unique sequences (3 per family)")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in ordered:
        grouped[str(row["family"])].append(
            _finite_float(row["paired_delta"], "paired delta")
        )
    if len(grouped) != 5 or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("bootstrap requires exactly 3 sequences in each of 5 families")
    count = int(replicates)
    if count <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    rng = np.random.default_rng(int(seed))
    samples = np.empty(count, dtype=np.float64)
    family_arrays = [np.asarray(grouped[family], dtype=np.float64) for family in sorted(grouped)]
    for replicate in range(count):
        sampled = [values[rng.integers(0, 3, size=3)] for values in family_arrays]
        samples[replicate] = float(np.mean(np.concatenate(sampled)))
    all_values = np.concatenate(family_arrays)
    return BootstrapResult(
        sequence_count=15,
        family_count=5,
        replicates=count,
        seed=int(seed),
        mean_delta=float(np.mean(all_values)),
        ci_lower=float(np.quantile(samples, 0.025)),
        ci_upper=float(np.quantile(samples, 0.975)),
    )


@dataclass(frozen=True, slots=True)
class GateDecision:
    data_status: str
    gate_status: str
    conditions: tuple[int, ...]
    failed_conditions: tuple[int, ...]
    insufficient_conditions: tuple[int, ...]


def _ratio(hits: Any, events: Any, name: str) -> float:
    hit_count = _finite_float(hits, f"{name} hits")
    event_count = _integer(events, f"{name} events")
    if event_count <= 0 or hit_count < 0 or hit_count > event_count:
        raise ValueError(f"invalid {name} event/hit counts")
    return hit_count / event_count


def evaluate_phase3b_gate(evidence: Mapping[str, Any]) -> GateDecision:
    failed: list[int] = []
    insufficient: list[int] = []

    condition_1 = (
        _finite_float(evidence["selected_joint_coverage"], "joint coverage") >= 0.85
        and len(evidence["selected_joint_coverage_by_family"]) == 5
        and all(
            _finite_float(value, "family coverage") >= 0.80
            for value in evidence["selected_joint_coverage_by_family"].values()
        )
    )
    if not condition_1:
        failed.append(1)

    ci = tuple(_finite_float(value, "paired CI") for value in evidence["paired_delta_ci95"])
    seed_values = [
        _finite_float(value, "base-seed delta")
        for value in evidence["paired_delta_by_base_seed"].values()
    ]
    family_values = [
        _finite_float(value, "family delta")
        for value in evidence["paired_delta_by_family"].values()
    ]
    condition_2 = (
        len(ci) == 2
        and ci[0] > 0.0
        and len(seed_values) == 3
        and all(value > 0.0 for value in seed_values)
        and len(family_values) == 5
        and sum(value > 0.0 for value in family_values) >= 4
    )
    if not condition_2:
        failed.append(2)

    lofo_values = [
        _finite_float(value, "LOFO delta")
        for value in evidence["lofo_delta_by_family"].values()
    ]
    degradation = [
        _finite_float(value, "LOFO relative degradation")
        for value in evidence["lofo_relative_degradation_by_family"].values()
    ]
    condition_3 = (
        _finite_float(evidence["lofo_aggregate_delta"], "LOFO aggregate delta") >= 0.0
        and len(lofo_values) == 5
        and sum(value > 0.0 for value in lofo_values) >= 3
        and len(degradation) == 5
        and sum(value > 0.10 for value in degradation) <= 1
    )
    if not condition_3:
        failed.append(3)

    total_events = _integer(evidence["total_tail_events"], "total events")
    group_events = _integer(evidence["group_tail_events"], "group events")
    if total_events < 10 or group_events < 10:
        insufficient.append(4)
    else:
        condition_4 = (
            _ratio(evidence["total_tail_hits"], total_events, "total tail") >= 0.70
            and _ratio(evidence["group_tail_hits"], group_events, "group tail") >= 0.70
            and _ratio(
                evidence["hotspot_hits"], evidence["hotspot_events"], "hotspot"
            ) >= 0.70
        )
        if not condition_4:
            failed.append(4)

    invalid_rates = [
        _finite_float(value, "ordinary invalid rate")
        for value in evidence["ordinary_invalid_or_empty_rate"].values()
    ]
    condition_5 = (
        _finite_float(
            evidence["selected_mean_physical_normalized_width"], "mean width"
        ) <= 0.75
        and len(invalid_rates) == len(ORDINARY_METHODS)
        and all(value == 0.0 for value in invalid_rates)
        and _finite_float(evidence["ratio1_singleton_coverage"], "ratio1 coverage") == 1.0
        and bool(evidence["all_timings_finite"])
    )
    if not condition_5:
        failed.append(5)

    condition_6 = bool(evidence["integrity_checks_complete"]) and bool(
        evidence["integrity_checks_passed"]
    )
    if not condition_6:
        failed.append(6)

    if failed:
        data_status = "FAIL"
    elif insufficient:
        data_status = "HOLD"
    else:
        data_status = "PASS"
    return GateDecision(
        data_status=data_status,
        gate_status="PENDING_SUPERVISOR",
        conditions=(1, 2, 3, 4, 5, 6),
        failed_conditions=tuple(failed),
        insufficient_conditions=tuple(insufficient),
    )


def build_summary(
    evidence: Mapping[str, Any],
    *,
    selected_method: str,
) -> dict[str, Any]:
    if selected_method not in ORDINARY_METHODS:
        raise ValueError("selected method must be ordinary")
    decision = evaluate_phase3b_gate(evidence)
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "selected_method": selected_method,
        "selected_k": 8,
        "data_status": decision.data_status,
        "gate_status": "PENDING_SUPERVISOR",
        "conditions_evaluated": list(decision.conditions),
        "failed_conditions": list(decision.failed_conditions),
        "insufficient_conditions": list(decision.insufficient_conditions),
    }


_RAW_REQUIRED_FIELDS = {
    "case_id",
    "sequence_id",
    "split",
    "family",
    "base_seed",
    "checkpoint",
    "reveal_mode",
    "reveal_ratio",
    "requested_k",
    "construction_seed",
    "method",
    "nearest_rms_distance",
    "joint_coverage",
    "uses_oracle",
    "upper_bound_only",
    "sequence_digest",
    "observation_digest",
    "ambiguity_digest",
    "support_digest",
}


def validate_raw_case_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete_universe: bool = False,
) -> bool:
    values = list(rows)
    if not values:
        raise ValueError("raw case rows cannot be empty")
    case_ids: set[str] = set()
    semantic_ids: set[tuple[Any, ...]] = set()
    for row in values:
        missing = _RAW_REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"raw row missing fields: {sorted(missing)}")
        case_id = str(row["case_id"])
        if case_id in case_ids:
            raise ValueError("duplicate case_id")
        case_ids.add(case_id)
        semantic = (
            str(row["sequence_id"]),
            _integer(row["checkpoint"], "checkpoint"),
            str(row["reveal_mode"]),
            _finite_float(row["reveal_ratio"], "reveal ratio"),
            _integer(row["requested_k"], "requested K"),
            str(row["method"]),
            _integer(row["construction_seed"], "construction seed"),
        )
        if semantic in semantic_ids:
            raise ValueError("duplicate semantic case identity")
        semantic_ids.add(semantic)
        _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
        _finite_float(row["joint_coverage"], "joint coverage")
        for digest_name in (
            "sequence_digest", "observation_digest", "ambiguity_digest", "support_digest"
        ):
            digest = str(row[digest_name])
            if len(digest) != 64:
                raise ValueError(f"{digest_name} must be a SHA-256 digest")
    if not allow_incomplete_universe:
        _validate_formal_raw_universe(values)
    return True


def _validate_formal_raw_universe(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_count = 15 * len(CHECKPOINTS) * len(REVEAL_MODES) * len(
        REVEAL_RATIOS
    ) * len(REQUESTED_K) * len(ALL_METHODS)
    if len(rows) != expected_count:
        raise ValueError(
            f"formal raw universe requires exactly {expected_count} rows"
        )
    specs = build_formal_sequence_specs()
    test_specs = {
        spec.sequence_id: spec for spec in specs if spec.split == "test"
    }
    if len(test_specs) != 15:
        raise ValueError("formal test specification universe mismatch")
    case_indices = _case_index_lookup(specs)
    for row in rows:
        sequence_id = str(row["sequence_id"])
        if sequence_id not in test_specs or str(row["split"]) != "test":
            raise ValueError("formal raw row is outside the frozen test split")
        spec = test_specs[sequence_id]
        if (
            _integer(row["record_index"], "record index") != spec.record_index
            or str(row["family"]) != spec.family
            or _integer(row["base_seed"], "base seed") != spec.base_seed
        ):
            raise ValueError("formal raw row sequence provenance mismatch")
        checkpoint = _integer(row["checkpoint"], "checkpoint")
        mode = str(row["reveal_mode"])
        ratio = _finite_float(row["reveal_ratio"], "reveal ratio")
        requested_k = _integer(row["requested_k"], "requested K")
        method = str(row["method"])
        if (
            checkpoint not in CHECKPOINTS
            or mode not in REVEAL_MODES
            or ratio not in REVEAL_RATIOS
            or requested_k not in REQUESTED_K
            or method not in ALL_METHODS
        ):
            raise ValueError("formal raw row contains an unfrozen case coordinate")
        mode_index = REVEAL_MODES.index(mode)
        stage_index = REVEAL_RATIOS.index(ratio)
        expected_case_index = case_indices[
            (
                "test",
                spec.record_index,
                checkpoint,
                mode_index,
                stage_index,
                requested_k,
            )
        ]
        if _integer(row["case_index"], "case index") != expected_case_index:
            raise ValueError("formal raw case index does not match the registry")
        if str(row["case_id"]) != f"case-{expected_case_index}-{method}":
            raise ValueError("formal raw case ID does not match the registry")
        if _integer(row["construction_seed"], "construction seed") != reveal_seed(
            spec.record_index, checkpoint, mode_index
        ):
            raise ValueError("formal raw construction seed provenance mismatch")
        expected_oracle = method == "oracle_support_upper_bound"
        if (
            _bool_value(row["uses_oracle"]) != expected_oracle
            or _bool_value(row["upper_bound_only"]) != expected_oracle
        ):
            raise ValueError("formal raw oracle isolation flags are invalid")
        actual_k = _integer(row["actual_k"], "actual K")
        expected_actual_k = 1 if ratio == 1.0 else requested_k
        if actual_k != expected_actual_k:
            raise ValueError("formal ratio-1/K support-size contract is invalid")


@dataclass(frozen=True, slots=True)
class RecomputedArtifacts(_AttributeMapping):
    manifest: Mapping[str, Any]
    raw_case_rows: tuple[Mapping[str, Any], ...]
    raw_sequence_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def _raw_summary(
    manifest: Mapping[str, Any],
    raw_case_rows: Sequence[Mapping[str, Any]],
    raw_sequence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_distances = [
        _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
        for row in raw_case_rows
    ]
    sequence_distances = [
        _finite_float(row["nearest_rms_distance"], "sequence nearest RMS distance")
        for row in raw_sequence_rows
    ]
    common = {
        "protocol_sha256": str(manifest.get("protocol_sha256", PROTOCOL_SHA256)),
        "selected_method": str(manifest.get("selected_method", "minimax_subset")),
        "selected_k": _integer(manifest.get("selected_k", 8), "selected K"),
        "raw_case_count": len(raw_case_rows),
        "raw_sequence_count": len(raw_sequence_rows),
        "mean_case_nearest_rms_distance": float(np.mean(case_distances)),
        "mean_sequence_nearest_rms_distance": float(np.mean(sequence_distances)),
        "data_status": str(manifest.get("data_status", "HOLD")),
        "gate_status": "PENDING_SUPERVISOR",
    }
    if "gate_evidence" in manifest:
        summary = build_summary(
            manifest["gate_evidence"],
            selected_method=str(manifest.get("selected_method", "minimax_subset")),
        )
        summary.update(
            {
                "raw_case_count": common["raw_case_count"],
                "raw_sequence_count": common["raw_sequence_count"],
                "mean_case_nearest_rms_distance": common[
                    "mean_case_nearest_rms_distance"
                ],
                "mean_sequence_nearest_rms_distance": common[
                    "mean_sequence_nearest_rms_distance"
                ],
            }
        )
        if "test_total_traffic_dependence" in manifest:
            summary["test_total_traffic_dependence"] = _canonical(
                manifest["test_total_traffic_dependence"]
            )
        return summary
    return common


def recompute_artifacts(
    manifest: Mapping[str, Any],
    raw_case_rows: Sequence[Mapping[str, Any]],
    raw_sequence_rows: Sequence[Mapping[str, Any]],
    *,
    expected_summary: Mapping[str, Any] | None = None,
    allow_incomplete_universe: bool = False,
) -> RecomputedArtifacts:
    cases = sorted(
        (dict(row) for row in raw_case_rows),
        key=lambda row: str(row.get("case_id", "")),
    )
    sequences = sorted(
        (dict(row) for row in raw_sequence_rows),
        key=lambda row: (
            str(row.get("sequence_id", "")),
            str(row.get("method", "")),
            _integer(row.get("requested_k", 0), "requested K"),
        ),
    )
    if not cases or not sequences:
        raise ValueError("recompute requires nonempty raw case and sequence rows")
    if not allow_incomplete_universe:
        validate_raw_case_rows(cases, allow_incomplete_universe=False)

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    aggregation_cases = [
        row for row in cases
        if "reveal_ratio" not in row
        or _finite_float(row["reveal_ratio"], "reveal ratio") < 1.0
    ]
    for row in aggregation_cases:
        identity = (
            str(row["sequence_id"]),
            str(row["method"]),
            _integer(row["requested_k"], "requested K"),
        )
        grouped[identity].append(
            _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
        )
    sequence_identities: set[tuple[str, str, int]] = set()
    for row in sequences:
        identity = (
            str(row["sequence_id"]),
            str(row["method"]),
            _integer(row["requested_k"], "requested K"),
        )
        if identity in sequence_identities:
            raise ValueError("recompute found duplicate sequence metric identity")
        sequence_identities.add(identity)
        if identity not in grouped:
            raise ValueError("recompute sequence row has no raw cases")
        expected_distance = float(np.mean(grouped[identity]))
        actual_distance = _finite_float(
            row["nearest_rms_distance"], "sequence nearest RMS distance"
        )
        if abs(expected_distance - actual_distance) > 1e-12:
            raise ValueError("recompute mismatch indicates corrupt raw metrics")
        if "raw_case_count" in row and _integer(row["raw_case_count"], "raw case count") != len(
            grouped[identity]
        ):
            raise ValueError("recompute raw case count mismatch")
    if not allow_incomplete_universe and set(grouped) != sequence_identities:
        raise ValueError("recompute sequence universe incomplete")
    summary = _raw_summary(manifest, cases, sequences)
    if expected_summary is not None and _canonical_bytes(summary) != _canonical_bytes(
        dict(expected_summary)
    ):
        raise ValueError("recomputed summary mismatch indicates corruption")
    return RecomputedArtifacts(
        manifest=dict(manifest),
        raw_case_rows=tuple(cases),
        raw_sequence_rows=tuple(sequences),
        summary=summary,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_canonical(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("artifact CSV cannot be empty")
    fieldnames = list(values[0])
    if any(set(row) != set(fieldnames) for row in values):
        raise ValueError("artifact CSV rows have inconsistent schema")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def write_experiment_artifacts(
    output_directory: str | Path,
    *,
    manifest: Mapping[str, Any],
    raw_case_rows: Sequence[Mapping[str, Any]],
    raw_sequence_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "manifest.json", manifest)
    _write_csv(destination / "raw_case_metrics.csv", raw_case_rows)
    _write_csv(destination / "raw_sequence_metrics.csv", raw_sequence_rows)
    _write_json(destination / "summary.json", summary)
    return {
        "manifest": dict(manifest),
        "raw_case_rows": [dict(row) for row in raw_case_rows],
        "raw_sequence_rows": [dict(row) for row in raw_sequence_rows],
        "summary": dict(summary),
    }


def _toy_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def run_toy_experiment(output_directory: str | Path) -> dict[str, Any]:
    """Write a deterministic one-case smoke artifact set, never formal data."""

    manifest = {
        "protocol_sha256": PROTOCOL_SHA256,
        "experiment": "phase3b_toy",
        "selected_method": "minimax_subset",
        "selected_k": 8,
        "sequence_count": 1,
        "data_status": "HOLD",
    }
    raw_cases = [
        {
            "case_id": "toy-case-0",
            "sequence_id": "toy-sequence-0",
            "split": "test",
            "family": FORMAL_FAMILIES[0],
            "base_seed": FORMAL_BASE_SEEDS[0],
            "checkpoint": 32,
            "reveal_mode": REVEAL_MODES[0],
            "reveal_ratio": 0.0,
            "requested_k": 8,
            "construction_seed": 31_000_320,
            "method": "minimax_subset",
            "nearest_rms_distance": 0.4,
            "joint_coverage": 1,
            "uses_oracle": False,
            "upper_bound_only": False,
            "sequence_digest": _toy_digest("toy-sequence"),
            "observation_digest": _toy_digest("toy-observation"),
            "ambiguity_digest": _toy_digest("toy-ambiguity"),
            "support_digest": _toy_digest("toy-support"),
        }
    ]
    raw_sequences = [
        {
            "sequence_id": "toy-sequence-0",
            "method": "minimax_subset",
            "requested_k": 8,
            "nearest_rms_distance": 0.4,
            "joint_coverage": 1.0,
            "raw_case_count": 1,
        }
    ]
    recomputed = recompute_artifacts(
        manifest,
        raw_cases,
        raw_sequences,
        allow_incomplete_universe=True,
    )
    return write_experiment_artifacts(
        output_directory,
        manifest=manifest,
        raw_case_rows=raw_cases,
        raw_sequence_rows=raw_sequences,
        summary=recomputed.summary,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(matrices: Sequence[Any]) -> str:
    values = np.ascontiguousarray(np.stack(matrices), dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _positive_sequence_dependence(
    values: Sequence[Any], *, max_lag: int = 64
) -> dict[str, Any]:
    """Global-mean ACF and initial-positive-sequence ESS for one series.

    For ``rho_l = sum_t (x_t-xbar)(x_{t+l}-xbar) / sum_t
    (x_t-xbar)^2``, positive correlations are accumulated from lag one up to
    ``max_lag`` and stop immediately before the first non-positive value.  The
    estimator is ``n / (1 + 2 * sum(rho_l))``.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("dependence series must be finite, one-dimensional, and nontrivial")
    centered = array - float(array.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        return {
            "defined": False,
            "reason": "constant_series",
            "sample_count": len(array),
            "lag1_acf": None,
            "positive_lag_count": 0,
            "positive_sequence_ess": None,
        }
    correlations = [
        float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        for lag in range(1, min(int(max_lag), len(array) - 1) + 1)
    ]
    positive: list[float] = []
    for correlation in correlations:
        if correlation <= 0.0:
            break
        positive.append(correlation)
    ess = len(array) / (1.0 + 2.0 * sum(positive))
    return {
        "defined": True,
        "reason": None,
        "sample_count": len(array),
        "lag1_acf": correlations[0],
        "positive_lag_count": len(positive),
        "positive_sequence_ess": float(np.clip(ess, 1.0, len(array))),
    }


def _formal_test_dependence(
    specs: Sequence[SequenceSpec], sequences: Mapping[str, Any]
) -> dict[str, Any]:
    per_sequence: dict[str, Any] = {}
    for spec in specs:
        totals = [
            float(np.asarray(sequences[spec.sequence_id].matrices[checkpoint]).sum())
            for checkpoint in CHECKPOINTS
        ]
        per_sequence[spec.sequence_id] = _positive_sequence_dependence(
            totals, max_lag=64
        )
    if len(per_sequence) != 15:
        raise ValueError("test dependence requires exactly 15 sequences")
    defined = [row for row in per_sequence.values() if bool(row["defined"])]
    lag1_values = [float(row["lag1_acf"]) for row in defined]
    ess_values = [float(row["positive_sequence_ess"]) for row in defined]
    return {
        "series_definition": "total traffic at the 16 frozen test checkpoints",
        "acf_formula": (
            "rho_l=sum_t((x_t-xbar)*(x_(t+l)-xbar))/"
            "sum_t((x_t-xbar)^2)"
        ),
        "ess_formula": "n/(1+2*sum(rho_l)); stop before first rho_l<=0; max_lag=64",
        "sequence_weighting": "15 test sequences reported separately and equally",
        "max_lag": 64,
        "per_sequence": per_sequence,
        "aggregate": {
            "sequence_count": 15,
            "defined_sequence_count": len(defined),
            "mean_lag1_acf": None if not defined else float(np.mean(lag1_values)),
            "mean_positive_sequence_ess": (
                None if not defined else float(np.mean(ess_values))
            ),
            "sum_positive_sequence_ess": (
                None if not defined else float(np.sum(ess_values))
            ),
        },
    }


def _load_h1_digest_exclusion(project_root: Path) -> tuple[set[str], str]:
    manifest_path = project_root / "outputs" / "h1_predictability" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "formal Phase 3B requires the reviewed H1 manifest for digest exclusion"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("sequence_records", [])
    digests = {
        str(record.get("digest", record.get("sequence_digest", "")))
        for record in records
    }
    digests.discard("")
    if len(digests) != 75:
        raise ValueError("H1 digest exclusion manifest must contain 75 unique records")
    return digests, _file_sha256(manifest_path)


def _load_rear4_topology(project_root: Path) -> tuple[TopologyInfo, str]:
    topology_path = (
        project_root
        / "Data"
        / "Rear4GPU"
        / "Topology"
        / "pipeline_topology_no_switch.json"
    )
    data = json.loads(topology_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    node_indices = {node["id"]: index for index, node in enumerate(nodes)}
    edges = np.asarray(
        [
            [node_indices[link["source"]], node_indices[link["target"]]]
            for link in links
        ],
        dtype=np.int64,
    )
    capacities = np.asarray(
        [float(link.get("capacity_value", 1.0)) for link in links],
        dtype=np.float64,
    )
    edge_indices = {
        (int(source), int(destination)): index
        for index, (source, destination) in enumerate(edges)
    }
    constraints: list[tuple[list[int], float]] = []
    for group in data.get("bandwidth_groups", {}).values():
        indices = [
            edge_indices[
                (
                    node_indices[edge["source"]],
                    node_indices[edge["target"]],
                )
            ]
            for edge in group.get("edges", [])
            if (
                node_indices[edge["source"]],
                node_indices[edge["target"]],
            ) in edge_indices
        ]
        if indices:
            constraints.append((indices, float(group["max_bandwidth"])))
    minimum = float(capacities.min())
    if minimum <= 0.0:
        raise ValueError("Rear4GPU topology capacities must be positive")
    capacities /= minimum
    constraints = [(indices, limit / minimum) for indices, limit in constraints]
    topology = TopologyInfo(
        len(nodes),
        len(edges),
        edges,
        capacities,
        constraints,
        name="Rear4GPU",
    )
    return topology, _file_sha256(topology_path)


def _formal_observation(
    matrix: np.ndarray,
    *,
    spec: SequenceSpec,
    checkpoint: int,
    topology: PublicTopologyView,
    mode: str,
    ratio: float,
    seed: int,
) -> tuple[PartialObservationState, float]:
    values = np.asarray(matrix, dtype=np.int64)
    size = values.shape[0]
    entries = np.asarray(
        [
            (source, destination)
            for source in range(size)
            for destination in range(size)
            if source != destination
        ],
        dtype=np.int64,
    )
    random = np.random.default_rng(int(seed))
    order = entries[random.permutation(len(entries))]
    # DemandRevealProcess consumes this draw for every mode, including
    # partial_shards, before it constructs the token permutation.
    arrival = random.uniform(np.nextafter(0.0, 1.0), 1.0, size=len(entries))
    mask = np.eye(size, dtype=bool)
    observed = np.zeros_like(values)
    if mode == "partial_shards":
        units = np.asarray(
            [
                (source, destination)
                for source in range(size)
                for destination in range(size)
                if source != destination
                for _ in range(int(values[source, destination]))
            ],
            dtype=np.int64,
        )
        if len(units):
            units = units[random.permutation(len(units))]
        count = len(units) if ratio == 1.0 else int(math.floor(ratio * len(units)))
        for source, destination in units[:count]:
            observed[int(source), int(destination)] += 1
        if ratio == 1.0:
            mask[:, :] = True
            observed[:, :] = values
        else:
            for source, destination in entries:
                source_i, destination_i = int(source), int(destination)
                total = int(values[source_i, destination_i])
                if total > 0 and observed[source_i, destination_i] == total:
                    mask[source_i, destination_i] = True
    else:
        if mode == "time_based_arrival":
            selected = order[arrival <= ratio]
            if ratio == 1.0:
                selected = order
        else:
            count = len(order) if ratio == 1.0 else int(math.floor(ratio * len(order)))
            selected = order[:count]
        for source, destination in selected:
            mask[int(source), int(destination)] = True
        observed[mask] = values[mask]
    source_totals = (
        values.sum(axis=1)
        if mode in {"source_totals_first", "source_destination_totals_first"}
        else None
    )
    destination_totals = (
        values.sum(axis=0) if mode == "source_destination_totals_first" else None
    )
    actual_fraction = float(np.count_nonzero(mask & ~np.eye(size, dtype=bool)) / len(entries))
    stage = REVEAL_RATIOS.index(float(ratio))
    return (
        PartialObservationState(
            sequence_id=spec.sequence_id,
            sequence_step=int(checkpoint),
            family=spec.family,
            mode=mode,
            stage=stage,
            ratio=float(ratio),
            entry_mask=mask,
            observed_matrix=observed,
            unknown_mask=~mask,
            revealed_tokens=(),
            source_totals=source_totals,
            destination_totals=destination_totals,
            topology=topology,
            state_version=0,
        ),
        actual_fraction,
    )


def _formal_view(
    sequence: Any,
    spec: SequenceSpec,
    *,
    checkpoint: int,
    topology: PublicTopologyView,
    mode_index: int,
    ratio: float,
    normalizer: Any,
) -> tuple[AmbiguityConstructionView, float]:
    seed = reveal_seed(spec.record_index, checkpoint, mode_index)
    observation, actual_fraction = _formal_observation(
        np.asarray(sequence.matrices[checkpoint]),
        spec=spec,
        checkpoint=checkpoint,
        topology=topology,
        mode=REVEAL_MODES[mode_index],
        ratio=ratio,
        seed=seed,
    )
    view = AmbiguityConstructionView.from_observation(
        history_matrices=tuple(sequence.matrices[checkpoint - HISTORY_WINDOW : checkpoint]),
        history_offsets=tuple(range(-HISTORY_WINDOW, 0)),
        observation=observation,
        construction_seed=seed,
        normalizer=normalizer,
    )
    return view, actual_fraction


def _formal_calibration_rows(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str, Any],
    *,
    topology: PublicTopologyView,
    normalizer: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        sequence = sequences[spec.sequence_id]
        for checkpoint in CHECKPOINTS:
            descriptor = traffic_descriptor(sequence.matrices[checkpoint], topology)
            for mode_index, mode in enumerate(REVEAL_MODES):
                for ratio in UNKNOWN_RATIOS:
                    view, actual_fraction = _formal_view(
                        sequence,
                        spec,
                        checkpoint=checkpoint,
                        topology=topology,
                        mode_index=mode_index,
                        ratio=ratio,
                        normalizer=normalizer,
                    )
                    ambiguity = build_empirical_ambiguity_set(
                        view, calibration_radius=0.0
                    )
                    rows.append(
                        {
                            "family": spec.family,
                            "base_seed": spec.base_seed,
                            "sequence_id": spec.sequence_id,
                            "checkpoint": checkpoint,
                            "reveal_mode": mode,
                            "reveal_ratio": ratio,
                            "actual_entry_fraction": actual_fraction,
                            "observation_digest": hashlib.sha256(
                                _canonical_bytes(view)
                            ).hexdigest(),
                            "ambiguity_digest": hashlib.sha256(
                                ambiguity.to_canonical_bytes()
                            ).hexdigest(),
                            "score": calibration_exceedance_score(
                                ambiguity.lower_bounds,
                                ambiguity.upper_bounds,
                                descriptor,
                                normalizer.scale,
                            ),
                        }
                    )
    return rows


def _formal_validation_rows(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str, Any],
    *,
    topology: PublicTopologyView,
    normalizer: Any,
    radius: float,
    case_indices: Mapping[tuple[str, int, int, int, int, int], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        sequence = sequences[spec.sequence_id]
        for checkpoint in CHECKPOINTS:
            matrix = np.asarray(sequence.matrices[checkpoint])
            for mode_index, _ in enumerate(REVEAL_MODES):
                for stage_index, ratio in enumerate(UNKNOWN_RATIOS):
                    view, actual_fraction = _formal_view(
                        sequence,
                        spec,
                        checkpoint=checkpoint,
                        topology=topology,
                        mode_index=mode_index,
                        ratio=ratio,
                        normalizer=normalizer,
                    )
                    ambiguity = build_empirical_ambiguity_set(
                        view, calibration_radius=radius
                    )
                    observation_digest = hashlib.sha256(
                        _canonical_bytes(view)
                    ).hexdigest()
                    ambiguity_digest = hashlib.sha256(
                        ambiguity.to_canonical_bytes()
                    ).hexdigest()
                    for method in ORDINARY_METHODS:
                        case_index = case_indices[
                            (
                                "validation",spec.record_index,checkpoint,
                                mode_index,stage_index,8,
                            )
                        ]
                        distance,support_digest,replicate_count = (
                            _formal_support_evidence(
                                ambiguity,matrix,method=method,
                                case_index=case_index,
                            )
                        )
                        rows.append(
                            {
                                "sequence_id": spec.sequence_id,
                                "checkpoint": checkpoint,
                                "mode_index": mode_index,
                                "reveal_mode": REVEAL_MODES[mode_index],
                                "stage_index": stage_index,
                                "method": method,
                                "requested_k": 8,
                                "reveal_ratio": ratio,
                                "actual_entry_fraction": actual_fraction,
                                "nearest_rms_distance": distance,
                                "replicate_count": replicate_count,
                                "observation_digest": observation_digest,
                                "ambiguity_digest": ambiguity_digest,
                                "support_digest": support_digest,
                            }
                        )
    return rows


def _formal_support_evidence(
    ambiguity: Any,
    matrix: np.ndarray,
    *,
    method: str,
    case_index: int,
) -> tuple[float,str,int]:
    """Return the metric and digest of the exact support bytes used."""
    if method == "random_empirical":
        supports = [
            select_support(
                ambiguity,method=method,k=8,
                replicate_seed=replicate_seed(case_index,replicate),
            )
            for replicate in range(RANDOM_REPLICATES)
        ]
        distance = float(np.mean([
            truth_nearest_descriptor_distance(ambiguity,support,matrix)
            for support in supports
        ]))
        digest = hashlib.sha256(b"".join(
            support.to_canonical_bytes() for support in supports
        )).hexdigest()
        return distance,digest,RANDOM_REPLICATES
    support = select_support(ambiguity,method=method,k=8)
    return (
        truth_nearest_descriptor_distance(ambiguity,support,matrix),
        hashlib.sha256(support.to_canonical_bytes()).hexdigest(),
        1,
    )


def _choose_validation_for_count(
    rows: Sequence[Mapping[str, Any]], expected_sequences: int
) -> str:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sequence_id"]), str(row["method"]))].append(
            _finite_float(row["nearest_rms_distance"], "nearest RMS distance")
        )
    sequence_ids = sorted({key[0] for key in grouped})
    if len(sequence_ids) != expected_sequences:
        raise ValueError("validation sequence universe mismatch")
    means = {
        method: float(
            np.mean(
                [
                    np.mean(grouped[(sequence_id, method)])
                    for sequence_id in sequence_ids
                ]
            )
        )
        for method in ORDINARY_METHODS
    }
    minimum = min(means.values())
    tied = {method for method, value in means.items() if abs(value - minimum) <= 1e-12}
    return next(method for method in VALIDATION_TIE_ORDER if method in tied)


def _support_case_metrics(
    ambiguity: Any,
    support: Any,
    matrix: np.ndarray,
    *,
    physical_low: np.ndarray,
    physical_high: np.ndarray,
    total_threshold: float,
    group_indices: tuple[int, ...],
    group_thresholds: np.ndarray,
    requested_k: int,
) -> CaseMetrics:
    truth_descriptor = traffic_descriptor(matrix, ambiguity.topology)
    truth_hotspot = int(np.argmax(matrix.sum(axis=0)))
    support_hotspots = tuple(
        int(np.argmax(np.asarray(candidate).sum(axis=0)))
        for candidate in support.matrices
    )
    return compute_case_metrics(
        truth_descriptor=truth_descriptor,
        truth_matrix=matrix,
        support_descriptors=support.descriptor_vectors,
        support_matrices=np.stack(support.matrices),
        pool_descriptors=ambiguity.descriptor_vectors,
        fit_scale=ambiguity.normalizer.scale,
        lower=ambiguity.lower_bounds,
        upper=ambiguity.upper_bounds,
        physical_low=physical_low,
        physical_high=physical_high,
        truth_satisfies_observation=True,
        total_index=0,
        group_indices=group_indices,
        total_tail_threshold=total_threshold,
        group_tail_thresholds=group_thresholds,
        truth_hotspot_destination=truth_hotspot,
        support_hotspot_destinations=support_hotspots,
        requested_k=requested_k,
    )


def _average_case_metrics(values: Sequence[CaseMetrics]) -> dict[str, float]:
    return {
        field.name: float(np.mean([float(getattr(value, field.name)) for value in values]))
        for field in fields(CaseMetrics)
    }


def _formal_test_rows(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str, Any],
    *,
    topology: PublicTopologyView,
    normalizer: Any,
    radius: float,
    total_threshold: float,
    group_indices: tuple[int, ...],
    group_thresholds: np.ndarray,
    sequence_digests: Mapping[str, str],
    case_indices: Mapping[tuple[str, int, int, int, int, int], int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    physical_low, physical_high = physical_descriptor_bounds(
        topology, max_entry=FORMAL_MAX_ENTRY
    )
    raw_rows: list[dict[str, Any]] = []
    for spec in specs:
        sequence = sequences[spec.sequence_id]
        for checkpoint in CHECKPOINTS:
            matrix = np.asarray(sequence.matrices[checkpoint])
            for mode_index, mode in enumerate(REVEAL_MODES):
                for stage_index, ratio in enumerate(REVEAL_RATIOS):
                    view, actual_fraction = _formal_view(
                        sequence,
                        spec,
                        checkpoint=checkpoint,
                        topology=topology,
                        mode_index=mode_index,
                        ratio=ratio,
                        normalizer=normalizer,
                    )
                    construction_started = time.perf_counter()
                    ambiguity = build_empirical_ambiguity_set(
                        view, calibration_radius=radius
                    )
                    construction_seconds = (
                        time.perf_counter() - construction_started
                    )
                    ambiguity_digest = hashlib.sha256(
                        ambiguity.to_canonical_bytes()
                    ).hexdigest()
                    observation_digest = hashlib.sha256(
                        _canonical_bytes(view)
                    ).hexdigest()
                    for requested_k in REQUESTED_K:
                        case_index = case_indices[
                            (
                                "test",
                                spec.record_index,
                                checkpoint,
                                mode_index,
                                stage_index,
                                requested_k,
                            )
                        ]
                        for method in ALL_METHODS:
                            if method == "random_empirical" and ratio < 1.0:
                                selector_started = time.perf_counter()
                                supports = [
                                    select_support(
                                        ambiguity,
                                        method=method,
                                        k=requested_k,
                                        replicate_seed=replicate_seed(case_index, replicate),
                                    )
                                    for replicate in range(RANDOM_REPLICATES)
                                ]
                                selector_seconds = (
                                    time.perf_counter() - selector_started
                                )
                                metric_values = [
                                    _support_case_metrics(
                                        ambiguity,
                                        support,
                                        matrix,
                                        physical_low=physical_low,
                                        physical_high=physical_high,
                                        total_threshold=total_threshold,
                                        group_indices=group_indices,
                                        group_thresholds=group_thresholds,
                                        requested_k=requested_k,
                                    )
                                    for support in supports
                                ]
                                metrics = _average_case_metrics(metric_values)
                                support_digest = hashlib.sha256(
                                    b"".join(support.to_canonical_bytes() for support in supports)
                                ).hexdigest()
                                uses_oracle = False
                                upper_bound_only = False
                            else:
                                selector_started = time.perf_counter()
                                support = (
                                    oracle_support_upper_bound(
                                        ambiguity, truth=matrix, k=requested_k
                                    )
                                    if method == "oracle_support_upper_bound"
                                    else select_support(
                                        ambiguity,
                                        method=method,
                                        k=requested_k,
                                        **(
                                            {"replicate_seed": replicate_seed(case_index, 0)}
                                            if method == "random_empirical"
                                            else {}
                                        ),
                                    )
                                )
                                selector_seconds = (
                                    time.perf_counter() - selector_started
                                )
                                metric = _support_case_metrics(
                                    ambiguity,
                                    support,
                                    matrix,
                                    physical_low=physical_low,
                                    physical_high=physical_high,
                                    total_threshold=total_threshold,
                                    group_indices=group_indices,
                                    group_thresholds=group_thresholds,
                                    requested_k=requested_k,
                                )
                                metrics = {
                                    field.name: getattr(metric, field.name)
                                    for field in fields(CaseMetrics)
                                }
                                support_digest = hashlib.sha256(
                                    support.to_canonical_bytes()
                                ).hexdigest()
                                uses_oracle = bool(support.uses_oracle)
                                upper_bound_only = bool(support.upper_bound_only)
                            metrics = dict(metrics)
                            metrics["total_tail_events"] = metrics.pop(
                                "total_tail_event"
                            )
                            metrics["total_tail_hits"] = metrics.pop(
                                "total_tail_hit"
                            )
                            raw_rows.append(
                                {
                                    "case_id": f"case-{case_index}-{method}",
                                    "case_index": case_index,
                                    "sequence_id": spec.sequence_id,
                                    "split": "test",
                                    "family": spec.family,
                                    "base_seed": spec.base_seed,
                                    "record_index": spec.record_index,
                                    "checkpoint": checkpoint,
                                    "reveal_mode": mode,
                                    "reveal_ratio": ratio,
                                    "actual_entry_fraction": actual_fraction,
                                    "requested_k": requested_k,
                                    "construction_seed": reveal_seed(
                                        spec.record_index, checkpoint, mode_index
                                    ),
                                    "method": method,
                                    **metrics,
                                    "uses_oracle": uses_oracle,
                                    "upper_bound_only": upper_bound_only,
                                    "invalid_or_empty": 0,
                                    "construction_seconds": construction_seconds,
                                    "selector_seconds": selector_seconds,
                                    "sequence_digest": sequence_digests[spec.sequence_id],
                                    "observation_digest": observation_digest,
                                    "ambiguity_digest": ambiguity_digest,
                                    "support_digest": support_digest,
                                }
                            )
    sequence_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        if float(row["reveal_ratio"]) < 1.0:
            grouped[
                (str(row["sequence_id"]), str(row["method"]), int(row["requested_k"]))
            ].append(row)
    spec_by_id = {spec.sequence_id: spec for spec in specs}
    for identity in sorted(grouped):
        aggregate = aggregate_sequence_metrics(grouped[identity])
        spec = spec_by_id[identity[0]]
        sequence_rows.append(
            {
                **asdict(aggregate),
                "family": spec.family,
                "base_seed": spec.base_seed,
            }
        )
    return raw_rows, sequence_rows


def _formal_lofo_evidence(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str, Any],
    *,
    topology: PublicTopologyView,
    case_indices: Mapping[tuple[str, int, int, int, int, int], int],
) -> tuple[float, dict[str, float], dict[str, float]]:
    family_deltas: dict[str, float] = {}
    degradation: dict[str, float] = {}
    for held_family in FORMAL_FAMILIES:
        fold = build_lofo_fold(specs, held_out_family=held_family)
        fit_matrices = tuple(
            matrix
            for spec in fold.fit
            for matrix in sequences[spec.sequence_id].matrices
        )
        normalizer = fit_descriptor_normalizer(fit_matrices, topology)
        calibration_rows = _formal_calibration_rows(
            fold.calibration,
            sequences,
            topology=topology,
            normalizer=normalizer,
        )
        radius = calibrate_envelope_radius(
            calibration_rows, held_out_family=held_family
        )
        validation_rows = _formal_validation_rows(
            fold.validation,
            sequences,
            topology=topology,
            normalizer=normalizer,
            radius=radius,
            case_indices=case_indices,
        )
        method = _choose_validation_for_count(validation_rows, 12)
        selected_distances: list[float] = []
        random_distances: list[float] = []
        sequence_deltas: list[float] = []
        for spec in fold.test:
            sequence_selected: list[float] = []
            sequence_random: list[float] = []
            sequence = sequences[spec.sequence_id]
            for checkpoint in CHECKPOINTS:
                matrix = np.asarray(sequence.matrices[checkpoint])
                for mode_index in range(len(REVEAL_MODES)):
                    for stage_index, ratio in enumerate(UNKNOWN_RATIOS):
                        view, _ = _formal_view(
                            sequence,
                            spec,
                            checkpoint=checkpoint,
                            topology=topology,
                            mode_index=mode_index,
                            ratio=ratio,
                            normalizer=normalizer,
                        )
                        ambiguity = build_empirical_ambiguity_set(
                            view, calibration_radius=radius
                        )
                        case_index = case_indices[
                            (
                                "test",
                                spec.record_index,
                                checkpoint,
                                mode_index,
                                stage_index,
                                8,
                            )
                        ]
                        random_values = [
                            truth_nearest_descriptor_distance(
                                ambiguity,
                                select_support(
                                    ambiguity,
                                    method="random_empirical",
                                    k=8,
                                    replicate_seed=replicate_seed(
                                        case_index, replicate
                                    ),
                                ),
                                matrix,
                            )
                            for replicate in range(RANDOM_REPLICATES)
                        ]
                        random_distance = float(np.mean(random_values))
                        if method == "random_empirical":
                            selected_distance = random_distance
                        else:
                            selected_distance = truth_nearest_descriptor_distance(
                                ambiguity,
                                select_support(ambiguity, method=method, k=8),
                                matrix,
                            )
                        sequence_selected.append(selected_distance)
                        sequence_random.append(random_distance)
            selected_mean = float(np.mean(sequence_selected))
            random_mean = float(np.mean(sequence_random))
            selected_distances.append(selected_mean)
            random_distances.append(random_mean)
            sequence_deltas.append(random_mean - selected_mean)
        family_deltas[held_family] = float(np.mean(sequence_deltas))
        selected_mean = float(np.mean(selected_distances))
        random_mean = float(np.mean(random_distances))
        degradation[held_family] = (
            0.0 if random_mean == 0.0 else (selected_mean - random_mean) / random_mean
        )
    return float(np.mean(list(family_deltas.values()))), family_deltas, degradation


def _run_formal_schema_v1_disabled(output_directory: str | Path) -> dict[str, Any]:
    """Run the frozen formal design and write artifacts only after full validation."""

    project_root = Path(__file__).resolve().parents[2]
    protocol_path = (
        project_root / "docs" / "uncertainty_aiccl" / "PHASE3B_AMBIGUITY_PROTOCOL.md"
    )
    if _file_sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Phase 3B protocol SHA-256 mismatch")
    source_files_sha256 = {
        relative_path: _file_sha256(project_root / relative_path)
        for relative_path in (
            "rlccl/uncertainty/ambiguity.py",
            "rlccl/uncertainty/ambiguity_experiment.py",
            "scripts/run_phase3b_ambiguity.py",
        )
    }
    h1_digests, h1_manifest_digest = _load_h1_digest_exclusion(project_root)
    topology_info, topology_digest = _load_rear4_topology(project_root)
    topology = PublicTopologyView.from_topology_info(topology_info)
    specs = build_formal_sequence_specs()
    case_indices = _case_index_lookup(specs)
    sequences: dict[str, Any] = {}
    sequence_records: list[dict[str, Any]] = []
    for spec in specs:
        sequence = generate_long_horizon_sequence(
            LongHorizonTrafficConfig(**spec.generator_config)
        )
        if len(sequence.matrices) != FORMAL_SEQUENCE_LENGTH:
            raise ValueError("formal generator returned wrong sequence length")
        digest = _sequence_sha256(sequence.matrices)
        sequences[spec.sequence_id] = sequence
        sequence_records.append(
            {
                "sequence_id": spec.sequence_id,
                "split": spec.split,
                "sequence_digest": digest,
                "record_index": spec.record_index,
            }
        )
    validate_sequence_records(
        sequence_records, h1_sequence_digests=h1_digests
    )
    sequence_digests = {
        row["sequence_id"]: row["sequence_digest"] for row in sequence_records
    }
    fit_specs = tuple(spec for spec in specs if spec.split == "fit")
    fit_matrices = tuple(
        matrix for spec in fit_specs for matrix in sequences[spec.sequence_id].matrices
    )
    normalizer = fit_descriptor_normalizer(fit_matrices, topology)
    fit_descriptors = np.stack(
        [traffic_descriptor(matrix, topology) for matrix in fit_matrices]
    )
    group_start = 2 * topology.num_nodes + 3
    group_indices = tuple(range(group_start, fit_descriptors.shape[1]))
    total_threshold = float(np.quantile(fit_descriptors[:, 0], 0.9, method="linear"))
    group_thresholds = np.quantile(
        fit_descriptors[:, group_indices], 0.9, axis=0, method="linear"
    )

    calibration_specs = tuple(spec for spec in specs if spec.split == "calibration")
    calibration_rows = _formal_calibration_rows(
        calibration_specs,
        sequences,
        topology=topology,
        normalizer=normalizer,
    )
    radius = calibrate_envelope_radius(calibration_rows)
    validation_specs = tuple(spec for spec in specs if spec.split == "validation")
    validation_rows = _formal_validation_rows(
        validation_specs,
        sequences,
        topology=topology,
        normalizer=normalizer,
        radius=radius,
        case_indices=case_indices,
    )
    selection = select_validation_method(validation_rows)
    test_specs = tuple(spec for spec in specs if spec.split == "test")
    raw_cases, raw_sequences = _formal_test_rows(
        test_specs,
        sequences,
        topology=topology,
        normalizer=normalizer,
        radius=radius,
        total_threshold=total_threshold,
        group_indices=group_indices,
        group_thresholds=group_thresholds,
        sequence_digests=sequence_digests,
        case_indices=case_indices,
    )
    test_total_traffic_dependence = _formal_test_dependence(
        test_specs, sequences
    )

    # Fail closed before any artifact directory is created.  The exact raw
    # validator checks the 120,000-row Cartesian universe, registry IDs and
    # seeds, oracle isolation, and ratio-1 singleton actual K.  Recompute then
    # checks the exact 300-row sequence aggregation universe against raw data.
    integrity_checks_complete = False
    integrity_checks_passed = False
    validate_raw_case_rows(raw_cases, allow_incomplete_universe=False)
    recompute_artifacts(
        {
            "protocol_sha256": PROTOCOL_SHA256,
            "selected_method": selection.method,
            "selected_k": 8,
            "data_status": "HOLD",
        },
        raw_cases,
        raw_sequences,
        allow_incomplete_universe=False,
    )
    integrity_checks_complete = True
    integrity_checks_passed = True

    selected_rows = [
        row for row in raw_cases
        if row["method"] == selection.method
        and int(row["requested_k"]) == 8
        and float(row["reveal_ratio"]) < 1.0
    ]
    selected_sequence_rows = {
        str(row["sequence_id"]): row
        for row in raw_sequences
        if row["method"] == selection.method and int(row["requested_k"]) == 8
    }
    random_sequence_rows = {
        str(row["sequence_id"]): row
        for row in raw_sequences
        if row["method"] == "random_empirical" and int(row["requested_k"]) == 8
    }
    paired_rows = []
    for spec in test_specs:
        selected_row = selected_sequence_rows[spec.sequence_id]
        random_row = random_sequence_rows[spec.sequence_id]
        paired_rows.append(
            {
                "family": spec.family,
                "base_seed": spec.base_seed,
                "sequence_id": spec.sequence_id,
                "paired_delta": float(random_row["nearest_rms_distance"])
                - float(selected_row["nearest_rms_distance"]),
            }
        )
    bootstrap = family_stratified_sequence_bootstrap(paired_rows)
    lofo_aggregate, lofo_by_family, lofo_degradation = _formal_lofo_evidence(
        specs, sequences, topology=topology, case_indices=case_indices
    )
    coverage_by_family = {
        family: float(
            np.mean(
                [
                    float(row["joint_coverage"])
                    for row in selected_rows
                    if row["family"] == family
                ]
            )
        )
        for family in FORMAL_FAMILIES
    }
    evidence = {
        "selected_joint_coverage": float(
            np.mean([float(row["joint_coverage"]) for row in selected_rows])
        ),
        "selected_joint_coverage_by_family": coverage_by_family,
        "paired_delta_ci95": [bootstrap.ci_lower, bootstrap.ci_upper],
        "paired_delta_by_base_seed": {
            str(seed): float(
                np.mean(
                    [row["paired_delta"] for row in paired_rows if row["base_seed"] == seed]
                )
            )
            for seed in FORMAL_BASE_SEEDS
        },
        "paired_delta_by_family": {
            family: float(
                np.mean(
                    [row["paired_delta"] for row in paired_rows if row["family"] == family]
                )
            )
            for family in FORMAL_FAMILIES
        },
        "lofo_aggregate_delta": lofo_aggregate,
        "lofo_delta_by_family": lofo_by_family,
        "lofo_relative_degradation_by_family": lofo_degradation,
        "total_tail_hits": float(sum(row["total_tail_hits"] for row in selected_rows)),
        "total_tail_events": float(sum(row["total_tail_events"] for row in selected_rows)),
        "group_tail_hits": float(sum(row["group_tail_hits"] for row in selected_rows)),
        "group_tail_events": float(sum(row["group_tail_events"] for row in selected_rows)),
        "hotspot_hits": float(sum(row["hotspot_hit"] for row in selected_rows)),
        "hotspot_events": float(sum(row["hotspot_event"] for row in selected_rows)),
        "selected_mean_physical_normalized_width": float(
            np.mean(
                [row["physical_normalized_mean_width"] for row in selected_rows]
            )
        ),
        "ordinary_invalid_or_empty_rate": {
            method: float(
                np.mean(
                    [
                        row["invalid_or_empty"]
                        for row in raw_cases
                        if row["method"] == method and float(row["reveal_ratio"]) < 1.0
                    ]
                )
            )
            for method in ORDINARY_METHODS
        },
        "ratio1_singleton_coverage": float(
            np.mean(
                [
                    row["joint_coverage"]
                    for row in raw_cases
                    if row["method"] == selection.method
                    and int(row["requested_k"]) == 8
                    and float(row["reveal_ratio"]) == 1.0
                ]
            )
        ),
        "all_timings_finite": all(
            math.isfinite(float(row["construction_seconds"]))
            and math.isfinite(float(row["selector_seconds"]))
            for row in raw_cases
        ),
        "integrity_checks_complete": integrity_checks_complete,
        "integrity_checks_passed": integrity_checks_passed,
    }
    decision = evaluate_phase3b_gate(evidence)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": PROTOCOL_SHA256,
        "families": list(FORMAL_FAMILIES),
        "base_seeds": list(FORMAL_BASE_SEEDS),
        "splits": list(FORMAL_SPLITS),
        "sequence_length": FORMAL_SEQUENCE_LENGTH,
        "max_entry": FORMAL_MAX_ENTRY,
        "history_window": HISTORY_WINDOW,
        "checkpoints": list(CHECKPOINTS),
        "reveal_modes": list(REVEAL_MODES),
        "reveal_ratios": list(REVEAL_RATIOS),
        "requested_k": list(REQUESTED_K),
        "random_replicates": RANDOM_REPLICATES,
        "sequence_specs": [_canonical(spec) for spec in specs],
        "sequence_records": sequence_records,
        "h1_exclusion_manifest_sha256": h1_manifest_digest,
        "h1_excluded_sequence_digests": sorted(h1_digests),
        "topology": {"name": "Rear4GPU", "sha256": topology_digest},
        "normalizer_digest": normalizer.digest,
        "group_coefficients_digest": group_coefficients_digest(topology),
        "ambiguity_provenance": {
            "normalizer_digest": normalizer.digest,
            "group_coefficients_digest": group_coefficients_digest(topology),
            "history_cutoffs": list(CHECKPOINTS),
            "construction_seed_formula": (
                "31000000 + record_index*100000 + checkpoint*10 + mode_index"
            ),
        },
        "authorized_source_sha256": source_files_sha256,
        "test_total_traffic_dependence": test_total_traffic_dependence,
        "calibration_radius": radius,
        "selected_method": selection.method,
        "selected_k": 8,
        "validation_method_means": dict(selection.method_means),
        "gate_evidence": evidence,
        "data_status": decision.data_status,
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    recomputed = recompute_artifacts(
        manifest,
        raw_cases,
        raw_sequences,
        allow_incomplete_universe=False,
    )
    return write_experiment_artifacts(
        output_directory,
        manifest=manifest,
        raw_case_rows=raw_cases,
        raw_sequence_rows=raw_sequences,
        summary=recomputed.summary,
    )


# ---------------------------------------------------------------------------
# Schema-v2 integrity layer
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
# Publication-state sentinel: provisional evidence is fully recomputed, but its
# gate decision is deliberately non-final until integrity flags are committed.
_PROVISIONAL_DATA_STATUS = "HOLD"
AUTHORIZED_SOURCE_KEYS = {
    "rlccl/uncertainty/ambiguity.py",
    "rlccl/uncertainty/ambiguity_experiment.py",
    "scripts/run_phase3b_ambiguity.py",
}
ENVIRONMENT_KEYS = {"python", "python_executable", "numpy", "platform"}
RAW_ROW_COUNTS = {
    "raw_calibration_scores.csv": 4_800,
    "raw_validation_metrics.csv": 19_200,
    "raw_case_metrics.csv": 120_000,
    "raw_sequence_metrics.csv": 300,
    "raw_lofo_calibration_scores.csv": 19_200,
    "raw_lofo_validation_metrics.csv": 76_800,
    "raw_lofo_test_metrics.csv": 9_600,
    "raw_dependence_metrics.csv": 240,
}
RAW_TABLE_SCHEMAS = {
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
RAW_TABLE_IDENTITIES = {
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _v2_scalar(value: Any) -> Any:
    if value is None:
        return ["n", None]
    if type(value) is bool:
        return ["b", "true" if value else "false"]
    if type(value) is int:
        return ["i", str(value)]
    if type(value) is float:
        number = 0.0 if value == 0.0 else value
        if not math.isfinite(number):
            raise ValueError("canonical float must be finite")
        return ["f", number.hex().lower()]
    if type(value) is str:
        return ["s", value]
    if type(value) in {list, tuple}:
        return ["l", [_v2_scalar(item) for item in value]]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical mapping keys must be strings")
        return ["m", [[key, _v2_scalar(value[key])] for key in sorted(value)]]
    if isinstance(value, np.generic):
        return _v2_scalar(value.item())
    raise TypeError(f"unsupported canonical type: {type(value)!r}")


def canonical_object_sha256(value: Any) -> str:
    encoded = json.dumps(
        _v2_scalar(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_v2_type(value: Any, kind: str, name: str) -> None:
    if kind == "s":
        valid = type(value) is str
    elif kind == "i":
        valid = type(value) is int
    elif kind == "b":
        valid = type(value) is bool
    elif kind == "f":
        valid = type(value) is float and math.isfinite(value)
    elif kind == "f?":
        valid = value is None or (type(value) is float and math.isfinite(value))
    else:  # pragma: no cover - frozen schema above
        raise AssertionError(kind)
    if not valid:
        if kind in {"f", "f?"} and type(value) is float and not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        raise ValueError(f"{name} violates strict {kind} type")


def canonical_table_sha256(
    table_name: str,
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    scientific: bool = False,
) -> str:
    if table_name not in RAW_TABLE_SCHEMAS:
        raise ValueError("unknown raw table")
    schema = RAW_TABLE_SCHEMAS[table_name]
    expected_header = tuple(name for name, _ in schema)
    if tuple(header) != expected_header:
        raise ValueError("raw table column order mismatch")
    identities = RAW_TABLE_IDENTITIES[table_name]
    typed_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in rows:
        if set(source) != set(expected_header):
            raise ValueError("raw row schema mismatch")
        for name, kind in schema:
            _validate_v2_type(source[name], kind, name)
        identity = tuple(source[name] for name in identities)
        if identity in seen:
            raise ValueError("duplicate raw table identity")
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


def combined_scientific_evidence_sha256(digests: Mapping[str, str]) -> str:
    csv_names = ARTIFACT_NAMES[1:-1]
    if set(digests) != set(csv_names):
        raise ValueError("scientific digest map must contain exactly eight raw tables")
    ordered = [[name, str(digests[name])] for name in csv_names]
    if any(_SHA256_RE.fullmatch(value) is None for _, value in ordered):
        raise ValueError("scientific digest must be lowercase SHA-256")
    return hashlib.sha256(
        json.dumps(
            ["phase3b-scientific-v1", ordered],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _v2_header(table_name: str) -> tuple[str, ...]:
    return tuple(name for name, _ in RAW_TABLE_SCHEMAS[table_name])


def _v2_assert_digest(value: Any, name: str) -> str:
    digest = str(value)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _v2_close(actual: Any, expected: Any, label: str, *, atol: float = 1e-12) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label} derived mismatch")
        return
    if type(expected) in {int, float} and type(expected) is not bool:
        if type(actual) not in {int, float} or type(actual) is bool:
            raise ValueError(f"{label} derived type mismatch")
        if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > atol:
            raise ValueError(f"{label} derived mismatch")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} derived mapping mismatch")
        for key in expected:
            _v2_close(actual[key], expected[key], f"{label}.{key}", atol=atol)
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"{label} derived sequence mismatch")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _v2_close(left, right, f"{label}[{index}]", atol=atol)
        return
    if actual != expected:
        raise ValueError(f"{label} derived mismatch")


def _v2_validate_case_id(table_name: str, row: Mapping[str, Any]) -> None:
    if table_name == "raw_sequence_metrics":
        return
    if table_name == "raw_case_metrics":
        expected = f"case-{row['case_index']}-{row['method']}"
    else:
        expected = table_name + ":" + ":".join(
            str(row[name]) for name in RAW_TABLE_IDENTITIES[table_name]
        )
    if row["case_id"] != expected:
        raise ValueError("case identity/domain does not match raw coordinates or role/method")


def validate_schema_v2_table(
    table_name: str,
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_incomplete_universe: bool = False,
    expected_provenance: Mapping[str, Any] | None = None,
    expected_derived: Mapping[str, Any] | None = None,
) -> bool:
    values = [dict(row) for row in rows]
    canonical_table_sha256(table_name, header, values)
    provenance = dict(expected_provenance or {})
    sequence_records = provenance.get("sequence_records", {})
    fold_normalizers = provenance.get("lofo_fold_normalizer_digests", {})
    coordinate_digests: dict[tuple[Any, ...], tuple[str, str]] = {}
    for row in values:
        _v2_validate_case_id(table_name, row)
        for name in row:
            if name.endswith("_digest"):
                _v2_assert_digest(row[name], name)
        if "family" in row and row["family"] not in FORMAL_FAMILIES:
            raise ValueError("unknown family domain")
        if "split" in row and row["split"] not in set(FORMAL_SPLITS):
            raise ValueError("unknown split domain")
        if "checkpoint" in row and row["checkpoint"] not in CHECKPOINTS:
            raise ValueError("unknown checkpoint domain")
        if "reveal_mode" in row and row["reveal_mode"] not in REVEAL_MODES:
            raise ValueError("unknown reveal mode domain")
        if "mode_index" in row and row["mode_index"] != REVEAL_MODES.index(row["reveal_mode"]):
            raise ValueError("mode index/domain mismatch")
        if "construction_seed" in row:
            expected_construction_seed = reveal_seed(
                row["record_index"], row["checkpoint"], row["mode_index"]
            )
            if row["construction_seed"] != expected_construction_seed:
                raise ValueError("construction seed/reveal coordinate mismatch")
        if "stage_index" in row:
            ratios = REVEAL_RATIOS if table_name == "raw_case_metrics" else UNKNOWN_RATIOS
            if row["reveal_ratio"] not in ratios or row["stage_index"] != ratios.index(row["reveal_ratio"]):
                raise ValueError("stage index/reveal ratio domain mismatch")
        if "actual_entry_fraction" in row and not 0.0 <= row["actual_entry_fraction"] <= 1.0:
            raise ValueError("actual entry fraction domain violation")
        if "requested_k" in row and row["requested_k"] not in REQUESTED_K:
            raise ValueError("requested K domain violation")
        if table_name in {"raw_validation_metrics", "raw_lofo_validation_metrics", "raw_lofo_test_metrics"}:
            if row["requested_k"] != 8 or row["method"] not in ORDINARY_METHODS:
                raise ValueError("validation/LOFO method must be ordinary K8")
            if row["reveal_ratio"] >= 1.0:
                raise ValueError("validation/LOFO evidence must use unknown ratios")
            if row["uses_oracle"] or row["upper_bound_only"]:
                raise ValueError("oracle row is forbidden from validation/LOFO")
            if row["nearest_rms_distance"] < 0.0:
                raise ValueError("nearest distance domain violation")
            expected_replicates = RANDOM_REPLICATES if row["method"] == "random_empirical" else 1
            if row["replicate_count"] != expected_replicates:
                raise ValueError("validation/LOFO replicate-count domain violation")
        if table_name in {"raw_calibration_scores", "raw_lofo_calibration_scores"} and row["score"] < 0.0:
            raise ValueError("calibration score domain violation")
        if table_name.startswith("raw_lofo_"):
            held_out_family = row["held_out_family"]
            if held_out_family not in FORMAL_FAMILIES:
                raise ValueError("LOFO held-out family domain violation")
            expected_fold_id = (
                f"lofo-{FORMAL_FAMILIES.index(held_out_family)}-{held_out_family}"
            )
            if row["fold_id"] != expected_fold_id:
                raise ValueError("LOFO fold identity/held-out family mismatch")
            if row["fold_id"] not in fold_normalizers:
                raise ValueError("LOFO fold normalizer provenance missing")
            if row["normalizer_digest"] != fold_normalizers[row["fold_id"]]:
                raise ValueError("LOFO fold normalizer digest mismatch")
            if table_name == "raw_lofo_test_metrics":
                if row["family"] != row["held_out_family"] or row["role"] not in {"selected", "random_comparator"}:
                    raise ValueError("LOFO test family/role domain violation")
                if row["role"] == "random_comparator" and row["method"] != "random_empirical":
                    raise ValueError("LOFO random role method mismatch")
            elif row["family"] == row["held_out_family"]:
                raise ValueError("LOFO held-out family leaked into training evidence")
        if table_name == "raw_dependence_metrics" and row["total_traffic"] < 0.0:
            raise ValueError("dependence total traffic domain violation")
        if table_name == "raw_case_metrics":
            if row["method"] not in ALL_METHODS:
                raise ValueError("unknown test method")
            oracle = row["method"] == "oracle_support_upper_bound"
            if row["uses_oracle"] != oracle or row["upper_bound_only"] != oracle:
                raise ValueError("oracle isolation flags invalid")
            for name in (
                "nearest_rms_distance","nearest_matrix_l1_distance","covering_radius",
                "mean_pairwise_diversity","duplicate_fraction","component_coverage",
                "joint_coverage","physical_normalized_mean_width","total_tail_events",
                "total_tail_hits","group_tail_events","group_tail_hits","hotspot_events",
                "hotspot_hits","invalid_or_empty","construction_seconds","selector_seconds",
            ):
                if row[name] < 0.0:
                    raise ValueError(f"{name} finite nonnegative domain violation")
            if row["actual_k"] <= 0 or row["actual_k"] > row["requested_k"]:
                raise ValueError("actual K domain violation")
            if row["reveal_ratio"] == 1.0:
                if row["actual_k"] != 1 or row["replicate_count"] != 1:
                    raise ValueError("ratio1 actual K/replicate must be singleton")
            else:
                if row["actual_k"] != row["requested_k"]:
                    raise ValueError("unknown-ratio actual K must equal requested K")
                expected_replicates = RANDOM_REPLICATES if row["method"] == "random_empirical" else 1
                if row["replicate_count"] != expected_replicates:
                    raise ValueError("raw-case replicate-count domain violation")
            for name in (
                "actual_entry_fraction","duplicate_fraction","component_coverage",
                "joint_coverage","invalid_or_empty",
            ):
                if not 0.0 <= row[name] <= 1.0:
                    raise ValueError(f"{name} fraction/coverage domain violation")
            if row["zero_physical_range_components"] < 0:
                raise ValueError("zero physical range component count domain violation")
            for event_name,hit_name in (
                ("total_tail_events","total_tail_hits"),
                ("group_tail_events","group_tail_hits"),
                ("hotspot_events","hotspot_hits"),
            ):
                if row[hit_name] > row[event_name]:
                    raise ValueError(f"{hit_name} cannot exceed {event_name}")
        if table_name == "raw_sequence_metrics":
            if row["method"] not in ALL_METHODS or row["raw_case_count"] <= 0:
                raise ValueError("sequence aggregate domain violation")
            if row["nearest_rms_distance"] < 0.0:
                raise ValueError("sequence distance domain violation")
            for event_name,hit_name,recall_name in (
                ("total_tail_events","total_tail_hits","total_tail_recall"),
                ("group_tail_events","group_tail_hits","group_tail_recall"),
                ("hotspot_events","hotspot_hits","hotspot_recall"),
            ):
                events = row[event_name]
                hits = row[hit_name]
                recall = row[recall_name]
                if events < 0.0 or hits < 0.0 or hits > events:
                    raise ValueError("sequence event/hit domain violation")
                if events == 0.0:
                    if recall is not None:
                        raise ValueError("sequence recall must be null iff events are zero")
                elif recall is None or abs(recall-hits/events) > 1e-12:
                    raise ValueError("sequence recall must exactly equal hits/events")
        if sequence_records and "sequence_id" in row:
            if row["sequence_id"] not in sequence_records:
                raise ValueError("sequence provenance is absent from manifest")
            record = sequence_records[row["sequence_id"]]
            for name in ("split", "family", "base_seed", "record_index", "sequence_digest", "generator_config_digest"):
                if name in row and name in record and row[name] != record[name]:
                    raise ValueError(f"sequence {name} provenance mismatch")
        if "topology_digest" in row and "topology_digest" in provenance:
            if row["topology_digest"] != provenance["topology_digest"]:
                raise ValueError("topology digest provenance mismatch")
        if "normalizer_digest" in row and not table_name.startswith("raw_lofo_") and provenance.get("normalizer_digest") is not None:
            if row["normalizer_digest"] != provenance["normalizer_digest"]:
                raise ValueError("normalizer digest provenance mismatch")
        if {"record_index","checkpoint","mode_index","stage_index","observation_digest","ambiguity_digest"}.issubset(row):
            coordinate = (row["record_index"], row["checkpoint"], row["mode_index"], row["stage_index"], row.get("fold_id"))
            digests = (row["observation_digest"], row["ambiguity_digest"])
            if coordinate in coordinate_digests and coordinate_digests[coordinate] != digests:
                raise ValueError("observation/ambiguity digest differs at same coordinate")
            coordinate_digests[coordinate] = digests
        if "support_digest" in row:
            support_digests = provenance.get("support_digests", {})
            if row.get("case_id") in support_digests and row["support_digest"] != support_digests[row["case_id"]]:
                raise ValueError("support digest provenance mismatch")
            support_bytes = provenance.get("support_canonical_bytes", {})
            if row.get("case_id") in support_bytes:
                expected = hashlib.sha256(bytes(support_bytes[row["case_id"]])).hexdigest()
                if row["support_digest"] != expected:
                    raise ValueError("support canonical bytes digest mismatch")
    if table_name.startswith("raw_lofo_") and fold_normalizers:
        used = {row["fold_id"] for row in values}
        if len({fold_normalizers[fold] for fold in used}) != len(used):
            raise ValueError("LOFO fold normalizer digests must be unique and bound")
    return True


def _v2_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError(f"cannot derive {field} from an empty raw selection")
    return float(np.mean([float(row[field]) for row in rows]))


def validate_gate_raw_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_provenance: Mapping[str, Any],
    expected_derived: Mapping[str, Any],
    allow_incomplete_universe: bool = True,
) -> Mapping[str, Any]:
    values = [dict(row) for row in rows]
    validate_schema_v2_table(
        "raw_case_metrics", _v2_header("raw_case_metrics"), values,
        expected_provenance=expected_provenance,
        allow_incomplete_universe=allow_incomplete_universe,
    )
    method = str(expected_derived["selected_method"])
    selected_k = int(expected_derived["selected_k"])
    selected = [
        row for row in values
        if row["method"] == method and row["requested_k"] == selected_k
        and row["reveal_ratio"] < 1.0
    ]
    random_rows = [
        row for row in values
        if row["method"] == "random_empirical" and row["requested_k"] == selected_k
        and row["reveal_ratio"] < 1.0
    ]
    ratio1 = [
        row for row in values
        if row["method"] == method and row["requested_k"] == selected_k
        and row["reveal_ratio"] == 1.0
    ]
    selected_nearest = _v2_mean(selected, "nearest_rms_distance")
    random_nearest = _v2_mean(random_rows, "nearest_rms_distance")
    actual = {
        "selected_method":method,
        "selected_k":selected_k,
        "selected_nearest_rms_distance":selected_nearest,
        "random_nearest_rms_distance":random_nearest,
        "paired_delta":random_nearest - selected_nearest,
        "bootstrap_mean_delta":random_nearest - selected_nearest,
        "component_coverage":_v2_mean(selected, "component_coverage"),
        "joint_coverage":_v2_mean(selected, "joint_coverage"),
        "physical_normalized_mean_width":_v2_mean(selected, "physical_normalized_mean_width"),
        "total_tail_events":sum(float(row["total_tail_events"]) for row in selected),
        "total_tail_hits":sum(float(row["total_tail_hits"]) for row in selected),
        "group_tail_events":sum(float(row["group_tail_events"]) for row in selected),
        "group_tail_hits":sum(float(row["group_tail_hits"]) for row in selected),
        "hotspot_events":sum(float(row["hotspot_events"]) for row in selected),
        "hotspot_hits":sum(float(row["hotspot_hits"]) for row in selected),
        "invalid_or_empty_rate":_v2_mean(selected, "invalid_or_empty"),
        "ratio1_joint_coverage":_v2_mean(ratio1, "joint_coverage"),
        "ratio1_actual_k":next(iter({int(row["actual_k"]) for row in ratio1})),
        "all_timings_finite_nonnegative":all(
            math.isfinite(float(row["construction_seconds"]))
            and math.isfinite(float(row["selector_seconds"]))
            and float(row["construction_seconds"]) >= 0.0
            and float(row["selector_seconds"]) >= 0.0
            for row in values
        ),
    }
    _v2_close(actual, expected_derived, "gate raw evidence")
    return actual


def _v2_validate_row_provenance(
    row: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> None:
    records = expected_provenance.get("sequence_records", {})
    record = records.get(row.get("sequence_id"))
    if not isinstance(record, Mapping):
        raise ValueError("sequence provenance missing")
    for name in ("split","family","base_seed","record_index","sequence_digest","generator_config_digest"):
        if name in row and row[name] != record[name]:
            raise ValueError(f"sequence {name} provenance mismatch")
    if "topology_digest" in row and row["topology_digest"] != expected_provenance["topology_digest"]:
        raise ValueError("topology digest provenance mismatch")
    if "normalizer_digest" in row and row["normalizer_digest"] != expected_provenance["normalizer_digest"]:
        raise ValueError("normalizer digest provenance mismatch")


def validate_test_raw_aggregates(
    raw_case_rows: Sequence[Mapping[str, Any]],
    raw_sequence_rows: Sequence[Mapping[str, Any]],
    *,
    expected_provenance: Mapping[str, Any],
    expected_derived: Mapping[str, Any],
    exact: bool = True,
) -> Mapping[str, Any]:
    cases = list(raw_case_rows)
    sequences = list(raw_sequence_rows)
    if exact:
        if len(cases) != RAW_ROW_COUNTS["raw_case_metrics.csv"] or len(sequences) != RAW_ROW_COUNTS["raw_sequence_metrics.csv"]:
            raise ValueError("formal raw test universe row count mismatch")
        _validate_formal_raw_universe(cases)
    case_fields = set(_v2_header("raw_case_metrics"))
    sequence_fields = set(_v2_header("raw_sequence_metrics"))
    aggregates: dict[tuple[str, str, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    coordinate_digests: dict[tuple[int, int, int, int], tuple[str, str]] = {}
    selected: list[Mapping[str, Any]] = []
    ratio1: list[Mapping[str, Any]] = []
    ordinary_invalid: dict[str, list[float]] = defaultdict(list)
    method = str(expected_derived["selected_method"])
    selected_k = int(expected_derived["selected_k"])
    for row in cases:
        if set(row) != case_fields:
            raise ValueError("raw case schema mismatch")
        for name, kind in RAW_TABLE_SCHEMAS["raw_case_metrics"]:
            _validate_v2_type(row[name], kind, name)
        _v2_validate_case_id("raw_case_metrics", row)
        _v2_validate_row_provenance(row, expected_provenance)
        for name in ("sequence_digest","generator_config_digest","topology_digest","normalizer_digest","observation_digest","ambiguity_digest","support_digest"):
            _v2_assert_digest(row[name], name)
        coordinate = (row["record_index"],row["checkpoint"],row["mode_index"],row["stage_index"])
        pair = (row["observation_digest"],row["ambiguity_digest"])
        if coordinate in coordinate_digests and coordinate_digests[coordinate] != pair:
            raise ValueError("same-coordinate observation/ambiguity digest mismatch")
        coordinate_digests[coordinate] = pair
        if row["method"] not in ALL_METHODS:
            raise ValueError("raw test method domain violation")
        if not all(
            math.isfinite(float(row[name])) and float(row[name]) >= 0.0
            for name in (
                "nearest_rms_distance","nearest_matrix_l1_distance","covering_radius",
                "mean_pairwise_diversity","duplicate_fraction","component_coverage",
                "joint_coverage","physical_normalized_mean_width","total_tail_events",
                "total_tail_hits","group_tail_events","group_tail_hits","hotspot_events",
                "hotspot_hits","invalid_or_empty","construction_seconds","selector_seconds",
            )
        ):
            raise ValueError("raw test numeric/timing domain violation")
        if row["reveal_ratio"] < 1.0:
            identity = (row["sequence_id"], row["method"], row["requested_k"])
            bucket = aggregates[identity]
            bucket["raw_case_count"] += 1.0
            for name in (
                "nearest_rms_distance","total_tail_events","total_tail_hits",
                "group_tail_events","group_tail_hits","hotspot_events","hotspot_hits",
            ):
                bucket[name] += float(row[name])
            if row["method"] in ORDINARY_METHODS:
                ordinary_invalid[row["method"]].append(float(row["invalid_or_empty"]))
            if row["method"] == method and row["requested_k"] == selected_k:
                selected.append(row)
        elif row["method"] == method and row["requested_k"] == selected_k:
            ratio1.append(row)
    sequence_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in sequences:
        if set(row) != sequence_fields:
            raise ValueError("raw sequence schema mismatch")
        for name, kind in RAW_TABLE_SCHEMAS["raw_sequence_metrics"]:
            _validate_v2_type(row[name], kind, name)
        _v2_validate_row_provenance(row, expected_provenance)
        identity = (row["sequence_id"],row["method"],row["requested_k"])
        if identity in sequence_map or identity not in aggregates:
            raise ValueError("raw sequence aggregate identity mismatch")
        sequence_map[identity] = row
        bucket = aggregates[identity]
        count = int(bucket["raw_case_count"])
        expected_values = {
            "raw_case_count":count,
            "nearest_rms_distance":bucket["nearest_rms_distance"] / count,
            "total_tail_events":bucket["total_tail_events"],
            "total_tail_hits":bucket["total_tail_hits"],
            "total_tail_recall":None if bucket["total_tail_events"] == 0 else bucket["total_tail_hits"] / bucket["total_tail_events"],
            "group_tail_events":bucket["group_tail_events"],
            "group_tail_hits":bucket["group_tail_hits"],
            "group_tail_recall":None if bucket["group_tail_events"] == 0 else bucket["group_tail_hits"] / bucket["group_tail_events"],
            "hotspot_events":bucket["hotspot_events"],
            "hotspot_hits":bucket["hotspot_hits"],
            "hotspot_recall":None if bucket["hotspot_events"] == 0 else bucket["hotspot_hits"] / bucket["hotspot_events"],
        }
        for name, expected in expected_values.items():
            _v2_close(row[name], expected, f"sequence aggregate {name}")
    if exact and set(sequence_map) != set(aggregates):
        raise ValueError("raw sequence aggregate universe incomplete")
    paired = {
        sequence_id:float(sequence_map[(sequence_id,"random_empirical",selected_k)]["nearest_rms_distance"])
        - float(sequence_map[(sequence_id,method,selected_k)]["nearest_rms_distance"])
        for sequence_id in sorted({row["sequence_id"] for row in selected})
    }
    actual = {
        "selected_method":method,
        "selected_k":selected_k,
        "selected_joint_coverage":_v2_mean(selected,"joint_coverage"),
        "selected_component_coverage":_v2_mean(selected,"component_coverage"),
        "selected_mean_physical_normalized_width":_v2_mean(selected,"physical_normalized_mean_width"),
        "selected_total_tail_events":sum(float(row["total_tail_events"]) for row in selected),
        "selected_total_tail_hits":sum(float(row["total_tail_hits"]) for row in selected),
        "selected_group_tail_events":sum(float(row["group_tail_events"]) for row in selected),
        "selected_group_tail_hits":sum(float(row["group_tail_hits"]) for row in selected),
        "selected_hotspot_events":sum(float(row["hotspot_events"]) for row in selected),
        "selected_hotspot_hits":sum(float(row["hotspot_hits"]) for row in selected),
        "ordinary_invalid_or_empty_rate":{
            ordinary:float(np.mean(ordinary_invalid[ordinary])) for ordinary in ORDINARY_METHODS
        },
        "ratio1_singleton_coverage":_v2_mean(ratio1,"joint_coverage"),
        "ratio1_actual_k":next(iter({int(row["actual_k"]) for row in ratio1})),
        "all_timings_finite_nonnegative":all(
            math.isfinite(float(row["construction_seconds"]))
            and math.isfinite(float(row["selector_seconds"]))
            and float(row["construction_seconds"]) >= 0.0
            and float(row["selector_seconds"]) >= 0.0 for row in cases
        ),
        "paired_sequence_delta":paired,
    }
    _v2_close(actual, expected_derived, "formal test raw recomputation")
    return actual


def _v2_two_level_higher(rows: Sequence[Mapping[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence_id"])].append(float(row["score"]))
    if not grouped:
        raise ValueError("calibration evidence is empty")
    sequence_values = [
        float(np.quantile(values, 0.9, method="higher"))
        for _, values in sorted(grouped.items())
    ]
    return float(np.quantile(sequence_values, 0.9, method="higher"))


def _v2_validation_selection(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], str]:
    grouped: dict[tuple[str,str], list[float]] = defaultdict(list)
    for row in rows:
        if row["requested_k"] != 8 or row["reveal_ratio"] >= 1.0 or row["method"] not in ORDINARY_METHODS:
            raise ValueError("validation raw evidence must be ordinary K8 unknown-ratio")
        grouped[(row["sequence_id"],row["method"])].append(float(row["nearest_rms_distance"]))
    sequence_ids = sorted({key[0] for key in grouped})
    if not sequence_ids:
        raise ValueError("validation raw evidence is empty")
    means = {
        method:float(np.mean([
            float(np.mean(grouped[(sequence_id,method)])) for sequence_id in sequence_ids
            if (sequence_id,method) in grouped
        ]))
        for method in ORDINARY_METHODS
    }
    if any(sum((sequence_id, method) in grouped for sequence_id in sequence_ids) != len(sequence_ids) for method in ORDINARY_METHODS):
        raise ValueError("validation method/sequence evidence is incomplete")
    minimum = min(means.values())
    tied = {method for method, value in means.items() if abs(value-minimum) <= 1e-12}
    return means, next(method for method in VALIDATION_TIE_ORDER if method in tied)


def _v2_dependence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence_id"]].append(row)
    result: dict[str, Any] = {}
    defined: list[dict[str, Any]] = []
    for sequence_id in sorted(grouped):
        ordered = sorted(grouped[sequence_id], key=lambda row: row["checkpoint"])
        values = [float(row["total_traffic"]) for row in ordered]
        array = np.asarray(values, dtype=np.float64)
        centered = array - array.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 1e-15:
            evidence = {"lag1_acf":None,"positive_sequence_ess":None,"defined":False}
        else:
            correlations = [
                float(np.dot(centered[:-lag], centered[lag:]) / denominator)
                for lag in range(1, min(64, len(array)-1)+1)
            ]
            positive: list[float] = []
            for value in correlations:
                if value <= 0.0:
                    break
                positive.append(value)
            evidence = {
                "lag1_acf":correlations[0],
                "positive_sequence_ess":float(len(array)/(1.0+2.0*sum(positive))),
                "defined":True,
            }
            defined.append(evidence)
        result[sequence_id] = evidence
    result["aggregate"] = {
        "mean_lag1_acf":float(np.mean([item["lag1_acf"] for item in defined])) if defined else None,
        "mean_positive_sequence_ess":float(np.mean([item["positive_sequence_ess"] for item in defined])) if defined else None,
        "sum_positive_sequence_ess":float(np.sum([item["positive_sequence_ess"] for item in defined])) if defined else None,
    }
    return result


def _v2_derive_lofo(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    calibration = tables["raw_lofo_calibration_scores"]
    validation = tables["raw_lofo_validation_metrics"]
    test_rows = tables["raw_lofo_test_metrics"]
    fold_ids = sorted({row["fold_id"] for row in calibration} | {row["fold_id"] for row in validation} | {row["fold_id"] for row in test_rows})
    evidence: dict[str, Any] = {}
    deltas: list[float] = []
    for fold_id in fold_ids:
        fold_cal = [row for row in calibration if row["fold_id"] == fold_id]
        fold_val = [row for row in validation if row["fold_id"] == fold_id]
        fold_test = [row for row in test_rows if row["fold_id"] == fold_id]
        means, selected_method = _v2_validation_selection(fold_val)
        selected_rows = [row for row in fold_test if row["role"] == "selected"]
        random_rows = [row for row in fold_test if row["role"] == "random_comparator"]
        if any(row["method"] != selected_method for row in selected_rows):
            raise ValueError("LOFO selected role method must equal fold validation selector")
        if any(row["method"] != "random_empirical" for row in random_rows):
            raise ValueError("LOFO random role method must be random_empirical")
        selected = [float(row["nearest_rms_distance"]) for row in selected_rows]
        random_values = [float(row["nearest_rms_distance"]) for row in random_rows]
        if not selected or not random_values:
            raise ValueError("LOFO selected/random test evidence incomplete")
        selected_mean = float(np.mean(selected))
        random_mean = float(np.mean(random_values))
        delta = random_mean-selected_mean
        deltas.append(delta)
        evidence[fold_id] = {
            "held_out_family":fold_test[0]["held_out_family"],
            "calibration_radius":_v2_two_level_higher(fold_cal),
            "selected_method":selected_method,
            "family_delta":delta,
            "relative_degradation":0.0 if random_mean == 0.0 else (selected_mean-random_mean)/random_mean,
        }
    evidence["aggregate_delta"] = float(np.mean(deltas))
    return evidence


def validate_schema_v2_derived_evidence(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_provenance: Mapping[str, Any],
    expected_derived: Mapping[str, Any],
    allow_incomplete_universe: bool = False,
) -> Mapping[str, Any]:
    required = {
        "raw_calibration_scores","raw_validation_metrics",
        "raw_lofo_calibration_scores","raw_lofo_validation_metrics",
        "raw_lofo_test_metrics","raw_dependence_metrics",
    }
    if set(tables) != required:
        raise ValueError("derived evidence requires exact six raw tables")
    for name in required:
        validate_schema_v2_table(
            name, _v2_header(name), tables[name],
            expected_provenance=expected_provenance,
            allow_incomplete_universe=allow_incomplete_universe,
        )
    validation_means, selected = _v2_validation_selection(tables["raw_validation_metrics"])
    actual = {
        "calibration_radius":_v2_two_level_higher(tables["raw_calibration_scores"]),
        "validation_method_means":validation_means,
        "selected_method":selected,
        "selected_k":8,
        "lofo_fold_evidence":_v2_derive_lofo(tables),
        "test_total_traffic_dependence":_v2_dependence(tables["raw_dependence_metrics"]),
    }
    _v2_close(actual, expected_derived, "schema-v2 derived evidence")
    return actual


def _v2_exact_identity_set(
    rows: Sequence[Mapping[str,Any]], names: Sequence[str]
) -> set[tuple[Any,...]]:
    return {tuple(row[name] for name in names) for row in rows}


def _validate_v2_exact_universes(
    tables: Mapping[str,Sequence[Mapping[str,Any]]]
) -> None:
    specs = build_formal_sequence_specs()
    by_record = {spec.record_index:spec for spec in specs}
    calibration = [spec for spec in specs if spec.split == "calibration"]
    validation = [spec for spec in specs if spec.split == "validation"]
    test = [spec for spec in specs if spec.split == "test"]
    coordinates = [
        (checkpoint,mode_index,stage_index)
        for checkpoint in CHECKPOINTS
        for mode_index in range(len(REVEAL_MODES))
        for stage_index in range(len(UNKNOWN_RATIOS))
    ]
    expected_cal = {
        (spec.record_index,*coordinate) for spec in calibration for coordinate in coordinates
    }
    if _v2_exact_identity_set(
        tables["raw_calibration_scores"],
        ("record_index","checkpoint","mode_index","stage_index"),
    ) != expected_cal:
        raise ValueError("exact calibration identity universe mismatch")
    expected_val = {
        (spec.record_index,*coordinate,method)
        for spec in validation for coordinate in coordinates for method in ORDINARY_METHODS
    }
    if _v2_exact_identity_set(
        tables["raw_validation_metrics"],
        ("record_index","checkpoint","mode_index","stage_index","method"),
    ) != expected_val:
        raise ValueError("exact validation identity universe mismatch")
    expected_sequences = {
        (spec.record_index,method,requested_k)
        for spec in test for method in ALL_METHODS for requested_k in REQUESTED_K
    }
    if _v2_exact_identity_set(
        tables["raw_sequence_metrics"],
        ("record_index","method","requested_k"),
    ) != expected_sequences:
        raise ValueError("exact test sequence aggregate identity universe mismatch")
    expected_dependence = {
        (spec.record_index,checkpoint) for spec in test for checkpoint in CHECKPOINTS
    }
    if _v2_exact_identity_set(
        tables["raw_dependence_metrics"],("record_index","checkpoint")
    ) != expected_dependence:
        raise ValueError("exact dependence identity universe mismatch")
    expected_lofo_cal: set[tuple[Any,...]] = set()
    expected_lofo_val: set[tuple[Any,...]] = set()
    expected_lofo_test: set[tuple[Any,...]] = set()
    for held_index,held_family in enumerate(FORMAL_FAMILIES):
        fold_id = f"lofo-{held_index}-{held_family}"
        for spec in calibration:
            if spec.family != held_family:
                expected_lofo_cal.update(
                    (fold_id,spec.record_index,*coordinate)
                    for coordinate in coordinates
                )
        for spec in validation:
            if spec.family != held_family:
                expected_lofo_val.update(
                    (fold_id,spec.record_index,*coordinate,method)
                    for coordinate in coordinates for method in ORDINARY_METHODS
                )
        for spec in test:
            if spec.family == held_family:
                expected_lofo_test.update(
                    (fold_id,spec.record_index,*coordinate,role)
                    for coordinate in coordinates
                    for role in ("selected","random_comparator")
                )
    if _v2_exact_identity_set(
        tables["raw_lofo_calibration_scores"],
        ("fold_id","record_index","checkpoint","mode_index","stage_index"),
    ) != expected_lofo_cal:
        raise ValueError("exact LOFO calibration identity universe mismatch")
    if _v2_exact_identity_set(
        tables["raw_lofo_validation_metrics"],
        ("fold_id","record_index","checkpoint","mode_index","stage_index","method"),
    ) != expected_lofo_val:
        raise ValueError("exact LOFO validation identity universe mismatch")
    if _v2_exact_identity_set(
        tables["raw_lofo_test_metrics"],
        ("fold_id","record_index","checkpoint","mode_index","stage_index","role"),
    ) != expected_lofo_test:
        raise ValueError("exact LOFO test identity universe mismatch")
    expected_splits = {
        "raw_calibration_scores":"calibration",
        "raw_validation_metrics":"validation",
        "raw_case_metrics":"test","raw_sequence_metrics":"test",
        "raw_lofo_calibration_scores":"calibration",
        "raw_lofo_validation_metrics":"validation",
        "raw_lofo_test_metrics":"test","raw_dependence_metrics":"test",
    }
    for table_name,rows in tables.items():
        expected_split = expected_splits[table_name]
        for row in rows:
            spec = by_record.get(int(row["record_index"]))
            if spec is None or spec.split != expected_split or row["split"] != expected_split:
                raise ValueError(f"{table_name} split/record replacement is forbidden")
            if row["sequence_id"] != spec.sequence_id or row["family"] != spec.family or row["base_seed"] != spec.base_seed:
                raise ValueError(f"{table_name} sequence identity replacement is forbidden")


def validate_all_raw_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_provenance: Mapping[str, Any],
    expected_derived: Mapping[str, Any],
    allow_incomplete_universe: bool = False,
) -> bool:
    if set(tables) != set(RAW_TABLE_SCHEMAS):
        raise ValueError("exact eight raw tables are required")
    if not allow_incomplete_universe:
        _validate_v2_exact_universes(tables)
    for name, rows in tables.items():
        validate_schema_v2_table(
            name, _v2_header(name), rows,
            expected_provenance=expected_provenance,
            allow_incomplete_universe=allow_incomplete_universe,
        )
    validate_schema_v2_derived_evidence(
        {name:tables[name] for name in (
            "raw_calibration_scores","raw_validation_metrics",
            "raw_lofo_calibration_scores","raw_lofo_validation_metrics",
            "raw_lofo_test_metrics","raw_dependence_metrics",
        )},
        expected_provenance=expected_provenance,
        expected_derived=expected_derived,
        allow_incomplete_universe=allow_incomplete_universe,
    )
    return True


def _v2_sequence_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    specifications: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    for spec in build_formal_sequence_specs():
        specifications.append(_canonical(asdict(spec)))
        records[spec.sequence_id] = {
            "split":spec.split,
            "family":spec.family,
            "base_seed":spec.base_seed,
            "record_index":spec.record_index,
            "sequence_digest":hashlib.sha256(f"toy-sequence-{spec.record_index}".encode()).hexdigest(),
            "generator_config_digest":canonical_object_sha256(spec.generator_config),
        }
    return specifications, records


def _v2_make_row(
    table_name: str,
    spec: SequenceSpec,
    records: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint: int = 32,
    mode_index: int = 0,
    stage_index: int = 0,
    method: str = "minimax_subset",
    fold_id: str = "",
    held_out_family: str = "",
    role: str = "selected",
    normalizer_digest: str,
    topology_digest: str,
    case_index: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name, kind in RAW_TABLE_SCHEMAS[table_name]:
        row[name] = "" if kind == "s" else (False if kind == "b" else (0 if kind == "i" else (None if kind == "f?" else 0.0)))
    ratio = REVEAL_RATIOS[stage_index] if table_name == "raw_case_metrics" else UNKNOWN_RATIOS[stage_index]
    record = records[spec.sequence_id]
    values: dict[str, Any] = {
        "sequence_id":spec.sequence_id,"split":spec.split,"family":spec.family,
        "base_seed":spec.base_seed,"record_index":spec.record_index,
        "checkpoint":checkpoint,"mode_index":mode_index,"reveal_mode":REVEAL_MODES[mode_index],
        "stage_index":stage_index,"reveal_ratio":float(ratio),"actual_entry_fraction":float(ratio),
        "requested_k":8,"construction_seed":reveal_seed(spec.record_index,checkpoint,mode_index),
        "method":method,"replicate_count":8 if method == "random_empirical" and ratio < 1.0 else 1,
        "nearest_rms_distance":0.0,"nearest_matrix_l1_distance":1.0,
        "covering_radius":0.5,"mean_pairwise_diversity":0.2,"duplicate_fraction":0.0,
        "actual_k":1 if ratio == 1.0 else 8,"component_coverage":1.0,"joint_coverage":1.0,
        "physical_normalized_mean_width":0.5,"zero_physical_range_components":0,
        "total_tail_events":1.0,"total_tail_hits":1.0,"group_tail_events":1.0,"group_tail_hits":1.0,
        "hotspot_events":1.0,"hotspot_hits":1.0,"invalid_or_empty":0.0,
        "construction_seconds":0.001,"selector_seconds":0.001,
        "uses_oracle":False,"upper_bound_only":False,
        "sequence_digest":record["sequence_digest"],
        "generator_config_digest":record["generator_config_digest"],
        "topology_digest":topology_digest,"normalizer_digest":normalizer_digest,
        "observation_digest":hashlib.sha256(f"toy-observation-{spec.record_index}-{checkpoint}-{mode_index}-{stage_index}-{fold_id}".encode()).hexdigest(),
        "ambiguity_digest":hashlib.sha256(f"toy-ambiguity-{spec.record_index}-{checkpoint}-{mode_index}-{stage_index}-{fold_id}".encode()).hexdigest(),
        "support_digest":hashlib.sha256(f"toy-support-{table_name}-{spec.record_index}-{checkpoint}-{method}-{role}-{fold_id}".encode()).hexdigest(),
        "fold_id":fold_id,"held_out_family":held_out_family,"role":role,
        "score":0.0,"total_traffic":0.0,"case_index":case_index,
    }
    for name in row:
        if name in values:
            row[name] = values[name]
    if table_name != "raw_sequence_metrics":
        if table_name == "raw_case_metrics":
            row["case_id"] = f"case-{row['case_index']}-{row['method']}"
        else:
            row["case_id"] = table_name + ":" + ":".join(
                str(row[name]) for name in RAW_TABLE_IDENTITIES[table_name]
            )
    return row


def _v2_build_toy_tables() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    specs = build_formal_sequence_specs()
    specifications, records = _v2_sequence_records()
    topology_digest = hashlib.sha256(b"toy-topology-Rear4GPU").hexdigest()
    normalizer_digest = hashlib.sha256(b"toy-global-fit-normalizer").hexdigest()
    fold_normalizers = {
        f"lofo-{index}-{family}":hashlib.sha256(f"toy-lofo-normalizer-{index}".encode()).hexdigest()
        for index, family in enumerate(FORMAL_FAMILIES)
    }
    tables: dict[str, list[dict[str, Any]]] = {name:[] for name in RAW_TABLE_SCHEMAS}
    calibration_specs = [spec for spec in specs if spec.split == "calibration"]
    validation_specs = [spec for spec in specs if spec.split == "validation"]
    test_specs = [spec for spec in specs if spec.split == "test"]
    for index, spec in enumerate(calibration_specs):
        row = _v2_make_row(
            "raw_calibration_scores", spec, records,
            normalizer_digest=normalizer_digest, topology_digest=topology_digest,
        )
        row["score"] = 0.1 + index / 100.0
        tables["raw_calibration_scores"].append(row)
    method_distances = {
        "minimax_subset":0.30,"boundary_scenarios":0.30 + 5e-13,
        "worst_recent_cases":0.45,"random_empirical":0.50,
    }
    for spec in validation_specs:
        for method in ORDINARY_METHODS:
            row = _v2_make_row(
                "raw_validation_metrics", spec, records, method=method,
                normalizer_digest=normalizer_digest, topology_digest=topology_digest,
            )
            row["nearest_rms_distance"] = method_distances[method]
            row["case_id"] = "raw_validation_metrics:" + ":".join(
                str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_validation_metrics"]
            )
            tables["raw_validation_metrics"].append(row)
    registry = _case_index_lookup(specs)
    for spec in test_specs:
        unknown_index = registry[("test",spec.record_index,32,0,0,8)]
        for method in ORDINARY_METHODS:
            row = _v2_make_row(
                "raw_case_metrics", spec, records, method=method, case_index=unknown_index,
                normalizer_digest=normalizer_digest, topology_digest=topology_digest,
            )
            row["nearest_rms_distance"] = method_distances[method]
            tables["raw_case_metrics"].append(row)
            sequence = _v2_make_row(
                "raw_sequence_metrics", spec, records, method=method,
                normalizer_digest=normalizer_digest, topology_digest=topology_digest,
            )
            sequence.update({
                "raw_case_count":1,"nearest_rms_distance":method_distances[method],
                "total_tail_events":1.0,"total_tail_hits":1.0,"total_tail_recall":1.0,
                "group_tail_events":1.0,"group_tail_hits":1.0,"group_tail_recall":1.0,
                "hotspot_events":1.0,"hotspot_hits":1.0,"hotspot_recall":1.0,
            })
            tables["raw_sequence_metrics"].append(sequence)
        ratio1_index = registry[("test",spec.record_index,32,0,4,8)]
        ratio1 = _v2_make_row(
            "raw_case_metrics", spec, records, stage_index=4,
            method="minimax_subset", case_index=ratio1_index,
            normalizer_digest=normalizer_digest, topology_digest=topology_digest,
        )
        ratio1["nearest_rms_distance"] = 0.0
        tables["raw_case_metrics"].append(ratio1)
    for held_index, held_family in enumerate(FORMAL_FAMILIES):
        fold_id = f"lofo-{held_index}-{held_family}"
        fold_digest = fold_normalizers[fold_id]
        seen_cal = [spec for spec in calibration_specs if spec.family != held_family]
        seen_val = [spec for spec in validation_specs if spec.family != held_family]
        held_test = [spec for spec in test_specs if spec.family == held_family]
        for index, spec in enumerate(seen_cal):
            row = _v2_make_row(
                "raw_lofo_calibration_scores", spec, records,
                fold_id=fold_id, held_out_family=held_family,
                normalizer_digest=fold_digest, topology_digest=topology_digest,
            )
            row["score"] = 0.05 + index / 100.0
            tables["raw_lofo_calibration_scores"].append(row)
        for spec in seen_val:
            for method in ORDINARY_METHODS:
                row = _v2_make_row(
                    "raw_lofo_validation_metrics", spec, records, method=method,
                    fold_id=fold_id, held_out_family=held_family,
                    normalizer_digest=fold_digest, topology_digest=topology_digest,
                )
                row["nearest_rms_distance"] = method_distances[method]
                row["case_id"] = "raw_lofo_validation_metrics:" + ":".join(
                    str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_lofo_validation_metrics"]
                )
                tables["raw_lofo_validation_metrics"].append(row)
        for spec in held_test:
            for role, method, distance in (
                ("selected","minimax_subset",0.4 + held_index/100.0),
                ("random_comparator","random_empirical",0.6 + held_index/100.0),
            ):
                row = _v2_make_row(
                    "raw_lofo_test_metrics", spec, records, method=method,
                    fold_id=fold_id, held_out_family=held_family, role=role,
                    normalizer_digest=fold_digest, topology_digest=topology_digest,
                )
                row["nearest_rms_distance"] = distance
                row["case_id"] = "raw_lofo_test_metrics:" + ":".join(
                    str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_lofo_test_metrics"]
                )
                tables["raw_lofo_test_metrics"].append(row)
    for sequence_index, spec in enumerate(test_specs):
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            row = _v2_make_row(
                "raw_dependence_metrics", spec, records, checkpoint=checkpoint,
                normalizer_digest=normalizer_digest, topology_digest=topology_digest,
            )
            row["total_traffic"] = float(20 + sequence_index + 3*math.sin(checkpoint_index/2.0) + checkpoint_index/10.0)
            row["case_id"] = "raw_dependence_metrics:" + ":".join(
                str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_dependence_metrics"]
            )
            tables["raw_dependence_metrics"].append(row)
    provenance = {
        "topology_digest":topology_digest,"normalizer_digest":normalizer_digest,
        "lofo_fold_normalizer_digests":fold_normalizers,
        "sequence_records":records,"sequence_specs":specifications,
    }
    return tables, provenance


def _v2_gate_from_raw(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    lofo_evidence: Mapping[str, Any],
    *,
    selected_method: str,
    selected_k: int,
    integrity_complete: bool,
    integrity_passed: bool,
) -> dict[str, Any]:
    cases = tables["raw_case_metrics"]
    sequence_rows = tables["raw_sequence_metrics"]
    selected = [row for row in cases if row["method"] == selected_method and row["requested_k"] == selected_k and row["reveal_ratio"] < 1.0]
    ratio1 = [row for row in cases if row["method"] == selected_method and row["requested_k"] == selected_k and row["reveal_ratio"] == 1.0]
    selected_sequences = {(row["sequence_id"],row["method"],row["requested_k"]):row for row in sequence_rows}
    paired_rows: list[dict[str, Any]] = []
    record_map = {spec.sequence_id:spec for spec in build_formal_sequence_specs()}
    for sequence_id in sorted({row["sequence_id"] for row in selected}):
        selected_row = selected_sequences[(sequence_id,selected_method,selected_k)]
        random_row = selected_sequences[(sequence_id,"random_empirical",selected_k)]
        spec = record_map[sequence_id]
        paired_rows.append({
            "sequence_id":sequence_id,"family":spec.family,"base_seed":spec.base_seed,
            "paired_delta":float(random_row["nearest_rms_distance"])-float(selected_row["nearest_rms_distance"]),
        })
    bootstrap = family_stratified_sequence_bootstrap(
        paired_rows, replicates=10_000, seed=20260731
    )
    folds = [value for key, value in lofo_evidence.items() if key != "aggregate_delta"]
    lofo_delta = {row["held_out_family"]:float(row["family_delta"]) for row in folds}
    lofo_degradation = {row["held_out_family"]:float(row["relative_degradation"]) for row in folds}
    return {
        "selected_joint_coverage":_v2_mean(selected,"joint_coverage"),
        "selected_joint_coverage_by_family":{
            family:_v2_mean([row for row in selected if row["family"] == family],"joint_coverage")
            for family in FORMAL_FAMILIES
        },
        "paired_delta_ci95":[bootstrap.ci_lower,bootstrap.ci_upper],
        "paired_delta_by_base_seed":{
            str(seed):float(np.mean([row["paired_delta"] for row in paired_rows if row["base_seed"] == seed]))
            for seed in FORMAL_BASE_SEEDS
        },
        "paired_delta_by_family":{
            family:float(np.mean([row["paired_delta"] for row in paired_rows if row["family"] == family]))
            for family in FORMAL_FAMILIES
        },
        "lofo_aggregate_delta":float(lofo_evidence["aggregate_delta"]),
        "lofo_delta_by_family":lofo_delta,
        "lofo_relative_degradation_by_family":lofo_degradation,
        "total_tail_hits":sum(float(row["total_tail_hits"]) for row in selected),
        "total_tail_events":sum(float(row["total_tail_events"]) for row in selected),
        "group_tail_hits":sum(float(row["group_tail_hits"]) for row in selected),
        "group_tail_events":sum(float(row["group_tail_events"]) for row in selected),
        "hotspot_hits":sum(float(row["hotspot_hits"]) for row in selected),
        "hotspot_events":sum(float(row["hotspot_events"]) for row in selected),
        "selected_mean_physical_normalized_width":_v2_mean(selected,"physical_normalized_mean_width"),
        "ordinary_invalid_or_empty_rate":{
            method:_v2_mean([row for row in cases if row["method"] == method and row["reveal_ratio"] < 1.0],"invalid_or_empty")
            for method in ORDINARY_METHODS
        },
        "ratio1_singleton_coverage":_v2_mean(ratio1,"joint_coverage"),
        "all_timings_finite":all(
            math.isfinite(float(row["construction_seconds"])) and math.isfinite(float(row["selector_seconds"]))
            and float(row["construction_seconds"]) >= 0.0 and float(row["selector_seconds"]) >= 0.0
            for row in cases
        ),
        "integrity_checks_complete":integrity_complete,
        "integrity_checks_passed":integrity_passed,
    }


def _v2_summary(
    manifest: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    decision = evaluate_phase3b_gate(manifest["gate_evidence"])
    return {
        "schema_version":SCHEMA_VERSION,"protocol_sha256":PROTOCOL_SHA256,
        "selected_method":manifest["selected_method"],"selected_k":manifest["selected_k"],
        "calibration_radius":manifest["calibration_radius"],
        "validation_method_means":manifest["validation_method_means"],
        "gate_evidence":manifest["gate_evidence"],
        "test_total_traffic_dependence":manifest["test_total_traffic_dependence"],
        "raw_row_counts":{f"{name}.csv":len(rows) for name, rows in tables.items()},
        "data_status":manifest["data_status"],"gate_status":"PENDING_SUPERVISOR",
        "conditions_evaluated":list(decision.conditions),
        "failed_conditions":list(decision.failed_conditions),
        "insufficient_conditions":list(decision.insufficient_conditions),
        "combined_scientific_evidence_sha256":manifest["combined_scientific_evidence_sha256"],
    }


def _v2_write_csv(path: Path, table_name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    header = _v2_header(table_name)
    canonical_table_sha256(table_name, header, rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name:("" if row[name] is None else ("true" if row[name] is True else ("false" if row[name] is False else str(row[name]))))
                for name in header
            })


def _v2_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name:_file_sha256(root/name) for name in sorted(AUTHORIZED_SOURCE_KEYS)}


def _validate_v2_manifest_provenance(
    manifest: Mapping[str,Any], *, allow_incomplete_universe: bool
) -> None:
    frozen = {
        "families":list(FORMAL_FAMILIES),"base_seeds":list(FORMAL_BASE_SEEDS),
        "splits":list(FORMAL_SPLITS),"sequence_length":FORMAL_SEQUENCE_LENGTH,
        "max_entry":FORMAL_MAX_ENTRY,"history_window":HISTORY_WINDOW,
        "checkpoints":list(CHECKPOINTS),"reveal_modes":list(REVEAL_MODES),
        "reveal_ratios":list(REVEAL_RATIOS),"requested_k":list(REQUESTED_K),
        "random_replicates":RANDOM_REPLICATES,
    }
    for name,expected in frozen.items():
        if manifest[name] != expected:
            raise ValueError(f"manifest frozen {name} mismatch")
    specs = build_formal_sequence_specs()
    expected_specs = [_canonical(asdict(spec)) for spec in specs]
    if manifest["sequence_specs"] != expected_specs:
        raise ValueError("manifest sequence_specs mismatch")
    records = manifest["sequence_records"]
    if not isinstance(records,Mapping) or set(records) != {spec.sequence_id for spec in specs}:
        raise ValueError("manifest sequence_records exact universe mismatch")
    expected_record_keys = {
        "split","family","base_seed","record_index",
        "sequence_digest","generator_config_digest",
    }
    sequence_digests: set[str] = set()
    for spec in specs:
        record = records[spec.sequence_id]
        if not isinstance(record,Mapping) or set(record) != expected_record_keys:
            raise ValueError("manifest sequence record exact key set mismatch")
        static = {
            "split":spec.split,"family":spec.family,"base_seed":spec.base_seed,
            "record_index":spec.record_index,
            "generator_config_digest":canonical_object_sha256(spec.generator_config),
        }
        if any(record[name] != value for name,value in static.items()):
            raise ValueError("manifest sequence record static/config provenance mismatch")
        sequence_digests.add(_v2_assert_digest(record["sequence_digest"],"sequence digest"))
    if len(sequence_digests) != len(specs):
        raise ValueError("manifest sequence digests must be unique")
    if manifest["topology"].get("name") != "Rear4GPU":
        raise ValueError("manifest topology name mismatch")
    _v2_assert_digest(manifest["topology"].get("sha256"),"topology digest")
    _v2_assert_digest(manifest["h1_exclusion_manifest_sha256"],"H1 exclusion manifest")
    if not isinstance(manifest["h1_excluded_sequence_digests"],list):
        raise ValueError("H1 exclusion digest list is invalid")
    h1_digests = {
        _v2_assert_digest(value,"H1 excluded sequence")
        for value in manifest["h1_excluded_sequence_digests"]
    }
    if len(h1_digests) != len(manifest["h1_excluded_sequence_digests"]):
        raise ValueError("H1 exclusion digests must be unique")
    if h1_digests & sequence_digests:
        raise ValueError("H1 exclusion overlap with Phase 3B corpus")
    _v2_assert_digest(manifest["group_coefficients_digest"],"group coefficients")
    if allow_incomplete_universe:
        expected_toy = {
            "topology":hashlib.sha256(b"toy-topology-Rear4GPU").hexdigest(),
            "normalizer":hashlib.sha256(b"toy-global-fit-normalizer").hexdigest(),
            "h1":hashlib.sha256(b"toy-h1-exclusion-manifest").hexdigest(),
            "group":hashlib.sha256(b"toy-group-coefficients").hexdigest(),
        }
        if manifest["topology"]["sha256"] != expected_toy["topology"]:
            raise ValueError("toy topology provenance mismatch")
        if manifest["normalizer_digest"] != expected_toy["normalizer"]:
            raise ValueError("toy global normalizer provenance mismatch")
        if manifest["h1_exclusion_manifest_sha256"] != expected_toy["h1"] or h1_digests:
            raise ValueError("toy H1 exclusion provenance mismatch")
        if manifest["group_coefficients_digest"] != expected_toy["group"]:
            raise ValueError("toy group-coefficients provenance mismatch")
        expected_fold_digests = {
            f"lofo-{index}-{family}":hashlib.sha256(
                f"toy-lofo-normalizer-{index}".encode()
            ).hexdigest()
            for index,family in enumerate(FORMAL_FAMILIES)
        }
        if manifest["lofo_fold_normalizer_digests"] != expected_fold_digests:
            raise ValueError("toy LOFO normalizer provenance mismatch")
        for spec in specs:
            expected_sequence = hashlib.sha256(
                f"toy-sequence-{spec.record_index}".encode()
            ).hexdigest()
            if records[spec.sequence_id]["sequence_digest"] != expected_sequence:
                raise ValueError("toy sequence digest provenance mismatch")
        return
    project_root = Path(__file__).resolve().parents[2]
    topology_info,topology_digest = _load_rear4_topology(project_root)
    if manifest["topology"] != {"name":"Rear4GPU","sha256":topology_digest}:
        raise ValueError("formal Rear4 topology provenance mismatch")
    actual_h1,actual_h1_manifest = _load_h1_digest_exclusion(project_root)
    if manifest["h1_exclusion_manifest_sha256"] != actual_h1_manifest or h1_digests != actual_h1:
        raise ValueError("formal H1 exclusion provenance mismatch")
    topology = PublicTopologyView.from_topology_info(topology_info)
    if manifest["group_coefficients_digest"] != group_coefficients_digest(topology):
        raise ValueError("formal group-coefficients provenance mismatch")
    generated: dict[str,Any] = {}
    for spec in specs:
        sequence = generate_long_horizon_sequence(
            LongHorizonTrafficConfig(**spec.generator_config)
        )
        generated[spec.sequence_id] = sequence
        if records[spec.sequence_id]["sequence_digest"] != _sequence_sha256(sequence.matrices):
            raise ValueError("formal regenerated sequence digest mismatch")
    fit_matrices = tuple(
        matrix for spec in specs if spec.split == "fit"
        for matrix in generated[spec.sequence_id].matrices
    )
    global_normalizer = fit_descriptor_normalizer(fit_matrices,topology)
    if manifest["normalizer_digest"] != global_normalizer.digest:
        raise ValueError("formal global fit normalizer digest mismatch")
    expected_folds: dict[str,str] = {}
    for held_index,held_family in enumerate(FORMAL_FAMILIES):
        fold_id = f"lofo-{held_index}-{held_family}"
        fold = build_lofo_fold(specs,held_out_family=held_family)
        fold_fit = tuple(
            matrix for spec in fold.fit
            for matrix in generated[spec.sequence_id].matrices
        )
        expected_folds[fold_id] = fit_descriptor_normalizer(fold_fit,topology).digest
    if manifest["lofo_fold_normalizer_digests"] != expected_folds:
        raise ValueError("formal LOFO fold normalizer digest mismatch")


def materialize_provisional_toy_artifacts(staging_directory: str | Path) -> Mapping[str, Any]:
    staging = Path(staging_directory)
    if staging.exists():
        raise FileExistsError("staging destination already exists")
    staging.mkdir(parents=True)
    tables, provenance = _v2_build_toy_tables()
    derived_tables = {name:tables[name] for name in (
        "raw_calibration_scores","raw_validation_metrics","raw_lofo_calibration_scores",
        "raw_lofo_validation_metrics","raw_lofo_test_metrics","raw_dependence_metrics",
    )}
    validation_means, selected_method = _v2_validation_selection(tables["raw_validation_metrics"])
    lofo_evidence = _v2_derive_lofo(tables)
    dependence = _v2_dependence(tables["raw_dependence_metrics"])
    calibration_radius = _v2_two_level_higher(tables["raw_calibration_scores"])
    gate = _v2_gate_from_raw(
        tables, lofo_evidence, selected_method=selected_method, selected_k=8,
        integrity_complete=False, integrity_passed=False,
    )
    logical: dict[str,str] = {}
    scientific: dict[str,str] = {}
    for table_name, rows in tables.items():
        filename = f"{table_name}.csv"
        _v2_write_csv(staging/filename, table_name, rows)
        header = _v2_header(table_name)
        logical[filename] = canonical_table_sha256(table_name,header,rows)
        scientific[filename] = canonical_table_sha256(table_name,header,rows,scientific=True)
    combined = combined_scientific_evidence_sha256(scientific)
    manifest: dict[str,Any] = {
        "schema_version":SCHEMA_VERSION,"protocol_sha256":PROTOCOL_SHA256,
        "artifact_names":list(ARTIFACT_NAMES),"artifact_logical_sha256":{},
        "artifact_scientific_sha256":scientific,
        "combined_scientific_evidence_sha256":combined,
        "authorized_source_sha256":_v2_source_hashes(),
        "families":list(FORMAL_FAMILIES),"base_seeds":list(FORMAL_BASE_SEEDS),"splits":list(FORMAL_SPLITS),
        "sequence_length":FORMAL_SEQUENCE_LENGTH,"max_entry":FORMAL_MAX_ENTRY,"history_window":HISTORY_WINDOW,
        "checkpoints":list(CHECKPOINTS),"reveal_modes":list(REVEAL_MODES),"reveal_ratios":list(REVEAL_RATIOS),
        "requested_k":list(REQUESTED_K),"random_replicates":RANDOM_REPLICATES,
        "sequence_specs":provenance["sequence_specs"],"sequence_records":provenance["sequence_records"],
        "h1_exclusion_manifest_sha256":hashlib.sha256(b"toy-h1-exclusion-manifest").hexdigest(),
        "h1_excluded_sequence_digests":[],
        "topology":{"name":"Rear4GPU","sha256":provenance["topology_digest"]},
        "normalizer_digest":provenance["normalizer_digest"],
        "group_coefficients_digest":hashlib.sha256(b"toy-group-coefficients").hexdigest(),
        "lofo_fold_normalizer_digests":provenance["lofo_fold_normalizer_digests"],
        "calibration_radius":calibration_radius,"selected_method":selected_method,"selected_k":8,
        "validation_method_means":validation_means,"lofo_fold_evidence":lofo_evidence,
        "test_total_traffic_dependence":dependence,"gate_evidence":gate,
        "data_status":_PROVISIONAL_DATA_STATUS,
        "summary_sha256":"",
        "environment":{
            "python":platform.python_version(),"python_executable":sys.executable,
            "numpy":np.__version__,"platform":platform.platform(),
        },
    }
    manifest["artifact_logical_sha256"] = dict(logical)
    summary = _v2_summary(manifest,tables)
    summary_digest = canonical_object_sha256(summary)
    manifest["summary_sha256"] = summary_digest
    manifest["artifact_logical_sha256"]["summary.json"] = summary_digest
    _write_json(staging/"summary.json",summary)
    _write_json(staging/"manifest.json",manifest)
    return read_back_artifacts(staging,integrity_expected=False,allow_incomplete_universe=True)


def _v2_read_csv(path: Path, table_name: str) -> list[dict[str, Any]]:
    schema = RAW_TABLE_SCHEMAS[table_name]
    expected_header = _v2_header(table_name)
    with path.open("r",encoding="utf-8",newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_header:
            raise ValueError("CSV header/column order mismatch")
        raw_rows = list(reader)
    rows: list[dict[str,Any]] = []
    for raw in raw_rows:
        row: dict[str,Any] = {}
        for name, kind in schema:
            text = raw[name]
            if kind == "s":
                value: Any = text
            elif kind == "b":
                if text not in {"true","false"}:
                    raise ValueError("CSV bool lexical form must be lowercase true/false")
                value = text == "true"
            elif kind == "i":
                if re.fullmatch(r"0|-?[1-9][0-9]*",text) is None:
                    raise ValueError("CSV integer lexical form is not canonical")
                value = int(text)
            elif kind == "f?" and text == "":
                value = None
            else:
                try:
                    value = float(text)
                except ValueError as error:
                    raise ValueError("CSV float lexical form is invalid") from error
                if not math.isfinite(value):
                    raise ValueError("CSV float must be finite")
            row[name] = value
        rows.append(row)
    return rows


def _v2_expected_test_chain(
    tables: Mapping[str,Sequence[Mapping[str,Any]]],
    manifest: Mapping[str,Any],
) -> dict[str,Any]:
    method = manifest["selected_method"]
    selected_k = manifest["selected_k"]
    cases = tables["raw_case_metrics"]
    sequences = tables["raw_sequence_metrics"]
    selected = [row for row in cases if row["method"] == method and row["requested_k"] == selected_k and row["reveal_ratio"] < 1.0]
    ratio1 = [row for row in cases if row["method"] == method and row["requested_k"] == selected_k and row["reveal_ratio"] == 1.0]
    sequence_map = {(row["sequence_id"],row["method"],row["requested_k"]):row for row in sequences}
    paired = {
        sequence_id:float(sequence_map[(sequence_id,"random_empirical",selected_k)]["nearest_rms_distance"])
        - float(sequence_map[(sequence_id,method,selected_k)]["nearest_rms_distance"])
        for sequence_id in sorted({row["sequence_id"] for row in selected})
    }
    gate = manifest["gate_evidence"]
    return {
        "selected_method":method,"selected_k":selected_k,
        "selected_joint_coverage":gate["selected_joint_coverage"],
        "selected_component_coverage":_v2_mean(selected,"component_coverage"),
        "selected_mean_physical_normalized_width":gate["selected_mean_physical_normalized_width"],
        "selected_total_tail_events":gate["total_tail_events"],"selected_total_tail_hits":gate["total_tail_hits"],
        "selected_group_tail_events":gate["group_tail_events"],"selected_group_tail_hits":gate["group_tail_hits"],
        "selected_hotspot_events":gate["hotspot_events"],"selected_hotspot_hits":gate["hotspot_hits"],
        "ordinary_invalid_or_empty_rate":gate["ordinary_invalid_or_empty_rate"],
        "ratio1_singleton_coverage":gate["ratio1_singleton_coverage"],
        "ratio1_actual_k":next(iter({int(row["actual_k"]) for row in ratio1})),
        "all_timings_finite_nonnegative":gate["all_timings_finite"],
        "paired_sequence_delta":paired,
    }


def read_back_artifacts(
    artifact_directory: str | Path,
    *,
    integrity_expected: bool,
    allow_incomplete_universe: bool = False,
) -> dict[str,Any]:
    directory = Path(artifact_directory)
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != set(ARTIFACT_NAMES):
        raise ValueError("artifact directory must contain exactly ten files")
    try:
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    except (UnicodeDecodeError,json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if not isinstance(manifest,dict) or set(manifest) != SCHEMA_V2_MANIFEST_KEYS:
        raise ValueError("manifest exact schema/key set mismatch")
    if not isinstance(summary,dict) or set(summary) != SCHEMA_V2_SUMMARY_KEYS:
        raise ValueError("summary exact schema/key set mismatch")
    if manifest["schema_version"] != SCHEMA_VERSION or summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema version mismatch")
    if manifest["protocol_sha256"] != PROTOCOL_SHA256 or summary["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("protocol SHA-256 mismatch")
    if manifest["artifact_names"] != list(ARTIFACT_NAMES):
        raise ValueError("artifact name/order manifest mismatch")
    if set(manifest["authorized_source_sha256"]) != AUTHORIZED_SOURCE_KEYS or manifest["authorized_source_sha256"] != _v2_source_hashes():
        raise ValueError("authorized source SHA-256 manifest mismatch")
    if set(manifest["environment"]) != ENVIRONMENT_KEYS:
        raise ValueError("environment exact key set mismatch")
    _validate_v2_manifest_provenance(
        manifest,allow_incomplete_universe=allow_incomplete_universe
    )
    _v2_assert_digest(manifest["topology"]["sha256"],"topology")
    _v2_assert_digest(manifest["normalizer_digest"],"normalizer")
    tables = {
        name:_v2_read_csv(directory/f"{name}.csv",name) for name in RAW_TABLE_SCHEMAS
    }
    if not allow_incomplete_universe:
        actual_counts = {f"{name}.csv":len(rows) for name,rows in tables.items()}
        if actual_counts != RAW_ROW_COUNTS:
            raise ValueError("formal eight-table raw row-count universe mismatch")
    logical = {
        f"{name}.csv":canonical_table_sha256(name,_v2_header(name),rows)
        for name,rows in tables.items()
    }
    scientific = {
        f"{name}.csv":canonical_table_sha256(name,_v2_header(name),rows,scientific=True)
        for name,rows in tables.items()
    }
    summary_digest = canonical_object_sha256(summary)
    logical["summary.json"] = summary_digest
    if manifest["artifact_logical_sha256"] != logical:
        raise ValueError("artifact logical digest mismatch")
    if manifest["artifact_scientific_sha256"] != scientific:
        raise ValueError("artifact scientific digest mismatch")
    combined = combined_scientific_evidence_sha256(scientific)
    if manifest["combined_scientific_evidence_sha256"] != combined or summary["combined_scientific_evidence_sha256"] != combined:
        raise ValueError("combined scientific evidence digest mismatch")
    if manifest["summary_sha256"] != summary_digest:
        raise ValueError("summary canonical digest mismatch")
    provenance = {
        "topology_digest":manifest["topology"]["sha256"],
        "normalizer_digest":manifest["normalizer_digest"],
        "lofo_fold_normalizer_digests":manifest["lofo_fold_normalizer_digests"],
        "sequence_records":manifest["sequence_records"],
    }
    expected_derived = {
        "calibration_radius":manifest["calibration_radius"],
        "validation_method_means":manifest["validation_method_means"],
        "selected_method":manifest["selected_method"],"selected_k":manifest["selected_k"],
        "lofo_fold_evidence":manifest["lofo_fold_evidence"],
        "test_total_traffic_dependence":manifest["test_total_traffic_dependence"],
    }
    validate_all_raw_tables(
        tables, expected_provenance=provenance, expected_derived=expected_derived,
        allow_incomplete_universe=allow_incomplete_universe,
    )
    validate_test_raw_aggregates(
        tables["raw_case_metrics"],tables["raw_sequence_metrics"],
        expected_provenance=provenance,
        expected_derived=_v2_expected_test_chain(tables,manifest),
        exact=not allow_incomplete_universe,
    )
    recomputed_gate = _v2_gate_from_raw(
        tables,manifest["lofo_fold_evidence"],
        selected_method=manifest["selected_method"],selected_k=manifest["selected_k"],
        integrity_complete=integrity_expected,integrity_passed=integrity_expected,
    )
    _v2_close(manifest["gate_evidence"],recomputed_gate,"raw-derived gate evidence")
    flags = manifest["gate_evidence"]
    if flags["integrity_checks_complete"] is not integrity_expected or flags["integrity_checks_passed"] is not integrity_expected:
        raise ValueError("integrity provisional/final flags mismatch")
    if integrity_expected:
        expected_status = evaluate_phase3b_gate(recomputed_gate).data_status
    else:
        expected_status = _PROVISIONAL_DATA_STATUS
    if manifest["data_status"] != expected_status:
        raise ValueError("raw-derived data status mismatch")
    expected_summary = _v2_summary(manifest,tables)
    if summary != expected_summary:
        raise ValueError("raw-derived summary recomputation mismatch")
    return {"manifest":manifest,"summary":summary,"tables":tables}


def finalize_staged_artifacts(
    staging_directory: str | Path,
    *,
    allow_incomplete_universe: bool = False,
) -> dict[str,Any]:
    staging = Path(staging_directory)
    provisional = read_back_artifacts(
        staging,integrity_expected=False,
        allow_incomplete_universe=allow_incomplete_universe,
    )
    manifest = provisional["manifest"]
    tables = provisional["tables"]
    gate = _v2_gate_from_raw(
        tables,manifest["lofo_fold_evidence"],
        selected_method=manifest["selected_method"],selected_k=manifest["selected_k"],
        integrity_complete=True,integrity_passed=True,
    )
    manifest["gate_evidence"] = gate
    manifest["data_status"] = evaluate_phase3b_gate(gate).data_status
    summary = _v2_summary(manifest,tables)
    summary_digest = canonical_object_sha256(summary)
    manifest["summary_sha256"] = summary_digest
    manifest["artifact_logical_sha256"]["summary.json"] = summary_digest
    _write_json(staging/"summary.json",summary)
    _write_json(staging/"manifest.json",manifest)
    return read_back_artifacts(
        staging,integrity_expected=True,
        allow_incomplete_universe=allow_incomplete_universe,
    )


def publish_artifacts_atomically(
    staging_directory: str | Path,
    destination_directory: str | Path,
) -> Path:
    staging = Path(staging_directory)
    destination = Path(destination_directory)
    if staging.parent.resolve() != destination.parent.resolve():
        raise ValueError("staging and destination must have the same parent")
    if not staging.name.startswith(".phase3b-staging-"):
        raise ValueError("staging directory prefix is invalid")
    if destination.exists():
        raise FileExistsError("destination already exists; overwrite is forbidden")
    if not staging.is_dir() or {path.name for path in staging.iterdir()} != set(ARTIFACT_NAMES):
        raise ValueError("staging directory must contain exact ten artifacts")
    return staging.rename(destination)


def _v2_convert_calibration_rows(
    legacy_rows: Sequence[Mapping[str,Any]],
    specs_by_id: Mapping[str,SequenceSpec],
    records: Mapping[str,Mapping[str,Any]],
    *,
    table_name: str,
    topology_digest: str,
    normalizer_digest: str,
    fold_id: str = "",
    held_out_family: str = "",
) -> list[dict[str,Any]]:
    converted: list[dict[str,Any]] = []
    for source in legacy_rows:
        spec = specs_by_id[str(source["sequence_id"])]
        mode_index = REVEAL_MODES.index(str(source["reveal_mode"]))
        stage_index = UNKNOWN_RATIOS.index(float(source["reveal_ratio"]))
        row = _v2_make_row(
            table_name,spec,records,checkpoint=int(source["checkpoint"]),
            mode_index=mode_index,stage_index=stage_index,fold_id=fold_id,
            held_out_family=held_out_family,normalizer_digest=normalizer_digest,
            topology_digest=topology_digest,
        )
        row["actual_entry_fraction"] = float(source["actual_entry_fraction"])
        row["observation_digest"] = str(source["observation_digest"])
        row["ambiguity_digest"] = str(source["ambiguity_digest"])
        row["score"] = float(source["score"])
        row["case_id"] = table_name + ":" + ":".join(
            str(row[name]) for name in RAW_TABLE_IDENTITIES[table_name]
        )
        converted.append(row)
    return converted


def _v2_convert_validation_rows(
    legacy_rows: Sequence[Mapping[str,Any]],
    specs_by_id: Mapping[str,SequenceSpec],
    records: Mapping[str,Mapping[str,Any]],
    *,
    table_name: str,
    topology_digest: str,
    normalizer_digest: str,
    fold_id: str = "",
    held_out_family: str = "",
) -> list[dict[str,Any]]:
    converted: list[dict[str,Any]] = []
    for source in legacy_rows:
        spec = specs_by_id[str(source["sequence_id"])]
        method = str(source["method"])
        checkpoint = int(source["checkpoint"])
        mode_index = int(source["mode_index"])
        stage_index = int(source["stage_index"])
        row = _v2_make_row(
            table_name,spec,records,checkpoint=checkpoint,mode_index=mode_index,
            stage_index=stage_index,method=method,fold_id=fold_id,
            held_out_family=held_out_family,normalizer_digest=normalizer_digest,
            topology_digest=topology_digest,
        )
        row["nearest_rms_distance"] = float(source["nearest_rms_distance"])
        row["actual_entry_fraction"] = float(source["actual_entry_fraction"])
        row["replicate_count"] = int(source["replicate_count"])
        row["observation_digest"] = str(source["observation_digest"])
        row["ambiguity_digest"] = str(source["ambiguity_digest"])
        row["support_digest"] = str(source["support_digest"])
        row["case_id"] = table_name + ":" + ":".join(
            str(row[name]) for name in RAW_TABLE_IDENTITIES[table_name]
        )
        converted.append(row)
    return converted


def _v2_convert_test_rows(
    legacy_cases: Sequence[Mapping[str,Any]],
    legacy_sequences: Sequence[Mapping[str,Any]],
    specs_by_id: Mapping[str,SequenceSpec],
    records: Mapping[str,Mapping[str,Any]],
    *,
    topology_digest: str,
    normalizer_digest: str,
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    cases: list[dict[str,Any]] = []
    for source in legacy_cases:
        spec = specs_by_id[str(source["sequence_id"])]
        mode_index = REVEAL_MODES.index(str(source["reveal_mode"]))
        stage_index = REVEAL_RATIOS.index(float(source["reveal_ratio"]))
        method = str(source["method"])
        row = _v2_make_row(
            "raw_case_metrics",spec,records,checkpoint=int(source["checkpoint"]),
            mode_index=mode_index,stage_index=stage_index,method=method,
            normalizer_digest=normalizer_digest,topology_digest=topology_digest,
            case_index=int(source["case_index"]),
        )
        float_fields = (
            "nearest_rms_distance","nearest_matrix_l1_distance","covering_radius",
            "mean_pairwise_diversity","duplicate_fraction","component_coverage",
            "joint_coverage","physical_normalized_mean_width","total_tail_events",
            "total_tail_hits","group_tail_events","group_tail_hits","invalid_or_empty",
            "construction_seconds","selector_seconds",
        )
        for name in float_fields:
            row[name] = float(source[name])
        row["hotspot_events"] = float(source.get("hotspot_events",source.get("hotspot_event",0.0)))
        row["hotspot_hits"] = float(source.get("hotspot_hits",source.get("hotspot_hit",0.0)))
        row["zero_physical_range_components"] = int(source["zero_physical_range_components"])
        row["requested_k"] = int(source["requested_k"])
        row["actual_k"] = int(source["actual_k"])
        row["actual_entry_fraction"] = float(source["actual_entry_fraction"])
        row["replicate_count"] = RANDOM_REPLICATES if method == "random_empirical" and row["reveal_ratio"] < 1.0 else 1
        row["uses_oracle"] = bool(source["uses_oracle"])
        row["upper_bound_only"] = bool(source["upper_bound_only"])
        for name in ("observation_digest","ambiguity_digest","support_digest"):
            row[name] = str(source[name])
        cases.append(row)
    sequences: list[dict[str,Any]] = []
    for source in legacy_sequences:
        spec = specs_by_id[str(source["sequence_id"])]
        row = _v2_make_row(
            "raw_sequence_metrics",spec,records,method=str(source["method"]),
            normalizer_digest=normalizer_digest,topology_digest=topology_digest,
        )
        for name in (
            "raw_case_count","nearest_rms_distance","total_tail_events","total_tail_hits",
            "total_tail_recall","group_tail_events","group_tail_hits","group_tail_recall",
            "hotspot_events","hotspot_hits","hotspot_recall",
        ):
            value = source[name]
            row[name] = int(value) if name == "raw_case_count" else (None if value is None else float(value))
        row["requested_k"] = int(source["requested_k"])
        sequences.append(row)
    return cases,sequences


def _v2_formal_dependence_rows(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str,Any],
    records: Mapping[str,Mapping[str,Any]],
    *,
    topology_digest: str,
    normalizer_digest: str,
) -> list[dict[str,Any]]:
    rows: list[dict[str,Any]] = []
    for spec in specs:
        for checkpoint in CHECKPOINTS:
            row = _v2_make_row(
                "raw_dependence_metrics",spec,records,checkpoint=checkpoint,
                normalizer_digest=normalizer_digest,topology_digest=topology_digest,
            )
            row["total_traffic"] = float(np.asarray(sequences[spec.sequence_id].matrices[checkpoint]).sum())
            row["case_id"] = "raw_dependence_metrics:" + ":".join(
                str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_dependence_metrics"]
            )
            rows.append(row)
    return rows


def _v2_formal_lofo_tables(
    specs: Sequence[SequenceSpec],
    sequences: Mapping[str,Any],
    records: Mapping[str,Mapping[str,Any]],
    *,
    topology: PublicTopologyView,
    topology_digest: str,
    case_indices: Mapping[tuple[str,int,int,int,int,int],int],
) -> tuple[dict[str,list[dict[str,Any]]],dict[str,str]]:
    specs_by_id = {spec.sequence_id:spec for spec in specs}
    tables = {
        "raw_lofo_calibration_scores":[],"raw_lofo_validation_metrics":[],
        "raw_lofo_test_metrics":[],
    }
    normalizers: dict[str,str] = {}
    for held_index, held_family in enumerate(FORMAL_FAMILIES):
        fold_id = f"lofo-{held_index}-{held_family}"
        fold = build_lofo_fold(specs,held_out_family=held_family)
        fit_matrices = tuple(matrix for spec in fold.fit for matrix in sequences[spec.sequence_id].matrices)
        normalizer = fit_descriptor_normalizer(fit_matrices,topology)
        normalizers[fold_id] = normalizer.digest
        legacy_cal = _formal_calibration_rows(fold.calibration,sequences,topology=topology,normalizer=normalizer)
        tables["raw_lofo_calibration_scores"].extend(_v2_convert_calibration_rows(
            legacy_cal,specs_by_id,records,table_name="raw_lofo_calibration_scores",
            topology_digest=topology_digest,normalizer_digest=normalizer.digest,
            fold_id=fold_id,held_out_family=held_family,
        ))
        radius = calibrate_envelope_radius(legacy_cal,held_out_family=held_family)
        legacy_val = _formal_validation_rows(
            fold.validation,sequences,topology=topology,normalizer=normalizer,
            radius=radius,case_indices=case_indices,
        )
        tables["raw_lofo_validation_metrics"].extend(_v2_convert_validation_rows(
            legacy_val,specs_by_id,records,table_name="raw_lofo_validation_metrics",
            topology_digest=topology_digest,normalizer_digest=normalizer.digest,
            fold_id=fold_id,held_out_family=held_family,
        ))
        selected_method = _choose_validation_for_count(legacy_val,12)
        for spec in fold.test:
            sequence = sequences[spec.sequence_id]
            for checkpoint in CHECKPOINTS:
                matrix = np.asarray(sequence.matrices[checkpoint])
                for mode_index in range(len(REVEAL_MODES)):
                    for stage_index,ratio in enumerate(UNKNOWN_RATIOS):
                        view,actual_fraction = _formal_view(
                            sequence,spec,checkpoint=checkpoint,topology=topology,
                            mode_index=mode_index,ratio=ratio,normalizer=normalizer,
                        )
                        ambiguity = build_empirical_ambiguity_set(view,calibration_radius=radius)
                        case_index = case_indices[("test",spec.record_index,checkpoint,mode_index,stage_index,8)]
                        random_supports = [
                            select_support(
                                ambiguity,method="random_empirical",k=8,
                                replicate_seed=replicate_seed(case_index,replicate),
                            ) for replicate in range(RANDOM_REPLICATES)
                        ]
                        random_distance = float(np.mean([
                            truth_nearest_descriptor_distance(
                                ambiguity,support,matrix,
                            ) for support in random_supports
                        ]))
                        random_digest = hashlib.sha256(b"".join(
                            support.to_canonical_bytes() for support in random_supports
                        )).hexdigest()
                        if selected_method == "random_empirical":
                            selected_distance = random_distance
                            selected_digest = random_digest
                            selected_replicates = RANDOM_REPLICATES
                        else:
                            selected_support = select_support(
                                ambiguity,method=selected_method,k=8
                            )
                            selected_distance = truth_nearest_descriptor_distance(
                                ambiguity,selected_support,matrix,
                            )
                            selected_digest = hashlib.sha256(
                                selected_support.to_canonical_bytes()
                            ).hexdigest()
                            selected_replicates = 1
                        observation_digest = hashlib.sha256(
                            _canonical_bytes(view)
                        ).hexdigest()
                        ambiguity_digest = hashlib.sha256(
                            ambiguity.to_canonical_bytes()
                        ).hexdigest()
                        for role,method,distance,support_digest,replicate_count in (
                            ("selected",selected_method,selected_distance,selected_digest,selected_replicates),
                            ("random_comparator","random_empirical",random_distance,random_digest,RANDOM_REPLICATES),
                        ):
                            row = _v2_make_row(
                                "raw_lofo_test_metrics",spec,records,checkpoint=checkpoint,
                                mode_index=mode_index,stage_index=stage_index,method=method,
                                fold_id=fold_id,held_out_family=held_family,role=role,
                                normalizer_digest=normalizer.digest,topology_digest=topology_digest,
                            )
                            row["nearest_rms_distance"] = float(distance)
                            row["actual_entry_fraction"] = float(actual_fraction)
                            row["replicate_count"] = int(replicate_count)
                            row["observation_digest"] = observation_digest
                            row["ambiguity_digest"] = ambiguity_digest
                            row["support_digest"] = support_digest
                            row["case_id"] = "raw_lofo_test_metrics:" + ":".join(
                                str(row[name]) for name in RAW_TABLE_IDENTITIES["raw_lofo_test_metrics"]
                            )
                            tables["raw_lofo_test_metrics"].append(row)
    return tables,normalizers


def _v2_stage_bundle(
    staging: Path,
    tables: Mapping[str,Sequence[Mapping[str,Any]]],
    *,
    provenance: Mapping[str,Any],
    h1_manifest_digest: str,
    h1_digests: Sequence[str],
    topology_name: str,
    group_digest: str,
    allow_incomplete_universe: bool,
) -> dict[str,Any]:
    if staging.exists():
        raise FileExistsError("staging destination already exists")
    if not allow_incomplete_universe:
        counts = {f"{name}.csv":len(rows) for name,rows in tables.items()}
        if counts != RAW_ROW_COUNTS:
            raise ValueError("formal raw table row counts are not exact")
    staging.mkdir(parents=True)
    validation_means,selected_method = _v2_validation_selection(tables["raw_validation_metrics"])
    lofo_evidence = _v2_derive_lofo(tables)
    dependence = _v2_dependence(tables["raw_dependence_metrics"])
    calibration_radius = _v2_two_level_higher(tables["raw_calibration_scores"])
    gate = _v2_gate_from_raw(
        tables,lofo_evidence,selected_method=selected_method,selected_k=8,
        integrity_complete=False,integrity_passed=False,
    )
    logical: dict[str,str] = {}
    scientific: dict[str,str] = {}
    for name,rows in tables.items():
        _v2_write_csv(staging/f"{name}.csv",name,rows)
        logical[f"{name}.csv"] = canonical_table_sha256(name,_v2_header(name),rows)
        scientific[f"{name}.csv"] = canonical_table_sha256(name,_v2_header(name),rows,scientific=True)
    combined = combined_scientific_evidence_sha256(scientific)
    manifest: dict[str,Any] = {
        "schema_version":2,"protocol_sha256":PROTOCOL_SHA256,"artifact_names":list(ARTIFACT_NAMES),
        "artifact_logical_sha256":logical,"artifact_scientific_sha256":scientific,
        "combined_scientific_evidence_sha256":combined,"authorized_source_sha256":_v2_source_hashes(),
        "families":list(FORMAL_FAMILIES),"base_seeds":list(FORMAL_BASE_SEEDS),"splits":list(FORMAL_SPLITS),
        "sequence_length":FORMAL_SEQUENCE_LENGTH,"max_entry":FORMAL_MAX_ENTRY,"history_window":HISTORY_WINDOW,
        "checkpoints":list(CHECKPOINTS),"reveal_modes":list(REVEAL_MODES),"reveal_ratios":list(REVEAL_RATIOS),
        "requested_k":list(REQUESTED_K),"random_replicates":RANDOM_REPLICATES,
        "sequence_specs":provenance["sequence_specs"],"sequence_records":provenance["sequence_records"],
        "h1_exclusion_manifest_sha256":h1_manifest_digest,"h1_excluded_sequence_digests":list(h1_digests),
        "topology":{"name":topology_name,"sha256":provenance["topology_digest"]},
        "normalizer_digest":provenance["normalizer_digest"],"group_coefficients_digest":group_digest,
        "lofo_fold_normalizer_digests":provenance["lofo_fold_normalizer_digests"],
        "calibration_radius":calibration_radius,"selected_method":selected_method,"selected_k":8,
        "validation_method_means":validation_means,"lofo_fold_evidence":lofo_evidence,
        "test_total_traffic_dependence":dependence,"gate_evidence":gate,
        "data_status":_PROVISIONAL_DATA_STATUS,
        "summary_sha256":"","environment":{
            "python":platform.python_version(),"python_executable":sys.executable,
            "numpy":np.__version__,"platform":platform.platform(),
        },
    }
    summary = _v2_summary(manifest,tables)
    summary_digest = canonical_object_sha256(summary)
    manifest["summary_sha256"] = summary_digest
    manifest["artifact_logical_sha256"]["summary.json"] = summary_digest
    _write_json(staging/"summary.json",summary)
    _write_json(staging/"manifest.json",manifest)
    return read_back_artifacts(
        staging,integrity_expected=False,
        allow_incomplete_universe=allow_incomplete_universe,
    )


def run_formal_experiment(output_directory: str | Path) -> dict[str,Any]:
    """Generate schema-v2 formal evidence through the reviewed atomic pipeline."""
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError("formal destination already exists; overwrite is forbidden")
    project_root = Path(__file__).resolve().parents[2]
    protocol_path = project_root/"docs"/"uncertainty_aiccl"/"PHASE3B_AMBIGUITY_PROTOCOL.md"
    if _file_sha256(protocol_path).upper() != PROTOCOL_SHA256:
        raise ValueError("Phase 3B protocol SHA-256 mismatch")
    h1_digests,h1_manifest_digest = _load_h1_digest_exclusion(project_root)
    topology_info,topology_digest = _load_rear4_topology(project_root)
    topology = PublicTopologyView.from_topology_info(topology_info)
    specs = build_formal_sequence_specs()
    specs_by_id = {spec.sequence_id:spec for spec in specs}
    sequences: dict[str,Any] = {}
    records: dict[str,dict[str,Any]] = {}
    validation_records: list[dict[str,Any]] = []
    for spec in specs:
        sequence = generate_long_horizon_sequence(LongHorizonTrafficConfig(**spec.generator_config))
        if len(sequence.matrices) != FORMAL_SEQUENCE_LENGTH:
            raise ValueError("formal generator returned wrong sequence length")
        digest = _sequence_sha256(sequence.matrices)
        sequences[spec.sequence_id] = sequence
        records[spec.sequence_id] = {
            "split":spec.split,"family":spec.family,"base_seed":spec.base_seed,
            "record_index":spec.record_index,"sequence_digest":digest,
            "generator_config_digest":canonical_object_sha256(spec.generator_config),
        }
        validation_records.append({
            "sequence_id":spec.sequence_id,"split":spec.split,
            "sequence_digest":digest,"record_index":spec.record_index,
        })
    validate_sequence_records(validation_records,h1_sequence_digests=h1_digests)
    fit_specs = [spec for spec in specs if spec.split == "fit"]
    fit_matrices = tuple(matrix for spec in fit_specs for matrix in sequences[spec.sequence_id].matrices)
    normalizer = fit_descriptor_normalizer(fit_matrices,topology)
    fit_descriptors = np.stack([traffic_descriptor(matrix,topology) for matrix in fit_matrices])
    group_start = 2*topology.num_nodes+3
    group_indices = tuple(range(group_start,fit_descriptors.shape[1]))
    total_threshold = float(np.quantile(fit_descriptors[:,0],0.9,method="linear"))
    group_thresholds = np.quantile(fit_descriptors[:,group_indices],0.9,axis=0,method="linear")
    case_indices = _case_index_lookup(specs)
    calibration_specs = [spec for spec in specs if spec.split == "calibration"]
    legacy_cal = _formal_calibration_rows(calibration_specs,sequences,topology=topology,normalizer=normalizer)
    calibration_rows = _v2_convert_calibration_rows(
        legacy_cal,specs_by_id,records,table_name="raw_calibration_scores",
        topology_digest=topology_digest,normalizer_digest=normalizer.digest,
    )
    radius = calibrate_envelope_radius(legacy_cal)
    validation_specs = [spec for spec in specs if spec.split == "validation"]
    legacy_val = _formal_validation_rows(
        validation_specs,sequences,topology=topology,normalizer=normalizer,
        radius=radius,case_indices=case_indices,
    )
    validation_rows = _v2_convert_validation_rows(
        legacy_val,specs_by_id,records,table_name="raw_validation_metrics",
        topology_digest=topology_digest,normalizer_digest=normalizer.digest,
    )
    test_specs = [spec for spec in specs if spec.split == "test"]
    legacy_cases,legacy_sequences = _formal_test_rows(
        test_specs,sequences,topology=topology,normalizer=normalizer,radius=radius,
        total_threshold=total_threshold,group_indices=group_indices,
        group_thresholds=group_thresholds,
        sequence_digests={key:value["sequence_digest"] for key,value in records.items()},
        case_indices=case_indices,
    )
    case_rows,sequence_rows = _v2_convert_test_rows(
        legacy_cases,legacy_sequences,specs_by_id,records,
        topology_digest=topology_digest,normalizer_digest=normalizer.digest,
    )
    lofo_tables,fold_normalizers = _v2_formal_lofo_tables(
        specs,sequences,records,topology=topology,topology_digest=topology_digest,
        case_indices=case_indices,
    )
    tables: dict[str,Sequence[Mapping[str,Any]]] = {
        "raw_calibration_scores":calibration_rows,"raw_validation_metrics":validation_rows,
        "raw_case_metrics":case_rows,"raw_sequence_metrics":sequence_rows,
        **lofo_tables,
        "raw_dependence_metrics":_v2_formal_dependence_rows(
            test_specs,sequences,records,topology_digest=topology_digest,
            normalizer_digest=normalizer.digest,
        ),
    }
    provenance = {
        "sequence_specs":[_canonical(asdict(spec)) for spec in specs],
        "sequence_records":records,"topology_digest":topology_digest,
        "normalizer_digest":normalizer.digest,
        "lofo_fold_normalizer_digests":fold_normalizers,
    }
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = destination.parent/(
        f".phase3b-staging-{destination.name}-{uuid.uuid4().hex}"
    )
    _v2_stage_bundle(
        staging,tables,provenance=provenance,h1_manifest_digest=h1_manifest_digest,
        h1_digests=sorted(h1_digests),topology_name="Rear4GPU",
        group_digest=group_coefficients_digest(topology),allow_incomplete_universe=False,
    )
    finalize_staged_artifacts(staging,allow_incomplete_universe=False)
    publish_artifacts_atomically(staging,destination)
    return read_back_artifacts(
        destination,integrity_expected=True,allow_incomplete_universe=False,
    )


__all__ = [
    "ALL_METHODS",
    "ARTIFACT_NAMES",
    "AUTHORIZED_SOURCE_KEYS",
    "CHECKPOINTS",
    "FORMAL_BASE_SEEDS",
    "FORMAL_FAMILIES",
    "FORMAL_MAX_ENTRY",
    "FORMAL_SEQUENCE_LENGTH",
    "FORMAL_SPLITS",
    "HISTORY_WINDOW",
    "ORDINARY_METHODS",
    "RANDOM_REPLICATES",
    "RAW_ROW_COUNTS",
    "RAW_TABLE_IDENTITIES",
    "RAW_TABLE_SCHEMAS",
    "REQUESTED_K",
    "REVEAL_MODES",
    "REVEAL_RATIOS",
    "SAME_MOMENT_VARIANTS",
    "SCHEMA_VERSION",
    "SCHEMA_V2_MANIFEST_KEYS",
    "SCHEMA_V2_SUMMARY_KEYS",
    "aggregate_random_replicates",
    "aggregate_sequence_metrics",
    "build_case_registry",
    "build_formal_sequence_specs",
    "build_lofo_fold",
    "build_summary",
    "calibrate_envelope_radius",
    "calibration_exceedance_score",
    "canonical_object_sha256",
    "canonical_table_sha256",
    "combined_scientific_evidence_sha256",
    "compute_case_metrics",
    "evaluate_phase3b_gate",
    "family_stratified_sequence_bootstrap",
    "finalize_staged_artifacts",
    "materialize_provisional_toy_artifacts",
    "publish_artifacts_atomically",
    "read_back_artifacts",
    "recompute_artifacts",
    "replicate_seed",
    "reveal_seed",
    "run_formal_experiment",
    "run_toy_experiment",
    "select_validation_method",
    "validate_raw_case_rows",
    "validate_all_raw_tables",
    "validate_gate_raw_evidence",
    "validate_schema_v2_derived_evidence",
    "validate_schema_v2_table",
    "validate_sequence_records",
    "validate_test_raw_aggregates",
    "write_experiment_artifacts",
]
