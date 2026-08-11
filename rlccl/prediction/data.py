"""Leakage-safe data preparation for the Gate H1 prediction protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rlccl.models.traffic_predictor import summary_vector, traffic_summary
from rlccl.traffic.long_horizon_generator import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
)


FORMAL_FAMILIES = tuple(LONG_HORIZON_FAMILIES)
FORMAL_BASE_SEEDS = (42, 142, 242)
FORMAL_SEQUENCE_LENGTH = 1024
FORMAL_SPLITS = ("fit", "fit", "validation", "calibration", "test")
RECENT_STEPS = 8
MOMENT_WINDOW = 16


@dataclass(frozen=True)
class SequenceSpec:
    """Immutable identity and generator configuration for one full sequence."""

    sequence_id: str
    family: str
    family_index: int
    base_seed: int
    seed_index: int
    sequence_index: int
    actual_seed: int
    split: str
    dynamics_variant: str | None
    sequence_length: int
    generator_config: dict[str, Any]


@dataclass(frozen=True)
class HistoryExamples:
    """Array representation whose predictors can only consume ``X_<t``."""

    sequence_ids: np.ndarray
    families: np.ndarray
    seeds: np.ndarray
    steps: np.ndarray
    history_last_steps: np.ndarray
    recent_history: np.ndarray
    moment_features: np.ndarray
    previous_targets: np.ndarray
    ewma_targets: np.ndarray
    targets: np.ndarray
    hotspot_targets: np.ndarray
    previous_hotspots: np.ndarray


@dataclass
class Standardizers:
    """Fit-split statistics for recent-history inputs and continuous targets."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    def transform_inputs(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.input_mean) / self.input_scale

    def transform_targets(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.target_mean) / self.target_scale

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array * self.target_scale + self.target_mean

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "input_mean": self.input_mean.copy(),
            "input_scale": self.input_scale.copy(),
            "target_mean": self.target_mean.copy(),
            "target_scale": self.target_scale.copy(),
        }


def _generator_config(
    family: str,
    actual_seed: int,
    dynamics_variant: str | None,
) -> dict[str, Any]:
    return asdict(
        LongHorizonTrafficConfig(
            num_nodes=4,
            sequence_length=FORMAL_SEQUENCE_LENGTH,
            family=family,
            seed=actual_seed,
            mean_level=2.0,
            std_level=1.5,
            max_entry=8,
            calibration_candidates=1,
            topology_name="Rear4GPU",
            dynamics_variant=dynamics_variant,
        )
    )


def build_formal_sequence_specs() -> list[SequenceSpec]:
    """Return the frozen 75-sequence topology without generating traffic."""

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
                sequence_id = (
                    f"{family}-base{base_seed}-sequence{sequence_index}-seed{actual_seed}"
                )
                specs.append(
                    SequenceSpec(
                        sequence_id=sequence_id,
                        family=family,
                        family_index=family_index,
                        base_seed=base_seed,
                        seed_index=seed_index,
                        sequence_index=sequence_index,
                        actual_seed=actual_seed,
                        split=split,
                        dynamics_variant=variant,
                        sequence_length=FORMAL_SEQUENCE_LENGTH,
                        generator_config=_generator_config(family, actual_seed, variant),
                    )
                )
    return specs


def generate_formal_sequence(spec: SequenceSpec) -> Any:
    """Generate exactly one preregistered sequence and apply its canonical id."""

    sequence = generate_long_horizon_sequence(
        LongHorizonTrafficConfig(**dict(spec.generator_config))
    )
    sequence.sequence_id = spec.sequence_id
    sequence.metadata["formal_base_seed"] = spec.base_seed
    sequence.metadata["formal_sequence_index"] = spec.sequence_index
    sequence.metadata["formal_split"] = spec.split
    return sequence


def sequence_digest(matrices: Iterable[np.ndarray]) -> str:
    """Hash shapes, dtypes, and contents of a complete matrix sequence."""

    digest = hashlib.sha256()
    count = 0
    for matrix in matrices:
        array = np.ascontiguousarray(np.asarray(matrix))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
        count += 1
    digest.update(np.asarray([count], dtype=np.int64).tobytes())
    return digest.hexdigest()


def validate_split_records(records: Iterable[Mapping[str, Any]]) -> None:
    """Reject any sequence-id or complete-sequence digest overlap."""

    seen_ids: dict[str, str] = {}
    seen_digests: dict[str, str] = {}
    allowed = {"fit", "validation", "calibration", "test"}
    for record in records:
        sequence_id = str(record["sequence_id"])
        digest = str(record["digest"])
        split = str(record["split"])
        if split not in allowed:
            raise ValueError(f"Unknown split: {split}")
        if sequence_id in seen_ids:
            raise ValueError(
                f"sequence id overlap/duplicate: {sequence_id} in {seen_ids[sequence_id]} and {split}"
            )
        if digest in seen_digests:
            raise ValueError(
                f"digest overlap/duplicate: {digest} in {seen_digests[digest]} and {split}"
            )
        seen_ids[sequence_id] = split
        seen_digests[digest] = split


def _moment_feature(history: np.ndarray) -> np.ndarray:
    mean = history.mean(axis=0)
    variance = history.var(axis=0, ddof=0)
    off_diagonal = ~np.eye(mean.shape[0], dtype=bool)
    return np.concatenate((mean[off_diagonal], variance[off_diagonal]))


def _ewma_vectors(vectors: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    result = np.empty_like(vectors, dtype=np.float64)
    state = np.asarray(vectors[0], dtype=np.float64).copy()
    for index, vector in enumerate(vectors):
        if index:
            state = alpha * vector + (1.0 - alpha) * state
        result[index] = state
    return result


def build_history_examples(
    sequence: Any,
    *,
    group_coefficients: np.ndarray,
    recent_steps: int = RECENT_STEPS,
    moment_window: int = MOMENT_WINDOW,
) -> HistoryExamples:
    """Materialize examples at ``t>=8`` using only matrices through ``t-1``."""

    if recent_steps <= 0 or moment_window <= 0:
        raise ValueError("history windows must be positive")
    matrices = np.asarray(sequence.matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("sequence matrices must have shape [time, node, node]")
    groups = np.asarray(group_coefficients, dtype=np.float64)
    if groups.ndim != 3 or groups.shape[1:] != matrices.shape[1:]:
        raise ValueError("group_coefficients must have shape [group, node, node]")
    if len(matrices) <= recent_steps:
        raise ValueError("sequence is too short for the recent-history window")

    summaries = [traffic_summary(matrix, groups) for matrix in matrices]
    vectors = np.stack([summary_vector(item) for item in summaries]).astype(np.float64)
    hotspots = np.asarray(
        [item["hotspot_destination"] for item in summaries], dtype=np.int64
    )
    ewma = _ewma_vectors(vectors)
    steps = np.arange(recent_steps, len(matrices), dtype=np.int64)
    recent = np.stack([vectors[step - recent_steps : step] for step in steps])
    moments = np.stack(
        [
            _moment_feature(matrices[max(0, step - moment_window) : step])
            for step in steps
        ]
    )
    return HistoryExamples(
        sequence_ids=np.full(len(steps), str(sequence.sequence_id), dtype=object),
        families=np.full(len(steps), str(sequence.family), dtype=object),
        seeds=np.full(len(steps), int(sequence.seed), dtype=np.int64),
        steps=steps,
        history_last_steps=steps - 1,
        recent_history=recent,
        moment_features=moments,
        previous_targets=vectors[steps - 1].copy(),
        ewma_targets=ewma[steps - 1].copy(),
        targets=vectors[steps].copy(),
        hotspot_targets=hotspots[steps].copy(),
        previous_hotspots=hotspots[steps - 1].copy(),
    )


def fit_standardizers(inputs: np.ndarray, targets: np.ndarray) -> Standardizers:
    """Fit immutable scaling statistics from a caller-supplied fit split."""

    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim < 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("inputs and targets must be nonempty same-length arrays")
    reduction = tuple(range(x.ndim - 1))
    input_mean = x.mean(axis=reduction)
    input_scale = x.std(axis=reduction, ddof=0)
    input_scale = np.where(input_scale < 1e-8, 1.0, input_scale)
    target_mean = y.mean(axis=0)
    target_scale = y.std(axis=0, ddof=0)
    target_scale = np.where(target_scale < 1e-8, 1.0, target_scale)
    return Standardizers(input_mean, input_scale, target_mean, target_scale)


def build_lofo_fold(
    specs: Sequence[SequenceSpec], held_out_family: str
) -> dict[str, list[SequenceSpec]]:
    """Build a fold with held-out family absent from all model/calibration splits."""

    if held_out_family not in FORMAL_FAMILIES:
        raise ValueError(f"Unknown held-out family: {held_out_family}")
    result = {split: [] for split in ("fit", "validation", "calibration", "test")}
    for spec in specs:
        if spec.split == "test":
            if spec.family == held_out_family:
                result["test"].append(spec)
        elif spec.family != held_out_family:
            result[spec.split].append(spec)
    return result
