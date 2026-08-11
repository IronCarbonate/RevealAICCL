"""Deterministic Phase 4 experiment primitives and artifact validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import csv
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ARTIFACT_NAMES = (
    "manifest.json", "h1_best_point_model.json", "raw_validation_metrics.csv",
    "raw_test_episode_metrics.csv", "raw_test_sequence_metrics.csv",
    "raw_test_execution_events.csv", "raw_timing_metrics.csv", "summary.json",
)
COMMON_COLUMNS = (
    "schema_version", "split", "coordinate_id", "sequence_id", "family", "base_seed",
    "sequence_digest", "checkpoint", "checkpoint_index", "reveal_mode", "mode_index",
    "reveal_seed", "topology_digest", "config_digest", "method", "role", "uses_oracle",
    "executable",
)
VALIDATION_COLUMNS = COMMON_COLUMNS + (
    "horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max", "risk_lambda",
    "completion_slots", "total_online_ns", "end_to_end_latency_ms", "legality",
    "discrete_timeout", "wall_timeout", "row_digest",
)
EPISODE_COLUMNS = COMMON_COLUMNS + (
    "reference_kind", "horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max",
    "risk_lambda", "completion_slots", "lower_bound_slots", "oracle_regret_slots",
    "total_online_ns", "runner_wall_ns", "end_to_end_latency_ms", "end_to_end_regret_ms",
    "first_action_slot", "reveal_lead_lag_slots", "prefix_planned_batches",
    "prefix_planned_actions", "prefix_executed_batches", "prefix_executed_actions",
    "discarded_unexecuted_batches", "discarded_unexecuted_actions", "wasted_executed_actions",
    "wasted_unexecuted_actions", "reveal_replan_events", "exhaustion_replan_events",
    "invalidation_replan_events", "true_replan_events", "residual_repair_actions",
    "no_common_action_events", "fallback_events", "unreachable_od_count", "legality",
    "illegal_reason", "discrete_timeout", "wall_timeout", "row_digest",
)
EVENT_COLUMNS = COMMON_COLUMNS + (
    "event_index", "slot", "stage", "state_version_before", "state_version_after",
    "plan_revision", "event_kind", "reason", "observation_digest", "residual_state_digest",
    "support_digest", "requested_k", "actual_k", "batch_index", "batch_count",
    "action_count", "local_token_ordinal", "truth_binding_digest", "edge_index",
    "before_distance", "after_distance", "commit_legal", "elapsed_ns",
    "event_payload_digest", "row_digest",
)
SEQUENCE_COLUMNS = COMMON_COLUMNS + (
    "episode_count", "completion_mean", "completion_median", "completion_p95", "completion_p99",
    "completion_cvar95", "end_to_end_mean_ms", "end_to_end_median_ms", "end_to_end_p95_ms",
    "end_to_end_p99_ms", "end_to_end_cvar95_ms", "oracle_regret_mean_slots",
    "total_online_mean_ns", "total_online_p95_ns", "total_online_p99_ns", "legality_rate",
    "discrete_timeout_rate", "wall_timeout_rate", "prefix_executed_actions_sum",
    "discarded_actions_sum", "true_replan_sum", "residual_repair_actions_sum", "row_digest",
)
TIMING_COLUMNS = COMMON_COLUMNS + ("component", "elapsed_ns", "row_digest")
EXACT_COLUMNS = {
    "raw_validation_metrics.csv": VALIDATION_COLUMNS,
    "raw_test_episode_metrics.csv": EPISODE_COLUMNS,
    "raw_test_execution_events.csv": EVENT_COLUMNS,
}
EXACT_ROW_COUNTS = {"raw_validation_metrics.csv": 9600, "raw_test_episode_metrics.csv": 2700,
                    "raw_test_sequence_metrics.csv": 135, "raw_timing_metrics.csv": 21600}
PRIMARY_KEYS = {
    "raw_validation_metrics.csv": ("coordinate_id", "method", "horizon", "prefix", "risk_lambda"),
    "raw_test_episode_metrics.csv": ("coordinate_id", "method"),
    "raw_test_execution_events.csv": ("coordinate_id", "method", "event_index"),
}

BOOL_COLUMNS = {"uses_oracle", "executable", "legality", "discrete_timeout", "wall_timeout", "commit_legal"}
FLOAT_COLUMNS = {
    "risk_lambda", "end_to_end_latency_ms", "end_to_end_regret_ms",
    "completion_mean", "completion_median", "completion_p95", "completion_p99",
    "completion_cvar95", "end_to_end_mean_ms", "end_to_end_median_ms",
    "end_to_end_p95_ms", "end_to_end_p99_ms", "end_to_end_cvar95_ms",
    "oracle_regret_mean_slots", "total_online_mean_ns", "total_online_p95_ns",
    "total_online_p99_ns", "legality_rate", "discrete_timeout_rate", "wall_timeout_rate",
}
INT_COLUMNS = {
    "schema_version", "base_seed", "checkpoint", "checkpoint_index", "mode_index",
    "reveal_seed", "horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max",
    "completion_slots", "total_online_ns", "runner_wall_ns", "lower_bound_slots",
    "oracle_regret_slots", "first_action_slot", "reveal_lead_lag_slots",
    "prefix_planned_batches", "prefix_planned_actions", "prefix_executed_batches",
    "prefix_executed_actions", "discarded_unexecuted_batches", "discarded_unexecuted_actions",
    "wasted_executed_actions", "wasted_unexecuted_actions", "reveal_replan_events",
    "exhaustion_replan_events", "invalidation_replan_events", "true_replan_events",
    "residual_repair_actions", "no_common_action_events", "fallback_events",
    "unreachable_od_count", "event_index", "slot", "stage", "state_version_before",
    "state_version_after", "plan_revision", "actual_k", "batch_index", "batch_count",
    "action_count", "local_token_ordinal", "edge_index", "before_distance", "after_distance",
    "elapsed_ns", "episode_count", "prefix_executed_actions_sum", "discarded_actions_sum",
    "true_replan_sum", "residual_repair_actions_sum",
}
DEADLINE_KIND = "cooperative_not_preemptive"
FAMILIES = ("regime_switching_long", "stochastic_volatility", "rare_shock_recovery",
            "hotspot_random_walk", "same_moments_different_dynamics")
BASE_SEEDS = (642, 742, 842)
SPLITS = ("fit", "validation", "test")
CHECKPOINTS = (32, 96, 160, 224)
REVEAL_MODES = ("random_entries", "source_totals_first",
                "source_destination_totals_first", "partial_shards",
                "time_based_arrival")
REVEAL_RATIOS = (0.0, .25, .5, .75, 1.0)
FORMAL_DEADLINE_NS = 10_000_000_000
METHODS = ("full_information_lower_bound", "full_information_executable_reference",
           "wait_until_known", "partial_current_only", "long_term_mean_point_plan",
           "previous_value_point_plan", "h1_best_point_plan", "scenario_robust_prefix",
           "oracle_scenario_robust_reference")
LEGAL_HP = ((2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 4),
            (16, 1), (16, 2), (16, 4), (16, 8))
H1_MANIFEST_SHA256 = "C702D8CEA33BCEC805FA0AB4B1EEA58C7E0BCBF6AAEF697E01523BB86D65B48C"
PHASE3B_MANIFEST_SHA256 = "DF8218052A635A683CE0CA848BB31171C740A4FC9C8E31DDB764BB60F2DEE527"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    role: str
    uses_oracle: bool
    executable: bool
    reference_kind: str


METHOD_REGISTRY = {
    name: MethodSpec(
        "lower_bound" if name == METHODS[0] else "executable_reference" if name == METHODS[1]
        else "oracle_ceiling" if name == METHODS[-1] else "ordinary",
        name in (METHODS[0], METHODS[1], METHODS[-1]), name != METHODS[0],
        "truth_assisted_support_ceiling_not_proven_performance_bound" if name == METHODS[-1]
        else "provable_full_information_lower_bound" if name == METHODS[0]
        else "full_information_feasible_scheduler_not_optimal" if name == METHODS[1]
        else "ordinary_comparator" if name in METHODS[2:4]
        else "ordinary_point_comparator" if name in METHODS[4:7]
        else "ordinary_h2_candidate",
    ) for name in METHODS
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items()) if k != "row_digest"}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha(value: Any) -> str:
    def encode(item: Any) -> bytes:
        if item is None:
            return b"n;"
        if isinstance(item, (bool, np.bool_)):
            return b"b:1;" if bool(item) else b"b:0;"
        if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            return f"i:{int(item)};".encode("ascii")
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("canonical values must be finite")
            return f"f:{number.hex()};".encode("ascii")
        if isinstance(item, str):
            raw = item.encode("utf-8")
            return f"s:{len(raw)}:".encode("ascii") + raw + b";"
        if isinstance(item, np.ndarray):
            array = np.asarray(item)
            return encode({"dtype": array.dtype.str, "shape": list(array.shape),
                           "data": array.reshape(-1).tolist()})
        if isinstance(item, Mapping):
            pairs = sorted(((str(key), value) for key, value in item.items()), key=lambda pair: pair[0].encode("utf-8"))
            return f"m:{len(pairs)}:".encode("ascii") + b"".join(encode(key) + encode(value) for key, value in pairs)
        if isinstance(item, (tuple, list)):
            return f"l:{len(item)}:".encode("ascii") + b"".join(encode(value) for value in item)
        if hasattr(item, "__dataclass_fields__"):
            return encode({name: getattr(item, name) for name in item.__dataclass_fields__})
        raise TypeError(f"unsupported canonical type: {type(item).__name__}")
    return hashlib.sha256(encode(value)).hexdigest()


def row_digest(row: Mapping[str, Any]) -> str:
    return _sha([(key, value) for key, value in row.items() if key != "row_digest"])


def event_payload_digest(row: Mapping[str, Any]) -> str:
    fields = tuple(name for name in EVENT_COLUMNS[len(COMMON_COLUMNS):]
                   if name not in {"event_payload_digest", "row_digest"})
    return _sha([(name, row[name]) for name in fields if name in row])


def config_digest(*, method: str, role: str, horizon: int, prefix: int,
                  requested_k: int, risk_lambda: float, phase3b_recipe_digest: str,
                  h1_model_digest: str, slot_duration_ms: float,
                  cooperative_deadline_ns: int, checker_version: str) -> str:
    return _sha(locals())


def coordinate_id(*, sequence_digest: str, checkpoint: int, reveal_mode: str,
                  reveal_seed: int, topology_digest: str, ratios: Sequence[float],
                  cadence_slots: int, time_limit: int, checker_version: str) -> str:
    return _sha(locals())


def phase4_seeds(*, record_index: int, checkpoint_index: int, mode_index: int) -> tuple[int, int]:
    suffix = int(record_index) * 1000 + int(checkpoint_index) * 10 + int(mode_index)
    return 202608010 + suffix, 203608010 + suffix


def rotated_method_order(coordinate_id: str, methods: Sequence[str] = METHODS) -> tuple[str, ...]:
    values = tuple(methods)
    if not values:
        return ()
    shift = int(coordinate_id[:8], 16) % len(values)
    return values[shift:] + values[:shift]


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    record_index: int
    family: str
    base_seed: int
    split: str
    sequence_id: str
    actual_seed: int
    variant_index: int
    variant: str
    generator_config: Mapping[str, Any]
    sequence_length: int = 256


def build_formal_sequence_specs() -> tuple[SequenceSpec, ...]:
    rows = []
    for family in FAMILIES:
        for base_seed in BASE_SEEDS:
            for split in SPLITS:
                index = len(rows)
                family_index = FAMILIES.index(family)
                split_index = SPLITS.index(split)
                actual_seed = base_seed + family_index * 1_000_000 + split_index * 10_000
                variant_index = (BASE_SEEDS.index(base_seed) + split_index) % 4
                variants = ("smooth", "random_switching", "long_regime", "shock_recovery")
                generator_config = {
                    "sequence_length": 256, "num_nodes": 4, "family": family,
                    "seed": actual_seed, "mean_level": 2.0, "std_level": 1.5,
                    "max_entry": 8, "calibration_candidates": 1,
                    "topology_name": "Rear4GPU",
                    "dynamics_variant": (variants[variant_index]
                                         if family == "same_moments_different_dynamics" else None),
                }
                rows.append(SequenceSpec(index, family, base_seed, split,
                                         f"p4-{family}-base{base_seed}-{split}-seed{actual_seed}",
                                         actual_seed, variant_index, variants[variant_index], generator_config))
    return tuple(rows)


def toy_manifest() -> Mapping[str, Any]:
    excluded_h1 = (_sha("old-h1-sequence"),)
    excluded_phase3b = (_sha("old-phase3b-sequence"),)
    records = tuple({"sequence_id": spec.sequence_id, "sequence_digest": _sha(spec.sequence_id)}
                    for spec in build_formal_sequence_specs())
    truth = np.asarray([[0, 2, 1, 0], [1, 0, 1, 1], [0, 1, 0, 2], [1, 0, 1, 0]], dtype=np.int64)
    history = tuple(np.array(truth, copy=True) for _ in range(32))
    return {
        "old_manifests": {"h1": H1_MANIFEST_SHA256, "phase3b": PHASE3B_MANIFEST_SHA256},
        "h1_excluded_sequence_digests": excluded_h1,
        "phase3b_excluded_sequence_digests": excluded_phase3b,
        "excluded_sequence_digests": excluded_h1 + excluded_phase3b,
        "sequence_records": records,
        "toy_truth_matrix": truth,
        "toy_history_matrices": history,
    }


def validate_fresh_sequence_digests(digests: Sequence[str], excluded: Sequence[str]) -> None:
    values = tuple(digests)
    if len(values) != len(set(values)) or set(values) & set(excluded):
        raise ValueError("fresh sequence digest collision/overlap with excluded corpus")


def materialize_formal_sequence_records(*, excluded_sequence_digests: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    """Generate the frozen 45 sequences in memory and fail closed on any digest overlap."""
    from rlccl.traffic.long_horizon_generator import LongHorizonTrafficConfig, generate_long_horizon_sequence
    records = []
    for spec in build_formal_sequence_specs():
        sequence = generate_long_horizon_sequence(LongHorizonTrafficConfig(**dict(spec.generator_config)))
        sequence.sequence_id = spec.sequence_id
        digest = _sha(tuple(np.asarray(matrix, dtype=np.int64) for matrix in sequence.matrices))
        records.append({"record_index": spec.record_index, "sequence_id": spec.sequence_id,
                        "family": spec.family, "base_seed": spec.base_seed, "split": spec.split,
                        "actual_seed": spec.actual_seed, "variant_index": spec.variant_index,
                        "variant": spec.variant, "sequence_length": len(sequence.matrices),
                        "sequence_digest": digest, "sequence": sequence})
    validate_fresh_sequence_digests([record["sequence_digest"] for record in records],
                                    excluded_sequence_digests)
    if len(records) != 45 or any(record["sequence_length"] != 256 for record in records):
        raise ValueError("formal sequence universe is incomplete")
    return tuple(records)


@dataclass(frozen=True, slots=True)
class ToyEpisode:
    world: Any
    reveal_process: Any
    method: str


def build_episode(manifest: Mapping[str, Any], method: str) -> ToyEpisode:
    if method not in METHOD_REGISTRY:
        raise ValueError("unknown method")
    from rlccl.envs.problem import TopologyInfo
    from rlccl.uncertainty.problem import UncertainProblemInstance
    from rlccl.uncertainty.reveal import DemandRevealProcess
    nodes = 4
    edges = np.asarray([(source, destination) for source in range(nodes)
                        for destination in range(nodes) if source != destination], dtype=np.int64)
    topology = TopologyInfo(nodes, len(edges), edges, np.full(len(edges), 2.0), [], name="phase4-toy")
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=np.asarray(manifest["toy_truth_matrix"]), topology_info=topology,
        time_limit=80, sequence_id="p4-toy", sequence_step=32,
        family=FAMILIES[0], generator_metadata={"formal": False},
    )
    reveal = DemandRevealProcess(problem=world, mode="random_entries",
                                 ratios=(0.0, .25, .5, .75, 1.0), seed=202608010)
    return ToyEpisode(world, reveal, str(method))


def build_formal_episode(*, truth_matrix: np.ndarray, topology: Any, method: str,
                         sequence_id: str, sequence_step: int, family: str,
                         reveal_mode: str, reveal_seed: int) -> ToyEpisode:
    if method not in METHOD_REGISTRY or reveal_mode not in REVEAL_MODES:
        raise ValueError("unknown formal method or reveal mode")
    from rlccl.uncertainty.problem import UncertainProblemInstance
    from rlccl.uncertainty.reveal import DemandRevealProcess
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=np.asarray(truth_matrix, dtype=np.int64), topology_info=topology,
        time_limit=80, sequence_id=str(sequence_id), sequence_step=int(sequence_step),
        family=str(family), generator_metadata={"formal": True},
    )
    reveal = DemandRevealProcess(problem=world, mode=reveal_mode,
                                 ratios=REVEAL_RATIOS, seed=int(reveal_seed))
    return ToyEpisode(world, reveal, method)


@dataclass(frozen=True, slots=True)
class PublicEpisodeResult:
    scientific_rows: tuple[Mapping[str, Any], ...]
    final_world_digest: str
    rng_result_digest: str
    timing_metrics: Mapping[str, int] | None = None


def _fit_toy_h1_model(manifest: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    from rlccl.prediction.models import RecentHistoryMLP
    history = np.asarray(manifest["toy_history_matrices"], dtype=np.float64)
    vectors = history.reshape(len(history), -1)
    recent = np.stack([vectors[index - 8:index] for index in range(8, len(vectors))])
    targets = vectors[8:]
    input_mean = recent.mean(axis=(0, 1)); input_scale = recent.std(axis=(0, 1))
    input_scale = np.where(input_scale < 1e-8, 1.0, input_scale)
    target_mean = targets.mean(axis=0); target_scale = fit_target_scale(targets)
    normalized_x = (recent - input_mean) / input_scale
    normalized_y = (targets - target_mean) / target_scale
    if len(normalized_x) < 256:
        repeats = math.ceil(256 / len(normalized_x))
        normalized_x = np.tile(normalized_x, (repeats, 1, 1))[:256]
        normalized_y = np.tile(normalized_y, (repeats, 1))[:256]
    model = RecentHistoryMLP(8 * vectors.shape[1], vectors.shape[1], seed=20260731)
    wall_start = time.perf_counter_ns(); cpu_start = time.process_time_ns()
    model.fit(normalized_x, normalized_y)
    fit_wall_ns = time.perf_counter_ns() - wall_start
    fit_cpu_ns = time.process_time_ns() - cpu_start
    raw = model._model
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (*raw.coefs_, *raw.intercepts_))
    state = {
        "input_mean": input_mean, "input_scale": input_scale,
        "target_mean": target_mean, "target_scale": target_scale,
        "parameter_arrays": arrays, "model_state_sha256": _sha(arrays),
        "fit_wall_ns": fit_wall_ns, "fit_cpu_ns": fit_cpu_ns,
        "fit_example_count": len(normalized_x),
    }
    return model, state


def fit_h1_best_point_model(*, fit_sequences: Sequence[Any],
                            group_coefficients: np.ndarray) -> tuple[Any, Mapping[str, Any]]:
    """Fit the frozen H1 MLP on exactly the fresh fit universe (15 x 248 examples)."""
    from rlccl.prediction.data import build_history_examples, fit_standardizers
    from rlccl.prediction.models import RecentHistoryMLP
    import rlccl.prediction.models as model_source_module
    if len(fit_sequences) != 15:
        raise ValueError("H1 fit requires exactly 15 fresh fit sequences")
    examples = [build_history_examples(sequence, group_coefficients=group_coefficients,
                                       recent_steps=8) for sequence in fit_sequences]
    recent = np.concatenate([item.recent_history for item in examples], axis=0)
    targets = np.concatenate([item.targets for item in examples], axis=0)
    if recent.shape[0] != 3720 or targets.shape[0] != 3720:
        raise ValueError("H1 fit universe must contain exactly 3720 examples")
    standardizers = fit_standardizers(recent, targets)
    normalized_x = standardizers.transform_inputs(recent)
    normalized_y = standardizers.transform_targets(targets)
    model = RecentHistoryMLP(8 * targets.shape[1], targets.shape[1], seed=20260731)
    wall_start = time.perf_counter_ns(); cpu_start = time.process_time_ns()
    model.fit(normalized_x, normalized_y)
    wall_ns = time.perf_counter_ns() - wall_start; cpu_ns = time.process_time_ns() - cpu_start
    if not np.all(np.isfinite(model.predict(normalized_x[:1]))):
        raise ValueError("H1 model produced nonfinite inference")
    raw = model._model
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (*raw.coefs_, *raw.intercepts_))
    artifact = {
        "schema_version": 1, "model_name": "recent_history_mlp", "config": h1_fit_contract().config,
        "fit_sequence_records": tuple(str(sequence.sequence_id) for sequence in fit_sequences),
        "fit_example_count": 3720, "input_mean": standardizers.input_mean,
        "input_scale": standardizers.input_scale, "target_mean": standardizers.target_mean,
        "target_scale": standardizers.target_scale, "parameter_arrays": arrays,
        "model_state_sha256": _sha(arrays),
        "source_sha256": hashlib.sha256(Path(model_source_module.__file__).read_bytes()).hexdigest(),
        "group_coefficients_digest": _sha(np.asarray(group_coefficients)),
        "fit_wall_ns": wall_ns, "fit_cpu_ns": cpu_ns,
    }
    return model, artifact


def _build_ordinary_boundary_support(*, history_matrices: Sequence[np.ndarray],
                                     trusted_observation: Any, view: Any,
                                     construction_seed: int) -> tuple[Any, int, int]:
    from rlccl.uncertainty.ambiguity import (AmbiguityConstructionView,
                                             build_empirical_ambiguity_set,
                                             fit_descriptor_normalizer, select_support)
    from rlccl.scheduling.scenario_adapter import scenario_support_from_selected
    history = tuple(np.asarray(matrix, dtype=np.int64) for matrix in history_matrices[-32:])
    if len(history) != 32:
        raise ValueError("ordinary Phase3B support requires the exact 32-step history")
    construction_start = time.perf_counter_ns()
    normalizer = fit_descriptor_normalizer(history, trusted_observation.topology)
    ambiguity_view = AmbiguityConstructionView.from_observation(
        history_matrices=history, history_offsets=tuple(range(-32, 0)),
        observation=trusted_observation, construction_seed=int(construction_seed),
        normalizer=normalizer,
    )
    ambiguity = build_empirical_ambiguity_set(
        ambiguity_view, calibration_radius=0.34327919716983946,
    )
    construction_ns = time.perf_counter_ns() - construction_start
    selection_start = time.perf_counter_ns()
    selected = select_support(ambiguity, method="boundary_scenarios", k=8)
    support = scenario_support_from_selected(selected, observation=view)
    selection_ns = time.perf_counter_ns() - selection_start
    return support, construction_ns, selection_ns


def _reconcile_point_candidate(*, candidate: Any, history_matrices: Sequence[np.ndarray],
                               trusted_observation: Any, construction_seed: int) -> np.ndarray:
    from rlccl.uncertainty.ambiguity import (AmbiguityConstructionView,
                                             fit_descriptor_normalizer,
                                             reconcile_candidate)
    history = tuple(np.asarray(matrix, dtype=np.int64) for matrix in history_matrices[-32:])
    if len(history) != 32:
        raise ValueError("point support requires the exact 32-step history")
    normalizer = fit_descriptor_normalizer(history, trusted_observation.topology)
    ambiguity_view = AmbiguityConstructionView.from_observation(
        history_matrices=history, history_offsets=tuple(range(-32, 0)),
        observation=trusted_observation, construction_seed=int(construction_seed),
        normalizer=normalizer,
    )
    rounded = np.rint(np.asarray(candidate, dtype=np.float64)).astype(np.int64)
    rounded = np.clip(rounded, 0, 8)
    np.fill_diagonal(rounded, 0)
    return reconcile_candidate(rounded, ambiguity_view)


def _h1_candidate_from_actual_pool(*, history_matrices: Sequence[np.ndarray],
                                   trusted_observation: Any, model: Any,
                                   model_state: Mapping[str, Any],
                                   construction_seed: int) -> tuple[np.ndarray, int, int, int]:
    """Infer a summary, then select only from the current reconciled Phase3B pool."""
    from rlccl.models.traffic_predictor import summary_vector, traffic_summary
    from rlccl.uncertainty.ambiguity import (AmbiguityConstructionView,
                                             build_empirical_ambiguity_set,
                                             fit_descriptor_normalizer)
    history = tuple(np.asarray(matrix, dtype=np.int64) for matrix in history_matrices[-32:])
    groups = np.asarray(model_state["group_coefficients"], dtype=np.float64)
    construction_start = time.perf_counter_ns()
    normalizer = fit_descriptor_normalizer(history, trusted_observation.topology)
    ambiguity_view = AmbiguityConstructionView.from_observation(
        history_matrices=history, history_offsets=tuple(range(-32, 0)),
        observation=trusted_observation, construction_seed=int(construction_seed),
        normalizer=normalizer,
    )
    ambiguity = build_empirical_ambiguity_set(ambiguity_view,
                                               calibration_radius=0.34327919716983946)
    construction_ns = time.perf_counter_ns() - construction_start
    history_vectors = np.stack([summary_vector(traffic_summary(matrix, groups)) for matrix in history])
    normalized = ((history_vectors[-8:] - np.asarray(model_state["input_mean"])) /
                  np.asarray(model_state["input_scale"]))[None, ...]
    inference_start = time.perf_counter_ns()
    prediction = (model.predict(normalized)[0] * np.asarray(model_state["target_scale"]) +
                  np.asarray(model_state["target_mean"]))
    inference_ns = time.perf_counter_ns() - inference_start
    if not np.all(np.isfinite(prediction)):
        raise ValueError("H1 produced a nonfinite summary prediction")
    selection_start = time.perf_counter_ns()
    candidate_vectors = np.stack([
        summary_vector(traffic_summary(matrix, groups)) for matrix in ambiguity.support_matrices
    ])
    selected = select_h1_point_candidate(
        prediction=prediction, candidate_summaries=candidate_vectors,
        fit_target_scale=model_state["target_scale"],
        history_offsets=ambiguity.history_offsets,
    )
    matrix = np.asarray(ambiguity.support_matrices[selected.pool_index], dtype=np.int64)
    return matrix, construction_ns, time.perf_counter_ns() - selection_start, inference_ns


def _checker_rejection_reason(error: BaseException) -> str:
    if isinstance(error, TypeError):
        return "type_error"
    message = str(error).lower()
    ordered = (
        ("stale", "stale_observation"), ("unrevealed", "unrevealed_token"),
        ("duplicate", "duplicate_token"), ("edge_index", "edge_range"),
        ("source possession", "source_possession"),
        ("destination already", "destination_possession"),
        ("edge capacity", "edge_capacity"), ("shared group", "shared_group"),
    )
    return next((reason for fragment, reason in ordered if fragment in message),
                "other_rejected")


def run_public_episode(*, manifest: Mapping[str, Any], coordinate_id: str, method: str,
                       episode: ToyEpisode, episode_cache: dict[Any, Any],
                       planner_config: tuple[int, int, float] | None = None,
                       deadline_ns: int | None = None,
                       construction_seed: int = 203608010) -> PublicEpisodeResult:
    from rlccl.scheduling.robust_prefix import RobustPrefixConfig, RobustPrefixPlanner, build_scheduling_view, enumerate_candidates, pack_candidate_batch
    from rlccl.scheduling.scenario_adapter import scenario_support_from_matrices, oracle_support_from_matrices
    from rlccl.scheduling.recourse import (RecourseState, bind_action, bind_first_batch,
                                           record_committed_batch)
    from rlccl.uncertainty.execution import Proposal, commit_proposal
    spec = METHOD_REGISTRY[method]
    key = f"public:{method}"
    episode_cache[key] = _sha((coordinate_id, method))
    world, reveal = episode.world, episode.reveal_process
    events: list[Mapping[str, Any]] = []
    zero_digest = "0" * 64

    def emit(kind: str, reason: str = "NONE", **updates: Any) -> None:
        emit_updates = {
            "coordinate_id": coordinate_id, "method": method, "role": spec.role,
            "uses_oracle": spec.uses_oracle, "executable": spec.executable,
            **updates,
        }
        events.append(_event_row(len(events), kind, reason, **emit_updates))

    emit("episode_start", slot=-1, stage=-1, observation_digest=zero_digest,
         residual_state_digest=zero_digest, support_digest=zero_digest,
         batch_count=0, action_count=0)
    timing = {name: 0 for name in ("h1_inference", "ambiguity_construction", "support_selection",
                                   "prefix_synthesis", "recourse_repair", "fallback", "checker_commit")}
    h1_bundle = None
    if method == METHODS[6]:
        model_key = ("immutable_h1_model", _sha(manifest["toy_history_matrices"]))
        if model_key not in episode_cache:
            episode_cache[model_key] = manifest.get("h1_model_bundle") or _fit_toy_h1_model(manifest)
        h1_bundle = episode_cache[model_key]
    online_start = time.perf_counter_ns()
    legality = True
    wait_latch = None
    executed_actions = 0
    deadline_hit = False
    cursor = 0
    first_action = 80
    completion = 81
    if not spec.executable:
        first_action = 0
        from rlccl.uncertainty.evaluation import _oracle_completion_lower_bound
        completion = _oracle_completion_lower_bound(
            np.asarray(manifest["toy_truth_matrix"], dtype=np.int64),
            world.topology_info, world.time_limit,
        )
        cursor = min(completion, world.time_limit)
    else:
        configured_h, configured_p, configured_lambda = planner_config or (8, 4, .5)
        planner = RobustPrefixPlanner(RobustPrefixConfig(
            configured_h, configured_p,
            0.0 if method in METHODS[4:7] else configured_lambda,
            1 if method in METHODS[4:7] else 8,
        ))
        history = tuple(np.asarray(matrix) for matrix in manifest["toy_history_matrices"])
        recourse_state = None
        active_stage = None
        for slot in range(80):
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                completion = 81
                deadline_hit = True
                break
            cursor = slot
            complete = all(bool(world._possession[index, destination])
                           for index, (_, destination, _) in enumerate(world._atomic))
            if complete:
                completion = cursor
                break
            public_stage = min(slot // 4, 4)
            stage = 4 if method == METHODS[1] else public_stage
            trusted = reveal.observation_for_stage(stage)
            view = build_scheduling_view(trusted)
            if method == METHODS[2] and stage < 4:
                latch = (stage, trusted.state_version)
                if wait_latch != latch:
                    emit("wait_latch_entered", "no_common_action", slot=slot, stage=stage,
                         state_version_before=trusted.state_version,
                         state_version_after=trusted.state_version,
                         observation_digest=view.observation_digest,
                         residual_state_digest=view.residual_state_digest,
                         support_digest=zero_digest, batch_count=0, action_count=0)
                    wait_latch = latch
                continue
            if method in METHODS[4:9]:
                needs_plan = (recourse_state is None or active_stage != stage or
                              recourse_state.plan is None or not recourse_state.plan.batches)
                if needs_plan:
                    if recourse_state is not None and active_stage != stage and recourse_state.plan is not None:
                        discarded_batches = len(recourse_state.plan.batches)
                        discarded_actions = sum(len(batch) for batch in recourse_state.plan.batches)
                        if discarded_batches:
                            emit("suffix_discarded", "reveal", slot=slot, stage=stage,
                                 state_version_before=trusted.state_version,
                                 state_version_after=trusted.state_version,
                                 plan_revision=recourse_state.current_revision,
                                 observation_digest=view.observation_digest,
                                 residual_state_digest=view.residual_state_digest,
                                 support_digest=recourse_state.plan.support_digest,
                                 batch_index=recourse_state.next_batch_index,
                                 batch_count=discarded_batches, action_count=discarded_actions)
                    matrices = history[-8:]
                    point_candidate = None
                    if method == METHODS[4]:
                        point_candidate = np.mean(np.stack(history[-32:]), axis=0)
                    elif method == METHODS[5]:
                        point_candidate = np.array(history[-1], dtype=np.int64, copy=True)
                    if method == METHODS[6]:
                        if h1_bundle is None:
                            raise RuntimeError("immutable H1 model was not prepared before online timing")
                        model, model_state = h1_bundle
                        if "group_coefficients" in model_state:
                            point_candidate, construction_ns, selection_ns, inference_ns = _h1_candidate_from_actual_pool(
                                history_matrices=history, trusted_observation=trusted,
                                model=model, model_state=model_state,
                                construction_seed=construction_seed,
                            )
                            timing["ambiguity_construction"] += construction_ns
                            timing["support_selection"] += selection_ns
                            timing["h1_inference"] += inference_ns
                            matrices = (point_candidate,)
                            point_candidate = None
                        else:
                            inference_start = time.perf_counter_ns()
                            vectors = np.asarray(history, dtype=np.float64).reshape(len(history), -1)
                            normalized = ((vectors[-8:] - model_state["input_mean"]) /
                                          model_state["input_scale"])[None, ...]
                            prediction = (model.predict(normalized)[0] * model_state["target_scale"] +
                                          model_state["target_mean"])
                            point_candidate = prediction.reshape(np.asarray(history[-1]).shape)
                            timing["h1_inference"] += time.perf_counter_ns() - inference_start
                    if stage == 4:
                        matrices = (np.asarray(trusted.observed_matrix),)
                    elif point_candidate is not None:
                        point_start = time.perf_counter_ns()
                        matrices = (_reconcile_point_candidate(
                            candidate=point_candidate, history_matrices=history,
                            trusted_observation=trusted,
                            construction_seed=construction_seed,
                        ),)
                        timing["support_selection"] += time.perf_counter_ns() - point_start
                    if method == METHODS[7]:
                        support, construction_ns, selection_ns = _build_ordinary_boundary_support(
                            history_matrices=history, trusted_observation=trusted, view=view,
                            construction_seed=construction_seed,
                        )
                        timing["ambiguity_construction"] += construction_ns
                        timing["support_selection"] += selection_ns
                    elif method == METHODS[8]:
                        oracle_provenance: Mapping[str, Any] = {"toy": True}
                        if stage < 4:
                            from rlccl.uncertainty.ambiguity import (
                                AmbiguityConstructionView, build_empirical_ambiguity_set,
                                fit_descriptor_normalizer, oracle_support_upper_bound,
                            )
                            ordinary_history = tuple(
                                np.asarray(matrix, dtype=np.int64) for matrix in history[-32:]
                            )
                            if len(ordinary_history) != 32:
                                raise ValueError("oracle support requires the exact 32-step history")
                            ambiguity_start = time.perf_counter_ns()
                            normalizer = fit_descriptor_normalizer(
                                ordinary_history, trusted.topology,
                            )
                            ambiguity_view = AmbiguityConstructionView.from_observation(
                                history_matrices=ordinary_history,
                                history_offsets=tuple(range(-32, 0)),
                                observation=trusted,
                                construction_seed=int(construction_seed),
                                normalizer=normalizer,
                            )
                            ambiguity = build_empirical_ambiguity_set(
                                ambiguity_view, calibration_radius=0.34327919716983946,
                            )
                            timing["ambiguity_construction"] += (
                                time.perf_counter_ns() - ambiguity_start
                            )
                            selection_start = time.perf_counter_ns()
                            selected = oracle_support_upper_bound(
                                ambiguity,
                                truth=np.asarray(manifest["toy_truth_matrix"], dtype=np.int64),
                                k=8,
                            )
                            matrices = selected.matrices
                            oracle_weights = selected.weights
                            oracle_provenance = {
                                "selector": selected.method,
                                "selected_indices": selected.selected_indices,
                                "history_offsets": selected.history_offsets,
                                "approximation": selected.approximation,
                                "observation_constraint_fingerprint": (
                                    ambiguity.observation_constraint_fingerprint
                                ),
                                "normalizer_digest": ambiguity.normalizer_digest,
                                "group_coefficients_digest": ambiguity.group_coefficients_digest,
                                "history_cutoff": ambiguity.history_cutoff,
                                "construction_seed": ambiguity.construction_seed,
                            }
                        else:
                            selection_start = time.perf_counter_ns()
                            oracle_weights = (1.0,)
                        support = oracle_support_from_matrices(
                            matrices=matrices, weights=oracle_weights,
                            requested_k=8, provenance=oracle_provenance, observation=view,
                        )
                        timing["support_selection"] += time.perf_counter_ns() - selection_start
                    else:
                        selection_start = time.perf_counter_ns()
                        support = scenario_support_from_matrices(
                            matrices=matrices, weights=(1.0 / len(matrices),) * len(matrices),
                            method="boundary_scenarios",
                            requested_k=1 if method in METHODS[4:7] else 8, uses_oracle=False,
                            upper_bound_only=False, provenance={"toy": True}, observation=view,
                        )
                        timing["support_selection"] += time.perf_counter_ns() - selection_start
                    synthesis_start = time.perf_counter_ns()
                    plan = planner.plan(view, support)
                    timing["prefix_synthesis"] += time.perf_counter_ns() - synthesis_start
                    revision = 0 if recourse_state is None else recourse_state.current_revision + 1
                    if revision:
                        from dataclasses import replace as _replace
                        plan = _replace(plan, revision=revision)
                    recourse_state = (RecourseState.initial(plan) if recourse_state is None else
                                      RecourseState(plan=plan, executed_actions=recourse_state.executed_actions,
                                                    discarded_actions=(recourse_state.discarded_actions +
                                                                       (discarded_actions if active_stage != stage else 0)),
                                                    execution_start_state_version=recourse_state.execution_start_state_version,
                                                    reason="reveal" if active_stage != stage else "exhaustion",
                                                    current_revision=revision))
                    active_stage = stage
                    emit("plan_built", "initial" if revision == 0 else
                         "reveal" if recourse_state.reason == "reveal" else "exhaustion",
                         slot=slot, stage=stage,
                         state_version_before=trusted.state_version,
                         state_version_after=trusted.state_version,
                         plan_revision=revision, observation_digest=view.observation_digest,
                         residual_state_digest=view.residual_state_digest,
                         requested_k=support.requested_k, actual_k=support.actual_k,
                         support_digest=support.digest, batch_index=-1,
                         batch_count=len(plan.batches),
                         action_count=sum(len(batch) for batch in plan.batches))
                    if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                        deadline_hit = True; completion = 81
                        break
                structural = recourse_state.plan.batches[0] if recourse_state.plan.batches else ()
            else:
                candidates = enumerate_candidates(view)
                structural = pack_candidate_batch(candidates, view.topology)
            if not structural:
                latch = (stage, trusted.state_version)
                if wait_latch != latch:
                    fallback_start = time.perf_counter_ns()
                    emit("wait_latch_entered", "no_common_action", slot=slot, stage=stage,
                         state_version_before=trusted.state_version,
                         state_version_after=trusted.state_version,
                         observation_digest=view.observation_digest,
                         residual_state_digest=view.residual_state_digest,
                         support_digest=(recourse_state.plan.support_digest
                                         if recourse_state is not None and recourse_state.plan is not None
                                         else zero_digest), batch_count=0, action_count=0)
                    timing["fallback"] += time.perf_counter_ns() - fallback_start
                    wait_latch = latch
                continue
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                deadline_hit = True; completion = 81
                break
            proposal = (bind_first_batch(recourse_state, view, trusted_observation=trusted)
                        if method in METHODS[4:9] else Proposal.from_transfers(tuple(
                            bind_action(view, local_token_ordinal=item.local_token_ordinal,
                                        edge_index=item.edge_index, trusted_observation=trusted)
                            for item in structural)))
            transaction = {
                "slot": slot, "stage": stage,
                "state_version_before": trusted.state_version,
                "state_version_after": trusted.state_version,
                "plan_revision": (recourse_state.current_revision
                                    if recourse_state is not None else -1),
                "observation_digest": view.observation_digest,
                "residual_state_digest": view.residual_state_digest,
                "support_digest": (recourse_state.plan.support_digest
                                   if recourse_state is not None and recourse_state.plan is not None
                                   else zero_digest),
                "batch_index": (recourse_state.next_batch_index
                                if recourse_state is not None else -1),
                "batch_count": (len(recourse_state.plan.batches)
                                if recourse_state is not None and recourse_state.plan is not None else 1),
            }
            emit("proposal_bound", action_count=len(proposal.actions), **transaction)
            checker_start = time.perf_counter_ns()
            try:
                result = commit_proposal(world, trusted, proposal)
            except (TypeError, ValueError) as error:
                timing["checker_commit"] += time.perf_counter_ns() - checker_start
                legality = False
                emit("checker_rejected", _checker_rejection_reason(error),
                     action_count=len(proposal.actions), **transaction)
                completion = 81
                break
            timing["checker_commit"] += time.perf_counter_ns() - checker_start
            if not result.legal:
                legality = False
                emit("checker_rejected", "other_rejected", action_count=len(proposal.actions), **transaction)
                completion = 81
                break
            if proposal.actions and first_action == 80:
                first_action = slot
            executed_actions += len(proposal.actions)
            for item, action in zip(structural, proposal.actions):
                emit("action_committed", local_token_ordinal=item.local_token_ordinal,
                     truth_binding_digest=_sha(str(action.token_id)),
                     edge_index=action.edge_index, before_distance=item.before_distance,
                     after_distance=item.after_distance, commit_legal=True,
                     action_count=1, **transaction)
            if method in METHODS[4:9]:
                repair_start = time.perf_counter_ns()
                fresh = reveal.observation_for_stage(stage)
                recourse_state = record_committed_batch(recourse_state, proposal=proposal,
                                                         commit_result=result, fresh_observation=fresh)
                timing["recourse_repair"] += time.perf_counter_ns() - repair_start
            emit("batch_committed", action_count=len(proposal.actions), commit_legal=True,
                 **{**transaction, "state_version_after": result.state_version})
            cursor = slot + 1
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                deadline_hit = True; completion = 81
                break
        else:
            cursor = 80
            completion = 81
        if completion == 81 and all(bool(world._possession[index, destination])
                                    for index, (_, destination, _) in enumerate(world._atomic)):
            completion = cursor
    total_online_ns = 0 if not spec.executable else time.perf_counter_ns() - online_start
    end_reason = ("lower_bound_complete" if not spec.executable and completion <= 80 else
                  "lower_bound_timeout" if not spec.executable else
                  "complete" if completion <= 80 else "illegal" if not legality else
                  "wall_timeout" if deadline_hit else "discrete_timeout")
    end_cursor = (completion if not spec.executable and completion <= 80 else
                  min(cursor, 80) if completion > 80 else completion)
    emit("episode_end", end_reason, slot=end_cursor,
         stage=min(end_cursor // 4, 4), batch_count=0, action_count=0)
    timing_rows = finalize_timing(total_online_ns=total_online_ns, components=timing)
    rows = ({"method": method, "role": spec.role, "uses_oracle": spec.uses_oracle,
             "executable": spec.executable, "completion_slots": completion,
             "first_action_slot": first_action, "prefix_executed_actions": executed_actions,
             "legality": legality, "total_online_ns": total_online_ns,
             "events": tuple(events)},)
    world_digest = _sha(np.asarray(world._possession, dtype=np.int8))
    return PublicEpisodeResult(rows, world_digest, _sha(("rng", method, tuple(events))), timing_rows)


@dataclass(frozen=True, slots=True)
class H1Contract:
    fit_sequence_count: int = 15
    fit_steps: tuple[int, ...] = tuple(range(8, 256))
    fit_example_count: int = 3720
    config: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", {
            "recent_steps": 8, "hidden_layer_sizes": (32,), "activation": "tanh",
            "solver": "adam", "alpha": 1e-4, "batch_size": 256,
            "learning_rate_init": 1e-3, "max_iter": 80, "shuffle": True,
            "early_stopping": False, "seed": 20260731,
        })


def h1_fit_contract() -> H1Contract:
    return H1Contract()


def fit_target_scale(targets: Any) -> np.ndarray:
    scale = np.std(np.asarray(targets, dtype=float), axis=0)
    return np.where(scale > 0, scale, 1.0)


@dataclass(frozen=True, slots=True)
class PointCandidate:
    pool_size: int
    pool_index: int
    history_offset: int


def select_h1_point_candidate(*, prediction: Any, candidate_summaries: Any,
                              fit_target_scale: Any, history_offsets: Any) -> PointCandidate:
    values = np.asarray(candidate_summaries, dtype=float)
    distances = np.sum(((values - np.asarray(prediction)) / np.asarray(fit_target_scale)) ** 2, axis=1)
    minimum = float(distances.min())
    tied = np.flatnonzero(np.abs(distances - minimum) <= 1e-12)
    offsets = np.asarray(history_offsets, dtype=int)
    index = int(max(tied, key=lambda item: (int(offsets[item]), int(item))))
    return PointCandidate(len(values), index, int(offsets[index]))


@dataclass(frozen=True, slots=True)
class ValidationKey:
    sequence_id: str
    method: str
    horizon: int = 0
    prefix: int = 0
    risk_lambda: float = 0.0


def build_toy_validation_registry(*, materialize_metrics: bool) -> tuple[ValidationKey, ...]:
    rows = []
    for sequence in range(15):
        for coordinate in range(20):
            for horizon, prefix in LEGAL_HP:
                for risk in (0.0, .5, 1.0):
                    rows.append(ValidationKey(f"validation-{sequence}", METHODS[7], horizon, prefix, risk))
            rows.extend((ValidationKey(f"validation-{sequence}", METHODS[2]),
                         ValidationKey(f"validation-{sequence}", METHODS[3])))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class SelectedConfig:
    horizon: int
    prefix: int
    risk_lambda: float


def select_validation_config(rows: Sequence[Mapping[str, Any]]) -> SelectedConfig:
    configs = {(int(r["horizon"]), int(r["prefix"]), float(r["risk_lambda"])) for r in rows if r["method"] == METHODS[7]}
    scored = []
    for config in configs:
        selected = [r for r in rows if r["method"] == METHODS[7] and (r["horizon"], r["prefix"], r["risk_lambda"]) == config]
        by_sequence = {}
        for row in selected: by_sequence.setdefault(row["sequence_id"], []).append(row)
        sequence_means = [np.mean([float(r["end_to_end_latency_ms"]) for r in values]) for values in by_sequence.values()]
        sequence_cvars = [sequence_distribution([float(r["end_to_end_latency_ms"]) for r in values]).cvar95 for values in by_sequence.values()]
        online = [np.mean([float(r["total_online_ns"]) for r in values]) for values in by_sequence.values()]
        scored.append(((float(np.mean(sequence_means)), float(np.mean(sequence_cvars)), float(np.mean(online)),
                        config[0], config[1], config[2]), config))
    best_score, best_config = scored[0]
    for score, candidate in scored[1:]:
        decision = 0
        for left, right in zip(score[:3], best_score[:3]):
            if abs(left - right) > 1e-12:
                decision = -1 if left < right else 1
                break
        if decision == 0:
            decision = -1 if score[3:] < best_score[3:] else 1
        if decision < 0:
            best_score, best_config = score, candidate
    return SelectedConfig(*best_config)


def select_primary_comparator(rows: Sequence[Mapping[str, Any]]) -> str:
    choices = (METHODS[2], METHODS[3])
    scores = {}
    for name in choices:
        selected = [r for r in rows if r["method"] == name]
        by_sequence = {}
        for row in selected: by_sequence.setdefault(row["sequence_id"], []).append(float(row["end_to_end_latency_ms"]))
        scores[name] = float(np.mean([np.mean(values) for values in by_sequence.values()]))
    if abs(scores[choices[0]] - scores[choices[1]]) <= 1e-12:
        return METHODS[3]
    return min(choices, key=scores.get)


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    sequence_id: str
    method: str


@dataclass(frozen=True, slots=True)
class TestRegistry:
    episode_keys: tuple[EpisodeKey, ...]
    sequence_keys: tuple[str, ...]


def build_toy_test_registry() -> TestRegistry:
    sequences = tuple(f"test-{i}" for i in range(15))
    episodes = tuple(EpisodeKey(seq, method) for seq in sequences for _ in range(20) for method in METHODS)
    sequence_keys = tuple(f"{seq}:{method}" for seq in sequences for method in METHODS)
    return TestRegistry(episodes, sequence_keys)


def completion_after_slot(slot: int) -> int: return int(slot) + 1
def unfinished_completion(*, time_limit: int) -> int: return int(time_limit) + 1
def end_to_end_latency_ms(*, completion_slots: int, total_online_ns: int) -> float: return float(completion_slots) + float(total_online_ns) / 1e6
def reveal_lead_lag_slots(completion_slots: int) -> int: return int(completion_slots) - 16


@dataclass(frozen=True, slots=True)
class DeadlineOutcome:
    wall_timeout: bool
    discrete_timeout: bool
    completion_slots: int
    total_online_ns: int


def deadline_outcome(*, elapsed_ns: int, deadline_ns: int) -> DeadlineOutcome:
    timed = int(elapsed_ns) > int(deadline_ns)
    return DeadlineOutcome(timed, timed, 81 if timed else 0, int(elapsed_ns))


def finalize_timing(*, total_online_ns: int, components: Mapping[str, int]) -> Mapping[str, int]:
    names = ("h1_inference", "ambiguity_construction", "support_selection", "prefix_synthesis",
             "recourse_repair", "fallback", "checker_commit", "unattributed")
    if int(total_online_ns) < 0 or any(int(value) < 0 for value in components.values()):
        raise ValueError("timing values must be nonnegative")
    result = {name: int(components.get(name, 0)) for name in names}
    result["unattributed"] = int(total_online_ns) - sum(result[name] for name in names[:-1])
    if result["unattributed"] < 0: raise ValueError("timing components exceed total")
    return result


@dataclass(frozen=True, slots=True)
class Distribution:
    mean: float
    p95: float
    p99: float
    cvar95: float


def sequence_distribution(values: Any) -> Distribution:
    data = np.asarray(values, dtype=float)
    q95 = float(np.quantile(data, .95, method="higher")); q99 = float(np.quantile(data, .99, method="higher"))
    return Distribution(float(data.mean()), q95, q99, float(data[data >= q95].mean()))


@dataclass(frozen=True, slots=True)
class Decomposition:
    family_deltas: np.ndarray
    base_seed_deltas: np.ndarray


def decompose_sequence_deltas(deltas: Any, *, families: Sequence[str], base_seeds: Sequence[int]) -> Decomposition:
    data = np.asarray(deltas, dtype=float)
    return Decomposition(data.mean(axis=1), data.mean(axis=0))


@dataclass(frozen=True, slots=True)
class Bootstrap:
    samples: np.ndarray
    mean: float
    lower: float
    upper: float


def family_stratified_bootstrap(deltas: Any, *, samples: int, seed: int) -> Bootstrap:
    data = np.asarray(deltas, dtype=float); rng = np.random.default_rng(seed)
    draws = data[np.arange(data.shape[0])[:, None], rng.integers(0, data.shape[1], size=(data.shape[0], samples))].mean(axis=0)
    ordered = np.sort(draws)
    lower = float(ordered[math.ceil(.025 * (samples - 1))])
    upper = float(ordered[math.ceil(.975 * (samples - 1))])
    return Bootstrap(draws, float(draws.mean()), lower, upper)


def positive_sequence_ess(values: Any) -> float:
    data = np.asarray(values, dtype=float)
    if len(data) < 2: return float(len(data))
    centered = data - data.mean(); variance = float(centered @ centered)
    if variance == 0: return float(len(data))
    accepted = []
    for lag in range(1, min(3, len(data) - 1) + 1):
        rho = float(centered[:-lag] @ centered[lag:] / variance)
        if rho <= 0:
            break
        accepted.append(rho)
    ess = len(data) / (1.0 + 2.0 * sum(accepted))
    return max(1.0, min(float(len(data)), float(ess)))


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    data_status: str
    gate_status: str
    conditions: tuple[int, ...]
    failed_conditions: tuple[int, ...]


def evaluate_h2_conditions(evidence: Mapping[str, Any]) -> GateResult:
    paired = evidence["paired"]
    tests = {
        1: all(paired[c]["mean_e2e_delta"] > 0 and paired[c]["ci_lower"] > 0 for c in ("wait", "partial")),
        2: all(paired[c]["mean_cvar95_delta"] <= 0 for c in ("wait", "partial")),
        3: (all(v > 0 for v in evidence["base_seed_deltas"].values()) and
            sum(v < 0 for v in evidence["family_deltas"].values()) <= 1 and
            all(v <= .1 for v in evidence["family_relative_degradation"].values())),
        4: all(v == 1.0 for v in evidence["legality_rates"].values()),
        5: all(paired[c][f"robust_{k}_timeout_rate"] <= paired[c][f"comparator_{k}_timeout_rate"]
               for c in ("wait", "partial") for k in ("discrete", "wall")),
        6: evidence["scheduling_only_delta"] > 0 and evidence["end_to_end_delta"] >= 0 and evidence["overhead_included"],
        7: evidence["test_sequence_count"] == 15 and evidence["test_episode_count"] == 2700,
        8: all(evidence[name] for name in ("fresh_exclusion_complete", "capability_isolation_complete",
                                           "artifact_chain_complete", "focused_tests_complete")),
    }
    failed = tuple(index for index, passed in tests.items() if not passed)
    status = "FAIL" if failed else "PASS" if evidence.get("environment_complete", False) else "HOLD"
    return GateResult(not failed and status == "PASS", status, str(evidence.get("gate_status", "PENDING_SUPERVISOR")), tuple(tests), failed)


_EVENT_REASONS = {
    "episode_start": {"NONE"}, "plan_built": {"initial", "reveal", "exhaustion", "invalidation"},
    "suffix_discarded": {"reveal", "invalidation"},
    "wait_latch_entered": {"no_common_action"},
    "checker_rejected": {"type_error", "stale_observation", "unrevealed_token",
                         "duplicate_token", "edge_range", "source_possession",
                         "destination_possession", "edge_capacity", "shared_group",
                         "other_rejected"},
    "episode_end": {"complete", "discrete_timeout", "wall_timeout", "illegal",
                    "lower_bound_complete", "lower_bound_timeout"},
    "proposal_bound": {"NONE"}, "action_committed": {"NONE"}, "batch_committed": {"NONE"},
}


def validate_event_reason(kind: str, reason: str) -> None:
    if kind not in _EVENT_REASONS or reason not in _EVENT_REASONS[kind]:
        raise ValueError("invalid event kind/reason pair")


def validate_event_ledger(rows: Sequence[Mapping[str, Any]], *,
                          require_canonical_sentinels: bool = False) -> None:
    if not rows or rows[0]["event_kind"] != "episode_start" or rows[-1]["event_kind"] != "episode_end":
        raise ValueError("event sentinel order")
    if sum(r["event_kind"] == "episode_start" for r in rows) != 1 or sum(r["event_kind"] == "episode_end" for r in rows) != 1:
        raise ValueError("event sentinels must be unique")
    if [int(r["event_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("event indices must be contiguous")
    for row in rows:
        validate_event_reason(str(row["event_kind"]), str(row["reason"]))
        if row_digest(row) != row.get("row_digest") or event_payload_digest(row) != row.get("event_payload_digest"):
            raise ValueError("event digest mismatch")
        if int(row["elapsed_ns"]) != 0:
            raise ValueError("event elapsed_ns is an inapplicable zero sentinel")
        kind = str(row["event_kind"])
        if kind in {"episode_start", "episode_end"}:
            sentinel = (int(row["state_version_before"]), int(row["state_version_after"]),
                        int(row["plan_revision"]), int(row["batch_index"]),
                        int(row["batch_count"]), int(row["action_count"]))
            if sentinel != (-1, -1, -1, -1, 0, 0):
                raise ValueError("episode sentinel fields must use frozen -1/0 values")
            if any(row[name] != "0" * 64 for name in (
                    "observation_digest", "residual_state_digest", "support_digest",
                    "truth_binding_digest")):
                raise ValueError("episode sentinel digests must be zero")
        if kind != "action_committed" and any(int(row[name]) != -1 for name in (
                "local_token_ordinal", "edge_index", "before_distance", "after_distance")):
            raise ValueError("non-action identity/distance fields must be -1")
        if kind != "action_committed" and row["truth_binding_digest"] != "0" * 64:
            raise ValueError("non-action truth binding digest must be zero")
        if (kind == "action_committed" and not bool(row["commit_legal"])) or (
                kind not in {"action_committed", "batch_committed"} and bool(row["commit_legal"])):
            raise ValueError("illegal commit_legal sentinel/domain mismatch")
        if require_canonical_sentinels and kind == "batch_committed" and not bool(row["commit_legal"]):
            raise ValueError("canonical committed batch requires commit_legal=true")
    method = str(rows[0]["method"])
    reject_positions = [i for i, r in enumerate(rows) if r["event_kind"] == "checker_rejected"]
    if reject_positions and (rows[-1]["reason"] != "illegal" or
                             any(r["event_kind"] != "episode_end" for r in rows[reject_positions[0] + 1:])):
        raise ValueError("checker rejection must be terminal illegal")
    if any((int(r["requested_k"]) != 0 or int(r["actual_k"]) != 0) for r in rows if r["event_kind"] != "plan_built"):
        raise ValueError("requested/actual K are nonzero only on plan events")
    plans = [r for r in rows if r["event_kind"] == "plan_built"]
    for row in plans:
        if require_canonical_sentinels and int(row["batch_index"]) != -1:
            raise ValueError("canonical plan_built batch index must be -1")
        requested, actual, stage = int(row["requested_k"]), int(row["actual_k"]), int(row["stage"])
        if method in {METHODS[4], METHODS[5], METHODS[6]} and (requested, actual) != (1, 1):
            raise ValueError("point plan requested/actual K")
        if method == METHODS[7] and (requested != 8 or actual not in ({1} if stage == 4 else {8})):
            raise ValueError("ordinary robust stage actual K")
        if method == METHODS[8] and (requested != 8 or actual not in ({1} if stage == 4 else set(range(1, 9)))):
            raise ValueError("oracle robust stage actual K")
    transactions = [row for row in rows if row["event_kind"] in {
        "proposal_bound", "action_committed", "batch_committed", "checker_rejected"}]
    if method not in METHODS[4:9] and any(int(row["plan_revision"]) != -1 for row in transactions):
        raise ValueError("direct transaction plan revision must be -1")
    if method in METHODS[4:9] and any(int(row["plan_revision"]) < 0 for row in transactions):
        raise ValueError("prefix transaction requires a concrete plan revision")
    rejects = [i for i, r in enumerate(rows) if r["event_kind"] == "checker_rejected"]
    if rejects and (rows[-1]["reason"] != "illegal" or any(r["event_kind"] != "episode_end" for r in rows[rejects[0]+1:])):
        raise ValueError("checker reject must be terminal illegal")
    end = rows[-1]
    end_slot = int(end["slot"])
    committed_batches = [row for row in rows if row["event_kind"] == "batch_committed"]
    if end["reason"] == "illegal" and (not rejects or end_slot != int(rows[rejects[0]]["slot"])):
        raise ValueError("illegal termination cursor must equal precommit rejection slot")
    if end["reason"] == "complete":
        expected_cursor = int(committed_batches[-1]["slot"]) + 1 if committed_batches else 0
        if end_slot != expected_cursor:
                raise ValueError("batch completion cursor must be successful commit slot plus one")
    if end["reason"] in {"discrete_timeout", "lower_bound_timeout"} and end_slot != 80:
        raise ValueError("discrete/lower-bound timeout cursor must be 80")
    if end["reason"] == "lower_bound_complete" and not 0 <= end_slot <= 80:
        raise ValueError("lower-bound completion cursor outside frozen range")
    if end["reason"] == "wall_timeout" and not 0 <= end_slot <= 80:
        raise ValueError("wall-timeout cursor outside frozen range")
    latest_plan: Mapping[str, Any] | None = None
    index = 1
    transaction_fields = ("slot", "stage", "state_version_before", "plan_revision",
                          "observation_digest", "residual_state_digest", "support_digest",
                          "batch_index", "batch_count")
    while index < len(rows) - 1:
        row = rows[index]
        if row["event_kind"] == "plan_built":
            latest_plan = row
        if row["event_kind"] != "proposal_bound":
            if row["event_kind"] in {"action_committed", "batch_committed"}:
                raise ValueError("action or batch outside proposal transaction")
            index += 1
            continue
        cursor = index + 1
        actions: list[Mapping[str, Any]] = []
        while cursor < len(rows) - 1 and rows[cursor]["event_kind"] == "action_committed":
            actions.append(rows[cursor])
            cursor += 1
        if cursor >= len(rows) - 1 or rows[cursor]["event_kind"] != "batch_committed":
            if cursor < len(rows) - 1 and rows[cursor]["event_kind"] == "checker_rejected" and not actions:
                index = cursor + 1
                continue
            raise ValueError("proposal transaction has no terminal batch")
        batch = rows[cursor]
        transaction_rows = [row, *actions, batch]
        if any(any(item[field] != row[field] for field in transaction_fields) for item in transaction_rows):
            raise ValueError("proposal/action/batch transaction continuity mismatch")
        if method in METHODS[4:9] and latest_plan is None:
            raise ValueError("prefix proposal transaction has no plan")
        if (latest_plan is not None and
                (latest_plan["support_digest"] != row["support_digest"] or
                 latest_plan["plan_revision"] != row["plan_revision"])):
            raise ValueError("proposal support/revision does not link to current plan")
        before = int(row["state_version_before"])
        if any(int(item["state_version_before"]) != before or
               int(item["state_version_after"]) != before for item in actions):
            raise ValueError("action state version continuity")
        if (int(batch["state_version_after"]) != before + 1 or
                int(batch["action_count"]) != len(actions) or
                int(row["action_count"]) != len(actions)):
            raise ValueError("batch state/count transaction")
        index = cursor + 1


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    prefix_planned_batches: int = 0
    prefix_planned_actions: int = 0
    prefix_executed_actions: int = 0
    prefix_executed_batches: int = 0
    fallback_events: int = 0
    no_common_action_events: int = 0
    reveal_replan_events: int = 0
    exhaustion_replan_events: int = 0
    invalidation_replan_events: int = 0
    true_replan_events: int = 0
    requested_k: int = 0
    actual_k_min: int = 0
    actual_k_max: int = 0
    completion_slots: int = 81
    legality: bool = True
    discrete_timeout: bool = False
    wall_timeout: bool = False
    wasted_executed_actions: int = 0
    residual_repair_actions: int = 0
    discarded_unexecuted_batches: int = 0
    discarded_unexecuted_actions: int = 0
    wasted_unexecuted_actions: int = 0


def recompute_episode_from_events(rows: Sequence[Mapping[str, Any]]) -> EpisodeMetrics:
    validate_event_ledger(rows)
    plans = [r for r in rows if r["event_kind"] == "plan_built"]
    actions = [r for r in rows if r["event_kind"] == "action_committed"]
    end = rows[-1]
    discarded = [r for r in rows if r["event_kind"] == "suffix_discarded"]
    return EpisodeMetrics(
        prefix_planned_batches=sum(int(r["batch_count"]) for r in plans),
        prefix_planned_actions=sum(int(r["action_count"]) for r in plans),
        prefix_executed_actions=len(actions),
        prefix_executed_batches=sum(r["event_kind"] == "batch_committed" for r in rows),
        fallback_events=sum(r["event_kind"] == "wait_latch_entered" for r in rows),
        no_common_action_events=sum(r["event_kind"] == "wait_latch_entered" for r in rows),
        reveal_replan_events=sum(r["event_kind"] == "plan_built" and r["reason"] == "reveal" for r in rows),
        exhaustion_replan_events=sum(r["event_kind"] == "plan_built" and r["reason"] == "exhaustion" for r in rows),
        invalidation_replan_events=sum(r["event_kind"] == "plan_built" and r["reason"] == "invalidation" for r in rows),
        true_replan_events=sum(r["event_kind"] == "plan_built" and r["reason"] in {"reveal","exhaustion","invalidation"} for r in rows),
        requested_k=max((int(r["requested_k"]) for r in plans), default=0),
        actual_k_min=min((int(r["actual_k"]) for r in plans), default=0),
        actual_k_max=max((int(r["actual_k"]) for r in plans), default=0),
        completion_slots=int(end["slot"]) if end["reason"] in {"complete", "lower_bound_complete"} else 81,
        legality=end["reason"] != "illegal",
        discrete_timeout=end["reason"] in {"discrete_timeout", "wall_timeout", "lower_bound_timeout"},
        wall_timeout=end["reason"] == "wall_timeout", wasted_executed_actions=0,
        residual_repair_actions=sum(int(r["slot"]) >= 16 for r in actions),
        discarded_unexecuted_batches=sum(int(r["batch_count"]) for r in discarded),
        discarded_unexecuted_actions=sum(int(r["action_count"]) for r in discarded),
        wasted_unexecuted_actions=sum(int(r["action_count"]) for r in discarded),
    )


def validate_plan_support_links(rows: Sequence[Mapping[str, Any]], expected: Mapping[int, str]) -> None:
    for row in rows:
        if row["event_kind"] == "plan_built" and expected.get(int(row["event_index"])) != row["support_digest"]:
            raise ValueError("support digest plan link mismatch")


def validate_episode_against_events(episode: EpisodeMetrics, rows: Sequence[Mapping[str, Any]]) -> None:
    expected = recompute_episode_from_events(rows)
    if episode != expected:
        raise ValueError("episode actual K min/max or executed counter/event ledger mismatch")


def validate_episode_termination(rows: Sequence[Mapping[str, Any]], episode: Mapping[str, Any]) -> None:
    end = rows[-1]; reason = end["reason"]
    expected = {"completion_slots": int(end["slot"]) if reason in {"complete", "lower_bound_complete"} else 81,
                "legality": reason != "illegal", "discrete_timeout": reason in {"discrete_timeout", "wall_timeout", "lower_bound_timeout"},
                "wall_timeout": reason == "wall_timeout"}
    if any(episode[key] != value for key, value in expected.items()):
        raise ValueError("end cursor completion legality timeout reason mismatch")


def synthetic_episode_row(**overrides: Any) -> Mapping[str, Any]:
    row = {"unreachable_od_count": 0, "prefix_executed_actions": 0}
    row.update(overrides); return row


def validate_published_episode(episode: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    if int(episode["unreachable_od_count"]) != 0 or any("unreachable" in str(r.get("reason", "")) for r in rows):
        raise ValueError("published episode must have zero unreachable OD")


def synthetic_sequence_row() -> Mapping[str, Any]:
    row = {"split": "test", "coordinate_id": "ALL", "checkpoint": -1, "checkpoint_index": -1,
           "reveal_mode": "ALL", "mode_index": -1, "reveal_seed": -1,
           "method": METHODS[7], "config_digest": "3" * 64, "row_digest": ""}
    row["row_digest"] = row_digest(row)
    return row


def validate_sequence_row(row: Mapping[str, Any]) -> None:
    if (row.get("split"), row.get("coordinate_id"), row.get("checkpoint"), row.get("checkpoint_index")) != ("test", "ALL", -1, -1):
        raise ValueError("sequence sentinel mismatch")
    if (row.get("reveal_mode"), row.get("mode_index"), row.get("reveal_seed")) != ("ALL", -1, -1):
        raise ValueError("sequence reveal sentinel mismatch")
    if row_digest(row) != row.get("row_digest"):
        raise ValueError("sequence row digest mismatch")


def _base_row(method: str = METHODS[7]) -> dict[str, Any]:
    spec = METHOD_REGISTRY[method]
    coordinate = coordinate_id(sequence_digest="2" * 64, checkpoint=32,
                               reveal_mode="random_entries", reveal_seed=202608010,
                               topology_digest="3" * 64, ratios=(0.0, .25, .5, .75, 1.0),
                               cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1")
    return {
        "schema_version": 1, "split": "test", "coordinate_id": coordinate,
        "sequence_id": "toy", "family": FAMILIES[0], "base_seed": 642,
        "sequence_digest": "2" * 64, "checkpoint": 32, "checkpoint_index": 0,
        "reveal_mode": "random_entries", "mode_index": 0, "reveal_seed": 202608010,
        "topology_digest": "3" * 64, "config_digest": "4" * 64, "method": method,
        "role": spec.role, "uses_oracle": spec.uses_oracle, "executable": spec.executable,
    }


def _row_for(columns: Sequence[str], *, kind: str) -> dict[str, Any]:
    row = _base_row()
    defaults: dict[str, Any] = {name: 0 for name in columns}
    defaults.update(row)
    for name in ("legality", "commit_legal"): defaults[name] = True
    for name in ("discrete_timeout", "wall_timeout"): defaults[name] = False
    defaults.update({"horizon": 8, "prefix": 4, "requested_k": 8,
                     "actual_k_min": 1, "actual_k_max": 8, "risk_lambda": .5,
                     "reference_kind": "NONE", "illegal_reason": "NONE"})
    if kind == "event":
        defaults.update({"event_index": 0, "slot": 0, "stage": 0, "event_kind": "plan_built",
                         "reason": "initial", "observation_digest": "5" * 64,
                         "residual_state_digest": "6" * 64, "support_digest": "7" * 64,
                         "actual_k": 8, "truth_binding_digest": "0" * 64})
        defaults["event_payload_digest"] = event_payload_digest(defaults)
    result = {name: defaults.get(name, 0) for name in columns}
    result["row_digest"] = row_digest(result)
    return result


def _event_row(index: int, kind: str, reason: str, **updates: Any) -> dict[str, Any]:
    row = _row_for(EVENT_COLUMNS, kind="event")
    row.update({"event_index": index, "event_kind": kind, "reason": reason,
                "state_version_before": -1, "state_version_after": -1,
                "plan_revision": -1, "observation_digest": "0" * 64,
                "residual_state_digest": "0" * 64, "support_digest": "0" * 64,
                "requested_k": 0, "actual_k": 0, "batch_index": -1,
                "batch_count": 0, "action_count": 0, "elapsed_ns": 0,
                "local_token_ordinal": -1, "edge_index": -1,
                "before_distance": -1, "after_distance": -1,
                "truth_binding_digest": "0" * 64,
                "commit_legal": kind in {"action_committed", "batch_committed"}})
    row.update(updates)
    row["commit_legal"] = kind in {"action_committed", "batch_committed"}
    row["event_payload_digest"] = event_payload_digest(row)
    row["row_digest"] = row_digest(row)
    return row


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (str(bool(value)).lower() if key in BOOL_COLUMNS else
                      str(float(value)) if key in FLOAT_COLUMNS else
                      str(int(value)) if key in INT_COLUMNS else str(value))
                for key, value in row.items()
            }
            normalized = dict(parse_csv_row(path.name, serialized))
            if "event_payload_digest" in normalized:
                normalized["event_payload_digest"] = event_payload_digest(normalized)
            if "row_digest" in normalized:
                normalized["row_digest"] = row_digest(normalized)
            writer.writerow({key: (str(value).lower() if isinstance(value, bool) else value)
                             for key, value in normalized.items()})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_logical_sha(path: Path) -> str:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [parse_csv_row(path.name, row) for row in csv.DictReader(handle)]
        return _sha(rows)
    return _sha(json.loads(path.read_text(encoding="utf-8")))


def _artifact_scientific_sha(path: Path) -> str:
    if path.suffix != ".csv":
        return _artifact_logical_sha(path)
    volatile = {"coordinate_id", "sequence_id", "family", "base_seed", "sequence_digest",
                "reveal_seed", "config_digest", "elapsed_ns", "total_online_ns", "runner_wall_ns",
                "end_to_end_latency_ms", "end_to_end_regret_ms", "row_digest", "event_payload_digest"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [parse_csv_row(path.name, row) for row in csv.DictReader(handle)]
    return _sha([{key: value for key, value in row.items() if key not in volatile} for row in rows])


def write_toy_artifacts(directory: Path, *, final: bool) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    validation = _row_for(VALIDATION_COLUMNS, kind="validation")
    second_coordinate = coordinate_id(sequence_digest="2" * 64, checkpoint=96,
                                      reveal_mode="random_entries", reveal_seed=202608020,
                                      topology_digest="3" * 64, ratios=(0.0, .25, .5, .75, 1.0),
                                      cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1")
    validation_second = dict(validation, coordinate_id=second_coordinate, checkpoint=96,
                             checkpoint_index=1, reveal_seed=202608020, method=METHODS[3],
                             horizon=0, prefix=0, requested_k=0, actual_k_min=0,
                             actual_k_max=0, risk_lambda=0.0)
    validation_second["row_digest"] = row_digest(validation_second)
    third_coordinate = coordinate_id(sequence_digest="2" * 64, checkpoint=160,
                                     reveal_mode="random_entries", reveal_seed=202608030,
                                     topology_digest="3" * 64, ratios=(0.0, .25, .5, .75, 1.0),
                                     cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1")
    validation_wait = dict(validation_second, coordinate_id=third_coordinate, checkpoint=160,
                           checkpoint_index=2, reveal_seed=202608030, method=METHODS[2])
    validation_wait["row_digest"] = row_digest(validation_wait)
    episode = _row_for(EPISODE_COLUMNS, kind="episode")
    episode.update({"prefix_planned_batches": 1, "prefix_planned_actions": 1,
                    "prefix_executed_batches": 1, "prefix_executed_actions": 1,
                    "residual_repair_actions": 0, "completion_slots": 1,
                    "first_action_slot": 0, "actual_k_min": 8, "actual_k_max": 8,
                    "end_to_end_latency_ms": 1.0, "reveal_lead_lag_slots": -15})
    episode["row_digest"] = row_digest(episode)
    event_common = {"slot": 0, "stage": 0, "state_version_before": 0,
                    "state_version_after": 0, "plan_revision": 0,
                    "observation_digest": "5" * 64, "residual_state_digest": "6" * 64,
                    "support_digest": "7" * 64, "batch_index": 0, "batch_count": 1}
    events = (
        _event_row(0, "episode_start", "NONE", slot=-1, stage=-1,
                   observation_digest="0" * 64, residual_state_digest="0" * 64,
                   support_digest="0" * 64, batch_count=0, action_count=0),
        _event_row(1, "plan_built", "initial", requested_k=8, actual_k=8,
                   action_count=1, **{**event_common, "batch_index": -1}),
        _event_row(2, "proposal_bound", "NONE", action_count=1, **event_common),
        _event_row(3, "action_committed", "NONE", action_count=1,
                   local_token_ordinal=0, edge_index=0, before_distance=1,
                   after_distance=0, truth_binding_digest="9" * 64,
                   commit_legal=True, **event_common),
        _event_row(4, "batch_committed", "NONE", action_count=1,
                   **{**event_common, "state_version_after": 1}),
        _event_row(5, "episode_end", "complete", slot=1, stage=0,
                   observation_digest="0" * 64, residual_state_digest="0" * 64,
                   support_digest="0" * 64, batch_count=0, action_count=0),
    )
    validation_rows = tuple(sorted((validation, validation_second, validation_wait), key=lambda row: (
        row["coordinate_id"], row["method"], row["horizon"], row["prefix"], row["risk_lambda"])))
    _write_csv(directory / "raw_validation_metrics.csv", VALIDATION_COLUMNS, validation_rows)
    _write_csv(directory / "raw_test_episode_metrics.csv", EPISODE_COLUMNS, (episode,))
    _write_csv(directory / "raw_test_execution_events.csv", EVENT_COLUMNS, events)
    sequence = _row_for(SEQUENCE_COLUMNS, kind="sequence")
    sequence.update({"coordinate_id": "ALL", "checkpoint": -1, "checkpoint_index": -1,
                     "reveal_mode": "ALL", "mode_index": -1, "reveal_seed": -1,
                     "episode_count": 1, "completion_mean": 1.0, "completion_median": 1.0,
                     "completion_p95": 1.0, "completion_p99": 1.0, "completion_cvar95": 1.0,
                     "end_to_end_mean_ms": 1.0, "end_to_end_median_ms": 1.0,
                     "end_to_end_p95_ms": 1.0, "end_to_end_p99_ms": 1.0,
                     "end_to_end_cvar95_ms": 1.0, "legality_rate": 1.0,
                     "prefix_executed_actions_sum": 1, "residual_repair_actions_sum": 0})
    sequence["row_digest"] = row_digest(sequence)
    _write_csv(directory / "raw_test_sequence_metrics.csv", SEQUENCE_COLUMNS, (sequence,))
    timing_rows_list = []
    for component in ("h1_inference", "ambiguity_construction", "support_selection",
                      "prefix_synthesis", "recourse_repair", "fallback", "checker_commit", "unattributed"):
        timing = _row_for(TIMING_COLUMNS, kind="timing")
        timing.update({"component": component, "elapsed_ns": 0}); timing["row_digest"] = row_digest(timing)
        timing_rows_list.append(timing)
    timing_rows = tuple(sorted(timing_rows_list, key=lambda row: (row["coordinate_id"], row["method"], row["component"])))
    _write_csv(directory / "raw_timing_metrics.csv", TIMING_COLUMNS, timing_rows)
    toy_h1_manifest = toy_manifest()
    _, h1_state = _fit_toy_h1_model(toy_h1_manifest)
    parameter_arrays = tuple({"dtype": array.dtype.str, "shape": list(array.shape),
                              "data": [float(value).hex() for value in array.reshape(-1)]}
                             for array in h1_state["parameter_arrays"])
    h1_artifact = {
        "schema_version": 1, "model_name": "recent_history_mlp", "config": h1_fit_contract().config,
        "fit_sequence_records": [record["sequence_id"] for record in toy_h1_manifest["sequence_records"][:15]],
        "fit_example_count": h1_state["fit_example_count"], "input_mean": np.asarray(h1_state["input_mean"]).tolist(),
        "input_scale": np.asarray(h1_state["input_scale"]).tolist(),
        "target_mean": np.asarray(h1_state["target_mean"]).tolist(),
        "target_scale": np.asarray(h1_state["target_scale"]).tolist(),
        "parameter_arrays": parameter_arrays, "model_state_sha256": h1_state["model_state_sha256"],
        "source_sha256": _sha("rlccl.prediction.models.RecentHistoryMLP"),
        "group_coefficients_digest": _sha("toy-complete-topology"),
        "fit_wall_ns": h1_state["fit_wall_ns"], "fit_cpu_ns": h1_state["fit_cpu_ns"],
    }
    (directory / "h1_best_point_model.json").write_text(json.dumps(_plain(h1_artifact), sort_keys=True), encoding="utf-8")
    scientific_names = tuple(name for name in ARTIFACT_NAMES
                             if name not in {"manifest.json", "summary.json"})
    scientific = {name: _artifact_scientific_sha(directory / name) for name in scientific_names}
    paired_evidence = {
        "wait": {"mean_e2e_delta": 1.0, "ci_lower": .1, "mean_cvar95_delta": 0.0,
                 "robust_discrete_timeout_rate": 0.0, "comparator_discrete_timeout_rate": 0.0,
                 "robust_wall_timeout_rate": 0.0, "comparator_wall_timeout_rate": 0.0},
        "partial": {"mean_e2e_delta": 1.0, "ci_lower": .1, "mean_cvar95_delta": 0.0,
                    "robust_discrete_timeout_rate": 0.0, "comparator_discrete_timeout_rate": 0.0,
                    "robust_wall_timeout_rate": 0.0, "comparator_wall_timeout_rate": 0.0},
    }
    gate_evidence = {
        "paired": paired_evidence, "base_seed_deltas": {seed: 1.0 for seed in BASE_SEEDS},
        "family_deltas": {family: 1.0 for family in FAMILIES},
        "family_relative_degradation": {family: 0.0 for family in FAMILIES},
        "legality_rates": {name: 1.0 for name, item in METHOD_REGISTRY.items() if item.executable},
        "scheduling_only_delta": 1.0, "end_to_end_delta": 1.0, "overhead_included": True,
        "test_sequence_count": 1, "test_episode_count": 1,
        "fresh_exclusion_complete": True, "capability_isolation_complete": True,
        "artifact_chain_complete": True, "focused_tests_complete": True,
        "environment_complete": True, "gate_status": "PENDING_SUPERVISOR",
    }
    gate_result = evaluate_h2_conditions(gate_evidence)
    summary = {
        "schema_version": 1, "study_name": "phase4_early_planning_toy",
        "integrity_complete": bool(final), "evidence_complete": bool(final),
        "data_status": gate_result.data_status if final else "HOLD", "gate_status": "PENDING_SUPERVISOR",
        "selected_config": {"horizon": 8, "prefix": 4, "requested_k": 8, "risk_lambda": .5},
        "selected_primary_comparator": METHODS[3], "test_sequence_count": 1,
        "test_episode_count": 1, "method_metrics": {}, "comparator_evidence": paired_evidence,
        "seed_evidence": gate_evidence["base_seed_deltas"],
        "family_evidence": {"deltas": gate_evidence["family_deltas"],
                            "relative_degradation": gate_evidence["family_relative_degradation"]},
        "timeout_evidence": paired_evidence,
        "legality_evidence": gate_evidence["legality_rates"],
        "timing_evidence": {"scheduling_only_delta": 1.0, "end_to_end_delta": 1.0,
                            "overhead_included": True},
        "conditions_1_to_8": {str(index): index not in gate_result.failed_conditions for index in range(1, 9)},
        "failed_conditions": list(gate_result.failed_conditions), "insufficient_conditions": [],
        "combined_scientific_evidence_sha256": _sha(scientific),
    }
    (directory / "summary.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    hashed_names = tuple(name for name in ARTIFACT_NAMES if name != "manifest.json")
    physical = {name: _file_sha(directory / name) for name in hashed_names}
    logical = {name: _artifact_logical_sha(directory / name) for name in hashed_names}
    method_registry = {name: {"role": value.role, "uses_oracle": value.uses_oracle,
                              "executable": value.executable, "reference_kind": value.reference_kind}
                       for name, value in METHOD_REGISTRY.items()}
    manifest = {
        "schema_version": 1, "study_name": "phase4_early_planning_toy",
        "protocol_sha256": "4246D661D3E9E316B10F730E3AC17B61BDBD15C6677965EFDC1C24F7898F2068",
        "authorized_source_sha256": _file_sha(Path(__file__)), "authorized_test_sha256": {},
        "runner_sha256": "0" * 64, "environment": {"formal": False},
        "old_manifests": toy_h1_manifest["old_manifests"],
        "excluded_sequence_digests": list(toy_h1_manifest["excluded_sequence_digests"]),
        "sequence_records": [{"sequence_id": "toy", "sequence_digest": "2" * 64}],
        "families": list(FAMILIES), "base_seeds": list(BASE_SEEDS), "splits": list(SPLITS),
        "checkpoints": [32, 96, 160, 224],
        "reveal_modes": ["random_entries", "source_totals_first", "source_destination_totals_first",
                         "partial_shards", "time_based_arrival"],
        "reveal_ratios": [0.0, .25, .5, .75, 1.0],
        "seeds": {"mlp": 20260731, "bootstrap": 20260801},
        "topology": {"name": "Rear4GPU"},
        "phase3b_recipe": {"method": "boundary_scenarios", "requested_k": 8,
                           "calibration_radius": 0.34327919716983946},
        "h1_model": {"model_state_sha256": h1_state["model_state_sha256"]},
        "method_registry": method_registry,
        "validation_config_universe": {"legal_hp": LEGAL_HP, "risk_lambda": (0.0, .5, 1.0)},
        "selected_config": summary["selected_config"],
        "selected_primary_comparator": summary["selected_primary_comparator"],
        "timing_contract": {"slot_duration_ms": 1.0, "deadline_kind": DEADLINE_KIND},
        "statistics_contract": {"bootstrap_seed": 20260801, "samples": 10000, "ess_lags": [1, 2, 3]},
        "artifact_names": list(ARTIFACT_NAMES), "artifact_row_counts": {
            "raw_validation_metrics.csv": 3, "raw_test_episode_metrics.csv": 1,
            "raw_test_sequence_metrics.csv": 1, "raw_test_execution_events.csv": 6,
            "raw_timing_metrics.csv": 8},
        "artifact_logical_sha256": logical,
        "artifact_scientific_sha256": scientific,
        "integrity_complete": bool(final), "evidence_complete": bool(final),
        "data_status": gate_result.data_status if final else "HOLD", "gate_status": "PENDING_SUPERVISOR",
        "summary_sha256": physical["summary.json"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return directory


def parse_csv_row(filename: str, row: Mapping[str, str]) -> Mapping[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        text = str(value)
        if key in BOOL_COLUMNS:
            if text not in {"true", "false"}:
                raise ValueError(f"{filename}:{key} must be lowercase boolean")
            parsed[key] = text == "true"
        elif key in FLOAT_COLUMNS:
            number = float(text)
            if not math.isfinite(number) or str(number) != text:
                raise ValueError(f"{filename}:{key} must be finite canonical float")
            parsed[key] = number
        elif key in INT_COLUMNS:
            number = int(text)
            if str(number) != text:
                raise ValueError(f"{filename}:{key} must be canonical decimal integer")
            parsed[key] = number
        else:
            parsed[key] = text
    return parsed


@dataclass(frozen=True, slots=True)
class ArtifactReadback:
    validation_chain: tuple[str, ...]
    test_chain: tuple[str, ...]


def read_back_artifacts(directory: Path, *, require_final: bool) -> ArtifactReadback:
    directory = Path(directory)
    if set(path.name for path in directory.iterdir()) != set(ARTIFACT_NAMES):
        raise ValueError("artifact name universe mismatch")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    exact_manifest_keys = {
        "schema_version", "study_name", "protocol_sha256", "authorized_source_sha256",
        "authorized_test_sha256", "runner_sha256", "environment", "old_manifests",
        "excluded_sequence_digests", "sequence_records", "families", "base_seeds", "splits",
        "checkpoints", "reveal_modes", "reveal_ratios", "seeds", "topology", "phase3b_recipe",
        "h1_model", "method_registry", "validation_config_universe", "selected_config",
        "selected_primary_comparator", "timing_contract", "statistics_contract", "artifact_names",
        "artifact_row_counts", "artifact_logical_sha256", "artifact_scientific_sha256",
        "integrity_complete", "evidence_complete", "data_status", "gate_status", "summary_sha256",
    }
    if set(manifest) != exact_manifest_keys:
        raise ValueError("manifest exact key schema mismatch")
    formal = manifest.get("study_name") == "phase4_early_planning_formal"
    if formal:
        records = manifest.get("sequence_records", [])
        if (len(records) != 45 or [int(record["record_index"]) for record in records] != list(range(45)) or
                len({record["sequence_id"] for record in records}) != 45 or
                len({record["sequence_digest"] for record in records}) != 45 or
                set(record["sequence_digest"] for record in records) & set(manifest["excluded_sequence_digests"])):
            raise ValueError("formal manifest sequence universe/exclusion mismatch")
        frozen = build_formal_sequence_specs()
        for record, spec in zip(records, frozen):
            if (record["sequence_id"], int(record["actual_seed"]), record["split"],
                    int(record["sequence_length"])) != (
                    spec.sequence_id, spec.actual_seed, spec.split, 256):
                raise ValueError("formal sequence identity/seed formula mismatch")
    if require_final and not (manifest.get("integrity_complete") and manifest.get("evidence_complete")):
        raise ValueError("artifact set is not final")
    if not require_final and (manifest.get("integrity_complete") or manifest.get("evidence_complete") or
                              manifest.get("data_status") != "HOLD" or
                              manifest.get("gate_status") != "PENDING_SUPERVISOR"):
        raise ValueError("provisional manifest must be false/false/HOLD/PENDING_SUPERVISOR")
    for name, expected in manifest["artifact_logical_sha256"].items():
        if _artifact_logical_sha(directory / name) != expected:
            raise ValueError("artifact logical hash mismatch")
    for name, expected in manifest["artifact_scientific_sha256"].items():
        if _artifact_scientific_sha(directory / name) != expected:
            raise ValueError("artifact scientific hash mismatch")
    if _file_sha(directory / "summary.json") != manifest.get("summary_sha256"):
        raise ValueError("summary hash mismatch")
    h1_artifact = json.loads((directory / "h1_best_point_model.json").read_text(encoding="utf-8"))
    h1_keys = {"schema_version", "model_name", "config", "fit_sequence_records",
               "fit_example_count", "input_mean", "input_scale", "target_mean",
               "target_scale", "parameter_arrays", "model_state_sha256", "source_sha256",
               "group_coefficients_digest", "fit_wall_ns", "fit_cpu_ns"}
    if set(h1_artifact) != h1_keys:
        raise ValueError("H1 artifact exact key schema mismatch")
    if formal:
        parameters = tuple(np.asarray(value, dtype=np.float64) for value in h1_artifact["parameter_arrays"])
        if (int(h1_artifact["fit_example_count"]) != 3720 or
                len(h1_artifact["fit_sequence_records"]) != 15 or
                not all(np.all(np.isfinite(value)) for value in parameters) or
                _sha(parameters) != h1_artifact["model_state_sha256"]):
            raise ValueError("H1 artifact fit universe/model state digest mismatch")
    parsed_tables: dict[str, list[Mapping[str, Any]]] = {}
    for name, expected_columns in EXACT_COLUMNS.items():
        with (directory / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise ValueError("artifact exact schema mismatch")
            parsed_tables[name] = [parse_csv_row(name, row) for row in reader]
        for row in parsed_tables[name]:
            if row_digest(row) != row.get("row_digest"):
                raise ValueError("artifact row digest mismatch")
    for name, columns in (("raw_test_sequence_metrics.csv", SEQUENCE_COLUMNS),
                          ("raw_timing_metrics.csv", TIMING_COLUMNS)):
        with (directory / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                raise ValueError("artifact exact schema mismatch")
            parsed_tables[name] = [parse_csv_row(name, row) for row in reader]
        if any(row_digest(row) != row.get("row_digest") for row in parsed_tables[name]):
            raise ValueError("artifact row digest mismatch")
    toy_counts = {"raw_validation_metrics.csv": 3, "raw_test_episode_metrics.csv": 1,
                  "raw_test_sequence_metrics.csv": 1, "raw_test_execution_events.csv": 6,
                  "raw_timing_metrics.csv": 8}
    expected_counts = toy_counts if manifest.get("study_name") == "phase4_early_planning_toy" else EXACT_ROW_COUNTS
    for name, count in expected_counts.items():
        if len(parsed_tables.get(name, ())) != count:
            raise ValueError("artifact exact row count mismatch")
    for name, key in PRIMARY_KEYS.items():
        rows = parsed_tables[name]
        identities = [tuple(row[field] for field in key) for row in rows]
        if len(identities) != len(set(identities)) or identities != sorted(identities):
            raise ValueError("artifact primary key duplicate or sort violation")
    record_indices = {record["sequence_id"]: int(record.get("record_index", 0))
                      for record in manifest.get("sequence_records", [])}
    for row in parsed_tables["raw_validation_metrics.csv"]:
        if row["method"] not in {METHODS[2], METHODS[3], METHODS[7]} or row["role"] != "ordinary" or row["uses_oracle"]:
            raise ValueError("validation method role/oracle domain")
        if row["method"] == METHODS[7]:
            if row["requested_k"] != 8 or not (1 <= row["actual_k_min"] <= row["actual_k_max"] <= 8):
                raise ValueError("validation robust actual K domain")
        elif any(row[field] != 0 for field in ("horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max")):
            raise ValueError("direct validation fields must be zero")
        expected_seed, _ = phase4_seeds(record_index=record_indices.get(row["sequence_id"], 0),
                                        checkpoint_index=int(row["checkpoint_index"]),
                                        mode_index=int(row["mode_index"]))
        if int(row["reveal_seed"]) != expected_seed:
            raise ValueError("validation reveal seed formula mismatch")
        expected_coordinate = coordinate_id(
            sequence_digest=row["sequence_digest"], checkpoint=int(row["checkpoint"]),
            reveal_mode=row["reveal_mode"], reveal_seed=int(row["reveal_seed"]),
            topology_digest=row["topology_digest"], ratios=(0.0, .25, .5, .75, 1.0),
            cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1",
        )
        if row["coordinate_id"] != expected_coordinate:
            raise ValueError("validation coordinate digest mismatch")
    if formal:
        validation_rows = parsed_tables["raw_validation_metrics.csv"]
        coordinates = {row["coordinate_id"] for row in validation_rows}
        robust = [row for row in validation_rows if row["method"] == METHODS[7]]
        if (len(coordinates) != 300 or len(robust) != 9000 or
                any(sum(row["method"] == method for row in validation_rows) != 300
                    for method in (METHODS[2], METHODS[3]))):
            raise ValueError("formal validation coordinate/method universe mismatch")
        config_counts: dict[tuple[int, int, float], int] = {}
        for row in robust:
            key = (row["horizon"], row["prefix"], row["risk_lambda"])
            config_counts[key] = config_counts.get(key, 0) + 1
        if set(config_counts) != {(h, p, risk) for h, p in LEGAL_HP for risk in (0.0, .5, 1.0)} or set(config_counts.values()) != {300}:
            raise ValueError("formal validation config universe mismatch")
    event_rows = parsed_tables["raw_test_execution_events.csv"]
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in event_rows:
        groups.setdefault((row["coordinate_id"], row["method"]), []).append(row)
    for rows in groups.values():
        validate_event_ledger(rows, require_canonical_sentinels=True)
    episode_rows = parsed_tables["raw_test_episode_metrics.csv"]
    episode_keys = {(row["coordinate_id"], row["method"]) for row in episode_rows}
    if set(groups) != episode_keys:
        raise ValueError("every episode must have exactly one execution event ledger")
    if manifest.get("study_name") != "phase4_early_planning_toy" and len(groups) != 2700:
        raise ValueError("formal event ledger universe must contain exactly 2700 episodes")
    if formal:
        coordinate_methods: dict[str, set[str]] = {}
        for row in episode_rows:
            coordinate_methods.setdefault(row["coordinate_id"], set()).add(row["method"])
        if len(coordinate_methods) != 300 or any(methods != set(METHODS) for methods in coordinate_methods.values()):
            raise ValueError("formal test coordinate/method universe mismatch")
    for row in episode_rows:
        registry = METHOD_REGISTRY.get(row["method"])
        if registry is None or (row["role"], row["uses_oracle"], row["executable"]) != (registry.role, registry.uses_oracle, registry.executable):
            raise ValueError("episode method capability domain")
        ledger = groups.get((row["coordinate_id"], row["method"]))
        if ledger is None:
            raise ValueError("episode execution event ledger missing")
        recomputed = recompute_episode_from_events(ledger)
        for field in ("prefix_planned_batches", "prefix_planned_actions", "prefix_executed_batches",
                          "prefix_executed_actions", "fallback_events", "no_common_action_events",
                          "reveal_replan_events", "exhaustion_replan_events", "invalidation_replan_events",
                          "true_replan_events", "wasted_executed_actions", "residual_repair_actions",
                          "discarded_unexecuted_batches", "discarded_unexecuted_actions",
                          "wasted_unexecuted_actions",
                          "requested_k", "actual_k_min", "actual_k_max", "completion_slots"):
            if int(row[field]) != int(getattr(recomputed, field)):
                raise ValueError("episode counters disagree with event ledger")
        if (bool(row["legality"]), bool(row["discrete_timeout"]), bool(row["wall_timeout"])) != (
                recomputed.legality, recomputed.discrete_timeout, recomputed.wall_timeout):
            raise ValueError("episode legality/timeout disagrees with event ledger")
    sequence_keys = [(row["sequence_id"], row["method"]) for row in parsed_tables["raw_test_sequence_metrics.csv"]]
    if len(sequence_keys) != len(set(sequence_keys)) or sequence_keys != sorted(sequence_keys):
        raise ValueError("sequence primary key duplicate or sort violation")
    for row in parsed_tables["raw_test_sequence_metrics.csv"]:
        if any((row["split"] != "test", row["coordinate_id"] != "ALL",
                row["checkpoint"] != -1, row["checkpoint_index"] != -1,
                row["reveal_mode"] != "ALL", row["mode_index"] != -1,
                row["reveal_seed"] != -1)):
            raise ValueError("sequence sentinel mismatch")
    timing_keys = [(row["coordinate_id"], row["method"], row["component"])
                   for row in parsed_tables["raw_timing_metrics.csv"]]
    if len(timing_keys) != len(set(timing_keys)) or timing_keys != sorted(timing_keys):
        raise ValueError("timing primary key duplicate or sort violation")
    components = {"h1_inference", "ambiguity_construction", "support_selection", "prefix_synthesis",
                  "recourse_repair", "fallback", "checker_commit", "unattributed"}
    timing_groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in parsed_tables["raw_timing_metrics.csv"]:
        timing_groups.setdefault((row["coordinate_id"], row["method"]), []).append(row)
    for episode in episode_rows:
        rows = timing_groups.get((episode["coordinate_id"], episode["method"]), [])
        if {row["component"] for row in rows} != components:
            raise ValueError("timing table must contain exactly eight components per episode")
        if sum(int(row["elapsed_ns"]) for row in rows) != int(episode["total_online_ns"]):
            raise ValueError("timing components do not sum to episode online total")
    episodes_by_sequence: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for episode in episode_rows:
        episodes_by_sequence.setdefault((episode["sequence_id"], episode["method"]), []).append(episode)
    for row in parsed_tables["raw_test_sequence_metrics.csv"]:
        episodes = episodes_by_sequence.get((row["sequence_id"], row["method"]), [])
        if int(row["episode_count"]) != len(episodes) or not episodes:
            raise ValueError("sequence episode universe mismatch")
        completion = sequence_distribution([float(item["completion_slots"]) for item in episodes])
        latency = sequence_distribution([float(item["end_to_end_latency_ms"]) for item in episodes])
        expected_values = {
            "completion_mean": completion.mean, "completion_p95": completion.p95,
            "completion_p99": completion.p99, "completion_cvar95": completion.cvar95,
            "completion_median": float(np.median([float(item["completion_slots"]) for item in episodes])),
            "end_to_end_mean_ms": latency.mean, "end_to_end_p95_ms": latency.p95,
            "end_to_end_p99_ms": latency.p99, "end_to_end_cvar95_ms": latency.cvar95,
            "end_to_end_median_ms": float(np.median([float(item["end_to_end_latency_ms"]) for item in episodes])),
            "total_online_mean_ns": float(np.mean([int(item["total_online_ns"]) for item in episodes])),
            "total_online_p95_ns": sequence_distribution([int(item["total_online_ns"]) for item in episodes]).p95,
            "total_online_p99_ns": sequence_distribution([int(item["total_online_ns"]) for item in episodes]).p99,
            "legality_rate": float(np.mean([bool(item["legality"]) for item in episodes])),
            "discrete_timeout_rate": float(np.mean([bool(item["discrete_timeout"]) for item in episodes])),
            "wall_timeout_rate": float(np.mean([bool(item["wall_timeout"]) for item in episodes])),
            "prefix_executed_actions_sum": sum(int(item["prefix_executed_actions"]) for item in episodes),
            "discarded_actions_sum": sum(int(item["discarded_unexecuted_actions"]) for item in episodes),
            "true_replan_sum": sum(int(item["true_replan_events"]) for item in episodes),
            "residual_repair_actions_sum": sum(int(item["residual_repair_actions"]) for item in episodes),
        }
        if any(abs(float(row[field]) - float(value)) > 1e-12 for field, value in expected_values.items()):
            raise ValueError("sequence metrics do not recompute from episodes")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    exact_summary_keys = {
        "schema_version", "study_name", "integrity_complete", "evidence_complete", "data_status",
        "gate_status", "selected_config", "selected_primary_comparator", "test_sequence_count",
        "test_episode_count", "method_metrics", "comparator_evidence", "seed_evidence",
        "family_evidence", "timeout_evidence", "legality_evidence", "timing_evidence",
        "conditions_1_to_8", "failed_conditions", "insufficient_conditions",
        "combined_scientific_evidence_sha256",
    }
    if set(summary) != exact_summary_keys:
        raise ValueError("summary exact key schema mismatch")
    if require_final and not (summary["integrity_complete"] and summary["evidence_complete"]):
        raise ValueError("final summary completion flags are false")
    if not require_final and (summary["integrity_complete"] or summary["evidence_complete"] or
                              summary["data_status"] != "HOLD" or
                              summary["gate_status"] != "PENDING_SUPERVISOR"):
        raise ValueError("provisional summary must be false/false/HOLD/PENDING_SUPERVISOR")
    if summary.get("gate_status") != "PENDING_SUPERVISOR":
        raise ValueError("gate status must remain pending supervisor")
    selected = select_validation_config(parsed_tables["raw_validation_metrics.csv"])
    if summary["selected_config"] != {"horizon": selected.horizon, "prefix": selected.prefix,
                                      "requested_k": 8, "risk_lambda": selected.risk_lambda}:
        raise ValueError("summary selected config does not recompute from validation raw")
    if summary["selected_primary_comparator"] != select_primary_comparator(parsed_tables["raw_validation_metrics.csv"]):
        raise ValueError("summary primary comparator does not recompute from validation raw")
    reconstructed_evidence = {
        "paired": summary["comparator_evidence"],
        "base_seed_deltas": summary["seed_evidence"],
        "family_deltas": summary["family_evidence"]["deltas"],
        "family_relative_degradation": summary["family_evidence"]["relative_degradation"],
        "legality_rates": summary["legality_evidence"],
        **summary["timing_evidence"],
        "test_sequence_count": summary["test_sequence_count"],
        "test_episode_count": summary["test_episode_count"],
        "fresh_exclusion_complete": True, "capability_isolation_complete": True,
        "artifact_chain_complete": True, "focused_tests_complete": True,
        "environment_complete": True, "gate_status": "PENDING_SUPERVISOR",
    }
    gate = evaluate_h2_conditions(reconstructed_evidence)
    expected_conditions = {str(index): index not in gate.failed_conditions for index in range(1, 9)}
    expected_data_status = gate.data_status if require_final else "HOLD"
    if (summary["data_status"] != expected_data_status or summary["failed_conditions"] != list(gate.failed_conditions)
            or summary["conditions_1_to_8"] != expected_conditions):
        raise ValueError("summary Gate conditions do not recompute from raw evidence")
    if manifest["data_status"] != summary["data_status"] or manifest["gate_status"] != summary["gate_status"]:
        raise ValueError("manifest/summary Gate status mismatch")
    if formal:
        independently_rebuilt = _build_formal_summary(
            parsed_tables["raw_validation_metrics.csv"], episode_rows,
            parsed_tables["raw_test_sequence_metrics.csv"], selected,
            summary["selected_primary_comparator"])
        if not require_final:
            independently_rebuilt = {**independently_rebuilt,
                                     "integrity_complete": False,
                                     "evidence_complete": False,
                                     "data_status": "HOLD"}
        if independently_rebuilt != summary:
            raise ValueError("formal summary/evidence/Gate does not independently recompute from raw rows")
    return ArtifactReadback(("raw_validation_metrics", "config_selection", "primary_selection"),
                            ("execution_events", "test_episode", "test_sequence", "conditions_1_to_8", "summary"))


def _finalize_toy_artifacts(directory: Path) -> None:
    directory = Path(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({"integrity_complete": True, "evidence_complete": True,
                    "data_status": "FAIL" if summary.get("failed_conditions") else "PASS"})
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"integrity_complete": True, "evidence_complete": True,
                     "data_status": summary["data_status"]})
    names = tuple(name for name in ARTIFACT_NAMES if name != "manifest.json")
    manifest["artifact_logical_sha256"] = {name: _artifact_logical_sha(directory / name) for name in names}
    manifest["artifact_scientific_sha256"] = {
        name: _artifact_scientific_sha(directory / name) for name in names if name != "summary.json"
    }
    manifest["summary_sha256"] = _file_sha(summary_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def publish_toy_artifacts(destination: Path) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.parent / f".phase4-staging-{_sha(str(destination))[:12]}"
    write_toy_artifacts(staging, final=False)
    read_back_artifacts(staging, require_final=False)
    _finalize_toy_artifacts(staging)
    read_back_artifacts(staging, require_final=True)
    staging.replace(destination)
    return destination


def _normalize_runtime_events(events: Sequence[Mapping[str, Any]],
                              common: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    normalized = []
    for source in events:
        row = {name: source.get(name, 0) for name in EVENT_COLUMNS}
        row.update(common)
        row.update({name: source[name] for name in EVENT_COLUMNS if name in source})
        row["event_payload_digest"] = event_payload_digest(row)
        row["row_digest"] = row_digest(row)
        normalized.append(row)
    validate_event_ledger(normalized, require_canonical_sentinels=True)
    return tuple(normalized)


def _episode_row_from_runtime(*, common: Mapping[str, Any], result: PublicEpisodeResult,
                              lower_bound_slots: int, horizon: int, prefix: int,
                              risk_lambda: float, runner_wall_ns: int) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    events = _normalize_runtime_events(result.scientific_rows[0]["events"], common)
    metrics = recompute_episode_from_events(events)
    method = str(common["method"]); spec = METHOD_REGISTRY[method]
    total_online_ns = int(result.scientific_rows[0]["total_online_ns"])
    row = _row_for(EPISODE_COLUMNS, kind="episode")
    row.update(common)
    row.update({
        "reference_kind": spec.reference_kind, "horizon": int(horizon),
        "prefix": int(prefix), "requested_k": metrics.requested_k,
        "actual_k_min": metrics.actual_k_min, "actual_k_max": metrics.actual_k_max,
        "risk_lambda": float(risk_lambda), "completion_slots": metrics.completion_slots,
        "lower_bound_slots": int(lower_bound_slots),
        "oracle_regret_slots": metrics.completion_slots - int(lower_bound_slots),
        "total_online_ns": total_online_ns, "runner_wall_ns": int(runner_wall_ns),
        "end_to_end_latency_ms": end_to_end_latency_ms(
            completion_slots=metrics.completion_slots, total_online_ns=total_online_ns),
        "end_to_end_regret_ms": (end_to_end_latency_ms(
            completion_slots=metrics.completion_slots, total_online_ns=total_online_ns) -
            float(lower_bound_slots)),
        "first_action_slot": int(result.scientific_rows[0]["first_action_slot"]),
        "reveal_lead_lag_slots": reveal_lead_lag_slots(metrics.completion_slots),
        "prefix_planned_batches": metrics.prefix_planned_batches,
        "prefix_planned_actions": metrics.prefix_planned_actions,
        "prefix_executed_batches": metrics.prefix_executed_batches,
        "prefix_executed_actions": metrics.prefix_executed_actions,
        "discarded_unexecuted_batches": metrics.discarded_unexecuted_batches,
        "discarded_unexecuted_actions": metrics.discarded_unexecuted_actions,
        "wasted_executed_actions": metrics.wasted_executed_actions,
        "wasted_unexecuted_actions": metrics.wasted_unexecuted_actions,
        "reveal_replan_events": metrics.reveal_replan_events,
        "exhaustion_replan_events": metrics.exhaustion_replan_events,
        "invalidation_replan_events": metrics.invalidation_replan_events,
        "true_replan_events": metrics.true_replan_events,
        "residual_repair_actions": metrics.residual_repair_actions,
        "no_common_action_events": metrics.no_common_action_events,
        "fallback_events": metrics.fallback_events, "unreachable_od_count": 0,
        "legality": metrics.legality,
        "illegal_reason": events[-1]["reason"] if not metrics.legality else "NONE",
        "discrete_timeout": metrics.discrete_timeout, "wall_timeout": metrics.wall_timeout,
    })
    row = {name: row[name] for name in EPISODE_COLUMNS}
    row["row_digest"] = row_digest(row)
    timing_rows = []
    for component, elapsed in result.timing_metrics.items():
        timing = _row_for(TIMING_COLUMNS, kind="timing")
        timing.update(common); timing.update({"component": component, "elapsed_ns": int(elapsed)})
        timing = {name: timing[name] for name in TIMING_COLUMNS}
        timing["row_digest"] = row_digest(timing); timing_rows.append(timing)
    return row, events, tuple(timing_rows)


def _validation_row_from_runtime(*, common: Mapping[str, Any], result: PublicEpisodeResult,
                                 horizon: int, prefix: int, risk_lambda: float) -> Mapping[str, Any]:
    events = _normalize_runtime_events(result.scientific_rows[0]["events"], common)
    metrics = recompute_episode_from_events(events)
    total = int(result.scientific_rows[0]["total_online_ns"])
    row = _row_for(VALIDATION_COLUMNS, kind="validation"); row.update(common)
    row.update({"horizon": int(horizon), "prefix": int(prefix),
                "requested_k": metrics.requested_k, "actual_k_min": metrics.actual_k_min,
                "actual_k_max": metrics.actual_k_max, "risk_lambda": float(risk_lambda),
                "completion_slots": metrics.completion_slots, "total_online_ns": total,
                "end_to_end_latency_ms": end_to_end_latency_ms(
                    completion_slots=metrics.completion_slots, total_online_ns=total),
                "legality": metrics.legality, "discrete_timeout": metrics.discrete_timeout,
                "wall_timeout": metrics.wall_timeout})
    row = {name: row[name] for name in VALIDATION_COLUMNS}; row["row_digest"] = row_digest(row)
    return row


def _sequence_rows_from_episodes(episodes: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault((str(episode["sequence_id"]), str(episode["method"])), []).append(episode)
    output = []
    for key in sorted(groups):
        values = groups[key]; first = values[0]
        completion_values = [float(row["completion_slots"]) for row in values]
        latency_values = [float(row["end_to_end_latency_ms"]) for row in values]
        completion = sequence_distribution(completion_values); latency = sequence_distribution(latency_values)
        online = sequence_distribution([float(row["total_online_ns"]) for row in values])
        row = _row_for(SEQUENCE_COLUMNS, kind="sequence")
        row.update({name: first[name] for name in COMMON_COLUMNS})
        row.update({"coordinate_id": "ALL", "checkpoint": -1, "checkpoint_index": -1,
                    "reveal_mode": "ALL", "mode_index": -1, "reveal_seed": -1,
                    "episode_count": len(values), "completion_mean": completion.mean,
                    "completion_median": float(np.median(completion_values)),
                    "completion_p95": completion.p95, "completion_p99": completion.p99,
                    "completion_cvar95": completion.cvar95,
                    "end_to_end_mean_ms": latency.mean,
                    "end_to_end_median_ms": float(np.median(latency_values)),
                    "end_to_end_p95_ms": latency.p95, "end_to_end_p99_ms": latency.p99,
                    "end_to_end_cvar95_ms": latency.cvar95,
                    "oracle_regret_mean_slots": float(np.mean([row_["oracle_regret_slots"] for row_ in values])),
                    "total_online_mean_ns": online.mean, "total_online_p95_ns": online.p95,
                    "total_online_p99_ns": online.p99,
                    "legality_rate": float(np.mean([row_["legality"] for row_ in values])),
                    "discrete_timeout_rate": float(np.mean([row_["discrete_timeout"] for row_ in values])),
                    "wall_timeout_rate": float(np.mean([row_["wall_timeout"] for row_ in values])),
                    "prefix_executed_actions_sum": sum(int(row_["prefix_executed_actions"]) for row_ in values),
                    "discarded_actions_sum": sum(int(row_["discarded_unexecuted_actions"]) for row_ in values),
                    "true_replan_sum": sum(int(row_["true_replan_events"]) for row_ in values),
                    "residual_repair_actions_sum": sum(int(row_["residual_repair_actions"]) for row_ in values)})
        row = {name: row[name] for name in SEQUENCE_COLUMNS}; row["row_digest"] = row_digest(row)
        output.append(row)
    return tuple(output)


def _formal_common(*, record: Mapping[str, Any], checkpoint: int, checkpoint_index: int,
                   reveal_mode: str, mode_index: int, reveal_seed: int,
                   topology_digest: str, method: str, horizon: int, prefix: int,
                   risk_lambda: float, phase3b_digest: str, h1_digest: str) -> Mapping[str, Any]:
    spec = METHOD_REGISTRY[method]
    coordinate = coordinate_id(
        sequence_digest=record["sequence_digest"], checkpoint=checkpoint,
        reveal_mode=reveal_mode, reveal_seed=reveal_seed,
        topology_digest=topology_digest, ratios=REVEAL_RATIOS,
        cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1",
    )
    requested = 1 if method in METHODS[4:7] else 8 if method in METHODS[7:9] else 0
    digest = config_digest(
        method=method, role=spec.role, horizon=horizon, prefix=prefix,
        requested_k=requested, risk_lambda=risk_lambda,
        phase3b_recipe_digest=phase3b_digest, h1_model_digest=h1_digest,
        slot_duration_ms=1.0, cooperative_deadline_ns=FORMAL_DEADLINE_NS,
        checker_version="phase1-atomic-v1",
    )
    return {
        "schema_version": 1, "split": str(record["split"]), "coordinate_id": coordinate,
        "sequence_id": str(record["sequence_id"]), "family": str(record["family"]),
        "base_seed": int(record["base_seed"]), "sequence_digest": str(record["sequence_digest"]),
        "checkpoint": int(checkpoint), "checkpoint_index": int(checkpoint_index),
        "reveal_mode": reveal_mode, "mode_index": int(mode_index),
        "reveal_seed": int(reveal_seed), "topology_digest": topology_digest,
        "config_digest": digest, "method": method, "role": spec.role,
        "uses_oracle": spec.uses_oracle, "executable": spec.executable,
    }


def _build_formal_summary(validation: Sequence[Mapping[str, Any]],
                          episodes: Sequence[Mapping[str, Any]],
                          sequences: Sequence[Mapping[str, Any]],
                          selected: SelectedConfig, primary: str, *,
                          focused_tests_complete: bool = True,
                          environment_complete: bool = True) -> Mapping[str, Any]:
    indexed = {(row["sequence_id"], row["method"]): row for row in sequences}
    paired = {}
    comparator_names = {"wait": METHODS[2], "partial": METHODS[3]}
    delta_by_comparator: dict[str, np.ndarray] = {}
    for label, comparator in comparator_names.items():
        matrix = np.zeros((len(FAMILIES), len(BASE_SEEDS)), dtype=np.float64)
        cvar = []
        for fi, family in enumerate(FAMILIES):
            for si, seed in enumerate(BASE_SEEDS):
                robust = next(row for row in sequences if row["family"] == family and row["base_seed"] == seed and row["method"] == METHODS[7])
                other = next(row for row in sequences if row["family"] == family and row["base_seed"] == seed and row["method"] == comparator)
                matrix[fi, si] = float(other["end_to_end_mean_ms"]) - float(robust["end_to_end_mean_ms"])
                cvar.append(float(other["end_to_end_cvar95_ms"]) - float(robust["end_to_end_cvar95_ms"]))
        delta_by_comparator[label] = matrix
        bootstrap = family_stratified_bootstrap(matrix, samples=10000, seed=20260801)
        robust_episodes = [row for row in episodes if row["method"] == METHODS[7]]
        comparator_episodes = [row for row in episodes if row["method"] == comparator]
        paired[label] = {
            "mean_e2e_delta": float(matrix.mean()), "ci_lower": bootstrap.lower,
            "ci_upper": bootstrap.upper, "mean_cvar95_delta": float(np.mean(cvar)),
            "robust_discrete_timeout_rate": float(np.mean([row["discrete_timeout"] for row in robust_episodes])),
            "comparator_discrete_timeout_rate": float(np.mean([row["discrete_timeout"] for row in comparator_episodes])),
            "robust_wall_timeout_rate": float(np.mean([row["wall_timeout"] for row in robust_episodes])),
            "comparator_wall_timeout_rate": float(np.mean([row["wall_timeout"] for row in comparator_episodes])),
            "sequence_ess": positive_sequence_ess(matrix.reshape(-1)),
        }
    primary_label = "wait" if primary == METHODS[2] else "partial"
    matrix = delta_by_comparator[primary_label]
    decomposition = decompose_sequence_deltas(matrix, families=FAMILIES, base_seeds=BASE_SEEDS)
    family_deltas = {family: float(decomposition.family_deltas[index]) for index, family in enumerate(FAMILIES)}
    base_deltas = {str(seed): float(decomposition.base_seed_deltas[index]) for index, seed in enumerate(BASE_SEEDS)}
    relative = {}
    for family in FAMILIES:
        comparator_value = float(np.mean([
            row["end_to_end_mean_ms"] for row in sequences
            if row["family"] == family and row["method"] == primary
        ]))
        relative[family] = max(0.0, -family_deltas[family]) / max(abs(comparator_value), 1e-12)
    legality = {method: float(np.mean([row["legality"] for row in episodes if row["method"] == method])) for method in METHODS}
    scheduling_delta = float(np.mean([
        indexed[(sequence_id, primary)]["completion_mean"] - indexed[(sequence_id, METHODS[7])]["completion_mean"]
        for sequence_id in {row["sequence_id"] for row in sequences}
    ]))
    evidence = {
        "paired": paired, "base_seed_deltas": base_deltas, "family_deltas": family_deltas,
        "family_relative_degradation": relative, "legality_rates": legality,
        "scheduling_only_delta": scheduling_delta,
        "end_to_end_delta": float(matrix.mean()), "overhead_included": True,
        "test_sequence_count": 15, "test_episode_count": 2700,
        "fresh_exclusion_complete": True, "capability_isolation_complete": True,
        "artifact_chain_complete": True,
        "focused_tests_complete": bool(focused_tests_complete),
        "environment_complete": bool(environment_complete),
        "gate_status": "PENDING_SUPERVISOR",
    }
    gate = evaluate_h2_conditions(evidence)
    method_metrics = {method: {
        "completion_mean": float(np.mean([row["completion_mean"] for row in sequences if row["method"] == method])),
        "end_to_end_mean_ms": float(np.mean([row["end_to_end_mean_ms"] for row in sequences if row["method"] == method])),
    } for method in METHODS}
    return {
        "schema_version": 1, "study_name": "phase4_early_planning_formal",
        "integrity_complete": True, "evidence_complete": True,
        "data_status": gate.data_status, "gate_status": "PENDING_SUPERVISOR",
        "selected_config": {"horizon": selected.horizon, "prefix": selected.prefix,
                            "requested_k": 8, "risk_lambda": selected.risk_lambda},
        "selected_primary_comparator": primary, "test_sequence_count": 15,
        "test_episode_count": 2700, "method_metrics": method_metrics,
        "comparator_evidence": paired, "seed_evidence": base_deltas,
        "family_evidence": {"deltas": family_deltas, "relative_degradation": relative},
        "timeout_evidence": {label: {key: value for key, value in data.items() if "timeout_rate" in key} for label, data in paired.items()},
        "legality_evidence": legality,
        "timing_evidence": {"scheduling_only_delta": scheduling_delta,
                            "end_to_end_delta": float(matrix.mean()), "overhead_included": True},
        "conditions_1_to_8": {str(index): index not in gate.failed_conditions for index in range(1, 9)},
        "failed_conditions": list(gate.failed_conditions), "insufficient_conditions": [],
        "combined_scientific_evidence_sha256": _sha((validation, episodes, sequences)),
    }


def _publish_formal_artifacts(*, destination: Path, manifest: Mapping[str, Any],
                              h1_artifact: Mapping[str, Any],
                              validation: Sequence[Mapping[str, Any]],
                              episodes: Sequence[Mapping[str, Any]],
                              sequences: Sequence[Mapping[str, Any]],
                              events: Sequence[Mapping[str, Any]],
                              timing: Sequence[Mapping[str, Any]],
                              summary: Mapping[str, Any]) -> Path:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.parent / f".phase4-staging-{_sha(str(destination.resolve()))[:12]}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        _write_csv(staging / "raw_validation_metrics.csv", VALIDATION_COLUMNS, validation)
        _write_csv(staging / "raw_test_episode_metrics.csv", EPISODE_COLUMNS, episodes)
        _write_csv(staging / "raw_test_sequence_metrics.csv", SEQUENCE_COLUMNS, sequences)
        _write_csv(staging / "raw_test_execution_events.csv", EVENT_COLUMNS, events)
        _write_csv(staging / "raw_timing_metrics.csv", TIMING_COLUMNS, timing)
        (staging / "h1_best_point_model.json").write_text(json.dumps(_plain(h1_artifact), sort_keys=True), encoding="utf-8")
        names = tuple(name for name in ARTIFACT_NAMES if name != "manifest.json")
        row_counts = {"raw_validation_metrics.csv": len(validation),
                      "raw_test_episode_metrics.csv": len(episodes),
                      "raw_test_sequence_metrics.csv": len(sequences),
                      "raw_test_execution_events.csv": len(events),
                      "raw_timing_metrics.csv": len(timing)}

        def write_phase(phase_summary: Mapping[str, Any], *, final: bool) -> None:
            (staging / "summary.json").write_text(
                json.dumps(_plain(phase_summary), sort_keys=True), encoding="utf-8")
            phase_manifest = dict(manifest)
            phase_manifest.update({
                "artifact_row_counts": row_counts,
                "artifact_logical_sha256": {name: _artifact_logical_sha(staging / name) for name in names},
                "artifact_scientific_sha256": {name: _artifact_scientific_sha(staging / name) for name in names if name != "summary.json"},
                "integrity_complete": final, "evidence_complete": final,
                "data_status": phase_summary["data_status"],
                "gate_status": "PENDING_SUPERVISOR",
                "summary_sha256": _file_sha(staging / "summary.json"),
            })
            (staging / "manifest.json").write_text(
                json.dumps(_plain(phase_manifest), sort_keys=True), encoding="utf-8")

        provisional_summary = {**summary, "integrity_complete": False,
                               "evidence_complete": False, "data_status": "HOLD",
                               "gate_status": "PENDING_SUPERVISOR"}
        write_phase(provisional_summary, final=False)
        read_back_artifacts(staging, require_final=False)
        write_phase(summary, final=True)
        read_back_artifacts(staging, require_final=True)
        staging.replace(destination)
    except BaseException:
        # Fail closed and leave the staging directory for forensic inspection.
        raise
    return destination


@dataclass(frozen=True, slots=True)
class FormalResourceEstimate:
    sequence_count: int = 45
    validation_episode_count: int = 9600
    test_episode_count: int = 2700
    maximum_slot_iterations: int = 984000
    fixed_csv_rows_excluding_events: int = 34035
    minimum_event_rows: int = 5400
    deadline_upper_bound_hours: float = 34.166666666666664


def estimate_formal_resources() -> FormalResourceEstimate:
    """Static fail-safe bound: 12,300 episodes, 80 slots, and 10 s/episode."""
    return FormalResourceEstimate()


def run_formal_experiment(*, destination: Path, project_root: Path,
                          excluded_sequence_digests: Sequence[str],
                          protocol_sha256: str, authorized_source_sha256: str,
                          authorized_test_sha256: str,
                          focused_tests_complete: bool,
                          environment_complete: bool) -> Path:
    """Run and atomically publish the frozen formal universe; never overwrites output.

    Merely importing this function performs no generation and creates no directory.
    All 12,300 method episodes are evaluated before the single final directory rename.
    """
    destination = Path(destination); project_root = Path(project_root)
    if destination.exists():
        raise FileExistsError(destination)
    for name, digest in (("protocol", protocol_sha256), ("source", authorized_source_sha256),
                         ("test", authorized_test_sha256)):
        if len(str(digest)) != 64 or any(char not in "0123456789abcdefABCDEF" for char in str(digest)):
            raise ValueError(f"{name} SHA-256 must be a 64-digit hexadecimal digest")
    if len(tuple(excluded_sequence_digests)) != len(set(excluded_sequence_digests)):
        raise ValueError("excluded sequence digest universe contains duplicates")
    if not excluded_sequence_digests:
        raise ValueError("formal run requires the complete nonempty old-corpus exclusion universe")
    if not focused_tests_complete or not environment_complete:
        raise ValueError("formal publication requires completed focused tests and admitted environment")
    from rlccl.models.traffic_predictor import deterministic_group_coefficients
    from rlccl.uncertainty.ambiguity import group_coefficients_digest
    from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology
    from rlccl.uncertainty.evaluation import _oracle_completion_lower_bound

    materialized = materialize_formal_sequence_records(
        excluded_sequence_digests=excluded_sequence_digests)
    by_split = {split: [record for record in materialized if record["split"] == split]
                for split in SPLITS}
    if any(len(by_split[split]) != 15 for split in SPLITS):
        raise ValueError("formal split universe must contain exactly 15 sequences per split")
    topology, topology_digest = _load_rear4_topology(project_root)
    groups = deterministic_group_coefficients(topology)
    h1_model, h1_artifact = fit_h1_best_point_model(
        fit_sequences=[record["sequence"] for record in by_split["fit"]],
        group_coefficients=groups,
    )
    h1_artifact = {
        **h1_artifact,
        "fit_sequence_records": tuple({"sequence_id": record["sequence_id"],
                                       "sequence_digest": record["sequence_digest"]}
                                      for record in by_split["fit"]),
        "group_coefficients_digest": group_coefficients_digest(topology),
    }
    h1_bundle = (h1_model, {**h1_artifact, "group_coefficients": groups})
    h1_digest = str(h1_artifact["model_state_sha256"])
    phase3b_recipe = {"method": "boundary_scenarios", "requested_k": 8,
                      "calibration_radius": 0.34327919716983946}
    phase3b_digest = _sha(phase3b_recipe)

    validation_rows: list[Mapping[str, Any]] = []
    for record in by_split["validation"]:
        matrices = tuple(np.asarray(matrix, dtype=np.int64) for matrix in record["sequence"].matrices)
        coordinate_manifest = {"toy_history_matrices": (), "toy_truth_matrix": None}
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            history = matrices[checkpoint - 32:checkpoint]
            truth = matrices[checkpoint]
            coordinate_manifest.update({"toy_history_matrices": history, "toy_truth_matrix": truth})
            for mode_index, reveal_mode in enumerate(REVEAL_MODES):
                reveal_seed, construction_seed = phase4_seeds(record_index=record["record_index"],
                                                               checkpoint_index=checkpoint_index,
                                                               mode_index=mode_index)
                for horizon, prefix in LEGAL_HP:
                    for risk_lambda in (0.0, .5, 1.0):
                        method = METHODS[7]
                        common = _formal_common(
                            record=record, checkpoint=checkpoint, checkpoint_index=checkpoint_index,
                            reveal_mode=reveal_mode, mode_index=mode_index,
                            reveal_seed=reveal_seed, topology_digest=topology_digest, method=method,
                            horizon=horizon, prefix=prefix, risk_lambda=risk_lambda,
                            phase3b_digest=phase3b_digest, h1_digest=h1_digest)
                        episode = build_formal_episode(
                            truth_matrix=truth, topology=topology, method=method,
                            sequence_id=record["sequence_id"], sequence_step=checkpoint,
                            family=record["family"], reveal_mode=reveal_mode,
                            reveal_seed=reveal_seed)
                        result = run_public_episode(
                            manifest=coordinate_manifest, coordinate_id=common["coordinate_id"],
                            method=method, episode=episode, episode_cache={},
                            planner_config=(horizon, prefix, risk_lambda),
                            deadline_ns=time.perf_counter_ns() + FORMAL_DEADLINE_NS,
                            construction_seed=construction_seed)
                        validation_rows.append(_validation_row_from_runtime(
                            common=common, result=result, horizon=horizon, prefix=prefix,
                            risk_lambda=risk_lambda))
                for method in (METHODS[2], METHODS[3]):
                    common = _formal_common(
                        record=record, checkpoint=checkpoint, checkpoint_index=checkpoint_index,
                        reveal_mode=reveal_mode, mode_index=mode_index,
                        reveal_seed=reveal_seed, topology_digest=topology_digest, method=method,
                        horizon=0, prefix=0, risk_lambda=0.0,
                        phase3b_digest=phase3b_digest, h1_digest=h1_digest)
                    episode = build_formal_episode(
                        truth_matrix=truth, topology=topology, method=method,
                        sequence_id=record["sequence_id"], sequence_step=checkpoint,
                        family=record["family"], reveal_mode=reveal_mode,
                        reveal_seed=reveal_seed)
                    result = run_public_episode(
                        manifest=coordinate_manifest, coordinate_id=common["coordinate_id"],
                        method=method, episode=episode, episode_cache={},
                        deadline_ns=time.perf_counter_ns() + FORMAL_DEADLINE_NS,
                        construction_seed=construction_seed)
                    validation_rows.append(_validation_row_from_runtime(
                        common=common, result=result, horizon=0, prefix=0, risk_lambda=0.0))
    if len(validation_rows) != 9600:
        raise ValueError("formal validation runner did not materialize exactly 9600 rows")
    selected = select_validation_config(validation_rows)
    primary = select_primary_comparator(validation_rows)

    episode_rows: list[Mapping[str, Any]] = []
    event_rows: list[Mapping[str, Any]] = []
    timing_rows: list[Mapping[str, Any]] = []
    for record in by_split["test"]:
        matrices = tuple(np.asarray(matrix, dtype=np.int64) for matrix in record["sequence"].matrices)
        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
            history = matrices[checkpoint - 32:checkpoint]; truth = matrices[checkpoint]
            coordinate_manifest = {"toy_history_matrices": history, "toy_truth_matrix": truth,
                                   "h1_model_bundle": h1_bundle}
            lower_bound = _oracle_completion_lower_bound(truth, topology, 80)
            for mode_index, reveal_mode in enumerate(REVEAL_MODES):
                reveal_seed, construction_seed = phase4_seeds(record_index=record["record_index"],
                                                               checkpoint_index=checkpoint_index,
                                                               mode_index=mode_index)
                coordinate = coordinate_id(
                    sequence_digest=record["sequence_digest"], checkpoint=checkpoint,
                    reveal_mode=reveal_mode, reveal_seed=reveal_seed,
                    topology_digest=topology_digest, ratios=REVEAL_RATIOS,
                    cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1")
                for method in rotated_method_order(coordinate):
                    is_prefix = method in METHODS[4:9]
                    horizon = selected.horizon if is_prefix else 0
                    prefix = selected.prefix if is_prefix else 0
                    risk_lambda = (0.0 if method in METHODS[4:7] else
                                   selected.risk_lambda if is_prefix else 0.0)
                    common = _formal_common(
                        record=record, checkpoint=checkpoint, checkpoint_index=checkpoint_index,
                        reveal_mode=reveal_mode, mode_index=mode_index,
                        reveal_seed=reveal_seed, topology_digest=topology_digest, method=method,
                        horizon=horizon, prefix=prefix, risk_lambda=risk_lambda,
                        phase3b_digest=phase3b_digest, h1_digest=h1_digest)
                    wall_start = time.perf_counter_ns()
                    episode = build_formal_episode(
                        truth_matrix=truth, topology=topology, method=method,
                        sequence_id=record["sequence_id"], sequence_step=checkpoint,
                        family=record["family"], reveal_mode=reveal_mode,
                        reveal_seed=reveal_seed)
                    result = run_public_episode(
                        manifest=coordinate_manifest, coordinate_id=coordinate, method=method,
                        episode=episode, episode_cache={},
                        planner_config=(selected.horizon, selected.prefix, selected.risk_lambda),
                        deadline_ns=time.perf_counter_ns() + FORMAL_DEADLINE_NS,
                        construction_seed=construction_seed)
                    episode_row, events, timings = _episode_row_from_runtime(
                        common=common, result=result, lower_bound_slots=lower_bound,
                        horizon=horizon, prefix=prefix, risk_lambda=risk_lambda,
                        runner_wall_ns=time.perf_counter_ns() - wall_start)
                    episode_rows.append(episode_row); event_rows.extend(events); timing_rows.extend(timings)
    sequence_rows = list(_sequence_rows_from_episodes(episode_rows))
    if (len(episode_rows), len(sequence_rows), len(timing_rows)) != (2700, 135, 21600):
        raise ValueError("formal test runner row universe is incomplete")
    validation_rows.sort(key=lambda row: tuple(row[field] for field in PRIMARY_KEYS["raw_validation_metrics.csv"]))
    episode_rows.sort(key=lambda row: (row["coordinate_id"], row["method"]))
    event_rows.sort(key=lambda row: (row["coordinate_id"], row["method"], row["event_index"]))
    sequence_rows.sort(key=lambda row: (row["sequence_id"], row["method"]))
    timing_rows.sort(key=lambda row: (row["coordinate_id"], row["method"], row["component"]))
    summary = _build_formal_summary(
        validation_rows, episode_rows, sequence_rows, selected, primary,
        focused_tests_complete=focused_tests_complete,
        environment_complete=environment_complete)
    sequence_records = [{key: value for key, value in record.items() if key != "sequence"}
                        for record in materialized]
    manifest = {
        "schema_version": 1, "study_name": "phase4_early_planning_formal",
        "protocol_sha256": protocol_sha256,
        "authorized_source_sha256": authorized_source_sha256,
        "authorized_test_sha256": authorized_test_sha256,
        "runner_sha256": _file_sha(Path(__file__)),
        "environment": {"formal": True, "python": os.sys.version},
        "old_manifests": {"h1": H1_MANIFEST_SHA256, "phase3b": PHASE3B_MANIFEST_SHA256},
        "excluded_sequence_digests": list(excluded_sequence_digests),
        "sequence_records": sequence_records, "families": list(FAMILIES),
        "base_seeds": list(BASE_SEEDS), "splits": list(SPLITS),
        "checkpoints": list(CHECKPOINTS), "reveal_modes": list(REVEAL_MODES),
        "reveal_ratios": list(REVEAL_RATIOS),
        "seeds": {"mlp": 20260731, "bootstrap": 20260801},
        "topology": {"name": "Rear4GPU", "sha256": topology_digest},
        "phase3b_recipe": phase3b_recipe,
        "h1_model": {"model_state_sha256": h1_digest},
        "method_registry": {name: {field: getattr(spec, field) for field in spec.__dataclass_fields__}
                            for name, spec in METHOD_REGISTRY.items()},
        "validation_config_universe": {"legal_hp": LEGAL_HP, "risk_lambda": (0.0, .5, 1.0)},
        "selected_config": summary["selected_config"],
        "selected_primary_comparator": primary,
        "timing_contract": {"slot_duration_ms": 1.0, "deadline_kind": DEADLINE_KIND,
                            "cooperative_deadline_ns": FORMAL_DEADLINE_NS},
        "statistics_contract": {"bootstrap_seed": 20260801, "samples": 10000,
                                "ess_lags": [1, 2, 3]},
        "artifact_names": list(ARTIFACT_NAMES), "artifact_row_counts": {},
        "artifact_logical_sha256": {}, "artifact_scientific_sha256": {},
        "integrity_complete": False, "evidence_complete": False,
        "data_status": "HOLD", "gate_status": "PENDING_SUPERVISOR",
        "summary_sha256": "0" * 64,
    }
    return _publish_formal_artifacts(
        destination=destination, manifest=manifest, h1_artifact=h1_artifact,
        validation=validation_rows, episodes=episode_rows, sequences=sequence_rows,
        events=event_rows, timing=timing_rows, summary=summary)


def run_ordinary_stage(*, view: Any, support: Any, plan: Any, method: str) -> Mapping[str, Any]:
    return {
        "method": str(method), "stage": int(view.stage), "ratio": float(view.ratio),
        "state_version": int(view.state_version), "observation_digest": view.observation_digest,
        "residual_state_digest": view.residual_state_digest, "support_digest": support.digest,
        "requested_k": int(support.requested_k), "actual_k": int(support.actual_k),
        "structural_actions": tuple(plan.structural_actions),
    }


@dataclass(frozen=True, slots=True)
class OrdinaryPrefixRun:
    scientific_rows: tuple[Mapping[str, Any], ...]


def run_ordinary_prefix(*, world: Any, reveal_process: Any, history_matrices: Sequence[np.ndarray],
                        method: str, config: Any, stop_before_slot: int) -> OrdinaryPrefixRun:
    from rlccl.scheduling.robust_prefix import RobustPrefixPlanner, build_scheduling_view
    from rlccl.scheduling.scenario_adapter import scenario_support_from_matrices
    from rlccl.scheduling.recourse import RecourseState, bind_first_batch, record_committed_batch
    from rlccl.uncertainty.execution import commit_proposal
    rows: list[Mapping[str, Any]] = []
    planner = RobustPrefixPlanner(config)
    state = None
    active_stage = None
    for slot in range(int(stop_before_slot)):
        stage = min(slot // 4, 4)
        trusted = reveal_process.observation_for_stage(stage)
        view = build_scheduling_view(trusted)
        if state is None or active_stage != stage or state.plan is None or not state.plan.batches:
            support, _, _ = _build_ordinary_boundary_support(
                history_matrices=history_matrices, trusted_observation=trusted, view=view,
                construction_seed=203608010 + stage,
            )
            revision = 0 if state is None else state.plan.revision + 1
            plan = planner.plan(view, support)
            if revision:
                from dataclasses import replace as _replace
                plan = _replace(plan, revision=revision)
            state = RecourseState.initial(plan) if state is None else RecourseState(
                plan=plan, executed_actions=state.executed_actions,
                discarded_actions=state.discarded_actions,
                execution_start_state_version=state.execution_start_state_version,
                reason="reveal" if active_stage != stage else "exhaustion",
                current_revision=revision,
            )
            active_stage = stage
            rows.append({**run_ordinary_stage(view=view, support=support, plan=plan, method=method),
                         "slot": slot, "event_kind": "plan_built"})
        if not state.plan.batches:
            rows.append({"method": method, "slot": slot, "stage": stage, "event_kind": "wait",
                         "observation_digest": view.observation_digest})
            continue
        executed_structural = tuple((item.local_token_ordinal, item.edge_index)
                                    for item in state.plan.batches[0])
        proposal = bind_first_batch(state, view, trusted_observation=trusted)
        result = commit_proposal(world, trusted, proposal)
        fresh = reveal_process.observation_for_stage(stage)
        state = record_committed_batch(state, proposal=proposal, commit_result=result,
                                       fresh_observation=fresh)
        rows.append({"method": method, "slot": slot, "stage": stage,
                     "event_kind": "batch_committed", "action_count": len(proposal.actions),
                     "state_version": result.state_version,
                     "structural_actions": executed_structural})
    return OrdinaryPrefixRun(tuple(rows))


__all__ = ["ARTIFACT_NAMES", "DEADLINE_KIND", "row_digest", "event_payload_digest",
           "config_digest", "coordinate_id", "phase4_seeds", "rotated_method_order",
           "run_ordinary_stage", "run_ordinary_prefix", "OrdinaryPrefixRun",
           "run_formal_experiment", "estimate_formal_resources", "FormalResourceEstimate"]
