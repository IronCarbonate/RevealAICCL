"""Independent diagnostics for generated traffic-matrix sequences.

The audit intentionally consumes the public :class:`TrafficSequence` output.  It
does not alter or reach into the generator, so intermediate pre-clip/pre-round
values that the current API discards are reported as unavailable rather than
reconstructed speculatively.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from rlccl.traffic.types import TrafficSequence


DEFAULT_ACF_LAGS = (1, 2, 4, 8, 16, 32, 64)


def _as_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _describe(values: Sequence[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            key: None
            for key in ("mean", "variance", "std", "min", "max", "p50", "p90", "p95", "p99", "p99_9")
        }
    result: dict[str, float | None] = {
        "mean": _as_float(array.mean()),
        "variance": _as_float(array.var(ddof=0)),
        "std": _as_float(array.std(ddof=0)),
        "min": _as_float(array.min()),
        "max": _as_float(array.max()),
    }
    for label, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("p99_9", 0.999)):
        result[label] = _as_float(np.quantile(array, quantile))
    return result


def run_lengths(mask: Sequence[bool] | np.ndarray) -> list[int]:
    """Return lengths of consecutive true runs."""
    runs: list[int] = []
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _max_run(mask: np.ndarray) -> int:
    runs = run_lengths(mask)
    return max(runs, default=0)


def autocorrelation(
    values: Sequence[float] | np.ndarray,
    lags: Iterable[int] = DEFAULT_ACF_LAGS,
) -> dict[str, Any]:
    """Compute global-mean-normalized ACF and a positive-sequence ESS.

    ESS is undefined for a constant series and is therefore emitted as null.
    """
    array = np.asarray(values, dtype=np.float64)
    centered = array - array.mean()
    denominator = float(np.dot(centered, centered))
    requested = tuple(sorted({int(lag) for lag in lags if int(lag) > 0}))
    if denominator <= 1e-15:
        return {
            "defined": False,
            "reason": "constant series",
            "values": {str(lag): None for lag in requested},
            "effective_sample_size": None,
        }

    acf: dict[str, float | None] = {}
    for lag in requested:
        acf[str(lag)] = (
            _as_float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            if lag < len(array)
            else None
        )

    positive_sum = 0.0
    for lag in range(1, len(array)):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
        if rho <= 0.0:
            break
        positive_sum += rho
    ess = len(array) / (1.0 + 2.0 * positive_sum)
    return {
        "defined": True,
        "reason": None,
        "values": acf,
        "effective_sample_size": float(np.clip(ess, 1.0, len(array))),
    }


def _skewness(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=0))
    if array.size == 0 or std <= 1e-15:
        return None
    return _as_float(np.mean(((array - array.mean()) / std) ** 3))


def _rolling_moment_metrics(
    stack: np.ndarray,
    mean_ref: np.ndarray,
    var_ref: np.ndarray,
    window_size: int,
    epsilon_mean: float,
    epsilon_var: float,
) -> dict[str, Any]:
    length = int(stack.shape[0])
    if window_size > length:
        return {
            "available": False,
            "reason": f"window_size={window_size} exceeds sequence_length={length}",
            "window_size": int(window_size),
            "num_windows": 0,
        }

    values = stack.astype(np.float64, copy=False)
    cumulative = np.concatenate(
        [np.zeros((1,) + values.shape[1:], dtype=np.float64), np.cumsum(values, axis=0)],
        axis=0,
    )
    cumulative_sq = np.concatenate(
        [np.zeros((1,) + values.shape[1:], dtype=np.float64), np.cumsum(values * values, axis=0)],
        axis=0,
    )
    sums = cumulative[window_size:] - cumulative[:-window_size]
    sums_sq = cumulative_sq[window_size:] - cumulative_sq[:-window_size]
    means = sums / float(window_size)
    variances = np.maximum(sums_sq / float(window_size) - means * means, 0.0)
    axes = tuple(range(1, means.ndim))
    mean_denominator = float(np.linalg.norm(mean_ref)) + 1e-8
    var_denominator = float(np.linalg.norm(var_ref)) + 1e-8
    mean_errors = np.sqrt(np.sum((means - mean_ref) ** 2, axis=axes)) / mean_denominator
    var_errors = np.sqrt(np.sum((variances - var_ref) ** 2, axis=axes)) / var_denominator
    violations = (mean_errors > epsilon_mean) | (var_errors > epsilon_var)
    return {
        "available": True,
        "reason": None,
        "window_size": int(window_size),
        "num_windows": int(len(mean_errors)),
        "mean_error": _describe(mean_errors),
        "variance_error": _describe(var_errors),
        "mean_violation_fraction": float(np.mean(mean_errors > epsilon_mean)),
        "variance_violation_fraction": float(np.mean(var_errors > epsilon_var)),
        "any_violation_fraction": float(np.mean(violations)),
        "max_consecutive_violation_windows": _max_run(violations),
        "epsilon_mean": float(epsilon_mean),
        "epsilon_var": float(epsilon_var),
    }


def _periodicity(stack: np.ndarray, max_period_lag: int) -> dict[str, Any]:
    length = int(stack.shape[0])
    tokens = [matrix.tobytes() for matrix in np.ascontiguousarray(stack)]
    limit = min(int(max_period_lag), length // 2)
    ratios: dict[str, float] = {}
    candidate_lags = {lag for lag in DEFAULT_ACF_LAGS if lag <= limit}
    candidate_lags.update(lag for lag in range(1, limit + 1) if tokens[lag] == tokens[0])
    detected: int | None = None
    strongest_lag: int | None = None
    strongest_ratio = -1.0
    for lag in sorted(candidate_lags):
        ratio = float(np.mean([left == right for left, right in zip(tokens[:-lag], tokens[lag:])]))
        ratios[str(lag)] = ratio
        if ratio > strongest_ratio:
            strongest_lag, strongest_ratio = lag, ratio
        if detected is None and ratio == 1.0:
            # Confirm byte equality with an array equality check before declaring a period.
            if np.array_equal(stack[:-lag], stack[lag:]):
                detected = lag
    return {
        "detected_exact_period": detected,
        "strongest_checked_lag": strongest_lag,
        "strongest_exact_repeat_ratio": strongest_ratio if strongest_lag is not None else None,
        "exact_repeat_ratio_by_lag": ratios,
        "max_period_lag": limit,
    }


def audit_sequence(
    sequence: TrafficSequence,
    *,
    short_window: int = 16,
    medium_window: int = 128,
    long_window: int = 512,
    acf_lags: Iterable[int] = DEFAULT_ACF_LAGS,
    max_period_lag: int = 512,
    generation_seconds: float | None = None,
) -> dict[str, Any]:
    """Audit one generated sequence; the sequence is the statistical unit."""
    if not sequence.matrices:
        raise ValueError("Traffic sequence must contain at least one matrix")
    stack = np.stack(sequence.matrices, axis=0)
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        raise ValueError("Traffic matrices must stack to [time, nodes, nodes]")
    length, num_nodes, _ = stack.shape
    off_diagonal = ~np.eye(num_nodes, dtype=bool)

    total = stack.sum(axis=(1, 2)).astype(np.float64)
    total_stats = _describe(total)
    total_mean = float(total.mean())
    total_std = float(total.std(ddof=0))
    high_2 = total > total_mean + 2.0 * total_std
    high_3 = total > total_mean + 3.0 * total_std
    burst_positions = np.flatnonzero(high_2)
    total_stats.update(
        {
            "max_to_mean_ratio": float(total.max() / total_mean) if total_mean > 0 else None,
            "fraction_above_mean_plus_2std": float(high_2.mean()),
            "fraction_above_mean_plus_3std": float(high_3.mean()),
            "high_load_run_lengths": run_lengths(high_2),
            "high_load_run_length_stats": _describe(run_lengths(high_2)),
            "burst_intervals": np.diff(burst_positions).astype(int).tolist(),
            "burst_interval_stats": _describe(np.diff(burst_positions)),
        }
    )

    adjacent_delta = np.diff(stack.astype(np.float64), axis=0)
    adjacent_l1 = np.abs(adjacent_delta).sum(axis=(1, 2))
    adjacent_l2 = np.sqrt((adjacent_delta * adjacent_delta).sum(axis=(1, 2)))
    normalization = max(float(num_nodes * (num_nodes - 1) * max(int(stack.max()), 1)), 1.0)
    receive = stack.sum(axis=1).astype(np.float64)
    hotspots = np.argmax(receive, axis=1)
    # The generic boolean helper is not suitable for categorical dwell times.
    hotspot_runs = []
    dwell = 1
    for previous, current in zip(hotspots[:-1], hotspots[1:]):
        if current == previous:
            dwell += 1
        else:
            hotspot_runs.append(dwell)
            dwell = 1
    hotspot_runs.append(dwell)

    tokens = [matrix.tobytes() for matrix in np.ascontiguousarray(stack)]
    unique_count = len(set(tokens))
    near_threshold = 0.05
    normalized_adjacent_l1 = adjacent_l1 / normalization
    regime_records = None
    regime_record_name = None
    for name in ("regime_dwell_records", "volatility_dwell_records", "variant_dwell_records"):
        if sequence.metadata.get(name):
            regime_records = sequence.metadata[name]
            regime_record_name = name
            break
    temporal = {
        "total_traffic_acf": autocorrelation(total, acf_lags),
        "adjacent_matrix_l1": _describe(adjacent_l1),
        "adjacent_matrix_l2": _describe(adjacent_l2),
        "adjacent_normalized_l1": _describe(normalized_adjacent_l1),
        "hotspot_destination_dwell_lengths": hotspot_runs,
        "hotspot_destination_dwell_stats": _describe(hotspot_runs),
        "hotspot_destination_migrations": int(np.sum(hotspots[1:] != hotspots[:-1])),
        "hotspot_destination_series": hotspots.astype(int).tolist(),
        "regime_dwell": (
            {
                "available": True,
                "source": regime_record_name,
                "lengths": [int(record["length"]) for record in regime_records],
                "statistics": _describe([record["length"] for record in regime_records]),
            }
            if regime_records is not None
            else {
                "available": False,
                "reason": "generator output contains no latent regime labels",
            }
        ),
        "exact_duplicate_ratio": float(1.0 - unique_count / length),
        "num_unique_matrices": unique_count,
        "adjacent_near_duplicate_ratio": (
            float(np.mean(normalized_adjacent_l1 <= near_threshold)) if length > 1 else None
        ),
        "near_duplicate_normalized_l1_threshold": near_threshold,
        "periodicity": _periodicity(stack, max_period_lag),
        "labeled_hotspot_destination_series": sequence.metadata.get("hotspot_destination"),
        "labeled_hotspot_source_series": sequence.metadata.get("hotspot_source"),
        "shock_count": int(np.sum(sequence.metadata.get("shock_flags", []))),
    }

    send = stack.sum(axis=2).astype(np.float64)
    nonzero_total = np.maximum(total, 1.0)
    send_share = send.max(axis=1) / nonzero_total
    receive_share = receive.max(axis=1) / nonzero_total
    probabilities = stack[:, off_diagonal].astype(np.float64) / nonzero_total[:, None]
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0
    log_probabilities[positive] = np.log(probabilities[positive])
    entropy = -np.sum(probabilities * log_probabilities, axis=1)
    spatial = {
        "source_load_by_node": {
            str(node): _describe(send[:, node]) for node in range(num_nodes)
        },
        "destination_load_by_node": {
            str(node): _describe(receive[:, node]) for node in range(num_nodes)
        },
        "off_diagonal_sparsity": _describe(np.mean(stack[:, off_diagonal] == 0, axis=1)),
        "max_source_share": _describe(send_share),
        "max_destination_share": _describe(receive_share),
        "off_diagonal_entropy": _describe(entropy),
        "normalized_off_diagonal_entropy": _describe(
            entropy / np.log(max(num_nodes * (num_nodes - 1), 2))
        ),
        "destination_hotspot_strength": _describe(receive.max(axis=1) * num_nodes / nonzero_total),
        "source_load_variance": _describe(np.var(send, axis=1, ddof=0)),
        "destination_load_variance": _describe(np.var(receive, axis=1, ddof=0)),
        "source_load_skewness": _skewness(send.ravel()),
        "destination_load_skewness": _skewness(receive.ravel()),
        "bandwidth_group_concentration": {
            "available": False,
            "reason": (
                "TrafficSequence has source-destination demand only; mapping demand to routed "
                "topology bandwidth groups requires a routing/schedule policy and is not identifiable here"
            ),
        },
    }

    windows = {
        str(window): _rolling_moment_metrics(
            stack,
            sequence.mean_ref,
            sequence.var_ref,
            window,
            sequence.bounds.epsilon_mean,
            sequence.bounds.epsilon_var,
        )
        for window in dict.fromkeys((short_window, medium_window, long_window))
    }

    attempt = int(sequence.metadata.get("generation_attempt", 1))
    intermediate_available = "pre_clip_statistics" in sequence.metadata
    generation = {
        "attempts": attempt,
        "rejections_before_acceptance": max(0, attempt - 1),
        "rejection_rate_before_acceptance": float(max(0, attempt - 1) / attempt),
        "wall_time_seconds": None if generation_seconds is None else float(generation_seconds),
        "intermediate_instrumentation_available": intermediate_available,
        "pre_clip": sequence.metadata.get("pre_clip_statistics"),
        "post_clip_pre_round": sequence.metadata.get("post_clip_statistics"),
        "post_round_pre_diagonal": None,
        "unavailable_reason": None if intermediate_available else (
            "generate_traffic_sequence returns only final integer matrices and does not retain "
            "pre-clip, pre-round, or rejected candidate arrays"
        ),
        "final_integer": {
            "dtype": str(stack.dtype),
            "minimum": int(stack.min()),
            "maximum": int(stack.max()),
            "diagonal_nonzero_count": int(np.count_nonzero(np.diagonal(stack, axis1=1, axis2=2))),
            "off_diagonal_zero_fraction": float(np.mean(stack[:, off_diagonal] == 0)),
            "off_diagonal_at_max_fraction": float(
                np.mean(stack[:, off_diagonal] == int(stack.max()))
            ),
        },
        "smoothing_indicator": {
            "mean_adjacent_normalized_l1": (
                float(normalized_adjacent_l1.mean()) if length > 1 else None
            ),
            "note": "descriptive output statistic; the generator exposes no explicit smoothing stage",
        },
    }

    return {
        "sequence_id": sequence.sequence_id,
        "family": sequence.family,
        "dynamics_variant": sequence.metadata.get("dynamics_variant"),
        "seed": int(sequence.seed),
        "sequence_length": int(length),
        "num_nodes": int(num_nodes),
        "topology_name": sequence.topology_name,
        "generator_kind": sequence.metadata.get("generator_kind", "legacy short-horizon generator"),
        "metadata_usage": sequence.metadata.get("metadata_usage"),
        "constraint_status": sequence.metadata.get("constraint_status"),
        "calibration_candidates_evaluated": sequence.metadata.get(
            "calibration_candidates_evaluated"
        ),
        "spatial_distribution_validation": sequence.metadata.get(
            "spatial_distribution_validation"
        ),
        "generator_window_size": int(sequence.metadata.get("window_size", short_window)),
        "total_traffic": total_stats,
        "temporal": temporal,
        "spatial": spatial,
        "multi_window_moments": windows,
        "generation": generation,
        "generator_multi_scale_constraints": sequence.metadata.get("multi_scale_constraints"),
        "generator_matrix_multi_scale_constraints": sequence.metadata.get(
            "matrix_multi_scale_constraints"
        ),
    }


def summarize_audits(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scalar evidence while preserving family/length/seed groups."""
    grouped: dict[tuple[str, int, int, str | None], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record["family"],
            record["sequence_length"],
            record["seed_base"],
            record.get("dynamics_variant"),
        )
        grouped.setdefault(key, []).append(record)

    summaries = []
    for (family, length, seed_base, dynamics_variant), rows in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        successes = [row for row in rows if row.get("status") == "success"]
        failures = len(rows) - len(successes)
        item: dict[str, Any] = {
            "family": family,
            "sequence_length": length,
            "seed_base": seed_base,
            "dynamics_variant": dynamics_variant,
            "num_requested": len(rows),
            "num_succeeded": len(successes),
            "num_failed": failures,
            "generation_failure_rate": failures / len(rows),
        }
        if successes:
            item.update(
                {
                    "mean_generation_seconds": float(
                        np.mean([row["generation"]["wall_time_seconds"] for row in successes])
                    ),
                    "mean_generation_attempts": float(
                        np.mean([row["generation"]["attempts"] for row in successes])
                    ),
                    "mean_total_cv": float(
                        np.mean([
                            (row["total_traffic"]["std"] / row["total_traffic"]["mean"])
                            if row["total_traffic"]["mean"] else 0.0
                            for row in successes
                        ])
                    ),
                    "mean_exact_duplicate_ratio": float(
                        np.mean([row["temporal"]["exact_duplicate_ratio"] for row in successes])
                    ),
                    "detected_exact_periods": sorted(
                        Counter(
                            row["temporal"]["periodicity"]["detected_exact_period"]
                            for row in successes
                        ).items(),
                        key=lambda pair: str(pair[0]),
                    ),
                    "window_violation_fraction": {
                        window: float(
                            np.mean([
                                row["multi_window_moments"][window]["any_violation_fraction"]
                                for row in successes
                                if row["multi_window_moments"][window]["available"]
                            ])
                        )
                        if any(row["multi_window_moments"][window]["available"] for row in successes)
                        else None
                        for window in successes[0]["multi_window_moments"]
                    },
                }
            )
        summaries.append(item)
    return {"groups": summaries, "num_records": len(records)}
