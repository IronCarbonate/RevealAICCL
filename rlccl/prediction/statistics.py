"""Sequence-level metrics, uncertainty calibration, bootstrap, and H1 Gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PRIMARY_TARGETS = (
    "total_traffic",
    "source_load_vector",
    "destination_load_vector",
)
EXPECTED_BASE_SEEDS = {42, 142, 242}
EXPECTED_FAMILIES = {
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    first = _average_ranks(actual)
    second = _average_ranks(predicted)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.sqrt((first @ first) * (second @ second)))
    return float(first @ second / denominator) if denominator > 0.0 else 0.0


def sequence_target_metrics(
    actual: Mapping[str, np.ndarray], predicted: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Compute each target block separately with equal component weighting."""

    if set(actual) != set(predicted):
        raise ValueError("actual and predicted target blocks must match")
    result: dict[str, dict[str, float]] = {}
    for name in actual:
        truth = np.asarray(actual[name], dtype=np.float64)
        estimate = np.asarray(predicted[name], dtype=np.float64)
        if truth.shape != estimate.shape or truth.size == 0:
            raise ValueError(f"shape mismatch for target {name}")
        if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
            raise ValueError(f"target {name} contains NaN/Inf")
        flat_truth = truth.reshape(-1)
        flat_estimate = estimate.reshape(-1)
        errors = flat_estimate - flat_truth
        denominator = float(np.sum((flat_truth - flat_truth.mean()) ** 2))
        squared = float(errors @ errors)
        r2 = 1.0 - squared / denominator if denominator > 0.0 else (1.0 if squared == 0 else 0.0)
        result[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "r2": float(r2),
            "spearman": _spearman(flat_truth, flat_estimate),
        }
    return result


def hotspot_from_destination_loads(destination_loads: np.ndarray) -> np.ndarray:
    loads = np.asarray(destination_loads, dtype=np.float64)
    if loads.ndim != 2 or loads.shape[1] == 0:
        raise ValueError("destination_loads must be a nonempty 2-D array")
    return np.argmax(loads, axis=1).astype(np.int64)


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    samples: np.ndarray


def family_stratified_paired_bootstrap(
    deltas: np.ndarray,
    families: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20260731,
) -> BootstrapResult:
    """Resample complete test sequences within family, then equally average."""

    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    labels = np.asarray(families, dtype=object).reshape(-1)
    if values.shape != labels.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("deltas/families must be same-length, nonempty, and finite")
    if samples <= 0:
        raise ValueError("samples must be positive")
    unique = list(dict.fromkeys(labels.tolist()))
    groups = [np.flatnonzero(labels == family) for family in unique]
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        selected = [rng.choice(group, size=len(group), replace=True) for group in groups]
        draws[draw] = float(np.mean(values[np.concatenate(selected)]))
    lower, upper = np.quantile(draws, (0.025, 0.975), method="linear")
    return BootstrapResult(float(values.mean()), float(lower), float(upper), draws)


def autocorrelation(values: np.ndarray, *, max_lag: int = 64) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or max_lag < 0:
        raise ValueError("values must be nonempty and max_lag nonnegative")
    limit = min(int(max_lag), len(array) - 1)
    result = np.zeros(int(max_lag) + 1, dtype=np.float64)
    result[0] = 1.0
    centered = array - array.mean()
    variance = float(centered @ centered)
    if variance <= 0.0:
        return result
    for lag in range(1, limit + 1):
        result[lag] = float(centered[:-lag] @ centered[lag:] / variance)
    return result


def positive_sequence_ess(values: np.ndarray, *, max_lag: int = 64) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    correlations = autocorrelation(array, max_lag=max_lag)
    positive: list[float] = []
    for rho in correlations[1 : min(len(array), len(correlations))]:
        if rho <= 0.0:
            break
        positive.append(float(rho))
    return float(len(array) / (1.0 + 2.0 * sum(positive)))


def quantile_interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal_coverage: float = 0.80,
) -> dict[str, float]:
    truth = np.asarray(actual, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if truth.shape != low.shape or truth.shape != high.shape or not truth.size:
        raise ValueError("actual/lower/upper shapes must match and be nonempty")
    if not np.isfinite(truth).all() or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("interval arrays contain NaN/Inf")
    if np.any(low > high):
        raise ValueError("interval lower bound exceeds upper bound")
    coverage = float(np.mean((truth >= low) & (truth <= high)))
    return {
        "coverage": coverage,
        "interval_width": float(np.mean(high - low)),
        "calibration_error": abs(coverage - float(nominal_coverage)),
    }


def scenario_envelope_metrics(
    actual: np.ndarray,
    scenarios: np.ndarray,
    *,
    nominal_coverage: float = 0.80,
) -> dict[str, Any]:
    truth = np.asarray(actual, dtype=np.float64)
    values = np.asarray(scenarios, dtype=np.float64)
    if values.ndim != truth.ndim + 1 or values.shape[0] != truth.shape[0] or values.shape[2:] != truth.shape[1:]:
        raise ValueError("scenarios must have shape [example, scenario, ...target]")
    if values.shape[1] != 64:
        raise ValueError("Gate H1 requires exactly 64 joint scenarios")
    lower = np.quantile(values, 0.10, axis=1, method="linear")
    upper = np.quantile(values, 0.90, axis=1, method="linear")
    metrics: dict[str, Any] = quantile_interval_metrics(
        truth, lower, upper, nominal_coverage=nominal_coverage
    )
    metrics.update({"lower": lower, "upper": upper})
    return metrics


def evaluate_h1_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically evaluate the seven preregistered H1 conditions."""

    seeds = evidence["seed_mean_deltas"]
    families = evidence["family_mean_deltas"]
    primary = evidence["primary_ci_lowers"]
    lofo = evidence["lofo"]
    probability = evidence["probability"]
    tail = evidence["tail"]
    condition_1 = float(evidence["pooled_total_ci_lower"]) > 0.0
    condition_2 = (
        set(seeds) == EXPECTED_BASE_SEEDS
        and all(float(value) > 0.0 for value in seeds.values())
        and set(families) == EXPECTED_FAMILIES
        and sum(float(value) > 0.0 for value in families.values()) >= 4
    )
    condition_3 = set(primary) == set(PRIMARY_TARGETS) and all(
        float(primary[name]) > 0.0 for name in PRIMARY_TARGETS
    )
    lofo_family = lofo["family_mean_deltas"]
    lofo_changes = lofo["relative_rmse_changes"]
    condition_4 = (
        float(lofo["aggregate_mean_delta"]) >= 0.0
        and set(lofo_family) == EXPECTED_FAMILIES
        and set(lofo_changes) == EXPECTED_FAMILIES
        and sum(float(value) > 0.0 for value in lofo_family.values()) >= 3
        and sum(float(value) > 0.10 for value in lofo_changes.values()) <= 1
    )
    family_calibration = probability["family_interval_calibration_errors"]
    condition_5 = (
        float(probability["overall_interval_calibration_error"]) <= 0.05
        and set(family_calibration) == EXPECTED_FAMILIES
        and all(
            float(value) <= 0.10
            for value in family_calibration.values()
        )
        and float(probability["overall_scenario_calibration_error"]) <= 0.10
    )
    sufficient_tail = int(tail["event_count"]) >= 10
    condition_6 = sufficient_tail and tail.get("recall") is not None and float(tail["recall"]) >= 0.70
    condition_7 = bool(evidence["integrity_tests_passed"]) and not bool(
        evidence["supervisor_veto"]
    )
    conditions = {
        "condition_1": bool(condition_1),
        "condition_2": bool(condition_2),
        "condition_3": bool(condition_3),
        "condition_4": bool(condition_4),
        "condition_5": bool(condition_5),
        "condition_6": bool(condition_6),
        "condition_7": bool(condition_7),
    }
    reasons = [name for name, passed in conditions.items() if not passed]
    if not sufficient_tail:
        reasons.append("insufficient_events")
        explicit_failures = [
            passed
            for name, passed in conditions.items()
            if name != "condition_6"
        ]
        decision = "HOLD" if all(explicit_failures) else "FAIL"
    else:
        decision = "PASS" if all(conditions.values()) else "FAIL"
    return {"decision": decision, "conditions": conditions, "reasons": reasons}
