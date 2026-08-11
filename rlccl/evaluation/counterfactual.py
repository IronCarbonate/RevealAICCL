"""Pure helpers for same-current-traffic/different-history diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..traffic.moment_estimator import SlidingMomentEstimator
from ..traffic.moment_validation import relative_l2_error
from ..traffic.types import MomentContext


@dataclass(frozen=True)
class CounterfactualHistoryPair:
    """One current matrix paired with two strictly prior history windows."""

    pair_id: str
    family: str
    seed: int
    current_matrix: np.ndarray
    history_a: tuple[np.ndarray, ...]
    history_b: tuple[np.ndarray, ...]
    mean_ref: np.ndarray
    var_ref: np.ndarray

    def __post_init__(self) -> None:
        current = np.asarray(self.current_matrix, dtype=np.int64)
        object.__setattr__(self, "current_matrix", current)
        for name in ("history_a", "history_b"):
            history = tuple(np.asarray(item, dtype=np.int64) for item in getattr(self, name))
            if not history:
                raise ValueError(f"{name} must not be empty")
            if any(item.shape != current.shape for item in history):
                raise ValueError(f"{name} matrix shape mismatch")
            if any(np.shares_memory(item, current) for item in history):
                raise ValueError(f"{name} must contain prior matrices, not current_matrix")
            object.__setattr__(self, name, history)
        mean_ref = np.asarray(self.mean_ref, dtype=np.float64)
        var_ref = np.asarray(self.var_ref, dtype=np.float64)
        if mean_ref.shape != current.shape or var_ref.shape != current.shape:
            raise ValueError("Reference shape mismatch")
        object.__setattr__(self, "mean_ref", mean_ref)
        object.__setattr__(self, "var_ref", var_ref)


def context_from_prior_history(
    history: Sequence[np.ndarray],
    current_matrix: np.ndarray,
    mean_ref: np.ndarray,
    var_ref: np.ndarray,
    *,
    window_size: int,
    min_history: int,
) -> MomentContext:
    """Build context after updating with prior history only, never current X."""
    if len(history) > window_size:
        history = history[-window_size:]
    estimator = SlidingMomentEstimator(
        num_nodes=int(np.asarray(current_matrix).shape[0]),
        window_size=window_size,
        min_history=min_history,
    )
    for matrix in history:
        estimator.update(matrix)
    before = estimator.history_length
    context = estimator.get_context(current_matrix, mean_ref, var_ref)
    if estimator.history_length != before:
        raise AssertionError("get_context mutated estimator history")
    return context


def context_distance(a: MomentContext, b: MomentContext) -> dict[str, float]:
    """Return normalized first/second-moment distances between two contexts."""
    mean = relative_l2_error(a.mean_matrix, b.mean_matrix)
    variance = relative_l2_error(a.var_matrix, b.var_matrix)
    node = relative_l2_error(
        np.concatenate([a.send_mean, a.recv_mean]),
        np.concatenate([b.send_mean, b.recv_mean]),
    )
    return {
        "mean_relative_l2": float(mean),
        "variance_relative_l2": float(variance),
        "node_load_relative_l2": float(node),
        "combined": float(mean + variance + node),
    }


def action_edit_distance(
    lhs: Sequence[tuple[int, ...]], rhs: Sequence[tuple[int, ...]]
) -> int:
    """Levenshtein distance between complete slot/chunk/edge action sequences."""
    previous = list(range(len(rhs) + 1))
    for i, left in enumerate(lhs, start=1):
        current = [i]
        for j, right in enumerate(rhs, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
        previous = current
    return int(previous[-1])


def sparse_schedule(schedule: Sequence[np.ndarray]) -> list[list[int]]:
    """Serialize every scheduled transfer as [slot, chunk, edge]."""
    result: list[list[int]] = []
    for slot, matrix in enumerate(schedule):
        chunks, edges = np.where(np.asarray(matrix) != 0)
        result.extend([[slot, int(chunk), int(edge)] for chunk, edge in zip(chunks, edges)])
    return result


def schedule_edge_use(schedule: Sequence[np.ndarray]) -> np.ndarray:
    if not schedule:
        return np.empty(0, dtype=np.int64)
    return np.stack([np.asarray(item, dtype=np.int64) for item in schedule]).sum(axis=(0, 1))


def edge_use_l1(lhs: Sequence[np.ndarray], rhs: Sequence[np.ndarray]) -> int:
    left = schedule_edge_use(lhs)
    right = schedule_edge_use(rhs)
    if left.shape != right.shape:
        raise ValueError("Schedule edge dimensions differ")
    return int(np.abs(left - right).sum())


def json_ready_context(context: MomentContext) -> dict[str, Any]:
    """Compact audit-only context summary; excludes current/future matrices."""
    return {
        "history_length": int(context.history_length),
        "window_size": int(context.window_size),
        "mean_drift": float(context.mean_drift),
        "var_drift": float(context.var_drift),
        "confidence": float(context.confidence),
        "mean_matrix": np.asarray(context.mean_matrix).tolist(),
        "var_matrix": np.asarray(context.var_matrix).tolist(),
    }
