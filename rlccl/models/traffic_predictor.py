"""Deterministic history-only predictors for low-dimensional traffic summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass
class RidgeMultiOutput:
    """Small dependency-free multi-output ridge regressor with standardization."""

    alpha: float = 1.0
    x_mean: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeMultiOutput":
        features = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
            raise ValueError("x and y must be same-length 2-D arrays")
        if len(features) == 0:
            raise ValueError("Cannot fit an empty dataset")
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0, ddof=0)
        self.x_scale[self.x_scale < 1e-8] = 1.0
        standardized = (features - self.x_mean) / self.x_scale
        self.y_mean = targets.mean(axis=0)
        centered_targets = targets - self.y_mean
        gram = standardized.T @ standardized
        regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(self.alpha)
        self.weights = np.linalg.solve(
            gram + regularizer, standardized.T @ centered_targets
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.x_mean, self.x_scale, self.y_mean, self.weights)):
            raise RuntimeError("Regressor is not fitted")
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.x_mean.shape[0]:
            raise ValueError("Feature dimension mismatch")
        return (features - self.x_mean) / self.x_scale @ self.weights + self.y_mean

    def state(self, prefix: str) -> dict[str, np.ndarray]:
        if any(value is None for value in (self.x_mean, self.x_scale, self.y_mean, self.weights)):
            raise RuntimeError("Regressor is not fitted")
        return {
            f"{prefix}_x_mean": self.x_mean,
            f"{prefix}_x_scale": self.x_scale,
            f"{prefix}_y_mean": self.y_mean,
            f"{prefix}_weights": self.weights,
            f"{prefix}_alpha": np.asarray([self.alpha], dtype=np.float64),
        }

    @classmethod
    def from_state(cls, values: Any, prefix: str) -> "RidgeMultiOutput":
        return cls(
            alpha=float(values[f"{prefix}_alpha"][0]),
            x_mean=np.asarray(values[f"{prefix}_x_mean"], dtype=np.float64),
            x_scale=np.asarray(values[f"{prefix}_x_scale"], dtype=np.float64),
            y_mean=np.asarray(values[f"{prefix}_y_mean"], dtype=np.float64),
            weights=np.asarray(values[f"{prefix}_weights"], dtype=np.float64),
        )


def deterministic_group_coefficients(topology: Any) -> np.ndarray:
    """Map demand entries to shared groups using deterministic shortest paths.

    This is an offered-load proxy, not the realized load of a learned schedule.
    Ties are resolved by the smallest edge index so regeneration is exact.
    """
    num_groups = len(topology.shared_constraints)
    coefficients = np.zeros(
        (num_groups, topology.V, topology.V), dtype=np.float64
    )
    if num_groups == 0:
        return coefficients
    edge_groups: list[list[int]] = [[] for _ in range(topology.E)]
    for group, (edges, _) in enumerate(topology.shared_constraints):
        for edge in edges:
            edge_groups[int(edge)].append(group)
    outgoing: dict[int, list[int]] = {node: [] for node in range(topology.V)}
    for edge, source in enumerate(topology.edge_src):
        outgoing[int(source)].append(edge)
    for source in range(topology.V):
        for destination in range(topology.V):
            if source == destination:
                continue
            node = source
            visited = set()
            while node != destination:
                if node in visited:
                    raise RuntimeError("Shortest-path proxy encountered a cycle")
                visited.add(node)
                candidates = [
                    edge
                    for edge in outgoing[node]
                    if topology.dist_matrix[int(topology.edge_dst[edge]), destination]
                    < topology.dist_matrix[node, destination]
                ]
                if not candidates:
                    raise RuntimeError(f"No path from {source} to {destination}")
                edge = min(candidates)
                for group in edge_groups[edge]:
                    coefficients[group, source, destination] += 1.0
                node = int(topology.edge_dst[edge])
    return coefficients


def traffic_summary(matrix: np.ndarray, group_coefficients: np.ndarray) -> dict[str, Any]:
    traffic = np.asarray(matrix, dtype=np.float64)
    if traffic.ndim != 2 or traffic.shape[0] != traffic.shape[1]:
        raise ValueError("matrix must be square")
    source = traffic.sum(axis=1)
    destination = traffic.sum(axis=0)
    mask = ~np.eye(traffic.shape[0], dtype=bool)
    group_load = (
        np.einsum("gij,ij->g", group_coefficients, traffic)
        if group_coefficients.size
        else np.empty(0, dtype=np.float64)
    )
    return {
        "total": float(traffic.sum()),
        "source_load": source,
        "destination_load": destination,
        "hotspot_destination": int(np.argmax(destination)),
        "hotspot_strength": float(
            destination.max() / max(float(destination.mean()), 1e-8)
        ),
        "sparsity": float(np.mean(traffic[mask] == 0)),
        "bandwidth_group_load": group_load,
    }


def summary_vector(summary: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray([summary["total"]], dtype=np.float64),
            np.asarray(summary["source_load"], dtype=np.float64),
            np.asarray(summary["destination_load"], dtype=np.float64),
            np.asarray(
                [summary["hotspot_strength"], summary["sparsity"]],
                dtype=np.float64,
            ),
            np.asarray(summary["bandwidth_group_load"], dtype=np.float64),
        ]
    )


def _moment_feature(history: np.ndarray) -> np.ndarray:
    mean = history.mean(axis=0)
    variance = history.var(axis=0, ddof=0)
    mask = ~np.eye(mean.shape[0], dtype=bool)
    return np.concatenate([mean[mask], variance[mask]])


def build_history_examples(
    sequences: Sequence[Any],
    *,
    group_coefficients: np.ndarray,
    history_window: int,
    recent_steps: int,
    min_history: int,
) -> list[dict[str, Any]]:
    """Materialize prediction examples using X_0..X_(t-1) only."""
    if min(history_window, recent_steps, min_history) <= 0:
        raise ValueError("History lengths must be positive")
    examples: list[dict[str, Any]] = []
    start = max(recent_steps, min_history)
    for sequence in sequences:
        matrices = np.asarray(sequence.matrices, dtype=np.float64)
        summaries = [traffic_summary(matrix, group_coefficients) for matrix in matrices]
        vectors = np.stack([summary_vector(item) for item in summaries], axis=0)
        hotspots = np.asarray(
            [item["hotspot_destination"] for item in summaries], dtype=np.int64
        )
        for step in range(start, len(matrices)):
            history_start = max(0, step - history_window)
            moment_history = matrices[history_start:step]
            recent_history = vectors[step - recent_steps : step]
            # Explicit audit guards: neither feature slice can include step t.
            if moment_history.shape[0] == 0 or recent_history.shape[0] != recent_steps:
                raise AssertionError("Invalid history-only feature slice")
            examples.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "family": sequence.family,
                    "seed": int(sequence.seed),
                    "step": int(step),
                    "moment_features": _moment_feature(moment_history),
                    "recent_features": recent_history.reshape(-1),
                    "previous_target": vectors[step - 1].copy(),
                    "previous_hotspot": int(hotspots[step - 1]),
                    "target": vectors[step].copy(),
                    "hotspot_target": int(hotspots[step]),
                    "history_last_step": int(step - 1),
                }
            )
    return examples


class TrafficPredictorSuite:
    """Constant, previous-value, moment-only and recent-sequence predictors."""

    def __init__(self, num_nodes: int, group_count: int, alpha: float = 10.0):
        self.num_nodes = int(num_nodes)
        self.group_count = int(group_count)
        self.alpha = float(alpha)
        self.moment = RidgeMultiOutput(alpha)
        self.recent = RidgeMultiOutput(alpha)
        self.moment_hotspot = RidgeMultiOutput(alpha)
        self.recent_hotspot = RidgeMultiOutput(alpha)
        self.constant_target: np.ndarray | None = None
        self.constant_hotspot: int | None = None

    @property
    def target_dim(self) -> int:
        return 1 + 2 * self.num_nodes + 2 + self.group_count

    def fit(self, examples: Sequence[dict[str, Any]]) -> "TrafficPredictorSuite":
        if not examples:
            raise ValueError("examples must not be empty")
        moment = np.stack([item["moment_features"] for item in examples])
        recent = np.stack([item["recent_features"] for item in examples])
        target = np.stack([item["target"] for item in examples])
        labels = np.asarray([item["hotspot_target"] for item in examples], dtype=np.int64)
        if target.shape[1] != self.target_dim:
            raise ValueError("Target dimension does not match topology")
        one_hot = np.eye(self.num_nodes, dtype=np.float64)[labels]
        self.constant_target = target.mean(axis=0)
        counts = np.bincount(labels, minlength=self.num_nodes)
        self.constant_hotspot = int(np.argmax(counts))
        self.moment.fit(moment, target)
        self.recent.fit(recent, target)
        self.moment_hotspot.fit(moment, one_hot)
        self.recent_hotspot.fit(recent, one_hot)
        return self

    def predict(self, examples: Sequence[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
        if self.constant_target is None or self.constant_hotspot is None:
            raise RuntimeError("Predictor suite is not fitted")
        moment = np.stack([item["moment_features"] for item in examples])
        recent = np.stack([item["recent_features"] for item in examples])
        previous = np.stack([item["previous_target"] for item in examples])
        previous_hotspot = np.asarray(
            [item["previous_hotspot"] for item in examples], dtype=np.int64
        )
        target = np.stack([item["target"] for item in examples])
        labels = np.asarray([item["hotspot_target"] for item in examples], dtype=np.int64)
        count = len(examples)
        return {
            "target": {"continuous": target, "hotspot": labels},
            "constant": {
                "continuous": np.repeat(self.constant_target[None, :], count, axis=0),
                "hotspot": np.full(count, self.constant_hotspot, dtype=np.int64),
            },
            "previous": {"continuous": previous, "hotspot": previous_hotspot},
            "moment_only": {
                "continuous": self.moment.predict(moment),
                "hotspot": np.argmax(self.moment_hotspot.predict(moment), axis=1),
            },
            "recent_history": {
                "continuous": self.recent.predict(recent),
                "hotspot": np.argmax(self.recent_hotspot.predict(recent), axis=1),
            },
            "oracle_current_summary": {
                "continuous": target.copy(),
                "hotspot": labels.copy(),
            },
        }

    def save(self, path: str) -> None:
        if self.constant_target is None or self.constant_hotspot is None:
            raise RuntimeError("Predictor suite is not fitted")
        state: dict[str, np.ndarray] = {
            "num_nodes": np.asarray([self.num_nodes], dtype=np.int64),
            "group_count": np.asarray([self.group_count], dtype=np.int64),
            "constant_target": self.constant_target,
            "constant_hotspot": np.asarray([self.constant_hotspot], dtype=np.int64),
        }
        for name in ("moment", "recent", "moment_hotspot", "recent_hotspot"):
            state.update(getattr(self, name).state(name))
        np.savez_compressed(path, **state)

    @classmethod
    def load(cls, path: str) -> "TrafficPredictorSuite":
        with np.load(path) as values:
            result = cls(
                num_nodes=int(values["num_nodes"][0]),
                group_count=int(values["group_count"][0]),
                alpha=float(values["moment_alpha"][0]),
            )
            result.constant_target = np.asarray(values["constant_target"], dtype=np.float64)
            result.constant_hotspot = int(values["constant_hotspot"][0])
            for name in ("moment", "recent", "moment_hotspot", "recent_hotspot"):
                setattr(result, name, RidgeMultiOutput.from_state(values, name))
        return result


def stack_field(examples: Iterable[dict[str, Any]], name: str) -> np.ndarray:
    return np.stack([item[name] for item in examples])
