"""Observation builders for Phase C partial-demand experiments."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any

import numpy as np

from ..traffic.matrix_utils import validate_traffic_matrix


OBSERVATION_MODES = (
    "random_entries",
    "source_totals",
    "source_destination_totals",
    "partial_shards",
)


@dataclass(frozen=True)
class PartialDemandObservation:
    mode: str
    hide_ratio: float | None
    observed_matrix: np.ndarray
    observation_demands: np.ndarray
    revealed_chunk_mask: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.observed_matrix, dtype=np.int64)
        demands = np.asarray(self.observation_demands, dtype=np.float32)
        revealed = np.asarray(self.revealed_chunk_mask, dtype=bool)
        validate_traffic_matrix(matrix)
        if demands.ndim != 2 or revealed.shape != (demands.shape[0],):
            raise ValueError("Invalid observation demand/chunk-mask shape")
        if np.any(demands < 0) or not np.all(np.isfinite(demands)):
            raise ValueError("Observation demands must be finite and nonnegative")
        object.__setattr__(self, "observed_matrix", matrix)
        object.__setattr__(self, "observation_demands", demands)
        object.__setattr__(self, "revealed_chunk_mask", revealed)


def _chunk_map(matrix: np.ndarray) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    chunk = 0
    for source in range(matrix.shape[0]):
        for destination in range(matrix.shape[1]):
            for _ in range(int(matrix[source, destination])):
                result.append((chunk, source, destination))
                chunk += 1
    return result


def _row_balanced(source_totals: np.ndarray) -> np.ndarray:
    totals = np.asarray(source_totals, dtype=np.int64)
    num_nodes = len(totals)
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for source, total in enumerate(totals):
        destinations = [node for node in range(num_nodes) if node != source]
        base, remainder = divmod(int(total), len(destinations))
        matrix[source, destinations] = base
        for destination in destinations[:remainder]:
            matrix[source, destination] += 1
    return matrix


def _transport_from_margins(
    source_totals: np.ndarray, destination_totals: np.ndarray
) -> np.ndarray:
    """Integral max-flow using only row/column totals and structural zero diagonal."""
    rows = np.asarray(source_totals, dtype=np.int64)
    columns = np.asarray(destination_totals, dtype=np.int64)
    if rows.sum() != columns.sum() or np.any(rows < 0) or np.any(columns < 0):
        raise ValueError("Invalid transportation margins")
    n = len(rows)
    source_node = 0
    row_offset = 1
    column_offset = 1 + n
    sink = 1 + 2 * n
    size = sink + 1
    capacity = np.zeros((size, size), dtype=np.int64)
    for row in range(n):
        capacity[source_node, row_offset + row] = rows[row]
    total = int(rows.sum())
    for row in range(n):
        for column in range(n):
            if row != column:
                capacity[row_offset + row, column_offset + column] = total
    for column in range(n):
        capacity[column_offset + column, sink] = columns[column]

    residual = capacity.copy()
    flow = np.zeros_like(capacity)
    delivered = 0
    while True:
        parent = np.full(size, -1, dtype=np.int64)
        parent[source_node] = source_node
        queue = deque([source_node])
        while queue and parent[sink] < 0:
            node = queue.popleft()
            for neighbor in np.flatnonzero(residual[node] > 0):
                neighbor = int(neighbor)
                if parent[neighbor] < 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if parent[sink] < 0:
            break
        amount = total
        node = sink
        while node != source_node:
            previous = int(parent[node])
            amount = min(amount, int(residual[previous, node]))
            node = previous
        node = sink
        while node != source_node:
            previous = int(parent[node])
            residual[previous, node] -= amount
            residual[node, previous] += amount
            flow[previous, node] += amount
            flow[node, previous] -= amount
            node = previous
        delivered += amount
    if delivered != total:
        raise RuntimeError("Traffic margins are infeasible under zero diagonal")
    matrix = flow[row_offset : row_offset + n, column_offset : column_offset + n]
    if not np.array_equal(matrix.sum(axis=1), rows) or not np.array_equal(
        matrix.sum(axis=0), columns
    ):
        raise AssertionError("Transportation flow does not preserve margins")
    return matrix.astype(np.int64)


def _demands_for_proxy(
    true_matrix: np.ndarray, proxy_matrix: np.ndarray, num_chunks: int
) -> np.ndarray:
    """Assign proxy destinations to existing source-owned chunk identities."""
    n = true_matrix.shape[0]
    demands = np.zeros((num_chunks, n), dtype=np.float32)
    chunks_by_source: dict[int, list[int]] = {source: [] for source in range(n)}
    for chunk, source, _ in _chunk_map(true_matrix):
        chunks_by_source[source].append(chunk)
    for source in range(n):
        destinations = []
        for destination in range(n):
            destinations.extend([destination] * int(proxy_matrix[source, destination]))
        if len(destinations) != len(chunks_by_source[source]):
            raise AssertionError("Proxy row total differs from available source chunks")
        for chunk, destination in zip(chunks_by_source[source], destinations):
            demands[chunk, destination] = 1.0
    return demands


def build_partial_observation(
    matrix: np.ndarray,
    true_demands: np.ndarray,
    *,
    mode: str,
    hide_ratio: float | None,
    seed: int,
) -> PartialDemandObservation:
    """Build a policy observation without changing ground-truth execution demand."""
    traffic = np.asarray(matrix, dtype=np.int64)
    demands = np.asarray(true_demands, dtype=np.int64)
    validate_traffic_matrix(traffic)
    chunks = _chunk_map(traffic)
    if len(chunks) != demands.shape[0] or demands.shape[1] != traffic.shape[0]:
        raise ValueError("true_demands do not match the traffic matrix chunk encoding")
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"Unsupported observation mode: {mode}")
    if mode in {"random_entries", "partial_shards"}:
        if hide_ratio is None or not 0.0 <= hide_ratio < 1.0:
            raise ValueError("random_entries/partial_shards require hide_ratio in [0,1)")
    rng = np.random.default_rng(seed)
    observed = np.zeros_like(traffic)
    observation_demands = np.zeros_like(demands, dtype=np.float32)
    revealed = np.zeros(len(chunks), dtype=bool)

    if mode == "random_entries":
        entries = [(i, j) for i in range(traffic.shape[0]) for j in range(traffic.shape[1]) if i != j]
        hidden_count = int(round(float(hide_ratio) * len(entries)))
        hidden_indices = set(
            int(value)
            for value in rng.choice(len(entries), size=hidden_count, replace=False)
        )
        visible_entries = {
            entry for index, entry in enumerate(entries) if index not in hidden_indices
        }
        for chunk, source, destination in chunks:
            if (source, destination) in visible_entries:
                observed[source, destination] += 1
                observation_demands[chunk] = demands[chunk]
                revealed[chunk] = True
        metadata = {
            "visible_entry_count": len(visible_entries),
            "total_off_diagonal_entries": len(entries),
            "representation_note": (
                "Chunk identities/source ownership remain known to preserve the action space; "
                "hidden destination entries are zero in policy demand features."
            ),
        }
    elif mode == "partial_shards":
        visible_count = int(round((1.0 - float(hide_ratio)) * len(chunks)))
        visible_chunks = set(
            int(value)
            for value in rng.choice(len(chunks), size=visible_count, replace=False)
        )
        for chunk, source, destination in chunks:
            if chunk in visible_chunks:
                observed[source, destination] += 1
                observation_demands[chunk] = demands[chunk]
                revealed[chunk] = True
        metadata = {
            "visible_chunk_count": visible_count,
            "total_chunk_count": len(chunks),
        }
    elif mode == "source_totals":
        observed = _row_balanced(traffic.sum(axis=1))
        observation_demands = _demands_for_proxy(traffic, observed, len(chunks))
        revealed[:] = False
        metadata = {
            "imputation": "deterministic balanced off-diagonal allocation preserving source totals",
            "provided": "source totals only",
        }
        hide_ratio = None
    else:
        observed = _transport_from_margins(traffic.sum(axis=1), traffic.sum(axis=0))
        observation_demands = _demands_for_proxy(traffic, observed, len(chunks))
        revealed[:] = False
        metadata = {
            "imputation": (
                "deterministic integral max-flow using only source/destination totals "
                "and the zero-diagonal structural constraint"
            ),
            "provided": "source and destination totals",
        }
        hide_ratio = None

    validate_traffic_matrix(observed)
    return PartialDemandObservation(
        mode=mode,
        hide_ratio=hide_ratio,
        observed_matrix=observed,
        observation_demands=observation_demands,
        revealed_chunk_mask=revealed,
        metadata=metadata,
    )
