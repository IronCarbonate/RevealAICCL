"""Finite, leakage-resistant empirical traffic ambiguity sets for Phase 3B.

The ordinary construction path consumes only :class:`AmbiguityConstructionView`.
It has no access to sequence identity, generator metadata, the unrevealed matrix,
or any executable scheduling capability.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np

from .observation import PublicTopologyView, readonly_array


HISTORY_WINDOW = 32
MAX_ENTRY = 8


def _topology_size(topology: Any) -> int:
    return int(getattr(topology, "num_nodes", getattr(topology, "V", 0)))


def _topology_edges(topology: Any) -> np.ndarray:
    return np.asarray(topology.edges, dtype=np.int64)


def _topology_constraints(topology: Any) -> tuple[tuple[tuple[int, ...], float], ...]:
    return tuple(
        (tuple(int(index) for index in indices), float(limit))
        for indices, limit in topology.shared_constraints
    )


def _readonly_float(value: Any) -> np.ndarray:
    return readonly_array(value, dtype=np.float64)


def _validate_finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} dimensionality mismatch")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate_matrix(value: Any, *, size: int | None = None, max_entry: int = MAX_ENTRY) -> np.ndarray:
    raw = _validate_finite_array(value, name="traffic matrix", ndim=2)
    if raw.shape[0] != raw.shape[1]:
        raise ValueError("traffic matrix must be square")
    if size is not None and raw.shape != (size, size):
        raise ValueError("traffic matrix shape mismatch")
    if not np.equal(raw, np.floor(raw)).all():
        raise ValueError("traffic matrix must contain integers")
    matrix = np.asarray(raw, dtype=np.int64)
    if np.any(matrix < 0) or np.any(matrix > max_entry):
        raise ValueError(f"traffic matrix entries must be within [0,{max_entry}]")
    if np.any(np.diag(matrix) != 0):
        raise ValueError("traffic matrix diagonal must be zero")
    return matrix


def _canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": np.ascontiguousarray(value).tobytes().hex(),
        }
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _copy_topology(topology: Any) -> PublicTopologyView:
    if isinstance(topology, PublicTopologyView):
        return PublicTopologyView(
            num_nodes=int(topology.num_nodes),
            num_edges=int(topology.num_edges),
            edges=np.asarray(topology.edges),
            capacities=np.asarray(topology.capacities),
            shared_constraints=topology.shared_constraints,
            name=topology.name,
        )
    return PublicTopologyView.from_topology_info(topology)


def _shortest_path_edges(topology: Any, source: int, destination: int) -> tuple[int, ...]:
    """Return the lexicographically deterministic shortest-hop edge path."""

    size = _topology_size(topology)
    edges = _topology_edges(topology)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(size)]
    for edge_index, (left, right) in enumerate(edges):
        adjacency[int(left)].append((int(right), edge_index))
    for options in adjacency:
        options.sort(key=lambda item: (item[0], item[1]))
    queue: list[int] = [int(source)]
    paths: dict[int, tuple[int, ...]] = {int(source): ()}
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        if node == destination:
            return paths[node]
        for neighbor, edge_index in adjacency[node]:
            candidate = paths[node] + (edge_index,)
            if neighbor not in paths:
                paths[neighbor] = candidate
                queue.append(neighbor)
    raise ValueError(f"topology has no directed path {source}->{destination}")


def _group_coefficients(topology: Any) -> np.ndarray:
    size = _topology_size(topology)
    constraints = _topology_constraints(topology)
    coefficients = np.zeros((len(constraints), size, size), dtype=np.float64)
    memberships = [set(indices) for indices, _ in constraints]
    for source in range(size):
        for destination in range(size):
            if source == destination:
                continue
            path = _shortest_path_edges(topology, source, destination)
            for group_index, group_edges in enumerate(memberships):
                coefficients[group_index, source, destination] = float(
                    sum(edge in group_edges for edge in path)
                )
    return coefficients


def group_coefficients_digest(topology: Any) -> str:
    """Return the canonical digest of deterministic path/group coefficients."""

    return _digest(_group_coefficients(topology))


def descriptor_names(topology: Any) -> tuple[str, ...]:
    size = _topology_size(topology)
    if size < 2:
        raise ValueError("topology must have at least two nodes")
    group_count = len(_topology_constraints(topology))
    return (
        "total_traffic",
        *(f"source_load_{index}" for index in range(size)),
        *(f"destination_load_{index}" for index in range(size)),
        "hotspot_strength",
        "sparsity",
        *(f"bandwidth_group_load_{index}" for index in range(group_count)),
    )


def traffic_descriptor(matrix: Any, topology: Any) -> np.ndarray:
    size = _topology_size(topology)
    values = _validate_matrix(matrix, size=size)
    source_loads = values.sum(axis=1, dtype=np.float64)
    destination_loads = values.sum(axis=0, dtype=np.float64)
    total = float(source_loads.sum())
    mean_destination = float(destination_loads.mean())
    hotspot = 0.0 if mean_destination == 0.0 else float(destination_loads.max() / mean_destination)
    off_diagonal = ~np.eye(size, dtype=bool)
    sparsity = float(np.mean(values[off_diagonal] == 0))
    coefficients = _group_coefficients(topology)
    group_loads = np.einsum("gij,ij->g", coefficients, values, optimize=True)
    return np.asarray(
        [total, *source_loads, *destination_loads, hotspot, sparsity, *group_loads],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class DescriptorNormalizer:
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        center = _readonly_float(self.center)
        scale = _readonly_float(self.scale)
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("normalizer center/scale shape mismatch")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("normalizer must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("normalizer scale must be positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @property
    def digest(self) -> str:
        return _digest((self.center, self.scale))


def fit_descriptor_normalizer(history_matrices: Sequence[Any], topology: Any) -> DescriptorNormalizer:
    matrices = tuple(history_matrices)
    if not matrices:
        raise ValueError("fit history cannot be empty")
    vectors = np.stack([traffic_descriptor(matrix, topology) for matrix in matrices])
    center = vectors.mean(axis=0)
    scale = vectors.std(axis=0, ddof=0)
    scale[scale < 1e-8] = 1.0
    return DescriptorNormalizer(center=center, scale=scale)


def physical_descriptor_bounds(topology: Any, *, max_entry: int = MAX_ENTRY) -> tuple[np.ndarray, np.ndarray]:
    if int(max_entry) <= 0:
        raise ValueError("max_entry must be positive")
    size = _topology_size(topology)
    coefficients = _group_coefficients(topology)
    low = np.zeros(1 + 2 * size + 2 + coefficients.shape[0], dtype=np.float64)
    high = np.asarray(
        [
            size * (size - 1) * max_entry,
            *([((size - 1) * max_entry)] * size),
            *([((size - 1) * max_entry)] * size),
            float(size),
            1.0,
            *(coefficients.sum(axis=(1, 2)) * max_entry),
        ],
        dtype=np.float64,
    )
    return _readonly_float(low), _readonly_float(high)


@dataclass(frozen=True, slots=True)
class AmbiguityConstructionView:
    history_matrices: tuple[np.ndarray, ...]
    history_offsets: tuple[int, ...]
    observed_matrix: np.ndarray
    entry_mask: np.ndarray
    mode: str
    stage: int
    ratio: float
    source_totals: np.ndarray | None
    destination_totals: np.ndarray | None
    topology: PublicTopologyView
    construction_seed: int
    normalizer: DescriptorNormalizer
    history_cutoff: int

    def __post_init__(self) -> None:
        matrices = tuple(self.history_matrices)
        if len(matrices) != HISTORY_WINDOW:
            raise ValueError("history must contain exactly 32 matrices")
        offsets = tuple(int(value) for value in self.history_offsets)
        if offsets != tuple(range(-HISTORY_WINDOW, 0)):
            raise ValueError("history offsets must be strictly increasing -32..-1")
        topology = _copy_topology(self.topology)
        size = int(topology.num_nodes)
        frozen_matrices = tuple(
            readonly_array(_validate_matrix(matrix, size=size), dtype=np.int64)
            for matrix in matrices
        )
        observed = readonly_array(
            _validate_matrix(self.observed_matrix, size=size), dtype=np.int64
        )
        mask = readonly_array(self.entry_mask, dtype=bool)
        if mask.shape != (size, size):
            raise ValueError("entry mask shape mismatch")
        if not np.all(np.diag(mask)):
            raise ValueError("diagonal entries must be exact")
        source_totals = _validated_totals(self.source_totals, size, "source")
        destination_totals = _validated_totals(self.destination_totals, size, "destination")
        ratio = float(self.ratio)
        if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError("ratio must be finite and in [0,1]")
        if not isinstance(self.normalizer, DescriptorNormalizer):
            raise TypeError("normalizer must be a DescriptorNormalizer")
        if self.normalizer.center.shape != (len(descriptor_names(topology)),):
            raise ValueError("normalizer descriptor dimension mismatch")
        _lower_and_capacity(
            observed,
            mask,
            str(self.mode),
            source_totals,
            destination_totals,
        )
        object.__setattr__(self, "history_matrices", frozen_matrices)
        object.__setattr__(self, "history_offsets", offsets)
        object.__setattr__(self, "observed_matrix", observed)
        object.__setattr__(self, "entry_mask", mask)
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "stage", int(self.stage))
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "source_totals", source_totals)
        object.__setattr__(self, "destination_totals", destination_totals)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "construction_seed", int(self.construction_seed))
        cutoff = int(self.history_cutoff)
        if cutoff < HISTORY_WINDOW:
            raise ValueError("history cutoff must admit the complete 32-step window")
        object.__setattr__(self, "history_cutoff", cutoff)
        object.__setattr__(
            self,
            "normalizer",
            DescriptorNormalizer(self.normalizer.center, self.normalizer.scale),
        )

    @classmethod
    def from_observation(
        cls,
        *,
        history_matrices: Sequence[Any],
        history_offsets: Sequence[int],
        observation: Any,
        construction_seed: int,
        normalizer: DescriptorNormalizer,
    ) -> "AmbiguityConstructionView":
        return cls(
            history_matrices=tuple(history_matrices),
            history_offsets=tuple(history_offsets),
            observed_matrix=np.asarray(observation.observed_matrix),
            entry_mask=np.asarray(observation.entry_mask),
            mode=str(observation.mode),
            stage=int(observation.stage),
            ratio=float(observation.ratio),
            source_totals=(
                None if observation.source_totals is None else np.asarray(observation.source_totals)
            ),
            destination_totals=(
                None
                if observation.destination_totals is None
                else np.asarray(observation.destination_totals)
            ),
            topology=_copy_topology(observation.topology),
            construction_seed=int(construction_seed),
            normalizer=normalizer,
            history_cutoff=int(observation.sequence_step),
        )


def _validated_totals(value: Any | None, size: int, label: str) -> np.ndarray | None:
    if value is None:
        return None
    raw = _validate_finite_array(value, name=f"{label} totals", ndim=1)
    if raw.shape != (size,) or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{label} totals shape/integer mismatch")
    totals = np.asarray(raw, dtype=np.int64)
    if np.any(totals < 0):
        raise ValueError(f"{label} totals have negative margin")
    return readonly_array(totals, dtype=np.int64)


def _lower_and_capacity(
    observed: np.ndarray,
    mask: np.ndarray,
    mode: str,
    source_totals: np.ndarray | None,
    destination_totals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    size = observed.shape[0]
    lower = np.zeros((size, size), dtype=np.int64)
    capacity = np.zeros((size, size), dtype=np.int64)
    for source in range(size):
        for destination in range(size):
            if source == destination:
                if observed[source, destination] != 0:
                    raise ValueError("diagonal observation must be zero")
                continue
            if bool(mask[source, destination]):
                lower[source, destination] = int(observed[source, destination])
                capacity[source, destination] = 0
            elif mode == "partial_shards":
                lower[source, destination] = int(observed[source, destination])
                capacity[source, destination] = MAX_ENTRY - lower[source, destination]
            else:
                capacity[source, destination] = MAX_ENTRY
    if np.any(lower < 0) or np.any(lower > MAX_ENTRY) or np.any(capacity < 0):
        raise ValueError("observation lower bound exceeds entry cap 8")
    if source_totals is not None:
        residual = source_totals - lower.sum(axis=1)
        if np.any(residual < 0) or np.any(residual > capacity.sum(axis=1)):
            raise ValueError("source residual margin is infeasible for capacity")
    if destination_totals is not None:
        residual = destination_totals - lower.sum(axis=0)
        if np.any(residual < 0) or np.any(residual > capacity.sum(axis=0)):
            raise ValueError("destination residual margin is infeasible for capacity")
    if source_totals is not None and destination_totals is not None:
        if int(source_totals.sum()) != int(destination_totals.sum()):
            raise ValueError("source/destination residual margins violate conservation")
    return lower, capacity


def _source_only_completion(
    candidate: np.ndarray,
    lower: np.ndarray,
    capacity: np.ndarray,
    source_totals: np.ndarray,
) -> np.ndarray:
    result = lower.copy()
    size = result.shape[0]
    for source in range(size):
        remaining = int(source_totals[source] - result[source].sum())
        while remaining > 0:
            active = [
                destination
                for destination in range(size)
                if destination != source
                and capacity[source, destination] > result[source, destination] - lower[source, destination]
            ]
            if not active:
                raise ValueError("source residual margin is infeasible")
            remaining_capacity = {
                destination: int(
                    capacity[source, destination]
                    - (result[source, destination] - lower[source, destination])
                )
                for destination in active
            }
            preference = {
                destination: max(
                    int(candidate[source, destination] - result[source, destination]), 0
                )
                for destination in active
            }
            weights = preference
            if sum(weights.values()) == 0:
                weights = remaining_capacity
            total_weight = float(sum(weights.values()))
            quotas = {
                destination: remaining * weights[destination] / total_weight
                for destination in active
            }
            allocated = 0
            for destination in active:
                amount = min(
                    remaining_capacity[destination],
                    int(math.floor(quotas[destination])),
                    remaining - allocated,
                )
                if amount > 0:
                    result[source, destination] += amount
                    allocated += amount
                if allocated == remaining:
                    break
            remaining -= allocated
            if remaining == 0:
                continue
            ordered = sorted(
                active,
                key=lambda destination: (
                    -(quotas[destination] - math.floor(quotas[destination])),
                    -preference[destination],
                    destination,
                ),
            )
            extra = 0
            for destination in ordered:
                used = result[source, destination] - lower[source, destination]
                if used < capacity[source, destination] and remaining > 0:
                    result[source, destination] += 1
                    remaining -= 1
                    extra += 1
            if allocated == 0 and extra == 0:
                raise ValueError("source completion made no progress")
    if not np.array_equal(result.sum(axis=1), source_totals):
        raise ValueError("source completion margin mismatch")
    return result


@dataclass(slots=True)
class _FlowEdge:
    left: int
    right: int
    capacity: int
    cost: int
    reverse: int
    index: int


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    all_edges: list[_FlowEdge],
    left: int,
    right: int,
    capacity: int,
    cost: int,
) -> _FlowEdge:
    forward_index = len(all_edges)
    reverse_index = forward_index + 1
    forward = _FlowEdge(left, right, int(capacity), int(cost), reverse_index, forward_index)
    reverse = _FlowEdge(right, left, 0, -int(cost), forward_index, reverse_index)
    graph[left].append(forward)
    graph[right].append(reverse)
    all_edges.extend((forward, reverse))
    return forward


def _shortest_flow_path(
    node_count: int,
    all_edges: Sequence[_FlowEdge],
    source: int,
    sink: int,
) -> tuple[int, ...] | None:
    infinity = 10**18
    distances = [infinity] * node_count
    paths: list[tuple[int, ...] | None] = [None] * node_count
    distances[source] = 0
    paths[source] = ()
    for _ in range(node_count - 1):
        changed = False
        for edge in all_edges:
            if edge.capacity <= 0 or paths[edge.left] is None:
                continue
            candidate_distance = distances[edge.left] + edge.cost
            candidate_path = paths[edge.left] + (edge.index,)
            if (
                candidate_distance < distances[edge.right]
                or (
                    candidate_distance == distances[edge.right]
                    and (paths[edge.right] is None or candidate_path < paths[edge.right])
                )
            ):
                distances[edge.right] = candidate_distance
                paths[edge.right] = candidate_path
                changed = True
        if not changed:
            break
    return paths[sink]


def _transport_completion(
    candidate: np.ndarray,
    lower: np.ndarray,
    capacity: np.ndarray,
    source_totals: np.ndarray,
    destination_totals: np.ndarray,
) -> np.ndarray:
    size = lower.shape[0]
    row_residual = np.asarray(source_totals - lower.sum(axis=1), dtype=np.int64)
    column_residual = np.asarray(destination_totals - lower.sum(axis=0), dtype=np.int64)
    if np.any(row_residual < 0) or np.any(column_residual < 0):
        raise ValueError("negative residual margin is infeasible")
    if int(row_residual.sum()) != int(column_residual.sum()):
        raise ValueError("residual margins violate conservation")

    source_node = 0
    row_start = 1
    column_start = row_start + size
    sink_node = column_start + size
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink_node + 1)]
    all_edges: list[_FlowEdge] = []
    for source in range(size):
        _add_flow_edge(
            graph, all_edges, source_node, row_start + source, int(row_residual[source]), 0
        )
    unit_arcs: dict[tuple[int, int], list[_FlowEdge]] = {}
    for source in range(size):
        for destination in range(size):
            if source == destination:
                continue
            arcs: list[_FlowEdge] = []
            for unit in range(1, int(capacity[source, destination]) + 1):
                new_value = int(lower[source, destination] + unit)
                old_value = new_value - 1
                cost = 1 + abs(new_value - int(candidate[source, destination])) - abs(
                    old_value - int(candidate[source, destination])
                )
                arcs.append(
                    _add_flow_edge(
                        graph,
                        all_edges,
                        row_start + source,
                        column_start + destination,
                        1,
                        int(cost),
                    )
                )
            unit_arcs[(source, destination)] = arcs
    for destination in range(size):
        _add_flow_edge(
            graph,
            all_edges,
            column_start + destination,
            sink_node,
            int(column_residual[destination]),
            0,
        )

    for _ in range(int(row_residual.sum())):
        path = _shortest_flow_path(len(graph), all_edges, source_node, sink_node)
        if path is None:
            raise ValueError("integer min-cost flow has infeasible residual margins")
        for edge_index in path:
            edge = all_edges[edge_index]
            if edge.capacity <= 0:
                raise ValueError("invalid residual flow path")
            edge.capacity -= 1
            all_edges[edge.reverse].capacity += 1

    result = lower.copy()
    for (source, destination), arcs in unit_arcs.items():
        result[source, destination] += sum(edge.capacity == 0 for edge in arcs)
    if not np.array_equal(result.sum(axis=1), source_totals):
        raise ValueError("source min-cost-flow margin mismatch")
    if not np.array_equal(result.sum(axis=0), destination_totals):
        raise ValueError("destination min-cost-flow margin mismatch")
    return result


def _validate_reconciled(matrix: np.ndarray, view: AmbiguityConstructionView) -> None:
    result = _validate_matrix(matrix, size=view.observed_matrix.shape[0])
    exact = np.asarray(view.entry_mask, dtype=bool)
    if not np.array_equal(result[exact], np.asarray(view.observed_matrix)[exact]):
        raise ValueError("reconciled matrix violates exact observation")
    if view.mode == "partial_shards":
        non_exact = ~exact
        if np.any(result[non_exact] < np.asarray(view.observed_matrix)[non_exact]):
            raise ValueError("reconciled matrix violates partial-shard lower bound")
    if view.source_totals is not None and not np.array_equal(
        result.sum(axis=1), view.source_totals
    ):
        raise ValueError("reconciled matrix violates source totals")
    if view.destination_totals is not None and not np.array_equal(
        result.sum(axis=0), view.destination_totals
    ):
        raise ValueError("reconciled matrix violates destination totals")


def reconcile_candidate(candidate: Any, view: AmbiguityConstructionView) -> np.ndarray:
    if not isinstance(view, AmbiguityConstructionView):
        raise TypeError("reconciliation requires AmbiguityConstructionView")
    size = view.observed_matrix.shape[0]
    history_candidate = _validate_matrix(candidate, size=size)
    lower, capacity = _lower_and_capacity(
        np.asarray(view.observed_matrix),
        np.asarray(view.entry_mask),
        view.mode,
        view.source_totals,
        view.destination_totals,
    )
    if view.source_totals is None and view.destination_totals is not None:
        raise ValueError("destination totals without source totals are unsupported/infeasible")
    if view.source_totals is None:
        result = lower.copy()
        for source in range(size):
            for destination in range(size):
                if source == destination or capacity[source, destination] == 0:
                    continue
                result[source, destination] = min(
                    MAX_ENTRY,
                    max(int(history_candidate[source, destination]), int(lower[source, destination])),
                )
    elif view.destination_totals is None:
        result = _source_only_completion(
            history_candidate, lower, capacity, np.asarray(view.source_totals)
        )
    else:
        result = _transport_completion(
            history_candidate,
            lower,
            capacity,
            np.asarray(view.source_totals),
            np.asarray(view.destination_totals),
        )
    _validate_reconciled(result, view)
    return readonly_array(result, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class EmpiricalAmbiguitySet:
    support_matrices: tuple[np.ndarray, ...]
    history_offsets: tuple[int, ...]
    descriptor_vectors: np.ndarray
    descriptor_labels: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    empirical_mean: np.ndarray
    empirical_variance: np.ndarray
    delta_mean: np.ndarray
    variance_low: np.ndarray
    variance_high: np.ndarray
    uniform_witness: np.ndarray
    normalizer: DescriptorNormalizer
    topology: PublicTopologyView
    observation_constraint_fingerprint: str
    normalizer_digest: str
    group_coefficients_digest: str
    history_cutoff: int
    construction_seed: int
    singleton_control: bool
    uses_oracle: bool = False
    upper_bound_only: bool = False

    def __post_init__(self) -> None:
        matrices = tuple(readonly_array(matrix, dtype=np.int64) for matrix in self.support_matrices)
        vectors = _readonly_float(self.descriptor_vectors)
        if not matrices or vectors.shape[0] != len(matrices):
            raise ValueError("ambiguity support cannot be empty")
        object.__setattr__(self, "support_matrices", matrices)
        for name in (
            "lower_bounds",
            "upper_bounds",
            "empirical_mean",
            "empirical_variance",
            "delta_mean",
            "variance_low",
            "variance_high",
            "uniform_witness",
        ):
            object.__setattr__(self, name, _readonly_float(getattr(self, name)))
        object.__setattr__(self, "descriptor_vectors", vectors)
        object.__setattr__(self, "topology", _copy_topology(self.topology))
        if str(self.normalizer_digest) != self.normalizer.digest:
            raise ValueError("normalizer provenance digest mismatch")
        if str(self.group_coefficients_digest) != group_coefficients_digest(self.topology):
            raise ValueError("group coefficient provenance digest mismatch")
        if int(self.history_cutoff) < HISTORY_WINDOW:
            raise ValueError("history cutoff provenance is invalid")
        object.__setattr__(self, "normalizer_digest", str(self.normalizer_digest))
        object.__setattr__(
            self, "group_coefficients_digest", str(self.group_coefficients_digest)
        )
        object.__setattr__(self, "history_cutoff", int(self.history_cutoff))
        object.__setattr__(self, "construction_seed", int(self.construction_seed))

    @property
    def probability_support_size(self) -> int:
        return len(self.support_matrices)

    def validate_probability_weights(self, weights: Any) -> bool:
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (self.probability_support_size,):
            raise ValueError("probability weight shape mismatch")
        if not np.isfinite(values).all():
            raise ValueError("probability weights must be finite")
        if np.any(values < 0.0):
            raise ValueError("probability weights must be nonnegative")
        if abs(float(values.sum()) - 1.0) > 1e-10:
            raise ValueError("probability weights must be normalized")
        weighted_mean = values @ self.descriptor_vectors
        if np.any(np.abs(weighted_mean - self.empirical_mean) > self.delta_mean + 1e-10):
            raise ValueError("probability weights violate ambiguity mean moments")
        centered_square = (self.descriptor_vectors - self.empirical_mean) ** 2
        weighted_variance = values @ centered_square
        if np.any(weighted_variance < self.variance_low - 1e-10) or np.any(
            weighted_variance > self.variance_high + 1e-10
        ):
            raise ValueError("probability weights violate ambiguity variance moments")
        return True

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self)


def build_empirical_ambiguity_set(
    view: AmbiguityConstructionView,
    *,
    calibration_radius: float,
) -> EmpiricalAmbiguitySet:
    if not isinstance(view, AmbiguityConstructionView):
        raise TypeError("ordinary construction accepts only AmbiguityConstructionView")
    radius = float(calibration_radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("calibration radius must be finite and nonnegative")
    if view.ratio == 1.0:
        matrix = readonly_array(view.observed_matrix, dtype=np.int64)
        _validate_reconciled(matrix, view)
        matrices = (matrix,)
        offsets = (-1,)
        singleton = True
    else:
        matrices = tuple(reconcile_candidate(candidate, view) for candidate in view.history_matrices)
        offsets = view.history_offsets
        singleton = False
    vectors = np.stack([traffic_descriptor(matrix, view.topology) for matrix in matrices])
    empirical_mean = vectors.mean(axis=0)
    empirical_variance = ((vectors - empirical_mean) ** 2).mean(axis=0)
    delta_mean = 0.25 * view.normalizer.scale
    variance_low = 0.5 * empirical_variance
    variance_high = 1.5 * empirical_variance + 0.01 * view.normalizer.scale**2
    raw_low = vectors.min(axis=0)
    raw_high = vectors.max(axis=0)
    physical_low, physical_high = physical_descriptor_bounds(view.topology, max_entry=MAX_ENTRY)
    if singleton:
        lower_bounds = vectors[0].copy()
        upper_bounds = vectors[0].copy()
    else:
        lower_bounds = np.maximum(physical_low, raw_low - radius * view.normalizer.scale)
        upper_bounds = np.minimum(physical_high, raw_high + radius * view.normalizer.scale)
    uniform = np.full(len(matrices), 1.0 / len(matrices), dtype=np.float64)
    fingerprint = _digest(
        (
            view.observed_matrix,
            view.entry_mask,
            view.mode,
            view.stage,
            view.ratio,
            view.source_totals,
            view.destination_totals,
            view.normalizer.digest,
            group_coefficients_digest(view.topology),
            view.history_offsets,
            view.history_cutoff,
        )
    )
    result = EmpiricalAmbiguitySet(
        support_matrices=matrices,
        history_offsets=offsets,
        descriptor_vectors=vectors,
        descriptor_labels=descriptor_names(view.topology),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        empirical_mean=empirical_mean,
        empirical_variance=empirical_variance,
        delta_mean=delta_mean,
        variance_low=variance_low,
        variance_high=variance_high,
        uniform_witness=uniform,
        normalizer=view.normalizer,
        topology=view.topology,
        observation_constraint_fingerprint=fingerprint,
        normalizer_digest=view.normalizer.digest,
        group_coefficients_digest=group_coefficients_digest(view.topology),
        history_cutoff=view.history_cutoff,
        construction_seed=view.construction_seed,
        singleton_control=singleton,
    )
    result.validate_probability_weights(uniform)
    return result


def _standardized_distances(vectors: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = np.asarray(vectors, dtype=np.float64) / np.asarray(scale, dtype=np.float64)
    differences = standardized[:, None, :] - standardized[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def _validate_selection_inputs(
    vectors: Any,
    scale: Any,
    history_offsets: Any,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    values = np.asarray(vectors, dtype=np.float64)
    scales = np.asarray(scale, dtype=np.float64)
    offsets = np.asarray(history_offsets, dtype=np.int64)
    requested = int(k)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("selector vectors must be finite and two-dimensional")
    if scales.shape != (values.shape[1],) or np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise ValueError("selector scale must be finite and positive")
    if offsets.shape != (values.shape[0],):
        raise ValueError("selector history offsets shape mismatch")
    if requested <= 0 or requested > values.shape[0]:
        raise ValueError("selector K exceeds finite support")
    return values, scales, offsets, requested


def greedy_minimax_indices(
    vectors: Any,
    *,
    scale: Any,
    history_offsets: Any,
    k: int,
) -> tuple[int, ...]:
    values, scales, offsets, requested = _validate_selection_inputs(
        vectors, scale, history_offsets, k
    )
    distances = _standardized_distances(values, scales)
    radii = distances.max(axis=1)
    minimum = float(radii.min())
    medoid_candidates = np.flatnonzero(np.isclose(radii, minimum, rtol=0.0, atol=1e-12))
    first = int(max(medoid_candidates, key=lambda index: (int(offsets[index]), int(index))))
    selected = [first]
    while len(selected) < requested:
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        farthest = float(np.max(nearest))
        candidates = np.flatnonzero(np.isclose(nearest, farthest, rtol=0.0, atol=1e-12))
        choice = int(max(candidates, key=lambda index: (int(offsets[index]), int(index))))
        selected.append(choice)
    return tuple(selected)


def _farthest_fill(
    values: np.ndarray,
    scales: np.ndarray,
    offsets: np.ndarray,
    selected: list[int],
    requested: int,
) -> None:
    distances = _standardized_distances(values, scales)
    while len(selected) < requested:
        if not selected:
            selected.extend(greedy_minimax_indices(values, scale=scales, history_offsets=offsets, k=1))
            continue
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        farthest = float(np.max(nearest))
        candidates = np.flatnonzero(np.isclose(nearest, farthest, rtol=0.0, atol=1e-12))
        selected.append(
            int(max(candidates, key=lambda index: (int(offsets[index]), int(index))))
        )


def boundary_indices(
    vectors: Any,
    *,
    lower: Any,
    upper: Any,
    scale: Any,
    history_offsets: Any,
    k: int,
) -> tuple[int, ...]:
    values, scales, offsets, requested = _validate_selection_inputs(
        vectors, scale, history_offsets, k
    )
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != scales.shape or high.shape != scales.shape:
        raise ValueError("boundary shape mismatch")
    selected: list[int] = []
    for descriptor_index in range(values.shape[1]):
        for target in (low[descriptor_index], high[descriptor_index]):
            if len(selected) == requested:
                return tuple(selected)
            available = [index for index in range(len(values)) if index not in selected]
            distances = {
                index: abs(values[index, descriptor_index] - target) / scales[descriptor_index]
                for index in available
            }
            minimum = min(distances.values())
            tied = [
                index for index in available
                if abs(distances[index] - minimum) <= 1e-12
            ]
            selected.append(
                int(max(tied, key=lambda index: (int(offsets[index]), int(index))))
            )
    _farthest_fill(values, scales, offsets, selected, requested)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class SelectedSupport:
    method: str
    requested_k: int
    actual_k: int
    selected_indices: tuple[int, ...]
    history_offsets: tuple[int, ...]
    matrices: tuple[np.ndarray, ...]
    descriptor_vectors: np.ndarray
    weights: tuple[float, ...]
    uses_oracle: bool
    upper_bound_only: bool
    severity_definition: str | None = None
    approximation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matrices",
            tuple(readonly_array(matrix, dtype=np.int64) for matrix in self.matrices),
        )
        object.__setattr__(self, "descriptor_vectors", _readonly_float(self.descriptor_vectors))

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self)


def _severity(ambiguity: EmpiricalAmbiguitySet) -> np.ndarray:
    values = ambiguity.descriptor_vectors
    center = ambiguity.normalizer.center
    scale = ambiguity.normalizer.scale
    size = ambiguity.topology.num_nodes
    sparsity_index = 2 * size + 2
    ordinary_indices = [
        0,
        *range(1, 1 + size),
        *range(1 + size, 1 + 2 * size),
        1 + 2 * size,
        *range(sparsity_index + 1, values.shape[1]),
    ]
    components = (values[:, ordinary_indices] - center[ordinary_indices]) / scale[ordinary_indices]
    density = -(values[:, sparsity_index] - center[sparsity_index]) / scale[sparsity_index]
    return np.maximum(components.max(axis=1), density)


def _make_support(
    ambiguity: EmpiricalAmbiguitySet,
    *,
    method: str,
    requested_k: int,
    indices: Sequence[int],
    severity_definition: str | None = None,
    approximation: str | None = None,
) -> SelectedSupport:
    chosen = tuple(int(index) for index in indices)
    actual = len(chosen)
    return SelectedSupport(
        method=method,
        requested_k=int(requested_k),
        actual_k=actual,
        selected_indices=chosen,
        history_offsets=tuple(ambiguity.history_offsets[index] for index in chosen),
        matrices=tuple(ambiguity.support_matrices[index] for index in chosen),
        descriptor_vectors=ambiguity.descriptor_vectors[list(chosen)],
        weights=(1.0 / actual,) * actual,
        uses_oracle=False,
        upper_bound_only=False,
        severity_definition=severity_definition,
        approximation=approximation,
    )


def select_support(
    ambiguity: EmpiricalAmbiguitySet,
    *,
    method: str,
    k: int,
    replicate_seed: int | None = None,
) -> SelectedSupport:
    if not isinstance(ambiguity, EmpiricalAmbiguitySet):
        raise TypeError("selector requires EmpiricalAmbiguitySet")
    requested = int(k)
    if requested <= 0:
        raise ValueError("K must be positive")
    if ambiguity.singleton_control:
        return _make_support(
            ambiguity,
            method=str(method),
            requested_k=requested,
            indices=(0,),
            approximation=(
                "deterministic_greedy_k_center" if method == "minimax_subset" else None
            ),
        )
    if requested > len(ambiguity.support_matrices):
        raise ValueError("K exceeds finite empirical support")
    method = str(method)
    if method == "random_empirical":
        if replicate_seed is None:
            raise ValueError("random_empirical requires explicit replicate_seed")
        indices = tuple(
            int(index)
            for index in np.random.default_rng(int(replicate_seed)).permutation(
                len(ambiguity.support_matrices)
            )[:requested]
        )
        return _make_support(
            ambiguity, method=method, requested_k=requested, indices=indices
        )
    if method == "worst_recent_cases":
        severity = _severity(ambiguity)
        indices = tuple(
            sorted(
                range(len(severity)),
                key=lambda index: (-float(severity[index]), -int(ambiguity.history_offsets[index])),
            )[:requested]
        )
        return _make_support(
            ambiguity,
            method=method,
            requested_k=requested,
            indices=indices,
            severity_definition="max_upper_fit_standardized_with_density",
        )
    if method == "boundary_scenarios":
        indices = boundary_indices(
            ambiguity.descriptor_vectors,
            lower=ambiguity.lower_bounds,
            upper=ambiguity.upper_bounds,
            scale=ambiguity.normalizer.scale,
            history_offsets=ambiguity.history_offsets,
            k=requested,
        )
        return _make_support(
            ambiguity, method=method, requested_k=requested, indices=indices
        )
    if method == "minimax_subset":
        indices = greedy_minimax_indices(
            ambiguity.descriptor_vectors,
            scale=ambiguity.normalizer.scale,
            history_offsets=ambiguity.history_offsets,
            k=requested,
        )
        return _make_support(
            ambiguity,
            method=method,
            requested_k=requested,
            indices=indices,
            approximation="deterministic_greedy_k_center",
        )
    raise ValueError(f"unknown ordinary selector: {method}")


def _rms_distance(left: np.ndarray, right: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((left - right) / scale) ** 2)))


def support_covering_radius(
    ambiguity: EmpiricalAmbiguitySet,
    support: SelectedSupport,
) -> float:
    distances = [
        min(
            _rms_distance(candidate, chosen, ambiguity.normalizer.scale)
            for chosen in support.descriptor_vectors
        )
        for candidate in ambiguity.descriptor_vectors
    ]
    return float(max(distances, default=0.0))


def oracle_support_upper_bound(
    ambiguity: EmpiricalAmbiguitySet,
    *,
    truth: Any,
    k: int,
) -> SelectedSupport:
    requested = int(k)
    if requested <= 0:
        raise ValueError("oracle K must be positive")
    matrix = readonly_array(
        _validate_matrix(truth, size=ambiguity.topology.num_nodes), dtype=np.int64
    )
    truth_vector = traffic_descriptor(matrix, ambiguity.topology)
    candidate_count = (
        0
        if ambiguity.singleton_control
        else min(max(requested - 1, 0), len(ambiguity.support_matrices))
    )
    selected: list[int] = []
    if candidate_count:
        standardized_pool = ambiguity.descriptor_vectors / ambiguity.normalizer.scale
        truth_standardized = truth_vector / ambiguity.normalizer.scale
        while len(selected) < candidate_count:
            available = [
                index for index in range(len(ambiguity.support_matrices)) if index not in selected
            ]
            anchors = [truth_standardized] + [standardized_pool[index] for index in selected]
            nearest = {
                index: min(
                    float(np.linalg.norm(standardized_pool[index] - anchor))
                    for anchor in anchors
                )
                for index in available
            }
            farthest = max(nearest.values())
            tied = [index for index in available if abs(nearest[index] - farthest) <= 1e-12]
            selected.append(
                int(max(tied, key=lambda index: (ambiguity.history_offsets[index], index)))
            )
    matrices = (matrix,) + tuple(ambiguity.support_matrices[index] for index in selected)
    descriptors = np.vstack(
        [truth_vector, *[ambiguity.descriptor_vectors[index] for index in selected]]
    )
    actual = len(matrices)
    return SelectedSupport(
        method="oracle_support_upper_bound",
        requested_k=requested,
        actual_k=actual,
        selected_indices=(-1, *selected),
        history_offsets=(0, *[ambiguity.history_offsets[index] for index in selected]),
        matrices=matrices,
        descriptor_vectors=descriptors,
        weights=(1.0 / actual,) * actual,
        uses_oracle=True,
        upper_bound_only=True,
        approximation="truth_plus_deterministic_greedy_k_center",
    )


def truth_nearest_descriptor_distance(
    ambiguity: EmpiricalAmbiguitySet,
    support: SelectedSupport,
    truth: Any,
) -> float:
    vector = traffic_descriptor(truth, ambiguity.topology)
    return min(
        _rms_distance(vector, candidate, ambiguity.normalizer.scale)
        for candidate in support.descriptor_vectors
    )


__all__ = [
    "AmbiguityConstructionView",
    "DescriptorNormalizer",
    "EmpiricalAmbiguitySet",
    "SelectedSupport",
    "boundary_indices",
    "build_empirical_ambiguity_set",
    "descriptor_names",
    "fit_descriptor_normalizer",
    "group_coefficients_digest",
    "greedy_minimax_indices",
    "oracle_support_upper_bound",
    "physical_descriptor_bounds",
    "reconcile_candidate",
    "select_support",
    "support_covering_radius",
    "traffic_descriptor",
    "truth_nearest_descriptor_distance",
]
