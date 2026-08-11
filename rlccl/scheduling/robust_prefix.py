"""Deterministic public-token robust prefix planning for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from rlccl.uncertainty.observation import PublicTopologyView, readonly_array

SCORE_TIE_TOLERANCE = 1e-12
_LEGAL = {(2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 4),
          (16, 1), (16, 2), (16, 4), (16, 8)}


class UnreachableScenarioError(ValueError):
    pass


class InconsistentResidualError(ValueError):
    pass


class StaleSupportError(ValueError):
    pass


def _digest(value: Any) -> str:
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
                raise ValueError("digest values must be finite")
            return f"f:{number.hex()};".encode("ascii")
        if isinstance(item, str):
            raw = item.encode("utf-8")
            return f"s:{len(raw)}:".encode("ascii") + raw + b";"
        if isinstance(item, np.ndarray):
            return encode({"dtype": item.dtype.str, "shape": list(item.shape),
                           "data": item.reshape(-1).tolist()})
        if hasattr(item, "__dataclass_fields__"):
            return encode({name: getattr(item, name) for name in item.__dataclass_fields__})
        if isinstance(item, dict):
            pairs = sorted(((str(key), val) for key, val in item.items()), key=lambda pair: pair[0].encode("utf-8"))
            return f"m:{len(pairs)}:".encode("ascii") + b"".join(encode(key) + encode(val) for key, val in pairs)
        if isinstance(item, (tuple, list)):
            return f"l:{len(item)}:".encode("ascii") + b"".join(encode(value) for value in item)
        if isinstance(item, np.generic):
            return encode(item.item())
        raise TypeError(f"unsupported digest type: {type(item).__name__}")
    return hashlib.sha256(encode(value)).hexdigest()


def _topology(topology: Any) -> tuple[int, np.ndarray, np.ndarray, tuple[Any, ...]]:
    nodes = int(getattr(topology, "num_nodes", getattr(topology, "V", 0)))
    edges = np.asarray(getattr(topology, "edges"), dtype=np.int64)
    capacities = np.asarray(getattr(topology, "capacities"), dtype=np.float64)
    groups = tuple(getattr(topology, "shared_constraints", ()))
    return nodes, edges, capacities, groups


@dataclass(frozen=True, slots=True)
class PublicRevealedToken:
    local_ordinal: int
    source: int
    destination: int
    holders: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SchedulingObservationView:
    stage: int
    ratio: float
    state_version: int
    observed_matrix: np.ndarray
    entry_mask: np.ndarray
    source_totals: np.ndarray | None
    destination_totals: np.ndarray | None
    revealed_tokens: tuple[PublicRevealedToken, ...]
    topology: PublicTopologyView
    observation_digest: str
    residual_state_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_matrix", readonly_array(self.observed_matrix, dtype=np.int64))
        object.__setattr__(self, "entry_mask", readonly_array(self.entry_mask, dtype=bool))
        object.__setattr__(self, "source_totals", None if self.source_totals is None else readonly_array(self.source_totals, dtype=np.int64))
        object.__setattr__(self, "destination_totals", None if self.destination_totals is None else readonly_array(self.destination_totals, dtype=np.int64))
        object.__setattr__(self, "topology", PublicTopologyView(
            self.topology.num_nodes, self.topology.num_edges, self.topology.edges,
            self.topology.capacities, self.topology.shared_constraints, self.topology.name,
        ))


def build_scheduling_view(observation: Any) -> SchedulingObservationView:
    tokens = tuple(PublicRevealedToken(index, int(token.source), int(token.destination), tuple(token.holders))
                   for index, token in enumerate(observation.revealed_tokens))
    topology = PublicTopologyView(
        observation.topology.num_nodes, observation.topology.num_edges,
        observation.topology.edges, observation.topology.capacities,
        observation.topology.shared_constraints, observation.topology.name,
    )
    demand_key = (int(observation.stage), float(observation.ratio), np.asarray(observation.observed_matrix),
                  np.asarray(observation.entry_mask), observation.source_totals, observation.destination_totals,
                  tuple((t.source, t.destination) for t in tokens), topology)
    residual_key = (demand_key, tuple((t.local_ordinal, t.holders) for t in tokens))
    return SchedulingObservationView(
        int(observation.stage), float(observation.ratio), int(observation.state_version),
        observation.observed_matrix, observation.entry_mask, observation.source_totals,
        observation.destination_totals, tokens, topology, _digest(demand_key), _digest(residual_key),
    )


def atomic_units(raw: float) -> int:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("capacity must be finite and nonnegative")
    return int(math.floor(value))


def _usable(topology: Any, edge: int) -> bool:
    _, _, capacities, groups = _topology(topology)
    if atomic_units(capacities[edge]) < 1:
        return False
    return all(atomic_units(limit) >= 1 for indices, limit in groups if edge in indices)


def canonical_shortest_path(topology: Any, source: int, destination: int) -> tuple[int, ...]:
    nodes, edges, _, _ = _topology(topology)
    source, destination = int(source), int(destination)
    if source == destination:
        return ()
    outgoing: list[list[int]] = [[] for _ in range(nodes)]
    for index, (left, right) in enumerate(edges):
        if _usable(topology, index):
            outgoing[int(left)].append(index)
    for values in outgoing:
        values.sort(key=lambda index: (int(edges[index, 1]), index))
    queue = [source]
    parents: dict[int, tuple[int, int]] = {}
    seen = {source}
    for node in queue:
        for edge in outgoing[node]:
            target = int(edges[edge, 1])
            if target in seen:
                continue
            seen.add(target)
            parents[target] = (node, edge)
            if target == destination:
                path: list[int] = []
                cursor = target
                while cursor != source:
                    cursor, parent_edge = parents[cursor]
                    path.append(parent_edge)
                return tuple(reversed(path))
            queue.append(target)
    raise UnreachableScenarioError(f"unreachable OD {source}->{destination}")


def _distance(topology: Any, source: int, destination: int) -> float:
    try:
        return float(len(canonical_shortest_path(topology, source, destination)))
    except UnreachableScenarioError:
        return math.inf


@dataclass(frozen=True, slots=True)
class CandidateDistances:
    source_distance: float
    edge_target_distance: float
    before_global: float
    after_global: float


def candidate_distances(topology: Any, *, holders: Sequence[int], destination: int, edge_index: int) -> CandidateDistances:
    _, edges, _, _ = _topology(topology)
    source, target = map(int, edges[int(edge_index)])
    before = min(_distance(topology, holder, destination) for holder in holders)
    target_distance = _distance(topology, target, destination)
    return CandidateDistances(_distance(topology, source, destination), target_distance,
                              before, min(before, target_distance))


def is_strict_global_progress(topology: Any, *, holders: Sequence[int], destination: int, edge_index: int) -> bool:
    d = candidate_distances(topology, holders=holders, destination=destination, edge_index=edge_index)
    return math.isfinite(d.before_global) and d.before_global > 0 and d.after_global == d.before_global - 1


@dataclass(frozen=True, slots=True)
class CandidateAction:
    local_token_ordinal: int
    edge_index: int
    before_distance: int
    after_distance: int


def enumerate_candidates(view: SchedulingObservationView) -> tuple[CandidateAction, ...]:
    if view.ratio == 0 or not view.revealed_tokens:
        return ()
    _, edges, _, _ = _topology(view.topology)
    output: list[CandidateAction] = []
    for token in view.revealed_tokens:
        before = min(_distance(view.topology, holder, token.destination) for holder in token.holders)
        if not math.isfinite(before) or before <= 0:
            continue
        for edge, (source, target) in enumerate(edges):
            if int(source) not in token.holders or int(target) in token.holders or not _usable(view.topology, edge):
                continue
            after = min(before, _distance(view.topology, int(target), token.destination))
            if after == before - 1:
                output.append(CandidateAction(token.local_ordinal, edge, int(before), int(after)))
    return tuple(sorted(output, key=lambda item: (item.local_token_ordinal, item.edge_index)))


@dataclass(frozen=True, slots=True)
class BatchLoads:
    edge_units: np.ndarray
    group_units: np.ndarray


def batch_loads(batch: Sequence[CandidateAction], topology: Any) -> BatchLoads:
    _, edges, _, groups = _topology(topology)
    edge = np.zeros(len(edges), dtype=np.int64)
    for item in batch:
        edge[item.edge_index] += 1
    group = np.asarray([sum(edge[index] for index in indices) for indices, _ in groups], dtype=np.int64)
    return BatchLoads(edge, group)


def can_add_candidate(batch: Sequence[CandidateAction], candidate: CandidateAction, topology: Any) -> bool:
    if any(item.local_token_ordinal == candidate.local_token_ordinal for item in batch):
        return False
    _, _, capacities, groups = _topology(topology)
    loads = batch_loads((*batch, candidate), topology)
    if any(loads.edge_units[index] > atomic_units(capacities[index]) for index in range(len(capacities))):
        return False
    return all(loads.group_units[index] <= atomic_units(limit) for index, (_, limit) in enumerate(groups))


def pack_candidate_batch(candidates: Sequence[CandidateAction], topology: Any) -> tuple[CandidateAction, ...]:
    batch: list[CandidateAction] = []
    for candidate in candidates:
        if can_add_candidate(batch, candidate, topology):
            batch.append(candidate)
    return tuple(batch)


@dataclass(frozen=True, slots=True)
class ResidualProjection:
    edge_loads: np.ndarray
    group_loads: np.ndarray
    digest: str


def project_residual_scenario(scenario: Any, view: SchedulingObservationView) -> ResidualProjection:
    matrix = np.asarray(scenario)
    if matrix.shape != view.observed_matrix.shape or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("scenario shape/type mismatch")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise InconsistentResidualError("negative or noninteger scenario")
    matrix = matrix.astype(np.int64)
    nodes, edges, _, groups = _topology(view.topology)
    revealed_counts = np.zeros_like(matrix)
    for token in view.revealed_tokens:
        revealed_counts[token.source, token.destination] += 1
    if np.any(matrix < revealed_counts):
        raise InconsistentResidualError("insufficient scenario load for revealed tokens")
    loads = np.zeros(len(edges), dtype=np.float64)
    for source in range(nodes):
        for destination in range(nodes):
            amount = int(matrix[source, destination])
            if amount <= 0 or source == destination:
                continue
            try:
                path = canonical_shortest_path(view.topology, source, destination)
            except UnreachableScenarioError:
                raise UnreachableScenarioError("positive unreachable scenario demand")
            loads[list(path)] += amount
    for token in view.revealed_tokens:
        try:
            origin = canonical_shortest_path(view.topology, token.source, token.destination)
        except UnreachableScenarioError as exc:
            raise InconsistentResidualError("revealed token unreachable") from exc
        loads[list(origin)] -= 1
        if token.destination not in token.holders:
            holder = min(token.holders, key=lambda node: (_distance(view.topology, node, token.destination), node))
            path = canonical_shortest_path(view.topology, holder, token.destination)
            loads[list(path)] += 1
    if np.any(loads < -1e-12):
        raise InconsistentResidualError("insufficient revealed scenario load")
    loads = np.maximum(loads, 0)
    group = np.asarray([sum(loads[index] for index in indices) for indices, _ in groups], dtype=np.float64)
    loads.setflags(write=False); group.setflags(write=False)
    return ResidualProjection(loads, group, _digest((loads, group)))


def criticality_for_edge(*, residual_edge_load: float, edge_units: int,
                         incident_group_loads: Sequence[float], incident_group_units: Sequence[int]) -> float:
    values = [float(residual_edge_load), *map(float, incident_group_loads)]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("loads must be finite nonnegative")
    result = values[0] / max(int(edge_units), 1)
    result += sum(load / max(int(units), 1) for load, units in zip(values[1:], incident_group_units))
    return float(result)


def candidate_utility(*, criticality: float, after_distance: float) -> float:
    if not math.isfinite(float(criticality)) or not math.isfinite(float(after_distance)):
        raise ValueError("Q inputs must be finite")
    return 1.0 + 0.25 * float(criticality) - 0.05 * float(after_distance)


def robust_score(q: Any, weights: Any, *, risk_lambda: float) -> float:
    values = np.asarray(q, dtype=np.float64)
    raw = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or raw.shape != values.shape or not np.all(np.isfinite(values)):
        raise ValueError("Q must be finite vector")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError("weights must be finite nonnegative with positive sum")
    normalized = raw / raw.sum()
    mean = float(normalized @ values)
    variance = float(normalized @ ((values - mean) ** 2))
    return mean - float(risk_lambda) * math.sqrt(max(variance, 0.0))


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: CandidateAction
    robust_score: float
    scenario_std: float


def score_candidates(candidates: Sequence[CandidateAction], support: Any, *, risk_lambda: float) -> tuple[CandidateScore, ...]:
    projections = [project_residual_scenario(matrix, support.observation) for matrix in support.matrices]
    _, _, capacities, groups = _topology(support.observation.topology)
    output = []
    normalized = np.asarray(support.weights, dtype=float); normalized /= normalized.sum()
    for candidate in candidates:
        incident = [index for index, (indices, _) in enumerate(groups) if candidate.edge_index in indices]
        q = []
        for projection in projections:
            crit = criticality_for_edge(
                residual_edge_load=projection.edge_loads[candidate.edge_index],
                edge_units=atomic_units(capacities[candidate.edge_index]),
                incident_group_loads=tuple(projection.group_loads[index] for index in incident),
                incident_group_units=tuple(atomic_units(groups[index][1]) for index in incident),
            )
            q.append(candidate_utility(criticality=crit, after_distance=candidate.after_distance))
        qv = np.asarray(q); mean = float(normalized @ qv)
        std = float(np.sqrt(normalized @ ((qv - mean) ** 2)))
        output.append(CandidateScore(candidate, robust_score(qv, normalized, risk_lambda=risk_lambda), std))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RobustPrefixConfig:
    horizon: int
    prefix: int
    risk_lambda: float
    requested_k: int

    def __post_init__(self) -> None:
        if (int(self.horizon), int(self.prefix)) not in _LEGAL:
            raise ValueError("nonallowlisted H/P")
        if float(self.risk_lambda) not in (0.0, 0.5, 1.0):
            raise ValueError("risk lambda")
        if int(self.requested_k) not in (1, 8):
            raise ValueError("requested K must be point K1 or robust K8")


@dataclass(frozen=True, slots=True)
class SimulationStep:
    projected_load_digest: str


@dataclass(frozen=True, slots=True)
class PrefixPlan:
    origin_stage: int
    origin_state_version: int
    revision: int
    support_digest: str
    config_digest: str
    batches: tuple[tuple[CandidateAction, ...], ...]
    structural_actions: tuple[tuple[int, int], ...]
    simulation_trace: tuple[SimulationStep, ...]


class RobustPrefixPlanner:
    def __init__(self, config: RobustPrefixConfig) -> None:
        self.config = config

    def plan(self, observation: SchedulingObservationView, support: Any) -> PrefixPlan:
        if support.stage != observation.stage:
            raise StaleSupportError("support stage mismatch")
        if support.observation_digest != observation.observation_digest:
            raise StaleSupportError("support observation digest mismatch")
        projected = tuple(project_residual_scenario(matrix, observation).digest for matrix in support.matrices)
        static_digest = _digest(projected)
        simulated = observation
        planned: list[tuple[CandidateAction, ...]] = []
        trace: list[SimulationStep] = []
        _, edges, _, _ = _topology(observation.topology)
        for _slot in range(self.config.horizon):
            candidates = enumerate_candidates(simulated)
            if not candidates:
                break
            scores = score_candidates(candidates, support, risk_lambda=self.config.risk_lambda)
            ordered = [
                item.candidate for item in sorted(
                    scores,
                    key=lambda item: (-item.robust_score, item.candidate.local_token_ordinal,
                                      item.candidate.edge_index),
                )
            ]
            batch = pack_candidate_batch(ordered, simulated.topology)
            if not batch:
                break
            planned.append(batch)
            trace.append(SimulationStep(static_digest))
            holders = [set(token.holders) for token in simulated.revealed_tokens]
            for action in batch:
                holders[action.local_token_ordinal].add(int(edges[action.edge_index, 1]))
            tokens = tuple(
                replace(token, holders=tuple(sorted(holders[token.local_ordinal])))
                for token in simulated.revealed_tokens
            )
            simulated = replace(simulated, revealed_tokens=tokens,
                                residual_state_digest=_digest(tuple((t.local_ordinal, t.holders) for t in tokens)))
        batches = tuple(planned[: self.config.prefix])
        trace_prefix = tuple(trace[: self.config.prefix])
        structural = tuple((item.local_token_ordinal, item.edge_index) for one in batches for item in one)
        return PrefixPlan(observation.stage, observation.state_version, 0, support.digest,
                          _digest(self.config), batches, structural, trace_prefix)


__all__ = [
    "SchedulingObservationView", "PublicRevealedToken", "CandidateAction", "RobustPrefixConfig",
    "RobustPrefixPlanner", "PrefixPlan", "UnreachableScenarioError", "InconsistentResidualError",
    "StaleSupportError", "atomic_units", "canonical_shortest_path", "enumerate_candidates",
    "pack_candidate_batch", "batch_loads", "project_residual_scenario", "robust_score",
    "criticality_for_edge", "candidate_utility", "score_candidates", "candidate_distances",
    "is_strict_global_progress", "can_add_candidate", "SCORE_TIE_TOLERANCE",
]
