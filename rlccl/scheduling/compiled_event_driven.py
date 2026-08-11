"""Compiled, incremental equivalent of the frozen partial-current scheduler.

The module deliberately keeps the reference scheduler/checker untouched.  A
``StaticPlanCompiler`` pays all graph-search cost once, ``IncrementalState``
accepts reveal/commit deltas into preallocated arrays, ``FastBinder`` reproduces
the reference candidate and packing order, and ``DynamicGuard`` is an
independent fail-closed checker used for R2-C0 differential testing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np

from rlccl.scheduling.robust_prefix import CandidateAction, atomic_units
from rlccl.uncertainty.execution import Proposal, TransferAction
from rlccl.uncertainty.observation import TruthTokenId


_INF = np.iinfo(np.int32).max // 4


@dataclass(frozen=True, slots=True)
class RouteTemplate:
    destination: int
    edge_index: int
    source: int
    target: int
    source_distance: int
    target_distance: int
    group_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StaticProof:
    topology_path_validity: bool
    endpoint_validity: bool
    resource_group_mapping: bool
    template_static_legality: bool
    canonical_path_cases: int
    template_cases: int
    digest: str

    @property
    def valid(self) -> bool:
        return (
            self.topology_path_validity
            and self.endpoint_validity
            and self.resource_group_mapping
            and self.template_static_legality
        )


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    nodes: int
    edges: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    capacity_units: np.ndarray
    group_edges: tuple[tuple[int, ...], ...]
    group_limits: np.ndarray
    edge_to_groups: tuple[tuple[int, ...], ...]
    usable_edges: np.ndarray
    distances: np.ndarray
    canonical_paths: tuple[tuple[tuple[int, ...], ...], ...]
    templates_by_destination: tuple[tuple[RouteTemplate, ...], ...]
    proof: StaticProof
    template_order: str = "token_ordinal_then_edge_index"


class StaticPlanCompiler:
    """Compile every topology-dependent decision; runtime never performs BFS."""

    def __init__(self) -> None:
        self.compile_bfs_sources = 0

    def compile(self, topology: Any) -> CompiledPlan:
        nodes = int(getattr(topology, "num_nodes", getattr(topology, "V", 0)))
        edges = np.array(getattr(topology, "edges"), dtype=np.int64, copy=True)
        capacities = np.array(getattr(topology, "capacities"), dtype=np.float64, copy=True)
        raw_groups = tuple(getattr(topology, "shared_constraints", ()))
        if nodes <= 0 or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("invalid topology shape")
        if capacities.shape != (len(edges),):
            raise ValueError("capacity shape mismatch")
        if np.any(edges < 0) or np.any(edges >= nodes):
            raise ValueError("edge endpoint outside topology")

        capacity_units = np.asarray([atomic_units(value) for value in capacities], dtype=np.int64)
        group_edges: list[tuple[int, ...]] = []
        group_limits: list[int] = []
        edge_to_groups: list[list[int]] = [[] for _ in range(len(edges))]
        for group_index, (indices, limit) in enumerate(raw_groups):
            normalized = tuple(int(edge) for edge in indices)
            if len(set(normalized)) != len(normalized):
                raise ValueError("duplicate edge in bandwidth group")
            if any(edge < 0 or edge >= len(edges) for edge in normalized):
                raise ValueError("bandwidth group edge outside topology")
            group_edges.append(normalized)
            group_limits.append(atomic_units(limit))
            for edge in normalized:
                edge_to_groups[edge].append(group_index)

        group_limits_array = np.asarray(group_limits, dtype=np.int64)
        usable = capacity_units >= 1
        for edge, memberships in enumerate(edge_to_groups):
            if any(group_limits_array[group] < 1 for group in memberships):
                usable[edge] = False

        outgoing: list[list[int]] = [[] for _ in range(nodes)]
        for edge, (source, target) in enumerate(edges):
            if usable[edge]:
                outgoing[int(source)].append(edge)
        for values in outgoing:
            values.sort(key=lambda edge: (int(edges[edge, 1]), int(edge)))

        distances = np.full((nodes, nodes), _INF, dtype=np.int32)
        paths: list[list[tuple[int, ...]]] = [[() for _ in range(nodes)] for _ in range(nodes)]
        for source in range(nodes):
            self.compile_bfs_sources += 1
            distances[source, source] = 0
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
                    distances[source, target] = distances[source, node] + 1
                    queue.append(target)
            for destination in range(nodes):
                if destination == source:
                    paths[source][destination] = ()
                elif distances[source, destination] < _INF:
                    reverse: list[int] = []
                    cursor = destination
                    while cursor != source:
                        cursor, edge = parents[cursor]
                        reverse.append(edge)
                    paths[source][destination] = tuple(reversed(reverse))

        templates: list[tuple[RouteTemplate, ...]] = []
        for destination in range(nodes):
            one: list[RouteTemplate] = []
            for edge, (source, target) in enumerate(edges):
                target_distance = int(distances[int(target), destination])
                if not usable[edge] or target_distance >= _INF:
                    continue
                one.append(RouteTemplate(
                    destination=destination,
                    edge_index=edge,
                    source=int(source),
                    target=int(target),
                    source_distance=int(distances[int(source), destination]),
                    target_distance=target_distance,
                    group_indices=tuple(edge_to_groups[edge]),
                ))
            templates.append(tuple(one))

        canonical_paths = tuple(tuple(row) for row in paths)
        endpoint_validity = all(
            0 <= template.source < nodes and 0 <= template.target < nodes
            for group in templates for template in group
        )
        mapping_validity = all(
            tuple(index for index, indices in enumerate(group_edges) if edge in indices)
            == tuple(edge_to_groups[edge])
            for edge in range(len(edges))
        )
        template_legality = all(
            bool(usable[template.edge_index])
            and tuple(edge_to_groups[template.edge_index]) == template.group_indices
            and int(edges[template.edge_index, 0]) == template.source
            and int(edges[template.edge_index, 1]) == template.target
            and template.target_distance == int(distances[template.target, template.destination])
            for group in templates for template in group
        )
        path_validity = True
        path_cases = 0
        for source in range(nodes):
            for destination in range(nodes):
                path = canonical_paths[source][destination]
                if source == destination:
                    path_validity &= path == ()
                    continue
                if distances[source, destination] >= _INF:
                    path_validity &= path == ()
                    continue
                path_cases += 1
                cursor = source
                for edge in path:
                    path_validity &= bool(usable[edge]) and int(edges[edge, 0]) == cursor
                    cursor = int(edges[edge, 1])
                path_validity &= cursor == destination and len(path) == int(distances[source, destination])

        digest_material = (
            nodes,
            tuple(map(tuple, edges.tolist())),
            tuple(capacity_units.tolist()),
            tuple(group_edges),
            tuple(group_limits),
            tuple(tuple(path for path in row) for row in canonical_paths),
            tuple(tuple((t.destination, t.edge_index, t.source, t.target,
                         t.source_distance, t.target_distance, t.group_indices)
                        for t in group) for group in templates),
        )
        digest = hashlib.sha256(repr(digest_material).encode("utf-8")).hexdigest()
        proof = StaticProof(
            topology_path_validity=bool(path_validity),
            endpoint_validity=bool(endpoint_validity),
            resource_group_mapping=bool(mapping_validity),
            template_static_legality=bool(template_legality),
            canonical_path_cases=path_cases,
            template_cases=sum(len(group) for group in templates),
            digest=digest,
        )
        if not proof.valid:
            raise ValueError("static proof failed")

        for array in (edges, capacity_units, group_limits_array, usable, distances):
            array.setflags(write=False)
        return CompiledPlan(
            nodes=nodes,
            edges=edges,
            edge_sources=edges[:, 0],
            edge_targets=edges[:, 1],
            capacity_units=capacity_units,
            group_edges=tuple(group_edges),
            group_limits=group_limits_array,
            edge_to_groups=tuple(tuple(values) for values in edge_to_groups),
            usable_edges=usable,
            distances=distances,
            canonical_paths=canonical_paths,
            templates_by_destination=tuple(templates),
            proof=proof,
        )


@dataclass(frozen=True, slots=True)
class StateDigest:
    state_version: int
    ready_bitmap: int
    pending_ready_bitmap: int
    revealed_count: int
    residual_demand: tuple[tuple[int, ...], ...]
    holders: tuple[tuple[int, ...], ...]
    committed: tuple[tuple[int, ...], ...]
    link_credits: tuple[int, ...]
    group_credits: tuple[int, ...]


class IncrementalState:
    """Preallocated reveal/commit state updated only by deltas on the fast path."""

    def __init__(
        self,
        plan: CompiledPlan,
        *,
        max_tokens: int,
        max_chunks: int = 8,
        sequence_id: str = "",
        sequence_step: int = 0,
    ) -> None:
        if max_tokens <= 0 or max_chunks <= 0:
            raise ValueError("state capacities must be positive")
        self.plan = plan
        self.max_tokens = int(max_tokens)
        self.max_chunks = int(max_chunks)
        self.sequence_id = str(sequence_id)
        self.sequence_step = int(sequence_step)
        self.stage = 0
        self.ratio = 0.0
        self.state_version = 0
        self.revealed_count = 0
        self.staged_count = 0
        self.ready_bitmap = 0
        self.pending_ready_bitmap = 0
        self.token_ids: list[TruthTokenId | None] = [None] * self.max_tokens
        self.token_ordinals: dict[TruthTokenId, int] = {}
        self.sources = np.full(self.max_tokens, -1, dtype=np.int32)
        self.destinations = np.full(self.max_tokens, -1, dtype=np.int32)
        self.holders = np.zeros((self.max_tokens, plan.nodes), dtype=bool)
        self.ready = np.zeros(self.max_tokens, dtype=bool)
        self.committed_bitmap = np.zeros((self.max_tokens, len(plan.edges)), dtype=bool)
        self.residual_token = np.zeros(self.max_tokens, dtype=bool)
        self.remaining_hops = np.full(self.max_tokens, _INF, dtype=np.int32)
        self.residual_demand = np.zeros((plan.nodes, plan.nodes), dtype=np.int64)
        self.chunk_starts = np.full(self.max_chunks, -1, dtype=np.int32)
        self.chunk_counts = np.zeros(self.max_chunks, dtype=np.int32)
        self.chunk_consumed = np.zeros(self.max_chunks, dtype=bool)
        self.link_credits = np.array(plan.capacity_units, copy=True)
        self.group_credits = np.array(plan.group_limits, copy=True)
        self._scratch_selected_tokens = np.zeros(self.max_tokens, dtype=bool)
        self._scratch_seen_tokens = np.zeros(self.max_tokens, dtype=bool)
        self._scratch_edge_load = np.zeros(len(plan.edges), dtype=np.int64)
        self.delta_update_count = 0
        self.full_rebuild_count = 0

    @classmethod
    def from_observation(
        cls, plan: CompiledPlan, observation: Any, *, max_tokens: int | None = None,
    ) -> "IncrementalState":
        capacity = max(int(max_tokens or len(observation.revealed_tokens) or 1), len(observation.revealed_tokens))
        state = cls(
            plan,
            max_tokens=capacity,
            max_chunks=max(8, int(getattr(observation, "stage", 0)) + 1),
            sequence_id=str(observation.sequence_id),
            sequence_step=int(observation.sequence_step),
        )
        state.state_version = int(observation.state_version)
        state._append_ready_tokens(observation.revealed_tokens)
        state.stage = int(observation.stage)
        state.ratio = float(observation.ratio)
        if observation.revealed_tokens:
            state.ready_bitmap |= 1 << min(state.stage, state.max_chunks - 1)
        return state

    def _write_staged_token(self, ordinal: int, token: Any) -> None:
        if ordinal >= self.max_tokens:
            raise ValueError("token capacity exceeded")
        if not isinstance(token.token_id, TruthTokenId):
            raise TypeError("compiled state requires revealed truth token IDs")
        if token.token_id in self.token_ordinals:
            raise ValueError("duplicate revealed token")
        source, destination = int(token.source), int(token.destination)
        if not (0 <= source < self.plan.nodes and 0 <= destination < self.plan.nodes) or source == destination:
            raise ValueError("invalid token endpoint")
        holders = tuple(int(value) for value in token.holders)
        if not holders or any(value < 0 or value >= self.plan.nodes for value in holders):
            raise ValueError("invalid token holder")
        self.token_ids[ordinal] = token.token_id
        self.token_ordinals[token.token_id] = ordinal
        self.sources[ordinal] = source
        self.destinations[ordinal] = destination
        self.holders[ordinal, :] = False
        self.holders[ordinal, list(holders)] = True

    def stage_ready_chunk(self, chunk: int, tokens: Sequence[Any]) -> None:
        index = int(chunk)
        if index < 0 or index >= self.max_chunks:
            raise ValueError("chunk outside state capacity")
        bit = 1 << index
        if self.pending_ready_bitmap & bit or self.ready_bitmap & bit:
            raise ValueError("chunk replay")
        start = self.staged_count
        values = tuple(tokens)
        if start + len(values) > self.max_tokens:
            raise ValueError("token capacity exceeded")
        for offset, token in enumerate(values):
            self._write_staged_token(start + offset, token)
        self.chunk_starts[index] = start
        self.chunk_counts[index] = len(values)
        self.staged_count += len(values)
        self.pending_ready_bitmap |= bit

    def consume_pending_chunk(self, chunk: int) -> None:
        index = int(chunk)
        bit = 1 << index
        if not self.pending_ready_bitmap & bit:
            raise ValueError("chunk is not pending-ready")
        start, count = int(self.chunk_starts[index]), int(self.chunk_counts[index])
        if start != self.revealed_count:
            raise ValueError("out-of-order ready chunk would reorder token ordinals")
        self._activate_range(start, count)
        self.pending_ready_bitmap &= ~bit
        self.ready_bitmap |= bit
        self.chunk_consumed[index] = True
        self.delta_update_count += 1

    def _activate_range(self, start: int, count: int) -> None:
        for ordinal in range(start, start + count):
            source, destination = int(self.sources[ordinal]), int(self.destinations[ordinal])
            self.ready[ordinal] = True
            residual = not bool(self.holders[ordinal, destination])
            self.residual_token[ordinal] = residual
            if residual:
                self.residual_demand[source, destination] += 1
            holder_indices = np.flatnonzero(self.holders[ordinal])
            distances = self.plan.distances[holder_indices, destination]
            self.remaining_hops[ordinal] = int(distances.min()) if len(distances) else _INF
        self.revealed_count += count

    def _append_ready_tokens(self, tokens: Iterable[Any]) -> None:
        values = tuple(tokens)
        start = self.staged_count
        for offset, token in enumerate(values):
            self._write_staged_token(start + offset, token)
        self.staged_count += len(values)
        self._activate_range(start, len(values))
        if values:
            self.delta_update_count += 1

    def ingest_observation_delta(self, observation: Any, *, chunk: int | None = None) -> None:
        if self.sequence_id and str(observation.sequence_id) != self.sequence_id:
            raise ValueError("observation sequence mismatch")
        if int(observation.sequence_step) != self.sequence_step:
            raise ValueError("observation step mismatch")
        if int(observation.state_version) != self.state_version:
            raise ValueError("stale observation")
        tokens = tuple(observation.revealed_tokens)
        if len(tokens) < self.revealed_count:
            raise ValueError("revealed prefix shrank")
        for ordinal in range(self.revealed_count):
            token = tokens[ordinal]
            if token.token_id != self.token_ids[ordinal]:
                raise ValueError("revealed token order changed")
            observed_holders = np.zeros(self.plan.nodes, dtype=bool)
            observed_holders[list(token.holders)] = True
            if not np.array_equal(observed_holders, self.holders[ordinal]):
                raise ValueError("holder state divergence")
        suffix = tokens[self.revealed_count:]
        if suffix:
            index = int(observation.stage if chunk is None else chunk)
            if index >= self.max_chunks:
                index = self.max_chunks - 1
            self.stage_ready_chunk(index, suffix)
            self.consume_pending_chunk(index)
        self.stage = int(observation.stage)
        self.ratio = float(observation.ratio)

    def reset_slot_credits(self) -> None:
        np.copyto(self.link_credits, self.plan.capacity_units)
        np.copyto(self.group_credits, self.plan.group_limits)
        self._scratch_selected_tokens.fill(False)

    def digest(self) -> StateDigest:
        holders = tuple(
            tuple(int(value) for value in np.flatnonzero(self.holders[index]))
            for index in range(self.revealed_count)
        )
        committed = tuple(
            tuple(int(value) for value in np.flatnonzero(self.committed_bitmap[index]))
            for index in range(self.revealed_count)
        )
        return StateDigest(
            state_version=self.state_version,
            ready_bitmap=self.ready_bitmap,
            pending_ready_bitmap=self.pending_ready_bitmap,
            revealed_count=self.revealed_count,
            residual_demand=tuple(tuple(int(value) for value in row) for row in self.residual_demand),
            holders=holders,
            committed=committed,
            link_credits=tuple(int(value) for value in self.link_credits),
            group_credits=tuple(int(value) for value in self.group_credits),
        )


@dataclass(frozen=True, slots=True)
class BoundBatch:
    state_version: int
    candidates: tuple[CandidateAction, ...]
    selected: tuple[CandidateAction, ...]
    proposal: Proposal


class FastBinder:
    """Template lookup, dynamic filter, deterministic selection, and binding."""

    def __init__(self, plan: CompiledPlan) -> None:
        if not plan.proof.valid:
            raise ValueError("compiled plan has no valid static proof")
        self.plan = plan
        self.runtime_bfs_calls = 0

    def enumerate_candidates(self, state: IncrementalState) -> tuple[CandidateAction, ...]:
        if state.ratio == 0.0 or state.revealed_count == 0:
            return ()
        output: list[CandidateAction] = []
        for ordinal in range(state.revealed_count):
            if not state.ready[ordinal] or not state.residual_token[ordinal]:
                continue
            destination = int(state.destinations[ordinal])
            before = int(state.remaining_hops[ordinal])
            if before <= 0 or before >= _INF:
                continue
            for template in self.plan.templates_by_destination[destination]:
                if (
                    state.holders[ordinal, template.source]
                    and not state.holders[ordinal, template.target]
                    and template.target_distance == before - 1
                ):
                    output.append(CandidateAction(
                        local_token_ordinal=ordinal,
                        edge_index=template.edge_index,
                        before_distance=before,
                        after_distance=before - 1,
                    ))
        return tuple(output)

    def select_batch(
        self, state: IncrementalState, candidates: Sequence[CandidateAction],
    ) -> tuple[CandidateAction, ...]:
        state.reset_slot_credits()
        selected: list[CandidateAction] = []
        for candidate in candidates:
            ordinal, edge = int(candidate.local_token_ordinal), int(candidate.edge_index)
            if ordinal < 0 or ordinal >= state.revealed_count or state._scratch_selected_tokens[ordinal]:
                continue
            if state.link_credits[edge] < 1:
                continue
            groups = self.plan.edge_to_groups[edge]
            if any(state.group_credits[group] < 1 for group in groups):
                continue
            state._scratch_selected_tokens[ordinal] = True
            state.link_credits[edge] -= 1
            for group in groups:
                state.group_credits[group] -= 1
            selected.append(candidate)
        return tuple(selected)

    def bind(
        self, state: IncrementalState, selected: Sequence[CandidateAction],
    ) -> Proposal:
        actions: list[TransferAction] = []
        for candidate in selected:
            ordinal = int(candidate.local_token_ordinal)
            if ordinal < 0 or ordinal >= state.revealed_count or not state.ready[ordinal]:
                raise ValueError("candidate token is hidden or not ready")
            token_id = state.token_ids[ordinal]
            if not isinstance(token_id, TruthTokenId):
                raise ValueError("candidate has no executable truth token")
            actions.append(TransferAction(token_id, int(candidate.edge_index)))
        return Proposal.from_transfers(tuple(actions))

    def step(self, state: IncrementalState) -> BoundBatch:
        candidates = self.enumerate_candidates(state)
        selected = self.select_batch(state, candidates)
        return BoundBatch(state.state_version, candidates, selected, self.bind(state, selected))


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str | None
    applied_actions: int
    state_version: int


class DynamicGuard:
    """Independent fail-closed equivalent of the trusted deterministic checker."""

    def __init__(self, plan: CompiledPlan) -> None:
        self.plan = plan

    def check(
        self,
        state: IncrementalState,
        proposal: Proposal,
        *,
        require_scheduler_semantics: bool = False,
        expected_state_version: int | None = None,
    ) -> GuardDecision:
        try:
            if expected_state_version is not None and int(expected_state_version) != state.state_version:
                raise ValueError("stale_state_version")
            if not isinstance(proposal, Proposal):
                raise TypeError("proposal_type")
            if proposal.scenario_set is not None:
                raise ValueError("scenario_only")
            if proposal.is_wait:
                return GuardDecision(True, None, 0, state.state_version)
            state._scratch_seen_tokens.fill(False)
            state._scratch_edge_load.fill(0)
            for action in proposal.actions:
                ordinal = state.token_ordinals.get(action.token_id)
                if ordinal is None or ordinal >= state.revealed_count or not state.ready[ordinal]:
                    raise ValueError("unrevealed")
                if state._scratch_seen_tokens[ordinal]:
                    raise ValueError("duplicate_in_slot")
                state._scratch_seen_tokens[ordinal] = True
                edge = int(action.edge_index)
                if edge < 0 or edge >= len(self.plan.edges):
                    raise ValueError("edge_range")
                source, target = int(self.plan.edge_sources[edge]), int(self.plan.edge_targets[edge])
                if not state.holders[ordinal, source]:
                    raise ValueError("source_holder")
                if state.holders[ordinal, target] or state.committed_bitmap[ordinal, edge]:
                    raise ValueError("duplicate_commit")
                if require_scheduler_semantics:
                    destination = int(state.destinations[ordinal])
                    before = int(state.remaining_hops[ordinal])
                    target_distance = int(self.plan.distances[target, destination])
                    if not state.residual_token[ordinal] or before <= 0 or target_distance != before - 1:
                        raise ValueError("residual_demand")
                state._scratch_edge_load[edge] += 1
            if np.any(state._scratch_edge_load > self.plan.capacity_units):
                raise ValueError("edge_capacity")
            for group, indices in enumerate(self.plan.group_edges):
                if sum(int(state._scratch_edge_load[edge]) for edge in indices) > int(self.plan.group_limits[group]):
                    raise ValueError("bandwidth_group")
            return GuardDecision(True, None, len(proposal.actions), state.state_version + 1)
        except Exception as error:
            return GuardDecision(False, str(error), 0, state.state_version)

    def apply(
        self,
        state: IncrementalState,
        proposal: Proposal,
        *,
        require_scheduler_semantics: bool = False,
        expected_state_version: int | None = None,
    ) -> GuardDecision:
        decision = self.check(
            state, proposal, require_scheduler_semantics=require_scheduler_semantics,
            expected_state_version=expected_state_version,
        )
        if not decision.accepted or proposal.is_wait:
            return decision
        state.reset_slot_credits()
        for action in proposal.actions:
            ordinal = state.token_ordinals[action.token_id]
            edge = int(action.edge_index)
            target = int(self.plan.edge_targets[edge])
            state.holders[ordinal, target] = True
            state.committed_bitmap[ordinal, edge] = True
            state.link_credits[edge] -= 1
            for group in self.plan.edge_to_groups[edge]:
                state.group_credits[group] -= 1
            destination = int(state.destinations[ordinal])
            if target == destination and state.residual_token[ordinal]:
                source = int(state.sources[ordinal])
                state.residual_demand[source, destination] -= 1
                state.residual_token[ordinal] = False
            holders = np.flatnonzero(state.holders[ordinal])
            distances = self.plan.distances[holders, destination]
            state.remaining_hops[ordinal] = int(distances.min()) if len(distances) else _INF
        state.state_version = decision.state_version
        state.delta_update_count += 1
        return decision


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    bound: BoundBatch
    decision: GuardDecision


class CompiledEventDrivenRuntime:
    """Single-process adapter from EventBridge-ready chunks to compiled commit.

    The native EventBridge owns event polling; this adapter is called only for
    a slot whose atomic ready bit was published.  It does not perform polling,
    IPC, serialization, graph search, or observation rebuilds.
    """

    def __init__(self, plan: CompiledPlan, state: IncrementalState) -> None:
        if state.plan is not plan:
            raise ValueError("runtime state/plan identity mismatch")
        self.plan = plan
        self.state = state
        self.binder = FastBinder(plan)
        self.guard = DynamicGuard(plan)

    def stage_router_chunk(self, chunk: int, revealed_tokens: Sequence[Any]) -> None:
        self.state.stage_ready_chunk(chunk, revealed_tokens)

    def consume_event_ready(self, chunk: int, *, stage: int, ratio: float) -> None:
        self.state.consume_pending_chunk(chunk)
        self.state.stage = int(stage)
        self.state.ratio = float(ratio)

    def schedule_and_commit(self) -> RuntimeStep:
        bound = self.binder.step(self.state)
        decision = self.guard.apply(
            self.state,
            bound.proposal,
            require_scheduler_semantics=True,
            expected_state_version=bound.state_version,
        )
        return RuntimeStep(bound, decision)


def structural_signature(candidates: Sequence[CandidateAction]) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (int(item.local_token_ordinal), int(item.edge_index),
         int(item.before_distance), int(item.after_distance))
        for item in candidates
    )


__all__ = [
    "BoundBatch",
    "CompiledPlan",
    "CompiledEventDrivenRuntime",
    "DynamicGuard",
    "FastBinder",
    "GuardDecision",
    "IncrementalState",
    "RouteTemplate",
    "RuntimeStep",
    "StateDigest",
    "StaticPlanCompiler",
    "StaticProof",
    "structural_signature",
]
