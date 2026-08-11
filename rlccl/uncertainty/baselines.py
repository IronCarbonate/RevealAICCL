"""Leakage-safe reference baselines for the partial observation API."""

from __future__ import annotations

from collections import deque

import numpy as np

from .execution import Proposal, TransferAction
from .observation import PartialObservationState, SanitizedHistoryView
from .scenarios import ScenarioSet


def _distance_to_destination(
    num_nodes: int, edges: np.ndarray, destination: int
) -> np.ndarray:
    """Directed unweighted distance, computed only from public topology."""
    incoming: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        incoming[int(target)].append(int(source))
    distance = np.full(num_nodes, num_nodes + 1, dtype=np.int64)
    distance[int(destination)] = 0
    queue: deque[int] = deque((int(destination),))
    while queue:
        target = queue.popleft()
        for source in sorted(incoming[target]):
            if distance[source] > distance[target] + 1:
                distance[source] = distance[target] + 1
                queue.append(source)
    return distance


def _direct_revealed_proposal(observation: PartialObservationState) -> Proposal:
    """Choose a deterministic, single-slot legal subset from public state.

    Capacity and shared-group budgets are reserved while scanning revealed
    tokens in their stable public order.  Each token receives at most one hop,
    so a value received in this proposal can never be forwarded in the same
    slot.  Routes are shortest public-topology next hops, not private paths.
    """
    edges = np.asarray(observation.topology.edges, dtype=np.int64)
    capacities = np.asarray(observation.topology.capacities, dtype=np.float64)
    edge_load = np.zeros(observation.topology.num_edges, dtype=np.float64)
    group_load = np.zeros(len(observation.topology.shared_constraints), dtype=np.float64)
    groups_by_edge: list[list[tuple[int, float]]] = [
        [] for _ in range(observation.topology.num_edges)
    ]
    for group_index, (edge_indices, limit) in enumerate(
        observation.topology.shared_constraints
    ):
        for edge_index in edge_indices:
            groups_by_edge[int(edge_index)].append((group_index, float(limit)))

    distances: dict[int, np.ndarray] = {}
    actions: list[TransferAction] = []
    for token in observation.revealed_tokens:
        if token.destination in token.holders:
            continue
        distance = distances.setdefault(
            token.destination,
            _distance_to_destination(
                observation.topology.num_nodes, edges, token.destination
            ),
        )
        holders = set(token.holders)
        candidates: list[tuple[int, int]] = []
        for edge_index, (source, target) in enumerate(edges):
            source_i, target_i = int(source), int(target)
            if source_i not in holders or target_i in holders:
                continue
            if distance[source_i] == distance[target_i] + 1:
                candidates.append((int(distance[target_i]), edge_index))
        candidates.sort()
        for _, edge_index in candidates:
            if edge_load[edge_index] + 1.0 > capacities[edge_index] + 1e-12:
                continue
            if any(
                group_load[group_index] + 1.0 > limit + 1e-12
                for group_index, limit in groups_by_edge[edge_index]
            ):
                continue
            actions.append(TransferAction(token.token_id, edge_index))
            edge_load[edge_index] += 1.0
            for group_index, _ in groups_by_edge[edge_index]:
                group_load[group_index] += 1.0
            break
    return Proposal.from_transfers(tuple(actions)) if actions else Proposal.wait()


def _validate_history(
    observation: PartialObservationState,
    history: SanitizedHistoryView | None,
) -> SanitizedHistoryView | None:
    if history is None:
        return None
    if history.sequence_id != observation.sequence_id:
        raise ValueError("History and observation must belong to the same sequence")
    if history.current_step != observation.sequence_step:
        raise ValueError("History boundary must match the current sequence step")
    expected_shape = observation.observed_matrix.shape
    if any(matrix.shape != expected_shape for matrix in history.matrices):
        raise ValueError("History matrix shape must match the current topology")
    return history


def _history_scenario(
    observation: PartialObservationState,
    history: SanitizedHistoryView | None,
    *,
    method: str,
) -> Proposal:
    checked = _validate_history(observation, history)
    if checked is None or not checked.matrices:
        return Proposal.wait()
    if method == "long_term_mean":
        matrix = np.rint(np.mean(np.stack(checked.matrices), axis=0)).astype(np.int64)
        scenario_id = "long-term-mean"
    elif method == "previous_value":
        matrix = np.array(checked.matrices[-1], dtype=np.int64, copy=True)
        scenario_id = "previous-value"
    else:
        raise ValueError(f"Unsupported history scenario method: {method}")
    return Proposal.scenario_only(
        ScenarioSet.from_matrices(
            matrices=(matrix,),
            weights=(1.0,),
            scenario_ids=(scenario_id,),
            provenance=("same-sequence-X-before-t",),
        )
    )


class WaitUntilKnownBaseline:
    method = "wait_until_known"

    def propose(
        self,
        observation: PartialObservationState,
        *,
        history: SanitizedHistoryView | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Proposal:
        _validate_history(observation, history)
        del scenarios
        if observation.ratio < 1.0:
            return Proposal.wait()
        return _direct_revealed_proposal(observation)


class PartialCurrentOnlyBaseline:
    method = "partial_current_only"

    def propose(
        self,
        observation: PartialObservationState,
        *,
        history: SanitizedHistoryView | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Proposal:
        _validate_history(observation, history)
        del scenarios
        return _direct_revealed_proposal(observation)


class LongTermMeanBaseline:
    method = "long_term_mean"

    def propose(
        self,
        observation: PartialObservationState,
        *,
        history: SanitizedHistoryView | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Proposal:
        del scenarios
        _validate_history(observation, history)
        if observation.revealed_tokens:
            return _direct_revealed_proposal(observation)
        return _history_scenario(
            observation, history, method="long_term_mean"
        )


class PreviousValueBaseline:
    method = "previous_value"

    def propose(
        self,
        observation: PartialObservationState,
        *,
        history: SanitizedHistoryView | None = None,
        scenarios: ScenarioSet | None = None,
    ) -> Proposal:
        del scenarios
        _validate_history(observation, history)
        if observation.revealed_tokens:
            return _direct_revealed_proposal(observation)
        return _history_scenario(observation, history, method="previous_value")


class FullInformationOracle:
    """Evaluator-only lower-bound reference; never produces executable actions."""

    method = "full_information_oracle"
    uses_oracle = True
    upper_bound_only = True


__all__ = [
    "FullInformationOracle",
    "LongTermMeanBaseline",
    "PartialCurrentOnlyBaseline",
    "PreviousValueBaseline",
    "WaitUntilKnownBaseline",
]
