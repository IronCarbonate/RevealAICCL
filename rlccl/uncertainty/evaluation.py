"""Paired evaluation rebuilt independently from an immutable manifest."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from ..envs.problem import TopologyInfo
from ..traffic.matrix_utils import validate_traffic_matrix
from .baselines import (
    FullInformationOracle,
    LongTermMeanBaseline,
    PartialCurrentOnlyBaseline,
    PreviousValueBaseline,
    WaitUntilKnownBaseline,
)
from .execution import Proposal
from .metrics import RecourseMetrics
from .observation import PartialObservationState, SanitizedHistoryView
from .problem import UncertainProblemInstance
from .reveal import DemandRevealProcess


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """Immutable paired-evaluation identity and canonical input digests.

    ``timeout`` is common outer invocation-budget provenance only; this pure
    NumPy runner does not claim wall-clock enforcement.  The per-method
    ``timeout`` raw-row boolean is instead determined solely by whether the
    method completes within ``time_limit`` discrete execution slots.
    """

    manifest_id: str
    sequence_id: str
    family: str
    history_provenance: str
    truth_digest: str
    topology_digest: str
    config_digest: str
    reveal_mode: str
    ratios: tuple[float, ...]
    reveal_seed: int
    timeout: int
    time_limit: int
    checker_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratios", tuple(float(value) for value in self.ratios))
        if self.timeout <= 0:
            raise ValueError("timeout limit must be positive")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")

    @classmethod
    def create(
        cls,
        *,
        manifest_id: str,
        sequence_id: str,
        family: str,
        history_provenance: str,
        truth_matrix: np.ndarray,
        topology_info: TopologyInfo,
        reveal_mode: str,
        ratios: Sequence[float],
        reveal_seed: int,
        timeout: int,
        time_limit: int,
        checker_version: str,
    ) -> "EvaluationManifest":
        """Build all three digests from the actual evaluation inputs."""
        raw_truth = np.asarray(truth_matrix)
        validate_traffic_matrix(raw_truth)
        truth = np.array(raw_truth, dtype=np.int64, copy=True)
        normalized_ratios = tuple(float(value) for value in ratios)
        return cls(
            manifest_id=str(manifest_id),
            sequence_id=str(sequence_id),
            family=str(family),
            history_provenance=str(history_provenance),
            truth_digest=_truth_digest(truth),
            topology_digest=_topology_digest(topology_info),
            config_digest=_config_digest(
                reveal_mode=reveal_mode,
                ratios=normalized_ratios,
                reveal_seed=reveal_seed,
                timeout=timeout,
                time_limit=time_limit,
                checker_version=checker_version,
            ),
            reveal_mode=str(reveal_mode),
            ratios=normalized_ratios,
            reveal_seed=int(reveal_seed),
            timeout=int(timeout),
            time_limit=int(time_limit),
            checker_version=str(checker_version),
        )


@dataclass(slots=True)
class _Episode:
    world: UncertainProblemInstance
    reveal_process: DemandRevealProcess
    manifest: EvaluationManifest
    _slot: int = 0

    def next_observation(self) -> PartialObservationState:
        observation = self.reveal_process.observation_for_slot(self._slot)
        self._slot += 1
        return observation

    def next_full_observation(self) -> PartialObservationState:
        """Oracle-only: full demand is available before execution slot zero."""
        observation = self.reveal_process.full_observation()
        self._slot += 1
        return observation


def _truth_digest(matrix: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()


def _canonical_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _topology_digest(topology: TopologyInfo) -> str:
    """Versioned canonical topology digest required by §11.9."""
    edges = np.asarray(topology.edges)
    capacities = np.asarray(topology.capacities, dtype=np.float64)
    payload = {
        "version": "uncertainty-topology-v1",
        "V": int(topology.V),
        "E": int(topology.E),
        "edges": [[int(source), int(destination)] for source, destination in edges],
        "capacities_float_hex": [float(value).hex() for value in capacities],
        "shared_groups": [
            {
                "edge_indices": [int(edge) for edge in edge_indices],
                "limit_float_hex": float(limit).hex(),
            }
            for edge_indices, limit in topology.shared_constraints
        ],
    }
    return _canonical_json_digest(payload)


def _config_digest(
    *,
    reveal_mode: str,
    ratios: Sequence[float],
    reveal_seed: int,
    timeout: int,
    time_limit: int,
    checker_version: str,
) -> str:
    """Versioned canonical reveal/evaluation configuration digest."""
    payload = {
        "version": "uncertainty-config-v1",
        "reveal_mode": str(reveal_mode),
        "normalized_ratios_float_hex": [float(value).hex() for value in ratios],
        "reveal_seed": int(reveal_seed),
        "timeout": int(timeout),
        "time_limit": int(time_limit),
        "checker_version": str(checker_version),
    }
    return _canonical_json_digest(payload)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _oracle_completion_lower_bound(
    truth: np.ndarray, topology: TopologyInfo, time_limit: int
) -> int:
    """Compute the four-term provable completion lower bound from §11.8.

    Capacities are floored to atomic unit actions.  Directed shortest paths use
    only positive-unit-capacity edges.  Shared groups are deliberately ignored
    as an optimistic relaxation, never as additional capacity.
    """
    positive_pairs = np.argwhere(np.asarray(truth) > 0)
    if len(positive_pairs) == 0:
        return 0

    nodes = int(topology.V)
    edges = np.asarray(topology.edges, dtype=np.int64)
    unit_capacity = np.asarray(
        [max(0, math.floor(float(value))) for value in topology.capacities],
        dtype=np.int64,
    )
    total_unit_capacity = int(unit_capacity.sum())
    if total_unit_capacity <= 0:
        return int(time_limit) + 1

    distance = np.full((nodes, nodes), nodes + 1, dtype=np.int64)
    np.fill_diagonal(distance, 0)
    outgoing_capacity = np.zeros(nodes, dtype=np.int64)
    incoming_capacity = np.zeros(nodes, dtype=np.int64)
    for edge_index, (source, destination) in enumerate(edges):
        capacity = int(unit_capacity[edge_index])
        if capacity <= 0:
            continue
        source_i, destination_i = int(source), int(destination)
        distance[source_i, destination_i] = 1
        outgoing_capacity[source_i] += capacity
        incoming_capacity[destination_i] += capacity
    for intermediate in range(nodes):
        distance = np.minimum(
            distance,
            distance[:, [intermediate]] + distance[[intermediate], :],
        )

    lb_path = 0
    total_work = 0
    for source, destination in positive_pairs:
        source_i, destination_i = int(source), int(destination)
        hops = int(distance[source_i, destination_i])
        if (
            hops > nodes
            or outgoing_capacity[source_i] <= 0
            or incoming_capacity[destination_i] <= 0
        ):
            return int(time_limit) + 1
        count = int(truth[source_i, destination_i])
        lb_path = max(lb_path, hops)
        total_work += count * hops

    lb_work = _ceil_div(total_work, total_unit_capacity)
    lb_source = max(
        _ceil_div(int(truth[source].sum()), int(outgoing_capacity[source]))
        for source in range(nodes)
        if int(truth[source].sum()) > 0
    )
    lb_destination = max(
        _ceil_div(int(truth[:, destination].sum()), int(incoming_capacity[destination]))
        for destination in range(nodes)
        if int(truth[:, destination].sum()) > 0
    )
    lower_bound = max(lb_path, lb_work, lb_source, lb_destination)
    return int(time_limit) + 1 if lower_bound > int(time_limit) else int(lower_bound)


def _world_complete(world: UncertainProblemInstance) -> bool:
    return all(
        bool(world._possession[token_index, destination])
        for token_index, (_, destination, _) in enumerate(world._atomic)
    )


def _proposal_signature(proposal: Proposal) -> tuple[Any, ...]:
    """Stable public-only signature used by the recourse definition."""
    if proposal.scenario_set is not None:
        scenarios = proposal.scenario_set
        return (
            "scenario",
            tuple(
                (str(matrix.dtype), tuple(matrix.shape), matrix.tobytes())
                for matrix in scenarios.matrices
            ),
            scenarios.weights,
            scenarios.scenario_ids,
            scenarios.provenance,
        )
    if proposal.actions:
        return (
            "actions",
            tuple((str(action.token_id), action.edge_index) for action in proposal.actions),
        )
    return ("wait",)


class PairedEvaluationRunner:
    """Trusted evaluator; every method receives a fresh world/reveal process."""

    def __init__(
        self,
        *,
        manifest: EvaluationManifest,
        truth_matrix: np.ndarray,
        history_matrices: Sequence[np.ndarray],
        topology_info: TopologyInfo,
        generator_metadata: Mapping[str, Any],
    ) -> None:
        if not isinstance(manifest, EvaluationManifest):
            raise TypeError("manifest must be EvaluationManifest")
        raw_truth = np.asarray(truth_matrix)
        validate_traffic_matrix(raw_truth)
        if raw_truth.shape != (int(topology_info.V), int(topology_info.V)):
            raise ValueError("Truth matrix shape must match topology")
        truth = np.array(raw_truth, dtype=np.int64, copy=True)
        if _truth_digest(truth) != manifest.truth_digest:
            raise ValueError("Manifest truth digest does not match input truth")
        if _topology_digest(topology_info) != manifest.topology_digest:
            raise ValueError("Manifest topology digest does not match input topology")
        if _config_digest(
            reveal_mode=manifest.reveal_mode,
            ratios=manifest.ratios,
            reveal_seed=manifest.reveal_seed,
            timeout=manifest.timeout,
            time_limit=manifest.time_limit,
            checker_version=manifest.checker_version,
        ) != manifest.config_digest:
            raise ValueError("Manifest config digest does not match evaluation config")

        history: list[np.ndarray] = []
        for matrix in history_matrices:
            raw_matrix = np.asarray(matrix)
            validate_traffic_matrix(raw_matrix)
            if raw_matrix.shape != truth.shape:
                raise ValueError("Each history matrix shape must match truth/topology")
            history.append(np.array(raw_matrix, dtype=np.int64, copy=True))

        self.manifest = manifest
        self._truth = truth
        self._history = tuple(history)
        self._topology = TopologyInfo(
            int(topology_info.V),
            int(topology_info.E),
            np.array(topology_info.edges, dtype=np.int64, copy=True),
            np.array(topology_info.capacities, dtype=np.float64, copy=True),
            [
                (list(int(edge) for edge in edge_indices), float(limit))
                for edge_indices, limit in topology_info.shared_constraints
            ],
            name=getattr(topology_info, "name", None),
        )
        self._metadata = deepcopy(dict(generator_metadata))

    def build_episode(self, method: str) -> _Episode:
        if method not in {
            "full_information_oracle",
            "wait_until_known",
            "partial_current_only",
            "long_term_mean",
            "previous_value",
        }:
            raise ValueError(f"Unknown evaluation method: {method}")
        world = UncertainProblemInstance.from_traffic_matrix(
            truth_matrix=self._truth,
            topology_info=self._topology,
            time_limit=self.manifest.time_limit,
            sequence_id=self.manifest.sequence_id,
            sequence_step=len(self._history),
            family=self.manifest.family,
            generator_metadata=deepcopy(self._metadata),
        )
        reveal_process = DemandRevealProcess(
            problem=world,
            mode=self.manifest.reveal_mode,
            ratios=self.manifest.ratios,
            seed=self.manifest.reveal_seed,
        )
        return _Episode(world, reveal_process, self.manifest)

    def _history_view(self) -> SanitizedHistoryView:
        return SanitizedHistoryView.from_completed_matrices(
            matrices=self._history,
            steps=tuple(range(len(self._history))),
            sequence_id=self.manifest.sequence_id,
            current_step=len(self._history),
        )

    def _row(
        self,
        *,
        method: str,
        world: UncertainProblemInstance,
        completion: int,
        oracle_completion: int,
        reveal_stage: int,
        reveal_wait: int,
        recourse_count: int,
        replanned_actions: int,
        wasted_plan: int,
        synthesis_time_ms: float,
        replan_time_ms: float,
        legality: bool,
        legal_attempts: int,
        commit_attempts: int,
        timeout: bool,
        uses_oracle: bool,
        upper_bound_only: bool,
        legality_error: str | None = None,
    ) -> dict[str, Any]:
        topology_name = getattr(world.topology_info, "name", None) or "unnamed"
        metrics = RecourseMetrics(
            completion=int(completion),
            oracle_regret=float(completion - oracle_completion),
            reveal_wait=int(reveal_wait),
            recourse_count=int(recourse_count),
            replanned_actions=int(replanned_actions),
            wasted_plan=int(wasted_plan),
            synthesis_time_ms=float(synthesis_time_ms),
            replan_time_ms=float(replan_time_ms),
            legality=bool(legality),
            timeout=bool(timeout),
            sequence_id=self.manifest.sequence_id,
            family=self.manifest.family,
            seed=self.manifest.reveal_seed,
            topology=str(topology_name),
            reveal_stage=int(reveal_stage),
            reveal_mode=self.manifest.reveal_mode,
            method=method,
            manifest_id=self.manifest.manifest_id,
            truth_digest=self.manifest.truth_digest,
            topology_digest=self.manifest.topology_digest,
            config_digest=self.manifest.config_digest,
            checker_version=self.manifest.checker_version,
        )
        row = metrics.to_raw_row()
        row.update(
            {
                "legal": metrics.legality,
                "legality_rate": (
                    1.0 if commit_attempts == 0 else legal_attempts / commit_attempts
                ),
                "uses_oracle": bool(uses_oracle),
                "upper_bound_only": bool(upper_bound_only),
                "reveal_seed": self.manifest.reveal_seed,
                "history_provenance": self.manifest.history_provenance,
                "ratios": self.manifest.ratios,
                "time_limit": self.manifest.time_limit,
                "timeout_limit": self.manifest.timeout,
            }
        )
        if legality_error is not None:
            row["legality_error"] = legality_error
        return row

    def run_oracle(self, oracle: FullInformationOracle) -> dict[str, Any]:
        if not isinstance(oracle, FullInformationOracle):
            raise TypeError("run_oracle requires FullInformationOracle")
        episode = self.build_episode(oracle.method)
        synthesis_start = perf_counter()
        completion = _oracle_completion_lower_bound(
            self._truth, self._topology, self.manifest.time_limit
        )
        synthesis_time_ms = (perf_counter() - synthesis_start) * 1000.0
        final_stage = len(self.manifest.ratios) - 1
        row = self._row(
            method=oracle.method,
            world=episode.world,
            completion=completion,
            oracle_completion=completion,
            reveal_stage=final_stage,
            reveal_wait=0,
            recourse_count=0,
            replanned_actions=0,
            wasted_plan=0,
            synthesis_time_ms=synthesis_time_ms,
            replan_time_ms=0.0,
            legality=True,
            legal_attempts=0,
            commit_attempts=0,
            timeout=completion == self.manifest.time_limit + 1,
            uses_oracle=True,
            upper_bound_only=True,
        )
        row.update(
            {
                "reference_kind": "provable_full_information_lower_bound",
                "executable": False,
                "legality_basis": "vacuous_no_executable_actions",
            }
        )
        return row

    def _run_ordinary(
        self, baseline: Any, *, oracle_completion: int | None = None
    ) -> dict[str, Any]:
        if oracle_completion is None:
            oracle_completion = int(self.run_oracle(FullInformationOracle())["completion"])
        episode = self.build_episode(baseline.method)
        history = self._history_view()
        completion = 0 if _world_complete(episode.world) else self.manifest.time_limit + 1
        first_action_slot: int | None = None
        previous_signature: tuple[Any, ...] | None = None
        recourse_count = 0
        replanned_actions = 0
        wasted_plan: int | None = None
        synthesis_time_ms = 0.0
        replan_time_ms = 0.0
        legal_attempts = 0
        commit_attempts = 0
        legality = True
        error: str | None = None
        final_stage = 0

        for slot in range(self.manifest.time_limit):
            if _world_complete(episode.world):
                break
            observation = episode.next_observation()
            final_stage = observation.stage
            start = perf_counter()
            proposal = baseline.propose(observation, history=history, scenarios=None)
            elapsed = (perf_counter() - start) * 1000.0
            if slot == 0:
                synthesis_time_ms = elapsed
            else:
                replan_time_ms += elapsed
                replanned_actions += len(proposal.actions)

            signature = _proposal_signature(proposal)
            if previous_signature is not None and signature != previous_signature:
                recourse_count += 1
            previous_signature = signature

            if proposal.scenario_set is not None:
                # Scenario-only plans remain non-executable.  Their first-plan
                # waste is evaluated privately only after planning, as L1 to
                # final truth; the planner never receives that truth.
                if wasted_plan is None:
                    planned = np.asarray(proposal.scenario_set.matrices[0])
                    wasted_plan = int(np.abs(planned - self._truth).sum())
            elif proposal.actions:
                if first_action_slot is None:
                    first_action_slot = slot
                commit_attempts += 1
                try:
                    result = episode.world.commit(observation, proposal)
                except (TypeError, ValueError) as caught:
                    legality = False
                    error = str(caught)
                    break
                legal_attempts += int(result.legal)

            if _world_complete(episode.world):
                completion = slot + 1
                break

        timeout = not _world_complete(episode.world)
        return self._row(
            method=baseline.method,
            world=episode.world,
            completion=completion,
            oracle_completion=oracle_completion,
            reveal_stage=final_stage,
            reveal_wait=(
                self.manifest.time_limit if first_action_slot is None else first_action_slot
            ),
            recourse_count=recourse_count,
            replanned_actions=replanned_actions,
            wasted_plan=0 if wasted_plan is None else wasted_plan,
            synthesis_time_ms=synthesis_time_ms,
            replan_time_ms=replan_time_ms,
            legality=legality,
            legal_attempts=legal_attempts,
            commit_attempts=commit_attempts,
            timeout=timeout,
            uses_oracle=False,
            upper_bound_only=False,
            legality_error=error,
        )

    def run_default_baselines(self) -> list[dict[str, Any]]:
        oracle = self.run_oracle(FullInformationOracle())
        oracle_completion = int(oracle["completion"])
        return [
            oracle,
            self._run_ordinary(
                WaitUntilKnownBaseline(), oracle_completion=oracle_completion
            ),
            self._run_ordinary(
                PartialCurrentOnlyBaseline(), oracle_completion=oracle_completion
            ),
            self._run_ordinary(
                LongTermMeanBaseline(), oracle_completion=oracle_completion
            ),
            self._run_ordinary(
                PreviousValueBaseline(), oracle_completion=oracle_completion
            ),
        ]


__all__ = ["EvaluationManifest", "PairedEvaluationRunner"]
