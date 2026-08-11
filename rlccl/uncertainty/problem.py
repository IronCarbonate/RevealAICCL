"""Private NumPy world for reveal-aware collective communication."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from ..envs.problem import TopologyInfo
from ..traffic.matrix_utils import validate_traffic_matrix
from .observation import PublicTopologyView, TruthTokenId


class UncertainProblemInstance:
    """Evaluator-owned truth and mutable token-possession state.

    Instances of this class are capabilities and must never be supplied to an
    ordinary planner.  The public boundary is :class:`PartialObservationState`.
    """

    def __init__(
        self,
        *,
        truth_matrix: np.ndarray,
        topology_info: TopologyInfo,
        time_limit: int,
        sequence_id: str,
        sequence_step: int,
        family: str,
        generator_metadata: Mapping[str, Any],
    ) -> None:
        validate_traffic_matrix(np.asarray(truth_matrix))
        truth = np.array(truth_matrix, dtype=np.int64, copy=True)
        if truth.shape != (int(topology_info.V), int(topology_info.V)):
            raise ValueError("Traffic matrix and topology node count mismatch")
        if int(time_limit) <= 0:
            raise ValueError("time_limit must be positive")

        edges = np.array(topology_info.edges, dtype=np.int64, copy=True)
        capacities = np.array(topology_info.capacities, dtype=np.float64, copy=True)
        constraints = [
            (list(int(edge) for edge in edge_indices), float(limit))
            for edge_indices, limit in topology_info.shared_constraints
        ]
        self.topology_info = TopologyInfo(
            int(topology_info.V),
            int(topology_info.E),
            edges,
            capacities,
            constraints,
            name=getattr(topology_info, "name", None),
        )
        self.public_topology = PublicTopologyView.from_topology_info(self.topology_info)
        self.time_limit = int(time_limit)
        self.sequence_id = str(sequence_id)
        self.sequence_step = int(sequence_step)
        self.family = str(family)

        self._truth = truth
        self._generator_metadata = deepcopy(dict(generator_metadata))
        atomic: list[tuple[int, int, int]] = []
        local_by_pair: dict[tuple[int, int], list[int]] = {}
        for source in range(truth.shape[0]):
            for destination in range(truth.shape[1]):
                if source == destination:
                    continue
                indices: list[int] = []
                for local_index in range(int(truth[source, destination])):
                    token_index = len(atomic)
                    atomic.append((source, destination, local_index))
                    indices.append(token_index)
                local_by_pair[(source, destination)] = indices
        self._atomic = tuple(atomic)
        self._pair_indices = local_by_pair
        self._possession = np.zeros((len(atomic), truth.shape[0]), dtype=bool)
        for token_index, (source, _, _) in enumerate(atomic):
            self._possession[token_index, source] = True
        self._public_to_private: dict[TruthTokenId, int] = {}
        self._private_to_public: dict[int, TruthTokenId] = {}
        self._state_version = 0

    @classmethod
    def from_traffic_matrix(
        cls,
        *,
        truth_matrix: np.ndarray,
        topology_info: TopologyInfo,
        time_limit: int,
        sequence_id: str,
        sequence_step: int,
        family: str,
        generator_metadata: Mapping[str, Any],
    ) -> "UncertainProblemInstance":
        return cls(
            truth_matrix=truth_matrix,
            topology_info=topology_info,
            time_limit=time_limit,
            sequence_id=sequence_id,
            sequence_step=sequence_step,
            family=family,
            generator_metadata=generator_metadata,
        )

    @property
    def _token_count(self) -> int:
        return len(self._atomic)

    @property
    def _node_count(self) -> int:
        return self._truth.shape[0]

    def _indices_for_pair(self, source: int, destination: int) -> tuple[int, ...]:
        return tuple(self._pair_indices[(int(source), int(destination))])

    def _issue_token_id(self, token_index: int, *, reveal_seed: int) -> TruthTokenId:
        existing = self._private_to_public.get(int(token_index))
        if existing is not None:
            return existing
        source, destination, local_index = self._atomic[int(token_index)]
        material = (
            f"{self.sequence_id}|{self.sequence_step}|{reveal_seed}|"
            f"{source}|{destination}|{local_index}"
        ).encode("utf-8")
        public_id = TruthTokenId("truth:" + sha256(material).hexdigest())
        self._private_to_public[int(token_index)] = public_id
        self._public_to_private[public_id] = int(token_index)
        return public_id

    def _token_record(self, token_index: int) -> tuple[int, int, tuple[int, ...]]:
        source, destination, _ = self._atomic[int(token_index)]
        holders = tuple(int(value) for value in np.flatnonzero(self._possession[int(token_index)]))
        return source, destination, holders

    def commit(self, observation: Any, proposal: Any) -> Any:
        """Validate and atomically apply one communication slot."""
        from .execution import commit_proposal

        return commit_proposal(self, observation, proposal)


__all__ = ["UncertainProblemInstance"]
