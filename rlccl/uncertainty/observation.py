"""Immutable policy-facing views for the Phase 1 uncertainty environment."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..traffic.matrix_utils import validate_traffic_matrix


def readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return an owned NumPy copy whose write flag is disabled."""
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _frozen_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class TruthTokenId:
    """Opaque public identity issued only when a real demand token is revealed."""

    opaque_value: str

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_value, str) or not self.opaque_value:
            raise ValueError("TruthTokenId requires a non-empty opaque string")

    def __str__(self) -> str:
        return self.opaque_value


@dataclass(frozen=True, slots=True)
class RevealedDemandToken:
    token_id: TruthTokenId
    source: int
    destination: int
    holders: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.token_id, TruthTokenId):
            raise TypeError("RevealedDemandToken requires a TruthTokenId")
        if self.source < 0 or self.destination < 0 or self.source == self.destination:
            raise ValueError("Invalid revealed demand endpoints")
        holders = tuple(int(value) for value in self.holders)
        if any(value < 0 for value in holders):
            raise ValueError("Holder indices must be nonnegative")
        object.__setattr__(self, "holders", holders)


@dataclass(frozen=True, slots=True)
class PublicTopologyView:
    """Deep-copied static topology safe for ordinary planners."""

    num_nodes: int
    num_edges: int
    edges: np.ndarray
    capacities: np.ndarray
    shared_constraints: tuple[tuple[tuple[int, ...], float], ...]
    name: str | None = None

    def __post_init__(self) -> None:
        edges = readonly_array(self.edges, dtype=np.int64)
        capacities = readonly_array(self.capacities, dtype=np.float64)
        if edges.shape != (self.num_edges, 2):
            raise ValueError("Public topology edge shape mismatch")
        if capacities.shape != (self.num_edges,):
            raise ValueError("Public topology capacity shape mismatch")
        constraints = tuple(
            (tuple(int(edge) for edge in edge_indices), float(limit))
            for edge_indices, limit in self.shared_constraints
        )
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "capacities", capacities)
        object.__setattr__(self, "shared_constraints", constraints)

    @classmethod
    def from_topology_info(cls, topology: Any) -> "PublicTopologyView":
        return cls(
            num_nodes=int(topology.V),
            num_edges=int(topology.E),
            edges=np.asarray(topology.edges),
            capacities=np.asarray(topology.capacities),
            shared_constraints=tuple(topology.shared_constraints),
            name=getattr(topology, "name", None),
        )

    @property
    def edge_src(self) -> np.ndarray:
        return readonly_array(self.edges[:, 0], dtype=np.int64)

    @property
    def edge_dst(self) -> np.ndarray:
        return readonly_array(self.edges[:, 1], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class PartialObservationState:
    """The only current-demand object accepted by an ordinary planner."""

    sequence_id: str
    sequence_step: int
    family: str
    mode: str
    stage: int
    ratio: float
    entry_mask: np.ndarray
    observed_matrix: np.ndarray
    unknown_mask: np.ndarray
    revealed_tokens: tuple[RevealedDemandToken, ...]
    source_totals: np.ndarray | None
    destination_totals: np.ndarray | None
    topology: PublicTopologyView
    state_version: int

    def __post_init__(self) -> None:
        entry_mask = readonly_array(self.entry_mask, dtype=bool)
        observed = readonly_array(self.observed_matrix, dtype=np.int64)
        unknown = readonly_array(self.unknown_mask, dtype=bool)
        if observed.ndim != 2 or observed.shape[0] != observed.shape[1]:
            raise ValueError("Observed matrix must be square")
        if entry_mask.shape != observed.shape or unknown.shape != observed.shape:
            raise ValueError("Observation masks must match observed matrix")
        if not np.array_equal(unknown, ~entry_mask):
            raise ValueError("unknown_mask must be the complement of entry_mask")
        if np.any(np.diag(observed) != 0):
            raise ValueError("Observed matrix diagonal must be zero")
        tokens = tuple(self.revealed_tokens)
        if len({token.token_id for token in tokens}) != len(tokens):
            raise ValueError("Revealed token IDs must be unique")
        source_totals = (
            None
            if self.source_totals is None
            else readonly_array(self.source_totals, dtype=np.int64)
        )
        destination_totals = (
            None
            if self.destination_totals is None
            else readonly_array(self.destination_totals, dtype=np.int64)
        )
        if source_totals is not None and source_totals.shape != (observed.shape[0],):
            raise ValueError("source_totals shape mismatch")
        if destination_totals is not None and destination_totals.shape != (observed.shape[0],):
            raise ValueError("destination_totals shape mismatch")
        object.__setattr__(self, "entry_mask", entry_mask)
        object.__setattr__(self, "observed_matrix", observed)
        object.__setattr__(self, "unknown_mask", unknown)
        object.__setattr__(self, "revealed_tokens", tokens)
        object.__setattr__(self, "source_totals", source_totals)
        object.__setattr__(self, "destination_totals", destination_totals)

    @property
    def executable_token_ids(self) -> tuple[TruthTokenId, ...]:
        return tuple(token.token_id for token in self.revealed_tokens)

    def to_policy_payload(self) -> Mapping[str, Any]:
        """Return a new read-only payload with no evaluator-private capability."""
        topology_payload = _frozen_mapping(
            {
                "num_nodes": self.topology.num_nodes,
                "num_edges": self.topology.num_edges,
                "edges": readonly_array(self.topology.edges),
                "capacities": readonly_array(self.topology.capacities),
                "shared_constraints": self.topology.shared_constraints,
                "name": self.topology.name,
            }
        )
        return _frozen_mapping(
            {
                "sequence_id": self.sequence_id,
                "sequence_step": self.sequence_step,
                "family": self.family,
                "mode": self.mode,
                "stage": self.stage,
                "ratio": self.ratio,
                "entry_mask": readonly_array(self.entry_mask),
                "observed_matrix": readonly_array(self.observed_matrix),
                "unknown_mask": readonly_array(self.unknown_mask),
                "revealed_tokens": self.revealed_tokens,
                "source_totals": (
                    None
                    if self.source_totals is None
                    else readonly_array(self.source_totals)
                ),
                "destination_totals": (
                    None
                    if self.destination_totals is None
                    else readonly_array(self.destination_totals)
                ),
                "topology": topology_payload,
                "state_version": self.state_version,
            }
        )


@dataclass(frozen=True, slots=True)
class SanitizedHistoryView:
    """Completed matrices from one sequence, never including the current step."""

    matrices: tuple[np.ndarray, ...]
    steps: tuple[int, ...]
    sequence_id: str
    current_step: int

    def __post_init__(self) -> None:
        raw_matrices = tuple(np.asarray(matrix) for matrix in self.matrices)
        for matrix in raw_matrices:
            validate_traffic_matrix(matrix)
        matrices = tuple(readonly_array(matrix, dtype=np.int64) for matrix in raw_matrices)
        steps = tuple(int(step) for step in self.steps)
        if len(matrices) != len(steps):
            raise ValueError("History matrices and steps must have equal lengths")
        if tuple(sorted(steps)) != steps or len(set(steps)) != len(steps):
            raise ValueError("History steps must be strictly increasing")
        if any(step >= self.current_step for step in steps):
            raise ValueError("History steps must be < current step; current/future X is forbidden")
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_completed_matrices(
        cls,
        *,
        matrices: Sequence[np.ndarray],
        steps: Sequence[int],
        sequence_id: str,
        current_step: int,
    ) -> "SanitizedHistoryView":
        return cls(
            matrices=tuple(matrices),
            steps=tuple(steps),
            sequence_id=str(sequence_id),
            current_step=int(current_step),
        )


__all__ = [
    "PartialObservationState",
    "PublicTopologyView",
    "RevealedDemandToken",
    "SanitizedHistoryView",
    "TruthTokenId",
    "readonly_array",
]
