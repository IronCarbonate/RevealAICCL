"""Public proposal types and trusted deterministic execution checks."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from .observation import PartialObservationState, TruthTokenId
from .scenarios import ScenarioSet

if TYPE_CHECKING:
    from .problem import UncertainProblemInstance


@dataclass(frozen=True, slots=True)
class TransferAction:
    token_id: TruthTokenId
    edge_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_id, TruthTokenId):
            raise TypeError("TransferAction requires TruthTokenId; scenario tokens are not executable")
        if isinstance(self.edge_index, (bool, np.bool_)) or not isinstance(
            self.edge_index, (int, np.integer)
        ):
            raise TypeError("edge_index must be an integer")
        object.__setattr__(self, "edge_index", int(self.edge_index))


@dataclass(frozen=True, slots=True)
class Proposal:
    actions: tuple[TransferAction, ...] = ()
    scenario_set: ScenarioSet | None = None

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if any(not isinstance(action, TransferAction) for action in actions):
            raise TypeError("Proposal actions must be TransferAction objects")
        if self.scenario_set is not None and actions:
            raise ValueError("A scenario-only proposal cannot contain executable actions")
        object.__setattr__(self, "actions", actions)

    @classmethod
    def wait(cls) -> "Proposal":
        return cls()

    @classmethod
    def from_transfers(cls, actions: tuple[TransferAction, ...]) -> "Proposal":
        return cls(actions=tuple(actions))

    @classmethod
    def scenario_only(cls, scenario_set: ScenarioSet) -> "Proposal":
        if not isinstance(scenario_set, ScenarioSet):
            raise TypeError("scenario_only requires ScenarioSet")
        return cls(scenario_set=scenario_set)

    @property
    def is_wait(self) -> bool:
        return not self.actions and self.scenario_set is None

    def to_public_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "actions": tuple(
                    MappingProxyType(
                        {"token_id": action.token_id, "edge_index": action.edge_index}
                    )
                    for action in self.actions
                ),
                "is_wait": self.is_wait,
                "scenario_set": self.scenario_set,
            }
        )


@dataclass(frozen=True, slots=True)
class CommitResult:
    legal: bool
    applied_actions: int
    state_version: int


def validate_legacy_schedule_matrix(
    matrix: Any, *, expected_shape: tuple[int, int]
) -> np.ndarray:
    candidate = np.asarray(matrix)
    if candidate.ndim != 2:
        raise ValueError("Legacy schedule matrix must be 2-D")
    if tuple(candidate.shape) != tuple(expected_shape):
        raise ValueError(f"Legacy schedule matrix shape must be {expected_shape}")
    if not np.issubdtype(candidate.dtype, np.number):
        raise ValueError("Legacy schedule matrix must contain numeric binary values")
    numeric = candidate.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Legacy schedule matrix must contain finite values, not NaN/Inf")
    if np.any(numeric < 0):
        raise ValueError("Legacy schedule matrix must be nonnegative")
    if np.any((numeric != 0) & (numeric != 1)):
        raise ValueError("Legacy schedule matrix must be binary")
    result = np.array(numeric, dtype=np.int8, copy=True)
    result.setflags(write=False)
    return result


def commit_proposal(
    world: "UncertainProblemInstance",
    observation: PartialObservationState,
    proposal: Proposal,
) -> CommitResult:
    if not isinstance(observation, PartialObservationState):
        raise TypeError("commit requires PartialObservationState")
    if not isinstance(proposal, Proposal):
        raise TypeError("commit requires Proposal")
    if proposal.scenario_set is not None:
        raise ValueError("Scenario tokens are planning-only and not executable")
    if observation.sequence_id != world.sequence_id or observation.sequence_step != world.sequence_step:
        raise ValueError("Observation belongs to a different problem stage")
    if observation.state_version != world._state_version:
        raise ValueError("stale observation cannot be committed")
    if proposal.is_wait:
        return CommitResult(True, 0, world._state_version)

    executable = set(observation.executable_token_ids)
    seen: set[TruthTokenId] = set()
    edge_load = np.zeros(world.topology_info.E, dtype=np.float64)
    resolved: list[tuple[int, int, int]] = []
    for action in proposal.actions:
        if action.token_id in seen:
            raise ValueError("Duplicate token action conflict in one slot")
        seen.add(action.token_id)
        if action.token_id not in executable or action.token_id not in world._public_to_private:
            raise ValueError("Token is unrevealed and not executable at this stage")
        edge = int(action.edge_index)
        if edge < 0 or edge >= world.topology_info.E:
            raise ValueError("edge_index is outside valid edge range")
        token_index = world._public_to_private[action.token_id]
        source = int(world.topology_info.edge_src[edge])
        destination = int(world.topology_info.edge_dst[edge])
        if not world._possession[token_index, source]:
            raise ValueError("Edge source is not a current token holder; source possession required")
        if world._possession[token_index, destination]:
            raise ValueError("Edge destination already holds this token")
        edge_load[edge] += 1.0
        resolved.append((token_index, source, destination))

    if np.any(edge_load > np.asarray(world.topology_info.capacities, dtype=float) + 1e-12):
        raise ValueError("Edge capacity/bandwidth exceeded")
    for edge_indices, limit in world.topology_info.shared_constraints:
        load = float(edge_load[np.asarray(edge_indices, dtype=np.int64)].sum())
        if load > float(limit) + 1e-12:
            raise ValueError("Shared group bandwidth limit exceeded")

    # Apply only after every check.  Sources above came from the pre-slot state,
    # so a token received here cannot be forwarded in this same slot.
    for token_index, _, destination in resolved:
        world._possession[token_index, destination] = True
    world._state_version += 1
    return CommitResult(True, len(resolved), world._state_version)


__all__ = [
    "CommitResult",
    "Proposal",
    "TransferAction",
    "commit_proposal",
    "validate_legacy_schedule_matrix",
]
