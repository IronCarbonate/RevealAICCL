"""Append-only execution ledger and current-state prefix repair."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from rlccl.uncertainty.execution import Proposal, TransferAction, commit_proposal
from rlccl.uncertainty.observation import TruthTokenId


class StalePlanError(ValueError):
    pass


class ReplayedActionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutedActionIdentity:
    truth_token_id: Any
    edge_index: int
    plan_revision: int
    batch_ordinal: int


@dataclass(frozen=True, slots=True)
class ExecutedLedger:
    actions: tuple[ExecutedActionIdentity, ...] = ()

    @classmethod
    def empty(cls) -> "ExecutedLedger":
        return cls()

    def can_append(self, identity: ExecutedActionIdentity) -> bool:
        return identity not in self.actions

    def append(self, identity: ExecutedActionIdentity) -> "ExecutedLedger":
        if not self.can_append(identity):
            raise ReplayedActionError("exact executed action replay")
        return ExecutedLedger(self.actions + (identity,))


@dataclass(frozen=True, slots=True)
class RecourseState:
    plan: Any | None
    executed_actions: tuple[ExecutedActionIdentity, ...] = ()
    discarded_actions: int = 0
    execution_start_state_version: int = 0
    reason: str = "initial"
    wait_latch: tuple[int, int] | None = None
    next_batch_index: int = 0
    current_revision: int = 0

    @classmethod
    def initial(cls, plan: Any) -> "RecourseState":
        return cls(plan=plan, execution_start_state_version=int(plan.origin_state_version),
                   current_revision=int(plan.revision))

    @classmethod
    def waiting(cls, view: Any) -> "RecourseState":
        return cls(plan=None, execution_start_state_version=int(view.state_version), reason="wait")


def _validate_binding(state: RecourseState, view: Any, trusted: Any | None) -> None:
    if state.plan is None or not state.plan.batches:
        raise StalePlanError("plan has no executable batch")
    plan = state.plan
    if plan.revision < 0 or int(plan.revision) != int(state.current_revision):
        raise StalePlanError("stale plan revision")
    if plan.origin_stage != view.stage:
        raise StalePlanError("plan stage mismatch")
    if plan.origin_state_version != view.state_version:
        raise StalePlanError("plan state version mismatch")
    if trusted is not None:
        from rlccl.scheduling.robust_prefix import build_scheduling_view
        if int(trusted.stage) != int(view.stage):
            raise StalePlanError("trusted stage mismatch")
        if int(trusted.state_version) != int(view.state_version):
            raise StalePlanError("trusted state version mismatch")
        trusted_structural = tuple(
            (int(token.source), int(token.destination), tuple(int(x) for x in token.holders))
            for token in trusted.revealed_tokens
        )
        public_structural = tuple((token.source, token.destination, token.holders) for token in view.revealed_tokens)
        if trusted_structural != public_structural:
            raise StalePlanError("trusted revealed tuple mismatch")
        rebuilt = build_scheduling_view(trusted)
        if rebuilt.observation_digest != view.observation_digest or rebuilt.residual_state_digest != view.residual_state_digest:
            raise StalePlanError("trusted observation/view digest mismatch")
    from rlccl.scheduling.robust_prefix import candidate_distances
    edges = np.asarray(view.topology.edges, dtype=np.int64)
    for item in plan.batches[0]:
        token = view.revealed_tokens[item.local_token_ordinal]
        source, destination = map(int, edges[item.edge_index])
        if source not in token.holders or destination in token.holders:
            raise StalePlanError("next batch holder precondition invalid")
        distances = candidate_distances(view.topology, holders=token.holders,
                                        destination=token.destination, edge_index=item.edge_index)
        if (distances.before_global, distances.after_global) != (item.before_distance, item.after_distance):
            raise StalePlanError("next batch distance precondition invalid")


def bind_action(
    view: Any, *, local_token_ordinal: int, edge_index: int,
    trusted_observation: Any | None = None,
) -> TransferAction:
    if isinstance(local_token_ordinal, (bool, np.bool_)) or not isinstance(local_token_ordinal, (int, np.integer)):
        raise TypeError("local token ordinal must be an integer")
    ordinal = int(local_token_ordinal)
    if ordinal < 0 or ordinal >= len(view.revealed_tokens):
        raise IndexError("local token ordinal is hidden or outside the revealed tuple")
    if trusted_observation is None:
        raise ValueError("trusted observation is required to bind opaque truth token identity")
    if ordinal >= len(trusted_observation.revealed_tokens):
        raise ValueError("trusted revealed tuple mismatch")
    truth_id = trusted_observation.revealed_tokens[ordinal].token_id
    if not isinstance(truth_id, TruthTokenId):
        raise TypeError("only revealed truth tokens are executable")
    return TransferAction(truth_id, int(edge_index))


def bind_first_batch(
    state: RecourseState, view: Any, *, trusted_observation: Any | None = None,
) -> Proposal:
    _validate_binding(state, view, trusted_observation)
    if trusted_observation is None:
        raise ValueError("trusted observation is required for executable binding")
    actions = tuple(
        bind_action(view, local_token_ordinal=item.local_token_ordinal,
                    edge_index=item.edge_index, trusted_observation=trusted_observation)
        for item in state.plan.batches[0]
    )
    return Proposal.from_transfers(actions)


def record_committed_batch(
    state: RecourseState, *, proposal: Proposal, commit_result: Any,
    fresh_observation: Any,
) -> RecourseState:
    if not commit_result.legal:
        raise ValueError("cannot record an illegal commit")
    if int(fresh_observation.state_version) != int(commit_result.state_version):
        raise StalePlanError("fresh observation state version does not match commit")
    fresh_by_id = {token.token_id: token for token in fresh_observation.revealed_tokens}
    edges = np.asarray(fresh_observation.topology.edges, dtype=np.int64)
    for action in proposal.actions:
        token = fresh_by_id.get(action.token_id)
        if token is None:
            raise StalePlanError("fresh revealed tuple lost a committed token")
        destination = int(edges[int(action.edge_index), 1])
        if destination not in token.holders:
            raise StalePlanError("fresh holder state does not contain committed destination")
    identities = tuple(
        ExecutedActionIdentity(action.token_id, int(action.edge_index),
                               int(state.plan.revision), int(state.next_batch_index))
        for action in proposal.actions
    )
    ledger = ExecutedLedger(tuple(state.executed_actions))
    for identity in identities:
        ledger = ledger.append(identity)
    remaining = tuple(state.plan.batches[1:])
    structural = tuple((action.local_token_ordinal, action.edge_index) for batch in remaining for action in batch)
    plan = replace(state.plan, origin_state_version=int(commit_result.state_version),
                   batches=remaining, structural_actions=structural,
                   simulation_trace=tuple(state.plan.simulation_trace[1:]))
    return replace(state, plan=plan, executed_actions=ledger.actions, reason="commit",
                   next_batch_index=state.next_batch_index + 1)


def repair_prefix(state: RecourseState, view: Any, support: Any, planner: Any, *, reason: str) -> RecourseState:
    discarded = sum(len(batch) for batch in state.plan.batches) if state.plan is not None else 0
    fresh = planner.plan(view, support)
    revision = (state.plan.revision + 1) if state.plan is not None else 0
    return replace(state, plan=replace(fresh, revision=revision),
                   discarded_actions=state.discarded_actions + discarded, reason=str(reason),
                   wait_latch=None, next_batch_index=0, current_revision=revision)


@dataclass(frozen=True, slots=True)
class WaitOutcome:
    state: RecourseState
    no_common_delta: int
    fallback_delta: int


def enter_wait_latch(state: RecourseState, view: Any) -> WaitOutcome:
    key = (int(view.stage), int(view.state_version))
    delta = int(state.wait_latch != key)
    return WaitOutcome(replace(state, wait_latch=key), delta, delta)


@dataclass(frozen=True, slots=True)
class Transition:
    reason: str
    reveal_delta: int
    exhaustion_delta: int
    invalidation_delta: int


def choose_transition(*, new_stage: bool, prefix_exhausted: bool, precondition_valid: bool) -> Transition:
    reason = "reveal" if new_stage else "exhaustion" if prefix_exhausted else "invalidation" if not precondition_valid else "continue"
    return Transition(reason, int(reason == "reveal"), int(reason == "exhaustion"), int(reason == "invalidation"))


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    legality: bool
    stopped: bool
    illegal_reason: str | None
    fallback_delta: int
    no_common_delta: int
    commit_result: Any | None
    proposal: Proposal | None = None


def execute_first_batch(*, world: Any, state: RecourseState, view: Any, trusted_observation: Any) -> ExecutionOutcome:
    try:
        proposal = bind_first_batch(state, view, trusted_observation=trusted_observation)
        result = commit_proposal(world, trusted_observation, proposal)
    except ValueError as error:
        reason = "stale_observation" if "stale" in str(error).lower() else "checker_rejection"
        return ExecutionOutcome(False, True, reason, 0, 0, None, None)
    return ExecutionOutcome(bool(result.legal), not bool(result.legal), None if result.legal else "checker_rejection", 0, 0, result, proposal)


__all__ = [
    "ExecutedActionIdentity", "ExecutedLedger", "RecourseState", "StalePlanError",
    "ReplayedActionError", "bind_action", "bind_first_batch", "record_committed_batch",
    "repair_prefix", "enter_wait_latch", "choose_transition", "execute_first_batch",
]
