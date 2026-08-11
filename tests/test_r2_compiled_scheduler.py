"""R2-C0 unit contracts for the compiled partial-current scheduler."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from rlccl.envs.problem import TopologyInfo
from rlccl.scheduling.compiled_event_driven import (
    CompiledEventDrivenRuntime,
    DynamicGuard,
    FastBinder,
    IncrementalState,
    StaticPlanCompiler,
    structural_signature,
)
from rlccl.scheduling.recourse import bind_action
from rlccl.scheduling.robust_prefix import (
    build_scheduling_view,
    canonical_shortest_path,
    enumerate_candidates,
    pack_candidate_batch,
)
from rlccl.uncertainty.execution import Proposal, TransferAction, commit_proposal
from rlccl.uncertainty.problem import UncertainProblemInstance
from rlccl.uncertainty.reveal import DemandRevealProcess


def _topology(
    edges: tuple[tuple[int, int], ...],
    *,
    capacities: tuple[float, ...] | None = None,
    groups: tuple[tuple[tuple[int, ...], float], ...] = (),
    nodes: int = 4,
) -> TopologyInfo:
    return TopologyInfo(
        nodes,
        len(edges),
        np.asarray(edges, dtype=np.int64),
        np.asarray(capacities or (1.0,) * len(edges), dtype=np.float64),
        list(groups),
        name="r2-c0-test",
    )


def _episode(topology: TopologyInfo, truth: np.ndarray, ratio: float = 1.0):
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth,
        topology_info=topology,
        time_limit=80,
        sequence_id="r2-unit",
        sequence_step=8,
        family="r2-unit",
        generator_metadata={},
    )
    reveal = DemandRevealProcess(
        problem=world,
        mode="partial_shards",
        ratios=(0.0, ratio, 1.0) if ratio < 1.0 else (0.0, 1.0),
        seed=20260810,
    )
    observation = reveal.observation_for_stage(1)
    return world, reveal, observation


def _old_new(observation):
    view = build_scheduling_view(observation)
    old_candidates = enumerate_candidates(view)
    old_selected = pack_candidate_batch(old_candidates, view.topology)
    plan = StaticPlanCompiler().compile(view.topology)
    state = IncrementalState.from_observation(plan, observation, max_tokens=64)
    bound = FastBinder(plan).step(state)
    return view, old_candidates, old_selected, plan, state, bound


def test_static_compiler_matches_canonical_bfs_with_destination_first_tie() -> None:
    edges = ((3, 2), (2, 0), (0, 3), (1, 2), (2, 3), (3, 0),
             (1, 3), (2, 1), (3, 1), (0, 1), (1, 2))
    topology = _topology(edges)
    plan = StaticPlanCompiler().compile(topology)
    assert plan.proof.valid
    for source in range(4):
        for destination in range(4):
            if source == destination:
                assert plan.canonical_paths[source][destination] == ()
            else:
                assert plan.canonical_paths[source][destination] == canonical_shortest_path(
                    topology, source, destination,
                )
    assert plan.canonical_paths[0][2][0] == 9


def test_compiled_candidate_and_tie_order_are_exact() -> None:
    topology = _topology(((0, 1), (0, 2), (1, 3), (2, 3), (1, 0), (2, 0), (3, 0)))
    truth = np.zeros((4, 4), dtype=np.int64)
    truth[0, 3] = 2
    _, _, observation = _episode(topology, truth)
    _, old_candidates, old_selected, _, _, bound = _old_new(observation)
    assert structural_signature(bound.candidates) == structural_signature(old_candidates)
    assert structural_signature(bound.selected) == structural_signature(old_selected)
    # Token 0 takes the lower edge index. Edge-0 capacity is then exhausted,
    # so token 1 deterministically takes the other equal-length route.
    assert tuple(item.edge_index for item in bound.selected) == (0, 1)


def test_pending_hidden_chunk_cannot_change_current_action() -> None:
    topology = _topology(((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)), nodes=4)
    truth = np.zeros((4, 4), dtype=np.int64)
    truth[0, 3] = 2
    _, reveal, observation = _episode(topology, truth, ratio=0.5)
    view, _, _, plan, state, _ = _old_new(observation)
    binder = FastBinder(plan)
    before = structural_signature(binder.step(state).selected)
    full = reveal.full_observation()
    hidden = full.revealed_tokens[len(view.revealed_tokens):]
    state.stage_ready_chunk(7, hidden)
    after = structural_signature(binder.step(state).selected)
    assert before == after
    assert state.pending_ready_bitmap != 0 and state.revealed_count == len(view.revealed_tokens)


def test_guard_and_old_checker_accept_and_apply_identically() -> None:
    topology = _topology(((0, 1), (1, 2), (0, 2)), nodes=3)
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 1
    world, _, observation = _episode(topology, truth)
    view, _, old_selected, plan, state, bound = _old_new(observation)
    old_proposal = Proposal.from_transfers(tuple(
        bind_action(view, local_token_ordinal=item.local_token_ordinal,
                    edge_index=item.edge_index, trusted_observation=observation)
        for item in old_selected
    ))
    old = commit_proposal(world, observation, old_proposal)
    new = DynamicGuard(plan).apply(state, bound.proposal, require_scheduler_semantics=True)
    assert old.legal == new.accepted
    assert old.applied_actions == new.applied_actions
    assert old.state_version == new.state_version
    np.testing.assert_array_equal(world._possession[:state.revealed_count], state.holders[:state.revealed_count])


def test_duplicate_commit_rejected_by_both_checkers() -> None:
    topology = _topology(((0, 1),), nodes=2)
    truth = np.asarray(((0, 1), (0, 0)), dtype=np.int64)
    world, _, observation = _episode(topology, truth)
    _, _, _, plan, state, bound = _old_new(observation)
    guard = DynamicGuard(plan)
    assert commit_proposal(world, observation, bound.proposal).legal
    assert guard.apply(state, bound.proposal, require_scheduler_semantics=True).accepted
    fresh = DemandRevealProcess(problem=world, mode="partial_shards", ratios=(0.0, 1.0), seed=20260810).full_observation()
    try:
        commit_proposal(world, fresh, bound.proposal)
        old_accept = True
    except ValueError:
        old_accept = False
    assert not old_accept
    assert not guard.check(state, bound.proposal).accepted


def test_edge_and_shared_group_capacity_match_reference() -> None:
    topology = _topology(
        ((0, 1), (2, 3)),
        capacities=(1.0, 1.0),
        groups=(((0, 1), 1.0),),
    )
    truth = np.zeros((4, 4), dtype=np.int64)
    truth[0, 1] = 1
    truth[2, 3] = 1
    world, _, observation = _episode(topology, truth)
    view, old_candidates, old_selected, plan, state, bound = _old_new(observation)
    assert len(old_candidates) == 2 and len(old_selected) == 1
    assert structural_signature(bound.selected) == structural_signature(old_selected)
    actions = tuple(
        bind_action(view, local_token_ordinal=item.local_token_ordinal,
                    edge_index=item.edge_index, trusted_observation=observation)
        for item in old_candidates
    )
    proposal = Proposal.from_transfers(actions)
    try:
        commit_proposal(deepcopy(world), observation, proposal)
        old_accept = True
    except ValueError:
        old_accept = False
    assert old_accept is False
    assert DynamicGuard(plan).check(state, proposal).accepted is False


def test_unrevealed_token_rejected_by_both() -> None:
    topology = _topology(((0, 1), (1, 2), (0, 2)), nodes=3)
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 2
    world, reveal, partial = _episode(topology, truth, ratio=0.5)
    full = reveal.full_observation()
    hidden = next(token for token in full.revealed_tokens if token.token_id not in partial.executable_token_ids)
    edge = int(np.flatnonzero(np.asarray(topology.edge_src) == hidden.source)[0])
    proposal = Proposal.from_transfers((TransferAction(hidden.token_id, edge),))
    plan = StaticPlanCompiler().compile(topology)
    state = IncrementalState.from_observation(plan, partial, max_tokens=8)
    try:
        commit_proposal(deepcopy(world), partial, proposal)
        old_accept = True
    except ValueError:
        old_accept = False
    assert not old_accept and not DynamicGuard(plan).check(state, proposal).accepted


def test_incremental_reveal_uses_delta_state_without_full_rebuild() -> None:
    topology = _topology(((0, 1), (1, 2), (0, 2)), nodes=3)
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 4
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth, topology_info=topology, time_limit=80,
        sequence_id="delta", sequence_step=8, family="delta", generator_metadata={},
    )
    reveal = DemandRevealProcess(problem=world, mode="partial_shards", ratios=(0.0, 0.25, 0.5, 0.75, 1.0), seed=9)
    plan = StaticPlanCompiler().compile(topology)
    state = IncrementalState(plan, max_tokens=4, sequence_id="delta", sequence_step=8)
    for stage in range(5):
        state.ingest_observation_delta(reveal.observation_for_stage(stage))
    assert state.revealed_count == 4
    assert state.delta_update_count == 4
    assert state.full_rebuild_count == 0


def test_runtime_candidate_lookup_does_not_call_reference_bfs(monkeypatch) -> None:
    topology = _topology(((0, 1), (1, 2), (0, 2)), nodes=3)
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 1
    _, _, observation = _episode(topology, truth)
    plan = StaticPlanCompiler().compile(topology)
    state = IncrementalState.from_observation(plan, observation)
    import rlccl.scheduling.robust_prefix as reference
    monkeypatch.setattr(reference, "canonical_shortest_path", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("BFS")))
    binder = FastBinder(plan)
    assert binder.step(state).selected
    assert binder.runtime_bfs_calls == 0


def test_single_process_event_ready_runtime_uses_pending_delta() -> None:
    topology = _topology(((0, 1), (1, 2), (0, 2)), nodes=3)
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 1
    _, reveal, observation = _episode(topology, truth)
    plan = StaticPlanCompiler().compile(topology)
    state = IncrementalState(plan, max_tokens=1, sequence_id="r2-unit", sequence_step=8)
    runtime = CompiledEventDrivenRuntime(plan, state)
    runtime.stage_router_chunk(0, observation.revealed_tokens)
    assert state.pending_ready_bitmap == 1 and state.ready_bitmap == 0
    assert not runtime.binder.step(state).selected
    runtime.consume_event_ready(0, stage=1, ratio=1.0)
    step = runtime.schedule_and_commit()
    assert step.decision.accepted and state.full_rebuild_count == 0


def test_static_compiler_fails_closed_on_bad_group_mapping() -> None:
    topology = _topology(((0, 1),), nodes=2, groups=(((2,), 1.0),))
    try:
        StaticPlanCompiler().compile(topology)
        accepted = True
    except ValueError:
        accepted = False
    assert not accepted
