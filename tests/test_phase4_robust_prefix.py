"""RED contracts for Phase 4 robust prefix planning and residual repair.

Production imports are deliberately delayed.  Before Phase 4 implementation
exists this file must collect cleanly and fail inside test bodies, never during
module import.  The contracts use only synthetic public observations/scenarios;
they do not generate a formal corpus or write ``outputs/``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rlccl.envs.problem import TopologyInfo
from rlccl.uncertainty.execution import Proposal, TransferAction, commit_proposal
from rlccl.uncertainty.ambiguity import (
    AmbiguityConstructionView,
    build_empirical_ambiguity_set,
    fit_descriptor_normalizer,
    oracle_support_upper_bound,
    select_support,
)
from rlccl.uncertainty.problem import UncertainProblemInstance
from rlccl.uncertainty.reveal import DemandRevealProcess
from rlccl.uncertainty.scenarios import ScenarioSet, ScenarioTokenId
from rlccl.uncertainty.observation import PublicTopologyView, TruthTokenId


RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
LEGAL_CONFIGS = (
    (2, 1),
    (4, 1), (4, 2),
    (8, 1), (8, 2), (8, 4),
    (16, 1), (16, 2), (16, 4), (16, 8),
)
FORBIDDEN_IMPORTS = {
    "torch",
    "rlccl.envs.decoder",
    "rlccl.evaluation.partial_demand",
    "rlccl.evaluation.sequence_evaluator",
}


def _api() -> Any:
    robust = importlib.import_module("rlccl.scheduling.robust_prefix")
    recourse = importlib.import_module("rlccl.scheduling.recourse")
    adapter = importlib.import_module("rlccl.scheduling.scenario_adapter")
    experiment = importlib.import_module("rlccl.scheduling.phase4_experiment")
    return SimpleNamespace(robust=robust, recourse=recourse, adapter=adapter, experiment=experiment)


def _topology(
    *,
    edges: tuple[tuple[int, int], ...] | None = None,
    capacities: tuple[float, ...] | None = None,
    groups: tuple[tuple[tuple[int, ...], float], ...] = (),
    nodes: int = 4,
) -> TopologyInfo:
    if edges is None:
        edges = tuple(
            (source, destination)
            for source in range(nodes)
            for destination in range(nodes)
            if source != destination
        )
    if capacities is None:
        capacities = (4.0,) * len(edges)
    return TopologyInfo(
        nodes,
        len(edges),
        np.asarray(edges, dtype=np.int64),
        np.asarray(capacities, dtype=np.float64),
        list(groups),
        name="phase4-red",
    )


def _truth() -> np.ndarray:
    return np.asarray(
        [[0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2], [2, 0, 1, 0]],
        dtype=np.int64,
    )


def _episode(
    *, mode: str = "partial_shards", seed: int = 202608010,
    topology: TopologyInfo | None = None,
) -> tuple[UncertainProblemInstance, DemandRevealProcess]:
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=_truth(), topology_info=_topology() if topology is None else topology,
        time_limit=80, sequence_id="private-sequence", sequence_step=32,
        family="private-family", generator_metadata={"latent": "forbidden"},
    )
    return world, DemandRevealProcess(problem=world, mode=mode, ratios=RATIOS, seed=seed)


def _commit_first_revealed_direct(
    world: UncertainProblemInstance, observation: Any,
) -> Any:
    token = observation.revealed_tokens[0]
    source = int(token.holders[0])
    matches = np.flatnonzero(
        (np.asarray(world.topology_info.edge_src) == source)
        & (np.asarray(world.topology_info.edge_dst) == int(token.destination))
    )
    assert len(matches) == 1
    return commit_proposal(
        world, observation,
        Proposal.from_transfers((TransferAction(token.token_id, int(matches[0])),)),
    )


def _view(stage: int = 2, *, mode: str = "partial_shards") -> Any:
    api = _api()
    _, reveal = _episode(mode=mode)
    observation = reveal.observation_for_stage(stage)
    return api.robust.build_scheduling_view(observation)


def _support(view: Any, *, matrices: tuple[np.ndarray, ...] | None = None) -> Any:
    api = _api()
    if matrices is None:
        # Duplicate contents are legal: Phase 3B selects unique history indices,
        # not necessarily byte-distinct matrices.
        matrices = tuple(_truth().copy() for _ in range(8))
    return api.adapter.scenario_support_from_matrices(
        matrices=matrices,
        weights=(1.0 / len(matrices),) * len(matrices),
        method="boundary_scenarios",
        requested_k=8,
        uses_oracle=False,
        upper_bound_only=False,
        provenance={"normalizer_digest": "1" * 64},
        observation=view,
    )


def _history() -> tuple[np.ndarray, ...]:
    return tuple(_truth().copy() for _ in range(32))


def _ordinary_phase3b_ambiguity(observation: Any) -> Any:
    history = _history()
    normalizer = fit_descriptor_normalizer(history, observation.topology)
    view = AmbiguityConstructionView.from_observation(
        history_matrices=history,
        history_offsets=tuple(range(-32, 0)),
        observation=observation,
        construction_seed=203608010,
        normalizer=normalizer,
    )
    return build_empirical_ambiguity_set(
        view, calibration_radius=0.34327919716983946,
    )


def _ordinary_phase3b_support(observation: Any) -> Any:
    return select_support(
        _ordinary_phase3b_ambiguity(observation), method="boundary_scenarios", k=8,
    )


def _scientific_payload(rows: Any) -> tuple[dict[str, Any], ...]:
    identity = {
        "coordinate_id", "sequence_id", "family", "sequence_digest",
        "row_digest", "event_payload_digest",
    }
    return tuple({key: value for key, value in row.items() if key not in identity} for row in rows)


# 1. Sanitized observation, cadence, and executable-token boundary.
def test_scheduling_view_is_frozen_and_strips_private_identity_and_opaque_ids() -> None:
    api = _api()
    view = _view(2)
    assert is_dataclass(view)
    names = {field.name for field in fields(view)}
    assert {"stage", "ratio", "state_version", "revealed_tokens", "topology"} <= names
    assert not names & {"family", "sequence_id", "world", "manifest", "future_reveal"}
    assert all(hasattr(token, "local_ordinal") for token in view.revealed_tokens)
    assert all(not hasattr(token, "token_id") and not hasattr(token, "opaque_value") for token in view.revealed_tokens)
    with pytest.raises(FrozenInstanceError):
        view.stage = 4
    assert "SchedulingObservationView" in api.robust.__all__


def test_scheduling_view_arrays_are_owned_readonly_copies_of_sources_and_observation() -> None:
    api = _api()
    _, reveal = _episode(mode="source_destination_totals_first")
    original = reveal.observation_for_stage(2)
    source_observed = np.array(original.observed_matrix, copy=True)
    source_mask = np.array(original.entry_mask, copy=True)
    source_edges = np.array(original.topology.edges, copy=True)
    source_capacities = np.array(original.topology.capacities, copy=True)
    source_topology = PublicTopologyView(
        num_nodes=original.topology.num_nodes,
        num_edges=original.topology.num_edges,
        edges=source_edges,
        capacities=source_capacities,
        shared_constraints=original.topology.shared_constraints,
        name=original.topology.name,
    )
    observation = replace(
        original, observed_matrix=source_observed, entry_mask=source_mask,
        unknown_mask=~source_mask, topology=source_topology,
    )
    view = api.robust.build_scheduling_view(observation)
    arrays = [view.observed_matrix, view.entry_mask, view.topology.edges, view.topology.capacities]
    for name in ("source_totals", "destination_totals"):
        value = getattr(view, name)
        if value is not None:
            arrays.append(value)
    arrays.extend((view.topology.edge_src, view.topology.edge_dst))
    snapshots = [np.array(value, copy=True) for value in arrays]
    assert all(value.flags.owndata and not value.flags.writeable for value in arrays)
    assert all(not np.shares_memory(value, source_observed) for value in arrays)
    assert not np.shares_memory(view.topology.edges, source_edges)
    assert not np.shares_memory(view.topology.capacities, source_capacities)

    source_observed[0, 1] += 7
    source_mask[0, 1] = ~source_mask[0, 1]
    source_edges[0, 0] = 99
    source_capacities[0] = 99.0
    observation.observed_matrix.setflags(write=True)
    observation.observed_matrix[0, 1] += 11
    observation.topology.edges.setflags(write=True)
    observation.topology.edges[0, 0] = 88
    observation.topology.capacities.setflags(write=True)
    observation.topology.capacities[0] = 88.0
    assert all(np.array_equal(value, expected) for value, expected in zip(arrays, snapshots))
    for value in arrays:
        with pytest.raises(ValueError):
            value.reshape(-1)[0] = value.reshape(-1)[0]


def test_observation_for_stage_uses_l4_cadence_and_refreshes_state() -> None:
    api = _api()
    world, reveal = _episode()
    for slot in range(24):
        stage = min(slot // 4, 4)
        assert reveal.observation_for_stage(stage).stage == stage
    first = reveal.observation_for_stage(2)
    view = api.robust.build_scheduling_view(first)
    assert view.state_version == first.state_version
    assert first.revealed_tokens
    token = first.revealed_tokens[0]
    source = token.holders[0]
    matches = np.flatnonzero(
        (np.asarray(world.topology_info.edge_src) == source)
        & (np.asarray(world.topology_info.edge_dst) == token.destination)
    )
    assert len(matches) == 1
    commit_proposal(
        world,
        first,
        Proposal.from_transfers((TransferAction(token.token_id, int(matches[0])),)),
    )
    fresh = reveal.observation_for_stage(2)
    assert fresh is not first
    assert fresh.state_version == first.state_version + 1 == world._state_version
    fresh_token = next(item for item in fresh.revealed_tokens if item.token_id == token.token_id)
    assert token.destination in fresh_token.holders


def test_ratio_zero_never_produces_executable_candidate_or_transfer() -> None:
    api = _api()
    view = _view(0, mode="source_destination_totals_first")
    candidates = api.robust.enumerate_candidates(view)
    assert candidates == ()
    plan = api.robust.RobustPrefixPlanner(
        api.robust.RobustPrefixConfig(4, 2, 0.5, requested_k=8)
    ).plan(view, _support(view))
    assert plan.batches == ()


def test_scenario_and_unrevealed_tokens_cannot_be_bound_to_transfer_actions() -> None:
    api = _api()
    view = _view(1)
    scenario_id = ScenarioTokenId("scenario:only")
    with pytest.raises((TypeError, ValueError)):
        api.recourse.bind_action(view, local_token_ordinal=scenario_id, edge_index=0)
    hidden_ordinal = len(view.revealed_tokens) + 1
    with pytest.raises((IndexError, ValueError)):
        api.recourse.bind_action(view, local_token_ordinal=hidden_ordinal, edge_index=0)


@pytest.mark.parametrize(
    "uses_oracle,upper_bound_only",
    [(True, False), (False, True), (True, True)],
)
def test_ordinary_adapter_rejects_oracle_or_upper_bound_support(
    uses_oracle: bool, upper_bound_only: bool,
) -> None:
    api = _api()
    view = _view(2)
    with pytest.raises((TypeError, ValueError), match="oracle|upper|ordinary"):
        api.adapter.scenario_support_from_matrices(
            matrices=tuple(_truth().copy() for _ in range(8)),
            weights=(0.125,) * 8,
            method="oracle_support_upper_bound" if uses_oracle else "boundary_scenarios",
            requested_k=8,
            uses_oracle=uses_oracle,
            upper_bound_only=upper_bound_only,
            provenance={"normalizer_digest": "1" * 64},
            observation=view,
        )


def test_support_observation_digest_and_stage_are_bound_to_exact_view() -> None:
    api = _api()
    view2 = _view(2)
    view3 = _view(3)
    same_stage_other_observation = _view(2, mode="random_entries")
    support2 = _support(view2)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(4, 2, 0.5, 8))
    for mismatched in (view3, same_stage_other_observation):
        with pytest.raises((ValueError, api.robust.StaleSupportError), match="stage|observation|digest"):
            planner.plan(mismatched, support2)


def test_trusted_binding_rejects_cross_stage_state_and_revealed_tuple() -> None:
    api = _api()
    world, reveal = _episode()
    trusted2 = reveal.observation_for_stage(2)
    view2 = api.robust.build_scheduling_view(trusted2)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(4, 2, 0.0, 8))
    state = api.recourse.RecourseState.initial(planner.plan(view2, _support(view2)))
    trusted3 = reveal.observation_for_stage(3)
    altered = replace(trusted2, revealed_tokens=tuple(reversed(trusted2.revealed_tokens)))
    for mismatched in (trusted3, altered):
        with pytest.raises((ValueError, api.recourse.StalePlanError), match="stage|state|revealed|tuple|digest"):
            api.recourse.bind_first_batch(state, view2, trusted_observation=mismatched)
    _commit_first_revealed_direct(world, trusted2)
    stale_state = reveal.observation_for_stage(2)
    stale_view = api.robust.build_scheduling_view(stale_state)
    with pytest.raises((ValueError, api.recourse.StalePlanError), match="state|stale|version"):
        api.recourse.bind_first_batch(state, stale_view, trusted_observation=stale_state)


def test_opaque_token_alpha_renaming_cannot_change_ordinary_plan() -> None:
    api = _api()
    _, reveal = _episode(seed=101)
    obs_a = reveal.observation_for_stage(3)
    obs_b = replace(
        obs_a,
        revealed_tokens=tuple(
            replace(token, token_id=TruthTokenId(f"renamed-{index}"))
            for index, token in enumerate(obs_a.revealed_tokens)
        ),
    )
    view_a = api.robust.build_scheduling_view(obs_a)
    view_b = api.robust.build_scheduling_view(obs_b)
    assert [str(token.token_id) for token in obs_a.revealed_tokens] != [str(token.token_id) for token in obs_b.revealed_tokens]
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(8, 4, 0.5, 8))
    assert planner.plan(view_a, _support(view_a)).structural_actions == planner.plan(view_b, _support(view_b)).structural_actions


def test_real_paired_hidden_future_and_oracle_call_leave_ordinary_outputs_identical() -> None:
    api = _api()
    topology = _topology()
    truth_a = _truth()
    world_a = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth_a, topology_info=topology, time_limit=80,
        sequence_id="private-a", sequence_step=32, family="family-a",
        generator_metadata={"future": "a"},
    )
    reveal_a = DemandRevealProcess(
        problem=world_a, mode="random_entries", ratios=RATIOS, seed=1234,
    )
    observation_a = reveal_a.observation_for_stage(1)
    hidden_cells = np.argwhere(~np.asarray(observation_a.entry_mask) & ~np.eye(4, dtype=bool))
    assert len(hidden_cells)
    source, destination = map(int, hidden_cells[-1])
    truth_b = truth_a.copy()
    truth_b[source, destination] += 1
    world_b = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth_b, topology_info=topology, time_limit=80,
        sequence_id="private-b", sequence_step=32, family="family-b",
        generator_metadata={"future": "different"},
    )
    reveal_b = DemandRevealProcess(
        problem=world_b, mode="random_entries", ratios=RATIOS, seed=1234,
    )
    observation_b = reveal_b.observation_for_stage(1)
    assert np.array_equal(observation_a.entry_mask, observation_b.entry_mask)
    assert np.array_equal(observation_a.observed_matrix, observation_b.observed_matrix)
    view_a = api.robust.build_scheduling_view(observation_a)
    view_b = api.robust.build_scheduling_view(observation_b)
    assert (view_a.stage, view_a.ratio, view_a.state_version) == (
        view_b.stage, view_b.ratio, view_b.state_version,
    )
    assert np.array_equal(view_a.observed_matrix, view_b.observed_matrix)
    assert np.array_equal(view_a.entry_mask, view_b.entry_mask)
    assert view_a.revealed_tokens == view_b.revealed_tokens

    selected_a = _ordinary_phase3b_support(observation_a)
    selected_b = _ordinary_phase3b_support(observation_b)
    assert selected_a.to_canonical_bytes() == selected_b.to_canonical_bytes()
    support_a = api.adapter.scenario_support_from_selected(selected_a, observation=view_a)
    support_b = api.adapter.scenario_support_from_selected(selected_b, observation=view_b)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(8, 4, 0.5, 8))
    plan_a = planner.plan(view_a, support_a)
    plan_b = planner.plan(view_b, support_b)
    assert plan_a.structural_actions == plan_b.structural_actions
    row_a = api.experiment.run_ordinary_stage(
        view=view_a, support=support_a, plan=plan_a, method="scenario_robust_prefix",
    )
    row_b = api.experiment.run_ordinary_stage(
        view=view_b, support=support_b, plan=plan_b, method="scenario_robust_prefix",
    )
    assert row_a == row_b

    run_a = api.experiment.run_ordinary_prefix(
        world=world_a, reveal_process=reveal_a, history_matrices=_history(),
        method="scenario_robust_prefix", config=planner.config, stop_before_slot=8,
    )
    run_b = api.experiment.run_ordinary_prefix(
        world=world_b, reveal_process=reveal_b, history_matrices=_history(),
        method="scenario_robust_prefix", config=planner.config, stop_before_slot=8,
    )
    assert _scientific_payload(run_a.scientific_rows) == _scientific_payload(run_b.scientific_rows)

    ambiguity = _ordinary_phase3b_ambiguity(observation_a)
    oracle_support_upper_bound(ambiguity, truth=truth_b, k=8)
    selected_after_oracle = _ordinary_phase3b_support(observation_a)
    assert selected_a.to_canonical_bytes() == selected_after_oracle.to_canonical_bytes()
    support_after = api.adapter.scenario_support_from_selected(selected_after_oracle, observation=view_a)
    plan_after = planner.plan(view_a, support_after)
    assert plan_a.structural_actions == plan_after.structural_actions
    assert row_a == api.experiment.run_ordinary_stage(
        view=view_a, support=support_after, plan=plan_after, method="scenario_robust_prefix",
    )


# 2. Capacity, canonical paths, candidates, residual scenarios, and Q.
@pytest.mark.parametrize("raw,expected", [(0.5, 0), (1.0, 1), (1.5, 1), (3.9, 3)])
def test_atomic_capacity_units_are_floored(raw: float, expected: int) -> None:
    assert _api().robust.atomic_units(raw) == expected


def test_fractional_negative_and_nonfinite_capacities_fail_closed() -> None:
    api = _api()
    for raw in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            api.robust.atomic_units(raw)


def test_destination_first_bfs_tie_is_not_edge_index_first() -> None:
    api = _api()
    # edge 9 -> dst1 must beat edge 2 -> dst3 despite its larger edge index.
    edges = ((3, 2), (2, 0), (0, 3), (1, 2), (2, 3), (3, 0),
             (1, 3), (2, 1), (3, 1), (0, 1), (1, 2))
    topology = _topology(edges=edges, capacities=(1.0,) * len(edges))
    assert api.robust.canonical_shortest_path(topology, 0, 2)[0] == 9


def test_candidate_strictly_reduces_global_best_holder_distance_and_is_unique_per_token() -> None:
    api = _api()
    candidates = api.robust.enumerate_candidates(_view(4))
    assert len({candidate.local_token_ordinal for candidate in candidates}) <= len(candidates)
    assert all(candidate.after_distance == candidate.before_distance - 1 for candidate in candidates)
    packed = api.robust.pack_candidate_batch(candidates, _view(4).topology)
    assert len({action.local_token_ordinal for action in packed}) == len(packed)


def test_overlapping_shared_groups_are_both_enforced_during_packing() -> None:
    api = _api()
    topology = _topology(
        nodes=4, edges=((0, 1), (1, 2), (2, 3)), capacities=(1.0, 1.0, 1.0),
        groups=(((0, 1), 1.0), ((1, 2), 1.0)),
    )
    candidates = (
        api.robust.CandidateAction(0, 0, 3, 2),
        api.robust.CandidateAction(1, 2, 2, 1),
        api.robust.CandidateAction(2, 1, 2, 1),
    )
    packed = api.robust.pack_candidate_batch(candidates, topology)
    assert [(item.local_token_ordinal, item.edge_index) for item in packed] == [(0, 0), (1, 2)]
    assert not api.robust.can_add_candidate((candidates[0],), candidates[2], topology)
    assert not api.robust.can_add_candidate((candidates[1],), candidates[2], topology)
    assert all(api.robust.batch_loads((item,), topology).edge_units[item.edge_index] == 1 for item in candidates)


def test_multi_holder_local_path_progress_is_rejected_without_global_minimum_improvement() -> None:
    api = _api()
    topology = _topology(nodes=4, edges=((0, 1), (1, 2), (2, 3)), capacities=(1.0,) * 3)
    distances = api.robust.candidate_distances(
        topology, holders=(0, 2), destination=3, edge_index=0,
    )
    assert (distances.source_distance, distances.edge_target_distance) == (3, 2)
    assert (distances.before_global, distances.after_global) == (1, 1)
    assert not api.robust.is_strict_global_progress(
        topology, holders=(0, 2), destination=3, edge_index=0,
    )


def test_positive_unreachable_scenario_fails_closed() -> None:
    api = _api()
    topology = _topology(nodes=4, edges=((0, 1),), capacities=(1.0,))
    _, reveal = _episode(topology=topology)
    view = api.robust.build_scheduling_view(reveal.observation_for_stage(0))
    scenario = np.zeros((4, 4), dtype=np.int64)
    scenario[0, 2] = 1
    with pytest.raises(api.robust.UnreachableScenarioError):
        api.robust.project_residual_scenario(scenario, view)


def test_residual_projection_exact_chain_vectors_after_move_and_completion() -> None:
    api = _api()
    topology = _topology(
        nodes=3, edges=((0, 1), (1, 2)), capacities=(1.0, 1.0),
        groups=(((0, 1), 2.0),),
    )
    truth = np.asarray([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth, topology_info=topology, time_limit=80,
        sequence_id="residual", sequence_step=32, family="private", generator_metadata={},
    )
    reveal = DemandRevealProcess(problem=world, mode="partial_shards", ratios=RATIOS, seed=5)
    first = reveal.observation_for_stage(4)
    token = first.revealed_tokens[0].token_id
    assert commit_proposal(world, first, Proposal.from_transfers((TransferAction(token, 0),))).legal
    moved_view = api.robust.build_scheduling_view(reveal.observation_for_stage(4))
    moved = api.robust.project_residual_scenario(truth, moved_view)
    assert np.array_equal(moved.edge_loads, np.asarray([0.0, 1.0]))
    assert np.array_equal(moved.group_loads, np.asarray([1.0]))
    second = reveal.observation_for_stage(4)
    assert commit_proposal(world, second, Proposal.from_transfers((TransferAction(token, 1),))).legal
    completed = api.robust.project_residual_scenario(
        truth, api.robust.build_scheduling_view(reveal.observation_for_stage(4))
    )
    assert np.array_equal(completed.edge_loads, np.asarray([0.0, 0.0]))
    assert np.array_equal(completed.group_loads, np.asarray([0.0]))


@pytest.mark.parametrize("scenario", [np.zeros((3, 3), dtype=np.int64), np.asarray([[0, 0, -1], [0, 0, 0], [0, 0, 0]])])
def test_residual_projection_rejects_insufficient_or_negative_load(scenario: np.ndarray) -> None:
    api = _api()
    topology = _topology(nodes=3, edges=((0, 1), (1, 2)), capacities=(1.0, 1.0))
    truth = np.asarray([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth, topology_info=topology, time_limit=80,
        sequence_id="insufficient", sequence_step=32, family="private", generator_metadata={},
    )
    reveal = DemandRevealProcess(problem=world, mode="partial_shards", ratios=RATIOS, seed=7)
    view = api.robust.build_scheduling_view(reveal.observation_for_stage(4))
    with pytest.raises((ValueError, api.robust.InconsistentResidualError), match="negative|insufficient|revealed"):
        api.robust.project_residual_scenario(scenario, view)


def test_weighted_population_robust_score_has_frozen_sign_and_tie_tolerance() -> None:
    api = _api()
    q = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    mean = float(weights @ q)
    std = float(np.sqrt(weights @ ((q - mean) ** 2)))
    assert api.robust.robust_score(q, weights, risk_lambda=0.5) == pytest.approx(mean - 0.5 * std)
    assert api.robust.SCORE_TIE_TOLERANCE == 1e-12


def test_score_normalizes_positive_weights_and_rejects_invalid_numeric_inputs() -> None:
    api = _api()
    q = np.asarray([1.0, 2.0, 4.0])
    normalized = np.asarray([0.2, 0.3, 0.5])
    assert api.robust.robust_score(q, np.asarray([2.0, 3.0, 5.0]), risk_lambda=0.5) == pytest.approx(
        api.robust.robust_score(q, normalized, risk_lambda=0.5)
    )
    for weights in (np.zeros(3), np.asarray([1.0, -1.0, 1.0]), np.asarray([1.0, np.nan, 1.0])):
        with pytest.raises(ValueError):
            api.robust.robust_score(q, weights, risk_lambda=0.5)
    for invalid_q in (np.asarray([1.0, np.nan, 2.0]), np.asarray([1.0, np.inf, 2.0])):
        with pytest.raises(ValueError):
            api.robust.robust_score(invalid_q, normalized, risk_lambda=0.5)
    for criticality, distance in ((np.nan, 1.0), (1.0, np.inf)):
        with pytest.raises(ValueError):
            api.robust.candidate_utility(criticality=criticality, after_distance=distance)


def test_criticality_and_q_match_explicit_hand_calculation() -> None:
    api = _api()
    # edge load 4/cap2 plus two overlapping group terms 6/3 and 3/1 = 7.
    criticality = api.robust.criticality_for_edge(
        residual_edge_load=4.0,
        edge_units=2,
        incident_group_loads=(6.0, 3.0),
        incident_group_units=(3, 1),
    )
    assert criticality == pytest.approx(7.0)
    assert api.robust.candidate_utility(criticality=criticality, after_distance=2) == pytest.approx(
        1.0 + 0.25 * 7.0 - 0.05 * 2.0
    )


def test_projected_scenario_load_is_static_within_one_horizon_simulation() -> None:
    api = _api()
    view = _view(3)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(8, 4, 0.5, 8))
    plan = planner.plan(view, _support(view))
    assert plan.simulation_trace
    assert len({step.projected_load_digest for step in plan.simulation_trace}) == 1


def test_ratio_one_k8_support_is_actual_k1_with_zero_scenario_std() -> None:
    api = _api()
    view = _view(4)
    support = _support(view, matrices=(_truth(),))
    assert support.requested_k == 8 and support.actual_k == 1
    scores = api.robust.score_candidates(api.robust.enumerate_candidates(view), support, risk_lambda=1.0)
    assert all(score.scenario_std == 0.0 for score in scores)


# 3. Prefix immutability, current-state repair, and unchanged checker.
@pytest.mark.parametrize("horizon,prefix", LEGAL_CONFIGS)
def test_exact_prefix_config_allowlist(horizon: int, prefix: int) -> None:
    api = _api()
    config = api.robust.RobustPrefixConfig(horizon, prefix, 0.5, requested_k=8)
    assert (config.horizon, config.prefix) == (horizon, prefix)


@pytest.mark.parametrize("horizon,prefix", [(2, 2), (4, 4), (8, 8), (2, 4), (3, 1)])
def test_nonallowlisted_prefix_config_is_rejected(horizon: int, prefix: int) -> None:
    with pytest.raises(ValueError):
        _api().robust.RobustPrefixConfig(horizon, prefix, 0.5, requested_k=8)


def test_prefix_plan_is_frozen_state_bound_and_keeps_at_most_p_batches() -> None:
    api = _api()
    view = _view(3)
    config = api.robust.RobustPrefixConfig(16, 8, 0.5, 8)
    plan = api.robust.RobustPrefixPlanner(config).plan(view, _support(view))
    assert is_dataclass(plan) and len(plan.batches) <= 8
    assert plan.origin_stage == view.stage and plan.origin_state_version == view.state_version
    assert len(plan.support_digest) == len(plan.config_digest) == 64
    with pytest.raises(FrozenInstanceError):
        plan.revision = 99


def test_real_commit_then_repair_preserves_holders_ledger_and_only_discards_suffix() -> None:
    api = _api()
    world, reveal = _episode()
    trusted = reveal.observation_for_stage(3)
    view = api.robust.build_scheduling_view(trusted)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(8, 4, 0.5, 8))
    state = api.recourse.RecourseState.initial(planner.plan(view, _support(view)))
    assert state.plan.batches
    original_suffix_actions = sum(len(batch) for batch in state.plan.batches[1:])
    proposal = api.recourse.bind_first_batch(state, view, trusted_observation=trusted)
    committed_tokens = tuple(action.token_id for action in proposal.actions)
    result = commit_proposal(world, trusted, proposal)
    assert result.legal and result.state_version == trusted.state_version + 1
    fresh = reveal.observation_for_stage(4)
    fresh_view = api.robust.build_scheduling_view(fresh)
    committed = api.recourse.record_committed_batch(
        state, proposal=proposal, commit_result=result, fresh_observation=fresh,
    )
    assert committed.executed_actions
    assert len(committed.executed_actions) == len(proposal.actions)
    assert {identity.truth_token_id for identity in committed.executed_actions} == set(committed_tokens)
    for token_id in committed_tokens:
        token = next(item for item in fresh.revealed_tokens if item.token_id == token_id)
        assert len(token.holders) >= 2
    repaired = api.recourse.repair_prefix(
        committed, fresh_view, _support(fresh_view), planner, reason="reveal",
    )
    assert tuple(repaired.executed_actions) == tuple(committed.executed_actions)
    assert repaired.plan.revision == committed.plan.revision + 1
    assert repaired.discarded_actions == original_suffix_actions
    assert repaired.plan.origin_state_version == result.state_version
    assert repaired.plan.origin_stage == 4
    assert repaired.execution_start_state_version == state.plan.origin_state_version
    assert repaired.reason == "reveal"


def test_state_version_break_and_checker_rejection_stop_illegal_without_fallback() -> None:
    api = _api()
    world, reveal = _episode()
    trusted = reveal.observation_for_stage(4)
    view = api.robust.build_scheduling_view(trusted)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(4, 2, 0.0, 8))
    state = api.recourse.RecourseState.initial(planner.plan(view, _support(view)))
    _commit_first_revealed_direct(world, trusted)
    stale_outcome = api.recourse.execute_first_batch(
        world=world, state=state, view=view, trusted_observation=trusted,
    )
    assert not stale_outcome.legality and stale_outcome.stopped
    assert stale_outcome.illegal_reason == "stale_observation"
    assert stale_outcome.fallback_delta == stale_outcome.no_common_delta == 0
    assert stale_outcome.commit_result is None


def test_stale_revision_is_rejected_before_checker_and_cannot_be_fallback_masked() -> None:
    api = _api()
    view = _view(2)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(4, 2, 0.0, 8))
    state = api.recourse.RecourseState.initial(planner.plan(view, _support(view)))
    stale = replace(state.plan, revision=state.plan.revision - 1)
    with pytest.raises(api.recourse.StalePlanError):
        api.recourse.bind_first_batch(replace(state, plan=stale), view)


def test_wait_latch_counts_fallback_only_once_until_stage_or_state_changes() -> None:
    api = _api()
    view = _view(0)
    state = api.recourse.RecourseState.waiting(view)
    first = api.recourse.enter_wait_latch(state, view)
    second = api.recourse.enter_wait_latch(first.state, view)
    assert (first.no_common_delta, first.fallback_delta) == (1, 1)
    assert (second.no_common_delta, second.fallback_delta) == (0, 0)


@pytest.mark.parametrize(
    "new_stage,exhausted,precondition_valid,expected",
    [
        (True, True, False, "reveal"),
        (False, True, False, "exhaustion"),
        (False, False, False, "invalidation"),
        (False, False, True, "continue"),
    ],
)
def test_prefix_transition_priority_is_mutually_exclusive(
    new_stage: bool, exhausted: bool, precondition_valid: bool, expected: str,
) -> None:
    api = _api()
    transition = api.recourse.choose_transition(
        new_stage=new_stage,
        prefix_exhausted=exhausted,
        precondition_valid=precondition_valid,
    )
    assert transition.reason == expected
    deltas = (transition.reveal_delta, transition.exhaustion_delta, transition.invalidation_delta)
    assert sum(deltas) == (0 if expected == "continue" else 1)


def test_empty_plan_enters_latch_once_even_when_transition_conditions_overlap() -> None:
    api = _api()
    view = _view(0)
    state = api.recourse.RecourseState.waiting(view)
    first = api.recourse.enter_wait_latch(state, view)
    repeated = api.recourse.enter_wait_latch(first.state, view)
    assert first.no_common_delta == first.fallback_delta == 1
    assert repeated.no_common_delta == repeated.fallback_delta == 0


def test_bound_batch_uses_fresh_truth_ids_and_authoritative_commit() -> None:
    api = _api()
    world, reveal = _episode()
    full = reveal.observation_for_stage(4)
    view = api.robust.build_scheduling_view(full)
    planner = api.robust.RobustPrefixPlanner(api.robust.RobustPrefixConfig(4, 2, 0.0, 8))
    state = api.recourse.RecourseState.initial(planner.plan(view, _support(view)))
    proposal = api.recourse.bind_first_batch(state, view, trusted_observation=full)
    assert isinstance(proposal, Proposal)
    assert all(isinstance(action, TransferAction) for action in proposal.actions)
    result = commit_proposal(world, full, proposal)
    assert result.legal


def test_executed_action_identity_cannot_replay_but_token_can_continue_on_new_edge() -> None:
    api = _api()
    identity = api.recourse.ExecutedActionIdentity("binding", 3, 1, 0)
    state = api.recourse.ExecutedLedger.empty().append(identity)
    with pytest.raises(api.recourse.ReplayedActionError):
        state.append(identity)
    assert state.can_append(api.recourse.ExecutedActionIdentity("binding", 4, 2, 0))


def test_same_truth_token_commits_across_two_chain_edges_but_exact_action_replay_fails() -> None:
    api = _api()
    topology = _topology(nodes=3, edges=((0, 1), (1, 2)), capacities=(1.0, 1.0))
    truth = np.asarray([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    world = UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth, topology_info=topology, time_limit=80,
        sequence_id="chain", sequence_step=32, family="private", generator_metadata={},
    )
    reveal = DemandRevealProcess(problem=world, mode="partial_shards", ratios=RATIOS, seed=17)
    first_observation = reveal.observation_for_stage(4)
    token = first_observation.revealed_tokens[0].token_id
    first = Proposal.from_transfers((TransferAction(token, 0),))
    assert commit_proposal(world, first_observation, first).legal
    second_observation = reveal.observation_for_stage(4)
    second = Proposal.from_transfers((TransferAction(token, 1),))
    assert commit_proposal(world, second_observation, second).legal
    replay_observation = reveal.observation_for_stage(4)
    with pytest.raises(ValueError, match="destination|already|replay"):
        commit_proposal(world, replay_observation, first)
    identity = api.recourse.ExecutedActionIdentity(str(token), 0, 0, 0)
    ledger = api.recourse.ExecutedLedger.empty().append(identity)
    with pytest.raises(api.recourse.ReplayedActionError):
        ledger.append(identity)


def test_phase4_modules_do_not_import_legacy_decoder_torch_or_old_partial_evaluator() -> None:
    _api()
    for module_name in (
        "rlccl.scheduling.robust_prefix", "rlccl.scheduling.recourse",
        "rlccl.scheduling.scenario_adapter",
    ):
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None
        source = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports |= {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(any(name == banned or name.startswith(banned + ".") for banned in FORBIDDEN_IMPORTS) for name in imports)


def test_ordinary_planner_signature_has_no_private_capabilities() -> None:
    api = _api()
    signature = inspect.signature(api.robust.RobustPrefixPlanner.plan)
    forbidden = {"world", "problem", "truth", "manifest", "reveal_process", "family", "sequence_id", "oracle"}
    assert not forbidden & set(signature.parameters)
