"""RED contracts for the Phase 4 paired experiment and artifact validator.

The tests are synthetic and safe: production imports occur only inside test
bodies, no formal traffic corpus is generated, and the official output path is
never created.  Missing Phase 4 modules therefore produce genuine RED failures
without collection errors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import copy
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest


FAMILIES = (
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
)
BASE_SEEDS = (642, 742, 842)
SPLITS = ("fit", "validation", "test")
CHECKPOINTS = (32, 96, 160, 224)
MODES = (
    "random_entries", "source_totals_first", "source_destination_totals_first",
    "partial_shards", "time_based_arrival",
)
RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = (
    "full_information_lower_bound",
    "full_information_executable_reference",
    "wait_until_known",
    "partial_current_only",
    "long_term_mean_point_plan",
    "previous_value_point_plan",
    "h1_best_point_plan",
    "scenario_robust_prefix",
    "oracle_scenario_robust_reference",
)
VALIDATION_METHODS = (
    "scenario_robust_prefix", "wait_until_known", "partial_current_only",
)
LEGAL_HP = (
    (2, 1), (4, 1), (4, 2), (8, 1), (8, 2),
    (8, 4), (16, 1), (16, 2), (16, 4), (16, 8),
)
COMPONENTS = (
    "h1_inference", "ambiguity_construction", "support_selection",
    "prefix_synthesis", "recourse_repair", "fallback", "checker_commit",
    "unattributed",
)
ARTIFACTS = (
    "manifest.json", "h1_best_point_model.json", "raw_validation_metrics.csv",
    "raw_test_episode_metrics.csv", "raw_test_sequence_metrics.csv",
    "raw_test_execution_events.csv", "raw_timing_metrics.csv", "summary.json",
)
COMMON_COLUMNS = (
    "schema_version", "split", "coordinate_id", "sequence_id", "family", "base_seed",
    "sequence_digest", "checkpoint", "checkpoint_index", "reveal_mode", "mode_index",
    "reveal_seed", "topology_digest", "config_digest", "method", "role", "uses_oracle",
    "executable",
)
VALIDATION_COLUMNS = COMMON_COLUMNS + (
    "horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max", "risk_lambda",
    "completion_slots", "total_online_ns", "end_to_end_latency_ms", "legality",
    "discrete_timeout", "wall_timeout", "row_digest",
)
EPISODE_COLUMNS = COMMON_COLUMNS + (
    "reference_kind", "horizon", "prefix", "requested_k", "actual_k_min", "actual_k_max",
    "risk_lambda", "completion_slots", "lower_bound_slots", "oracle_regret_slots",
    "total_online_ns", "runner_wall_ns", "end_to_end_latency_ms", "end_to_end_regret_ms",
    "first_action_slot", "reveal_lead_lag_slots", "prefix_planned_batches",
    "prefix_planned_actions", "prefix_executed_batches", "prefix_executed_actions",
    "discarded_unexecuted_batches", "discarded_unexecuted_actions", "wasted_executed_actions",
    "wasted_unexecuted_actions", "reveal_replan_events", "exhaustion_replan_events",
    "invalidation_replan_events", "true_replan_events", "residual_repair_actions",
    "no_common_action_events", "fallback_events", "unreachable_od_count", "legality",
    "illegal_reason", "discrete_timeout", "wall_timeout", "row_digest",
)
EVENT_COLUMNS = COMMON_COLUMNS + (
    "event_index", "slot", "stage", "state_version_before", "state_version_after",
    "plan_revision", "event_kind", "reason", "observation_digest", "residual_state_digest",
    "support_digest", "requested_k", "actual_k", "batch_index", "batch_count",
    "action_count", "local_token_ordinal", "truth_binding_digest", "edge_index",
    "before_distance", "after_distance", "commit_legal", "elapsed_ns",
    "event_payload_digest", "row_digest",
)
ZERO_DIGEST = "0" * 64
FORMAL_OUTPUT = Path("outputs/phase4_early_planning")
H1_MANIFEST_SHA256 = "C702D8CEA33BCEC805FA0AB4B1EEA58C7E0BCBF6AAEF697E01523BB86D65B48C"
PHASE3B_MANIFEST_SHA256 = "DF8218052A635A683CE0CA848BB31171C740A4FC9C8E31DDB764BB60F2DEE527"


def _api() -> Any:
    experiment = importlib.import_module("rlccl.scheduling.phase4_experiment")
    robust = importlib.import_module("rlccl.scheduling.robust_prefix")
    return SimpleNamespace(experiment=experiment, robust=robust)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coordinate(api: Any, *, sequence_digest: str = "1" * 64, checkpoint: int = 32,
                mode: str = MODES[0], reveal_seed: int = 202608010) -> str:
    return api.coordinate_id(
        sequence_digest=sequence_digest, checkpoint=checkpoint, reveal_mode=mode,
        reveal_seed=reveal_seed, topology_digest="2" * 64, ratios=RATIOS,
        cadence_slots=4, time_limit=80, checker_version="phase1-atomic-v1",
    )


def _event(api: Any, *, kind: str, index: int, reason: str = "NONE", slot: int = -1,
           stage: int = -1, method: str = "scenario_robust_prefix", **overrides: Any) -> dict[str, Any]:
    row = {
        "schema_version": 1, "split": "test", "coordinate_id": _coordinate(api),
        "sequence_id": "p4-toy", "family": FAMILIES[0], "base_seed": 642,
        "sequence_digest": "1" * 64, "checkpoint": 32, "checkpoint_index": 0,
        "reveal_mode": MODES[0], "mode_index": 0, "reveal_seed": 202608010,
        "topology_digest": "2" * 64, "config_digest": "3" * 64,
        "method": method, "role": "ordinary", "uses_oracle": False,
        "executable": True, "event_index": index, "slot": slot, "stage": stage,
        "state_version_before": -1, "state_version_after": -1,
        "plan_revision": -1, "event_kind": kind, "reason": reason,
        "observation_digest": ZERO_DIGEST, "residual_state_digest": ZERO_DIGEST,
        "support_digest": ZERO_DIGEST, "requested_k": 0, "actual_k": 0,
        "batch_index": -1, "batch_count": 0,
        "action_count": 0, "local_token_ordinal": -1,
        "truth_binding_digest": ZERO_DIGEST, "edge_index": -1,
        "before_distance": -1, "after_distance": -1, "commit_legal": False,
        "elapsed_ns": 0,
    }
    row.update(overrides)
    row["event_payload_digest"] = api.event_payload_digest(row)
    row["row_digest"] = api.row_digest(row)
    return row


def _minimal_ledger(api: Any) -> list[dict[str, Any]]:
    return [
        _event(api, kind="episode_start", index=0),
        _event(api, kind="episode_end", index=1, reason="discrete_timeout", slot=80, stage=4),
    ]


def _committed_action_ledger(api: Any) -> list[dict[str, Any]]:
    common = {
        "slot": 16, "stage": 4, "state_version_before": 0,
        "state_version_after": 0, "plan_revision": 0,
        "observation_digest": "4" * 64, "residual_state_digest": "5" * 64,
        "support_digest": "6" * 64, "batch_index": 0, "batch_count": 1,
        "action_count": 2,
    }
    return [
        _event(api, kind="episode_start", index=0),
        _event(
            api, kind="plan_built", index=1, reason="initial",
            requested_k=8, actual_k=1, **common,
        ),
        _event(api, kind="proposal_bound", index=2, **common),
        _event(
            api, kind="action_committed", index=3, local_token_ordinal=0,
            truth_binding_digest="a" * 64, edge_index=3, before_distance=2,
            after_distance=1, commit_legal=True, **{**common, "action_count": 1},
        ),
        _event(
            api, kind="action_committed", index=4, local_token_ordinal=1,
            truth_binding_digest="b" * 64, edge_index=4, before_distance=3,
            after_distance=2, commit_legal=True, **{**common, "action_count": 1},
        ),
        _event(api, kind="batch_committed", index=5, **{**common, "state_version_after": 1}),
        _event(api, kind="episode_end", index=6, reason="complete", slot=17, stage=4),
    ]


def _plain_gate_evidence() -> dict[str, Any]:
    return {
        "paired": {
            "wait": {"mean_e2e_delta": 2.0, "ci_lower": 0.5, "mean_cvar95_delta": -0.1,
                     "robust_discrete_timeout_rate": 0.01, "comparator_discrete_timeout_rate": 0.02,
                     "robust_wall_timeout_rate": 0.01, "comparator_wall_timeout_rate": 0.02},
            "partial": {"mean_e2e_delta": 1.0, "ci_lower": 0.25, "mean_cvar95_delta": -0.05,
                        "robust_discrete_timeout_rate": 0.01, "comparator_discrete_timeout_rate": 0.02,
                        "robust_wall_timeout_rate": 0.01, "comparator_wall_timeout_rate": 0.02},
        },
        "base_seed_deltas": {642: 1.0, 742: 0.5, 842: 0.25},
        "family_deltas": {family: 0.5 for family in FAMILIES},
        "family_relative_degradation": {family: 0.0 for family in FAMILIES},
        "legality_rates": {method: 1.0 for method in METHODS if method != "full_information_lower_bound"},
        "scheduling_only_delta": 3.0,
        "end_to_end_delta": 1.0,
        "overhead_included": True,
        "test_sequence_count": 15,
        "test_episode_count": 2_700,
        "fresh_exclusion_complete": True,
        "capability_isolation_complete": True,
        "artifact_chain_complete": True,
        "focused_tests_complete": True,
        "environment_complete": True,
        "gate_status": "PENDING_SUPERVISOR",
    }


def _plain_validation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence_index in range(15):
        sequence_id = f"plain-validation-{sequence_index}"
        for coordinate_index in range(20):
            coordinate_id = _sha(f"{sequence_id}:{coordinate_index}")
            for horizon, prefix in LEGAL_HP:
                for risk_lambda in (0.0, 0.5, 1.0):
                    rows.append({
                        "sequence_id": sequence_id, "coordinate_id": coordinate_id,
                        "method": "scenario_robust_prefix", "horizon": horizon,
                        "prefix": prefix, "requested_k": 8, "actual_k_min": 1,
                        "actual_k_max": 8, "risk_lambda": risk_lambda,
                        "end_to_end_latency_ms": 10.0, "total_online_ns": 100,
                    })
            for method in ("wait_until_known", "partial_current_only"):
                rows.append({
                    "sequence_id": sequence_id, "coordinate_id": coordinate_id,
                    "method": method, "horizon": 0, "prefix": 0, "requested_k": 0,
                    "actual_k_min": 0, "actual_k_max": 0, "risk_lambda": 0.0,
                    "end_to_end_latency_ms": 12.0, "total_online_ns": 0,
                })
    assert len(rows) == 9_600
    return rows


def _checker_rejected_ledger(api: Any, *, slot: int = 6) -> list[dict[str, Any]]:
    stable = {
        "slot": slot, "stage": 1, "state_version_before": 4,
        "state_version_after": 4, "plan_revision": 2,
        "observation_digest": "4" * 64, "residual_state_digest": "5" * 64,
        "support_digest": "6" * 64, "batch_index": 0, "batch_count": 1,
        "action_count": 1,
    }
    return [
        _event(api, kind="episode_start", index=0),
        _event(api, kind="proposal_bound", index=1, **stable),
        _event(api, kind="checker_rejected", index=2, reason="source_possession", **stable),
        _event(api, kind="episode_end", index=3, reason="illegal", slot=slot, stage=1),
    ]


def _plan_k_ledger(
    api: Any,
    *,
    method: str,
    plans: tuple[tuple[int, int, int], ...],
    role: str = "ordinary",
    uses_oracle: bool = False,
    executable: bool = True,
) -> list[dict[str, Any]]:
    rows = [
        _event(
            api, kind="episode_start", index=0, method=method,
            role=role, uses_oracle=uses_oracle, executable=executable,
        )
    ]
    for revision, (stage, requested_k, actual_k) in enumerate(plans):
        rows.append(_event(
            api, kind="plan_built", index=len(rows), reason="initial" if revision == 0 else "reveal",
            slot=stage * 4, stage=stage, method=method, role=role, uses_oracle=uses_oracle,
            executable=executable, state_version_before=0, state_version_after=0,
            plan_revision=revision, observation_digest=f"{stage + 1:x}" * 64,
            residual_state_digest=f"{stage + 5:x}" * 64,
            support_digest=f"{stage + 9:x}" * 64, requested_k=requested_k,
            actual_k=actual_k, batch_count=0, action_count=0,
        ))
    end_reason = "lower_bound_timeout" if not executable else "discrete_timeout"
    rows.append(_event(
        api, kind="episode_end", index=len(rows), reason=end_reason, slot=80, stage=4,
        method=method, role=role, uses_oracle=uses_oracle, executable=executable,
    ))
    return rows


def _rewrite_csv_row(
    api: Any,
    path: Path,
    changes: Mapping[str, Any],
    *,
    recompute_event_payload: bool = False,
    recompute_row_digest: bool = False,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    assert rows and set(changes) <= set(fieldnames)
    candidates = [
        index for index, row in enumerate(rows)
        if row.get("method") == "scenario_robust_prefix"
        and all(
            key not in {"observation_digest", "residual_state_digest", "support_digest"}
            or row.get(key) != ZERO_DIGEST
            for key in changes
        )
    ]
    assert candidates, f"toy artifact lacks a linked target row for {changes}"
    target = candidates[0]
    rows[target].update({key: str(value) for key, value in changes.items()})
    if recompute_event_payload:
        parsed = api.parse_csv_row(path.name, rows[target])
        rows[target]["event_payload_digest"] = api.event_payload_digest(parsed)
    if recompute_row_digest:
        parsed = api.parse_csv_row(path.name, rows[target])
        rows[target]["row_digest"] = api.row_digest(parsed)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _assert_acyclic(graph: Mapping[str, set[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise AssertionError(f"cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, set()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _scientific_without_order_identity_or_timing(value: Any) -> Any:
    volatile = {
        "coordinate_id", "sequence_id", "family", "base_seed", "sequence_digest",
        "reveal_seed", "config_digest", "method_order", "method_order_index",
        "elapsed_ns", "total_online_ns", "runner_wall_ns", "end_to_end_latency_ms",
        "end_to_end_regret_ms", "row_digest", "event_payload_digest",
    }
    if isinstance(value, Mapping):
        return {
            key: _scientific_without_order_identity_or_timing(item)
            for key, item in sorted(value.items()) if key not in volatile
        }
    if isinstance(value, (tuple, list)):
        return tuple(_scientific_without_order_identity_or_timing(item) for item in value)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, tuple(value.shape), tuple(value.reshape(-1).tolist()))
    return value


# 4. Fresh corpus, seeds, paired identities, and method registry.
def test_formal_spec_registry_is_exact_fresh45_without_generation() -> None:
    api = _api().experiment
    specs = api.build_formal_sequence_specs()
    assert len(specs) == 45
    assert {spec.family for spec in specs} == set(FAMILIES)
    assert {spec.base_seed for spec in specs} == set(BASE_SEEDS)
    assert {spec.split for spec in specs} == set(SPLITS)
    assert len({spec.sequence_id for spec in specs}) == len(specs)
    first = specs[0]
    assert first.record_index == 0 and first.actual_seed == 642
    assert first.sequence_length == 256


def test_reveal_and_construction_seed_formula_is_exact() -> None:
    api = _api().experiment
    assert api.phase4_seeds(record_index=0, checkpoint_index=0, mode_index=0) == (202608010, 203608010)
    assert api.phase4_seeds(record_index=44, checkpoint_index=3, mode_index=4) == (202652044, 203652044)


def test_coordinate_id_uses_addendum_fields_and_excludes_method_or_truth() -> None:
    api = _api().experiment
    first = _coordinate(api)
    assert len(first) == 64
    assert first == _coordinate(api)
    assert first != _coordinate(api, checkpoint=96)
    signature = inspect.signature(api.coordinate_id)
    assert set(signature.parameters) == {
        "sequence_digest", "checkpoint", "reveal_mode", "reveal_seed",
        "topology_digest", "ratios", "cadence_slots", "time_limit", "checker_version",
    }


def test_config_digest_uses_requested_not_actual_k_and_all_frozen_fields() -> None:
    api = _api().experiment
    kwargs = dict(
        method="scenario_robust_prefix", role="ordinary", horizon=8, prefix=4,
        requested_k=8, risk_lambda=0.5, phase3b_recipe_digest="4" * 64,
        h1_model_digest="5" * 64, slot_duration_ms=1.0,
        cooperative_deadline_ns=10_000_000_000, checker_version="phase1-atomic-v1",
    )
    digest = api.config_digest(**kwargs)
    assert len(digest) == 64
    assert "actual_k" not in inspect.signature(api.config_digest).parameters
    assert digest != api.config_digest(**{**kwargs, "prefix": 2})


def test_method_registry_exactly_matches_nine_roles_and_capabilities() -> None:
    registry = _api().experiment.METHOD_REGISTRY
    assert tuple(registry) == METHODS
    assert registry["full_information_lower_bound"].uses_oracle and not registry["full_information_lower_bound"].executable
    assert registry["oracle_scenario_robust_reference"].reference_kind == "truth_assisted_support_ceiling_not_proven_performance_bound"
    assert all(not registry[name].uses_oracle for name in METHODS[2:8])


def test_paired_worlds_are_fresh_and_method_order_is_digest_rotation() -> None:
    api = _api().experiment
    manifest = api.toy_manifest()
    episodes = [api.build_episode(manifest, method) for method in METHODS]
    assert len({id(item.world) for item in episodes}) == len(METHODS)
    assert len({id(item.reveal_process) for item in episodes}) == len(METHODS)
    order = api.rotated_method_order(_coordinate(api))
    assert set(order) == set(METHODS) and len(order) == 9
    assert order == api.rotated_method_order(_coordinate(api))


def test_public_episode_runner_is_scientifically_invariant_forward_vs_reverse_oracle_order() -> None:
    api = _api().experiment
    manifest = api.toy_manifest()
    coordinate_id = _coordinate(api)
    forward_order = METHODS
    reverse_order = tuple(reversed(METHODS))
    assert forward_order[-1] == "oracle_scenario_robust_reference"
    assert reverse_order[0] == "oracle_scenario_robust_reference"

    world_objects: list[Any] = []
    cache_objects: list[dict[Any, Any]] = []

    def execute(order: tuple[str, ...]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for method in order:
            episode = api.build_episode(manifest, method)
            assert all(episode.world is not prior for prior in world_objects)
            world_objects.append(episode.world)
            cache: dict[Any, Any] = {}
            assert all(cache is not prior for prior in cache_objects)
            cache_objects.append(cache)
            result = api.run_public_episode(
                manifest=manifest, coordinate_id=coordinate_id, method=method,
                episode=episode, episode_cache=cache,
            )
            outputs[method] = {
                "rows": _scientific_without_order_identity_or_timing(result.scientific_rows),
                "final_world": result.final_world_digest,
                "rng": result.rng_result_digest,
                "cache_keys": tuple(sorted(str(key) for key in cache)),
            }
        return outputs

    forward = execute(forward_order)
    reverse = execute(reverse_order)
    assert set(forward) == set(reverse) == set(METHODS)
    for method in METHODS:
        assert forward[method] == reverse[method]
    ordinary = METHODS[2:8]
    assert all(
        not any("oracle" in key.lower() for key in forward[method]["cache_keys"])
        for method in ordinary
    )


def test_oracle_recent_history_support_covers_each_fresh_partial_shard_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = _api()
    api = modules.experiment
    from rlccl.envs.problem import TopologyInfo

    nodes = 4
    edges = np.asarray([
        (source, destination)
        for source in range(nodes)
        for destination in range(nodes)
        if source != destination
    ], dtype=np.int64)
    topology = TopologyInfo(
        nodes, len(edges), edges, np.ones(len(edges), dtype=np.float64),
        [(tuple(range(len(edges))), 1.0)], name="phase4-oracle-undercoverage-red",
    )
    assert set(map(tuple, edges)) == {
        (source, destination)
        for source in range(nodes)
        for destination in range(nodes)
        if source != destination
    }
    assert all(int(value) >= 1 for value in topology.capacities)
    assert topology.shared_constraints == [(tuple(range(len(edges))), 1.0)]

    truth = np.full((nodes, nodes), 8, dtype=np.int64)
    np.fill_diagonal(truth, 0)
    histories = tuple(np.zeros_like(truth) for _ in range(32))
    assert all(matrix.flags.owndata for matrix in histories)
    assert all(
        not np.shares_memory(left, right)
        for index, left in enumerate(histories)
        for right in histories[index + 1:]
    )
    manifest = dict(api.toy_manifest())
    manifest.update({
        "toy_truth_matrix": truth,
        "toy_history_matrices": histories,
    })

    original_plan = modules.robust.RobustPrefixPlanner.plan
    support_sightings: list[tuple[str, int, bool, bool, int, int]] = []

    def checked_plan(planner: Any, view: Any, support: Any) -> Any:
        revealed_counts = np.zeros(truth.shape, dtype=np.int64)
        for token in view.revealed_tokens:
            revealed_counts[int(token.source), int(token.destination)] += 1
        assert support.stage == view.stage
        assert support.observation_digest == view.observation_digest
        assert support.actual_k == len(support.matrices)
        for member_index, matrix in enumerate(support.matrices):
            candidate = np.asarray(matrix)
            assert candidate.shape == truth.shape
            assert np.issubdtype(candidate.dtype, np.integer)
            assert np.all(candidate >= 0)
            assert np.all(candidate >= revealed_counts), (
                f"oracle stage {view.stage} support member {member_index} "
                "undercovers revealed tokens"
            )
        if support.uses_oracle:
            assert support.method == "oracle_scenario_robust_reference"
            assert support.upper_bound_only is True
            assert support.requested_k == 8
            assert 1 <= support.actual_k <= 8
            if view.stage == 4:
                assert support.actual_k == 1
                assert np.array_equal(support.matrices[0], truth)
        else:
            assert support.upper_bound_only is False
        support_sightings.append((
            support.method, int(view.stage), bool(support.uses_oracle),
            bool(support.upper_bound_only), int(support.requested_k),
            int(support.actual_k),
        ))
        plan = original_plan(planner, view, support)
        assert all(len(batch) <= 1 for batch in plan.batches)
        return plan

    monkeypatch.setattr(modules.robust.RobustPrefixPlanner, "plan", checked_plan)
    ordinary = "scenario_robust_prefix"
    oracle = "oracle_scenario_robust_reference"
    coordinate_id = _coordinate(
        api, sequence_digest="7" * 64, mode="partial_shards", reveal_seed=202608013,
    )
    episode_objects: list[Any] = []
    reveal_objects: list[Any] = []
    cache_objects: list[dict[Any, Any]] = []

    def execute(order: tuple[str, str]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for method in order:
            episode = api.build_formal_episode(
                truth_matrix=truth, topology=topology, method=method,
                sequence_id="phase4-oracle-undercoverage-red", sequence_step=32,
                family=FAMILIES[0], reveal_mode="partial_shards", reveal_seed=202608013,
            )
            cache: dict[Any, Any] = {}
            assert all(episode.world is not prior for prior in episode_objects)
            assert all(episode.reveal_process is not prior for prior in reveal_objects)
            assert all(cache is not prior for prior in cache_objects)
            episode_objects.append(episode.world)
            reveal_objects.append(episode.reveal_process)
            cache_objects.append(cache)
            result = api.run_public_episode(
                manifest=manifest, coordinate_id=coordinate_id, method=method,
                episode=episode, episode_cache=cache,
            )
            api.validate_event_ledger(
                result.scientific_rows[0]["events"], require_canonical_sentinels=True,
            )
            outputs[method] = {
                "rows": _scientific_without_order_identity_or_timing(result.scientific_rows),
                "final_world": result.final_world_digest,
                "rng": result.rng_result_digest,
                "cache_keys": tuple(sorted(str(key) for key in cache)),
            }
        return outputs

    forward = execute((ordinary, oracle))
    reverse = execute((oracle, ordinary))
    assert forward[ordinary] == reverse[ordinary]
    assert forward[oracle] == reverse[oracle]
    assert forward[ordinary]["rows"][0]["uses_oracle"] is False
    assert forward[oracle]["rows"][0]["uses_oracle"] is True
    assert not any("oracle" in key.lower() for key in forward[ordinary]["cache_keys"])
    assert all("oracle" in key.lower() for key in forward[oracle]["cache_keys"])
    assert set(forward[ordinary]["cache_keys"]).isdisjoint(forward[oracle]["cache_keys"])
    oracle_sightings = [item for item in support_sightings if item[2]]
    ordinary_sightings = [item for item in support_sightings if not item[2]]
    assert {item[1] for item in oracle_sightings} == {0, 1, 2, 3, 4}
    assert {item[1] for item in ordinary_sightings} == {0, 1, 2, 3, 4}
    assert all(not item[3] for item in ordinary_sightings)


def test_fresh_manifest_embeds_both_frozen_sources_and_rejects_digest_overlap() -> None:
    api = _api().experiment
    manifest = api.toy_manifest()
    assert manifest["old_manifests"] == {
        "h1": H1_MANIFEST_SHA256,
        "phase3b": PHASE3B_MANIFEST_SHA256,
    }
    excluded = set(manifest["excluded_sequence_digests"])
    assert excluded
    assert excluded == set(manifest["h1_excluded_sequence_digests"]) | set(
        manifest["phase3b_excluded_sequence_digests"]
    )
    new_digests = [record["sequence_digest"] for record in manifest["sequence_records"]]
    api.validate_fresh_sequence_digests(new_digests, excluded)
    with pytest.raises(ValueError, match="collision|overlap|excluded"):
        api.validate_fresh_sequence_digests([*new_digests[:-1], next(iter(excluded))], excluded)


def test_h1_fit_universe_mlp_parameters_and_target_scale_are_frozen() -> None:
    api = _api().experiment
    contract = api.h1_fit_contract()
    assert contract.fit_sequence_count == 15
    assert tuple(contract.fit_steps) == tuple(range(8, 256))
    assert contract.fit_example_count == 15 * 248 == 3_720
    assert contract.config == {
        "recent_steps": 8, "hidden_layer_sizes": (32,), "activation": "tanh",
        "solver": "adam", "alpha": 1e-4, "batch_size": 256,
        "learning_rate_init": 1e-3, "max_iter": 80, "shuffle": True,
        "early_stopping": False, "seed": 20260731,
    }
    scale = api.fit_target_scale(np.asarray([[2.0, 5.0], [2.0, 9.0]], dtype=np.float64))
    assert np.array_equal(scale, np.asarray([1.0, 2.0]))


def test_h1_point_uses_actual_unknown_pool32_and_ratio1_singleton_with_frozen_tie_break() -> None:
    api = _api().experiment
    summaries = np.arange(32, dtype=np.float64).reshape(32, 1) + 10.0
    summaries[30, 0] = -1.0
    summaries[31, 0] = 1.0
    offsets = np.arange(-32, 0, dtype=np.int64)
    chosen = api.select_h1_point_candidate(
        prediction=np.asarray([0.0]), candidate_summaries=summaries,
        fit_target_scale=np.asarray([1.0]), history_offsets=offsets,
    )
    assert chosen.pool_size == 32 and chosen.pool_index == 31 and chosen.history_offset == -1
    singleton = api.select_h1_point_candidate(
        prediction=np.asarray([99.0]), candidate_summaries=np.asarray([[7.0]]),
        fit_target_scale=np.asarray([1.0]), history_offsets=np.asarray([-8]),
    )
    assert singleton.pool_size == 1 and singleton.pool_index == 0


# 5. Exact validation/test universes and selection.
def test_validation_universe_is_exact_9600_and_only_three_canonical_methods() -> None:
    api = _api().experiment
    rows = api.build_toy_validation_registry(materialize_metrics=False)
    assert len(rows) == 9_600
    counts = Counter(row.method for row in rows)
    assert counts == Counter({"scenario_robust_prefix": 9_000, "wait_until_known": 300, "partial_current_only": 300})
    assert {row.method for row in rows} == set(VALIDATION_METHODS)


def test_validation_selects_by_sequence_equal_lexicographic_objective() -> None:
    api = _api().experiment
    rows = _plain_validation_rows()
    selected = api.select_validation_config(rows)
    assert (selected.horizon, selected.prefix, selected.risk_lambda) == (2, 1, 0.0)
    assert api.select_primary_comparator(rows) == "partial_current_only"


def test_test_universe_is_exact_2700_and_sequence_universe_135() -> None:
    api = _api().experiment
    registry = api.build_toy_test_registry()
    assert len(registry.episode_keys) == 2_700
    assert len(registry.sequence_keys) == 135
    assert {key.method for key in registry.episode_keys} == set(METHODS)
    assert all(sum(key.sequence_id == seq for key in registry.episode_keys) == 180 for seq in {item.sequence_id for item in registry.episode_keys})


# 6. Completion, timing, metrics, statistics, and Gate mechanics.
def test_completion_timeout_and_end_to_end_units_are_exact() -> None:
    api = _api().experiment
    assert api.completion_after_slot(0) == 1
    assert api.unfinished_completion(time_limit=80) == 81
    assert api.end_to_end_latency_ms(completion_slots=20, total_online_ns=3_500_000) == pytest.approx(23.5)
    assert api.reveal_lead_lag_slots(12) == -4
    assert api.reveal_lead_lag_slots(81) == 65


def test_cooperative_deadline_marks_both_timeouts_without_claiming_preemption() -> None:
    api = _api().experiment
    result = api.deadline_outcome(elapsed_ns=10_000_000_001, deadline_ns=10_000_000_000)
    assert result.wall_timeout and result.discrete_timeout and result.completion_slots == 81
    assert result.total_online_ns == 10_000_000_001
    assert api.DEADLINE_KIND == "cooperative_not_preemptive"


def test_exclusive_timing_has_seven_components_plus_unattributed_without_double_count() -> None:
    api = _api().experiment
    timing = api.finalize_timing(
        total_online_ns=100,
        components={name: (5 if name != "unattributed" else 0) for name in COMPONENTS},
    )
    assert tuple(timing) == COMPONENTS
    assert timing["unattributed"] == 65
    assert sum(timing.values()) == 100


def test_higher_quantile_cvar_and_sequence_equal_aggregation_are_frozen() -> None:
    api = _api().experiment
    values = np.arange(1, 21, dtype=np.float64)
    stats = api.sequence_distribution(values)
    assert stats.p95 == 20.0 and stats.p99 == 20.0
    assert stats.cvar95 == 20.0
    assert stats.mean == pytest.approx(10.5)


def test_family_seed_deltas_bootstrap_and_positive_sequence_ess_use_whole_sequences() -> None:
    api = _api().experiment
    deltas = np.arange(15, dtype=np.float64).reshape(5, 3)
    decomposition = api.decompose_sequence_deltas(deltas, families=FAMILIES, base_seeds=BASE_SEEDS)
    assert np.allclose(decomposition.family_deltas, deltas.mean(axis=1))
    assert np.allclose(decomposition.base_seed_deltas, deltas.mean(axis=0))
    boot = api.family_stratified_bootstrap(deltas, samples=10_000, seed=20260801)
    assert boot.samples.shape == (10_000,) and boot.lower <= boot.mean <= boot.upper
    assert 1.0 <= api.positive_sequence_ess(np.asarray([1.0, 2.0, 3.0, 4.0])) <= 4.0


def test_plain_gate_evidence_passes_all_conditions_and_remains_pending_supervisor() -> None:
    api = _api().experiment
    result = api.evaluate_h2_conditions(_plain_gate_evidence())
    assert result.passed and result.data_status == "PASS"
    assert result.gate_status == "PENDING_SUPERVISOR"
    assert tuple(result.conditions) == tuple(range(1, 9))


def test_each_gate_condition_fails_under_direct_plain_evidence_mutation() -> None:
    api = _api().experiment
    mutations: list[tuple[int, Any]] = []

    def mutation(condition: int, fn: Any) -> None:
        mutations.append((condition, fn))

    mutation(1, lambda e: e["paired"]["wait"].update(ci_lower=0.0))
    mutation(1, lambda e: e["paired"]["partial"].update(mean_e2e_delta=-0.01))
    mutation(2, lambda e: e["paired"]["wait"].update(mean_cvar95_delta=0.01))
    mutation(2, lambda e: e["paired"]["partial"].update(mean_cvar95_delta=0.01))
    mutation(3, lambda e: e["base_seed_deltas"].update({842: 0.0}))
    mutation(3, lambda e: e["family_deltas"].update({FAMILIES[3]: -0.1, FAMILIES[4]: -0.1}))
    mutation(3, lambda e: (
        e["family_deltas"].update({FAMILIES[4]: -0.1}),
        e["family_relative_degradation"].update({FAMILIES[4]: 0.1000001}),
    ))
    mutation(4, lambda e: e["legality_rates"].update({"scenario_robust_prefix": 0.999}))
    for comparator in ("wait", "partial"):
        for timeout_kind in ("discrete", "wall"):
            mutation(5, lambda e, c=comparator, k=timeout_kind: e["paired"][c].update({
                f"robust_{k}_timeout_rate": e["paired"][c][f"comparator_{k}_timeout_rate"] + 0.01
            }))
    mutation(6, lambda e: e.update(scheduling_only_delta=2.0, end_to_end_delta=-0.01))
    mutation(6, lambda e: e.update(overhead_included=False))
    mutation(7, lambda e: e.update(test_sequence_count=14))
    mutation(7, lambda e: e.update(test_episode_count=2_699))
    for field in (
        "fresh_exclusion_complete", "capability_isolation_complete",
        "artifact_chain_complete", "focused_tests_complete",
    ):
        mutation(8, lambda e, name=field: e.update({name: False}))

    for expected_condition, mutate in mutations:
        evidence = copy.deepcopy(_plain_gate_evidence())
        mutate(evidence)
        result = api.evaluate_h2_conditions(evidence)
        assert not result.passed
        assert expected_condition in result.failed_conditions


def test_explicit_gate_failure_has_priority_over_environment_hold() -> None:
    api = _api().experiment
    evidence = _plain_gate_evidence()
    evidence["paired"]["wait"]["ci_lower"] = -0.1
    evidence["environment_complete"] = False
    result = api.evaluate_h2_conditions(evidence)
    assert result.data_status == "FAIL"
    assert 1 in result.failed_conditions


# 7. Event ledger, sentinels, reasons, payload digests, and state-machine recomputation.
def test_event_payload_digest_uses_only_persisted_ordered_fields() -> None:
    api = _api().experiment
    row = _event(api, kind="episode_start", index=0)
    digest = row["event_payload_digest"]
    assert digest == api.event_payload_digest(row)
    row["slot"] = 1
    assert api.event_payload_digest(row) != digest
    row["event_payload_digest"] = "f" * 64
    assert api.event_payload_digest(row) != row["event_payload_digest"]
    row = _event(api, kind="plan_built", index=0, requested_k=8, actual_k=8)
    digest = row["event_payload_digest"]
    row["actual_k"] = 1
    assert api.event_payload_digest(row) != digest


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("episode_start", "NONE"), ("plan_built", "initial"),
        ("suffix_discarded", "reveal"), ("wait_latch_entered", "no_common_action"),
        ("checker_rejected", "edge_capacity"), ("episode_end", "complete"),
    ],
)
def test_event_kind_reason_registry_accepts_only_frozen_pairs(kind: str, reason: str) -> None:
    api = _api().experiment
    api.validate_event_reason(kind, reason)
    with pytest.raises(ValueError):
        api.validate_event_reason(kind, "invented_reason")


def test_event_sentinels_start_end_uniqueness_and_contiguous_index_fail_closed() -> None:
    api = _api().experiment
    rows = _minimal_ledger(api)
    api.validate_event_ledger(rows)
    assert rows[0]["slot"] == -1 and rows[0]["support_digest"] == ZERO_DIGEST
    with pytest.raises(ValueError):
        api.validate_event_ledger([rows[0], dict(rows[1], event_index=2)])
    with pytest.raises(ValueError):
        api.validate_event_ledger([rows[0], rows[0], rows[1]])


def test_wait_latch_fallback_and_replan_counters_recompute_only_from_events() -> None:
    api = _api().experiment
    rows = [
        _event(api, kind="episode_start", index=0),
        _event(api, kind="plan_built", index=1, reason="initial", slot=0, stage=0,
               state_version_before=0, state_version_after=0, support_digest="a" * 64,
               plan_revision=0, batch_count=0, action_count=0, requested_k=8, actual_k=8),
        _event(api, kind="wait_latch_entered", index=2, reason="no_common_action", slot=0, stage=0,
               state_version_before=0, state_version_after=0),
        _event(api, kind="plan_built", index=3, reason="reveal", slot=4, stage=1,
               state_version_before=0, state_version_after=0, support_digest="b" * 64,
               plan_revision=1, batch_count=2, action_count=2, requested_k=8, actual_k=8),
        _event(api, kind="episode_end", index=4, reason="discrete_timeout", slot=80, stage=4),
    ]
    metrics = api.recompute_episode_from_events(rows)
    assert metrics.fallback_events == metrics.no_common_action_events == 1
    assert metrics.reveal_replan_events == metrics.true_replan_events == 1


def test_action_events_recompute_waste_and_residual_without_opaque_token_plaintext() -> None:
    api = _api().experiment
    rows = _committed_action_ledger(api)
    metrics = api.recompute_episode_from_events(rows)
    assert metrics.wasted_executed_actions == 0
    assert metrics.residual_repair_actions == 2
    assert "opaque" not in json.dumps(rows)


def test_action_commit_requires_proposal_and_batch_events_in_same_chain() -> None:
    api = _api().experiment
    rows = _committed_action_ledger(api)
    api.validate_event_ledger(rows)
    for missing_kind in ("proposal_bound", "batch_committed"):
        tampered = [row for row in rows if row["event_kind"] != missing_kind]
        tampered = [replace_event_index(api, row, index) for index, row in enumerate(tampered)]
        with pytest.raises(ValueError, match="proposal|batch|chain"):
            api.validate_event_ledger(tampered)


def test_dynamic_robust_k_is_recomputed_from_stage_plan_events_and_linked_support() -> None:
    api = _api().experiment
    rows = _plan_k_ledger(
        api, method="scenario_robust_prefix", plans=((0, 8, 8), (3, 8, 8), (4, 8, 1)),
    )
    expected_support = {
        row["event_index"]: row["support_digest"]
        for row in rows if row["event_kind"] == "plan_built"
    }
    api.validate_event_ledger(rows)
    api.validate_plan_support_links(rows, expected_support)
    episode = api.recompute_episode_from_events(rows)
    assert (episode.requested_k, episode.actual_k_min, episode.actual_k_max) == (8, 1, 8)
    api.validate_episode_against_events(episode, rows)

    plan_indices = [index for index, row in enumerate(rows) if row["event_kind"] == "plan_built"]
    mutations = (
        (plan_indices[0], {"actual_k": 1}),
        (plan_indices[-1], {"actual_k": 8}),
        (0, {"requested_k": 8, "actual_k": 8}),
    )
    for index, changes in mutations:
        tampered = [dict(row) for row in rows]
        tampered[index].update(changes)
        tampered = [replace_event_index(api, row, i) for i, row in enumerate(tampered)]
        with pytest.raises(ValueError, match="stage|plan|requested|actual|K"):
            api.validate_event_ledger(tampered)

    for bad_min, bad_max in ((0, 8), (1, 7), (8, 1)):
        with pytest.raises(ValueError, match="actual|K|min|max|event"):
            api.validate_episode_against_events(
                replace(episode, actual_k_min=bad_min, actual_k_max=bad_max), rows,
            )

    support_tamper = [dict(row) for row in rows]
    support_tamper[plan_indices[1]]["support_digest"] = "f" * 64
    support_tamper = [replace_event_index(api, row, i) for i, row in enumerate(support_tamper)]
    with pytest.raises(ValueError, match="support|digest|plan"):
        api.validate_plan_support_links(support_tamper, expected_support)


@pytest.mark.parametrize(
    "method,role,uses_oracle,executable,plans,expected_min,expected_max",
    [
        ("full_information_lower_bound", "lower_bound", True, False, (), 0, 0),
        ("full_information_executable_reference", "executable_reference", True, True, (), 0, 0),
        ("wait_until_known", "ordinary", False, True, (), 0, 0),
        ("partial_current_only", "ordinary", False, True, (), 0, 0),
        ("long_term_mean_point_plan", "ordinary", False, True, ((0, 1, 1), (4, 1, 1)), 1, 1),
        ("previous_value_point_plan", "ordinary", False, True, ((3, 1, 1),), 1, 1),
        ("h1_best_point_plan", "ordinary", False, True, ((4, 1, 1),), 1, 1),
        ("oracle_scenario_robust_reference", "oracle_ceiling", True, True, ((0, 8, 8), (4, 8, 1)), 1, 8),
    ],
)
def test_method_specific_plan_k_domains_and_no_plan_zero_minmax(
    method: str, role: str, uses_oracle: bool, executable: bool,
    plans: tuple[tuple[int, int, int], ...], expected_min: int, expected_max: int,
) -> None:
    api = _api().experiment
    rows = _plan_k_ledger(
        api, method=method, plans=plans, role=role,
        uses_oracle=uses_oracle, executable=executable,
    )
    api.validate_event_ledger(rows)
    episode = api.recompute_episode_from_events(rows)
    assert (episode.actual_k_min, episode.actual_k_max) == (expected_min, expected_max)


def test_oracle_stage4_actual_k_must_be_singleton() -> None:
    api = _api().experiment
    rows = _plan_k_ledger(
        api, method="oracle_scenario_robust_reference", plans=((4, 8, 2),),
        role="oracle_ceiling", uses_oracle=True,
    )
    with pytest.raises(ValueError, match="oracle|stage|actual|singleton|K"):
        api.validate_event_ledger(rows)


def test_event_transaction_order_versions_continuity_and_action_count_fail_closed() -> None:
    api = _api().experiment
    rows = _committed_action_ledger(api)
    api.validate_event_ledger(rows)

    def rewritten(changes: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for index, row in enumerate(rows):
            changed = dict(row)
            changed.update(changes.get(index, {}))
            result.append(replace_event_index(api, changed, index))
        return result

    swapped = list(rows)
    swapped[4], swapped[5] = swapped[5], swapped[4]
    swapped = [replace_event_index(api, row, index) for index, row in enumerate(swapped)]
    invalid_ledgers = (
        swapped,
        rewritten({3: {"state_version_after": 1}}),
        rewritten({5: {"state_version_after": 0}}),
        rewritten({4: {"state_version_before": 1, "state_version_after": 1}}),
        rewritten({5: {"action_count": 1}}),
    )
    for invalid in invalid_ledgers:
        with pytest.raises(ValueError, match="order|state|version|count|transaction|continu"):
            api.validate_event_ledger(invalid)


def test_checker_rejection_is_terminal_illegal_and_cannot_be_followed_by_fallback_or_action() -> None:
    api = _api().experiment
    rows = _checker_rejected_ledger(api)
    api.validate_event_ledger(rows)
    api.validate_episode_termination(
        rows,
        {"completion_slots": 81, "legality": False, "discrete_timeout": False, "wall_timeout": False},
    )
    for forbidden_kind, reason in (("action_committed", "NONE"), ("batch_committed", "NONE"),
                                   ("wait_latch_entered", "no_common_action"), ("plan_built", "invalidation")):
        inserted = _event(
            api, kind=forbidden_kind, index=3, reason=reason, slot=6, stage=1,
            state_version_before=4, state_version_after=4, plan_revision=2,
        )
        tampered = [*rows[:3], inserted, replace_event_index(api, rows[-1], 4)]
        with pytest.raises(ValueError, match="reject|terminal|illegal"):
            api.validate_event_ledger(tampered)


@pytest.mark.parametrize(
    "reason,end_slot,completion,legality,discrete,wall",
    [
        ("complete", 0, 0, True, False, False),
        ("complete", 4, 4, True, False, False),
        ("discrete_timeout", 80, 81, True, True, False),
        ("wall_timeout", 5, 81, True, True, True),
    ],
)
def test_end_cursor_reason_completion_legality_and_timeout_joint_contract(
    reason: str, end_slot: int, completion: int, legality: bool, discrete: bool, wall: bool,
) -> None:
    api = _api().experiment
    rows = [
        _event(api, kind="episode_start", index=0),
        _event(api, kind="episode_end", index=1, reason=reason, slot=end_slot, stage=min(end_slot // 4, 4)),
    ]
    episode = {
        "completion_slots": completion, "legality": legality,
        "discrete_timeout": discrete, "wall_timeout": wall,
    }
    api.validate_episode_termination(rows, episode)
    for field, bad in (
        ("completion_slots", completion + 1), ("legality", not legality),
        ("discrete_timeout", not discrete), ("wall_timeout", not wall),
    ):
        with pytest.raises(ValueError, match="end|cursor|completion|legality|timeout|reason"):
            api.validate_episode_termination(rows, {**episode, field: bad})


def test_episode_aggregate_counter_must_recompute_exactly_from_event_ledger() -> None:
    api = _api().experiment
    rows = _committed_action_ledger(api)
    episode = api.recompute_episode_from_events(rows)
    api.validate_episode_against_events(episode, rows)
    with pytest.raises(ValueError, match="counter|executed|ledger"):
        api.validate_episode_against_events(replace(episode, prefix_executed_actions=3), rows)


def replace_event_index(api: Any, row: Mapping[str, Any], index: int) -> dict[str, Any]:
    changed = dict(row, event_index=index)
    changed["event_payload_digest"] = api.event_payload_digest(changed)
    changed["row_digest"] = api.row_digest(changed)
    return changed


def test_published_artifacts_require_unreachable_zero_and_no_unreachable_event() -> None:
    api = _api().experiment
    episode = api.synthetic_episode_row(unreachable_od_count=0)
    api.validate_published_episode(episode, _minimal_ledger(api))
    with pytest.raises(ValueError):
        api.validate_published_episode(dict(episode, unreachable_od_count=1), _minimal_ledger(api))


# 8. Exact schemas, PK/sort, corruption, and acyclic hash graph.
def test_exact_artifact_names_row_counts_and_primary_keys() -> None:
    api = _api().experiment
    assert api.ARTIFACT_NAMES == ARTIFACTS
    assert api.EXACT_ROW_COUNTS == {
        "raw_validation_metrics.csv": 9_600,
        "raw_test_episode_metrics.csv": 2_700,
        "raw_test_sequence_metrics.csv": 135,
        "raw_timing_metrics.csv": 21_600,
    }
    assert api.PRIMARY_KEYS["raw_validation_metrics.csv"] == ("coordinate_id", "method", "horizon", "prefix", "risk_lambda")
    assert api.PRIMARY_KEYS["raw_test_execution_events.csv"] == ("coordinate_id", "method", "event_index")
    assert api.EXACT_COLUMNS["raw_validation_metrics.csv"] == VALIDATION_COLUMNS
    assert api.EXACT_COLUMNS["raw_test_episode_metrics.csv"] == EPISODE_COLUMNS
    assert api.EXACT_COLUMNS["raw_test_execution_events.csv"] == EVENT_COLUMNS
    assert "actual_k" not in VALIDATION_COLUMNS and "actual_k" not in EPISODE_COLUMNS
    assert ("actual_k_min" in VALIDATION_COLUMNS and "actual_k_max" in VALIDATION_COLUMNS)
    assert ("requested_k" in EVENT_COLUMNS and "actual_k" in EVENT_COLUMNS)


def test_sequence_sentinels_and_method_config_digest_are_exact() -> None:
    api = _api().experiment
    row = api.synthetic_sequence_row()
    assert (row["split"], row["coordinate_id"], row["checkpoint"], row["checkpoint_index"]) == ("test", "ALL", -1, -1)
    assert (row["reveal_mode"], row["mode_index"], row["reveal_seed"]) == ("ALL", -1, -1)
    api.validate_sequence_row(row)


@pytest.mark.parametrize(
    "filename,changes,recompute_event,recompute_row",
    [
        ("raw_validation_metrics.csv", {"reveal_seed": "202608011"}, False, True),
        (
            "raw_validation_metrics.csv",
            {"requested_k": "8", "actual_k_min": "4", "actual_k_max": "4"},
            False, True,
        ),
        ("raw_test_episode_metrics.csv", {"role": "oracle_ceiling", "uses_oracle": "true"}, False, True),
        ("raw_test_execution_events.csv", {"observation_digest": "a" * 64}, True, True),
        ("raw_test_execution_events.csv", {"residual_state_digest": "b" * 64}, True, True),
        ("raw_test_execution_events.csv", {"support_digest": "c" * 64}, True, True),
        ("raw_test_episode_metrics.csv", {"prefix_executed_actions": "999"}, False, True),
    ],
)
def test_direct_persisted_linked_tamper_rejected_even_with_local_digests_recomputed(
    filename: str, changes: Mapping[str, Any], recompute_event: bool,
    recompute_row: bool, tmp_path: Path,
) -> None:
    api = _api().experiment
    directory = api.write_toy_artifacts(tmp_path / "toy", final=True)
    _rewrite_csv_row(
        api, directory / filename, changes,
        recompute_event_payload=recompute_event, recompute_row_digest=recompute_row,
    )
    with pytest.raises(ValueError):
        api.read_back_artifacts(directory, require_final=True)


def test_direct_json_and_row_universe_tamper_rejected_without_production_corruptor(tmp_path: Path) -> None:
    api = _api().experiment
    summary_dir = api.write_toy_artifacts(tmp_path / "summary", final=True)
    summary_path = summary_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gate_status"] = "PASS"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError):
        api.read_back_artifacts(summary_dir, require_final=True)

    missing_dir = api.write_toy_artifacts(tmp_path / "missing", final=True)
    path = missing_dir / "raw_timing_metrics.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 2
    path.write_text("\n".join([lines[0], *lines[2:]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        api.read_back_artifacts(missing_dir, require_final=True)


def test_validation_chain_starts_at_validation_raw_and_test_chain_starts_at_events(tmp_path: Path) -> None:
    api = _api().experiment
    directory = api.write_toy_artifacts(tmp_path / "toy", final=True)
    result = api.read_back_artifacts(directory, require_final=True)
    assert result.validation_chain == ("raw_validation_metrics", "config_selection", "primary_selection")
    assert result.test_chain == ("execution_events", "test_episode", "test_sequence", "conditions_1_to_8", "summary")


def test_artifact_hash_graph_is_acyclic_and_excludes_manifest_self_hash(tmp_path: Path) -> None:
    api = _api().experiment
    directory = api.write_toy_artifacts(tmp_path / "toy", final=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    graph = {
        "manifest.json": set(manifest["artifact_logical_sha256"]),
        "summary.json": set(manifest["artifact_scientific_sha256"]),
    }
    _assert_acyclic(graph)
    cyclic = {node: set(children) for node, children in graph.items()}
    cyclic["manifest.json"].add("manifest.json")
    with pytest.raises(AssertionError, match="cycle"):
        _assert_acyclic(cyclic)
    assert "manifest.json" not in graph["manifest.json"]
    assert "manifest.json" not in graph["summary.json"]
    assert "summary.json" not in graph["summary.json"]
    assert set(graph["summary.json"]) == {
        "h1_best_point_model.json", "raw_validation_metrics.csv",
        "raw_test_episode_metrics.csv", "raw_test_sequence_metrics.csv",
        "raw_test_execution_events.csv", "raw_timing_metrics.csv",
    }


def test_publish_spy_observes_two_real_readbacks_and_one_directory_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api().experiment
    destination = tmp_path / "phase4-final"
    readback_calls: list[tuple[Path, bool]] = []
    rename_pairs: set[tuple[str, str]] = set()
    original_readback = api.read_back_artifacts
    original_path_replace = Path.replace
    original_os_replace = os.replace

    def readback_spy(directory: Path, *, require_final: bool) -> Any:
        readback_calls.append((Path(directory), require_final))
        return original_readback(directory, require_final=require_final)

    def path_replace_spy(source: Path, target: Path) -> Path:
        if Path(source).is_dir():
            rename_pairs.add((str(Path(source).resolve()), str(Path(target).resolve())))
        return original_path_replace(source, target)

    def os_replace_spy(source: Any, target: Any, *args: Any, **kwargs: Any) -> None:
        if Path(source).is_dir():
            rename_pairs.add((str(Path(source).resolve()), str(Path(target).resolve())))
        original_os_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(api, "read_back_artifacts", readback_spy)
    monkeypatch.setattr(Path, "replace", path_replace_spy)
    monkeypatch.setattr(os, "replace", os_replace_spy)
    api.publish_toy_artifacts(destination)
    assert destination.is_dir()
    assert [final for _, final in readback_calls] == [False, True]
    assert len(rename_pairs) == 1
    source, target = next(iter(rename_pairs))
    assert Path(target) == destination.resolve()
    assert Path(source).name.startswith(".phase4-staging-")
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_status"] == "PENDING_SUPERVISOR"
    assert not tuple(tmp_path.glob(".phase4-staging-*"))
    with pytest.raises(FileExistsError):
        api.publish_toy_artifacts(destination)


def test_import_collection_and_toy_never_create_formal_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _api().experiment
    monkeypatch.chdir(tmp_path)
    assert not FORMAL_OUTPUT.exists()
    api.write_toy_artifacts(tmp_path / "toy", final=True)
    assert not FORMAL_OUTPUT.exists()
