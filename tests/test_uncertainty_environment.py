"""Phase 1 contract tests for the uncertainty environment.

This file intentionally precedes the implementation.  Ordinary planners in these
tests only receive ``PartialObservationState`` and optional sanitized history or
scenario views.  Trusted test/evaluator code may hold a world or reveal process,
but those objects must never occur in a planner-facing payload or signature.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Mapping

import numpy as np
import pytest

from rlccl.envs.problem import TopologyInfo
from rlccl.uncertainty.baselines import (
    FullInformationOracle,
    LongTermMeanBaseline,
    PartialCurrentOnlyBaseline,
    PreviousValueBaseline,
    WaitUntilKnownBaseline,
)
from rlccl.uncertainty.evaluation import EvaluationManifest, PairedEvaluationRunner
from rlccl.uncertainty.execution import (
    Proposal,
    TransferAction,
    validate_legacy_schedule_matrix,
)
from rlccl.uncertainty.metrics import RecourseMetrics
from rlccl.uncertainty.observation import (
    PartialObservationState,
    RevealedDemandToken,
    SanitizedHistoryView,
    TruthTokenId,
)
from rlccl.uncertainty.problem import UncertainProblemInstance
from rlccl.uncertainty.reveal import DemandRevealProcess
from rlccl.uncertainty.scenarios import (
    ScenarioDemandToken,
    ScenarioSet,
    ScenarioTokenId,
)


RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
ENTRY_PERMUTATION_MODES = (
    "random_entries",
    "source_totals_first",
    "source_destination_totals_first",
)
ALL_MODES = ENTRY_PERMUTATION_MODES + (
    "partial_shards",
    "time_based_arrival",
)
ORDINARY_BASELINES = (
    WaitUntilKnownBaseline,
    PartialCurrentOnlyBaseline,
    LongTermMeanBaseline,
    PreviousValueBaseline,
)
METHODS = (
    "full_information_oracle",
    "wait_until_known",
    "partial_current_only",
    "long_term_mean",
    "previous_value",
)

FORBIDDEN_CAPABILITY_NAMES = {
    "world",
    "problem",
    "uncertain_problem",
    "reveal_process",
    "oracle",
    "oracle_view",
    "manifest",
    "evaluator",
    "rng",
    "rng_state",
    "random_state",
    "truth_matrix",
    "full_matrix",
    "private_c",
    "private_token_count",
    "truth_chunk_map",
    "generator_metadata",
    "latent_regime",
    "shock_flags",
    "future_mask",
    "future_masks",
    "future_ratios",
    "arrival_times",
    "entry_order",
}


def _complete_topology(
    nodes: int = 4,
    *,
    capacity: float = 1024.0,
    shared_constraints: list[tuple[list[int], float]] | None = None,
) -> TopologyInfo:
    edges = np.asarray(
        [(source, destination) for source in range(nodes) for destination in range(nodes) if source != destination],
        dtype=np.int64,
    )
    capacities = np.full(len(edges), capacity, dtype=np.float64)
    return TopologyInfo(
        nodes,
        len(edges),
        edges,
        capacities,
        [] if shared_constraints is None else shared_constraints,
        name=f"complete{nodes}",
    )


def _traffic() -> np.ndarray:
    # Includes both true zero and nonzero off-diagonal entries.
    return np.asarray(
        [
            [0, 3, 0, 2],
            [1, 0, 4, 0],
            [0, 2, 0, 1],
            [3, 0, 2, 0],
        ],
        dtype=np.int64,
    )


def _alternate_traffic() -> np.ndarray:
    return np.asarray(
        [
            [0, 1, 2, 0],
            [0, 0, 1, 3],
            [4, 0, 0, 1],
            [1, 2, 0, 0],
        ],
        dtype=np.int64,
    )


def _world(
    matrix: np.ndarray | None = None,
    *,
    topology: TopologyInfo | None = None,
    metadata: Mapping[str, Any] | None = None,
    sequence_id: str = "sequence-7",
    sequence_step: int = 2,
) -> UncertainProblemInstance:
    truth = _traffic() if matrix is None else np.asarray(matrix)
    return UncertainProblemInstance.from_traffic_matrix(
        truth_matrix=truth,
        topology_info=_complete_topology(truth.shape[0]) if topology is None else topology,
        time_limit=12,
        sequence_id=sequence_id,
        sequence_step=sequence_step,
        family="contract-test",
        generator_metadata={} if metadata is None else dict(metadata),
    )


def _observations(
    matrix: np.ndarray | None = None,
    *,
    mode: str = "random_entries",
    seed: int = 730,
    ratios: tuple[float, ...] = RATIOS,
    topology: TopologyInfo | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[UncertainProblemInstance, DemandRevealProcess, tuple[PartialObservationState, ...]]:
    world = _world(matrix, topology=topology, metadata=metadata)
    process = DemandRevealProcess(
        problem=world,
        mode=mode,
        ratios=ratios,
        seed=seed,
    )
    return world, process, tuple(process)


def _off_diagonal(mask: np.ndarray) -> np.ndarray:
    nodes = mask.shape[0]
    return np.asarray(mask)[~np.eye(nodes, dtype=bool)]


def _edge_index(topology: TopologyInfo, source: int, destination: int) -> int:
    matches = np.flatnonzero(
        (np.asarray(topology.edge_src) == source)
        & (np.asarray(topology.edge_dst) == destination)
    )
    assert len(matches) == 1
    return int(matches[0])


def _public_value(value: Any) -> Any:
    """Convert a public payload into a stable, comparison-friendly value."""
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _public_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_public_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _public_value(getattr(value, field.name))
            for field in fields(value)
        }
    return value


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        _public_value(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_read_only_arrays(value: Any, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_read_only_arrays(key, seen)
            _assert_read_only_arrays(item, seen)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_read_only_arrays(item, seen)
        return
    if is_dataclass(value):
        for field in fields(value):
            _assert_read_only_arrays(getattr(value, field.name), seen)


def _assert_no_private_capability(value: Any, seen: set[int] | None = None) -> None:
    """Reject references/callbacks that could lead back to evaluator-private state."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    assert not isinstance(
        value,
        (
            UncertainProblemInstance,
            DemandRevealProcess,
            EvaluationManifest,
            FullInformationOracle,
        ),
    )
    if isinstance(value, np.ndarray) or value is None or isinstance(
        value, (str, bytes, int, float, bool, np.generic)
    ):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            assert normalized not in FORBIDDEN_CAPABILITY_NAMES
            _assert_no_private_capability(item, seen)
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _assert_no_private_capability(item, seen)
        return
    assert not inspect.isfunction(value)
    assert not inspect.ismethod(value)
    if is_dataclass(value):
        for field in fields(value):
            assert field.name.lower() not in FORBIDDEN_CAPABILITY_NAMES
            _assert_no_private_capability(getattr(value, field.name), seen)
        return
    values = vars(value) if hasattr(value, "__dict__") else {}
    for name, item in values.items():
        assert name.lower() not in FORBIDDEN_CAPABILITY_NAMES
        _assert_no_private_capability(item, seen)


def _assert_observation_sanitized(observation: PartialObservationState) -> None:
    assert isinstance(observation, PartialObservationState)
    _assert_read_only_arrays(observation)
    _assert_no_private_capability(observation)
    payload = observation.to_policy_payload()
    _assert_read_only_arrays(payload)
    _assert_no_private_capability(payload)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_all_modes_cover_frozen_ratios_and_are_reproducible(mode: str) -> None:
    _, _, first = _observations(mode=mode, seed=17)
    _, _, second = _observations(mode=mode, seed=17)
    assert tuple(item.ratio for item in first) == RATIOS
    assert len(first) == len(RATIOS)
    assert [_fingerprint(item.to_policy_payload()) for item in first] == [
        _fingerprint(item.to_policy_payload()) for item in second
    ]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_masks_and_public_tokens_are_monotone_and_final_stage_is_complete(mode: str) -> None:
    truth = _traffic()
    _, _, observations = _observations(truth, mode=mode, seed=19)
    previous_entries = np.zeros_like(truth, dtype=bool)
    previous_ids: set[TruthTokenId] = set()
    for observation in observations:
        entry_mask = np.asarray(observation.entry_mask, dtype=bool)
        token_ids = {token.token_id for token in observation.revealed_tokens}
        assert np.all(previous_entries <= entry_mask)
        assert previous_ids <= token_ids
        previous_entries = entry_mask
        previous_ids = token_ids
    final = observations[-1]
    assert np.all(_off_diagonal(final.entry_mask))
    assert len(final.revealed_tokens) == int(truth.sum())


@pytest.mark.parametrize("mode", ENTRY_PERMUTATION_MODES)
def test_entry_permutation_modes_use_off_diagonal_floor_denominator(mode: str) -> None:
    nodes = 4
    denominator = nodes * (nodes - 1)
    _, _, observations = _observations(mode=mode, seed=23)
    expected = [int(np.floor(ratio * denominator)) for ratio in RATIOS]
    expected[-1] = denominator
    actual = [int(_off_diagonal(item.entry_mask).sum()) for item in observations]
    assert actual == expected


def test_partial_shards_uses_private_token_floor_without_public_full_length_mask() -> None:
    truth = _traffic()
    private_count = int(truth.sum())
    _, _, observations = _observations(truth, mode="partial_shards", seed=29)
    expected = [int(np.floor(ratio * private_count)) for ratio in RATIOS]
    expected[-1] = private_count
    assert [len(item.revealed_tokens) for item in observations] == expected
    for observation in observations[:-1]:
        payload = observation.to_policy_payload()
        assert "token_mask" not in payload
        assert "total_token_count" not in payload
        assert "private_c" not in payload
        for array in _arrays_in(payload):
            assert array.shape != (private_count,)


def _arrays_in(value: Any) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, Mapping):
        for item in value.values():
            result.extend(_arrays_in(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.extend(_arrays_in(item))
    elif is_dataclass(value):
        for field in fields(value):
            result.extend(_arrays_in(getattr(value, field.name)))
    return result


def test_time_based_arrival_is_thresholded_monotone_not_forced_to_ratio_count() -> None:
    _, _, observations = _observations(mode="time_based_arrival", seed=31)
    counts = [int(_off_diagonal(item.entry_mask).sum()) for item in observations]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] == 12


def test_unknown_and_revealed_true_zero_are_distinguished_by_mask() -> None:
    truth = _traffic()
    _, _, observations = _observations(truth, mode="random_entries", seed=37)
    initial, final = observations[0], observations[-1]
    zero_positions = np.argwhere((truth == 0) & ~np.eye(4, dtype=bool))
    assert len(zero_positions) > 0
    source, destination = map(int, zero_positions[0])
    assert initial.observed_matrix[source, destination] == 0
    assert not initial.entry_mask[source, destination]
    assert final.observed_matrix[source, destination] == 0
    assert final.entry_mask[source, destination]
    assert initial.unknown_mask[source, destination]
    assert not final.unknown_mask[source, destination]


@pytest.mark.parametrize(
    "mode",
    ("random_entries", "partial_shards", "time_based_arrival"),
)
def test_ratio_zero_has_empty_actions_and_does_not_leak_private_count_or_padding(mode: str) -> None:
    low = np.zeros((4, 4), dtype=np.int64)
    low[0, 1] = 1
    high = np.full((4, 4), 40, dtype=np.int64)
    np.fill_diagonal(high, 0)
    _, _, low_observations = _observations(low, mode=mode, seed=41)
    _, _, high_observations = _observations(high, mode=mode, seed=41)
    low_initial, high_initial = low_observations[0], high_observations[0]
    assert low_initial.revealed_tokens == high_initial.revealed_tokens == ()
    assert _fingerprint(low_initial.to_policy_payload()) == _fingerprint(
        high_initial.to_policy_payload()
    )
    assert PartialCurrentOnlyBaseline().propose(low_initial).is_wait
    assert PartialCurrentOnlyBaseline().propose(high_initial).is_wait
    private_counts = {int(low.sum()), int(high.sum())}
    for observation in (low_initial, high_initial):
        for array in _arrays_in(observation.to_policy_payload()):
            assert not any(array.ndim > 0 and count in array.shape for count in private_counts)


@pytest.mark.parametrize(
    "mode,source_visible,destination_visible",
    (
        ("source_totals_first", True, False),
        ("source_destination_totals_first", True, True),
    ),
)
def test_totals_first_are_aggregate_only_and_never_create_proxy_tokens(
    mode: str, source_visible: bool, destination_visible: bool
) -> None:
    truth = _traffic()
    _, _, observations = _observations(truth, mode=mode, seed=43)
    initial = observations[0]
    assert not np.any(_off_diagonal(initial.entry_mask))
    assert initial.revealed_tokens == ()
    assert np.array_equal(initial.source_totals, truth.sum(axis=1)) is source_visible
    if destination_visible:
        np.testing.assert_array_equal(initial.destination_totals, truth.sum(axis=0))
    else:
        assert initial.destination_totals is None
    assert not np.any(initial.observed_matrix)
    assert PartialCurrentOnlyBaseline().propose(initial).is_wait


def test_observation_history_and_scenarios_are_deep_copied_read_only_and_sanitized() -> None:
    latent = {
        "latent_regime": ["future"] * 5,
        "shock_flags": [0, 1, 0, 0, 1],
        "arrival_times": [0.1, 0.2],
    }
    truth = _traffic()
    _, _, observations = _observations(
        truth, mode="random_entries", seed=47, metadata=latent
    )
    observation = observations[2]
    history_inputs = [truth.copy(), _alternate_traffic()]
    history = SanitizedHistoryView.from_completed_matrices(
        matrices=history_inputs,
        steps=(0, 1),
        sequence_id="sequence-7",
        current_step=2,
    )
    scenario_inputs = [np.asarray(observation.observed_matrix).copy()]
    scenarios = ScenarioSet.from_matrices(
        matrices=scenario_inputs,
        weights=(1.0,),
        scenario_ids=("history-0",),
        provenance=("history-only",),
    )
    history_before = _fingerprint(history)
    scenarios_before = _fingerprint(scenarios)
    history_inputs[0][:] = 999
    scenario_inputs[0][:] = 999
    assert _fingerprint(history) == history_before
    assert _fingerprint(scenarios) == scenarios_before
    for public_object in (observation, history, scenarios):
        _assert_read_only_arrays(public_object)
        _assert_no_private_capability(public_object)
    _assert_observation_sanitized(observation)
    assert "latent_regime" not in repr(observation.to_policy_payload())
    assert "shock_flags" not in repr(observation.to_policy_payload())
    assert "arrival_times" not in repr(observation.to_policy_payload())


def test_history_view_rejects_current_or_future_steps_and_is_sequence_isolated() -> None:
    matrices = [_traffic(), _alternate_traffic()]
    first = SanitizedHistoryView.from_completed_matrices(
        matrices=(matrices[0],),
        steps=(0,),
        sequence_id="first",
        current_step=1,
    )
    second = SanitizedHistoryView.from_completed_matrices(
        matrices=(matrices[1],),
        steps=(0,),
        sequence_id="second",
        current_step=1,
    )
    assert first.sequence_id != second.sequence_id
    assert not np.shares_memory(first.matrices[0], second.matrices[0])
    with pytest.raises(ValueError, match="current|future|<"):
        SanitizedHistoryView.from_completed_matrices(
            matrices=matrices,
            steps=(0, 2),
            sequence_id="bad",
            current_step=2,
        )


def test_truth_and_scenario_token_namespaces_are_runtime_separate() -> None:
    truth_id = TruthTokenId("opaque-public-token")
    scenario_id = ScenarioTokenId("scenario:s0:0")
    assert type(truth_id) is not type(scenario_id)
    assert truth_id != scenario_id
    assert not hasattr(truth_id, "chunk_index")
    assert str(scenario_id).startswith("scenario:s0:")
    scenario_token = ScenarioDemandToken(
        token_id=scenario_id,
        source=0,
        destination=1,
    )
    assert isinstance(scenario_token.token_id, ScenarioTokenId)
    with pytest.raises(TypeError, match="TruthTokenId|scenario"):
        TransferAction(token_id=scenario_id, edge_index=0)


def test_scenario_only_and_unrevealed_truth_tokens_cannot_be_committed() -> None:
    world, _, observations = _observations(mode="random_entries", seed=53)
    early, final = observations[0], observations[-1]
    future_token = final.revealed_tokens[0]
    edge = _edge_index(world.topology_info, future_token.source, future_token.destination)
    unrevealed = Proposal.from_transfers(
        (TransferAction(token_id=future_token.token_id, edge_index=edge),)
    )
    with pytest.raises(ValueError, match="unrevealed|executable|stage"):
        world.commit(early, unrevealed)

    scenario_set = ScenarioSet.from_matrices(
        matrices=(np.asarray(final.observed_matrix),),
        weights=(1.0,),
        scenario_ids=("s0",),
        provenance=("manual",),
    )
    with pytest.raises(ValueError, match="scenario|not executable"):
        world.commit(early, Proposal.scenario_only(scenario_set))


@pytest.mark.parametrize(
    "bad_value,message",
    (
        (np.nan, "finite|NaN"),
        (np.inf, "finite|Inf"),
        (-1.0, "nonnegative|negative"),
        (0.5, "binary"),
        (2.0, "binary"),
    ),
)
def test_legacy_matrix_validator_rejects_invalid_numeric_domain(
    bad_value: float, message: str
) -> None:
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[0, 0] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_legacy_schedule_matrix(matrix, expected_shape=(2, 3))


@pytest.mark.parametrize(
    "matrix",
    (
        np.zeros(3),
        np.zeros((2, 2)),
        np.zeros((2, 3, 1)),
    ),
)
def test_legacy_matrix_validator_rejects_bad_shape(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape|2-D"):
        validate_legacy_schedule_matrix(matrix, expected_shape=(2, 3))


def _single_token_world(
    *, capacity: float = 1.0
) -> tuple[UncertainProblemInstance, PartialObservationState, RevealedDemandToken]:
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 1] = 1
    topology = _complete_topology(3, capacity=capacity)
    world, _, observations = _observations(
        matrix,
        mode="random_entries",
        seed=59,
        topology=topology,
    )
    observation = observations[-1]
    return world, observation, observation.revealed_tokens[0]


def test_commit_rejects_bad_edge_duplicate_and_source_possession() -> None:
    world, observation, token = _single_token_world()
    with pytest.raises(ValueError, match="edge|range"):
        world.commit(
            observation,
            Proposal.from_transfers(
                (TransferAction(token.token_id, world.topology_info.E),)
            ),
        )

    correct_edge = _edge_index(world.topology_info, token.source, token.destination)
    duplicate = TransferAction(token.token_id, correct_edge)
    with pytest.raises(ValueError, match="duplicate|conflict"):
        world.commit(
            observation,
            Proposal.from_transfers((duplicate, duplicate)),
        )

    wrong_source = next(
        node
        for node in range(world.topology_info.V)
        if node not in (token.source, token.destination)
    )
    wrong_edge = _edge_index(world.topology_info, wrong_source, token.destination)
    with pytest.raises(ValueError, match="source|possession|holder"):
        world.commit(
            observation,
            Proposal.from_transfers((TransferAction(token.token_id, wrong_edge),)),
        )


def test_commit_rejects_destination_already_holds_and_stale_observation() -> None:
    world, observation, token = _single_token_world()
    edge = _edge_index(world.topology_info, token.source, token.destination)
    proposal = Proposal.from_transfers((TransferAction(token.token_id, edge),))
    result = world.commit(observation, proposal)
    assert result.legal
    with pytest.raises(ValueError, match="destination|already|stale|executable"):
        world.commit(observation, proposal)


def test_commit_rejects_edge_capacity() -> None:
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 1] = 2
    topology = _complete_topology(3, capacity=1.0)
    world, _, observations = _observations(
        matrix, mode="random_entries", seed=61, topology=topology
    )
    observation = observations[-1]
    tokens = observation.revealed_tokens
    edge = _edge_index(topology, 0, 1)
    proposal = Proposal.from_transfers(
        tuple(TransferAction(token.token_id, edge) for token in tokens)
    )
    with pytest.raises(ValueError, match="capacity|bandwidth"):
        world.commit(observation, proposal)


def test_commit_rejects_shared_group_limit() -> None:
    nodes = 4
    base = _complete_topology(nodes)
    first_edge = _edge_index(base, 0, 1)
    second_edge = _edge_index(base, 2, 3)
    topology = _complete_topology(
        nodes,
        shared_constraints=[([first_edge, second_edge], 1.0)],
    )
    matrix = np.zeros((nodes, nodes), dtype=np.int64)
    matrix[0, 1] = 1
    matrix[2, 3] = 1
    world, _, observations = _observations(
        matrix, mode="random_entries", seed=67, topology=topology
    )
    observation = observations[-1]
    by_pair = {(token.source, token.destination): token for token in observation.revealed_tokens}
    proposal = Proposal.from_transfers(
        (
            TransferAction(by_pair[(0, 1)].token_id, first_edge),
            TransferAction(by_pair[(2, 3)].token_id, second_edge),
        )
    )
    with pytest.raises(ValueError, match="shared|group|bandwidth"):
        world.commit(observation, proposal)


def test_more_than_512_candidates_are_hidden_truth_invariant_on_new_numpy_path() -> None:
    base = np.full((4, 4), 60, dtype=np.int64)
    np.fill_diagonal(base, 0)
    _, _, first_observations = _observations(
        base, mode="random_entries", seed=71
    )
    first = first_observations[3]
    assert len(first.revealed_tokens) > 512

    changed = base.copy()
    hidden = ~np.asarray(first.entry_mask, dtype=bool)
    hidden &= ~np.eye(4, dtype=bool)
    changed[hidden] = np.arange(1, int(hidden.sum()) + 1) * 137
    _, _, second_observations = _observations(
        changed, mode="random_entries", seed=71
    )
    second = second_observations[3]

    assert _fingerprint(first.to_policy_payload()) == _fingerprint(
        second.to_policy_payload()
    )
    assert tuple(first.executable_token_ids) == tuple(second.executable_token_ids)
    first_proposal = PartialCurrentOnlyBaseline().propose(first)
    second_proposal = PartialCurrentOnlyBaseline().propose(second)
    assert _fingerprint(first_proposal.to_public_payload()) == _fingerprint(
        second_proposal.to_public_payload()
    )
    assert "rlccl.envs.decoder" not in sys.modules
    assert "torch" not in sys.modules


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _manifest(matrix: np.ndarray | None = None) -> EvaluationManifest:
    truth = _traffic() if matrix is None else np.asarray(matrix)
    topology = _complete_topology(truth.shape[0])
    return EvaluationManifest.create(
        manifest_id="manifest-contract-001",
        sequence_id="sequence-7",
        family="contract-test",
        history_provenance="same-sequence-completed-only",
        truth_matrix=truth,
        topology_info=topology,
        reveal_mode="random_entries",
        ratios=RATIOS,
        reveal_seed=79,
        timeout=12,
        time_limit=12,
        checker_version="reveal-aware-v1",
    )


def test_evaluation_manifest_is_frozen_digest_only_and_not_policy_payload() -> None:
    manifest = _manifest()
    with pytest.raises(FrozenInstanceError):
        manifest.reveal_seed = 99
    values = vars(manifest) if hasattr(manifest, "__dict__") else {
        field.name: getattr(manifest, field.name) for field in fields(manifest)
    }
    assert "truth_digest" in values
    assert not any(name in values for name in ("truth", "truth_matrix", "demands", "world"))
    assert not any(isinstance(value, np.ndarray) for value in values.values())
    assert isinstance(manifest.ratios, tuple)


def _runner() -> PairedEvaluationRunner:
    truth = _traffic()
    history = (np.maximum(truth - 1, 0), _alternate_traffic())
    return PairedEvaluationRunner(
        manifest=_manifest(truth),
        truth_matrix=truth,
        history_matrices=history,
        topology_info=_complete_topology(4),
        generator_metadata={
            "latent_regime": ["future"] * 5,
            "shock_flags": [0, 1, 0, 0, 1],
        },
    )


def test_same_manifest_rebuilds_independent_world_reveal_and_rng_per_method() -> None:
    runner = _runner()
    episodes = {name: runner.build_episode(name) for name in METHODS}
    assert len({id(episode.world) for episode in episodes.values()}) == len(METHODS)
    assert len({id(episode.reveal_process) for episode in episodes.values()}) == len(METHODS)
    first = episodes["wait_until_known"].next_observation()
    episodes["wait_until_known"].next_observation()
    untouched = episodes["partial_current_only"].next_observation()
    assert _fingerprint(first.to_policy_payload()) == _fingerprint(
        untouched.to_policy_payload()
    )
    assert all(episode.manifest is runner.manifest for episode in episodes.values())


@pytest.mark.parametrize("baseline_type", ORDINARY_BASELINES)
def test_ordinary_planner_signature_has_no_private_or_oracle_parameter(
    baseline_type: type[Any],
) -> None:
    parameters = inspect.signature(baseline_type.propose).parameters
    assert tuple(parameters) == ("self", "observation", "history", "scenarios")
    assert parameters["observation"].annotation in (
        PartialObservationState,
        "PartialObservationState",
    )
    assert parameters["history"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["scenarios"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not ({name.lower() for name in parameters} & FORBIDDEN_CAPABILITY_NAMES)


def test_oracle_is_an_evaluator_only_upper_bound_not_an_ordinary_planner() -> None:
    oracle = FullInformationOracle()
    assert not hasattr(oracle, "propose")
    assert oracle.uses_oracle
    assert oracle.upper_bound_only
    runner = _runner()
    row = runner.run_oracle(oracle)
    assert row["method"] == "full_information_oracle"
    assert row["uses_oracle"] is True
    assert row["upper_bound_only"] is True


def test_history_baselines_only_depend_on_completed_x_before_t() -> None:
    truth = _traffic()
    history = SanitizedHistoryView.from_completed_matrices(
        matrices=(np.maximum(truth - 1, 0), _alternate_traffic()),
        steps=(0, 1),
        sequence_id="sequence-7",
        current_step=2,
    )
    _, _, first_observations = _observations(
        truth, mode="random_entries", seed=83
    )
    changed_current = truth.copy()
    changed_current[first_observations[0].unknown_mask] += 1000
    _, _, second_observations = _observations(
        changed_current, mode="random_entries", seed=83
    )
    for baseline in (LongTermMeanBaseline(), PreviousValueBaseline()):
        first = baseline.propose(first_observations[0], history=history)
        second = baseline.propose(second_observations[0], history=history)
        assert _fingerprint(first.to_public_payload()) == _fingerprint(
            second.to_public_payload()
        )


def test_paired_runner_records_common_provenance_and_legality_for_all_baselines() -> None:
    runner = _runner()
    rows = runner.run_default_baselines()
    assert {row["method"] for row in rows} == set(METHODS)
    common_fields = (
        "manifest_id",
        "truth_digest",
        "topology_digest",
        "config_digest",
        "reveal_mode",
        "reveal_seed",
        "timeout",
        "checker_version",
    )
    for field_name in common_fields:
        assert len({row[field_name] for row in rows}) == 1
    assert all(row["legal"] for row in rows)
    assert all(row["legality_rate"] == 1.0 for row in rows)
    ordinary = [row for row in rows if row["method"] != "full_information_oracle"]
    assert all(row["uses_oracle"] is False for row in ordinary)
    oracle = next(row for row in rows if row["method"] == "full_information_oracle")
    assert oracle["uses_oracle"] is True
    assert oracle["upper_bound_only"] is True


def test_recourse_metrics_fields_and_raw_provenance_are_complete() -> None:
    required = {
        "completion",
        "oracle_regret",
        "reveal_wait",
        "recourse_count",
        "replanned_actions",
        "wasted_plan",
        "synthesis_time_ms",
        "replan_time_ms",
        "legality",
        "timeout",
        "sequence_id",
        "family",
        "seed",
        "topology",
        "reveal_stage",
        "reveal_mode",
        "method",
        "manifest_id",
        "truth_digest",
        "topology_digest",
        "config_digest",
        "checker_version",
    }
    assert required <= {field.name for field in fields(RecourseMetrics)}
    metrics = RecourseMetrics(
        completion=4,
        oracle_regret=1.0,
        reveal_wait=2,
        recourse_count=1,
        replanned_actions=2,
        wasted_plan=1,
        synthesis_time_ms=3.5,
        replan_time_ms=0.75,
        legality=True,
        timeout=False,
        sequence_id="sequence-7",
        family="contract-test",
        seed=89,
        topology="complete4",
        reveal_stage=4,
        reveal_mode="random_entries",
        method="partial_current_only",
        manifest_id="manifest-contract-001",
        truth_digest="truth-digest",
        topology_digest="topology-digest",
        config_digest="config-digest",
        checker_version="reveal-aware-v1",
    )
    row = metrics.to_raw_row()
    assert required <= set(row)
    assert row["sequence_id"] == "sequence-7"
    assert row["manifest_id"] == "manifest-contract-001"
    assert row["legality"] is True
    assert row["timeout"] is False


def test_uncertainty_path_does_not_import_frozen_torch_decoder_stack() -> None:
    # Use a fresh interpreter so prior tests importing legacy Torch paths cannot
    # contaminate this assertion through the process-global module cache.
    script = textwrap.dedent(
        """
        import json
        import sys

        from rlccl.uncertainty import (
            baselines,
            evaluation,
            execution,
            metrics,
            observation,
            problem,
            reveal,
            scenarios,
        )

        forbidden = {
            "rlccl.envs.decoder",
            "rlccl.models",
            "rlccl.training",
            "rlccl.evaluation.sequence_evaluator",
            "torch",
        }
        print(json.dumps(sorted(forbidden.intersection(sys.modules))))
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def _manifest_for(
    truth: np.ndarray,
    topology: TopologyInfo,
    *,
    ratios: tuple[float, ...] = RATIOS,
    time_limit: int = 12,
    timeout_limit: int | None = None,
    mode: str = "random_entries",
    seed: int = 101,
) -> EvaluationManifest:
    return EvaluationManifest.create(
        manifest_id=f"manifest-hardening-{seed}",
        sequence_id="sequence-hardening",
        family="contract-hardening",
        history_provenance="same-sequence-completed-only",
        truth_matrix=truth,
        topology_info=topology,
        reveal_mode=mode,
        ratios=ratios,
        reveal_seed=seed,
        timeout=time_limit if timeout_limit is None else timeout_limit,
        time_limit=time_limit,
        checker_version="reveal-aware-v1",
    )


def _runner_for(
    truth: np.ndarray,
    topology: TopologyInfo,
    *,
    history: tuple[np.ndarray, ...] = (),
    ratios: tuple[float, ...] = RATIOS,
    time_limit: int = 12,
    mode: str = "random_entries",
    seed: int = 101,
) -> PairedEvaluationRunner:
    return PairedEvaluationRunner(
        manifest=_manifest_for(
            truth,
            topology,
            ratios=ratios,
            time_limit=time_limit,
            mode=mode,
            seed=seed,
        ),
        truth_matrix=truth,
        history_matrices=history,
        topology_info=topology,
        generator_metadata={},
    )


def test_history_baselines_emit_distinct_deterministic_scenario_only_plans() -> None:
    world, _, observations = _observations(mode="random_entries", seed=103)
    initial, final = observations[0], observations[-1]
    first_history = np.zeros_like(_traffic())
    second_history = _alternate_traffic()
    history = SanitizedHistoryView.from_completed_matrices(
        matrices=(first_history, second_history),
        steps=(0, 1),
        sequence_id=initial.sequence_id,
        current_step=initial.sequence_step,
    )

    mean_baseline = LongTermMeanBaseline()
    previous_baseline = PreviousValueBaseline()
    mean_plan = mean_baseline.propose(initial, history=history)
    previous_plan = previous_baseline.propose(initial, history=history)
    assert mean_plan.scenario_set is not None and not mean_plan.actions
    assert previous_plan.scenario_set is not None and not previous_plan.actions
    np.testing.assert_array_equal(
        mean_plan.scenario_set.matrices[0],
        np.rint(np.mean(np.stack(history.matrices), axis=0)).astype(np.int64),
    )
    np.testing.assert_array_equal(
        previous_plan.scenario_set.matrices[0], second_history
    )
    assert _fingerprint(mean_plan.to_public_payload()) != _fingerprint(
        previous_plan.to_public_payload()
    )
    assert _fingerprint(mean_plan.to_public_payload()) == _fingerprint(
        mean_baseline.propose(initial, history=history).to_public_payload()
    )
    with pytest.raises(ValueError, match="scenario|not executable"):
        world.commit(initial, mean_plan)

    for baseline in (mean_baseline, previous_baseline):
        executable_plan = baseline.propose(final, history=history)
        assert executable_plan.scenario_set is None
        assert all(
            isinstance(action.token_id, TruthTokenId)
            and action.token_id in final.executable_token_ids
            for action in executable_plan.actions
        )


def test_public_planner_prefilters_edge_and_shared_capacity_deterministically() -> None:
    matrix = np.zeros((4, 4), dtype=np.int64)
    matrix[0, 1] = 2
    matrix[2, 3] = 1
    base = _complete_topology(4, capacity=1.0)
    first_edge = _edge_index(base, 0, 1)
    second_edge = _edge_index(base, 2, 3)
    topology = _complete_topology(
        4,
        capacity=1.0,
        shared_constraints=[([first_edge, second_edge], 1.0)],
    )
    _, _, first_observations = _observations(
        matrix, mode="random_entries", seed=107, topology=topology
    )
    _, _, second_observations = _observations(
        matrix, mode="random_entries", seed=107, topology=topology
    )
    first = PartialCurrentOnlyBaseline().propose(first_observations[-1])
    second = PartialCurrentOnlyBaseline().propose(second_observations[-1])
    assert len(first.actions) == 1
    assert _fingerprint(first.to_public_payload()) == _fingerprint(
        second.to_public_payload()
    )


def test_public_planner_uses_deterministic_next_hop_without_same_slot_forwarding() -> None:
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    topology = TopologyInfo(3, 2, edges, np.ones(2), [], name="line3")
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 2] = 1
    world, _, observations = _observations(
        matrix, mode="random_entries", seed=109, topology=topology
    )
    first_observation = observations[-1]
    first_plan = PartialCurrentOnlyBaseline().propose(first_observation)
    assert tuple(action.edge_index for action in first_plan.actions) == (0,)
    assert world.commit(first_observation, first_plan).legal

    fresh_final = tuple(
        DemandRevealProcess(
            problem=world,
            mode="random_entries",
            ratios=RATIOS,
            seed=109,
        )
    )[-1]
    second_plan = PartialCurrentOnlyBaseline().propose(fresh_final)
    assert tuple(action.edge_index for action in second_plan.actions) == (1,)
    assert world.commit(fresh_final, second_plan).legal


def test_runner_slots_continue_after_final_reveal_and_metrics_are_real() -> None:
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    topology = TopologyInfo(3, 2, edges, np.ones(2), [], name="line3")
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 2] = 1
    runner = _runner_for(
        truth,
        topology,
        ratios=(0.0, 1.0),
        time_limit=4,
        seed=113,
    )
    rows = runner.run_default_baselines()
    oracle = next(row for row in rows if row["method"] == "full_information_oracle")
    partial = next(row for row in rows if row["method"] == "partial_current_only")
    assert oracle["completion"] == 2
    assert oracle["reveal_wait"] == 0
    assert partial["completion"] == 3
    assert partial["oracle_regret"] == 1
    assert partial["reveal_wait"] == 1
    assert partial["recourse_count"] == 2
    assert partial["replanned_actions"] == 2
    assert partial["legality"] is True and partial["timeout"] is False
    assert partial["synthesis_time_ms"] >= 0
    assert partial["replan_time_ms"] >= 0

    episode = runner.build_episode("partial_current_only")
    seen = [episode.next_observation() for _ in range(4)]
    assert [item.stage for item in seen] == [0, 1, 1, 1]
    assert [item.ratio for item in seen] == [0.0, 1.0, 1.0, 1.0]
    assert all(item.state_version == 0 for item in seen)


def test_runner_low_capacity_progress_timeout_and_scenario_waste_metrics() -> None:
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 1] = 3
    topology = _complete_topology(3, capacity=1.0)
    history = (np.zeros_like(truth), np.asarray([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    runner = _runner_for(
        truth,
        topology,
        history=history,
        ratios=(0.0, 1.0),
        time_limit=5,
        seed=127,
    )
    rows = runner.run_default_baselines()
    oracle = next(row for row in rows if row["method"] == "full_information_oracle")
    partial = next(row for row in rows if row["method"] == "partial_current_only")
    mean = next(row for row in rows if row["method"] == "long_term_mean")
    previous = next(row for row in rows if row["method"] == "previous_value")
    assert oracle["completion"] == 2
    assert partial["completion"] == 4
    assert partial["oracle_regret"] == 2
    assert partial["legality"] and partial["legality_rate"] == 1.0
    rounded_mean = np.rint(np.mean(np.stack(history), axis=0)).astype(np.int64)
    assert mean["wasted_plan"] == int(np.abs(rounded_mean - truth).sum())
    assert previous["wasted_plan"] == int(np.abs(history[-1] - truth).sum())

    timeout_runner = _runner_for(
        truth,
        topology,
        ratios=(0.0, 1.0),
        time_limit=1,
        seed=131,
    )
    timeout_partial = next(
        row
        for row in timeout_runner.run_default_baselines()
        if row["method"] == "partial_current_only"
    )
    assert timeout_partial["completion"] == 2  # frozen timeout convention: T + 1
    assert timeout_partial["timeout"] is True
    assert timeout_partial["reveal_wait"] == 1


def test_runner_legality_is_observed_not_hard_coded() -> None:
    truth = np.zeros((3, 3), dtype=np.int64)
    truth[0, 1] = 1
    topology = _complete_topology(3)
    runner = _runner_for(
        truth,
        topology,
        ratios=(0.0, 1.0),
        time_limit=3,
        seed=137,
    )

    class BadEdgeBaseline:
        method = "partial_current_only"

        def propose(
            self,
            observation: PartialObservationState,
            *,
            history: SanitizedHistoryView | None = None,
            scenarios: ScenarioSet | None = None,
        ) -> Proposal:
            del history, scenarios
            if not observation.revealed_tokens:
                return Proposal.wait()
            return Proposal.from_transfers(
                (
                    TransferAction(
                        observation.revealed_tokens[0].token_id,
                        observation.topology.num_edges,
                    ),
                )
            )

    row = runner._run_ordinary(BadEdgeBaseline())
    assert row["legality"] is False
    assert row["legal"] is False
    assert row["legality_rate"] < 1.0
    assert "edge" in row["legality_error"].lower()


def test_history_sequence_shape_and_manifest_truth_digest_are_enforced() -> None:
    _, _, observations = _observations(mode="random_entries", seed=139)
    observation = observations[0]
    wrong_sequence = SanitizedHistoryView.from_completed_matrices(
        matrices=(_traffic(),),
        steps=(0,),
        sequence_id="another-sequence",
        current_step=observation.sequence_step,
    )
    with pytest.raises(ValueError, match="sequence"):
        LongTermMeanBaseline().propose(observation, history=wrong_sequence)

    wrong_shape = SanitizedHistoryView.from_completed_matrices(
        matrices=(np.zeros((3, 3), dtype=np.int64),),
        steps=(0,),
        sequence_id=observation.sequence_id,
        current_step=observation.sequence_step,
    )
    with pytest.raises(ValueError, match="shape|topology"):
        PreviousValueBaseline().propose(observation, history=wrong_shape)

    truth = _traffic()
    topology = _complete_topology(4)
    bad_manifest = _manifest_for(truth, topology, seed=149)
    object.__setattr__(bad_manifest, "truth_digest", "0" * 64)
    with pytest.raises(ValueError, match="truth.*digest|digest.*truth"):
        PairedEvaluationRunner(
            manifest=bad_manifest,
            truth_matrix=truth,
            history_matrices=(),
            topology_info=topology,
            generator_metadata={},
        )
    with pytest.raises(ValueError, match="history.*shape|shape.*history"):
        _runner_for(
            truth,
            topology,
            history=(np.zeros((3, 3), dtype=np.int64),),
            seed=151,
        )


def test_paired_episode_generator_metadata_is_deep_isolated() -> None:
    truth = _traffic()
    topology = _complete_topology(4)
    metadata = {
        "nested": {"labels": ["stable", "private"]},
        "arrays": [np.asarray([1, 2, 3], dtype=np.int64)],
    }
    runner = PairedEvaluationRunner(
        manifest=_manifest_for(truth, topology, seed=157),
        truth_matrix=truth,
        history_matrices=(),
        topology_info=topology,
        generator_metadata=metadata,
    )
    metadata["nested"]["labels"].append("mutated-after-runner")
    metadata["arrays"][0][0] = 999

    first = runner.build_episode("wait_until_known")
    second = runner.build_episode("partial_current_only")
    assert first.world._generator_metadata["nested"]["labels"] == [
        "stable",
        "private",
    ]
    np.testing.assert_array_equal(
        first.world._generator_metadata["arrays"][0], np.asarray([1, 2, 3])
    )
    first.world._generator_metadata["nested"]["labels"].append("first-only")
    first.world._generator_metadata["arrays"][0][1] = 777
    assert second.world._generator_metadata["nested"]["labels"] == [
        "stable",
        "private",
    ]
    np.testing.assert_array_equal(
        second.world._generator_metadata["arrays"][0], np.asarray([1, 2, 3])
    )
    assert "generator_metadata" not in repr(first.next_observation().to_policy_payload())


def test_raw_rows_carry_config_provenance_and_manifest_requires_positive_timeout() -> None:
    runner = _runner()
    rows = runner.run_default_baselines()
    for row in rows:
        assert row["history_provenance"] == runner.manifest.history_provenance
        assert row["ratios"] == runner.manifest.ratios
        assert row["time_limit"] == runner.manifest.time_limit
        assert row["timeout_limit"] == runner.manifest.timeout
        assert isinstance(row["timeout"], bool)
    for field_name in (
        "history_provenance",
        "ratios",
        "time_limit",
        "timeout_limit",
    ):
        assert len({_fingerprint(row[field_name]) for row in rows}) == 1

    truth = _traffic()
    topology = _complete_topology(4)
    with pytest.raises(ValueError, match="timeout.*positive|positive.*timeout"):
        _manifest_for(truth, topology, timeout_limit=0, seed=163)


def test_oracle_is_provable_nonexecutable_lower_bound_on_chain6_counterexample() -> None:
    edges = np.asarray([(node, node + 1) for node in range(5)], dtype=np.int64)
    topology = TopologyInfo(6, 5, edges, np.ones(5), [], name="chain6-cap1")
    truth = np.zeros((6, 6), dtype=np.int64)
    truth[0, 1] = 6
    truth[0, 5] = 1
    runner = _runner_for(
        truth,
        topology,
        ratios=RATIOS,
        time_limit=30,
        mode="time_based_arrival",
        seed=2,
    )
    rows = runner.run_default_baselines()
    oracle = next(row for row in rows if row["method"] == "full_information_oracle")
    ordinary = [row for row in rows if row["method"] != "full_information_oracle"]
    partial = next(row for row in ordinary if row["method"] == "partial_current_only")

    # §11.8: max(path=5, work=ceil(11/5)=3, source=7, dest=6) = 7.
    assert oracle["completion"] == 7
    assert oracle["oracle_regret"] == 0
    assert oracle["reference_kind"] == "provable_full_information_lower_bound"
    assert oracle["executable"] is False
    assert oracle["upper_bound_only"] is True
    assert oracle["legality"] is True
    assert oracle["legality_basis"] == "vacuous_no_executable_actions"
    assert oracle["synthesis_time_ms"] > 0
    assert oracle["replan_time_ms"] == 0
    assert partial["completion"] == 10  # do not rewrite ordinary completion
    for row in ordinary:
        assert row["oracle_regret"] == row["completion"] - 7
        assert row["oracle_regret"] >= 0


def test_oracle_lower_bound_empty_unreachable_zero_unit_capacity_and_over_t() -> None:
    empty = np.zeros((2, 2), dtype=np.int64)
    complete2 = _complete_topology(2, capacity=1.0)
    assert _runner_for(empty, complete2, time_limit=3, seed=167).run_oracle(
        FullInformationOracle()
    )["completion"] == 0

    chain_edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    unreachable_topology = TopologyInfo(
        3, 2, chain_edges, np.ones(2), [], name="forward-chain3"
    )
    unreachable = np.zeros((3, 3), dtype=np.int64)
    unreachable[2, 0] = 1
    assert _runner_for(
        unreachable, unreachable_topology, time_limit=4, seed=173
    ).run_oracle(FullInformationOracle())["completion"] == 5

    zero_unit_topology = TopologyInfo(
        2,
        1,
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0.999,), dtype=np.float64),
        [],
        name="subunit-edge",
    )
    positive = np.asarray(((0, 1), (0, 0)), dtype=np.int64)
    assert _runner_for(
        positive, zero_unit_topology, time_limit=4, seed=179
    ).run_oracle(FullInformationOracle())["completion"] == 5

    over_t = np.asarray(((0, 4), (0, 0)), dtype=np.int64)
    assert _runner_for(over_t, complete2, time_limit=3, seed=181).run_oracle(
        FullInformationOracle()
    )["completion"] == 4


def test_manifest_factory_and_runner_reject_topology_and_config_mismatch() -> None:
    truth = _traffic()
    topology = _complete_topology(4)
    manifest = EvaluationManifest.create(
        manifest_id="manifest-canonical-probe",
        sequence_id="sequence-canonical",
        family="contract-hardening",
        history_provenance="same-sequence-completed-only",
        truth_matrix=truth,
        topology_info=topology,
        reveal_mode="random_entries",
        ratios=RATIOS,
        reveal_seed=191,
        timeout=17,
        time_limit=12,
        checker_version="reveal-aware-v1",
    )
    repeated = EvaluationManifest.create(
        manifest_id="manifest-canonical-probe",
        sequence_id="sequence-canonical",
        family="contract-hardening",
        history_provenance="same-sequence-completed-only",
        truth_matrix=np.array(truth, copy=True),
        topology_info=_complete_topology(4),
        reveal_mode="random_entries",
        ratios=RATIOS,
        reveal_seed=191,
        timeout=17,
        time_limit=12,
        checker_version="reveal-aware-v1",
    )
    assert (
        manifest.truth_digest,
        manifest.topology_digest,
        manifest.config_digest,
    ) == (
        repeated.truth_digest,
        repeated.topology_digest,
        repeated.config_digest,
    )

    changed_topology = _complete_topology(4, capacity=1023.5)
    with pytest.raises(ValueError, match="topology.*digest|digest.*topology"):
        PairedEvaluationRunner(
            manifest=manifest,
            truth_matrix=truth,
            history_matrices=(),
            topology_info=changed_topology,
            generator_metadata={},
        )

    # Direct construction remains possible for serialization, but runner must
    # revalidate canonical config instead of trusting supplied digests.
    bogus_config = EvaluationManifest(
        manifest_id=manifest.manifest_id,
        sequence_id=manifest.sequence_id,
        family=manifest.family,
        history_provenance=manifest.history_provenance,
        truth_digest=manifest.truth_digest,
        topology_digest=manifest.topology_digest,
        config_digest=manifest.config_digest,
        reveal_mode=manifest.reveal_mode,
        ratios=manifest.ratios,
        reveal_seed=manifest.reveal_seed + 1,
        timeout=manifest.timeout,
        time_limit=manifest.time_limit,
        checker_version=manifest.checker_version,
    )
    with pytest.raises(ValueError, match="config.*digest|digest.*config"):
        PairedEvaluationRunner(
            manifest=bogus_config,
            truth_matrix=truth,
            history_matrices=(),
            topology_info=topology,
            generator_metadata={},
        )
