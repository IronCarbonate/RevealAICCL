"""RED contracts for the Phase 3B empirical ambiguity-set core.

Production imports are deliberately delayed.  The file must remain collectable
before ``rlccl.uncertainty.ambiguity`` exists so a missing implementation is a
real test failure, not a collection error.
"""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pytest


HISTORY_WINDOW = 32
REQUESTED_K = (1, 4, 8, 16)
ORDINARY_METHODS = (
    "random_empirical",
    "worst_recent_cases",
    "boundary_scenarios",
    "minimax_subset",
)

FORBIDDEN_IMPORTS = {
    "torch",
    "rlccl.uncertainty.execution",
    "rlccl.envs.decoder",
    "rlccl.scheduling",
}
FORBIDDEN_API_FRAGMENTS = {
    "proposal",
    "transferaction",
    "truthtokenid",
    "commit",
    "prefix",
    "horizon",
    "robustscore",
    "repair",
    "recourse",
}


def _resolved_imports(source: str, module_name: str) -> set[str]:
    """Resolve absolute and relative AST imports conservatively.

    ImportFrom aliases are included as possible submodules, so both
    ``from . import execution`` and ``from .execution import X`` are caught.
    """

    tree = ast.parse(source)
    imported: set[str] = set()
    package_parts = module_name.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = max(0, len(package_parts) - (node.level - 1))
            base_parts = package_parts[:keep]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
        else:
            base = node.module or ""
        if base:
            imported.add(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            imported.add(".".join(part for part in (base, alias.name) if part))
    return imported


def _normalized_public_name(name: Any) -> str:
    return "".join(character for character in str(name).casefold() if character != "_")


def _forbidden_public_names(names: Iterable[Any]) -> set[str]:
    return {
        str(name)
        for name in names
        if any(
            fragment in _normalized_public_name(name)
            for fragment in FORBIDDEN_API_FRAGMENTS
        )
    }


def _api() -> Any:
    return importlib.import_module("rlccl.uncertainty.ambiguity")


def _complete_topology() -> Any:
    from rlccl.envs.problem import TopologyInfo

    edges = np.asarray(
        [(source, destination) for source in range(4) for destination in range(4)
         if source != destination],
        dtype=np.int64,
    )
    egress = [
        [index for index, (source, _) in enumerate(edges) if int(source) == node]
        for node in range(4)
    ]
    ingress = [
        [index for index, (_, destination) in enumerate(edges) if int(destination) == node]
        for node in range(4)
    ]
    return TopologyInfo(
        4,
        len(edges),
        edges,
        np.ones(len(edges), dtype=np.float64),
        [(indices, 3.0) for indices in (*egress, *ingress)],
        name="phase3b-complete4",
    )


def _history(*, constant: int | None = None) -> tuple[np.ndarray, ...]:
    matrices: list[np.ndarray] = []
    for index in range(HISTORY_WINDOW):
        matrix = np.zeros((4, 4), dtype=np.int64)
        for source in range(4):
            for destination in range(4):
                if source != destination:
                    matrix[source, destination] = (
                        int(constant)
                        if constant is not None
                        else (index + 2 * source + destination) % 9
                    )
        matrices.append(matrix)
    return tuple(matrices)


def _observation(
    *,
    mode: str = "random_entries",
    ratio: float = 0.0,
    observed: np.ndarray | None = None,
    entry_mask: np.ndarray | None = None,
    source_totals: np.ndarray | None = None,
    destination_totals: np.ndarray | None = None,
    sequence_id: str = "private-sequence-A",
    family: str = "private-family-A",
) -> Any:
    from rlccl.uncertainty.observation import PartialObservationState, PublicTopologyView

    topology = PublicTopologyView.from_topology_info(_complete_topology())
    values = np.zeros((4, 4), dtype=np.int64) if observed is None else np.asarray(observed)
    mask = np.eye(4, dtype=bool) if entry_mask is None else np.asarray(entry_mask)
    return PartialObservationState(
        sequence_id=sequence_id,
        sequence_step=HISTORY_WINDOW,
        family=family,
        mode=mode,
        stage=0 if ratio < 1.0 else 4,
        ratio=float(ratio),
        entry_mask=mask,
        observed_matrix=values,
        unknown_mask=~mask,
        revealed_tokens=(),
        source_totals=source_totals,
        destination_totals=destination_totals,
        topology=topology,
        state_version=0,
    )


def _normalizer(module: Any, history: tuple[np.ndarray, ...] | None = None) -> Any:
    return module.fit_descriptor_normalizer(
        _history() if history is None else history,
        _complete_topology(),
    )


def _view(
    module: Any,
    *,
    history: tuple[np.ndarray, ...] | None = None,
    observation: Any | None = None,
    construction_seed: int = 19,
) -> Any:
    matrices = _history() if history is None else history
    return module.AmbiguityConstructionView.from_observation(
        history_matrices=matrices,
        history_offsets=tuple(range(-HISTORY_WINDOW, 0)),
        observation=_observation() if observation is None else observation,
        construction_seed=construction_seed,
        normalizer=_normalizer(module, matrices),
    )


def _fingerprint(value: Any) -> bytes:
    if hasattr(value, "to_canonical_bytes"):
        return bytes(value.to_canonical_bytes())

    def normalized(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return {
                "dtype": item.dtype.str,
                "shape": list(item.shape),
                "bytes": item.tobytes().hex(),
            }
        if isinstance(item, dict):
            return {str(key): normalized(val) for key, val in sorted(item.items())}
        if isinstance(item, (tuple, list)):
            return [normalized(val) for val in item]
        if hasattr(item, "__dataclass_fields__"):
            return {
                field.name: normalized(getattr(item, field.name))
                for field in fields(item)
            }
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(normalized(value), sort_keys=True, separators=(",", ":")).encode()


def test_phase3b_module_is_a_delayed_red_dependency() -> None:
    assert importlib.util.find_spec("rlccl.uncertainty.ambiguity") is not None


def test_descriptor_order_fit_only_normalizer_and_physical_bounds_are_exact() -> None:
    api = _api()
    topology = _complete_topology()
    names = api.descriptor_names(topology)
    assert names == (
        "total_traffic",
        "source_load_0", "source_load_1", "source_load_2", "source_load_3",
        "destination_load_0", "destination_load_1", "destination_load_2", "destination_load_3",
        "hotspot_strength",
        "sparsity",
        "bandwidth_group_load_0", "bandwidth_group_load_1",
        "bandwidth_group_load_2", "bandwidth_group_load_3",
        "bandwidth_group_load_4", "bandwidth_group_load_5",
        "bandwidth_group_load_6", "bandwidth_group_load_7",
    )
    matrix = _history()[7]
    descriptor = api.traffic_descriptor(matrix, topology)
    assert descriptor.shape == (19,)
    assert descriptor[0] == matrix.sum()
    np.testing.assert_array_equal(descriptor[1:5], matrix.sum(axis=1))
    np.testing.assert_array_equal(descriptor[5:9], matrix.sum(axis=0))
    assert descriptor[9] == matrix.sum(axis=0).max() / matrix.sum(axis=0).mean()
    assert descriptor[10] == np.mean(matrix[~np.eye(4, dtype=bool)] == 0)

    normalizer = api.fit_descriptor_normalizer(_history(), topology)
    expected = np.stack([api.traffic_descriptor(item, topology) for item in _history()])
    np.testing.assert_allclose(normalizer.center, expected.mean(axis=0), rtol=0, atol=1e-12)
    expected_scale = expected.std(axis=0, ddof=0)
    expected_scale[expected_scale < 1e-8] = 1.0
    np.testing.assert_allclose(normalizer.scale, expected_scale, rtol=0, atol=1e-12)
    low, high = api.physical_descriptor_bounds(topology, max_entry=8)
    assert low.shape == high.shape == (19,)
    assert low.tolist() == [0.0] * 19
    assert high[0] == 96 and np.all(high[1:9] == 24)
    assert high[9] == 4 and high[10] == 1
    assert np.isfinite(high).all() and np.all(high >= low)


def test_narrow_view_strips_private_identity_truth_metadata_and_capabilities() -> None:
    api = _api()
    view = _view(api)
    field_names = {field.name for field in fields(view)}
    forbidden = {
        "family", "sequence_id", "base_seed", "actual_seed", "reveal_seed",
        "generator_metadata", "metadata", "latent_regime", "shock_flags",
        "truth", "world", "manifest", "reveal_process", "future_mask",
        "future_order", "future_matrix", "future_demand", "future_history",
        "current_matrix", "current_demand", "current_truth", "next_matrix",
        "arrival_times", "callback", "closure",
    }
    assert field_names.isdisjoint(forbidden)
    assert not any(
        token in name.casefold()
        for name in field_names
        for token in ("current", "future", "next", "truth", "world")
    )
    payload = repr(view).lower()
    assert "private-sequence-a" not in payload
    assert "private-family-a" not in payload
    assert all(array.flags.writeable is False for array in view.history_matrices)
    assert view.observed_matrix.flags.writeable is False
    assert view.entry_mask.flags.writeable is False
    assert view.topology.edges.flags.writeable is False
    assert len(view.history_matrices) == 32
    assert tuple(view.history_offsets) == tuple(range(-32, 0))
    assert all(left < right < 0 for left, right in zip(view.history_offsets, view.history_offsets[1:]))

    signature = inspect.signature(api.build_empirical_ambiguity_set)
    forbidden_parameters = forbidden | {"observation", "problem", "oracle"}
    assert forbidden_parameters.isdisjoint(signature.parameters)


@pytest.mark.parametrize(
    "variant",
    (
        "31-history-matrices",
        "offset-includes-zero",
        "duplicate-offset",
        "out-of-order-offset",
        "shifted-negative-range",
    ),
)
def test_narrow_view_requires_exact_32_history_and_offsets_minus32_through_minus1(
    variant: str,
) -> None:
    api = _api()
    history = _history()
    offsets = list(range(-32, 0))
    if variant == "31-history-matrices":
        history = history[1:]
        offsets = list(range(-31, 0))
    elif variant == "offset-includes-zero":
        offsets = list(range(-31, 1))
    elif variant == "duplicate-offset":
        offsets[-1] = offsets[-2]
    elif variant == "out-of-order-offset":
        offsets[-2], offsets[-1] = offsets[-1], offsets[-2]
    elif variant == "shifted-negative-range":
        offsets = list(range(-33, -1))
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(variant)

    with pytest.raises(ValueError, match="32|history|offset|negative|increasing"):
        api.AmbiguityConstructionView.from_observation(
            history_matrices=history,
            history_offsets=tuple(offsets),
            observation=_observation(),
            construction_seed=19,
            normalizer=_normalizer(api),
        )


@pytest.mark.parametrize(
    "mutation",
    ("unrevealed_truth", "future_reveal", "metadata", "family", "sequence_id"),
)
def test_ordinary_construction_is_byte_invariant_to_forbidden_counterfactuals(
    mutation: str,
) -> None:
    api = _api()
    first_observation = _observation()
    second_observation = first_observation
    if mutation == "family":
        second_observation = replace(first_observation, family="changed-family")
    elif mutation == "sequence_id":
        second_observation = replace(first_observation, sequence_id="changed-sequence")
    # Other mutations are deliberately evaluator-private and therefore have no
    # constructor argument.  Their values exist only to make that boundary explicit.
    private_counterfactual = {
        "unrevealed_truth": np.full((4, 4), 8, dtype=np.int64),
        "future_reveal": np.ones((4, 4), dtype=bool),
        "metadata": {"latent_regime": [999], "shock_flags": [True]},
    }.get(mutation)
    del private_counterfactual

    first = api.build_empirical_ambiguity_set(_view(api, observation=first_observation), calibration_radius=0.0)
    second = api.build_empirical_ambiguity_set(_view(api, observation=second_observation), calibration_radius=0.0)
    assert _fingerprint(first) == _fingerprint(second)
    for method in ORDINARY_METHODS:
        kwargs = {"replicate_seed": 41000000} if method == "random_empirical" else {}
        assert _fingerprint(api.select_support(first, method=method, k=8, **kwargs)) == _fingerprint(
            api.select_support(second, method=method, k=8, **kwargs)
        )


def test_ratio_zero_without_aggregates_preserves_all_32_indexed_history_samples() -> None:
    api = _api()
    history = list(_history())
    history[-1] = history[-2].copy()  # duplicate bytes retain two empirical identities
    ambiguity = api.build_empirical_ambiguity_set(
        _view(api, history=tuple(history), observation=_observation()),
        calibration_radius=0.0,
    )
    assert ambiguity.history_offsets == tuple(range(-32, 0))
    assert len(ambiguity.support_matrices) == 32
    for expected, actual in zip(history, ambiguity.support_matrices):
        np.testing.assert_array_equal(actual, expected)
    assert ambiguity.support_matrices[-1].tobytes() == ambiguity.support_matrices[-2].tobytes()
    assert ambiguity.history_offsets[-1] != ambiguity.history_offsets[-2]


def test_exact_and_partial_shard_reconciliation_honors_lower_bounds_and_entry_cap() -> None:
    api = _api()
    candidate = np.full((4, 4), 7, dtype=np.int64)
    np.fill_diagonal(candidate, 0)

    exact_mask = np.eye(4, dtype=bool)
    exact_mask[0, 1] = True
    exact_observed = np.zeros((4, 4), dtype=np.int64)
    exact_observed[0, 1] = 3
    exact = api.reconcile_candidate(
        candidate,
        _view(api, observation=_observation(observed=exact_observed, entry_mask=exact_mask)),
    )
    assert exact[0, 1] == 3 and exact[0, 2] == 7
    assert exact.max() <= 8

    shard_observed = np.zeros((4, 4), dtype=np.int64)
    shard_observed[0, 2] = 5
    shard_candidate = candidate.copy()
    shard_candidate[0, 2] = 1
    shard = api.reconcile_candidate(
        shard_candidate,
        _view(api, observation=_observation(mode="partial_shards", observed=shard_observed)),
    )
    assert shard[0, 2] == 5
    assert shard.max() <= 8
    with pytest.raises(ValueError, match="8|cap|bound"):
        too_large = shard_observed.copy()
        too_large[0, 2] = 9
        _view(api, observation=_observation(mode="partial_shards", observed=too_large))


def test_source_only_reconciliation_handles_zero_row_saturation_and_candidate_weights() -> None:
    api = _api()
    totals = np.asarray([24, 3, 0, 5], dtype=np.int64)
    observation = _observation(mode="source_totals_first", source_totals=totals)
    first_candidate = np.zeros((4, 4), dtype=np.int64)
    first_candidate[1, 2] = 3
    first_candidate[3, 0] = 5
    second_candidate = np.zeros((4, 4), dtype=np.int64)
    second_candidate[1, 3] = 3
    second_candidate[3, 2] = 5
    first = api.reconcile_candidate(first_candidate, _view(api, observation=observation))
    second = api.reconcile_candidate(second_candidate, _view(api, observation=observation))
    np.testing.assert_array_equal(first.sum(axis=1), totals)
    np.testing.assert_array_equal(second.sum(axis=1), totals)
    assert np.all(first[0, np.arange(4) != 0] == 8)
    assert np.count_nonzero(first[2]) == 0
    assert first[1, 2] == 3 and second[1, 3] == 3
    assert first.tobytes() != second.tobytes()
    assert first.max() <= 8 and second.max() <= 8


def test_source_destination_min_cost_flow_is_candidate_sensitive_and_deterministic() -> None:
    api = _api()
    source_totals = np.asarray([2, 2, 0, 0], dtype=np.int64)
    destination_totals = np.asarray([0, 0, 2, 2], dtype=np.int64)
    observation = _observation(
        mode="source_destination_totals_first",
        source_totals=source_totals,
        destination_totals=destination_totals,
    )
    candidate_a = np.zeros((4, 4), dtype=np.int64)
    candidate_a[0, 2], candidate_a[1, 3] = 2, 2
    candidate_b = np.zeros((4, 4), dtype=np.int64)
    candidate_b[0, 3], candidate_b[1, 2] = 2, 2
    view = _view(api, observation=observation, construction_seed=71)
    result_a = api.reconcile_candidate(candidate_a, view)
    result_b = api.reconcile_candidate(candidate_b, view)
    np.testing.assert_array_equal(result_a, candidate_a)
    np.testing.assert_array_equal(result_b, candidate_b)
    assert result_a.tobytes() != result_b.tobytes()
    assert api.reconcile_candidate(candidate_a, view).tobytes() == result_a.tobytes()

    # Zero-cost ties with no candidate preference must still follow the frozen
    # arc/path tie break and therefore be byte deterministic.
    tied_observation = _observation(
        mode="source_destination_totals_first",
        source_totals=np.asarray([1, 1, 0, 0]),
        destination_totals=np.asarray([0, 0, 1, 1]),
    )
    tied_view = _view(api, observation=tied_observation)
    tied = api.reconcile_candidate(np.zeros((4, 4), dtype=np.int64), tied_view)
    assert api.reconcile_candidate(np.zeros((4, 4), dtype=np.int64), tied_view).tobytes() == tied.tobytes()
    np.testing.assert_array_equal(tied.sum(axis=1), [1, 1, 0, 0])
    np.testing.assert_array_equal(tied.sum(axis=0), [0, 0, 1, 1])


def test_source_destination_min_cost_flow_preserves_exact_entry_across_multiple_optima() -> None:
    api = _api()
    source_totals = np.asarray([2, 2, 2, 0], dtype=np.int64)
    destination_totals = np.asarray([0, 2, 2, 2], dtype=np.int64)
    exact_mask = np.eye(4, dtype=bool)
    exact_mask[0, 1] = True
    observed = np.zeros((4, 4), dtype=np.int64)
    observed[0, 1] = 1
    observation = _observation(
        mode="source_destination_totals_first",
        observed=observed,
        entry_mask=exact_mask,
        source_totals=source_totals,
        destination_totals=destination_totals,
    )

    candidate_a = np.zeros((4, 4), dtype=np.int64)
    candidate_a[0, 1], candidate_a[0, 2] = 1, 1
    candidate_a[1, 2], candidate_a[1, 3] = 1, 1
    candidate_a[2, 1], candidate_a[2, 3] = 1, 1
    candidate_b = np.zeros((4, 4), dtype=np.int64)
    candidate_b[0, 1], candidate_b[0, 3] = 1, 1
    candidate_b[1, 2] = 2
    candidate_b[2, 1], candidate_b[2, 3] = 1, 1

    view = _view(api, observation=observation, construction_seed=73)
    result_a = api.reconcile_candidate(candidate_a, view)
    result_b = api.reconcile_candidate(candidate_b, view)
    for result in (result_a, result_b):
        assert result[0, 1] == 1
        np.testing.assert_array_equal(result.sum(axis=1), source_totals)
        np.testing.assert_array_equal(result.sum(axis=0), destination_totals)
        assert result.max() <= 8
    np.testing.assert_array_equal(result_a, candidate_a)
    np.testing.assert_array_equal(result_b, candidate_b)
    assert result_a.tobytes() != result_b.tobytes()


@pytest.mark.parametrize(
    "source_totals,destination_totals,message",
    [
        (np.asarray([25, 0, 0, 0]), None, "capacity|infeasible|margin"),
        (np.asarray([2, 0, 0, 0]), np.asarray([0, 0, 1, 0]), "conserv|infeasible|margin"),
        (np.asarray([0, 0, 0, 0]), np.asarray([0, 0, 1, 0]), "conserv|infeasible|margin"),
    ],
)
def test_reconciliation_rejects_infeasible_margins(
    source_totals: np.ndarray,
    destination_totals: np.ndarray | None,
    message: str,
) -> None:
    api = _api()
    mode = "source_totals_first" if destination_totals is None else "source_destination_totals_first"
    observation = _observation(
        mode=mode,
        source_totals=source_totals,
        destination_totals=destination_totals,
    )
    with pytest.raises(ValueError, match=message):
        api.reconcile_candidate(np.zeros((4, 4), dtype=np.int64), _view(api, observation=observation))


def test_ratio_one_is_truth_consistent_singleton_control() -> None:
    api = _api()
    truth = _history()[9]
    mask = np.ones((4, 4), dtype=bool)
    observation = _observation(ratio=1.0, observed=truth, entry_mask=mask)
    ambiguity = api.build_empirical_ambiguity_set(
        _view(api, observation=observation), calibration_radius=999.0
    )
    assert len(ambiguity.support_matrices) == 1
    np.testing.assert_array_equal(ambiguity.support_matrices[0], truth)
    assert ambiguity.singleton_control is True
    for k in REQUESTED_K:
        selected = api.select_support(ambiguity, method="minimax_subset", k=k)
        assert selected.requested_k == k and selected.actual_k == 1
        np.testing.assert_array_equal(selected.matrices[0], truth)


def test_full_probability_ambiguity_matches_formula_and_uniform_witness() -> None:
    api = _api()
    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=0.0)
    values = np.asarray(ambiguity.descriptor_vectors, dtype=np.float64)
    expected_mean = values.mean(axis=0)
    expected_variance = ((values - expected_mean) ** 2).mean(axis=0)
    np.testing.assert_allclose(ambiguity.empirical_mean, expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ambiguity.empirical_variance, expected_variance, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ambiguity.delta_mean, 0.25 * ambiguity.normalizer.scale)
    np.testing.assert_allclose(ambiguity.variance_low, 0.5 * expected_variance)
    np.testing.assert_allclose(
        ambiguity.variance_high,
        1.5 * expected_variance + 0.01 * ambiguity.normalizer.scale**2,
    )
    uniform = np.full(32, 1.0 / 32.0)
    assert ambiguity.validate_probability_weights(uniform) is True
    np.testing.assert_array_equal(ambiguity.uniform_witness, uniform)
    assert ambiguity.uses_oracle is False and ambiguity.upper_bound_only is False

    # Nonnegative normalization alone is insufficient: choose an empirical
    # one-hot witness that provably violates at least one frozen mean/variance
    # moment constraint and require the validator to reject it.
    violating_one_hot: np.ndarray | None = None
    for index in range(32):
        weights = np.zeros(32, dtype=np.float64)
        weights[index] = 1.0
        weighted_mean = weights @ values
        weighted_variance = weights @ ((values - expected_mean) ** 2)
        violates_mean = np.any(
            np.abs(weighted_mean - expected_mean) > ambiguity.delta_mean + 1e-10
        )
        violates_variance = np.any(
            (weighted_variance < ambiguity.variance_low - 1e-10)
            | (weighted_variance > ambiguity.variance_high + 1e-10)
        )
        if violates_mean or violates_variance:
            violating_one_hot = weights
            break
    assert violating_one_hot is not None
    assert np.all(violating_one_hot >= 0.0)
    assert violating_one_hot.sum() == 1.0
    with pytest.raises(ValueError, match="mean|variance|moment|ambiguity"):
        ambiguity.validate_probability_weights(violating_one_hot)

    for invalid in (
        np.full(32, -1.0 / 32.0),
        np.full(32, 1.0 / 31.0),
        np.full(32, np.nan),
        np.full(32, np.inf),
    ):
        with pytest.raises(ValueError, match="finite|nonnegative|normal|probability|witness"):
            ambiguity.validate_probability_weights(invalid)


def test_point_envelope_is_calibrated_clipped_and_separate_from_probability_and_k_support() -> None:
    api = _api()
    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=1.5)
    raw_low = np.min(ambiguity.descriptor_vectors, axis=0)
    raw_high = np.max(ambiguity.descriptor_vectors, axis=0)
    physical_low, physical_high = api.physical_descriptor_bounds(_complete_topology(), max_entry=8)
    np.testing.assert_allclose(
        ambiguity.lower_bounds,
        np.maximum(physical_low, raw_low - 1.5 * ambiguity.normalizer.scale),
    )
    np.testing.assert_allclose(
        ambiguity.upper_bounds,
        np.minimum(physical_high, raw_high + 1.5 * ambiguity.normalizer.scale),
    )
    assert ambiguity.probability_support_size == 32
    selected = api.select_support(ambiguity, method="minimax_subset", k=8)
    assert selected.actual_k == 8
    assert not hasattr(selected, "delta_mean")
    assert tuple(selected.weights) == pytest.approx((1.0 / 8.0,) * 8)


def test_random_selector_uses_eight_explicit_replicates_without_replacement_and_nested_k() -> None:
    api = _api()
    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=0.0)
    for replicate in range(8):
        seed = 41_000_000 + replicate
        prefixes = []
        for k in REQUESTED_K:
            support = api.select_support(
                ambiguity, method="random_empirical", k=k, replicate_seed=seed
            )
            assert support.actual_k == k
            assert len(set(support.selected_indices)) == k
            assert tuple(support.weights) == pytest.approx((1.0 / k,) * k)
            prefixes.append(tuple(support.selected_indices))
        assert prefixes[1][:1] == prefixes[0]
        assert prefixes[2][:4] == prefixes[1]
        assert prefixes[3][:8] == prefixes[2]
        repeated = api.select_support(
            ambiguity, method="random_empirical", k=16, replicate_seed=seed
        )
        assert tuple(repeated.selected_indices) == prefixes[-1]


def test_worst_recent_selector_uses_newest_offset_for_exact_severity_ties() -> None:
    api = _api()
    constant_history = _history(constant=2)
    ambiguity = api.build_empirical_ambiguity_set(
        _view(api, history=constant_history), calibration_radius=0.0
    )
    support = api.select_support(ambiguity, method="worst_recent_cases", k=4)
    assert tuple(support.history_offsets) == (-1, -2, -3, -4)
    assert support.severity_definition == "max_upper_fit_standardized_with_density"


def test_boundary_selector_follows_descriptor_lower_then_upper_targets_and_newest_ties() -> None:
    api = _api()
    vectors = np.asarray([[0.0], [0.0], [2.0], [3.0], [3.0]])
    offsets = np.asarray([-5, -4, -3, -2, -1])
    selected = api.boundary_indices(
        vectors,
        lower=np.asarray([0.0]),
        upper=np.asarray([3.0]),
        scale=np.asarray([1.0]),
        history_offsets=offsets,
        k=2,
    )
    assert tuple(selected) == (1, 4)  # newest low, then newest high


def test_greedy_minimax_ties_and_nested_covering_radius_are_exact() -> None:
    api = _api()
    vectors = np.asarray([[0.0], [2.0], [4.0], [6.0]])
    offsets = np.asarray([-4, -3, -2, -1])
    selected = api.greedy_minimax_indices(
        vectors,
        scale=np.asarray([1.0]),
        history_offsets=offsets,
        k=3,
    )
    assert tuple(selected) == (2, 0, 3)  # newest tied medoid, farthest, newest tie

    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=0.0)
    prefixes = [
        api.select_support(ambiguity, method="minimax_subset", k=k)
        for k in REQUESTED_K
    ]
    assert tuple(prefixes[-1].selected_indices[:8]) == tuple(prefixes[2].selected_indices)
    radii = [api.support_covering_radius(ambiguity, item) for item in prefixes]
    assert all(left >= right - 1e-12 for left, right in zip(radii, radii[1:]))
    assert all(item.approximation == "deterministic_greedy_k_center" for item in prefixes)


@pytest.mark.parametrize("method", ORDINARY_METHODS)
def test_all_ordinary_selectors_obey_k_index_and_uniform_weight_contract(method: str) -> None:
    api = _api()
    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=0.25)
    for k in REQUESTED_K:
        kwargs = {"replicate_seed": 41_000_123} if method == "random_empirical" else {}
        support = api.select_support(ambiguity, method=method, k=k, **kwargs)
        assert support.method == method
        assert support.requested_k == support.actual_k == k
        assert len(set(support.selected_indices)) == k
        assert tuple(support.weights) == pytest.approx((1.0 / k,) * k)
        assert support.uses_oracle is False and support.upper_bound_only is False
        assert not any(
            callable(getattr(support, capability, None))
            for capability in ("execute", "apply", "commit", "schedule", "repair")
        )
        for matrix in support.matrices:
            assert matrix.shape == (4, 4)
            assert np.issubdtype(matrix.dtype, np.integer)
            assert np.all(matrix >= 0) and matrix.max() <= 8
            assert np.all(np.diag(matrix) == 0)


def test_oracle_is_exact_separate_nearest_zero_and_cannot_pollute_ordinary_cache() -> None:
    api = _api()
    ambiguity = api.build_empirical_ambiguity_set(_view(api), calibration_radius=0.0)
    before = {
        method: _fingerprint(
            api.select_support(
                ambiguity,
                method=method,
                k=8,
                **({"replicate_seed": 41_000_777} if method == "random_empirical" else {}),
            )
        )
        for method in ORDINARY_METHODS
    }
    truth_a = np.zeros((4, 4), dtype=np.int64)
    truth_a[0, 1] = 8
    truth_b = np.zeros((4, 4), dtype=np.int64)
    truth_b[2, 3] = 7
    oracle_a = api.oracle_support_upper_bound(ambiguity, truth=truth_a, k=8)
    oracle_b = api.oracle_support_upper_bound(ambiguity, truth=truth_b, k=8)
    assert oracle_a.uses_oracle and oracle_a.upper_bound_only
    assert oracle_b.uses_oracle and oracle_b.upper_bound_only
    np.testing.assert_array_equal(oracle_a.matrices[0], truth_a)
    np.testing.assert_array_equal(oracle_b.matrices[0], truth_b)
    assert api.truth_nearest_descriptor_distance(ambiguity, oracle_a, truth_a) == 0.0
    assert api.truth_nearest_descriptor_distance(ambiguity, oracle_b, truth_b) == 0.0
    assert _fingerprint(oracle_a) != _fingerprint(oracle_b)

    after = {
        method: _fingerprint(
            api.select_support(
                ambiguity,
                method=method,
                k=8,
                **({"replicate_seed": 41_000_777} if method == "random_empirical" else {}),
            )
        )
        for method in ORDINARY_METHODS
    }
    assert before == after
    suspicious = {
        name for name in vars(api)
        if name.lower() in {"oracle_cache", "_oracle_cache", "last_truth", "_last_truth"}
    }
    assert not suspicious


def test_phase3b_module_has_no_forbidden_imports_or_phase4_api() -> None:
    api = _api()
    source_path = Path(inspect.getsourcefile(api) or "")
    imported = _resolved_imports(
        source_path.read_text(encoding="utf-8"), api.__name__
    )
    assert not any(
        name == forbidden or name.startswith(forbidden + ".")
        for name in imported for forbidden in FORBIDDEN_IMPORTS
    )
    assert not _forbidden_public_names(getattr(api, "__all__", ()))

    # Guard the guard: relative imports and decorated public names must not
    # evade the source audit through spelling/casing/underscore variations.
    assert "rlccl.uncertainty.execution" in _resolved_imports(
        "from . import execution", "rlccl.uncertainty.ambiguity"
    )
    assert "rlccl.uncertainty.execution" in _resolved_imports(
        "from .execution import hidden", "rlccl.uncertainty.ambiguity"
    )
    assert _forbidden_public_names(
        ("build_prefix", "ROBUST_SCORE_report", "Truth_Token_IdFactory")
    ) == {"build_prefix", "ROBUST_SCORE_report", "Truth_Token_IdFactory"}

    script = """
import json, sys
import rlccl.uncertainty
before = set(sys.modules)
import rlccl.uncertainty.ambiguity
new = set(sys.modules) - before
forbidden = ('torch', 'rlccl.uncertainty.execution', 'rlccl.envs.decoder', 'rlccl.scheduling')
print(json.dumps(sorted(name for name in new if name in forbidden or name.startswith(tuple(x + '.' for x in forbidden)))))
"""
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
