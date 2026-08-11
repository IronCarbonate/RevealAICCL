"""Red-contract tests for Phase 2 / Gate H1 predictability.

Production imports are intentionally delayed until each test executes.  This
keeps the module collectable before ``rlccl.prediction`` exists while still
making the red phase fail honestly instead of being skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


FAMILIES = (
    "regime_switching_long",
    "stochastic_volatility",
    "rare_shock_recovery",
    "hotspot_random_walk",
    "same_moments_different_dynamics",
)
BASE_SEEDS = (42, 142, 242)
VARIANTS = ("smooth", "random_switching", "long_regime", "shock_recovery")
METHODS = (
    "long_term_mean",
    "previous_value",
    "ewma",
    "moment_only",
    "recent_history_mlp",
    "causal_tcn",
    "quantile_scenario",
)
PRIMARY_TARGETS = (
    "total_traffic",
    "source_load_vector",
    "destination_load_vector",
)


def _api() -> SimpleNamespace:
    return SimpleNamespace(
        data=importlib.import_module("rlccl.prediction.data"),
        models=importlib.import_module("rlccl.prediction.models"),
        calibration=importlib.import_module("rlccl.prediction.calibration"),
        statistics=importlib.import_module("rlccl.prediction.statistics"),
        artifacts=importlib.import_module("rlccl.prediction.artifacts"),
    )


@dataclass
class _Sequence:
    sequence_id: str
    family: str
    seed: int
    matrices: list[np.ndarray]
    metadata: dict[str, Any]


def _matrices(length: int = 24, nodes: int = 3) -> list[np.ndarray]:
    result = []
    for step in range(length):
        matrix = np.zeros((nodes, nodes), dtype=np.int64)
        matrix[0, 1] = 1 + step % 4
        matrix[1, 2] = 1 + (step // 2) % 3
        matrix[2, 0] = (step // 3) % 2
        result.append(matrix)
    return result


def _example_at(examples: Any, step: int) -> int:
    matches = np.flatnonzero(np.asarray(examples.steps) == step)
    assert len(matches) == 1
    return int(matches[0])


def _synthetic_raw_rows() -> list[dict[str, Any]]:
    rows = []
    for family in FAMILIES:
        for seed in BASE_SEEDS:
            common = {
                "sequence_id": f"{family}-{seed}",
                "family": family,
                "base_seed": seed,
                "target": "total_traffic",
                "raw_step_count": 1016,
            }
            rows.append(
                {
                    **common,
                    "method": "previous_value",
                    "rmse": 2.0,
                    "mae": 1.0,
                    "r2": 0.0,
                    "spearman": 0.0,
                    "delta_rmse": 0.0,
                }
            )
            rows.append(
                {
                    **common,
                    "method": "selected_recent",
                    "rmse": 1.0,
                    "mae": 0.5,
                    "r2": 0.5,
                    "spearman": 0.5,
                    "delta_rmse": 1.0,
                }
            )
    return rows


def _passing_gate_evidence() -> dict[str, Any]:
    return {
        "pooled_total_ci_lower": 0.01,
        "seed_mean_deltas": {seed: 0.02 for seed in BASE_SEEDS},
        "family_mean_deltas": {
            family: (0.02 if index < 4 else 0.0)
            for index, family in enumerate(FAMILIES)
        },
        "primary_ci_lowers": {target: 0.01 for target in PRIMARY_TARGETS},
        "lofo": {
            "aggregate_mean_delta": 0.01,
            "family_mean_deltas": {
                family: (0.01 if index < 3 else 0.0)
                for index, family in enumerate(FAMILIES)
            },
            "relative_rmse_changes": {family: 0.05 for family in FAMILIES},
        },
        "probability": {
            "overall_interval_calibration_error": 0.05,
            "family_interval_calibration_errors": {
                family: 0.10 for family in FAMILIES
            },
            "overall_scenario_calibration_error": 0.10,
        },
        "tail": {"event_count": 10, "recall": 0.70},
        "integrity_tests_passed": True,
        "supervisor_veto": False,
    }


def test_prediction_package_is_present_for_red_contract() -> None:
    assert importlib.util.find_spec("rlccl.prediction") is not None


def test_formal_specs_have_exact_75_sequences_splits_seeds_and_variants() -> None:
    api = _api()
    specs = tuple(api.data.build_formal_sequence_specs())
    assert len(specs) == 75
    assert tuple(api.data.FORMAL_FAMILIES) == FAMILIES
    assert tuple(api.data.FORMAL_BASE_SEEDS) == BASE_SEEDS
    counts = {split: sum(item.split == split for item in specs) for split in (
        "fit", "validation", "calibration", "test"
    )}
    assert counts == {"fit": 30, "validation": 15, "calibration": 15, "test": 15}
    assert len({item.sequence_id for item in specs}) == 75
    assert len({item.actual_seed for item in specs}) == 75
    for item in specs:
        family_index = FAMILIES.index(item.family)
        seed_index = BASE_SEEDS.index(item.base_seed)
        assert item.actual_seed == (
            item.base_seed + family_index * 1_000_000 + item.sequence_index * 10_000
        )
        expected_split = ("fit", "fit", "validation", "calibration", "test")[
            item.sequence_index
        ]
        assert item.split == expected_split
        expected_variant = (
            VARIANTS[(seed_index + item.sequence_index) % 4]
            if item.family == "same_moments_different_dynamics"
            else None
        )
        assert item.dynamics_variant == expected_variant
        assert item.sequence_length == 1024
        assert item.generator_config["mean_level"] == 2.0
        assert item.generator_config["std_level"] == 1.5
        assert item.generator_config["max_entry"] == 8
        assert item.generator_config["calibration_candidates"] == 1


def test_split_records_require_mutually_exclusive_ids_and_digests() -> None:
    api = _api()
    base = np.zeros((3, 3), dtype=np.int64)
    changed = base.copy()
    changed[0, 1] = 1
    first_digest = api.data.sequence_digest((base,))
    second_digest = api.data.sequence_digest((changed,))
    assert first_digest != second_digest
    records = (
        {"sequence_id": "fit-0", "digest": first_digest, "split": "fit"},
        {"sequence_id": "test-0", "digest": second_digest, "split": "test"},
    )
    api.data.validate_split_records(records)
    with pytest.raises(ValueError, match="sequence.*overlap|id.*overlap|duplicate"):
        api.data.validate_split_records(records + ({**records[0], "split": "test"},))
    with pytest.raises(ValueError, match="digest.*overlap|duplicate"):
        api.data.validate_split_records(
            records + ({"sequence_id": "test-1", "digest": first_digest, "split": "test"},)
        )


@pytest.mark.parametrize("mutation", ("current", "future", "metadata"))
def test_history_features_ignore_current_future_and_latent_metadata(mutation: str) -> None:
    api = _api()
    matrices = _matrices()
    original = _Sequence(
        "sequence-a",
        "family",
        42,
        [matrix.copy() for matrix in matrices],
        {"latent_regime": ["future"] * len(matrices), "shock_flags": [1] * len(matrices)},
    )
    changed = _Sequence(
        "sequence-b",
        "family",
        42,
        [matrix.copy() for matrix in matrices],
        {"latent_regime": ["other"] * len(matrices), "shock_flags": [0] * len(matrices)},
    )
    if mutation == "current":
        changed.matrices[8][0, 1] += 100
    elif mutation == "future":
        changed.matrices[9][0, 1] += 100
    groups = np.empty((0, 3, 3), dtype=np.float64)
    first = api.data.build_history_examples(original, group_coefficients=groups)
    second = api.data.build_history_examples(changed, group_coefficients=groups)
    left, right = _example_at(first, 8), _example_at(second, 8)
    np.testing.assert_array_equal(first.recent_history[left], second.recent_history[right])
    np.testing.assert_array_equal(first.moment_features[left], second.moment_features[right])
    np.testing.assert_array_equal(first.previous_targets[left], second.previous_targets[right])
    np.testing.assert_array_equal(first.ewma_targets[left], second.ewma_targets[right])
    assert first.history_last_steps[left] == second.history_last_steps[right] == 7
    if mutation == "current":
        assert not np.array_equal(first.targets[left], second.targets[right])
    else:
        np.testing.assert_array_equal(first.targets[left], second.targets[right])


def test_all_methods_are_registered_without_oracle_aliases() -> None:
    api = _api()
    assert tuple(api.models.METHOD_NAMES) == METHODS
    assert not ({"oracle", "current_truth", "ridge_as_tcn"} & set(api.models.METHOD_NAMES))


def test_ewma_uses_x0_initial_value_and_history_only_recurrence() -> None:
    api = _api()
    summaries = np.asarray([[2.0], [6.0], [10.0], [14.0]])
    predictions = api.models.ewma_history_predictions(summaries, alpha=0.30)
    expected = np.asarray([[2.0], [3.2], [5.24], [7.868]])
    np.testing.assert_allclose(predictions, expected, atol=1e-12, rtol=0)
    changed = summaries.copy()
    changed[2] = 1000.0
    counterfactual = api.models.ewma_history_predictions(changed, alpha=0.30)
    np.testing.assert_array_equal(predictions[:2], counterfactual[:2])


def test_fit_standardizers_are_fit_only_and_application_is_pure() -> None:
    api = _api()
    fit_x = np.arange(96, dtype=np.float64).reshape(8, 3, 4)
    fit_y = np.arange(40, dtype=np.float64).reshape(8, 5)
    standardizers = api.data.fit_standardizers(fit_x, fit_y)
    before = standardizers.state_dict()
    validation = np.full((4, 3, 4), 1e9)
    standardizers.transform_inputs(validation)
    after = standardizers.state_dict()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    np.testing.assert_allclose(standardizers.input_mean, fit_x.mean(axis=(0, 1)))
    np.testing.assert_allclose(standardizers.target_mean, fit_y.mean(axis=0))


def test_recent_model_selection_is_validation_only_with_mlp_tie_break() -> None:
    api = _api()
    scores = {
        "recent_history_mlp": np.asarray([2.0, 1.0, 3.0]),
        "causal_tcn": np.asarray([1.0, 1.0, 1.0]),
    }
    assert api.models.select_recent_backbone(scores) == "causal_tcn"
    tie = {"recent_history_mlp": np.ones(3), "causal_tcn": np.ones(3) + 1e-13}
    assert api.models.select_recent_backbone(tie) == "recent_history_mlp"
    assert tuple(api.models.select_recent_backbone.__code__.co_varnames[:1]) == (
        "validation_sequence_rmse",
    )


@pytest.mark.parametrize("held_out", FAMILIES)
def test_lofo_excludes_held_out_family_from_fit_validation_and_calibration(
    held_out: str,
) -> None:
    api = _api()
    fold = api.data.build_lofo_fold(api.data.build_formal_sequence_specs(), held_out)
    for split in ("fit", "validation", "calibration"):
        assert held_out not in {item.family for item in fold[split]}
    assert len(fold["test"]) == 3
    assert {item.family for item in fold["test"]} == {held_out}
    assert {item.base_seed for item in fold["test"]} == set(BASE_SEEDS)


def _small_tcn(api: SimpleNamespace, seed: int = 20260731) -> Any:
    return api.models.NumpyCausalTCN(
        input_dim=3,
        output_dim=2,
        kernel_size=3,
        hidden_channels=8,
        epochs=40,
        batch_size=256,
        learning_rate=5e-3,
        l2=1e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        seed=seed,
    )


@pytest.mark.parametrize(
    "parameter,index",
    (
        ("kernel", (0, 0, 0)),
        ("kernel", (2, 2, 7)),
        ("conv_bias", (3,)),
        ("head_weight", (0, 0)),
        ("head_bias", (1,)),
    ),
)
def test_tcn_finite_difference_gradient(parameter: str, index: tuple[int, ...]) -> None:
    api = _api()
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 8, 3))
    y = rng.normal(size=(4, 2))
    model = _small_tcn(api, seed=11)
    _, gradients = model.loss_and_gradients(x, y)
    state = model.state_dict()
    epsilon = 1e-6
    plus = {name: np.array(value, copy=True) for name, value in state.items()}
    minus = {name: np.array(value, copy=True) for name, value in state.items()}
    plus[parameter][index] += epsilon
    minus[parameter][index] -= epsilon
    model.load_state_dict(plus)
    high = model.loss(x, y)
    model.load_state_dict(minus)
    low = model.loss(x, y)
    numerical = (high - low) / (2 * epsilon)
    np.testing.assert_allclose(gradients[parameter][index], numerical, rtol=2e-4, atol=2e-5)


def test_tcn_kernel_updates_and_causal_lag_loss_decreases() -> None:
    api = _api()
    rng = np.random.default_rng(17)
    x = rng.normal(size=(160, 8, 3))
    y = np.stack(
        [np.tanh(1.7 * x[:, -3, 0]), x[:, -2, 1] - 0.5 * x[:, -1, 2]],
        axis=1,
    )
    model = _small_tcn(api, seed=19)
    initial_kernel = model.state_dict()["kernel"].copy()
    initial_loss = model.loss(x, y)
    history = model.fit(x, y)
    final_loss = model.loss(x, y)
    assert len(history) == 40
    assert not np.array_equal(initial_kernel, model.state_dict()["kernel"])
    assert final_loss < initial_loss * 0.65
    assert np.isfinite(history).all()


def test_tcn_hidden_is_causal_and_uses_shared_kernel() -> None:
    api = _api()
    rng = np.random.default_rng(23)
    x = rng.normal(size=(3, 8, 3))
    changed = x.copy()
    changed[:, 5:, :] += 1000.0
    model = _small_tcn(api, seed=29)
    original_hidden = model.temporal_hidden(x)
    changed_hidden = model.temporal_hidden(changed)
    np.testing.assert_allclose(original_hidden[:, :5], changed_hidden[:, :5], atol=1e-12)
    kernel = model.state_dict()["kernel"]
    assert kernel.shape == (3, 3, 8)
    assert model.representation_dim == 16  # final hidden concatenated with temporal mean


def test_tcn_same_seed_is_deterministic_and_npz_round_trip(tmp_path: Path) -> None:
    api = _api()
    rng = np.random.default_rng(31)
    x = rng.normal(size=(64, 8, 3))
    y = rng.normal(size=(64, 2))
    first = _small_tcn(api, seed=37)
    second = _small_tcn(api, seed=37)
    first.fit(x, y)
    second.fit(x, y)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    for name in first.state_dict():
        np.testing.assert_array_equal(first.state_dict()[name], second.state_dict()[name])
    path = tmp_path / "tcn.npz"
    first.save(path)
    loaded = api.models.NumpyCausalTCN.load(path)
    np.testing.assert_array_equal(first.predict(x), loaded.predict(x))


def test_neural_model_hyperparameters_match_frozen_protocol() -> None:
    api = _api()
    mlp = api.models.RecentHistoryMLP(input_dim=8 * 5, output_dim=5, seed=20260731)
    assert mlp.hidden_layer_sizes == (32,)
    assert mlp.activation == "tanh"
    assert mlp.solver == "adam"
    assert mlp.alpha == 1e-4
    assert mlp.batch_size == 256
    assert mlp.learning_rate_init == 1e-3
    assert mlp.max_iter == 80
    tcn = api.models.NumpyCausalTCN(input_dim=5, output_dim=5, seed=20260731)
    assert (tcn.kernel_size, tcn.hidden_channels, tcn.epochs) == (3, 8, 40)
    assert (tcn.batch_size, tcn.learning_rate, tcn.l2) == (256, 5e-3, 1e-4)


def test_residual_quantiles_are_ordered_and_joint_scenarios_preserve_rows() -> None:
    api = _api()
    calibration_point = np.zeros((20, 3), dtype=np.float64)
    residuals = np.asarray([[index, -index, index % 3] for index in range(20)], dtype=float)
    calibrator = api.calibration.ResidualCalibrator(seed=20260731).fit(
        calibration_point, residuals
    )
    point = np.asarray([[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]])
    prediction = calibrator.predict(point, stable_example_indices=np.asarray([7, 11]))
    assert prediction.scenarios.shape == (2, 64, 3)
    assert np.all(prediction.q10 <= prediction.q50)
    assert np.all(prediction.q50 <= prediction.q90)
    for example_index in range(2):
        sampled_residuals = prediction.scenarios[example_index] - point[example_index]
        for row in sampled_residuals:
            assert any(np.array_equal(row, candidate) for candidate in residuals)
    repeated = calibrator.predict(point, stable_example_indices=np.asarray([7, 11]))
    np.testing.assert_array_equal(prediction.scenarios, repeated.scenarios)


def test_quantile_interval_metrics_match_direct_componentwise_calculation() -> None:
    api = _api()
    truth = np.arange(8, dtype=np.float64).reshape(4, 2)
    lower = truth - 1.0
    upper = truth + 1.0
    lower[3, 1] = truth[3, 1] + 0.5
    metrics = api.statistics.quantile_interval_metrics(
        truth, lower, upper, nominal_coverage=0.80
    )
    covered = (truth >= lower) & (truth <= upper)
    assert metrics["coverage"] == pytest.approx(float(np.mean(covered)))
    assert metrics["coverage"] == pytest.approx(7 / 8)
    assert metrics["interval_width"] == pytest.approx(float(np.mean(upper - lower)))
    assert metrics["interval_width"] == pytest.approx(1.8125)
    assert metrics["calibration_error"] == pytest.approx(abs(7 / 8 - 0.80))


def test_64_scenario_central_envelope_coverage_uses_linear_quantiles() -> None:
    api = _api()
    offsets = np.concatenate(
        (
            -np.ones(8, dtype=np.float64),
            np.zeros(48, dtype=np.float64),
            np.ones(8, dtype=np.float64),
        )
    )
    scenarios = np.broadcast_to(offsets[None, :, None], (4, 64, 2)).copy()
    truth = np.asarray([[0.0, 1.0], [-1.0, 1.1], [2.0, 0.0], [0.5, -0.5]])
    metrics = api.statistics.scenario_envelope_metrics(
        truth, scenarios, nominal_coverage=0.80
    )
    expected_lower = np.quantile(scenarios, 0.10, axis=1, method="linear")
    expected_upper = np.quantile(scenarios, 0.90, axis=1, method="linear")
    np.testing.assert_allclose(metrics["lower"], expected_lower, rtol=0, atol=0)
    np.testing.assert_allclose(metrics["upper"], expected_upper, rtol=0, atol=0)
    assert scenarios.shape[1] == 64
    assert metrics["coverage"] == pytest.approx(6 / 8)
    assert metrics["calibration_error"] == pytest.approx(abs(6 / 8 - 0.80))


def test_tail_recall_reports_insufficient_instead_of_fabricating_value() -> None:
    api = _api()
    actual = np.asarray([0.0, 11.0, 12.0, 1.0])
    upper = np.asarray([0.0, 12.0, 8.0, 1.0])
    result = api.calibration.tail_event_recall(actual, upper, fit_threshold=10.0)
    assert result == {"status": "insufficient_events", "event_count": 2, "recall": None}


def test_primary_target_metrics_use_fixed_blocks_and_equal_component_weighting() -> None:
    api = _api()
    assert tuple(api.statistics.PRIMARY_TARGETS) == PRIMARY_TARGETS
    actual = {
        "total_traffic": np.asarray([[2.0], [4.0]]),
        "source_load_vector": np.asarray([[1.0, 3.0], [2.0, 4.0]]),
        "destination_load_vector": np.asarray([[2.0, 2.0], [1.0, 5.0]]),
    }
    predicted = {name: values + 1.0 for name, values in actual.items()}
    metrics = api.statistics.sequence_target_metrics(actual, predicted)
    for name in PRIMARY_TARGETS:
        assert metrics[name]["rmse"] == pytest.approx(1.0)
        assert metrics[name]["mae"] == pytest.approx(1.0)


def test_hotspot_argmax_uses_smallest_node_for_ties() -> None:
    api = _api()
    loads = np.asarray([[1.0, 5.0, 5.0], [7.0, 7.0, 2.0]])
    np.testing.assert_array_equal(
        api.statistics.hotspot_from_destination_loads(loads), np.asarray([1, 0])
    )


def test_family_stratified_bootstrap_is_10000_draw_deterministic_and_equal_weight() -> None:
    api = _api()
    families = np.repeat(np.asarray(FAMILIES, dtype=object), 3)
    deltas = np.repeat(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), 3)
    first = api.statistics.family_stratified_paired_bootstrap(
        deltas, families, samples=10_000, seed=20260731
    )
    second = api.statistics.family_stratified_paired_bootstrap(
        deltas, families, samples=10_000, seed=20260731
    )
    assert first.samples.shape == (10_000,)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_allclose(first.samples, 3.0)
    assert (first.mean, first.lower, first.upper) == pytest.approx((3.0, 3.0, 3.0))


def test_acf_and_positive_sequence_ess_follow_frozen_formula() -> None:
    api = _api()
    values = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0])
    acf = api.statistics.autocorrelation(values, max_lag=4)
    ess = api.statistics.positive_sequence_ess(values, max_lag=4)
    centered = values - values.mean()
    variance = float(centered @ centered)
    expected_lag1 = float(centered[:-1] @ centered[1:] / variance)
    assert acf[1] == pytest.approx(expected_lag1)
    positive = []
    for lag in range(1, 5):
        rho = float(centered[:-lag] @ centered[lag:] / variance)
        if rho <= 0:
            break
        positive.append(rho)
    assert ess == pytest.approx(len(values) / (1 + 2 * sum(positive)))


def test_gate_pass_requires_all_seven_preregistered_conditions() -> None:
    api = _api()
    result = api.statistics.evaluate_h1_gate(_passing_gate_evidence())
    assert result["decision"] == "PASS"
    assert tuple(result["conditions"]) == tuple(f"condition_{index}" for index in range(1, 8))
    assert all(result["conditions"].values())


def test_gate_fails_when_any_primary_ci_lower_is_not_strictly_positive() -> None:
    api = _api()
    evidence = _passing_gate_evidence()
    evidence["primary_ci_lowers"]["source_load_vector"] = 0.0
    result = api.statistics.evaluate_h1_gate(evidence)
    assert result["decision"] == "FAIL"
    assert result["conditions"]["condition_3"] is False


def test_gate_holds_when_tail_events_are_insufficient() -> None:
    api = _api()
    evidence = _passing_gate_evidence()
    evidence["tail"] = {"event_count": 9, "recall": None}
    result = api.statistics.evaluate_h1_gate(evidence)
    assert result["decision"] == "HOLD"
    assert result["conditions"]["condition_6"] is False
    assert "insufficient_events" in result["reasons"]


def test_gate_rejects_empty_family_and_lofo_maps() -> None:
    api = _api()
    evidence = _passing_gate_evidence()
    evidence["family_mean_deltas"] = {}
    evidence["lofo"]["family_mean_deltas"] = {}
    evidence["lofo"]["relative_rmse_changes"] = {}
    result = api.statistics.evaluate_h1_gate(evidence)
    assert result["decision"] == "FAIL"
    assert result["conditions"]["condition_2"] is False
    assert result["conditions"]["condition_4"] is False


def test_gate_tail_insufficiency_does_not_mask_an_explicit_failure() -> None:
    api = _api()
    evidence = _passing_gate_evidence()
    evidence["primary_ci_lowers"]["destination_load_vector"] = 0.0
    evidence["tail"] = {"event_count": 9, "recall": None}
    result = api.statistics.evaluate_h1_gate(evidence)
    assert result["decision"] == "FAIL"
    assert result["conditions"]["condition_3"] is False
    assert result["conditions"]["condition_6"] is False
    assert "insufficient_events" in result["reasons"]


@pytest.mark.parametrize(
    "corruption,condition",
    (
        ("seed_keys", "condition_2"),
        ("family_keys", "condition_2"),
        ("empty_family_calibration", "condition_5"),
        ("wrong_family_calibration", "condition_5"),
        ("lofo_relative_change_keys", "condition_4"),
    ),
)
def test_gate_requires_exact_seed_and_family_identities(
    corruption: str, condition: str
) -> None:
    api = _api()
    evidence = _passing_gate_evidence()
    if corruption == "seed_keys":
        evidence["seed_mean_deltas"] = {9001: 0.02, 9002: 0.02, 9003: 0.02}
    elif corruption == "family_keys":
        evidence["family_mean_deltas"] = {
            f"wrong_family_{index}": 0.02 for index in range(5)
        }
    elif corruption == "empty_family_calibration":
        evidence["probability"]["family_interval_calibration_errors"] = {}
    elif corruption == "wrong_family_calibration":
        evidence["probability"]["family_interval_calibration_errors"] = {
            f"wrong_family_{index}": 0.01 for index in range(5)
        }
    elif corruption == "lofo_relative_change_keys":
        evidence["lofo"]["relative_rmse_changes"] = {
            f"wrong_family_{index}": 0.05 for index in range(5)
        }
    else:
        raise AssertionError(corruption)
    result = api.statistics.evaluate_h1_gate(evidence)
    assert result["decision"] == "FAIL"
    assert result["conditions"][condition] is False


def test_manifest_records_splits_digests_generator_models_and_fixed_seeds() -> None:
    api = _api()
    specs = api.data.build_formal_sequence_specs()
    records = [
        {"sequence_id": item.sequence_id, "digest": f"digest-{index}", "split": item.split}
        for index, item in enumerate(specs)
    ]
    manifest = api.artifacts.build_manifest(specs=specs, sequence_records=records)
    assert manifest["schema_version"] == 1
    assert manifest["split_counts"] == {
        "fit": 30,
        "validation": 15,
        "calibration": 15,
        "test": 15,
    }
    assert manifest["random_seeds"] == {
        "model_shuffle_scenario_bootstrap": 20260731
    }
    assert manifest["methods"] == list(METHODS)
    assert manifest["model_hyperparameters"]["causal_tcn"]["kernel_size"] == 3
    assert len(manifest["sequence_records"]) == 75
    api.artifacts.validate_manifest(manifest)


def test_raw_rows_and_summary_are_finite_and_summary_is_recomputed() -> None:
    api = _api()
    raw_rows = _synthetic_raw_rows()
    api.artifacts.validate_raw_rows(raw_rows)
    summary = api.artifacts.recompute_summary(raw_rows)
    assert summary["independent_test_sequences"] == 15
    assert summary["raw_test_steps"] == 15 * 1016
    assert summary["methods"]["selected_recent"]["total_traffic"]["mean_delta_rmse"] == 1.0
    api.artifacts.validate_finite_tree(summary)
    corrupted = [dict(raw_rows[0], rmse=np.nan), *raw_rows[1:]]
    with pytest.raises(ValueError, match="NaN|finite"):
        api.artifacts.validate_raw_rows(corrupted)


def test_artifact_bundle_round_trip_can_be_reloaded_and_recomputed(tmp_path: Path) -> None:
    api = _api()
    specs = api.data.build_formal_sequence_specs()
    records = [
        {"sequence_id": item.sequence_id, "digest": f"digest-{index}", "split": item.split}
        for index, item in enumerate(specs)
    ]
    manifest = api.artifacts.build_manifest(specs=specs, sequence_records=records)
    raw_rows = _synthetic_raw_rows()
    summary = api.artifacts.recompute_summary(raw_rows)
    api.artifacts.write_artifact_bundle(
        tmp_path, manifest=manifest, raw_rows=raw_rows, summary=summary
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "manifest.json",
        "raw_sequence_metrics.csv",
        "summary.json",
    }
    loaded = api.artifacts.read_artifact_bundle(tmp_path)
    assert loaded["manifest"] == manifest
    api.artifacts.validate_raw_rows(loaded["raw_rows"])
    assert api.artifacts.recompute_summary(loaded["raw_rows"]) == loaded["summary"]
