from dataclasses import dataclass

import numpy as np

from rlccl.models import RidgeMultiOutput, TrafficPredictorSuite
from rlccl.models.traffic_predictor import build_history_examples


@dataclass
class _Sequence:
    sequence_id: str
    family: str
    seed: int
    matrices: list[np.ndarray]


def _matrices(offset=0):
    values = []
    for step in range(12):
        matrix = np.zeros((3, 3), dtype=np.int64)
        matrix[0, 1] = 1 + (step + offset) % 3
        matrix[1, 2] = 2 + (step // 2) % 2
        matrix[2, 0] = 1 + (step // 3) % 2
        values.append(matrix)
    return values


def test_ridge_multioutput_round_trip():
    x = np.arange(40, dtype=np.float64).reshape(10, 4)
    y = np.stack([x[:, 0] + x[:, 1], x[:, 2] - x[:, 3]], axis=1)
    model = RidgeMultiOutput(alpha=0.1).fit(x, y)
    prediction = model.predict(x)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()


def test_predictor_features_are_strictly_history_only(tmp_path):
    original = _Sequence("a", "family", 1, _matrices())
    changed_matrices = [item.copy() for item in original.matrices]
    changed_matrices[4][0, 1] += 5
    changed = _Sequence("b", "family", 2, changed_matrices)
    groups = np.empty((0, 3, 3), dtype=np.float64)
    examples_a = build_history_examples(
        [original],
        group_coefficients=groups,
        history_window=4,
        recent_steps=3,
        min_history=4,
    )
    examples_b = build_history_examples(
        [changed],
        group_coefficients=groups,
        history_window=4,
        recent_steps=3,
        min_history=4,
    )
    step_a = next(item for item in examples_a if item["step"] == 4)
    step_b = next(item for item in examples_b if item["step"] == 4)
    assert np.array_equal(step_a["moment_features"], step_b["moment_features"])
    assert np.array_equal(step_a["recent_features"], step_b["recent_features"])
    assert not np.array_equal(step_a["target"], step_b["target"])
    assert step_a["history_last_step"] == 3

    train = build_history_examples(
        [original],
        group_coefficients=groups,
        history_window=4,
        recent_steps=3,
        min_history=4,
    )
    suite = TrafficPredictorSuite(num_nodes=3, group_count=0, alpha=1.0).fit(train)
    before = suite.predict(examples_b)
    path = tmp_path / "predictor.npz"
    suite.save(str(path))
    loaded = TrafficPredictorSuite.load(str(path))
    after = loaded.predict(examples_b)
    for method in (
        "constant",
        "previous",
        "moment_only",
        "recent_history",
        "oracle_current_summary",
    ):
        assert np.allclose(before[method]["continuous"], after[method]["continuous"])
        assert np.array_equal(before[method]["hotspot"], after[method]["hotspot"])


def test_complete_sequences_remain_independent_examples():
    sequences = [
        _Sequence("train-sequence", "family", 1, _matrices()),
        _Sequence("test-sequence", "family", 2, _matrices(offset=1)),
    ]
    examples = build_history_examples(
        sequences,
        group_coefficients=np.empty((0, 3, 3)),
        history_window=4,
        recent_steps=3,
        min_history=4,
    )
    by_sequence = {name: [item for item in examples if item["sequence_id"] == name] for name in ("train-sequence", "test-sequence")}
    assert by_sequence["train-sequence"]
    assert by_sequence["test-sequence"]
    assert not ({item["sequence_id"] for item in by_sequence["train-sequence"]} & {item["sequence_id"] for item in by_sequence["test-sequence"]})
