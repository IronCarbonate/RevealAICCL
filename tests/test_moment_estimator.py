import copy

import numpy as np
import pytest

from rlccl.traffic.moment_estimator import SlidingMomentEstimator


def _matrix(a, b, c, d):
    return np.array([[0, a, b], [c, 0, d], [a, c, 0]], dtype=np.int64)


def test_warmup_confidence_and_history_window():
    estimator = SlidingMomentEstimator(3, window_size=2, min_history=2)
    reference = np.zeros((3, 3))
    current = _matrix(1, 2, 3, 4)
    cold = estimator.get_context(current, reference, reference)
    assert cold.history_length == 0
    assert cold.confidence == 0.0
    assert not cold.is_warm

    estimator.update(current)
    warming = estimator.get_context(current, reference, reference)
    assert warming.confidence == 0.5
    assert not warming.is_warm

    estimator.update(_matrix(2, 3, 4, 5))
    warm = estimator.get_context(current, reference, reference)
    assert warm.confidence == 1.0
    assert warm.is_warm
    estimator.update(_matrix(3, 4, 5, 6))
    assert estimator.history_length == 2


def test_state_dict_round_trip_and_independent_arrays():
    estimator = SlidingMomentEstimator(3, window_size=3, min_history=1)
    estimator.update(_matrix(1, 2, 3, 4))
    state = estimator.state_dict()
    restored = SlidingMomentEstimator(3, window_size=3, min_history=1)
    restored.load_state_dict(copy.deepcopy(state))
    assert restored.state_dict() == state
    state["history"][0][0][1] = 99
    assert restored.state_dict()["history"][0][0][1] == 1


def test_zero_variance_z_score_is_finite_and_clipped():
    estimator = SlidingMomentEstimator(3, window_size=2, min_history=1, z_clip=10)
    constant = _matrix(1, 1, 1, 1)
    estimator.update(constant)
    context = estimator.get_context(
        _matrix(8, 8, 8, 8), np.zeros((3, 3)), np.zeros((3, 3))
    )
    assert np.all(np.isfinite(context.current_send_z))
    assert np.all(np.isfinite(context.current_recv_z))
    assert np.max(np.abs(context.current_send_z)) <= 10


def test_send_std_comes_from_historical_row_totals():
    estimator = SlidingMomentEstimator(3, window_size=3, min_history=1)
    history = [_matrix(1, 2, 3, 4), _matrix(2, 4, 1, 3), _matrix(3, 6, 2, 2)]
    for matrix in history:
        estimator.update(matrix)
    context = estimator.get_context(history[-1], np.zeros((3, 3)), np.zeros((3, 3)))
    expected = np.stack(history).sum(axis=2).std(axis=0, ddof=0)
    np.testing.assert_allclose(context.send_std, expected)


def test_state_config_mismatch_is_rejected():
    state = SlidingMomentEstimator(3, 2, 1).state_dict()
    with pytest.raises(ValueError, match="num_nodes mismatch"):
        SlidingMomentEstimator(4, 2, 1).load_state_dict(state)
