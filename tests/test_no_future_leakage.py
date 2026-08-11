import numpy as np

from rlccl.traffic.moment_estimator import SlidingMomentEstimator


def _traffic(value):
    matrix = np.full((3, 3), value, dtype=np.int64)
    np.fill_diagonal(matrix, 0)
    return matrix


def test_current_matrix_changes_only_current_z_not_history_moments():
    estimator = SlidingMomentEstimator(3, window_size=4, min_history=2)
    estimator.update(_traffic(1))
    estimator.update(_traffic(2))
    state_before = estimator.state_dict()
    mean_ref = _traffic(1).astype(float)
    var_ref = np.ones((3, 3), dtype=float)
    np.fill_diagonal(var_ref, 0)

    low = estimator.get_context(_traffic(0), mean_ref, var_ref)
    high = estimator.get_context(_traffic(8), mean_ref, var_ref)

    np.testing.assert_array_equal(low.mean_matrix, high.mean_matrix)
    np.testing.assert_array_equal(low.var_matrix, high.var_matrix)
    assert not np.array_equal(low.current_send_z, high.current_send_z)
    assert estimator.state_dict() == state_before


def test_only_update_mutates_history():
    estimator = SlidingMomentEstimator(3, window_size=2, min_history=1)
    reference = np.zeros((3, 3))
    estimator.get_context(_traffic(3), reference, reference)
    assert estimator.history_length == 0
    estimator.update(_traffic(3))
    assert estimator.history_length == 1
    context = estimator.get_context(_traffic(7), reference, reference)
    np.testing.assert_array_equal(context.mean_matrix, _traffic(3))
