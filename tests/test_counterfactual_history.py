import numpy as np

from rlccl.evaluation.counterfactual import (
    CounterfactualHistoryPair,
    action_edit_distance,
    context_distance,
    context_from_prior_history,
    edge_use_l1,
    sparse_schedule,
)


def _matrix(a, b, c, d):
    return np.asarray(
        [[0, a, b], [c, 0, d], [b, a, 0]], dtype=np.int64
    )


def test_same_current_matrix_different_history_is_history_only():
    current = _matrix(2, 1, 3, 1)
    low = tuple(_matrix(1, 1, 1, 1) for _ in range(4))
    high = tuple(_matrix(5, 0, 4, 2) for _ in range(4))
    mean_ref = np.full((3, 3), 2.0)
    var_ref = np.full((3, 3), 1.0)
    np.fill_diagonal(mean_ref, 0.0)
    np.fill_diagonal(var_ref, 0.0)
    pair = CounterfactualHistoryPair(
        pair_id="pair-0",
        family="test",
        seed=42,
        current_matrix=current,
        history_a=low,
        history_b=high,
        mean_ref=mean_ref,
        var_ref=var_ref,
    )

    context_a = context_from_prior_history(
        pair.history_a,
        pair.current_matrix,
        pair.mean_ref,
        pair.var_ref,
        window_size=4,
        min_history=2,
    )
    context_b = context_from_prior_history(
        pair.history_b,
        pair.current_matrix,
        pair.mean_ref,
        pair.var_ref,
        window_size=4,
        min_history=2,
    )

    assert context_a.history_length == context_b.history_length == 4
    assert np.array_equal(context_a.mean_matrix, np.mean(low, axis=0))
    assert np.array_equal(context_b.mean_matrix, np.mean(high, axis=0))
    assert context_distance(context_a, context_b)["combined"] > 0.5
    assert not any(np.array_equal(item, current) for item in pair.history_a)
    assert not any(np.array_equal(item, current) for item in pair.history_b)


def test_schedule_comparison_helpers_cover_complete_schedule():
    left = [
        np.asarray([[1, 0], [0, 1]], dtype=np.int64),
        np.asarray([[0, 1], [0, 0]], dtype=np.int64),
    ]
    right = [
        np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        np.asarray([[0, 1], [0, 0]], dtype=np.int64),
    ]
    assert sparse_schedule(left) == [[0, 0, 0], [0, 1, 1], [1, 0, 1]]
    assert edge_use_l1(left, right) == 2
    assert action_edit_distance([(0, 0), (1, 1)], [(0, 1), (1, 1)]) == 1
