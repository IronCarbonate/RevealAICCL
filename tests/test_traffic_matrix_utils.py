import numpy as np
import pytest

from rlccl.traffic.matrix_utils import (
    scenario_to_traffic_matrix,
    traffic_matrix_to_scenario,
    validate_traffic_matrix,
)


def test_traffic_matrix_round_trip_and_chunk_count():
    matrix = np.array([[0, 2, 0], [1, 0, 3], [0, 4, 0]])
    scenario = traffic_matrix_to_scenario(matrix, sequence_id="s")
    assert scenario["C"] == int(matrix.sum())
    assert np.asarray(scenario["initial_state"]).shape == (matrix.sum(), 3)
    np.testing.assert_array_equal(scenario_to_traffic_matrix(scenario), matrix)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.zeros((2, 3)), "square"),
        (np.array([[0, -1], [0, 0]]), "nonnegative"),
        (np.array([[0, 1.5], [0, 0]]), "integer-valued"),
        (np.array([[1, 0], [0, 0]]), "diagonal"),
    ],
)
def test_invalid_matrix_rejected(matrix, message):
    with pytest.raises(ValueError, match=message):
        validate_traffic_matrix(matrix)


def test_all_zero_matrix_is_valid_and_caller_visible():
    scenario = traffic_matrix_to_scenario(np.zeros((3, 3), dtype=int))
    assert scenario["C"] == 0
    assert scenario["V"] == 3
    del scenario["traffic_matrix"]
    np.testing.assert_array_equal(
        scenario_to_traffic_matrix(scenario), np.zeros((3, 3), dtype=int)
    )


def test_reconstruct_without_stored_matrix():
    matrix = np.array([[0, 2], [1, 0]])
    scenario = traffic_matrix_to_scenario(matrix)
    del scenario["traffic_matrix"]
    np.testing.assert_array_equal(scenario_to_traffic_matrix(scenario), matrix)


def test_ambiguous_allgather_scenario_cannot_be_reconstructed():
    scenario = {
        "initial_state": [[1, 0, 0]],
        "demands": [[0, 1, 1]],
    }
    with pytest.raises(ValueError, match="exactly one"):
        scenario_to_traffic_matrix(scenario)
