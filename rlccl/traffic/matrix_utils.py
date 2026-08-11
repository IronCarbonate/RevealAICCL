"""Conversions between All-to-All-V traffic matrices and chunk scenarios."""

from typing import Any

import numpy as np


def validate_traffic_matrix(matrix: np.ndarray) -> None:
    """Validate square, nonnegative, integer-valued, zero-diagonal traffic."""
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Traffic matrix must be square, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Traffic matrix must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError("Traffic matrix must contain only finite values")
    if np.any(array < 0):
        raise ValueError("Traffic matrix entries must be nonnegative")
    if not np.all(array == np.floor(array)):
        raise ValueError("Traffic matrix entries must be integer-valued")
    if np.any(np.diag(array) != 0):
        raise ValueError("Traffic matrix diagonal must be zero")


def traffic_matrix_to_scenario(matrix: np.ndarray, **metadata: Any) -> dict[str, Any]:
    """Convert ``X[src, dst] = k`` into chunk initial state and demands.

    An all-zero matrix is valid and produces zero chunks.  A caller that needs a
    non-empty collective can reject or repair it before calling this function.
    """
    validate_traffic_matrix(matrix)
    traffic = np.asarray(matrix, dtype=np.int64)
    num_nodes = traffic.shape[0]
    num_chunks = int(traffic.sum())
    initial_state = np.zeros((num_chunks, num_nodes), dtype=np.int64)
    demands = np.zeros((num_chunks, num_nodes), dtype=np.int64)

    chunk = 0
    for src in range(num_nodes):
        for dst in range(num_nodes):
            count = int(traffic[src, dst])
            if count:
                chunk_slice = slice(chunk, chunk + count)
                initial_state[chunk_slice, src] = 1
                demands[chunk_slice, dst] = 1
                chunk += count

    scenario = dict(metadata)
    scenario.update(
        {
            "type": "all_to_all_v",
            "V": num_nodes,
            "traffic_matrix": traffic.tolist(),
            "C": num_chunks,
            "initial_state": initial_state.tolist(),
            "demands": demands.tolist(),
        }
    )
    return scenario


def scenario_to_traffic_matrix(scenario: dict[str, Any]) -> np.ndarray:
    """Return a stored matrix or reconstruct an unambiguous All-to-All-V one."""
    if scenario.get("traffic_matrix") is not None:
        matrix = np.asarray(scenario["traffic_matrix"])
        validate_traffic_matrix(matrix)
        return matrix.astype(np.int64, copy=True)

    initial_state = np.asarray(scenario.get("initial_state"))
    demands = np.asarray(scenario.get("demands"))
    if initial_state.size == 0 and demands.size == 0 and int(scenario.get("C", -1)) == 0:
        num_nodes = int(scenario.get("V", 0))
        if num_nodes <= 0:
            raise ValueError("Empty scenarios need positive V metadata")
        return np.zeros((num_nodes, num_nodes), dtype=np.int64)
    if initial_state.ndim != 2 or demands.ndim != 2 or initial_state.shape != demands.shape:
        raise ValueError("Scenario needs same-shaped 2-D initial_state and demands")

    num_chunks, num_nodes = initial_state.shape
    matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    for chunk in range(num_chunks):
        sources = np.flatnonzero(initial_state[chunk])
        destinations = np.flatnonzero(demands[chunk])
        if len(sources) != 1 or len(destinations) != 1:
            raise ValueError(
                "Cannot reconstruct All-to-All-V traffic: each chunk must have "
                "exactly one source and one destination"
            )
        src, dst = int(sources[0]), int(destinations[0])
        if src == dst:
            raise ValueError("All-to-All-V chunks cannot demand their source node")
        matrix[src, dst] += 1

    validate_traffic_matrix(matrix)
    return matrix
