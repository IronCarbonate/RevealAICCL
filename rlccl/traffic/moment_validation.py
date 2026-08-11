"""Validation of sliding-window moment constraints on final integer traffic."""

from typing import Any, Sequence

import numpy as np

from .matrix_utils import validate_traffic_matrix
from .types import TrafficSequence


def compute_window_moments(
    matrices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute element-wise population mean and variance for one window."""
    if not matrices:
        raise ValueError("At least one traffic matrix is required")
    for matrix in matrices:
        validate_traffic_matrix(matrix)
    stacked = np.stack(matrices, axis=0).astype(np.float64)
    return stacked.mean(axis=0), stacked.var(axis=0, ddof=0)


def relative_l2_error(
    actual: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Return ``||actual-reference||_2 / (||reference||_2 + eps)``."""
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if actual_array.shape != reference_array.shape:
        raise ValueError(
            f"Moment shape mismatch: {actual_array.shape} != {reference_array.shape}"
        )
    return float(
        np.linalg.norm(actual_array - reference_array)
        / (np.linalg.norm(reference_array) + eps)
    )


def validate_sequence_moment_bounds(sequence: TrafficSequence) -> dict[str, Any]:
    """Validate every configured complete sliding window in ``sequence``."""
    matrices = sequence.matrices
    if not matrices:
        raise ValueError("Traffic sequence must contain at least one matrix")
    for matrix in matrices:
        validate_traffic_matrix(matrix)
    if sequence.mean_ref.shape != matrices[0].shape or sequence.var_ref.shape != matrices[0].shape:
        raise ValueError("Reference moments must match traffic matrix shape")

    window_size = int(sequence.metadata.get("window_size", len(matrices)))
    validation_stride = int(sequence.metadata.get("validation_stride", 1))
    allowed_fraction = float(sequence.metadata.get("allowed_violation_fraction", 0.0))
    if window_size <= 0 or window_size > len(matrices):
        raise ValueError("window_size must be in [1, sequence_length]")
    if validation_stride <= 0:
        raise ValueError("validation_stride must be positive")
    if not 0.0 <= allowed_fraction <= 1.0:
        raise ValueError("allowed_violation_fraction must be in [0, 1]")

    windows = []
    for start in range(0, len(matrices) - window_size + 1, validation_stride):
        mean, variance = compute_window_moments(matrices[start : start + window_size])
        mean_error = relative_l2_error(mean, sequence.mean_ref)
        var_error = relative_l2_error(variance, sequence.var_ref)
        passed = (
            mean_error <= sequence.bounds.epsilon_mean
            and var_error <= sequence.bounds.epsilon_var
        )
        windows.append(
            {
                "start": start,
                "end": start + window_size,
                "mean_error": mean_error,
                "var_error": var_error,
                "passed": passed,
            }
        )

    violations = sum(not window["passed"] for window in windows)
    violation_fraction = violations / len(windows)
    return {
        "passed": violation_fraction <= allowed_fraction,
        "num_windows": len(windows),
        "num_violations": violations,
        "violation_fraction": violation_fraction,
        "allowed_violation_fraction": allowed_fraction,
        "max_mean_error": max(window["mean_error"] for window in windows),
        "max_var_error": max(window["var_error"] for window in windows),
        "windows": windows,
    }
