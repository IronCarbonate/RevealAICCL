"""Pooled signed-residual calibration for quantiles and joint scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibratedPrediction:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    scenarios: np.ndarray


class ResidualCalibrator:
    """Calibrate one selected point backbone without family-specific pools."""

    def __init__(self, seed: int = 20260731, scenario_count: int = 64):
        if scenario_count <= 0:
            raise ValueError("scenario_count must be positive")
        self.seed = int(seed)
        self.scenario_count = int(scenario_count)
        self.residuals: np.ndarray | None = None
        self.residual_quantiles: np.ndarray | None = None

    def fit(
        self, calibration_point: np.ndarray, signed_residuals: np.ndarray
    ) -> "ResidualCalibrator":
        points = np.asarray(calibration_point, dtype=np.float64)
        residuals = np.asarray(signed_residuals, dtype=np.float64)
        if points.ndim != 2 or residuals.ndim != 2 or points.shape != residuals.shape:
            raise ValueError("calibration points and signed residuals must share a 2-D shape")
        if not len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("calibration residuals must be nonempty and finite")
        self.residuals = residuals.copy()
        self.residual_quantiles = np.quantile(
            residuals, (0.10, 0.50, 0.90), axis=0, method="linear"
        )
        return self

    def predict(
        self, point: np.ndarray, *, stable_example_indices: np.ndarray
    ) -> CalibratedPrediction:
        if self.residuals is None or self.residual_quantiles is None:
            raise RuntimeError("ResidualCalibrator is not fitted")
        values = np.asarray(point, dtype=np.float64)
        indices = np.asarray(stable_example_indices, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != self.residuals.shape[1]:
            raise ValueError("point prediction dimension mismatch")
        if indices.shape != (len(values),):
            raise ValueError("one stable example index is required per prediction")
        scenarios = np.empty(
            (len(values), self.scenario_count, values.shape[1]), dtype=np.float64
        )
        for row, stable_index in enumerate(indices):
            rng = np.random.default_rng(self.seed + int(stable_index))
            selected = rng.integers(0, len(self.residuals), size=self.scenario_count)
            scenarios[row] = values[row] + self.residuals[selected]
        return CalibratedPrediction(
            q10=values + self.residual_quantiles[0],
            q50=values + self.residual_quantiles[1],
            q90=values + self.residual_quantiles[2],
            scenarios=scenarios,
        )


def tail_event_recall(
    actual_total: np.ndarray,
    predicted_q90: np.ndarray,
    *,
    fit_threshold: float,
    minimum_events: int = 10,
) -> dict[str, object]:
    """Evaluate the preregistered pooled tail event, preserving insufficiency."""

    actual = np.asarray(actual_total, dtype=np.float64).reshape(-1)
    upper = np.asarray(predicted_q90, dtype=np.float64).reshape(-1)
    if actual.shape != upper.shape or not np.isfinite(actual).all() or not np.isfinite(upper).all():
        raise ValueError("actual and q90 arrays must be same-length and finite")
    events = actual > float(fit_threshold)
    count = int(events.sum())
    if count < int(minimum_events):
        return {"status": "insufficient_events", "event_count": count, "recall": None}
    predicted_events = upper > float(fit_threshold)
    recall = float(np.mean(predicted_events[events]))
    return {"status": "ok", "event_count": count, "recall": recall}
