"""Paired statistics and interval-overlap helpers for R6-M9."""

from __future__ import annotations

import numpy as np


def interval_overlap(begin_a: int, end_a: int, begin_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(begin_a, begin_b))


def paired_bootstrap(
    progressive_ms, delayed_ms, *, samples: int = 10_000,
    seed: int = 20260816,
) -> dict[str, float | int | str]:
    progressive = np.asarray(progressive_ms, dtype=np.float64)
    delayed = np.asarray(delayed_ms, dtype=np.float64)
    if progressive.shape != delayed.shape or progressive.ndim != 1 or not len(progressive):
        raise ValueError("paired timing vectors must be non-empty and equal length")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    differences = delayed - progressive
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    medians = np.median(differences[indices], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    status = "PASS" if lower > 0 else "FAIL" if upper < 0 else "INCONCLUSIVE"
    return {
        "pairs": int(len(differences)),
        "median_d_minus_p_ms": float(np.median(differences)),
        "mean_d_minus_p_ms": float(np.mean(differences)),
        "median_progressive_ms": float(np.median(progressive)),
        "median_delayed_ms": float(np.median(delayed)),
        "relative_makespan_reduction": float(
            np.median(differences) / np.median(delayed)
        ),
        "bootstrap_samples": int(samples),
        "ci95_lower_ms": float(lower),
        "ci95_upper_ms": float(upper),
        "performance": status,
    }


__all__ = ["interval_overlap", "paired_bootstrap"]
