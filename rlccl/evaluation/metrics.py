"""Central metric definitions shared by AMR-AICCL evaluation scripts."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def cvar(values: np.ndarray, alpha: float, higher_is_worse: bool = True) -> float:
    """Return empirical CVaR for the worst ``1-alpha`` fraction."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    threshold = np.quantile(array, alpha if higher_is_worse else 1.0 - alpha)
    tail = array[array >= threshold] if higher_is_worse else array[array <= threshold]
    return float(tail.mean())


def summarize_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize standard completion, timeout, legality, and synthesis metrics."""
    records = list(rows)
    if not records:
        raise ValueError("At least one evaluation row is required")
    completion = np.asarray([row["completion_steps"] for row in records], dtype=np.float64)
    synthesis = np.asarray([row["synthesis_ms"] for row in records], dtype=np.float64)
    summary = {
        "num_collectives": len(records),
        "completion_steps_mean": float(completion.mean()),
        "completion_steps_median": float(np.median(completion)),
        "completion_steps_p95": float(np.percentile(completion, 95)),
        "completion_steps_p99": float(np.percentile(completion, 99)),
        "completion_steps_cvar90": cvar(completion, 0.90),
        "completion_steps_cvar95": cvar(completion, 0.95),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in records])),
        "legality_rate": float(np.mean([bool(row["legal"]) for row in records])),
        "synthesis_ms_mean": float(synthesis.mean()),
        "synthesis_ms_p95": float(np.percentile(synthesis, 95)),
    }
    if "fallback_level" in records[0]:
        summary["fallback_rate"] = float(
            np.mean([int(row["fallback_level"]) > 0 for row in records])
        )
    return summary
