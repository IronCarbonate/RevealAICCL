"""Raw reveal-aware recourse metrics and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RecourseMetrics:
    """One raw evaluation row under frozen discrete-time conventions.

    ``completion`` is elapsed execution slots, never a token count.  An
    unfinished run uses ``time_limit + 1``.  ``reveal_wait`` is the zero-based
    slot of the first executable action (or ``time_limit`` if none).
    ``recourse_count`` counts post-initial changes in the public proposal;
    ``replanned_actions`` counts truth-token actions proposed after slot zero.
    ``wasted_plan`` is the first scenario-only plan's L1 mismatch to final
    truth.  Initial proposal time and all later proposal time are reported
    separately as synthesis and replan time.
    """

    completion: int
    oracle_regret: float
    reveal_wait: int
    recourse_count: int
    replanned_actions: int
    wasted_plan: int
    synthesis_time_ms: float
    replan_time_ms: float
    legality: bool
    timeout: bool
    sequence_id: str
    family: str
    seed: int
    topology: str
    reveal_stage: int
    reveal_mode: str
    method: str
    manifest_id: str
    truth_digest: str
    topology_digest: str
    config_digest: str
    checker_version: str

    def to_raw_row(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["RecourseMetrics"]
