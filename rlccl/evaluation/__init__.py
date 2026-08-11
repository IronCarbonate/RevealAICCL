"""Evaluation helpers.

Torch-backed policy evaluation is loaded lazily so CPU-only, NumPy-based audit
utilities do not require PyTorch merely to import this package.
"""

from .metrics import cvar, summarize_rows

__all__ = [
    "CONTEXT_MODES",
    "build_shuffled_context_map",
    "cvar",
    "evaluate_sequence_policy",
    "summarize_rows",
]


def __getattr__(name):
    if name in {"CONTEXT_MODES", "build_shuffled_context_map", "evaluate_sequence_policy"}:
        from . import sequence_evaluator

        return getattr(sequence_evaluator, name)
    raise AttributeError(name)
