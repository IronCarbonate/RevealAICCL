"""Controlled MomentContext views used by V1 ablations."""

from dataclasses import replace

import numpy as np

from .types import MomentContext


def mean_only_context(context: MomentContext) -> MomentContext:
    """Keep historical first moments while removing variance/z information."""
    return replace(
        context,
        var_matrix=np.zeros_like(context.var_matrix),
        std_matrix=np.zeros_like(context.std_matrix),
        send_std=np.zeros_like(context.send_std),
        recv_std=np.zeros_like(context.recv_std),
        current_send_z=np.zeros_like(context.current_send_z),
        current_recv_z=np.zeros_like(context.current_recv_z),
        var_drift=0.0,
    )


def zero_moment_context(context: MomentContext) -> MomentContext:
    """Zero all statistical features while retaining shape/window metadata."""
    return replace(
        context,
        mean_matrix=np.zeros_like(context.mean_matrix),
        var_matrix=np.zeros_like(context.var_matrix),
        std_matrix=np.zeros_like(context.std_matrix),
        send_mean=np.zeros_like(context.send_mean),
        recv_mean=np.zeros_like(context.recv_mean),
        send_std=np.zeros_like(context.send_std),
        recv_std=np.zeros_like(context.recv_std),
        current_send_z=np.zeros_like(context.current_send_z),
        current_recv_z=np.zeros_like(context.current_recv_z),
        mean_drift=0.0,
        var_drift=0.0,
        confidence=0.0,
        is_warm=False,
    )
