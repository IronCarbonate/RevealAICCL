"""Sliding moment estimator whose context is read-only and history-only."""

from collections import deque
from typing import Any

import numpy as np

from .matrix_utils import validate_traffic_matrix
from .moment_validation import relative_l2_error
from .types import MomentContext


class SlidingMomentEstimator:
    """Estimate moments from completed matrices in one sequence only."""

    def __init__(
        self,
        num_nodes: int,
        window_size: int,
        min_history: int,
        eps: float = 1e-6,
        z_clip: float = 10.0,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_history <= 0 or min_history > window_size:
            raise ValueError("min_history must be in [1, window_size]")
        if eps <= 0 or z_clip <= 0:
            raise ValueError("eps and z_clip must be positive")
        self.num_nodes = int(num_nodes)
        self.window_size = int(window_size)
        self.min_history = int(min_history)
        self.eps = float(eps)
        self.z_clip = float(z_clip)
        self._history: deque[np.ndarray] = deque(maxlen=self.window_size)

    @property
    def history_length(self) -> int:
        return len(self._history)

    def _validate(self, matrix: np.ndarray, name: str) -> np.ndarray:
        validate_traffic_matrix(matrix)
        array = np.asarray(matrix, dtype=np.int64)
        expected = (self.num_nodes, self.num_nodes)
        if array.shape != expected:
            raise ValueError(f"{name} shape must be {expected}, got {array.shape}")
        return array

    def get_context(
        self,
        current_matrix: np.ndarray,
        mean_ref: np.ndarray,
        var_ref: np.ndarray,
    ) -> MomentContext:
        """Return context without mutating history or including ``current_matrix``."""
        current = self._validate(current_matrix, "current_matrix")
        mean_reference = np.asarray(mean_ref, dtype=np.float64)
        var_reference = np.asarray(var_ref, dtype=np.float64)
        expected = (self.num_nodes, self.num_nodes)
        if mean_reference.shape != expected or var_reference.shape != expected:
            raise ValueError(f"Reference moment shapes must be {expected}")
        if np.any(var_reference < 0):
            raise ValueError("var_ref must be nonnegative")

        if self._history:
            history = np.stack(tuple(self._history), axis=0).astype(np.float64)
            mean_matrix = history.mean(axis=0)
            var_matrix = history.var(axis=0, ddof=0)
            send_totals = history.sum(axis=2)
            recv_totals = history.sum(axis=1)
            send_std = send_totals.std(axis=0, ddof=0)
            recv_std = recv_totals.std(axis=0, ddof=0)
        else:
            mean_matrix = mean_reference.copy()
            var_matrix = var_reference.copy()
            # Independence approximation is used only for cold-start z scaling.
            send_std = np.sqrt(np.maximum(var_reference.sum(axis=1), 0.0))
            recv_std = np.sqrt(np.maximum(var_reference.sum(axis=0), 0.0))

        np.fill_diagonal(mean_matrix, 0.0)
        np.fill_diagonal(var_matrix, 0.0)
        std_matrix = np.sqrt(np.maximum(var_matrix, 0.0) + self.eps)
        np.fill_diagonal(std_matrix, 0.0)
        send_mean = mean_matrix.sum(axis=1)
        recv_mean = mean_matrix.sum(axis=0)
        current_send = current.sum(axis=1, dtype=np.float64)
        current_recv = current.sum(axis=0, dtype=np.float64)
        current_send_z = np.clip(
            (current_send - send_mean) / (send_std + self.eps),
            -self.z_clip,
            self.z_clip,
        )
        current_recv_z = np.clip(
            (current_recv - recv_mean) / (recv_std + self.eps),
            -self.z_clip,
            self.z_clip,
        )
        history_length = len(self._history)
        confidence = min(1.0, history_length / self.min_history)

        return MomentContext(
            history_length=history_length,
            window_size=self.window_size,
            mean_matrix=mean_matrix.copy(),
            var_matrix=var_matrix.copy(),
            std_matrix=std_matrix,
            send_mean=send_mean,
            recv_mean=recv_mean,
            send_std=np.asarray(send_std, dtype=np.float64),
            recv_std=np.asarray(recv_std, dtype=np.float64),
            current_send_z=current_send_z,
            current_recv_z=current_recv_z,
            mean_drift=relative_l2_error(mean_matrix, mean_reference, self.eps),
            var_drift=relative_l2_error(var_matrix, var_reference, self.eps),
            confidence=confidence,
            is_warm=history_length >= self.min_history,
        )

    def update(self, completed_matrix: np.ndarray) -> None:
        """Append one completed matrix, retaining only the configured window."""
        matrix = self._validate(completed_matrix, "completed_matrix")
        self._history.append(matrix.copy())

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "window_size": self.window_size,
            "min_history": self.min_history,
            "eps": self.eps,
            "z_clip": self.z_clip,
            "history": [matrix.tolist() for matrix in self._history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in ("num_nodes", "window_size", "min_history"):
            if int(state[name]) != getattr(self, name):
                raise ValueError(
                    f"Estimator {name} mismatch: {state[name]} != {getattr(self, name)}"
                )
        history = [self._validate(matrix, "history matrix") for matrix in state.get("history", [])]
        if len(history) > self.window_size:
            raise ValueError("Estimator state contains more history than window_size")
        self._history.clear()
        self._history.extend(matrix.copy() for matrix in history)
