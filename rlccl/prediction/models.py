"""Preregistered history-only baselines, MLP, and true NumPy causal TCN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


METHOD_NAMES = (
    "long_term_mean",
    "previous_value",
    "ewma",
    "moment_only",
    "recent_history_mlp",
    "causal_tcn",
    "quantile_scenario",
)


def ewma_history_predictions(summaries: np.ndarray, alpha: float = 0.30) -> np.ndarray:
    """Return post-update EWMA states; prediction at t consumes state t-1."""

    values = np.asarray(summaries, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("summaries must be a nonempty 2-D array")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    result = np.empty_like(values)
    state = values[0].copy()
    for index, value in enumerate(values):
        if index:
            state = float(alpha) * value + (1.0 - float(alpha)) * state
        result[index] = state
    return result


def select_recent_backbone(validation_sequence_rmse: Mapping[str, np.ndarray]) -> str:
    """Select only from validation sequence-total RMSE, with frozen MLP tie break."""

    expected = {"recent_history_mlp", "causal_tcn"}
    if set(validation_sequence_rmse) != expected:
        raise ValueError(f"validation scores must contain exactly {sorted(expected)}")
    means: dict[str, float] = {}
    for name, scores in validation_sequence_rmse.items():
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"validation scores for {name} must be nonempty and finite")
        means[name] = float(values.mean())
    difference = means["recent_history_mlp"] - means["causal_tcn"]
    if abs(difference) <= 1e-12:
        return "recent_history_mlp"
    return "recent_history_mlp" if difference < 0.0 else "causal_tcn"


@dataclass
class RidgeMultiOutput:
    """Fit-only standardized multi-output ridge used by moment_only."""

    alpha: float = 10.0
    x_mean: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeMultiOutput":
        features = np.asarray(x, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets) or not len(features):
            raise ValueError("x and y must be nonempty same-length 2-D arrays")
        self.x_mean = features.mean(axis=0)
        self.x_scale = features.std(axis=0, ddof=0)
        self.x_scale = np.where(self.x_scale < 1e-8, 1.0, self.x_scale)
        normalized = (features - self.x_mean) / self.x_scale
        self.y_mean = targets.mean(axis=0)
        centered = targets - self.y_mean
        gram = normalized.T @ normalized
        self.weights = np.linalg.solve(
            gram + float(self.alpha) * np.eye(gram.shape[0]), normalized.T @ centered
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if any(value is None for value in (self.x_mean, self.x_scale, self.y_mean, self.weights)):
            raise RuntimeError("RidgeMultiOutput is not fitted")
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.x_mean.shape[0]:
            raise ValueError("ridge feature shape mismatch")
        return (features - self.x_mean) / self.x_scale @ self.weights + self.y_mean


class RecentHistoryMLP:
    """Fixed sklearn MLP wrapper for flattened eight-step summary histories."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        seed: int = 20260731,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        activation: str = "tanh",
        solver: str = "adam",
        alpha: float = 1e-4,
        batch_size: int = 256,
        learning_rate_init: float = 1e-3,
        max_iter: int = 80,
    ):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.early_stopping = False
        self._model: Any | None = None

    def _features(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim > 2:
            values = values.reshape(len(values), -1)
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("MLP input dimension mismatch")
        return values

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RecentHistoryMLP":
        from sklearn.neural_network import MLPRegressor

        features = self._features(x)
        targets = np.asarray(y, dtype=np.float64)
        if targets.shape != (len(features), self.output_dim):
            raise ValueError("MLP target dimension mismatch")
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            shuffle=True,
            random_state=self.seed,
            early_stopping=False,
        )
        self._model.fit(features, targets)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RecentHistoryMLP is not fitted")
        result = np.asarray(self._model.predict(self._features(x)), dtype=np.float64)
        return result.reshape(-1, self.output_dim)


class NumpyCausalTCN:
    """One-layer float64 same-length left-padded causal Conv1D with Adam."""

    _PARAMETERS = ("kernel", "conv_bias", "head_weight", "head_bias")

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kernel_size: int = 3,
        hidden_channels: int = 8,
        epochs: int = 40,
        batch_size: int = 256,
        learning_rate: float = 5e-3,
        l2: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        seed: int = 20260731,
    ):
        if min(input_dim, output_dim, kernel_size, hidden_channels, epochs, batch_size) <= 0:
            raise ValueError("TCN dimensions and training counts must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = int(kernel_size)
        self.hidden_channels = int(hidden_channels)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        kernel_limit = np.sqrt(
            6.0 / (self.kernel_size * self.input_dim + self.hidden_channels)
        )
        head_limit = np.sqrt(6.0 / (2 * self.hidden_channels + self.output_dim))
        self.kernel = rng.uniform(
            -kernel_limit,
            kernel_limit,
            size=(self.kernel_size, self.input_dim, self.hidden_channels),
        ).astype(np.float64)
        self.conv_bias = np.zeros(self.hidden_channels, dtype=np.float64)
        self.head_weight = rng.uniform(
            -head_limit,
            head_limit,
            size=(2 * self.hidden_channels, self.output_dim),
        ).astype(np.float64)
        self.head_bias = np.zeros(self.output_dim, dtype=np.float64)

    @property
    def representation_dim(self) -> int:
        return 2 * self.hidden_channels

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] == 0 or values.shape[2] != self.input_dim:
            raise ValueError("TCN input must have shape [batch, time, input_dim]")
        if not np.isfinite(values).all():
            raise ValueError("TCN input contains NaN/Inf")
        return values

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._validate_x(x)
        batch, steps, _ = values.shape
        preactivation = np.broadcast_to(
            self.conv_bias, (batch, steps, self.hidden_channels)
        ).copy()
        # Kernel index 0 is the oldest lag and index K-1 is the current position.
        for position in range(steps):
            for kernel_index in range(self.kernel_size):
                source = position - (self.kernel_size - 1 - kernel_index)
                if source >= 0:
                    preactivation[:, position] += values[:, source] @ self.kernel[kernel_index]
        hidden = np.tanh(preactivation)
        representation = np.concatenate((hidden[:, -1], hidden.mean(axis=1)), axis=1)
        prediction = representation @ self.head_weight + self.head_bias
        return prediction, hidden, representation

    def temporal_hidden(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0]

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        prediction = self.predict(x)
        targets = np.asarray(y, dtype=np.float64)
        if targets.shape != prediction.shape:
            raise ValueError("TCN target shape mismatch")
        mse = float(np.mean((prediction - targets) ** 2))
        penalty = self.l2 * float(
            np.sum(self.kernel**2) + np.sum(self.head_weight**2)
        )
        return mse + penalty

    def loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        values = self._validate_x(x)
        targets = np.asarray(y, dtype=np.float64)
        prediction, hidden, representation = self._forward(values)
        if targets.shape != prediction.shape:
            raise ValueError("TCN target shape mismatch")
        batch, steps, _ = values.shape
        error = prediction - targets
        data_loss = float(np.mean(error**2))
        penalty = self.l2 * float(
            np.sum(self.kernel**2) + np.sum(self.head_weight**2)
        )
        d_prediction = (2.0 / error.size) * error
        head_weight_gradient = (
            representation.T @ d_prediction + 2.0 * self.l2 * self.head_weight
        )
        head_bias_gradient = d_prediction.sum(axis=0)
        d_representation = d_prediction @ self.head_weight.T
        d_hidden = np.broadcast_to(
            d_representation[:, self.hidden_channels :, None] / steps,
            (batch, self.hidden_channels, steps),
        ).transpose(0, 2, 1).copy()
        d_hidden[:, -1] += d_representation[:, : self.hidden_channels]
        d_preactivation = d_hidden * (1.0 - hidden**2)
        conv_bias_gradient = d_preactivation.sum(axis=(0, 1))
        kernel_gradient = np.zeros_like(self.kernel)
        for position in range(steps):
            for kernel_index in range(self.kernel_size):
                source = position - (self.kernel_size - 1 - kernel_index)
                if source >= 0:
                    kernel_gradient[kernel_index] += (
                        values[:, source].T @ d_preactivation[:, position]
                    )
        kernel_gradient += 2.0 * self.l2 * self.kernel
        return data_loss + penalty, {
            "kernel": kernel_gradient,
            "conv_bias": conv_bias_gradient,
            "head_weight": head_weight_gradient,
            "head_bias": head_bias_gradient,
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        values = self._validate_x(x)
        targets = np.asarray(y, dtype=np.float64)
        if targets.shape != (len(values), self.output_dim):
            raise ValueError("TCN target shape mismatch")
        first_moment = {
            name: np.zeros_like(getattr(self, name)) for name in self._PARAMETERS
        }
        second_moment = {
            name: np.zeros_like(getattr(self, name)) for name in self._PARAMETERS
        }
        rng = np.random.default_rng(self.seed)
        update = 0
        history = np.empty(self.epochs, dtype=np.float64)
        for epoch in range(self.epochs):
            permutation = rng.permutation(len(values))
            for start in range(0, len(values), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                _, gradients = self.loss_and_gradients(
                    values[batch_indices], targets[batch_indices]
                )
                update += 1
                for name in self._PARAMETERS:
                    gradient = gradients[name]
                    first_moment[name] = (
                        self.beta1 * first_moment[name] + (1.0 - self.beta1) * gradient
                    )
                    second_moment[name] = (
                        self.beta2 * second_moment[name]
                        + (1.0 - self.beta2) * gradient**2
                    )
                    corrected_first = first_moment[name] / (1.0 - self.beta1**update)
                    corrected_second = second_moment[name] / (1.0 - self.beta2**update)
                    parameter = getattr(self, name)
                    parameter -= self.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + self.epsilon
                    )
            history[epoch] = self.loss(values, targets)
        return history

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in self._PARAMETERS}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        if set(state) != set(self._PARAMETERS):
            raise ValueError(f"TCN state must contain exactly {self._PARAMETERS}")
        for name in self._PARAMETERS:
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != getattr(self, name).shape or not np.isfinite(value).all():
                raise ValueError(f"invalid TCN state for {name}")
            setattr(self, name, value.copy())

    def save(self, path: str | Path) -> None:
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "kernel_size": self.kernel_size,
            "hidden_channels": self.hidden_channels,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "seed": self.seed,
        }
        arrays = self.state_dict()
        arrays.update({f"config_{name}": np.asarray([value]) for name, value in config.items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyCausalTCN":
        with np.load(path, allow_pickle=False) as values:
            integer_names = {
                "input_dim",
                "output_dim",
                "kernel_size",
                "hidden_channels",
                "epochs",
                "batch_size",
                "seed",
            }
            names = (
                "input_dim",
                "output_dim",
                "kernel_size",
                "hidden_channels",
                "epochs",
                "batch_size",
                "learning_rate",
                "l2",
                "beta1",
                "beta2",
                "epsilon",
                "seed",
            )
            config = {
                name: (
                    int(values[f"config_{name}"][0])
                    if name in integer_names
                    else float(values[f"config_{name}"][0])
                )
                for name in names
            }
            model = cls(**config)
            model.load_state_dict({name: values[name] for name in cls._PARAMETERS})
        return model


class HistoryPredictorSuite:
    """Fit-split-only implementation of all six point methods in the registry."""

    def __init__(self, recent_steps: int, target_dim: int, seed: int = 20260731):
        self.recent_steps = int(recent_steps)
        self.target_dim = int(target_dim)
        self.seed = int(seed)
        self.moment = RidgeMultiOutput(alpha=10.0)
        self.mlp = RecentHistoryMLP(
            input_dim=self.recent_steps * self.target_dim,
            output_dim=self.target_dim,
            seed=self.seed,
        )
        self.tcn = NumpyCausalTCN(
            input_dim=self.target_dim, output_dim=self.target_dim, seed=self.seed
        )
        self.long_term_mean: np.ndarray | None = None
        self.input_mean: np.ndarray | None = None
        self.input_scale: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None

    def fit(
        self, moment_features: np.ndarray, recent_history: np.ndarray, targets: np.ndarray
    ) -> "HistoryPredictorSuite":
        recent = np.asarray(recent_history, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if recent.shape[1:] != (self.recent_steps, self.target_dim) or target.shape != (len(recent), self.target_dim):
            raise ValueError("suite training shapes do not match configuration")
        self.long_term_mean = target.mean(axis=0)
        self.input_mean = recent.mean(axis=(0, 1))
        self.input_scale = recent.std(axis=(0, 1), ddof=0)
        self.input_scale = np.where(self.input_scale < 1e-8, 1.0, self.input_scale)
        self.target_mean = target.mean(axis=0)
        self.target_scale = target.std(axis=0, ddof=0)
        self.target_scale = np.where(self.target_scale < 1e-8, 1.0, self.target_scale)
        normalized_recent = (recent - self.input_mean) / self.input_scale
        normalized_targets = (target - self.target_mean) / self.target_scale
        self.moment.fit(moment_features, target)
        self.mlp.fit(normalized_recent, normalized_targets)
        self.tcn.fit(normalized_recent, normalized_targets)
        return self

    def predict(
        self,
        moment_features: np.ndarray,
        recent_history: np.ndarray,
        previous_targets: np.ndarray,
        ewma_targets: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if any(
            value is None
            for value in (
                self.long_term_mean,
                self.input_mean,
                self.input_scale,
                self.target_mean,
                self.target_scale,
            )
        ):
            raise RuntimeError("HistoryPredictorSuite is not fitted")
        recent = np.asarray(recent_history, dtype=np.float64)
        normalized_recent = (recent - self.input_mean) / self.input_scale
        count = len(recent)
        restore = lambda value: value * self.target_scale + self.target_mean
        return {
            "long_term_mean": np.repeat(self.long_term_mean[None, :], count, axis=0),
            "previous_value": np.asarray(previous_targets, dtype=np.float64).copy(),
            "ewma": np.asarray(ewma_targets, dtype=np.float64).copy(),
            "moment_only": self.moment.predict(moment_features),
            "recent_history_mlp": restore(self.mlp.predict(normalized_recent)),
            "causal_tcn": restore(self.tcn.predict(normalized_recent)),
        }
