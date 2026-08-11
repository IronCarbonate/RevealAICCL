"""Serializable types used by moment-bounded traffic sequences."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _array(value: Any, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class MomentBounds:
    epsilon_mean: float
    epsilon_var: float
    norm: str = "relative_l2"

    def __post_init__(self) -> None:
        if self.epsilon_mean < 0 or self.epsilon_var < 0:
            raise ValueError("Moment tolerances must be nonnegative")
        if self.norm != "relative_l2":
            raise ValueError(f"Unsupported moment norm: {self.norm}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MomentBounds":
        return cls(**data)


@dataclass
class MomentContext:
    history_length: int
    window_size: int
    mean_matrix: np.ndarray
    var_matrix: np.ndarray
    std_matrix: np.ndarray
    send_mean: np.ndarray
    recv_mean: np.ndarray
    send_std: np.ndarray
    recv_std: np.ndarray
    current_send_z: np.ndarray
    current_recv_z: np.ndarray
    mean_drift: float
    var_drift: float
    confidence: float
    is_warm: bool

    _ARRAY_FIELDS = (
        "mean_matrix",
        "var_matrix",
        "std_matrix",
        "send_mean",
        "recv_mean",
        "send_std",
        "recv_std",
        "current_send_z",
        "current_recv_z",
    )

    def __post_init__(self) -> None:
        for name in self._ARRAY_FIELDS:
            setattr(self, name, _array(getattr(self, name)))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MomentContext":
        values = dict(data)
        for name in cls._ARRAY_FIELDS:
            values[name] = _array(values[name])
        return cls(**values)


@dataclass
class TrafficSequence:
    sequence_id: str
    topology_name: str
    family: str
    seed: int
    matrices: list[np.ndarray]
    mean_ref: np.ndarray
    var_ref: np.ndarray
    bounds: MomentBounds
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrices = [_array(matrix, dtype=np.int64) for matrix in self.matrices]
        self.mean_ref = _array(self.mean_ref)
        self.var_ref = _array(self.var_ref)
        if isinstance(self.bounds, dict):
            self.bounds = MomentBounds.from_dict(self.bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "topology_name": self.topology_name,
            "family": self.family,
            "seed": int(self.seed),
            "matrices": [matrix.tolist() for matrix in self.matrices],
            "mean_ref": self.mean_ref.tolist(),
            "var_ref": self.var_ref.tolist(),
            "bounds": self.bounds.to_dict(),
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrafficSequence":
        return cls(
            sequence_id=data["sequence_id"],
            topology_name=data["topology_name"],
            family=data["family"],
            seed=int(data["seed"]),
            matrices=[_array(matrix, dtype=np.int64) for matrix in data["matrices"]],
            mean_ref=_array(data["mean_ref"]),
            var_ref=_array(data["var_ref"]),
            bounds=MomentBounds.from_dict(data["bounds"]),
            metadata=dict(data.get("metadata", {})),
        )
