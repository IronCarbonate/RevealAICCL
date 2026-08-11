"""Validated, observation-bound scenario supports for ordinary scheduling."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from rlccl.uncertainty.observation import readonly_array


def _digest(value: Any) -> str:
    def encode(item: Any) -> bytes:
        if item is None:
            return b"n;"
        if isinstance(item, (bool, np.bool_)):
            return b"b:1;" if bool(item) else b"b:0;"
        if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            return f"i:{int(item)};".encode("ascii")
        if isinstance(item, (float, np.floating)):
            number = float(item)
            if not math.isfinite(number):
                raise ValueError("digest values must be finite")
            return f"f:{number.hex()};".encode("ascii")
        if isinstance(item, str):
            raw = item.encode("utf-8")
            return f"s:{len(raw)}:".encode("ascii") + raw + b";"
        if isinstance(item, np.ndarray):
            return encode({"dtype": item.dtype.str, "shape": list(item.shape),
                           "data": item.reshape(-1).tolist()})
        if isinstance(item, Mapping):
            pairs = sorted(((str(key), value) for key, value in item.items()), key=lambda pair: pair[0].encode("utf-8"))
            return f"m:{len(pairs)}:".encode("ascii") + b"".join(encode(key) + encode(value) for key, value in pairs)
        if isinstance(item, (tuple, list)):
            return f"l:{len(item)}:".encode("ascii") + b"".join(encode(value) for value in item)
        if isinstance(item, np.generic):
            return encode(item.item())
        raise TypeError(f"unsupported digest type: {type(item).__name__}")
    return hashlib.sha256(encode(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class ScenarioSupport:
    method: str
    requested_k: int
    actual_k: int
    matrices: tuple[np.ndarray, ...]
    weights: tuple[float, ...]
    provenance: Mapping[str, Any]
    stage: int
    observation_digest: str
    observation: Any
    digest: str
    uses_oracle: bool = False
    upper_bound_only: bool = False

    def __post_init__(self) -> None:
        matrices = tuple(readonly_array(matrix, dtype=np.int64) for matrix in self.matrices)
        weights = tuple(float(value) for value in self.weights)
        if not matrices or len(matrices) != self.actual_k or len(weights) != self.actual_k:
            raise ValueError("actual_k must match nonempty matrices and weights")
        shape = matrices[0].shape
        if len(shape) != 2 or shape[0] != shape[1] or any(matrix.shape != shape for matrix in matrices):
            raise ValueError("scenario matrices must be same-size square matrices")
        if any(np.any(matrix < 0) or np.any(np.diag(matrix) != 0) for matrix in matrices):
            raise ValueError("scenario matrices must be nonnegative with zero diagonal")
        if any(not np.isfinite(value) or value <= 0 for value in weights):
            raise ValueError("scenario weights must be finite and positive")
        total = sum(weights)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "weights", tuple(value / total for value in weights))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def scenario_support_from_matrices(
    *, matrices: Sequence[np.ndarray], weights: Sequence[float], method: str,
    requested_k: int, uses_oracle: bool, upper_bound_only: bool,
    provenance: Mapping[str, Any], observation: Any,
) -> ScenarioSupport:
    if uses_oracle or upper_bound_only:
        raise ValueError("oracle/upper-bound support is forbidden for an ordinary planner")
    raw_matrices = tuple(np.asarray(matrix) for matrix in matrices)
    raw_weights = tuple(float(value) for value in weights)
    requested = int(requested_k)
    if requested <= 0:
        raise ValueError("requested_k must be positive")
    if len(raw_matrices) != len(raw_weights) or not raw_matrices:
        raise ValueError("matrices and weights must have equal positive length")
    payload = (
        str(method), requested, raw_matrices, raw_weights, dict(provenance),
        int(observation.stage), str(observation.observation_digest),
    )
    return ScenarioSupport(
        method=str(method), requested_k=requested, actual_k=len(raw_matrices),
        matrices=raw_matrices, weights=raw_weights, provenance=dict(provenance),
        stage=int(observation.stage), observation_digest=str(observation.observation_digest),
        observation=observation, digest=_digest(payload), uses_oracle=False, upper_bound_only=False,
    )


def scenario_support_from_selected(selected: Any, *, observation: Any) -> ScenarioSupport:
    provenance = {
        "selected_indices": tuple(getattr(selected, "selected_indices", ())),
        "history_offsets": tuple(getattr(selected, "history_offsets", ())),
        "severity_definition": getattr(selected, "severity_definition", None),
        "approximation": getattr(selected, "approximation", None),
    }
    return scenario_support_from_matrices(
        matrices=selected.matrices, weights=selected.weights, method=selected.method,
        requested_k=selected.requested_k, uses_oracle=selected.uses_oracle,
        upper_bound_only=selected.upper_bound_only, provenance=provenance,
        observation=observation,
    )


def oracle_support_from_matrices(
    *, matrices: Sequence[np.ndarray], weights: Sequence[float], requested_k: int,
    provenance: Mapping[str, Any], observation: Any,
) -> ScenarioSupport:
    """Evaluator-only truth-assisted support; never callable through ordinary orchestration."""
    raw_matrices = tuple(np.asarray(matrix) for matrix in matrices)
    raw_weights = tuple(float(value) for value in weights)
    if not raw_matrices or len(raw_matrices) != len(raw_weights):
        raise ValueError("oracle matrices and weights must have equal positive length")
    if int(requested_k) != 8 or len(raw_matrices) > 8:
        raise ValueError("oracle support requires requested K8 and actual K in 1..8")
    unique_matrices: list[np.ndarray] = []
    unique_weights: list[float] = []
    positions: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for matrix, weight in zip(raw_matrices, raw_weights):
        canonical = np.ascontiguousarray(matrix, dtype=np.int64)
        key = (canonical.dtype.str, canonical.shape, canonical.tobytes())
        position = positions.get(key)
        if position is None:
            positions[key] = len(unique_matrices)
            unique_matrices.append(canonical)
            unique_weights.append(weight)
        else:
            unique_weights[position] += weight
    payload = ("oracle_scenario_robust_reference", requested_k, tuple(unique_matrices),
               tuple(unique_weights), dict(provenance), observation.stage, observation.observation_digest)
    return ScenarioSupport(
        method="oracle_scenario_robust_reference", requested_k=8,
        actual_k=len(unique_matrices), matrices=tuple(unique_matrices), weights=tuple(unique_weights),
        provenance=dict(provenance), stage=int(observation.stage),
        observation_digest=str(observation.observation_digest), observation=observation,
        digest=_digest(payload), uses_oracle=True, upper_bound_only=True,
    )


__all__ = ["ScenarioSupport", "scenario_support_from_matrices", "scenario_support_from_selected",
           "oracle_support_from_matrices"]
