"""Sanitized planning scenarios with a non-executable token namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..traffic.matrix_utils import validate_traffic_matrix
from .observation import readonly_array


@dataclass(frozen=True, slots=True)
class ScenarioTokenId:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.startswith("scenario:"):
            raise ValueError("Scenario token IDs must use the scenario: namespace")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ScenarioDemandToken:
    token_id: ScenarioTokenId
    source: int
    destination: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_id, ScenarioTokenId):
            raise TypeError("ScenarioDemandToken requires ScenarioTokenId")
        if self.source < 0 or self.destination < 0 or self.source == self.destination:
            raise ValueError("Invalid scenario demand endpoints")


@dataclass(frozen=True, slots=True)
class ScenarioSet:
    matrices: tuple[np.ndarray, ...]
    weights: tuple[float, ...]
    scenario_ids: tuple[str, ...]
    provenance: tuple[str, ...]
    scenario_tokens: tuple[tuple[ScenarioDemandToken, ...], ...]

    def __post_init__(self) -> None:
        raw_matrices = tuple(np.asarray(matrix) for matrix in self.matrices)
        for matrix in raw_matrices:
            validate_traffic_matrix(matrix)
        matrices = tuple(readonly_array(matrix, dtype=np.int64) for matrix in raw_matrices)
        weights = tuple(float(weight) for weight in self.weights)
        scenario_ids = tuple(str(value) for value in self.scenario_ids)
        provenance = tuple(str(value) for value in self.provenance)
        count = len(matrices)
        if not count or not (
            len(weights) == len(scenario_ids) == len(provenance) == count
        ):
            raise ValueError("Scenario fields must be non-empty and aligned")
        if len(set(scenario_ids)) != count:
            raise ValueError("scenario_ids must be unique")
        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Scenario weights must be finite and nonnegative")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("Scenario weights must have positive total")
        weights = tuple(weight / total for weight in weights)
        tokens = tuple(tuple(group) for group in self.scenario_tokens)
        if len(tokens) != count:
            raise ValueError("Scenario token groups must align with matrices")
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "scenario_ids", scenario_ids)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "scenario_tokens", tokens)

    @classmethod
    def from_matrices(
        cls,
        *,
        matrices: Sequence[np.ndarray],
        weights: Sequence[float],
        scenario_ids: Sequence[str],
        provenance: Sequence[str],
    ) -> "ScenarioSet":
        raw_matrices = tuple(np.asarray(matrix) for matrix in matrices)
        for matrix in raw_matrices:
            validate_traffic_matrix(matrix)
        copied = tuple(np.array(matrix, dtype=np.int64, copy=True) for matrix in raw_matrices)
        identifiers = tuple(str(value) for value in scenario_ids)
        all_tokens: list[tuple[ScenarioDemandToken, ...]] = []
        for scenario_id, matrix in zip(identifiers, copied):
            validate_traffic_matrix(matrix)
            tokens: list[ScenarioDemandToken] = []
            local_id = 0
            for source in range(matrix.shape[0]):
                for destination in range(matrix.shape[1]):
                    for _ in range(int(matrix[source, destination])):
                        tokens.append(
                            ScenarioDemandToken(
                                token_id=ScenarioTokenId(
                                    f"scenario:{scenario_id}:{local_id}"
                                ),
                                source=source,
                                destination=destination,
                            )
                        )
                        local_id += 1
            all_tokens.append(tuple(tokens))
        return cls(
            matrices=copied,
            weights=tuple(weights),
            scenario_ids=identifiers,
            provenance=tuple(provenance),
            scenario_tokens=tuple(all_tokens),
        )


__all__ = ["ScenarioDemandToken", "ScenarioSet", "ScenarioTokenId"]
