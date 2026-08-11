"""Deterministic, monotone partial-demand reveal processes."""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np

from .observation import PartialObservationState, RevealedDemandToken
from .problem import UncertainProblemInstance


_MODES = {
    "random_entries",
    "source_totals_first",
    "source_destination_totals_first",
    "partial_shards",
    "time_based_arrival",
}


class DemandRevealProcess:
    """Evaluator-owned deterministic reveal schedule."""

    def __init__(
        self,
        *,
        problem: UncertainProblemInstance,
        mode: str,
        ratios: Sequence[float],
        seed: int,
    ) -> None:
        if not isinstance(problem, UncertainProblemInstance):
            raise TypeError("problem must be UncertainProblemInstance")
        if mode not in _MODES:
            raise ValueError(f"Unsupported reveal mode: {mode}")
        normalized = tuple(float(value) for value in ratios)
        if not normalized or normalized[0] != 0.0 or normalized[-1] != 1.0:
            raise ValueError("Reveal ratios must start at 0 and end at 1")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in normalized):
            raise ValueError("Reveal ratios must be finite values in [0, 1]")
        if any(left >= right for left, right in zip(normalized, normalized[1:])):
            raise ValueError("Reveal ratios must be strictly increasing")
        self._problem = problem
        self._mode = str(mode)
        self._ratios = normalized
        self._seed = int(seed)
        random = np.random.default_rng(self._seed)
        nodes = problem._node_count
        entries = np.asarray(
            [(source, destination) for source in range(nodes) for destination in range(nodes) if source != destination],
            dtype=np.int64,
        )
        self._entry_order = entries[random.permutation(len(entries))]
        self._arrival = random.uniform(np.nextafter(0.0, 1.0), 1.0, size=len(entries))
        self._token_order = random.permutation(problem._token_count)

    def __iter__(self) -> Iterator[PartialObservationState]:
        for stage, ratio in enumerate(self._ratios):
            yield self._build_observation(stage, ratio)

    def observation_for_slot(self, slot: int) -> PartialObservationState:
        """Return a fresh observation for a discrete execution slot.

        Reveal stage ``r`` arrives immediately before slot ``r``.  Once the
        final stage has arrived, later slots receive freshly rebuilt full
        observations so holder/state-version changes remain current.
        """
        if isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer)):
            raise TypeError("slot must be a nonnegative integer")
        if int(slot) < 0:
            raise ValueError("slot must be nonnegative")
        stage = min(int(slot), len(self._ratios) - 1)
        return self._build_observation(stage, self._ratios[stage])

    def observation_for_stage(self, stage: int) -> PartialObservationState:
        """Return a fresh observation for one frozen reveal stage."""
        if isinstance(stage, (bool, np.bool_)) or not isinstance(stage, (int, np.integer)):
            raise TypeError("stage must be an integer")
        value = int(stage)
        if value < 0 or value >= len(self._ratios):
            raise ValueError("stage is outside the reveal schedule")
        return self._build_observation(value, self._ratios[value])

    def full_observation(self) -> PartialObservationState:
        """Return a fresh evaluator-only full-information observation."""
        stage = len(self._ratios) - 1
        return self._build_observation(stage, self._ratios[stage])

    def _build_observation(self, stage: int, ratio: float) -> PartialObservationState:
        world = self._problem
        nodes = world._node_count
        entry_mask = np.eye(nodes, dtype=bool)
        revealed_indices: list[int] = []

        if self._mode == "partial_shards":
            count = world._token_count if ratio == 1.0 else int(math.floor(ratio * world._token_count))
            revealed_indices = [int(value) for value in self._token_order[:count]]
            per_pair: dict[tuple[int, int], int] = {}
            for token_index in revealed_indices:
                source, destination, _ = world._atomic[token_index]
                pair = (source, destination)
                per_pair[pair] = per_pair.get(pair, 0) + 1
            for source in range(nodes):
                for destination in range(nodes):
                    if source == destination:
                        continue
                    pair_total = len(world._indices_for_pair(source, destination))
                    if ratio == 1.0 or (
                        pair_total > 0
                        and per_pair.get((source, destination), 0) == pair_total
                    ):
                        entry_mask[source, destination] = True
        else:
            if self._mode == "time_based_arrival":
                selected = self._entry_order[self._arrival <= ratio]
                if ratio == 1.0:
                    selected = self._entry_order
            else:
                denominator = len(self._entry_order)
                count = denominator if ratio == 1.0 else int(math.floor(ratio * denominator))
                selected = self._entry_order[:count]
            for source, destination in selected:
                source_i, destination_i = int(source), int(destination)
                entry_mask[source_i, destination_i] = True
                revealed_indices.extend(world._indices_for_pair(source_i, destination_i))

        observed = np.zeros_like(world._truth)
        if self._mode == "partial_shards":
            for token_index in revealed_indices:
                source, destination, _ = world._atomic[token_index]
                observed[source, destination] += 1
        else:
            observed[entry_mask] = world._truth[entry_mask]

        tokens: list[RevealedDemandToken] = []
        for token_index in revealed_indices:
            source, destination, holders = world._token_record(token_index)
            tokens.append(
                RevealedDemandToken(
                    token_id=world._issue_token_id(token_index, reveal_seed=self._seed),
                    source=source,
                    destination=destination,
                    holders=holders,
                )
            )

        source_totals = None
        destination_totals = None
        if self._mode in {"source_totals_first", "source_destination_totals_first"}:
            source_totals = world._truth.sum(axis=1)
        if self._mode == "source_destination_totals_first":
            destination_totals = world._truth.sum(axis=0)

        return PartialObservationState(
            sequence_id=world.sequence_id,
            sequence_step=world.sequence_step,
            family=world.family,
            mode=self._mode,
            stage=int(stage),
            ratio=float(ratio),
            entry_mask=entry_mask,
            observed_matrix=observed,
            unknown_mask=~entry_mask,
            revealed_tokens=tuple(tokens),
            source_totals=source_totals,
            destination_totals=destination_totals,
            topology=world.public_topology,
            state_version=world._state_version,
        )


__all__ = ["DemandRevealProcess"]
