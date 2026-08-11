"""Deterministic reference packing for router-derived variable-size AlltoAllv.

This module is intentionally a correctness substrate, not production MoE
packing.  Runtime callers may only pack assignments from chunks explicitly
marked completed *and* revealed.  Send counts are derived from those router
assignments; callers cannot inject final counts or offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Iterable, Mapping, Sequence

import numpy as np


PAYLOAD_FIELDS = 8
_CHECKSUM_MODULUS = (1 << 63) - 25
_CHECKSUM_COEFFICIENTS = (
    1_000_003,
    1_000_033,
    1_000_037,
    1_000_081,
    1_000_099,
    1_000_117,
    1_000_121,
)


@dataclass(frozen=True, slots=True)
class RouterAssignment:
    """One real token assignment emitted by the reference router."""

    token_id: int
    source_rank: int
    destination_rank: int
    expert_id: int
    chunk_id: int
    chunk_offset: int
    payload_word: int

    def record(self) -> tuple[int, ...]:
        prefix = (
            int(self.token_id),
            int(self.source_rank),
            int(self.destination_rank),
            int(self.expert_id),
            int(self.chunk_id),
            int(self.chunk_offset),
            int(self.payload_word),
        )
        return prefix + (payload_checksum(prefix),)


@dataclass(frozen=True, slots=True)
class DestinationLayout:
    """Destination lists and delta split metadata before payload flattening."""

    source_rank: int
    chunk_ids: tuple[int, ...]
    destination_lists: tuple[tuple[RouterAssignment, ...], ...]
    sendcounts_tokens: tuple[int, ...]
    offsets_tokens: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return sum(self.sendcounts_tokens)


@dataclass(frozen=True, slots=True)
class PackedShard:
    """Contiguous fixed-record payload with variable destination splits."""

    source_rank: int
    chunk_ids: tuple[int, ...]
    sendcounts_tokens: tuple[int, ...]
    offsets_tokens: tuple[int, ...]
    records: np.ndarray

    @property
    def sendcounts_elements(self) -> tuple[int, ...]:
        return tuple(value * PAYLOAD_FIELDS for value in self.sendcounts_tokens)

    @property
    def total_tokens(self) -> int:
        return int(self.records.shape[0])

    @property
    def total_bytes(self) -> int:
        return int(self.records.nbytes)


def payload_checksum(prefix: Sequence[int]) -> int:
    """Stable signed-int64-safe checksum for the seven identity fields."""

    if len(prefix) != PAYLOAD_FIELDS - 1:
        raise ValueError("payload checksum requires seven fields")
    value = 0
    for item, coefficient in zip(prefix, _CHECKSUM_COEFFICIENTS, strict=True):
        value = (value + (int(item) % _CHECKSUM_MODULUS) * coefficient) % _CHECKSUM_MODULUS
    return int(value)


def build_destination_layout(
    assignments: Iterable[RouterAssignment],
    *,
    world_size: int,
    source_rank: int,
    chunk_ids: Sequence[int],
    timing_sink: dict[str, float] | None = None,
) -> DestinationLayout:
    """Derive lists/counts/offsets exclusively from router assignments."""

    timing_start = time.perf_counter_ns()
    if world_size <= 0 or not 0 <= source_rank < world_size:
        raise ValueError("invalid world/source rank")
    normalized_chunks = tuple(int(value) for value in chunk_ids)
    if len(set(normalized_chunks)) != len(normalized_chunks):
        raise ValueError("duplicate chunk in descriptor")
    allowed = set(normalized_chunks)
    lists: list[list[RouterAssignment]] = [[] for _ in range(world_size)]
    seen: set[int] = set()
    for item in assignments:
        if int(item.source_rank) != source_rank:
            raise ValueError("assignment source does not match descriptor source")
        if int(item.chunk_id) not in allowed:
            raise ValueError("assignment belongs to an unrevealed descriptor chunk")
        if not 0 <= int(item.destination_rank) < world_size:
            raise ValueError("assignment destination outside process group")
        if int(item.destination_rank) != int(item.expert_id) % world_size:
            raise ValueError("destination rank is not router-expert-derived")
        token_id = int(item.token_id)
        if token_id in seen:
            raise ValueError("duplicate token in descriptor")
        seen.add(token_id)
        lists[int(item.destination_rank)].append(item)
    for values in lists:
        values.sort(key=lambda item: (int(item.token_id), int(item.chunk_id), int(item.chunk_offset)))
    counts = tuple(len(values) for values in lists)
    count_done = time.perf_counter_ns()
    offsets: list[int] = []
    cursor = 0
    for count in counts:
        offsets.append(cursor)
        cursor += count
    offset_done = time.perf_counter_ns()
    if timing_sink is not None:
        timing_sink["count_construction_us"] = (count_done - timing_start) / 1e3
        timing_sink["offset_construction_us"] = (offset_done - count_done) / 1e3
    return DestinationLayout(
        source_rank=source_rank,
        chunk_ids=normalized_chunks,
        destination_lists=tuple(tuple(values) for values in lists),
        sendcounts_tokens=counts,
        offsets_tokens=tuple(offsets),
    )


def pack_destination_layout(layout: DestinationLayout) -> PackedShard:
    """Flatten a validated layout in deterministic destination/token order."""

    rows = [item.record() for values in layout.destination_lists for item in values]
    records = (
        np.asarray(rows, dtype=np.int64).reshape((-1, PAYLOAD_FIELDS))
        if rows else np.empty((0, PAYLOAD_FIELDS), dtype=np.int64)
    )
    if not records.flags.c_contiguous:
        records = np.ascontiguousarray(records)
    records.setflags(write=False)
    return PackedShard(
        source_rank=layout.source_rank,
        chunk_ids=layout.chunk_ids,
        sendcounts_tokens=layout.sendcounts_tokens,
        offsets_tokens=layout.offsets_tokens,
        records=records,
    )


class ProgressivePackingState:
    """Fail-closed completed/revealed/dispatched state for delta descriptors."""

    def __init__(self, *, world_size: int, source_rank: int, max_chunks: int = 8) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.world_size = int(world_size)
        self.source_rank = int(source_rank)
        self.max_chunks = int(max_chunks)
        self.completed_bitmap = 0
        self.revealed_bitmap = 0
        self.dispatched_bitmap = 0
        self._assignments: list[tuple[RouterAssignment, ...] | None] = [None] * max_chunks
        self._dispatched_tokens: set[int] = set()
        self.descriptor_count = 0
        self.future_access_attempts = 0
        self.unrevealed_execution = 0
        self.duplicate_dispatch = 0
        self.stale_dispatch = 0

    def mark_completed(self, chunk: int, assignments: Sequence[RouterAssignment]) -> None:
        index = int(chunk)
        if not 0 <= index < self.max_chunks:
            raise ValueError("chunk outside state")
        bit = 1 << index
        if self.completed_bitmap & bit:
            raise ValueError("completed chunk replay")
        values = tuple(assignments)
        if any(int(item.chunk_id) != index for item in values):
            raise ValueError("assignment chunk mismatch")
        self._assignments[index] = values
        self.completed_bitmap |= bit

    def reveal(self, chunk: int) -> None:
        index = int(chunk)
        bit = 1 << index
        if not self.completed_bitmap & bit:
            self.future_access_attempts += 1
            raise ValueError("cannot reveal incomplete router chunk")
        if self.revealed_bitmap & bit:
            raise ValueError("revealed chunk replay")
        self.revealed_bitmap |= bit

    def build_delta_layout(
        self,
        chunk_ids: Sequence[int],
        *,
        timing_sink: dict[str, float] | None = None,
    ) -> DestinationLayout:
        """Validate a progressive delta and derive its destination layout."""

        normalized = tuple(int(value) for value in chunk_ids)
        if not normalized:
            raise ValueError("descriptor must name at least one chunk")
        mask = 0
        assignments: list[RouterAssignment] = []
        for index in normalized:
            if not 0 <= index < self.max_chunks:
                raise ValueError("descriptor chunk outside state")
            bit = 1 << index
            mask |= bit
            if not self.completed_bitmap & bit:
                self.future_access_attempts += 1
                raise ValueError("future router chunk access")
            if not self.revealed_bitmap & bit:
                self.unrevealed_execution += 1
                raise ValueError("unrevealed router chunk execution")
            if self.dispatched_bitmap & bit:
                self.stale_dispatch += 1
                raise ValueError("stale chunk dispatch")
            values = self._assignments[index]
            if values is None:
                raise RuntimeError("completed chunk has no assignments")
            assignments.extend(values)
        token_ids = [int(item.token_id) for item in assignments]
        if len(token_ids) != len(set(token_ids)) or any(
            token_id in self._dispatched_tokens for token_id in token_ids
        ):
            self.duplicate_dispatch += 1
            raise ValueError("duplicate token dispatch")
        layout = build_destination_layout(
            assignments,
            world_size=self.world_size,
            source_rank=self.source_rank,
            chunk_ids=normalized,
            timing_sink=timing_sink,
        )
        self.dispatched_bitmap |= mask
        self._dispatched_tokens.update(token_ids)
        self.descriptor_count += 1
        return layout

    def build_delta(self, chunk_ids: Sequence[int]) -> tuple[DestinationLayout, PackedShard]:
        layout = self.build_delta_layout(chunk_ids)
        packed = pack_destination_layout(layout)
        return layout, packed

    @property
    def dispatched_token_count(self) -> int:
        return len(self._dispatched_tokens)


def decode_records(records: np.ndarray) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(records, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != PAYLOAD_FIELDS:
        raise ValueError("invalid payload record shape")
    return tuple(tuple(int(item) for item in row) for row in values.tolist())


def payload_multiset_digest(records: Iterable[Sequence[int]]) -> str:
    normalized = sorted(tuple(int(item) for item in row) for row in records)
    digest = hashlib.sha256()
    for row in normalized:
        digest.update(np.asarray(row, dtype="<i8").tobytes())
    return digest.hexdigest()


def verify_received_records(
    records: np.ndarray,
    *,
    destination_rank: int,
    expected_by_token: Mapping[int, Sequence[int]],
) -> dict[str, int | bool | str]:
    """Verify identity, destination, checksum, corruption, loss, and duplication."""

    decoded = decode_records(records)
    seen: set[int] = set()
    duplicate = 0
    wrong_destination = 0
    corruption = 0
    unexpected = 0
    for row in decoded:
        token_id = int(row[0])
        duplicate += int(token_id in seen)
        seen.add(token_id)
        wrong_destination += int(int(row[2]) != int(destination_rank))
        corruption += int(payload_checksum(row[:7]) != int(row[7]))
        expected = expected_by_token.get(token_id)
        if expected is None:
            unexpected += 1
        elif tuple(int(item) for item in expected) != row:
            corruption += 1
    expected_tokens = set(int(value) for value in expected_by_token)
    lost = len(expected_tokens - seen)
    unexpected += len(seen - expected_tokens)
    passed = not any((lost, duplicate, wrong_destination, corruption, unexpected))
    return {
        "pass": passed,
        "received": len(decoded),
        "expected": len(expected_tokens),
        "lost": lost,
        "duplicate": duplicate,
        "wrong_destination": wrong_destination,
        "corruption": corruption,
        "unexpected": unexpected,
        "payload_multiset_digest": payload_multiset_digest(decoded),
    }
