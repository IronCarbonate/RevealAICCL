"""Preallocated, router-derived progressive forward data preparation.

This is an opt-in R5-P3 reference fast path.  It preserves the byte-level
ForwardPayload produced by reference_full_moe.pack_forward_payload while
removing per-descriptor Python token-list reconstruction.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from rlccl.transport.reference_a2av import RouterAssignment
from rlccl.transport.reference_full_moe import (
    FORWARD_META_FIELDS, ForwardPayload, feature_digest, identity_checksum,
)


_MODULUS = (1 << 63) - 25
_DESTINATION_COEFFICIENT = 1_000_037
_EXPERT_COEFFICIENT = 1_000_081


def _pinned_array(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, np.ndarray]:
    tensor = torch.empty(shape, dtype=dtype, pin_memory=True)
    return tensor, tensor.numpy()


class FastProgressiveDataPrep:
    """Fail-closed state plus preallocated destination/descriptor buffers."""

    def __init__(
        self, *, world_size: int, source_rank: int, tokens: np.ndarray,
        token_ids: np.ndarray, chunk_offsets: Sequence[int],
        descriptor_groups: Sequence[Sequence[int]],
    ) -> None:
        started = time.perf_counter_ns()
        self.world_size = int(world_size)
        self.source_rank = int(source_rank)
        self.tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        self.chunk_offsets = tuple(int(value) for value in chunk_offsets)
        self.max_chunks = len(self.chunk_offsets) - 1
        self.descriptor_groups = tuple(tuple(int(item) for item in group) for group in descriptor_groups)
        if self.world_size <= 0 or not 0 <= self.source_rank < self.world_size:
            raise ValueError("invalid world/source rank")
        if self.tokens.shape[0] != self.token_ids.size or self.chunk_offsets[-1] != self.token_ids.size:
            raise ValueError("fast data-prep token layout mismatch")
        if sorted(item for group in self.descriptor_groups for item in group) != list(range(self.max_chunks)):
            raise ValueError("descriptor groups must partition chunks in deterministic order")

        total_tokens = int(self.token_ids.size)
        self._feature_digests = np.fromiter(
            (feature_digest(row) for row in self.tokens), dtype=np.int64, count=total_tokens,
        )
        self._checksum_base = np.empty(total_tokens, dtype=np.int64)
        for chunk in range(self.max_chunks):
            left, right = self.chunk_offsets[chunk:chunk + 2]
            for position in range(left, right):
                offset = position - left
                self._checksum_base[position] = identity_checksum((
                    int(self.token_ids[position]), self.source_rank, 0, 0,
                    chunk, offset, position, int(self._feature_digests[position]),
                ))

        self._chunk_counts = np.zeros((self.max_chunks, self.world_size), dtype=np.int64)
        self._chunk_meta_tensors: list[list[torch.Tensor]] = []
        self._chunk_meta: list[list[np.ndarray]] = []
        self._chunk_feature_tensors: list[list[torch.Tensor]] = []
        self._chunk_features: list[list[np.ndarray]] = []
        for chunk in range(self.max_chunks):
            size = self.chunk_offsets[chunk + 1] - self.chunk_offsets[chunk]
            meta_tensors, meta_arrays, feature_tensors, feature_arrays = [], [], [], []
            for _ in range(self.world_size):
                meta_tensor, meta_array = _pinned_array((size, FORWARD_META_FIELDS), torch.int64)
                feature_tensor, feature_array = _pinned_array((size, self.tokens.shape[1]), torch.float32)
                meta_tensors.append(meta_tensor); meta_arrays.append(meta_array)
                feature_tensors.append(feature_tensor); feature_arrays.append(feature_array)
            self._chunk_meta_tensors.append(meta_tensors); self._chunk_meta.append(meta_arrays)
            self._chunk_feature_tensors.append(feature_tensors); self._chunk_features.append(feature_arrays)

        self._descriptor_meta_tensors: list[torch.Tensor] = []
        self._descriptor_meta: list[np.ndarray] = []
        self._descriptor_feature_tensors: list[torch.Tensor] = []
        self._descriptor_features: list[np.ndarray] = []
        for group in self.descriptor_groups:
            capacity = sum(
                self.chunk_offsets[chunk + 1] - self.chunk_offsets[chunk] for chunk in group
            )
            meta_tensor, meta_array = _pinned_array((capacity, FORWARD_META_FIELDS), torch.int64)
            feature_tensor, feature_array = _pinned_array((capacity, self.tokens.shape[1]), torch.float32)
            self._descriptor_meta_tensors.append(meta_tensor); self._descriptor_meta.append(meta_array)
            self._descriptor_feature_tensors.append(feature_tensor); self._descriptor_features.append(feature_array)

        self.completed_bitmap = 0
        self.revealed_bitmap = 0
        self.dispatched_bitmap = 0
        self._dispatched_positions = np.zeros(total_tokens, dtype=np.bool_)
        self.descriptor_count = 0
        self.future_access_attempts = 0
        self.unrevealed_execution = 0
        self.duplicate_dispatch = 0
        self.stale_dispatch = 0
        self.mark_completed_total_us = 0.0
        self._count_ticket: Any = None
        self.precompute_us = (time.perf_counter_ns() - started) / 1e3

    def mark_completed(
        self, chunk: int, assignments: Sequence[RouterAssignment], *, experts: np.ndarray,
    ) -> None:
        started = time.perf_counter_ns()
        index = int(chunk)
        if not 0 <= index < self.max_chunks:
            raise ValueError("chunk outside state")
        bit = 1 << index
        if self.completed_bitmap & bit:
            raise ValueError("completed chunk replay")
        left, right = self.chunk_offsets[index:index + 2]
        experts_array = np.asarray(experts, dtype=np.int64)
        if experts_array.size != right - left or len(assignments) != experts_array.size:
            raise ValueError("assignment cardinality mismatch")
        if any(
            int(item.chunk_id) != index or int(item.source_rank) != self.source_rank
            or int(item.expert_id) != int(experts_array[offset])
            for offset, item in enumerate(assignments)
        ):
            raise ValueError("router assignment mismatch")
        destinations = np.mod(experts_array, self.world_size)
        positions = np.arange(left, right, dtype=np.int64)
        chunk_offsets = positions - left
        for destination in range(self.world_size):
            selected = np.flatnonzero(destinations == destination)
            count = int(selected.size)
            self._chunk_counts[index, destination] = count
            if not count:
                continue
            global_positions = positions[selected]
            metadata = self._chunk_meta[index][destination]
            metadata[:count, 0] = self.token_ids[global_positions]
            metadata[:count, 1] = self.source_rank
            metadata[:count, 2] = destination
            metadata[:count, 3] = experts_array[selected]
            metadata[:count, 4] = index
            metadata[:count, 5] = chunk_offsets[selected]
            metadata[:count, 6] = global_positions
            metadata[:count, 7] = self._feature_digests[global_positions]
            metadata[:count, 8] = np.mod(
                self._checksum_base[global_positions]
                + destination * _DESTINATION_COEFFICIENT
                + experts_array[selected] * _EXPERT_COEFFICIENT,
                _MODULUS,
            )
            self._chunk_features[index][destination][:count] = self.tokens[global_positions]
        self.completed_bitmap |= bit
        self.mark_completed_total_us += (time.perf_counter_ns() - started) / 1e3

    def reveal(self, chunk: int) -> None:
        index = int(chunk)
        bit = 1 << index
        if not self.completed_bitmap & bit:
            self.future_access_attempts += 1
            raise ValueError("cannot reveal incomplete router chunk")
        if self.revealed_bitmap & bit:
            raise ValueError("revealed chunk replay")
        self.revealed_bitmap |= bit

    def build_delta_payload(
        self, chunk_ids: Sequence[int], *, timing_sink: dict[str, float] | None = None,
        count_exchange_launcher: Callable[[tuple[int, ...]], Any] | None = None,
    ) -> ForwardPayload:
        normalized = tuple(int(value) for value in chunk_ids)
        if not normalized:
            raise ValueError("descriptor must name at least one chunk")
        if self.descriptor_count >= len(self.descriptor_groups):
            raise ValueError("too many descriptors")
        if normalized != self.descriptor_groups[self.descriptor_count]:
            raise ValueError("descriptor order/granularity divergence")
        mask = 0
        positions = []
        for index in normalized:
            if not 0 <= index < self.max_chunks:
                raise ValueError("descriptor chunk outside state")
            bit = 1 << index; mask |= bit
            if not self.completed_bitmap & bit:
                self.future_access_attempts += 1
                raise ValueError("future router chunk access")
            if not self.revealed_bitmap & bit:
                self.unrevealed_execution += 1
                raise ValueError("unrevealed router chunk execution")
            if self.dispatched_bitmap & bit:
                self.stale_dispatch += 1
                raise ValueError("stale chunk dispatch")
            left, right = self.chunk_offsets[index:index + 2]
            positions.append(np.arange(left, right, dtype=np.int64))
        descriptor_positions = np.concatenate(positions)
        if np.any(self._dispatched_positions[descriptor_positions]):
            self.duplicate_dispatch += 1
            raise ValueError("duplicate token dispatch")

        count_start = time.perf_counter_ns()
        counts_array = np.sum(self._chunk_counts[list(normalized)], axis=0, dtype=np.int64)
        counts = tuple(int(value) for value in counts_array)
        count_done = time.perf_counter_ns()
        offsets_array = np.empty(self.world_size, dtype=np.int64)
        offsets_array[0] = 0
        if self.world_size > 1:
            np.cumsum(counts_array[:-1], out=offsets_array[1:])
        offsets = tuple(int(value) for value in offsets_array)
        offset_done = time.perf_counter_ns()

        if self._count_ticket is not None:
            raise RuntimeError("unconsumed count-exchange ticket")
        if count_exchange_launcher is not None:
            self._count_ticket = count_exchange_launcher(counts)

        pack_start = time.perf_counter_ns()
        descriptor_meta = self._descriptor_meta[self.descriptor_count]
        descriptor_features = self._descriptor_features[self.descriptor_count]
        cursor = 0
        for destination in range(self.world_size):
            for chunk in normalized:
                count = int(self._chunk_counts[chunk, destination])
                if count:
                    descriptor_meta[cursor:cursor + count] = self._chunk_meta[chunk][destination][:count]
                    descriptor_features[cursor:cursor + count] = self._chunk_features[chunk][destination][:count]
                    cursor += count
        pack_done = time.perf_counter_ns()
        if cursor != int(np.sum(counts_array)):
            raise RuntimeError("fast descriptor cardinality mismatch")

        self._dispatched_positions[descriptor_positions] = True
        self.dispatched_bitmap |= mask
        self.descriptor_count += 1
        if timing_sink is not None:
            timing_sink["count_construction_us"] = (count_done - count_start) / 1e3
            timing_sink["offset_construction_us"] = (offset_done - count_done) / 1e3
            timing_sink["packing_us"] = (pack_done - pack_start) / 1e3
            timing_sink["packing_start_host_ns"] = int(pack_start)
            timing_sink["packing_done_host_ns"] = int(pack_done)
            timing_sink["fast_mark_completed_total_us"] = self.mark_completed_total_us
        return ForwardPayload(
            sendcounts_tokens=counts, offsets_tokens=offsets,
            metadata=descriptor_meta[:cursor], features=descriptor_features[:cursor],
        )

    def take_count_ticket(self) -> Any:
        ticket = self._count_ticket
        self._count_ticket = None
        return ticket

    @property
    def dispatched_token_count(self) -> int:
        return int(np.count_nonzero(self._dispatched_positions))
