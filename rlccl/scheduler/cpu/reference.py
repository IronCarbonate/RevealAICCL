"""Deterministic CPU oracle for byte-for-byte GPU scheduler comparison.

This is a shadow adapter; it does not replace or modify the existing CPU
scheduler.  It freezes the R6 descriptor/action semantics at the new portable
IR boundary so CUDA and future Ascend implementations share one oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.compiled_plan import CompiledPlanBlob, validate_compiled_plan
from ..common.scheduler_schema import (
    CommittedAction, DeviceSchedulerError, RevealFlags, RevealRecord,
    SchedulerErrorCode,
)


@dataclass(frozen=True, slots=True)
class SchedulerRun:
    actions: tuple[CommittedAction, ...]
    errors: tuple[DeviceSchedulerError, ...]
    revealed_count: tuple[int, ...]
    committed_count: tuple[int, ...]
    processed_reveals: int


class CPUSchedulerShadow:
    """Single-consumer incremental scheduler with fail-closed commits."""

    def __init__(self, plan: CompiledPlanBlob) -> None:
        validate_compiled_plan(plan)
        self.plan = plan
        self.config = plan.config
        pairs = self.config.world_size * self.config.world_size
        self.revealed_count = [0] * pairs
        self.committed_count = [0] * pairs
        self.next_send_offset = [0] * pairs
        self.next_recv_offset = [0] * pairs
        self.last_reveal_epoch = -1
        self.descriptor_epoch = [-1] * self.config.max_descriptors
        self.next_action_id = 0
        self.actions: list[CommittedAction] = []
        self.errors: list[DeviceSchedulerError] = []
        self.processed_reveals = 0

    def _reject(self, code: SchedulerErrorCode, record: RevealRecord) -> None:
        self.errors.append(DeviceSchedulerError(
            int(code), self.next_action_id, record.descriptor_id,
            record.reveal_epoch,
        ))

    def process(self, record: RevealRecord, dst_ranks: tuple[int, ...]) -> None:
        cfg = self.config
        self.processed_reveals += 1
        if not 0 <= record.chunk_id < cfg.max_chunks:
            self._reject(SchedulerErrorCode.CHUNK_RANGE, record); return
        if not 0 <= record.descriptor_id < cfg.max_descriptors:
            self._reject(SchedulerErrorCode.DESCRIPTOR_RANGE, record); return
        if record.reveal_epoch <= self.last_reveal_epoch:
            self._reject(SchedulerErrorCode.STALE_REVEAL, record); return
        if self.descriptor_epoch[record.descriptor_id] >= 0:
            self._reject(SchedulerErrorCode.DUPLICATE_DESCRIPTOR, record); return
        left = record.assignment_begin
        right = left + record.assignment_count
        if left < 0 or right > len(dst_ranks):
            self._reject(SchedulerErrorCode.ASSIGNMENT_RANGE, record); return
        counts = [0] * cfg.world_size
        for destination in dst_ranks[left:right]:
            if not 0 <= destination < cfg.world_size:
                self._reject(SchedulerErrorCode.INVALID_DESTINATION_RANK, record); return
            counts[destination] += 1
        needed = sum(value > 0 for value in counts)
        if len(self.actions) + needed > cfg.action_queue_capacity:
            self._reject(SchedulerErrorCode.ACTION_QUEUE_OVERFLOW, record); return

        candidates: list[tuple[int, CommittedAction]] = []
        for dst, raw_count in enumerate(counts):
            if raw_count == 0:
                continue
            pair = cfg.source_rank * cfg.world_size + dst
            route_index = self.plan.rank_pair_to_route[pair]
            route = self.plan.route_templates[route_index] if 0 <= route_index < len(self.plan.route_templates) else None
            count = raw_count
            src_rank = cfg.source_rank
            reveal_epoch = record.reveal_epoch
            flags = RevealFlags(record.flags)
            if flags & RevealFlags.INJECT_FUTURE:
                count += 1
            if flags & RevealFlags.INJECT_ZERO_TOKEN_ACTION:
                count = 0
            available = self.revealed_count[pair] + raw_count
            if flags & RevealFlags.INJECT_UNREVEALED:
                available = 0
            if flags & RevealFlags.INJECT_STALE_ACTION:
                reveal_epoch = max(0, reveal_epoch - 1)
            if flags & RevealFlags.INJECT_DUPLICATE_ACTION:
                self.descriptor_epoch[record.descriptor_id] = record.reveal_epoch
            if flags & RevealFlags.INJECT_INVALID_RANK:
                src_rank = cfg.world_size
            route_id = route.route_id if route is not None else -1
            if flags & RevealFlags.INJECT_INVALID_ROUTE:
                route_id += 1
            src_offset = record.descriptor_id * self.plan.descriptor_stride + (route.send_region_base if route else 0)
            dst_offset = record.descriptor_id * self.plan.descriptor_stride + (route.recv_region_base if route else 0)
            if flags & RevealFlags.INJECT_OFFSET_OVERFLOW:
                src_offset = self.plan.region_bytes * 2
            byte_count = count * cfg.record_bytes
            if flags & RevealFlags.INJECT_BYTES_OVERFLOW:
                byte_count = (cfg.max_tokens_per_peer + 1) * cfg.record_bytes
            action = CommittedAction(
                self.next_action_id + len(candidates), record.descriptor_id,
                record.chunk_id, reveal_epoch, src_rank, dst, src_offset,
                dst_offset, count, byte_count, route_id,
                route.flags if route else 0,
            )
            code = self._guard(action, record, pair, available, route_index)
            if code is not SchedulerErrorCode.NONE:
                self._reject(code, record)
                return
            candidates.append((pair, action))

        for dst, value in enumerate(counts):
            pair = cfg.source_rank * cfg.world_size + dst
            self.revealed_count[pair] += value
        for pair, action in candidates:
            self.committed_count[pair] += action.token_count
            self.next_send_offset[pair] += action.bytes
            self.next_recv_offset[pair] += action.bytes
            self.actions.append(action)
        self.next_action_id += len(candidates)
        self.descriptor_epoch[record.descriptor_id] = record.reveal_epoch
        self.last_reveal_epoch = record.reveal_epoch

    def _guard(
        self, action: CommittedAction, record: RevealRecord, pair: int,
        available: int, route_index: int,
    ) -> SchedulerErrorCode:
        cfg = self.config
        if not 0 <= action.src_rank < cfg.world_size:
            return SchedulerErrorCode.INVALID_SOURCE_RANK
        if not 0 <= action.dst_rank < cfg.world_size:
            return SchedulerErrorCode.INVALID_DESTINATION_RANK
        if action.token_count <= 0:
            return SchedulerErrorCode.ZERO_TOKEN_ACTION
        if self.descriptor_epoch[action.descriptor_id] >= 0:
            return SchedulerErrorCode.DUPLICATE_DESCRIPTOR
        if action.reveal_epoch != record.reveal_epoch:
            return SchedulerErrorCode.STALE_REVEAL
        if available <= self.committed_count[pair]:
            return SchedulerErrorCode.UNREVEALED_DEMAND
        if self.committed_count[pair] + action.token_count > available:
            return SchedulerErrorCode.FUTURE_DEMAND
        if not 0 <= route_index < len(self.plan.route_templates):
            return SchedulerErrorCode.INVALID_ROUTE
        route = self.plan.route_templates[route_index]
        if (
            not self.plan.legality_flags[route_index]
            or route.src_rank != action.src_rank or route.dst_rank != action.dst_rank
            or route.route_id != action.route_id
        ):
            return SchedulerErrorCode.INVALID_ROUTE
        slot_bytes = 8 + cfg.max_tokens_per_peer * cfg.record_bytes
        if action.bytes > cfg.max_tokens_per_peer * cfg.record_bytes:
            return SchedulerErrorCode.BYTES_OVERFLOW
        if action.src_offset + 8 + action.bytes > self.plan.region_bytes:
            return SchedulerErrorCode.OFFSET_OVERFLOW
        if action.dst_offset + 8 + action.bytes > self.plan.region_bytes * 2:
            return SchedulerErrorCode.OFFSET_OVERFLOW
        if (action.src_offset % self.plan.descriptor_stride) + 8 + action.bytes > self.plan.descriptor_stride:
            return SchedulerErrorCode.OFFSET_OVERFLOW
        del slot_bytes
        return SchedulerErrorCode.NONE

    def run(self, records: tuple[RevealRecord, ...], dst_ranks: tuple[int, ...]) -> SchedulerRun:
        for record in records:
            self.process(record, dst_ranks)
        return SchedulerRun(
            tuple(self.actions), tuple(self.errors), tuple(self.revealed_count),
            tuple(self.committed_count), self.processed_reveals,
        )


__all__ = ["CPUSchedulerShadow", "SchedulerRun"]
