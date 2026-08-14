"""Fixed-width, backend-neutral scheduler records.

No class in this module owns a CUDA pointer, stream, communicator, tensor, or
transport handle.  Offsets are logical byte offsets into a later backend's
registered regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag


class SchedulerErrorCode(IntEnum):
    NONE = 0
    PLAN_CHECKSUM = 1
    REVEAL_QUEUE_OVERFLOW = 2
    ACTION_QUEUE_OVERFLOW = 3
    STALE_REVEAL = 4
    DUPLICATE_DESCRIPTOR = 5
    INVALID_SOURCE_RANK = 6
    INVALID_DESTINATION_RANK = 7
    ZERO_TOKEN_ACTION = 8
    FUTURE_DEMAND = 9
    UNREVEALED_DEMAND = 10
    OFFSET_OVERFLOW = 11
    BYTES_OVERFLOW = 12
    INVALID_ROUTE = 13
    DESCRIPTOR_RANGE = 14
    CHUNK_RANGE = 15
    ASSIGNMENT_RANGE = 16
    INTERNAL = 17


class RevealFlags(IntFlag):
    NONE = 0
    INJECT_FUTURE = 1 << 0
    INJECT_UNREVEALED = 1 << 1
    INJECT_STALE_ACTION = 1 << 2
    INJECT_DUPLICATE_ACTION = 1 << 3
    INJECT_OFFSET_OVERFLOW = 1 << 4
    INJECT_INVALID_RANK = 1 << 5
    INJECT_INVALID_ROUTE = 1 << 6
    INJECT_ZERO_TOKEN_ACTION = 1 << 7
    INJECT_BYTES_OVERFLOW = 1 << 8


class ActionFlags(IntFlag):
    NONE = 0
    LOGICAL_OFFSETS = 1 << 0
    SELF = 1 << 1


@dataclass(frozen=True, slots=True)
class RevealRecord:
    chunk_id: int
    reveal_epoch: int
    token_begin: int
    token_count: int
    assignment_begin: int
    assignment_count: int
    descriptor_id: int
    flags: int = 0

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.chunk_id, self.reveal_epoch, self.token_begin,
            self.token_count, self.assignment_begin, self.assignment_count,
            self.descriptor_id, self.flags,
        )


@dataclass(frozen=True, slots=True)
class CompiledRouteTemplate:
    src_rank: int
    dst_rank: int
    route_id: int
    channel_id: int
    send_region_base: int
    recv_region_base: int
    flags: int


@dataclass(frozen=True, slots=True)
class CommittedAction:
    action_id: int
    descriptor_id: int
    chunk_id: int
    reveal_epoch: int
    src_rank: int
    dst_rank: int
    src_offset: int
    dst_offset: int
    token_count: int
    bytes: int
    route_id: int
    flags: int

    def comparison_tuple(self) -> tuple[int, ...]:
        return (
            self.action_id, self.descriptor_id, self.chunk_id,
            self.reveal_epoch, self.src_rank, self.dst_rank,
            self.src_offset, self.dst_offset, self.token_count,
            self.bytes, self.route_id, self.flags,
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    world_size: int
    source_rank: int
    record_bytes: int
    max_descriptors: int
    max_chunks: int
    max_tokens_per_peer: int
    reveal_queue_capacity: int = 64
    action_queue_capacity: int = 256
    block_size: int = 32

    def validate(self) -> None:
        if not 1 <= self.world_size <= 64:
            raise ValueError("world_size must be in [1, 64]")
        if not 0 <= self.source_rank < self.world_size:
            raise ValueError("source_rank outside world")
        if self.record_bytes <= 0 or self.record_bytes % 8:
            raise ValueError("record_bytes must be a positive multiple of eight")
        if min(self.max_descriptors, self.max_chunks, self.max_tokens_per_peer) <= 0:
            raise ValueError("scheduler capacities must be positive")
        if min(self.reveal_queue_capacity, self.action_queue_capacity) <= 0:
            raise ValueError("queue capacities must be positive")
        if self.block_size != 32:
            raise ValueError("R6-M4 freezes the persistent kernel to one warp")


@dataclass(frozen=True, slots=True)
class DeviceSchedulerError:
    error_code: int
    action_id: int
    descriptor_id: int
    reveal_epoch: int


__all__ = [
    "ActionFlags", "CommittedAction", "CompiledRouteTemplate",
    "DeviceSchedulerError", "RevealFlags", "RevealRecord", "SchedulerConfig",
    "SchedulerErrorCode",
]
