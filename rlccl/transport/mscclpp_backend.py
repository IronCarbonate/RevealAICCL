"""Fail-closed bridge from guarded progressive descriptors to MSCCL++ puts.

This module intentionally contains no scheduler.  It accepts only a descriptor
whose scheduling step has already been accepted by ``DynamicGuard`` and emits
offset-based transfer actions for a long-lived, pre-registered buffer pair.
The CUDA/MSCCL++ implementation consumes these actions; unit tests can validate
the legality boundary without loading a GPU extension.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .reference_a2av import PAYLOAD_FIELDS, PackedShard


RECORD_BYTES = PAYLOAD_FIELDS * 8
COUNT_HEADER_BYTES = 8


@dataclass(frozen=True, slots=True)
class CommittedAction:
    action_id: int
    descriptor_id: int
    src_rank: int
    dst_rank: int
    src_offset: int
    dst_offset: int
    token_count: int
    bytes: int
    physical_bytes: int
    reveal_ids: tuple[int, ...]

    @property
    def is_remote(self) -> bool:
        return self.src_rank != self.dst_rank


@dataclass(frozen=True, slots=True)
class RegisteredBufferLayout:
    """Fixed offsets shared by both ranks for the lifetime of the runtime."""

    world_size: int
    max_descriptors: int
    max_tokens_per_peer_descriptor: int
    record_bytes: int = RECORD_BYTES
    alignment: int = 8

    def __post_init__(self) -> None:
        if self.world_size != 2:
            raise ValueError("R6-M1 is frozen to world_size=2")
        if self.max_descriptors <= 0 or self.max_tokens_per_peer_descriptor <= 0:
            raise ValueError("buffer capacities must be positive")
        if self.record_bytes <= 0 or self.record_bytes % 8:
            raise ValueError("record_bytes must be a positive multiple of 8")
        if self.alignment < 8 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a power of two and at least 8")

    @property
    def peer_stride(self) -> int:
        raw = COUNT_HEADER_BYTES + self.max_tokens_per_peer_descriptor * self.record_bytes
        return (raw + self.alignment - 1) & ~(self.alignment - 1)

    @property
    def descriptor_stride(self) -> int:
        return self.world_size * self.peer_stride

    @property
    def region_bytes(self) -> int:
        return self.max_descriptors * self.descriptor_stride

    @property
    def capacity_bytes(self) -> int:
        return 2 * self.region_bytes

    def _slot_offset(self, descriptor_id: int, peer_rank: int) -> int:
        if not 0 <= descriptor_id < self.max_descriptors:
            raise ValueError("descriptor exceeds registered buffer capacity")
        if not 0 <= peer_rank < self.world_size:
            raise ValueError("peer rank outside process group")
        return descriptor_id * self.descriptor_stride + peer_rank * self.peer_stride

    def send_offset(self, descriptor_id: int, destination_rank: int) -> int:
        return self._slot_offset(descriptor_id, destination_rank)

    def receive_offset(self, descriptor_id: int, source_rank: int) -> int:
        return self.region_bytes + self._slot_offset(descriptor_id, source_rank)


class MscclppCommittedAdapter:
    """Map guarded delta shards to runtime actions without rescheduling them."""

    def __init__(self, *, rank: int, layout: RegisteredBufferLayout) -> None:
        if not 0 <= rank < layout.world_size:
            raise ValueError("rank outside process group")
        self.rank = int(rank)
        self.layout = layout
        self._descriptor_ids: set[int] = set()
        self._action_id = 0
        self.actions_committed = 0
        self.payload_bytes_committed = 0
        self.physical_bytes_planned = 0
        self.future_access = 0
        self.unrevealed_access = 0
        self.stale_action = 0

    def commit_descriptor(
        self,
        packed: PackedShard,
        *,
        descriptor_id: int,
        guard_decision: Any,
        completed_chunks: Iterable[int],
        revealed_chunks: Iterable[int],
    ) -> tuple[CommittedAction, ...]:
        """Validate the commit boundary and allocate deterministic buffer slots.

        One action is emitted for every non-empty destination.  Its registered
        source slot starts with an int64 token-count header followed by payload
        records.  The header makes receiver placement self-describing without a
        count collective; ``bytes`` counts demand payload and ``physical_bytes``
        additionally counts that eight-byte control header.
        """

        index = int(descriptor_id)
        if index in self._descriptor_ids:
            self.stale_action += 1
            raise ValueError("stale descriptor replay")
        if not bool(getattr(guard_decision, "accepted", False)):
            raise ValueError("MSCCL++ adapter requires a DynamicGuard PASS")
        if int(getattr(guard_decision, "applied_actions", -1)) < 0:
            raise ValueError("invalid guard decision")
        if packed.source_rank != self.rank:
            raise ValueError("packed shard source rank mismatch")
        chunks = tuple(int(value) for value in packed.chunk_ids)
        if not chunks or len(chunks) != len(set(chunks)):
            raise ValueError("descriptor must name unique revealed chunks")
        completed = set(int(value) for value in completed_chunks)
        revealed = set(int(value) for value in revealed_chunks)
        if any(chunk not in completed for chunk in chunks):
            self.future_access += 1
            raise ValueError("future router chunk cannot enter MSCCL++")
        if any(chunk not in revealed for chunk in chunks):
            self.unrevealed_access += 1
            raise ValueError("unrevealed chunk cannot enter MSCCL++")
        if len(packed.sendcounts_tokens) != self.layout.world_size:
            raise ValueError("sendcount cardinality mismatch")
        if len(packed.offsets_tokens) != self.layout.world_size:
            raise ValueError("offset cardinality mismatch")

        actions: list[CommittedAction] = []
        for destination, (count, packed_offset) in enumerate(
            zip(packed.sendcounts_tokens, packed.offsets_tokens, strict=True)
        ):
            count = int(count)
            packed_offset = int(packed_offset)
            if count < 0 or packed_offset < 0:
                raise ValueError("negative count/offset")
            # A remote zero-token peer still receives the fixed int64 count
            # header.  This is a non-zero control transfer and pairs signal/wait
            # without manufacturing payload bytes.  Empty local slots need no op.
            if count == 0 and destination == self.rank:
                continue
            if count > self.layout.max_tokens_per_peer_descriptor:
                raise ValueError("peer payload exceeds registered slot capacity")
            if packed_offset + count > packed.total_tokens:
                raise ValueError("packed range exceeds descriptor payload")
            payload_bytes = count * self.layout.record_bytes
            physical_bytes = COUNT_HEADER_BYTES + payload_bytes
            base = self.layout.send_offset(index, destination)
            if base + physical_bytes > self.layout.capacity_bytes:
                raise ValueError("action exceeds registered buffer")
            action = CommittedAction(
                action_id=self._action_id,
                descriptor_id=index,
                src_rank=self.rank,
                dst_rank=destination,
                src_offset=base,
                dst_offset=self.layout.receive_offset(index, self.rank),
                token_count=count,
                bytes=payload_bytes,
                physical_bytes=physical_bytes,
                reveal_ids=chunks,
            )
            self._action_id += 1
            actions.append(action)

        self._descriptor_ids.add(index)
        self.actions_committed += len(actions)
        self.payload_bytes_committed += sum(action.bytes for action in actions)
        self.physical_bytes_planned += sum(action.physical_bytes for action in actions)
        return tuple(actions)

    def counters(self) -> dict[str, int]:
        return {
            "committed_actions": self.actions_committed,
            "committed_payload_bytes": self.payload_bytes_committed,
            "planned_physical_bytes": self.physical_bytes_planned,
            "future_access": self.future_access,
            "unrevealed_access": self.unrevealed_access,
            "stale_action": self.stale_action,
        }


def action_payload(action: CommittedAction) -> dict[str, int | list[int]]:
    result = asdict(action)
    result["reveal_ids"] = list(action.reveal_ids)
    return result


__all__ = [
    "COUNT_HEADER_BYTES",
    "RECORD_BYTES",
    "CommittedAction",
    "MscclppCommittedAdapter",
    "RegisteredBufferLayout",
    "action_payload",
]
