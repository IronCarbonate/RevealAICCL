"""Portable host mirror of the R6-M5 registered communication layout."""

from __future__ import annotations

from dataclasses import dataclass

from rlccl.scheduler.common.scheduler_schema import ActionFlags, CommittedAction


@dataclass(frozen=True, slots=True)
class PhysicalTransportAction:
    action_id: int
    descriptor_id: int
    src_rank: int
    dst_rank: int
    physical_src_offset: int
    physical_dst_offset: int
    payload_bytes: int
    physical_bytes: int


@dataclass(frozen=True, slots=True)
class GPURegisteredBufferLayout:
    world_size: int
    max_descriptors: int
    max_tokens_per_peer: int
    metadata_fields: int
    feature_width: int
    alignment: int = 8

    def __post_init__(self) -> None:
        if self.world_size != 2:
            raise ValueError("R6-M5 is frozen to two local GPUs")
        if min(self.max_descriptors, self.max_tokens_per_peer,
               self.metadata_fields, self.feature_width) <= 0:
            raise ValueError("M5 layout capacities must be positive")
        if self.alignment != 8:
            raise ValueError("MemoryChannel put<8> requires 8-byte alignment")
        if self.record_bytes % self.alignment:
            raise ValueError("packed record must preserve 8-byte alignment")

    @property
    def metadata_bytes(self) -> int:
        return self.metadata_fields * 8

    @property
    def feature_bytes(self) -> int:
        return self.feature_width * 4

    @property
    def record_bytes(self) -> int:
        return self.metadata_bytes + self.feature_bytes

    @property
    def peer_stride(self) -> int:
        raw = 8 + self.max_tokens_per_peer * self.record_bytes
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

    def send_offset(self, descriptor_id: int, destination_rank: int) -> int:
        self._check_slot(descriptor_id, destination_rank)
        return descriptor_id * self.descriptor_stride + destination_rank * self.peer_stride

    def receive_offset(self, descriptor_id: int, source_rank: int) -> int:
        self._check_slot(descriptor_id, source_rank)
        return self.region_bytes + descriptor_id * self.descriptor_stride + source_rank * self.peer_stride

    def _check_slot(self, descriptor_id: int, rank: int) -> None:
        if not 0 <= descriptor_id < self.max_descriptors:
            raise ValueError("descriptor outside registered layout")
        if not 0 <= rank < self.world_size:
            raise ValueError("rank outside registered layout")

    def map_action(self, action: CommittedAction) -> PhysicalTransportAction:
        """Validate and explicitly convert M4 logical offsets to M5 physical offsets."""
        if not action.flags & int(ActionFlags.LOGICAL_OFFSETS):
            raise ValueError("transport requires an explicitly logical M4 action")
        self._check_slot(action.descriptor_id, action.dst_rank)
        if not 0 <= action.src_rank < self.world_size:
            raise ValueError("source rank outside registered layout")
        if action.token_count <= 0 or action.token_count > self.max_tokens_per_peer:
            raise ValueError("token count outside peer slot")
        expected_src = self.send_offset(action.descriptor_id, action.dst_rank)
        expected_dst = self.receive_offset(action.descriptor_id, action.src_rank)
        if action.src_offset != expected_src or action.dst_offset != expected_dst:
            raise ValueError("logical action does not map to its unique physical slot")
        if action.bytes != action.token_count * self.record_bytes:
            raise ValueError("action byte count does not match packed records")
        physical_bytes = 8 + action.bytes
        if expected_src + physical_bytes > self.region_bytes:
            raise ValueError("physical send range exceeds registered region")
        if expected_dst + physical_bytes > self.capacity_bytes:
            raise ValueError("physical receive range exceeds registered buffer")
        if expected_src % 8 != expected_dst % 8:
            raise ValueError("put<8> source/destination phases differ")
        return PhysicalTransportAction(
            action.action_id, action.descriptor_id, action.src_rank,
            action.dst_rank, expected_src, expected_dst, action.bytes,
            physical_bytes,
        )


__all__ = ["GPURegisteredBufferLayout", "PhysicalTransportAction"]
