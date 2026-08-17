"""Host mirror of the R6-M7 direct/staged progressive dispatch window."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressiveDispatchLayout:
    world_size: int
    max_descriptors: int
    max_assignments_per_peer: int
    feature_width: int
    meta_storage_bytes: int = 32
    slot_header_bytes: int = 16
    alignment: int = 16

    def __post_init__(self) -> None:
        if min(self.world_size, self.max_descriptors,
               self.max_assignments_per_peer, self.feature_width) <= 0:
            raise ValueError("M7 layout dimensions must be positive")
        if self.meta_storage_bytes < 28 or self.meta_storage_bytes % 16:
            raise ValueError("M7 metadata storage must fit DispatchTokenMeta and align")
        if self.slot_header_bytes != 16 or self.alignment != 16:
            raise ValueError("M7 direct copy layout requires 16-byte header/alignment")

    @property
    def record_bytes(self) -> int:
        return self.meta_storage_bytes + self.feature_width * 4

    @property
    def peer_stride(self) -> int:
        raw = self.slot_header_bytes + self.max_assignments_per_peer * self.record_bytes
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

    def recv_offset(self, descriptor: int, source: int) -> int:
        return descriptor * self.descriptor_stride + source * self.peer_stride

    def staging_offset(self, descriptor: int, destination: int) -> int:
        return self.region_bytes + descriptor * self.descriptor_stride + destination * self.peer_stride


__all__ = ["ProgressiveDispatchLayout"]
