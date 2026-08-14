"""Backend-neutral contracts for job-level GPU device transports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from rlccl.transport.cuda.layout import PhysicalTransportAction


class DeviceTransportBackend(str, Enum):
    MSCCLPP = "mscclpp"
    NCCL_LSA = "nccl_lsa"
    NCCL_GIN = "nccl_gin"


@dataclass(frozen=True, slots=True)
class DeviceTransportConfig:
    backend: DeviceTransportBackend

    @classmethod
    def parse(cls, value: str | DeviceTransportBackend) -> "DeviceTransportConfig":
        try:
            return cls(DeviceTransportBackend(value))
        except ValueError as error:
            choices = ", ".join(item.value for item in DeviceTransportBackend)
            raise ValueError(f"transport_backend must be one of: {choices}") from error


@dataclass(frozen=True, slots=True)
class DeviceTransportCommand:
    """Exact transport-only projection of one frozen physical action."""

    action_id: int
    descriptor_id: int
    peer: int
    dst_offset: int
    src_offset: int
    bytes: int
    completion_id: int

    @classmethod
    def from_physical(
        cls, action: PhysicalTransportAction, *, completion_id: int
    ) -> "DeviceTransportCommand":
        if action.physical_bytes <= 0:
            raise ValueError("device transport rejects a zero-sized action")
        if action.physical_src_offset < 0 or action.physical_dst_offset < 0:
            raise ValueError("device transport rejects a negative offset")
        if action.physical_src_offset % 8 != action.physical_dst_offset % 8:
            raise ValueError("device transport requires matching 8-byte offset phase")
        if completion_id < 0:
            raise ValueError("completion_id must be non-negative")
        return cls(
            action_id=action.action_id,
            descriptor_id=action.descriptor_id,
            peer=action.dst_rank,
            dst_offset=action.physical_dst_offset,
            src_offset=action.physical_src_offset,
            bytes=action.physical_bytes,
            completion_id=completion_id,
        )


@dataclass(frozen=True, slots=True)
class DeviceTransportCapability:
    backend: DeviceTransportBackend
    runtime_available: bool
    reason: str
    nccl_version: str | None = None
    device_api: bool = False
    symmetric_window: bool = False
    lsa: bool = False
    gin: bool = False


@runtime_checkable
class DeviceTransportRuntime(Protocol):
    """Job owner; methods are job-level and never descriptor callbacks."""

    backend: DeviceTransportBackend

    def run(self, **job_inputs: object) -> dict[str, object]: ...

    def close(self) -> None: ...


__all__ = [
    "DeviceTransportBackend",
    "DeviceTransportCapability",
    "DeviceTransportCommand",
    "DeviceTransportConfig",
    "DeviceTransportRuntime",
]
