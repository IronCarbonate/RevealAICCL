"""Fail-closed NCCL Device API capability normalization for R6-M6."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from rlccl.transport.device_transport import (
    DeviceTransportBackend,
    DeviceTransportCapability,
)


MIN_QUERY_PROPERTIES_VERSION = (2, 29, 0)


def normalize_nccl_version(value: Any) -> tuple[int, int, int]:
    if isinstance(value, int):
        major = value // 10000
        minor = value % 10000 // 100
        patch = value % 100
        return major, minor, patch
    if isinstance(value, (tuple, list)) and 2 <= len(value) <= 3:
        parts = tuple(int(part) for part in value)
        return parts if len(parts) == 3 else (*parts, 0)
    raise ValueError("unsupported NCCL version encoding")


def capability_from_probe(probe: Mapping[str, Any]) -> DeviceTransportCapability:
    version = normalize_nccl_version(probe["nccl_version"])
    version_text = ".".join(map(str, version))
    if version < MIN_QUERY_PROPERTIES_VERSION:
        return DeviceTransportCapability(
            DeviceTransportBackend.NCCL_LSA,
            False,
            "NCCL_DEVICE_API_NOT_AVAILABLE: ncclCommQueryProperties requires NCCL >= 2.29",
            nccl_version=version_text,
        )
    device_api = bool(probe.get("device_api_support"))
    symmetric = bool(probe.get("symmetric_window"))
    lsa_size = int(probe.get("lsa_size", 0))
    lsa = device_api and symmetric and lsa_size >= 2
    gin_type = str(probe.get("gin_type", "NONE")).upper()
    gin = device_api and gin_type not in {"", "NONE", "NCCL_GIN_TYPE_NONE"}
    missing = []
    if not device_api:
        missing.append("deviceApiSupport=false")
    if not symmetric:
        missing.append("symmetric window unavailable")
    if lsa_size < 2:
        missing.append("LSA team size < 2")
    reason = "available" if not missing else "NCCL_DEVICE_API_NOT_AVAILABLE: " + "; ".join(missing)
    return DeviceTransportCapability(
        backend=DeviceTransportBackend.NCCL_LSA,
        runtime_available=lsa,
        reason=reason,
        nccl_version=version_text,
        device_api=device_api,
        symmetric_window=symmetric,
        lsa=lsa,
        gin=gin,
    )


def capability_json(capability: DeviceTransportCapability) -> dict[str, Any]:
    output = asdict(capability)
    output["backend"] = capability.backend.value
    return output


__all__ = [
    "MIN_QUERY_PROPERTIES_VERSION",
    "capability_from_probe",
    "capability_json",
    "normalize_nccl_version",
]
