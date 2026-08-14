"""Frozen action mapping and backend-neutral contracts for R6-M6."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rlccl.transport.cuda.layout import PhysicalTransportAction
from rlccl.transport.device_transport import (
    DeviceTransportBackend,
    DeviceTransportCommand,
    DeviceTransportConfig,
)
from rlccl.transport.nccl_device_capability import (
    capability_from_probe,
    capability_json,
    normalize_nccl_version,
)


def _physical() -> PhysicalTransportAction:
    return PhysicalTransportAction(
        action_id=7,
        descriptor_id=3,
        src_rank=0,
        dst_rank=1,
        physical_src_offset=128,
        physical_dst_offset=4096,
        payload_bytes=136,
        physical_bytes=144,
    )


@pytest.mark.parametrize("name", ["mscclpp", "nccl_lsa", "nccl_gin"])
def test_backend_selection_is_explicit_and_closed(name: str) -> None:
    config = DeviceTransportConfig.parse(name)
    assert config.backend.value == name
    with pytest.raises(ValueError, match="transport_backend"):
        DeviceTransportConfig.parse("auto")


def test_physical_action_maps_exactly_to_device_transport_parameters() -> None:
    action = _physical()
    command = DeviceTransportCommand.from_physical(action, completion_id=11)
    assert command.action_id == action.action_id
    assert command.descriptor_id == action.descriptor_id
    assert command.peer == action.dst_rank
    assert command.dst_offset == action.physical_dst_offset
    assert command.src_offset == action.physical_src_offset
    assert command.bytes == action.physical_bytes
    assert command.completion_id == 11


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("physical_bytes", 0, "zero-sized"),
        ("physical_src_offset", -8, "negative offset"),
        ("physical_src_offset", 4, "8-byte offset phase"),
    ],
)
def test_transport_mapping_fails_closed(field: str, value: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DeviceTransportCommand.from_physical(
            replace(_physical(), **{field: value}), completion_id=0,
        )


def test_transport_mapping_rejects_negative_completion() -> None:
    with pytest.raises(ValueError, match="completion_id"):
        DeviceTransportCommand.from_physical(_physical(), completion_id=-1)


def test_scheduler_and_packer_inputs_do_not_include_backend() -> None:
    action = _physical()
    assert not hasattr(action, "backend")
    assert set(item.value for item in DeviceTransportBackend) == {
        "mscclpp", "nccl_lsa", "nccl_gin",
    }


def test_nccl_version_normalization_handles_runtime_encodings() -> None:
    assert normalize_nccl_version(22907) == (2, 29, 7)
    assert normalize_nccl_version((2, 30)) == (2, 30, 0)


def test_pre_229_nccl_fails_closed_without_device_api_claim() -> None:
    capability = capability_from_probe({"nccl_version": (2, 27, 3)})
    assert not capability.runtime_available
    assert not capability.device_api
    assert "ncclCommQueryProperties" in capability.reason


def test_lsa_capability_requires_query_window_and_two_rank_team() -> None:
    capability = capability_from_probe({
        "nccl_version": (2, 29, 7),
        "device_api_support": True,
        "symmetric_window": True,
        "lsa_size": 2,
        "gin_type": "NCCL_GIN_TYPE_NONE",
    })
    assert capability.runtime_available and capability.lsa
    assert not capability.gin
    assert capability_json(capability)["backend"] == "nccl_lsa"


def test_gin_capability_is_reported_but_not_inferred_from_lsa() -> None:
    capability = capability_from_probe({
        "nccl_version": 23007,
        "device_api_support": True,
        "symmetric_window": True,
        "lsa_size": 2,
        "gin_type": "NCCL_GIN_TYPE_PROXY",
    })
    assert capability.lsa and capability.gin


def test_native_lsa_runtime_uses_only_real_device_api_transport() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "extensions" / "r6_m6_nccl_device" /
              "nccl_device_transport_runtime.cu").read_text(encoding="utf-8")
    for required in (
        "ncclCommQueryProperties", "ncclMemAlloc", "ncclCommWindowRegister",
        "NCCL_WIN_COLL_SYMMETRIC", "ncclDevCommCreate", "ncclGetPeerPointer",
        "completion.arrive", "completion.wait", "cuda::memory_order_release",
        "cuda::memory_order_acquire",
    ):
        assert required in source
    for forbidden in (
        "mscclpp::", "ncclAllToAll", "ncclAllToAllv", "ncclSend", "ncclRecv",
    ):
        assert forbidden not in source


def test_backend_neutral_device_contract_has_no_nccl_or_mscclpp_type() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "transport" / "cuda" /
              "device_transport.cuh").read_text(encoding="utf-8")
    assert "DeviceTransportRequest" in source
    assert "ncclDevComm" not in source
    assert "ncclWindow" not in source
    assert "mscclpp::" not in source


def test_gin_compile_surface_maps_action_and_separates_completion() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "rlccl" / "transport" / "cuda" /
              "nccl_gin_transport.cuh").read_text(encoding="utf-8")
    for required in (
        "ncclGin gin", "gin.put", "request.peer", "request.dst_offset",
        "request.src_offset", "request.bytes", "ncclGin_SignalInc",
        "ncclGin_CounterInc", "test_completion", "wait_completion",
        "waitSignal", "waitCounter",
        "NCCL_GIN_CONNECTION_FULL",
    ):
        assert required in source
