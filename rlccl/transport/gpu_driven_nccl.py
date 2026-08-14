"""Job-level ctypes owner for the R6-M6 NCCL LSA runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from rlccl.scheduler.common.compiled_plan import CompiledPlanBlob, validate_compiled_plan
from rlccl.transport.cuda.layout import GPURegisteredBufferLayout
from rlccl.transport.device_transport import DeviceTransportBackend


COUNTER_NAMES = (
    "scheduler_actions", "transport_actions", "gpu_pack_calls",
    "lsa_transfers", "lsa_bytes_transferred", "lsa_arrives",
    "lsa_waits", "transport_errors", "slot_replays", "future_access",
    "unrevealed_access", "stale_action",
)

CAPABILITY_NAMES = (
    "nccl_version", "device_api_support", "multimem_support", "gin_type",
    "n_lsa_teams", "lsa_size", "symmetric_window",
)


class GPUDrivenNcclLsaRuntime:
    """One NCCL communicator/window and one job-level four-role kernel."""

    backend = DeviceTransportBackend.NCCL_LSA

    @staticmethod
    def get_unique_id(library: str | Path) -> bytes:
        lib = ctypes.CDLL(str(library))
        lib.r6_m6_unique_id_size.restype = ctypes.c_size_t
        size = int(lib.r6_m6_unique_id_size())
        storage = ctypes.create_string_buffer(size)
        lib.r6_m6_get_unique_id.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if lib.r6_m6_get_unique_id(storage, size) != 0:
            lib.r6_m6_last_error.restype = ctypes.c_char_p
            raw = lib.r6_m6_last_error()
            raise RuntimeError(raw.decode(errors="replace") if raw else "NCCL unique ID failed")
        return storage.raw

    def __init__(
        self,
        library: str | Path,
        *,
        rank: int,
        device: int,
        unique_id: bytes,
        plan: CompiledPlanBlob,
        layout: GPURegisteredBufferLayout,
    ) -> None:
        validate_compiled_plan(plan)
        if plan.config.world_size != layout.world_size:
            raise ValueError("plan/layout world size mismatch")
        if plan.config.record_bytes != layout.record_bytes:
            raise ValueError("plan/layout record size mismatch")
        if plan.descriptor_stride != layout.descriptor_stride or plan.region_bytes != layout.region_bytes:
            raise ValueError("plan/layout offset semantics mismatch")
        if not unique_id:
            raise ValueError("NCCL unique ID must be non-empty")
        self.rank = int(rank)
        self.device = int(device)
        self.layout = layout
        self.plan = plan
        self.unique_id = bytes(unique_id)
        self._unique_id_storage = ctypes.create_string_buffer(self.unique_id)
        self._plan_storage = ctypes.create_string_buffer(plan.data)
        self._lib = ctypes.CDLL(str(library))
        self._lib.r6_m6_last_error.restype = ctypes.c_char_p
        self._lib.r6_m6_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ]
        self._lib.r6_m6_create.restype = ctypes.c_void_p
        self._lib.r6_m6_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_size_t,
        ]
        self._lib.r6_m6_counter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_m6_counter.restype = ctypes.c_uint64
        self._lib.r6_m6_capability.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_m6_capability.restype = ctypes.c_uint64
        for name in ("action_count", "trace_count", "timing_count"):
            function = getattr(self._lib, f"r6_m6_{name}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_size_t
        self._lib.r6_m6_copy_actions.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m6_copy_traces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m6_copy_timings.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m6_copy_buffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m6_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._lib.r6_m6_create(
            self.rank, self.device, ctypes.cast(self._unique_id_storage, ctypes.c_void_p),
            len(self.unique_id), layout.capacity_bytes,
            ctypes.cast(self._plan_storage, ctypes.c_void_p), len(plan.data),
        )
        if not self._handle:
            self._raise("create")
        self.cpu_per_descriptor_packing = 0
        self.cpu_per_descriptor_transport_submission = 0
        self.cpu_per_descriptor_cuda_launch = 0

    def _raise(self, operation: str) -> None:
        raw = self._lib.r6_m6_last_error()
        detail = raw.decode(errors="replace") if raw else "unknown error"
        raise RuntimeError(f"R6-M6 {operation} failed: {detail}")

    @staticmethod
    def _pointer(tensor: Any) -> ctypes.c_void_p:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("M6 runtime inputs must be contiguous CUDA tensors")
        return ctypes.c_void_p(tensor.data_ptr())

    def run(
        self,
        *,
        reveal_records: Any,
        destination_ranks: Any,
        expert_ids: Any,
        token_ids: Any,
        feature_digests: Any,
        features: Any,
        metadata: Any,
        expected_remote_actions: int,
        producer_delay_cycles: int,
        router_stream: int,
    ) -> dict[str, Any]:
        if reveal_records.ndim != 2 or reveal_records.shape[1] != 8:
            raise ValueError("reveal_records must have shape [N, 8]")
        total = int(destination_ranks.numel())
        if any(int(value.numel()) != total for value in (expert_ids, token_ids, feature_digests)):
            raise ValueError("M6 assignment tensor cardinality mismatch")
        if features.shape != (total, self.layout.feature_width):
            raise ValueError("M6 feature tensor shape mismatch")
        if metadata.shape != (total, self.layout.metadata_fields):
            raise ValueError("M6 metadata tensor shape mismatch")
        status = self._lib.r6_m6_run(
            self._handle, self._pointer(reveal_records), reveal_records.shape[0],
            self._pointer(destination_ranks), self._pointer(expert_ids),
            self._pointer(token_ids), self._pointer(feature_digests),
            self._pointer(features), self._pointer(metadata), total,
            self.layout.feature_width, int(expected_remote_actions),
            int(producer_delay_cycles), int(router_stream),
        )
        if status != 0:
            self._raise("run")
        action_count = int(self._lib.r6_m6_action_count(self._handle))
        trace_count = int(self._lib.r6_m6_trace_count(self._handle))
        timing_count = int(self._lib.r6_m6_timing_count(self._handle))
        actions = np.empty((action_count, 12), dtype=np.int64)
        traces = np.empty((trace_count, 19), dtype=np.uint64)
        timings = np.empty((timing_count, 6), dtype=np.uint64)
        for name, values in (("actions", actions), ("traces", traces), ("timings", timings)):
            function = getattr(self._lib, f"r6_m6_copy_{name}")
            if function(self._handle, values.ctypes.data_as(ctypes.c_void_p), len(values)) != 0:
                self._raise(f"copy {name}")
        registered_buffer = np.empty(self.layout.capacity_bytes, dtype=np.uint8)
        if self._lib.r6_m6_copy_buffer(
            self._handle, registered_buffer.ctypes.data_as(ctypes.c_void_p),
            registered_buffer.size,
        ) != 0:
            self._raise("copy registered buffer")
        return {
            "actions": actions,
            "traces": traces,
            "timings": timings,
            "counters": self.counters(),
            "capability": self.capability(),
            "registered_buffer": registered_buffer,
        }

    def counters(self) -> dict[str, int]:
        return {
            name: int(self._lib.r6_m6_counter(self._handle, index))
            for index, name in enumerate(COUNTER_NAMES)
        }

    def capability(self) -> dict[str, int]:
        return {
            name: int(self._lib.r6_m6_capability(self._handle, index))
            for index, name in enumerate(CAPABILITY_NAMES)
        }

    def close(self) -> None:
        if self._handle:
            self._lib.r6_m6_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = ["CAPABILITY_NAMES", "COUNTER_NAMES", "GPUDrivenNcclLsaRuntime"]
