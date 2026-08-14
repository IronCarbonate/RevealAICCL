"""Job-level ctypes owner for the R6-M5 GPU-driven MSCCL++ runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from rlccl.scheduler.common.compiled_plan import CompiledPlanBlob, validate_compiled_plan
from rlccl.transport.cuda.layout import GPURegisteredBufferLayout


COUNTER_NAMES = (
    "scheduler_actions", "transport_actions", "gpu_pack_calls",
    "mscclpp_put_calls", "mscclpp_bytes_transferred", "mscclpp_signals",
    "mscclpp_waits", "transport_errors", "slot_replays", "future_access",
    "unrevealed_access", "stale_action",
)


class GPUDrivenMscclppRuntime:
    """One registered buffer/channel and four job-level device consumers."""

    def __init__(
        self,
        library: str | Path,
        *,
        rank: int,
        device: int,
        registered_buffer: Any,
        endpoint: str,
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
        if registered_buffer.numel() * registered_buffer.element_size() != layout.capacity_bytes:
            raise ValueError("registered buffer capacity mismatch")
        if not registered_buffer.is_cuda or not registered_buffer.is_contiguous():
            raise ValueError("registered buffer must be contiguous CUDA memory")
        self.rank = int(rank)
        self.device = int(device)
        self.layout = layout
        self.plan = plan
        self.registered_buffer = registered_buffer
        self._plan_storage = ctypes.create_string_buffer(plan.data)
        self._lib = ctypes.CDLL(str(library))
        self._lib.r6_m5_last_error.restype = ctypes.c_char_p
        self._lib.r6_m5_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
        ]
        self._lib.r6_m5_create.restype = ctypes.c_void_p
        self._lib.r6_m5_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_size_t,
        ]
        self._lib.r6_m5_counter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_m5_counter.restype = ctypes.c_uint64
        for name in ("action_count", "trace_count", "timing_count"):
            function = getattr(self._lib, f"r6_m5_{name}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_size_t
        self._lib.r6_m5_copy_actions.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m5_copy_traces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m5_copy_timings.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_m5_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._lib.r6_m5_create(
            self.rank, self.device, ctypes.c_void_p(registered_buffer.data_ptr()),
            layout.capacity_bytes, endpoint.encode(),
            ctypes.cast(self._plan_storage, ctypes.c_void_p), len(plan.data),
        )
        if not self._handle:
            self._raise("create")
        self.cpu_per_descriptor_packing = 0
        self.cpu_per_descriptor_transport_submission = 0
        self.cpu_per_descriptor_cuda_launch = 0

    def _raise(self, operation: str) -> None:
        raw = self._lib.r6_m5_last_error()
        detail = raw.decode(errors="replace") if raw else "unknown error"
        raise RuntimeError(f"R6-M5 {operation} failed: {detail}")

    @staticmethod
    def _pointer(tensor: Any) -> ctypes.c_void_p:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("M5 runtime inputs must be contiguous CUDA tensors")
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
            raise ValueError("M5 assignment tensor cardinality mismatch")
        if features.shape != (total, self.layout.feature_width):
            raise ValueError("M5 feature tensor shape mismatch")
        if metadata.shape != (total, self.layout.metadata_fields):
            raise ValueError("M5 metadata tensor shape mismatch")
        status = self._lib.r6_m5_run(
            self._handle, self._pointer(reveal_records), reveal_records.shape[0],
            self._pointer(destination_ranks), self._pointer(expert_ids),
            self._pointer(token_ids), self._pointer(feature_digests),
            self._pointer(features), self._pointer(metadata), total,
            self.layout.feature_width, int(expected_remote_actions),
            int(producer_delay_cycles), int(router_stream),
        )
        if status != 0:
            self._raise("run")
        action_count = int(self._lib.r6_m5_action_count(self._handle))
        trace_count = int(self._lib.r6_m5_trace_count(self._handle))
        timing_count = int(self._lib.r6_m5_timing_count(self._handle))
        actions = np.empty((action_count, 12), dtype=np.int64)
        traces = np.empty((trace_count, 19), dtype=np.uint64)
        timings = np.empty((timing_count, 6), dtype=np.uint64)
        for name, values in (("actions", actions), ("traces", traces), ("timings", timings)):
            function = getattr(self._lib, f"r6_m5_copy_{name}")
            if function(self._handle, values.ctypes.data_as(ctypes.c_void_p), len(values)) != 0:
                self._raise(f"copy {name}")
        return {
            "actions": actions,
            "traces": traces,
            "timings": timings,
            "counters": self.counters(),
        }

    def counters(self) -> dict[str, int]:
        return {
            name: int(self._lib.r6_m5_counter(self._handle, index))
            for index, name in enumerate(COUNTER_NAMES)
        }

    def close(self) -> None:
        if self._handle:
            self._lib.r6_m5_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = ["COUNTER_NAMES", "GPUDrivenMscclppRuntime"]
