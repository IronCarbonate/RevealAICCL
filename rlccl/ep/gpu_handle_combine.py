"""Job-level ctypes owner for the R6-M8 handle-driven combine runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from rlccl.ep.gpu_progressive_ep import CAPABILITY_NAMES, COUNTER_NAMES, DISPATCH_META_DTYPE
from rlccl.ep.layout import ProgressiveDispatchLayout
from rlccl.scheduler.common.compiled_plan import CompiledPlanBlob, validate_compiled_plan


COMBINE_COUNTER_NAMES = (
    "rows_mapped", "local_returns", "remote_returns", "remote_bytes",
    "lsa_arrives", "lsa_waits", "contributions_reduced", "errors",
    "stale_handle", "range_bounds", "wrong_source_rank", "wrong_token",
    "wrong_topk_slot", "wrong_expert", "slot_collision", "missing_return",
    "corruption",
)


class GPUHandleCombineRuntime:
    """Own the frozen M7 forward path and the M8 full-handle return path."""

    @staticmethod
    def get_unique_id(library: str | Path) -> bytes:
        lib = ctypes.CDLL(str(library))
        lib.r6_m8_unique_id_size.restype = ctypes.c_size_t
        size = int(lib.r6_m8_unique_id_size())
        storage = ctypes.create_string_buffer(size)
        lib.r6_m8_get_unique_id.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if lib.r6_m8_get_unique_id(storage, size) != 0:
            lib.r6_m8_last_error.restype = ctypes.c_char_p
            raw = lib.r6_m8_last_error()
            raise RuntimeError(raw.decode(errors="replace") if raw else "NCCL unique ID failed")
        return storage.raw

    def __init__(
        self, library: str | Path, *, rank: int, device: int,
        unique_id: bytes, plan: CompiledPlanBlob,
        dispatch_layout: ProgressiveDispatchLayout,
        num_source_tokens: int, num_topk: int,
    ) -> None:
        validate_compiled_plan(plan)
        if plan.config.world_size != dispatch_layout.world_size:
            raise ValueError("M8 plan/layout world-size mismatch")
        self.rank = int(rank)
        self.device = int(device)
        self.plan = plan
        self.dispatch_layout = dispatch_layout
        self.num_source_tokens = int(num_source_tokens)
        self.num_topk = int(num_topk)
        self._uid = ctypes.create_string_buffer(bytes(unique_id))
        self._plan = ctypes.create_string_buffer(plan.data)
        self._lib = ctypes.CDLL(str(library))
        self._lib.r6_m8_last_error.restype = ctypes.c_char_p
        self._lib.r6_m8_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32,
        ]
        self._lib.r6_m8_create.restype = ctypes.c_void_p
        self._lib.r6_m8_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_size_t,
        ]
        self._lib.r6_m8_run.restype = ctypes.c_int
        for name in ("counter", "combine_counter", "capability"):
            function = getattr(self._lib, f"r6_m8_{name}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_int]
            function.restype = ctypes.c_uint64
        for name in (
            "num_recv_tokens", "num_local_experts", "return_trace_count",
            "num_source_tokens",
        ):
            function = getattr(self._lib, f"r6_m8_{name}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_size_t
        for name in (
            "copy_expert_counts", "copy_expert_offsets", "copy_recv_x",
            "copy_recv_metadata", "copy_return_traces", "copy_final_output",
        ):
            function = getattr(self._lib, f"r6_m8_{name}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        self._lib.r6_m8_destroy.argtypes = [ctypes.c_void_p]
        layout = dispatch_layout
        self._handle = self._lib.r6_m8_create(
            self.rank, self.device, ctypes.cast(self._uid, ctypes.c_void_p),
            len(unique_id), ctypes.cast(self._plan, ctypes.c_void_p), len(plan.data),
            layout.world_size, layout.max_descriptors,
            layout.max_assignments_per_peer, layout.feature_width,
            layout.peer_stride, layout.descriptor_stride, layout.region_bytes,
            self.num_source_tokens, self.num_topk,
        )
        if not self._handle:
            self._raise("create")
        self.python_callback_per_output = 0
        self.cpu_poll_per_output = 0
        self.cpu_return_construction_per_output = 0
        self.cpu_packing_per_output = 0
        self.cpu_transport_submission_per_output = 0
        self.cpu_cuda_launch_per_output = 0

    def _raise(self, operation: str) -> None:
        raw = self._lib.r6_m8_last_error()
        detail = raw.decode(errors="replace") if raw else "unknown error"
        raise RuntimeError(f"R6-M8 {operation} failed: {detail}")

    @staticmethod
    def _pointer(tensor: Any) -> ctypes.c_void_p:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("M8 inputs must be contiguous CUDA tensors")
        return ctypes.c_void_p(tensor.data_ptr())

    def _copy(self, name: str, values: np.ndarray) -> None:
        capacity = values.size if name in ("recv_x", "final_output") else len(values)
        if getattr(self._lib, f"r6_m8_copy_{name}")(
            self._handle, values.ctypes.data_as(ctypes.c_void_p), capacity,
        ) != 0:
            self._raise(f"copy {name}")

    def run(
        self, *, reveal_records: Any, x: Any, topk_idx: Any,
        topk_weights: Any, expert_weights: Any, experts_per_rank: int,
        producer_delay_cycles: int, router_stream: int,
    ) -> dict[str, Any]:
        if topk_idx.shape != (self.num_source_tokens, self.num_topk):
            raise ValueError("M8 topk_idx shape mismatch")
        if topk_weights.shape != topk_idx.shape:
            raise ValueError("M8 topk weight shape mismatch")
        hidden = self.dispatch_layout.feature_width
        if x.shape != (self.num_source_tokens, hidden):
            raise ValueError("M8 x shape mismatch")
        if expert_weights.shape != (experts_per_rank, hidden, hidden):
            raise ValueError("M8 expert weight shape mismatch")
        num_experts = int(experts_per_rank) * self.dispatch_layout.world_size
        status = self._lib.r6_m8_run(
            self._handle, self._pointer(reveal_records), reveal_records.shape[0],
            self._pointer(x), self._pointer(topk_idx), self._pointer(topk_weights),
            self._pointer(expert_weights), self.num_source_tokens, hidden,
            self.num_topk, int(experts_per_rank), num_experts,
            int(producer_delay_cycles), int(router_stream),
        )
        if status != 0:
            self._raise("run")
        num_recv = int(self._lib.r6_m8_num_recv_tokens(self._handle))
        num_local = int(self._lib.r6_m8_num_local_experts(self._handle))
        trace_count = int(self._lib.r6_m8_return_trace_count(self._handle))
        arrays = {
            "expert_counts": np.empty(num_local, dtype=np.uint32),
            "expert_offsets": np.empty(num_local + 1, dtype=np.uint32),
            "recv_x": np.empty((num_recv, hidden), dtype=np.float32),
            "recv_metadata": np.empty(num_recv, dtype=DISPATCH_META_DTYPE),
            "return_traces": np.empty((trace_count, 11), dtype=np.uint64),
            "final_output": np.empty((self.num_source_tokens, hidden), dtype=np.float32),
        }
        for name, values in arrays.items():
            self._copy(name, values)
        return {
            **arrays,
            "dispatch_counters": {
                name: int(self._lib.r6_m8_counter(self._handle, index))
                for index, name in enumerate(COUNTER_NAMES)
            },
            "combine_counters": {
                name: int(self._lib.r6_m8_combine_counter(self._handle, index))
                for index, name in enumerate(COMBINE_COUNTER_NAMES)
            },
            "capability": {
                name: int(self._lib.r6_m8_capability(self._handle, index))
                for index, name in enumerate(CAPABILITY_NAMES)
            },
            "handle": {
                "num_recv_tokens": num_recv,
                "num_local_experts": num_local,
                "num_source_tokens": self.num_source_tokens,
                "num_topk": self.num_topk,
                "generation": 1,
            },
        }

    def close(self) -> None:
        if self._handle:
            self._lib.r6_m8_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = ["COMBINE_COUNTER_NAMES", "GPUHandleCombineRuntime"]
