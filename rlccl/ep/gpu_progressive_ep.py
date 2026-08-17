"""Job-level Python owner for the R6-M7 NCCL LSA dispatch runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from rlccl.ep.layout import ProgressiveDispatchLayout
from rlccl.scheduler.common.compiled_plan import CompiledPlanBlob, validate_compiled_plan


COUNTER_NAMES = (
    "descriptor_commits", "shadow_actions", "assignments_scanned",
    "direct_remote_records", "direct_remote_bytes", "local_records",
    "lsa_arrives", "lsa_waits", "epilogue_records", "errors",
    "unauthorized_destination", "cursor_overflow", "future_access",
    "unrevealed_access", "stale_action",
)

CAPABILITY_NAMES = (
    "nccl_version", "device_api_support", "multimem_support", "gin_type",
    "n_lsa_teams", "lsa_size", "symmetric_window",
)

DISPATCH_META_DTYPE = np.dtype([
    ("src_rank", "<u4"), ("src_token_idx", "<u4"),
    ("expert_id", "<u4"), ("topk_slot", "<u4"),
    ("descriptor_id", "<u4"), ("reveal_epoch", "<u4"),
    ("topk_weight", "<f4"),
], align=False)


class GPUProgressiveEPRuntime:
    """Own one communicator, symmetric window, and persistent M7 pipeline."""

    @staticmethod
    def get_unique_id(library: str | Path) -> bytes:
        lib = ctypes.CDLL(str(library))
        lib.r6_m7_unique_id_size.restype = ctypes.c_size_t
        size = int(lib.r6_m7_unique_id_size())
        storage = ctypes.create_string_buffer(size)
        lib.r6_m7_get_unique_id.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if lib.r6_m7_get_unique_id(storage, size) != 0:
            lib.r6_m7_last_error.restype = ctypes.c_char_p
            raw = lib.r6_m7_last_error()
            raise RuntimeError(raw.decode(errors="replace") if raw else "NCCL unique ID failed")
        return storage.raw

    def __init__(
        self, library: str | Path, *, rank: int, device: int,
        unique_id: bytes, plan: CompiledPlanBlob,
        layout: ProgressiveDispatchLayout,
    ) -> None:
        validate_compiled_plan(plan)
        if plan.config.world_size != layout.world_size:
            raise ValueError("M7 plan/layout world-size mismatch")
        if plan.config.max_descriptors != layout.max_descriptors:
            raise ValueError("M7 plan/layout descriptor mismatch")
        if not unique_id:
            raise ValueError("NCCL unique ID must be non-empty")
        self.rank = int(rank)
        self.device = int(device)
        self.plan = plan
        self.layout = layout
        self._unique_id_storage = ctypes.create_string_buffer(bytes(unique_id))
        self._plan_storage = ctypes.create_string_buffer(plan.data)
        self._lib = ctypes.CDLL(str(library))
        self._lib.r6_m7_last_error.restype = ctypes.c_char_p
        self._lib.r6_m7_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ]
        self._lib.r6_m7_create.restype = ctypes.c_void_p
        self._lib.r6_m7_run.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        self._lib.r6_m7_run.restype = ctypes.c_int
        self._lib.r6_m7_counter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_m7_counter.restype = ctypes.c_uint64
        self._lib.r6_m7_capability.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_m7_capability.restype = ctypes.c_uint64
        for name in (
            "commit_count", "peer_plan_count", "shadow_action_count",
            "trace_count", "timing_count", "num_recv_tokens",
            "num_local_experts",
        ):
            function = getattr(self._lib, f"r6_m7_{name}")
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_size_t
        for name in (
            "copy_commits", "copy_peer_plans", "copy_shadow_actions",
            "copy_traces", "copy_timings", "copy_expert_counts",
            "copy_expert_offsets", "copy_recv_x", "copy_recv_metadata",
        ):
            function = getattr(self._lib, f"r6_m7_{name}")
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_int
        self._lib.r6_m7_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._lib.r6_m7_create(
            self.rank, self.device,
            ctypes.cast(self._unique_id_storage, ctypes.c_void_p),
            len(unique_id), ctypes.cast(self._plan_storage, ctypes.c_void_p),
            len(plan.data), layout.world_size, layout.max_descriptors,
            layout.max_assignments_per_peer, layout.feature_width,
            layout.peer_stride, layout.descriptor_stride, layout.region_bytes,
        )
        if not self._handle:
            self._raise("create")
        self.cpu_per_descriptor_packing = 0
        self.cpu_per_descriptor_transport_submission = 0
        self.cpu_per_descriptor_poll = 0
        self.cpu_per_descriptor_cuda_launch = 0

    def _raise(self, operation: str) -> None:
        raw = self._lib.r6_m7_last_error()
        detail = raw.decode(errors="replace") if raw else "unknown error"
        raise RuntimeError(f"R6-M7 {operation} failed: {detail}")

    @staticmethod
    def _pointer(tensor: Any) -> ctypes.c_void_p:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("M7 inputs must be contiguous CUDA tensors")
        return ctypes.c_void_p(tensor.data_ptr())

    def _copy(self, name: str, values: np.ndarray) -> None:
        function = getattr(self._lib, f"r6_m7_copy_{name}")
        capacity = values.size if name == "recv_x" else len(values)
        if function(
            self._handle, values.ctypes.data_as(ctypes.c_void_p), capacity,
        ) != 0:
            self._raise(f"copy {name}")

    def run(
        self, *, reveal_records: Any, x: Any, topk_idx: Any,
        topk_weights: Any, experts_per_rank: int,
        producer_delay_cycles: int, router_stream: int,
    ) -> dict[str, Any]:
        if reveal_records.ndim != 2 or reveal_records.shape[1] != 8:
            raise ValueError("reveal_records must have shape [N, 8]")
        if x.ndim != 2 or x.shape[1] != self.layout.feature_width:
            raise ValueError("x feature width does not match M7 layout")
        if topk_idx.shape != topk_weights.shape or topk_idx.ndim != 2:
            raise ValueError("topk tensors must have identical [tokens, topk] shape")
        if topk_idx.shape[0] != x.shape[0]:
            raise ValueError("x/topk token count mismatch")
        num_tokens, num_topk = map(int, topk_idx.shape)
        num_experts = int(experts_per_rank) * self.layout.world_size
        status = self._lib.r6_m7_run(
            self._handle, self._pointer(reveal_records), reveal_records.shape[0],
            self._pointer(x), self._pointer(topk_idx),
            self._pointer(topk_weights), num_tokens, self.layout.feature_width,
            num_topk, int(experts_per_rank), num_experts,
            int(producer_delay_cycles), int(router_stream),
        )
        if status != 0:
            self._raise("run")
        sizes = {
            name: int(getattr(self._lib, f"r6_m7_{name}")(self._handle))
            for name in (
                "commit_count", "peer_plan_count", "shadow_action_count",
                "trace_count", "timing_count", "num_recv_tokens",
                "num_local_experts",
            )
        }
        arrays = {
            "commits": np.empty((sizes["commit_count"], 10), dtype=np.uint64),
            "peer_plans": np.empty((sizes["peer_plan_count"], 7), dtype=np.uint64),
            "shadow_actions": np.empty((sizes["shadow_action_count"], 12), dtype=np.int64),
            "traces": np.empty((sizes["trace_count"], 11), dtype=np.uint64),
            "timings": np.empty((sizes["timing_count"], 8), dtype=np.uint64),
            "expert_counts": np.empty(sizes["num_local_experts"], dtype=np.uint32),
            "expert_offsets": np.empty(sizes["num_local_experts"] + 1, dtype=np.uint32),
            "recv_x": np.empty((sizes["num_recv_tokens"], self.layout.feature_width), dtype=np.float32),
            "recv_metadata": np.empty(sizes["num_recv_tokens"], dtype=DISPATCH_META_DTYPE),
        }
        for name, values in arrays.items():
            self._copy(name, values)
        return {
            **arrays,
            "handle": {
                "num_recv_tokens": sizes["num_recv_tokens"],
                "num_local_experts": sizes["num_local_experts"],
                "num_topk": num_topk, "generation": 1,
            },
            "counters": self.counters(), "capability": self.capability(),
        }

    def counters(self) -> dict[str, int]:
        return {name: int(self._lib.r6_m7_counter(self._handle, i))
                for i, name in enumerate(COUNTER_NAMES)}

    def capability(self) -> dict[str, int]:
        return {name: int(self._lib.r6_m7_capability(self._handle, i))
                for i, name in enumerate(CAPABILITY_NAMES)}

    def close(self) -> None:
        if self._handle:
            self._lib.r6_m7_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = [
    "CAPABILITY_NAMES", "COUNTER_NAMES", "DISPATCH_META_DTYPE",
    "GPUProgressiveEPRuntime",
]
