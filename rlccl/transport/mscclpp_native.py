"""ctypes owner for the R6-M1 MSCCL++ CUDA primitive runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path


class MscclppNativeRuntime:
    def __init__(
        self, library: str | Path, *, rank: int, device: int,
        buffer_ptr: int, buffer_bytes: int, endpoint: str,
    ) -> None:
        self._lib = ctypes.CDLL(str(library))
        self._lib.r6_mscclpp_last_error.restype = ctypes.c_char_p
        self._lib.r6_mscclpp_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
        ]
        self._lib.r6_mscclpp_create.restype = ctypes.c_void_p
        self._lib.r6_mscclpp_issue.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_uint64, ctypes.c_size_t,
        ]
        self._lib.r6_mscclpp_wait.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_mscclpp_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.r6_mscclpp_counter.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.r6_mscclpp_counter.restype = ctypes.c_uint64
        self._lib.r6_mscclpp_origin_host_ns.argtypes = [ctypes.c_void_p]
        self._lib.r6_mscclpp_origin_host_ns.restype = ctypes.c_uint64
        self._lib.r6_mscclpp_event_timing.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.r6_mscclpp_issue_host_timing.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.r6_mscclpp_destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._lib.r6_mscclpp_create(
            rank, device, ctypes.c_void_p(buffer_ptr), buffer_bytes, endpoint.encode(),
        )
        if not self._handle:
            self._raise("create")

    def _raise(self, operation: str) -> None:
        raw = self._lib.r6_mscclpp_last_error()
        detail = raw.decode(errors="replace") if raw else "unknown error"
        raise RuntimeError(f"MSCCL++ {operation} failed: {detail}")

    def issue(self, *, dst_offset: int, src_offset: int, bytes: int, stream: int) -> None:
        if self._lib.r6_mscclpp_issue(
            self._handle, dst_offset, src_offset, bytes, stream,
        ) != 0:
            self._raise("put/signal")

    def wait(self, *, stream: int) -> None:
        if self._lib.r6_mscclpp_wait(self._handle, stream) != 0:
            self._raise("wait")

    def synchronize(self, *, stream: int) -> None:
        if self._lib.r6_mscclpp_synchronize(self._handle, stream) != 0:
            self._raise("synchronize")

    def counters(self) -> dict[str, int]:
        return {
            name: int(self._lib.r6_mscclpp_counter(self._handle, index))
            for index, name in enumerate((
                "mscclpp_put_calls", "mscclpp_bytes_transferred",
                "mscclpp_signals", "mscclpp_waits",
            ))
        }

    @property
    def origin_host_ns(self) -> int:
        return int(self._lib.r6_mscclpp_origin_host_ns(self._handle))

    def event_timing(self, *, kind: str, index: int) -> dict[str, float | int]:
        if kind not in ("put", "wait"):
            raise ValueError("event kind must be put or wait")
        start, end = ctypes.c_float(), ctypes.c_float()
        if self._lib.r6_mscclpp_event_timing(
            self._handle, 0 if kind == "put" else 1, index,
            ctypes.byref(start), ctypes.byref(end),
        ) != 0:
            self._raise("event timing")
        return {
            "gpu_start_us": float(start.value),
            "gpu_end_us": float(end.value),
            "gpu_start_host_ns": self.origin_host_ns + int(start.value * 1000.0),
            "gpu_end_host_ns": self.origin_host_ns + int(end.value * 1000.0),
            "gpu_duration_us": float(end.value - start.value),
        }

    def issue_host_timing(self, *, index: int) -> dict[str, int]:
        wrapper, launch_call, launch_return = (
            ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
        )
        if self._lib.r6_mscclpp_issue_host_timing(
            self._handle, index, ctypes.byref(wrapper),
            ctypes.byref(launch_call), ctypes.byref(launch_return),
        ) != 0:
            self._raise("host issue timing")
        return {
            "native_wrapper_enter_host_ns": int(wrapper.value),
            "kernel_launch_call_host_ns": int(launch_call.value),
            "kernel_launch_return_host_ns": int(launch_return.value),
        }

    def close(self) -> None:
        if self._handle:
            self._lib.r6_mscclpp_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
