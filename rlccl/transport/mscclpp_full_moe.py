"""R6-M2 forward-only full-MoE transport over real MSCCL++ MemoryChannel.

Descriptors are submitted unchanged and may remain outstanding.  This module
does not schedule, merge, split, or reinterpret them.  Return traffic remains
on the frozen reference path so R6-M2 changes only forward communication.
"""

from __future__ import annotations

from types import SimpleNamespace
import time
from typing import Any

import numpy as np
import torch

from .mscclpp_backend import (
    COUNT_HEADER_BYTES,
    MscclppCommittedAdapter,
    RegisteredBufferLayout,
    action_payload,
)
from .mscclpp_native import MscclppNativeRuntime
from .reference_full_moe import FORWARD_META_FIELDS


class MscclppFullMoeForwardTransport:
    def __init__(
        self, *, library: str, rank: int, device: int,
        endpoint: str, comm_stream: torch.cuda.Stream,
        max_descriptors: int, max_tokens_per_peer_descriptor: int,
        feature_width: int, diagnostic_mode: str = "normal",
    ) -> None:
        self.rank = int(rank)
        self.device = int(device)
        self.comm_stream = comm_stream
        self.feature_width = int(feature_width)
        if diagnostic_mode not in ("normal", "dependency_resolved"):
            raise ValueError("unsupported R6-M3 diagnostic mode")
        self.diagnostic_mode = diagnostic_mode
        self.record_bytes = FORWARD_META_FIELDS * 8 + self.feature_width * 4
        self.layout = RegisteredBufferLayout(
            world_size=2, max_descriptors=max_descriptors,
            max_tokens_per_peer_descriptor=max_tokens_per_peer_descriptor,
            record_bytes=self.record_bytes,
        )
        self.buffer = torch.zeros(
            self.layout.capacity_bytes, dtype=torch.uint8,
            device=torch.device("cuda", device),
        )
        self.adapter = MscclppCommittedAdapter(rank=rank, layout=self.layout)
        self.runtime = MscclppNativeRuntime(
            library, rank=rank, device=device, buffer_ptr=self.buffer.data_ptr(),
            buffer_bytes=self.layout.capacity_bytes, endpoint=endpoint,
        )
        self._gpu_origin = torch.cuda.Event(enable_timing=True)
        self._gpu_origin.record(torch.cuda.current_stream(self.device))
        self._gpu_origin.synchronize()
        self._gpu_origin_host_ns = time.monotonic_ns()
        self.descriptors: list[dict[str, Any]] = []
        self._remote_put_event_indices: list[int] = []
        self._wait_event_indices: list[int] = []
        self._closed = False
        self._event_ids: set[int] = set()
        self._comm_sequence = 0

    def _records(self, payload: Any) -> np.ndarray:
        count = int(payload.total_tokens)
        records = np.empty((count, self.record_bytes), dtype=np.uint8)
        records[:, :FORWARD_META_FIELDS * 8] = (
            np.ascontiguousarray(payload.metadata, dtype=np.int64)
            .view(np.uint8).reshape(count, FORWARD_META_FIELDS * 8)
        )
        records[:, FORWARD_META_FIELDS * 8:] = (
            np.ascontiguousarray(payload.features, dtype=np.float32)
            .view(np.uint8).reshape(count, self.feature_width * 4)
        )
        return records

    def submit(self, payload: Any, meta: dict[str, Any]) -> None:
        adapter_issue_ns = time.monotonic_ns()
        descriptor_id = int(meta["descriptor_index"])
        meta["adapter_issue_host_ns"] = adapter_issue_ns
        meta["cpu_byte_pack_start_host_ns"] = time.monotonic_ns()
        records = self._records(payload)
        meta["cpu_byte_pack_done_host_ns"] = time.monotonic_ns()
        packed = SimpleNamespace(
            source_rank=self.rank,
            chunk_ids=tuple(int(value) for value in meta["chunk_ids"]),
            sendcounts_tokens=tuple(int(value) for value in payload.sendcounts_tokens),
            offsets_tokens=tuple(int(value) for value in payload.offsets_tokens),
            total_tokens=int(payload.total_tokens),
        )
        commit_start_ns = time.monotonic_ns()
        actions = self.adapter.commit_descriptor(
            packed, descriptor_id=descriptor_id,
            guard_decision=meta.pop("_guard_decision"),
            completed_chunks=meta.pop("_completed_chunks"),
            revealed_chunks=meta.pop("_revealed_chunks"),
        )
        commit_done_ns = time.monotonic_ns()
        meta["transport_commit_start_host_ns"] = commit_start_ns
        meta["committed_action_created_host_ns"] = commit_done_ns
        meta.pop("_count_ticket", None)
        staging_start = torch.cuda.Event(enable_timing=True)
        staging_end = torch.cuda.Event(enable_timing=True)
        producer_stream = torch.cuda.current_stream(self.device)
        meta["registered_buffer_staging_host_ns"] = time.monotonic_ns()
        staging_start.record(producer_stream)
        meta["backend"] = "mscclpp"
        meta["committed_actions"] = [action_payload(value) for value in actions]
        host_records = torch.from_numpy(records.reshape(-1))
        for destination, count in enumerate(payload.sendcounts_tokens):
            count = int(count)
            base = self.layout.send_offset(descriptor_id, destination)
            header = torch.tensor([count], dtype=torch.int64).view(torch.uint8)
            self.buffer[base:base + COUNT_HEADER_BYTES].copy_(header.to(self.buffer.device))
            if count:
                left = int(payload.offsets_tokens[destination]) * self.record_bytes
                right = left + count * self.record_bytes
                self.buffer[base + COUNT_HEADER_BYTES:base + COUNT_HEADER_BYTES + right - left].copy_(
                    host_records[left:right].to(self.buffer.device)
                )
        staging_end.record(producer_stream)
        event_id = id(staging_end)
        event_reused = event_id in self._event_ids
        self._event_ids.add(event_id)
        meta["pack_event_id"] = event_id
        meta["pack_event_producer_stream"] = int(producer_stream.cuda_stream)
        meta["pack_event_record_host_ns"] = time.monotonic_ns()
        meta["pack_event_reused"] = event_reused
        meta["dependency_resolved_sync"] = self.diagnostic_mode == "dependency_resolved"
        if self.diagnostic_mode == "dependency_resolved":
            staging_end.synchronize()
            meta["dependency_resolved_host_ns"] = time.monotonic_ns()
        else:
            meta["dependency_resolved_host_ns"] = 0
        self.comm_stream.wait_event(staging_end)
        meta["comm_wait_event_enqueue_host_ns"] = time.monotonic_ns()
        meta["comm_wait_event_id"] = event_id
        meta["comm_wait_event_stream"] = int(self.comm_stream.cuda_stream)
        issue_host_ns = time.monotonic_ns()
        meta["issue_time_host_ns"] = issue_host_ns
        remote_event_index = -1
        preceding = "origin_event" if self._comm_sequence == 0 else "previous_put_or_self_copy"
        for action in actions:
            if action.is_remote:
                remote_event_index = self.runtime.counters()["mscclpp_put_calls"]
                self.runtime.issue(
                    dst_offset=action.dst_offset, src_offset=action.src_offset,
                    bytes=action.physical_bytes,
                    stream=self.comm_stream.cuda_stream,
                )
                self._remote_put_event_indices.append(remote_event_index)
                self._comm_sequence += 1
            else:
                with torch.cuda.stream(self.comm_stream):
                    source = action.src_offset
                    target = action.dst_offset
                    size = action.physical_bytes
                    self.buffer[target:target + size].copy_(self.buffer[source:source + size])
        meta["put_launch_host_ns"] = issue_host_ns if remote_event_index >= 0 else 0
        meta["comm_stream_sequence_number"] = self._comm_sequence
        meta["preceding_operation"] = preceding
        meta["preceding_dependency"] = f"pack_event:{event_id}"
        meta["_staging_start_event"] = staging_start
        meta["_staging_end_event"] = staging_end
        meta["_put_event_index"] = remote_event_index
        self.descriptors.append(meta)

    def finish(self):
        # Every descriptor sends one remote signal, including an 8-byte zero
        # count header.  Queue all waits only at the forward boundary, allowing
        # all previously revealed descriptors to remain outstanding.
        for _ in self.descriptors:
            index = self.runtime.counters()["mscclpp_waits"]
            self.runtime.wait(stream=self.comm_stream.cuda_stream)
            self._wait_event_indices.append(index)
        self.runtime.synchronize(stream=self.comm_stream.cuda_stream)
        completed = []
        for descriptor_id, meta in enumerate(self.descriptors):
            rows_meta: list[np.ndarray] = []
            rows_features: list[np.ndarray] = []
            recvcounts = []
            wait_timing = self.runtime.event_timing(
                kind="wait", index=self._wait_event_indices[descriptor_id],
            )
            put_index = int(meta.pop("_put_event_index"))
            put_timing = (
                self.runtime.event_timing(kind="put", index=put_index)
                if put_index >= 0 else {
                    "gpu_start_us": 0.0, "gpu_end_us": 0.0,
                    "gpu_start_host_ns": 0, "gpu_end_host_ns": 0,
                    "gpu_duration_us": 0.0,
                }
            )
            put_host_timing = (
                self.runtime.issue_host_timing(index=put_index)
                if put_index >= 0 else {
                    "native_wrapper_enter_host_ns": 0,
                    "kernel_launch_call_host_ns": 0,
                    "kernel_launch_return_host_ns": 0,
                }
            )
            staging_start_event = meta.pop("_staging_start_event")
            staging_end_event = meta.pop("_staging_end_event")
            staging_start_us = float(self._gpu_origin.elapsed_time(staging_start_event) * 1e3)
            staging_end_us = float(self._gpu_origin.elapsed_time(staging_end_event) * 1e3)
            for source_rank in range(2):
                base = self.layout.receive_offset(descriptor_id, source_rank)
                count = int(self.buffer[base:base + 8].view(torch.int64).item())
                if not 0 <= count <= self.layout.max_tokens_per_peer_descriptor:
                    raise RuntimeError(f"MSCCL++ receive count outside slot: {count}")
                recvcounts.append(count)
                if not count:
                    continue
                raw = (
                    self.buffer[
                        base + COUNT_HEADER_BYTES:
                        base + COUNT_HEADER_BYTES + count * self.record_bytes
                    ].cpu().numpy().reshape(count, self.record_bytes)
                )
                rows_meta.append(
                    raw[:, :FORWARD_META_FIELDS * 8].copy()
                    .reshape(-1).view(np.int64).reshape(count, FORWARD_META_FIELDS)
                )
                rows_features.append(
                    raw[:, FORWARD_META_FIELDS * 8:].copy()
                    .reshape(-1).view(np.float32).reshape(count, self.feature_width)
                )
            metadata = (
                np.concatenate(rows_meta, axis=0)
                if rows_meta else np.empty((0, FORWARD_META_FIELDS), dtype=np.int64)
            )
            features = (
                np.concatenate(rows_features, axis=0)
                if rows_features else np.empty((0, self.feature_width), dtype=np.float32)
            )
            timing = {
                "backend": "mscclpp",
                **put_host_timing,
                "staging_gpu_start_host_ns": self._gpu_origin_host_ns + int(staging_start_us * 1e3),
                "staging_gpu_end_host_ns": self._gpu_origin_host_ns + int(staging_end_us * 1e3),
                "staging_gpu_duration_us": staging_end_us - staging_start_us,
                "payload_call_host_ns": int(meta["put_launch_host_ns"]),
                "payload_complete_host_ns": int(wait_timing["gpu_end_host_ns"]),
                "put_kernel_start_host_ns": int(put_timing["gpu_start_host_ns"]),
                "put_kernel_end_host_ns": int(put_timing["gpu_end_host_ns"]),
                "put_kernel_duration_us": float(put_timing["gpu_duration_us"]),
                "signal_host_ns": int(put_timing["gpu_end_host_ns"]),
                "remote_wait_complete_host_ns": int(wait_timing["gpu_end_host_ns"]),
                "wait_gpu_duration_us": float(wait_timing["gpu_duration_us"]),
                "a2av_completion_us": (
                    int(wait_timing["gpu_end_host_ns"]) - int(meta["put_launch_host_ns"])
                ) / 1e3,
                "h2d_us": 0.0, "count_exchange_us": 0.0,
                "count_wait_us": 0.0, "count_gpu_us": 0.0,
                "a2av_submit_us": 0.0, "d2h_us": 0.0,
                "count_h2d_overlap": False,
                "metadata_host_pinned": True, "values_host_pinned": True,
            }
            completed.append((metadata, features, tuple(recvcounts), timing))
        return completed

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "mscclpp", "channel": "MemoryChannel",
            "transport": "CudaIpc", "multiple_outstanding": True,
            "descriptor_count": len(self.descriptors),
            "diagnostic_mode": self.diagnostic_mode,
            **self.runtime.counters(), **self.adapter.counters(),
        }

    def close(self) -> None:
        if not self._closed:
            self.runtime.close()
            self._closed = True
