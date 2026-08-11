"""R4-A0/C0 reference full-MoE correctness over real bidirectional A2Av."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import (  # noqa: E402
    DynamicGuard, FastBinder, IncrementalState, StaticPlanCompiler, structural_signature,
)
from rlccl.transport.reference_a2av import (  # noqa: E402
    ProgressivePackingState, payload_multiset_digest,
)
from rlccl.transport.fast_progressive_data_prep import FastProgressiveDataPrep  # noqa: E402
from rlccl.transport.reference_full_moe import (  # noqa: E402
    FORWARD_META_FIELDS, RETURN_META_FIELDS, pack_forward_payload,
    pack_return_payload, reference_expert_mlp, seed_reference_experts,
    verify_forward_payload, verify_return_and_combine,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from scripts.run_r2_f0_integrated import (  # noqa: E402
    CHUNKS, D, EXPERTS, PARTIAL_CHUNKS, TOP_K, WAIT_TIMEOUT_NS,
    _action_signature, _load_bridge_extension,
)
from scripts.run_r3_a0_c0 import (  # noqa: E402
    CASE_ORDER, CONTROL_LIMIT_PER_CHUNK, TOTAL_TOKENS, _case_inputs,
    _control_tokens, _make_assignments, _oracle_replay_variable,
    distribution, sha256_file,
)


EXPERT_HIDDEN = 32
EXPERT_OUTPUT = 16
EXPERT_SEED = 20260812
ROUTER_SEED = 20260805
TRIGGERS = (0, 1, 2, 3, 4, 5, 7)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _reference_expert_mlp_with_events(
    features: torch.Tensor, expert_ids: np.ndarray,
    weights: tuple[Any, Any, Any, Any], expert_stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, tuple[int, ...], list[dict[str, Any]], torch.cuda.Event, torch.cuda.Event]:
    """R4 reference MLP with symmetric per-full-batch CUDA instrumentation.

    This is an opt-in R5-P2 substrate.  It preserves reference_expert_mlp's
    expert order, membership, GEMM shapes/count and index-copy layout.
    """
    w1, b1, w2, b2 = weights
    output = torch.empty(
        (features.shape[0], w2.shape[2]), dtype=torch.float32, device=features.device,
    )
    counts: list[int] = []
    tasks: list[dict[str, Any]] = []
    overall_start = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)
    expert_ids_array = np.asarray(expert_ids)
    with torch.inference_mode(), torch.cuda.stream(expert_stream):
        overall_start.record(expert_stream)
        for expert in range(w1.shape[0]):
            indices_np = np.flatnonzero(expert_ids_array == expert)
            counts.append(int(indices_np.size))
            if not indices_np.size:
                continue
            indices = torch.from_numpy(indices_np.astype(np.int64)).to(features.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(expert_stream)
            batch = features.index_select(0, indices)
            hidden = torch.relu(batch @ w1[expert].to(features.device) + b1[expert].to(features.device))
            values = hidden @ w2[expert].to(features.device) + b2[expert].to(features.device)
            output.index_copy_(0, indices, values)
            end_event.record(expert_stream)
            tasks.append({
                "expert": int(expert), "indices": indices_np.astype(np.int64),
                "batch_size": int(indices_np.size), "start_event": start_event,
                "end_event": end_event,
            })
        overall_end.record(expert_stream)
    return output, tuple(counts), tasks, overall_start, overall_end


def _start_count_exchange(
    *, sendcounts: Sequence[int], device: torch.device,
    count_stream: torch.cuda.Stream,
    trace_label: str | None = None,
) -> dict[str, Any]:
    host_counts = torch.tensor(
        tuple(int(value) for value in sendcounts), dtype=torch.int64, pin_memory=True,
    )
    recv_counts_device = torch.empty(dist.get_world_size(), dtype=torch.int64, device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started_ns = time.perf_counter_ns()
    annotation = record_function(trace_label) if trace_label is not None else None
    if annotation is not None:
        annotation.__enter__()
    with torch.cuda.stream(count_stream):
        start_event.record(count_stream)
        send_counts = host_counts.to(device, non_blocking=True)
        work = dist.all_to_all_single(recv_counts_device, send_counts, async_op=True)
        end_event.record(count_stream)
    return {
        "host_counts": host_counts, "send_counts": send_counts,
        "recv_counts_device": recv_counts_device, "work": work,
        "count_stream": count_stream, "start_event": start_event,
        "end_event": end_event, "started_ns": started_ns,
        "annotation": annotation,
    }


def _exchange_pair(
    *, metadata: np.ndarray, values: np.ndarray, sendcounts: Sequence[int],
    meta_width: int, value_width: int, device: torch.device,
    comm_stream: torch.cuda.Stream, stream_scoped_sync: bool = False,
    count_stream: torch.cuda.Stream | None = None,
    overlap_count_with_h2d: bool = False,
    prestarted_count_ticket: dict[str, Any] | None = None,
    trace_label_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], dict[str, float | bool]]:
    h2d_start = time.perf_counter_ns()
    host_meta = torch.from_numpy(np.ascontiguousarray(metadata.reshape(-1)))
    host_values = torch.from_numpy(np.ascontiguousarray(values.reshape(-1)))
    if overlap_count_with_h2d:
        if count_stream is None:
            raise ValueError("count/H2D overlap requires an independent count stream")
        ticket = prestarted_count_ticket or _start_count_exchange(
            sendcounts=sendcounts, device=device, count_stream=count_stream,
            trace_label=(f"R5P4|kind=count|{trace_label_prefix}" if trace_label_prefix else None),
        )
        count_start = int(ticket["started_ns"])
        recv_counts_device = ticket["recv_counts_device"]
        count_work = ticket["work"]
        count_start_event = ticket["start_event"]
        count_end_event = ticket["end_event"]
        with torch.cuda.stream(comm_stream):
            send_meta = host_meta.to(device, non_blocking=host_meta.is_pinned())
            send_values = host_values.to(device, non_blocking=host_values.is_pinned())
        h2d_done = time.perf_counter_ns()
    else:
        send_meta = host_meta.to(device)
        send_values = host_values.to(device)
        send_counts = torch.tensor(tuple(int(value) for value in sendcounts), dtype=torch.int64, device=device)
        recv_counts_device = torch.empty(dist.get_world_size(), dtype=torch.int64, device=device)
        h2d_done = time.perf_counter_ns()
        count_start = time.perf_counter_ns()
        with torch.cuda.stream(comm_stream):
            count_work = dist.all_to_all_single(recv_counts_device, send_counts, async_op=True)
        count_start_event = None
        count_end_event = None
    count_wait_start = time.perf_counter_ns()
    count_work.wait()
    if overlap_count_with_h2d:
        assert count_stream is not None
        count_stream.synchronize()
    recvcounts = tuple(int(value) for value in recv_counts_device.cpu().tolist())
    count_done = time.perf_counter_ns()
    count_annotation = ticket.get("annotation") if overlap_count_with_h2d else None
    if count_annotation is not None:
        count_annotation.__exit__(None, None, None)
    recv_meta = torch.empty(sum(recvcounts) * meta_width, dtype=torch.int64, device=device)
    recv_values = torch.empty(sum(recvcounts) * value_width, dtype=torch.float32, device=device)
    meta_start = time.perf_counter_ns()
    payload_context = (
        record_function(f"R5P4|kind=payload|{trace_label_prefix}")
        if trace_label_prefix else nullcontext()
    )
    with payload_context:
        with torch.cuda.stream(comm_stream):
            meta_work = dist.all_to_all_single(
                recv_meta, send_meta,
                output_split_sizes=[value * meta_width for value in recvcounts],
                input_split_sizes=[int(value) * meta_width for value in sendcounts],
                async_op=True,
            )
            value_work = dist.all_to_all_single(
                recv_values, send_values,
                output_split_sizes=[value * value_width for value in recvcounts],
                input_split_sizes=[int(value) * value_width for value in sendcounts],
                async_op=True,
            )
        submit_done = time.perf_counter_ns()
        meta_work.wait(); value_work.wait()
        if stream_scoped_sync:
            comm_stream.synchronize()
        else:
            torch.cuda.synchronize(device)
        payload_done = time.perf_counter_ns()
    # R5 progressive expert execution needs communication completion without
    # draining an independent expert stream.  This is opt-in so historical R4
    # behavior and artifacts remain unchanged.
    received_meta = recv_meta.cpu().numpy().reshape((-1, meta_width)).copy()
    received_values = recv_values.cpu().numpy().reshape((-1, value_width)).copy()
    d2h_done = time.perf_counter_ns()
    return received_meta, received_values, recvcounts, {
        "h2d_us": (h2d_done - h2d_start) / 1e3,
        "count_exchange_us": (count_done - count_start) / 1e3,
        "count_wait_us": (count_done - count_wait_start) / 1e3,
        "count_gpu_us": (
            float(count_start_event.elapsed_time(count_end_event) * 1e3)
            if count_start_event is not None and count_end_event is not None else 0.0
        ),
        "count_prestarted_before_packing": bool(prestarted_count_ticket is not None),
        "count_start_host_ns": int(count_start),
        "count_visible_host_ns": int(count_done),
        "payload_call_host_ns": int(meta_start),
        "payload_submit_return_host_ns": int(submit_done),
        "payload_complete_host_ns": int(payload_done),
        "a2av_submit_us": (submit_done - meta_start) / 1e3,
        "a2av_completion_us": (payload_done - meta_start) / 1e3,
        "d2h_us": (d2h_done - payload_done) / 1e3,
        "count_h2d_overlap": bool(overlap_count_with_h2d),
        "metadata_host_pinned": bool(host_meta.is_pinned()),
        "values_host_pinned": bool(host_values.is_pinned()),
    }


def _run_arm(
    *, arm: str, case_name: str, rank: int, topology: Any, plan: Any, bridge: Any,
    case_data: dict[str, Any], tokens_device: torch.Tensor,
    token_chunks: Sequence[torch.Tensor], mask_chunks: Sequence[torch.Tensor],
    router_weight: torch.Tensor, router_bias: torch.Tensor,
    expert_weights: tuple[Any, Any, Any, Any], router_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream, split_primary_timing: bool = False,
    progressive_expert: bool = False, expert_batch_threshold: int = 0,
    expert_stream: torch.cuda.Stream | None = None,
    stream_scoped_sync: bool = False, retain_final_output: bool = False,
    progressive_return: bool = False, instrument_full_expert_batches: bool = False,
    fast_data_prep: bool = False, count_stream: torch.cuda.Stream | None = None,
    overlap_count_with_h2d: bool = False,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    delayed = arm in ("D", "D1")
    if progressive_expert and delayed:
        raise ValueError("progressive expert requires progressive forward dispatch")
    if progressive_expert and (expert_stream is None or expert_batch_threshold <= 0):
        raise ValueError("progressive expert requires a stream and positive threshold")
    if progressive_return and progressive_expert:
        raise ValueError("R5-P2 progressive return forbids progressive expert execution")
    if progressive_return and not instrument_full_expert_batches:
        raise ValueError("progressive return requires symmetric full-batch instrumentation")
    if instrument_full_expert_batches and expert_stream is None:
        raise ValueError("full expert batch instrumentation requires an expert stream")
    if overlap_count_with_h2d and (not fast_data_prep or count_stream is None):
        raise ValueError("count/H2D overlap is only valid for fast data prep with a count stream")
    trial_id = f"r4-a0-{case_name}-{arm}-rank{rank}"
    total_controls = sum(min(value, CONTROL_LIMIT_PER_CHUNK) for value in case_data["chunk_sizes"])
    state = IncrementalState(
        plan, max_tokens=max(1, total_controls), max_chunks=CHUNKS,
        sequence_id=trial_id, sequence_step=8,
    )
    binder, guard = FastBinder(plan), DynamicGuard(plan)
    if fast_data_prep:
        packing = FastProgressiveDataPrep(
            world_size=2, source_rank=rank, tokens=case_data["tokens"],
            token_ids=case_data["token_ids"], chunk_offsets=case_data["chunk_offsets"],
            descriptor_groups=tuple((chunk,) for chunk in range(PARTIAL_CHUNKS)) + ((6, 7),),
        )
    else:
        packing = ProgressivePackingState(world_size=2, source_rank=rank, max_chunks=CHUNKS)
    bridge.reset_all()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    host_indices = [torch.empty(size, dtype=torch.int64, pin_memory=True) for size in case_data["chunk_sizes"]]
    host_numpy = [value.numpy() for value in host_indices]
    ready_ns = [0] * CHUNKS
    launch_ns = [0] * CHUNKS
    producer_error: list[BaseException | None] = [None]
    assignments_by_chunk: list[Any] = [None] * CHUNKS
    controls_by_chunk: list[Any] = [None] * CHUNKS
    bounds: list[Any] = []
    decisions: list[Any] = []
    pending: list[tuple[Any, dict[str, Any]]] = []
    forward_descriptors: list[dict[str, Any]] = []
    received_forward: list[dict[str, Any]] = []
    expert_buffers: dict[int, list[tuple[int, np.ndarray]]] = {
        expert: [] for expert in range(EXPERTS)
    }
    expert_tasks: list[dict[str, Any]] = []
    progressive_executed_indices: list[int] = []
    first_expert_launch_host_ns = 0
    forward_received_cursor = 0
    features_by_token = None if fast_data_prep else {
        int(token_id): case_data["tokens"][position]
        for position, token_id in enumerate(case_data["token_ids"])
    }
    position_by_token = None if fast_data_prep else {
        int(token_id): position for position, token_id in enumerate(case_data["token_ids"])
    }

    gpu_origin = torch.cuda.Event(enable_timing=True)
    torch.cuda.current_stream(tokens_device.device).record_event(gpu_origin)
    gpu_origin.synchronize()
    router_stream.wait_event(gpu_origin)
    comm_stream.wait_event(gpu_origin)
    if expert_stream is not None:
        expert_stream.wait_event(gpu_origin)

    def launch_expert_batch(expert: int, items: list[tuple[int, np.ndarray]], *, flush: bool) -> None:
        nonlocal first_expert_launch_host_ns
        if not items:
            return
        assert expert_stream is not None
        indices = np.asarray([item[0] for item in items], dtype=np.int64)
        features = np.ascontiguousarray(np.stack([item[1] for item in items]).astype(np.float32))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        launch_ns = time.monotonic_ns()
        if first_expert_launch_host_ns == 0:
            first_expert_launch_host_ns = launch_ns
        w1, b1, w2, b2 = expert_weights
        with torch.inference_mode(), torch.cuda.stream(expert_stream):
            batch = torch.from_numpy(features).to(tokens_device.device)
            start_event.record(expert_stream)
            hidden = torch.relu(batch @ w1[expert] + b1[expert])
            output = hidden @ w2[expert] + b2[expert]
            end_event.record(expert_stream)
        expert_tasks.append({
            "expert": expert, "indices": indices, "output": output,
            "batch_size": len(items), "flush": flush, "launch_host_ns": launch_ns,
            "start_event": start_event, "end_event": end_event,
        })
        progressive_executed_indices.extend(int(value) for value in indices)

    def drain_ready_experts(*, flush: bool) -> None:
        if not progressive_expert:
            return
        for expert in range(EXPERTS):
            buffer = expert_buffers[expert]
            while len(buffer) >= expert_batch_threshold or (flush and buffer):
                take = len(buffer) if flush and len(buffer) < expert_batch_threshold else expert_batch_threshold
                items = buffer[:take]
                del buffer[:take]
                launch_expert_batch(expert, items, flush=flush and not buffer)

    def producer() -> None:
        try:
            torch.cuda.set_device(rank)
            with torch.inference_mode():
                for chunk in range(CHUNKS):
                    launch_ns[chunk] = time.monotonic_ns()
                    router_context = (
                        record_function(
                            "R5P4|kind=router|"
                            f"seed={trace_context['seed']}|family={trace_context['family_index']}|"
                            f"job={trace_context['job']}|arm={arm}|item={chunk}"
                        ) if trace_context is not None else nullcontext()
                    )
                    with router_context:
                        with torch.cuda.stream(router_stream):
                            starts[chunk].record(router_stream)
                            indices, _ = router_topk(
                                token_chunks[chunk], router_weight, router_bias, TOP_K,
                                mask=mask_chunks[chunk],
                            )
                            host_indices[chunk].copy_(indices, non_blocking=True)
                            events[chunk].record(router_stream)
                    bridge.arm(chunk, events[chunk].cuda_event)
        except BaseException as error:
            producer_error[0] = error

    router_start_ns = time.monotonic_ns()
    thread = threading.Thread(target=producer, name=f"r4-router-{arm}-rank{rank}")
    thread.start()

    def consume(chunk: int, reveal: bool) -> None:
        ready_ns[chunk] = int(bridge.wait_ready(chunk, WAIT_TIMEOUT_NS))
        assignments = _make_assignments(case_data=case_data, rank=rank, chunk=chunk, experts=host_numpy[chunk])
        controls = _control_tokens(
            case_name=f"r4-{case_name}", rank=rank, chunk=chunk,
            assignments=assignments, case_data=case_data,
        )
        assignments_by_chunk[chunk], controls_by_chunk[chunk] = assignments, controls
        if fast_data_prep:
            packing.mark_completed(chunk, assignments, experts=host_numpy[chunk])
        else:
            packing.mark_completed(chunk, assignments)
        state.stage_ready_chunk(chunk, controls)
        if reveal:
            packing.reveal(chunk); state.consume_pending_chunk(chunk)

    def schedule() -> dict[str, float]:
        start = time.perf_counter_ns(); bound = binder.step(state); action = time.perf_counter_ns()
        decision = guard.apply(
            state, bound.proposal, require_scheduler_semantics=True,
            expected_state_version=bound.state_version,
        )
        done = time.perf_counter_ns()
        if not decision.accepted:
            raise RuntimeError("compiled checker failed closed")
        bounds.append(bound); decisions.append(decision)
        return {"aiccl_action_us": (action - start) / 1e3, "aiccl_control_us": (done - start) / 1e3}

    def build(chunk_ids: tuple[int, ...], trigger: int, control: dict[str, float]) -> tuple[Any, dict[str, Any]]:
        timing: dict[str, float] = {}
        descriptor_index = packing.descriptor_count
        trace_prefix = None
        if trace_context is not None:
            trace_prefix = (
                f"seed={trace_context['seed']}|family={trace_context['family_index']}|"
                f"job={trace_context['job']}|arm={arm}|item={descriptor_index}"
            )
        if fast_data_prep:
            count_launcher = None
            if overlap_count_with_h2d and not delayed:
                assert count_stream is not None
                count_launcher = lambda counts: _start_count_exchange(
                    sendcounts=counts, device=tokens_device.device, count_stream=count_stream,
                    trace_label=(f"R5P4|kind=count|{trace_prefix}" if trace_prefix else None),
                )
            payload = packing.build_delta_payload(
                chunk_ids, timing_sink=timing, count_exchange_launcher=count_launcher,
            )
            count_ticket = packing.take_count_ticket()
        else:
            layout = packing.build_delta_layout(chunk_ids, timing_sink=timing)
            pack_start = time.perf_counter_ns()
            assert features_by_token is not None and position_by_token is not None
            payload = pack_forward_payload(
                layout, features_by_token=features_by_token,
                original_position_by_token=position_by_token,
            )
            pack_done = time.perf_counter_ns()
            timing["packing_us"] = (pack_done - pack_start) / 1e3
            count_ticket = None
        meta = {
            "descriptor_index": packing.descriptor_count - 1, "trigger": trigger,
            "chunk_ids": list(chunk_ids), "sendcounts_tokens": list(payload.sendcounts_tokens),
            "offsets_tokens": list(payload.offsets_tokens), "tokens": payload.total_tokens,
            "metadata_digest": _digest(payload.metadata), "feature_digest": _digest(payload.features),
            "fast_data_prep": bool(fast_data_prep), **timing, **control,
        }
        if count_ticket is not None:
            meta["_count_ticket"] = count_ticket
        meta["descriptor_ready_host_ns"] = int(time.perf_counter_ns())
        if trace_prefix is not None:
            meta["trace_label_prefix"] = trace_prefix
        return payload, meta

    def communicate(payload: Any, meta: dict[str, Any]) -> None:
        nonlocal forward_received_cursor
        meta["communicate_enter_host_ns"] = int(time.perf_counter_ns())
        count_ticket = meta.pop("_count_ticket", None)
        recv_meta, recv_features, recvcounts, timing = _exchange_pair(
            metadata=payload.metadata, values=payload.features,
            sendcounts=payload.sendcounts_tokens, meta_width=FORWARD_META_FIELDS,
            value_width=D, device=tokens_device.device, comm_stream=comm_stream,
            stream_scoped_sync=stream_scoped_sync,
            count_stream=count_stream, overlap_count_with_h2d=overlap_count_with_h2d,
            prestarted_count_ticket=count_ticket,
            trace_label_prefix=meta.get("trace_label_prefix"),
        )
        verified = verify_forward_payload(
            recv_meta, recv_features, destination_rank=rank, world_size=2,
            recvcounts_tokens=recvcounts,
        )
        if not verified["pass"]:
            raise RuntimeError(f"forward payload verification failed: {verified}")
        meta.update({"recvcounts_tokens": list(recvcounts), "communication": timing, "verification": verified})
        forward_descriptors.append(meta)
        received_forward.append({"metadata": recv_meta, "features": recv_features, "recvcounts": recvcounts})
        if progressive_expert:
            for offset, (row, feature) in enumerate(zip(recv_meta, recv_features, strict=True)):
                expert_buffers[int(row[3])].append(
                    (forward_received_cursor + offset, np.asarray(feature, dtype=np.float32).copy())
                )
            forward_received_cursor += int(recv_meta.shape[0])
            drain_ready_experts(flush=False)

    for chunk in range(PARTIAL_CHUNKS):
        consume(chunk, True); state.stage, state.ratio = chunk + 1, (chunk + 1) / CHUNKS
        control = schedule(); payload, meta = build((chunk,), chunk, control)
        if delayed: pending.append((payload, meta))
        else: communicate(payload, meta)
    consume(6, False); consume(7, False)
    final_router_ns = int(ready_ns[7])
    for chunk in (6, 7):
        packing.reveal(chunk); state.consume_pending_chunk(chunk)
    state.stage, state.ratio = CHUNKS, 1.0
    control = schedule(); payload, meta = build((6, 7), 7, control)
    if delayed:
        pending.append((payload, meta))
        for value, value_meta in pending: communicate(value, value_meta)
    else:
        communicate(payload, meta)
    thread.join(timeout=120)
    if thread.is_alive() or producer_error[0] is not None:
        raise RuntimeError("router producer failed") from producer_error[0]

    # Non-progressive expert boundary: all seven forward descriptors are complete.
    forward_done_event = torch.cuda.Event(enable_timing=True)
    forward_done_event.record(comm_stream)
    forward_done_event.synchronize()
    forward_done_host_ns = time.monotonic_ns()
    expert_start = time.perf_counter_ns()
    all_forward_meta = np.concatenate([value["metadata"] for value in received_forward], axis=0)
    all_forward_features = np.concatenate([value["features"] for value in received_forward], axis=0)
    baseline_expert_start_event: torch.cuda.Event | None = None
    baseline_expert_end_event: torch.cuda.Event | None = None
    full_batch_tasks: list[dict[str, Any]] = []
    return_descriptors: list[dict[str, Any]] = []
    returned_meta_arrays: list[np.ndarray] = []
    returned_output_arrays: list[np.ndarray] = []
    returned_counts: list[tuple[int, ...]] = []
    return_start_host_ns = 0
    return_done_host_ns = 0
    return_error: list[BaseException | None] = [None]

    def execute_returns(
        output_source: np.ndarray | torch.Tensor,
        completion_by_expert: dict[int, torch.cuda.Event] | None,
    ) -> None:
        nonlocal return_start_host_ns, return_done_host_ns
        try:
            torch.cuda.set_device(rank)
            return_start_host_ns = time.monotonic_ns()
            cursor = 0
            for descriptor_index, forward in enumerate(received_forward):
                count = int(forward["metadata"].shape[0])
                descriptor_meta = all_forward_meta[cursor:cursor + count]
                dependencies = sorted({int(value) for value in descriptor_meta[:, 3]})
                wait_start_ns = time.monotonic_ns()
                if completion_by_expert is not None:
                    dependency_events = [completion_by_expert[value] for value in dependencies]
                    while not all(value.query() for value in dependency_events):
                        pass
                dependency_ready_ns = time.monotonic_ns()
                output_view = output_source[cursor:cursor + count]
                if isinstance(output_view, torch.Tensor):
                    output_slice = output_view.detach().cpu().numpy().copy()
                else:
                    output_slice = np.ascontiguousarray(output_view)
                cursor += count
                return_pack_start = time.perf_counter_ns()
                returned = pack_return_payload(
                    forward["metadata"], output_slice, expert_rank=rank, world_size=2,
                )
                return_pack_done = time.perf_counter_ns()
                gpu_start_event = torch.cuda.Event(enable_timing=True)
                gpu_end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(comm_stream):
                    gpu_start_event.record(comm_stream)
                submit_call_host_ns = time.monotonic_ns()
                recv_meta, recv_output, recvcounts, timing = _exchange_pair(
                    metadata=returned.metadata, values=returned.outputs,
                    sendcounts=returned.sendcounts_tokens, meta_width=RETURN_META_FIELDS,
                    value_width=EXPERT_OUTPUT, device=tokens_device.device,
                    comm_stream=comm_stream, stream_scoped_sync=stream_scoped_sync,
                )
                submit_return_host_ns = time.monotonic_ns()
                with torch.cuda.stream(comm_stream):
                    gpu_end_event.record(comm_stream)
                gpu_end_event.synchronize()
                returned_meta_arrays.append(recv_meta)
                returned_output_arrays.append(recv_output)
                returned_counts.append(recvcounts)
                return_descriptors.append({
                    "descriptor_index": descriptor_index,
                    "sendcounts_tokens": list(returned.sendcounts_tokens),
                    "offsets_tokens": list(returned.offsets_tokens),
                    "tokens": returned.total_tokens,
                    "metadata_digest": _digest(returned.metadata),
                    "output_digest": _digest(returned.outputs),
                    "packing_us": (return_pack_done - return_pack_start) / 1e3,
                    "communication": timing,
                    "expert_dependencies": dependencies,
                    "dependency_wait_us": (dependency_ready_ns - wait_start_ns) / 1e3,
                    "dependency_ready_host_ns": int(dependency_ready_ns),
                    "submit_call_host_ns": int(submit_call_host_ns),
                    "submit_return_host_ns": int(submit_return_host_ns),
                    "_gpu_start_event": gpu_start_event,
                    "_gpu_end_event": gpu_end_event,
                })
            return_done_host_ns = time.monotonic_ns()
        except BaseException as error:
            return_error[0] = error

    return_thread: threading.Thread | None = None
    if progressive_expert:
        drain_ready_experts(flush=True)
        assert expert_stream is not None
        expert_stream.synchronize()
        all_expert_outputs = np.empty((all_forward_meta.shape[0], EXPERT_OUTPUT), dtype=np.float32)
        for task in expert_tasks:
            all_expert_outputs[task["indices"]] = task["output"].cpu().numpy()
        expert_batch_shapes = tuple(
            int(np.count_nonzero(all_forward_meta[:, 3] == expert)) for expert in range(EXPERTS)
        )
        expert_gpu_done = time.perf_counter_ns()
        executed = sorted(progressive_executed_indices)
        if executed != list(range(all_forward_meta.shape[0])):
            raise RuntimeError("progressive expert token loss/duplication")
    else:
        if first_expert_launch_host_ns == 0:
            first_expert_launch_host_ns = time.monotonic_ns()
        expert_input = torch.from_numpy(all_forward_features).to(tokens_device.device)
        if instrument_full_expert_batches:
            assert expert_stream is not None
            (expert_output_device, expert_batch_shapes, full_batch_tasks,
             baseline_expert_start_event, baseline_expert_end_event) = _reference_expert_mlp_with_events(
                expert_input, all_forward_meta[:, 3], expert_weights, expert_stream,
            )
        else:
            with torch.inference_mode(), torch.cuda.stream(expert_stream) if expert_stream is not None else torch.cuda.device(tokens_device.device):
                active_stream = expert_stream if expert_stream is not None else torch.cuda.current_stream(tokens_device.device)
                baseline_expert_start_event = torch.cuda.Event(enable_timing=True)
                baseline_expert_end_event = torch.cuda.Event(enable_timing=True)
                baseline_expert_start_event.record(active_stream)
                expert_output_device, expert_batch_shapes = reference_expert_mlp(
                    expert_input, all_forward_meta[:, 3], expert_weights,
                )
                baseline_expert_end_event.record(active_stream)
        if progressive_return:
            completion_by_expert = {
                int(value["expert"]): value["end_event"] for value in full_batch_tasks
            }
            return_thread = threading.Thread(
                target=execute_returns,
                args=(expert_output_device, completion_by_expert),
                name=f"r5-return-{arm}-rank{rank}",
            )
            return_thread.start()
        if expert_stream is not None:
            expert_stream.synchronize()
        else:
            torch.cuda.synchronize(tokens_device.device)
        expert_gpu_done = time.perf_counter_ns()
        all_expert_outputs = expert_output_device.cpu().numpy().copy()
    expert_done = time.perf_counter_ns()
    expert_done_host_ns = time.monotonic_ns()
    if return_thread is not None:
        return_thread.join(timeout=180)
        if return_thread.is_alive():
            raise RuntimeError("progressive return worker timed out")
        if return_error[0] is not None:
            raise RuntimeError("progressive return worker failed") from return_error[0]

    router_gpu_end_us = float(gpu_origin.elapsed_time(events[7]) * 1e3)
    forward_gpu_end_us = float(gpu_origin.elapsed_time(forward_done_event) * 1e3)
    if progressive_expert:
        task_rows = []
        for task in expert_tasks:
            start_us = float(gpu_origin.elapsed_time(task["start_event"]) * 1e3)
            end_us = float(gpu_origin.elapsed_time(task["end_event"]) * 1e3)
            task_rows.append({
                "expert": int(task["expert"]), "batch_size": int(task["batch_size"]),
                "flush": bool(task["flush"]), "start_us": start_us, "end_us": end_us,
                "gpu_duration_us": end_us - start_us,
                "hidden_before_router_us": max(0.0, min(end_us, router_gpu_end_us) - start_us),
                "hidden_before_forward_us": max(0.0, min(end_us, forward_gpu_end_us) - start_us),
                "completed_before_router": end_us <= router_gpu_end_us,
                "completed_before_forward": end_us <= forward_gpu_end_us,
            })
        expert_gpu_active_us = sum(value["gpu_duration_us"] for value in task_rows)
        hidden_before_router_us = sum(value["hidden_before_router_us"] for value in task_rows)
        hidden_before_forward_us = sum(value["hidden_before_forward_us"] for value in task_rows)
        expert_gpu_end_us = max((value["end_us"] for value in task_rows), default=forward_gpu_end_us)
        batch_sizes = [value["batch_size"] for value in task_rows]
    else:
        assert baseline_expert_start_event is not None and baseline_expert_end_event is not None
        start_us = float(gpu_origin.elapsed_time(baseline_expert_start_event) * 1e3)
        expert_gpu_end_us = float(gpu_origin.elapsed_time(baseline_expert_end_event) * 1e3)
        expert_gpu_active_us = expert_gpu_end_us - start_us
        hidden_before_router_us = 0.0
        hidden_before_forward_us = 0.0
        batch_sizes = [int(value) for value in expert_batch_shapes if int(value) > 0]
        task_rows = [{
            "expert": -1, "batch_size": int(sum(batch_sizes)), "flush": True,
            "start_us": start_us, "end_us": expert_gpu_end_us,
            "gpu_duration_us": expert_gpu_active_us,
            "hidden_before_router_us": 0.0, "hidden_before_forward_us": 0.0,
            "completed_before_router": False, "completed_before_forward": False,
        }]
    full_batch_rows = []
    for task in full_batch_tasks:
        full_start_us = float(gpu_origin.elapsed_time(task["start_event"]) * 1e3)
        full_end_us = float(gpu_origin.elapsed_time(task["end_event"]) * 1e3)
        full_batch_rows.append({
            "expert": int(task["expert"]), "batch_size": int(task["batch_size"]),
            "index_digest": _digest(np.asarray(task["indices"], dtype=np.int64)),
            "start_us": full_start_us, "end_us": full_end_us,
            "gpu_duration_us": full_end_us - full_start_us,
        })
    expert_tail_after_forward_gpu_us = max(0.0, expert_gpu_end_us - forward_gpu_end_us)

    if not progressive_return:
        execute_returns(all_expert_outputs, None)
        if return_error[0] is not None:
            raise RuntimeError("delayed return worker failed") from return_error[0]

    assert baseline_expert_end_event is not None or progressive_expert
    final_expert_gpu_us = expert_gpu_end_us
    return_hidden_before_expert_us = 0.0
    return_gpu_start_before_expert = 0
    return_gpu_complete_before_expert = 0
    for descriptor in return_descriptors:
        gpu_start_us = float(gpu_origin.elapsed_time(descriptor.pop("_gpu_start_event")) * 1e3)
        gpu_end_us = float(gpu_origin.elapsed_time(descriptor.pop("_gpu_end_event")) * 1e3)
        hidden_us = max(0.0, min(gpu_end_us, final_expert_gpu_us) - gpu_start_us)
        descriptor.update({
            "gpu_start_us": gpu_start_us, "gpu_end_us": gpu_end_us,
            "gpu_duration_us": gpu_end_us - gpu_start_us,
            "hidden_before_final_expert_us": hidden_us,
            "gpu_start_before_final_expert": gpu_start_us < final_expert_gpu_us,
            "gpu_complete_before_final_expert": gpu_end_us <= final_expert_gpu_us,
        })
        return_hidden_before_expert_us += hidden_us
        return_gpu_start_before_expert += int(gpu_start_us < final_expert_gpu_us)
        return_gpu_complete_before_expert += int(gpu_end_us <= final_expert_gpu_us)

    # The actual combine is only a deterministic position scatter.  Keep it
    # separate from all identity/checksum/oracle work so R4-P0 can stop its
    # primary clock when the real output is ready.
    actual_combine_start = time.perf_counter_ns()
    combined = np.zeros((TOTAL_TOKENS, EXPERT_OUTPUT), dtype=np.float32)
    for recv_meta, recv_output in zip(returned_meta_arrays, returned_output_arrays, strict=True):
        positions = np.asarray(recv_meta[:, 4], dtype=np.int64)
        combined[positions] = recv_output
    actual_combine_done = time.perf_counter_ns()
    primary_done_host_ns = time.monotonic_ns()

    # Independently reconstruct expected expert outputs from original local tokens/top-k.
    oracle_start = time.perf_counter_ns()
    local_experts = np.concatenate([np.asarray(value, dtype=np.int64) for value in host_numpy])
    oracle_device, oracle_shapes = reference_expert_mlp(tokens_device, local_experts, expert_weights)
    torch.cuda.synchronize(tokens_device.device)
    oracle_outputs = oracle_device.cpu().numpy().copy()
    expected_expert = {int(token): int(local_experts[pos]) for pos, token in enumerate(case_data["token_ids"])}
    expected_position = {int(token): pos for pos, token in enumerate(case_data["token_ids"])}
    expected_output = {int(token): oracle_outputs[pos] for pos, token in enumerate(case_data["token_ids"])}
    chunk_by_token = {
        int(item.token_id): int(item.chunk_id)
        for values in assignments_by_chunk for item in (values or ())
    }
    verified_combined = np.zeros((TOTAL_TOKENS, EXPERT_OUTPUT), dtype=np.float32)
    filled = np.zeros(TOTAL_TOKENS, dtype=np.bool_)
    return_checks = []
    combine_start = time.perf_counter_ns()
    for descriptor_index, (recv_meta, recv_output, recvcounts) in enumerate(
        zip(returned_meta_arrays, returned_output_arrays, returned_counts, strict=True)
    ):
        chunk_ids = forward_descriptors[descriptor_index]["chunk_ids"]
        required_positions = [
            pos for pos, token in enumerate(case_data["token_ids"])
            if chunk_by_token[int(token)] in chunk_ids
        ]
        required_tokens = {int(case_data["token_ids"][pos]) for pos in required_positions}
        partial, check = verify_return_and_combine(
            recv_meta, recv_output, origin_rank=rank, recvcounts_tokens=recvcounts,
            expected_expert_by_token={token: expected_expert[token] for token in required_tokens},
            expected_position_by_token={token: expected_position[token] for token in required_tokens},
            expected_output_by_token={token: expected_output[token] for token in required_tokens},
            total_tokens=TOTAL_TOKENS, required_positions=required_positions,
        )
        if not check["pass"]:
            raise RuntimeError(f"return/combine verification failed: {check}")
        for position in required_positions:
            if filled[position]: raise RuntimeError("cross-descriptor duplicate combine")
            verified_combined[position] = partial[position]; filled[position] = True
        return_checks.append(check)
    combine_done = time.perf_counter_ns()
    final_match = bool(
        filled.all()
        and np.allclose(verified_combined, oracle_outputs, atol=2e-3, rtol=2e-3)
        and np.allclose(combined, verified_combined, atol=0.0, rtol=0.0)
    )

    selected = [structural_signature(bound.selected) for bound in bounds]
    actions = [_action_signature(bound.proposal) for bound in bounds]
    oracle = _oracle_replay_variable(
        topology=topology, trial_id=trial_id,
        chunk_tokens=[tuple(value or ()) for value in controls_by_chunk],
        selected_signatures=selected, action_signatures=actions,
        decisions=decisions, compiled_state=state,
    )
    semantic = {
        "runtime_bfs_calls": binder.runtime_bfs_calls, "full_rebuild_count": state.full_rebuild_count,
        "unrevealed_execution": packing.unrevealed_execution,
        "future_access": packing.future_access_attempts,
        "duplicate_dispatch": packing.duplicate_dispatch, "stale_dispatch": packing.stale_dispatch,
        "candidate_divergences": oracle["candidate_divergences"],
        "action_divergences": oracle["action_divergences"],
        "checker_divergences": oracle["checker_divergences"],
        "holder_divergences": oracle["holder_divergences"],
        "legal": oracle["legal"], "total": oracle["total"],
    }
    forward_checks = [value["verification"] for value in forward_descriptors]
    all_forward_tokens = [int(row[0]) for value in received_forward for row in value["metadata"]]
    forward_cross_duplicate = len(all_forward_tokens) - len(set(all_forward_tokens))
    correctness = {
        "forward_duplicate": sum(int(value["duplicate"]) for value in forward_checks),
        "forward_cross_duplicate": forward_cross_duplicate,
        "wrong_source": sum(int(value["wrong_source"]) for value in forward_checks),
        "wrong_expert": sum(int(value["wrong_expert"]) for value in forward_checks) + sum(int(value["wrong_expert"]) for value in return_checks),
        "wrong_destination": sum(int(value["wrong_destination"]) for value in forward_checks) + sum(int(value["wrong_destination"]) for value in return_checks),
        "wrong_return": sum(int(value["wrong_return"]) for value in return_checks),
        "wrong_position": sum(int(value["wrong_position"]) for value in return_checks),
        "corruption": sum(int(value["corruption"]) for value in forward_checks) + sum(int(value["corruption"]) for value in return_checks),
        "lost": sum(int(value["lost"]) for value in return_checks),
        "duplicate": sum(int(value["duplicate"]) for value in return_checks),
        "expert_output_mismatch": sum(int(value["expert_output_mismatch"]) for value in return_checks),
        "expert_execution_loss": 0,
        "expert_execution_duplicate": 0,
        "expert_future_access": 0,
        "return_future_access": 0,
        "final_combine_correct": final_match,
        "token_integrity": bool(filled.all()),
    }
    zero_semantic = ("runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution", "future_access", "duplicate_dispatch", "stale_dispatch", "candidate_divergences", "action_divergences", "checker_divergences", "holder_divergences")
    if not (all(value == 0 for key, value in correctness.items() if key not in ("final_combine_correct", "token_integrity"))
            and correctness["final_combine_correct"] and correctness["token_integrity"]
            and semantic["legal"] == semantic["total"] and all(semantic[key] == 0 for key in zero_semantic)):
        raise RuntimeError(f"R4 correctness gate failed: {correctness}, {semantic}")
    full_reference_done_host_ns = time.monotonic_ns()
    result = {
        "arm": arm, "case": case_name, "rank": rank,
        "router_assignment_digest": payload_multiset_digest([item.record() for values in assignments_by_chunk for item in (values or ())]),
        "topk_digests": [_digest(np.asarray(value, dtype=np.int64)) for value in host_numpy],
        "forward_descriptors": forward_descriptors,
        "expert": {
            "input_digest": _digest(all_forward_features), "output_digest": _digest(all_expert_outputs),
            "batch_shapes": list(expert_batch_shapes), "oracle_batch_shapes": list(oracle_shapes),
            "weight_shapes": [list(value.shape) for value in expert_weights],
            "weight_digest": hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in expert_weights)).hexdigest(),
            "progressive": progressive_expert,
            "batch_threshold": int(expert_batch_threshold) if progressive_expert else 0,
            "full_batch_instrumented": bool(instrument_full_expert_batches),
            "full_batch_index_digests": [
                {"expert": value["expert"], "batch_size": value["batch_size"],
                 "index_digest": value["index_digest"]}
                for value in full_batch_rows
            ],
            "gemm_count": 2 * len(full_batch_rows) if instrument_full_expert_batches else 2 * sum(int(value) > 0 for value in expert_batch_shapes),
        },
        "return_descriptors": return_descriptors,
        "return_progressive": bool(progressive_return),
        "final_output_digest": _digest(combined), "oracle_output_digest": _digest(oracle_outputs),
        "correctness": correctness, "semantic": semantic,
        "scheduler_actions": [[list(item) for item in value] for value in actions],
        "timing": {
            "first_router_launch_host_ns": int(launch_ns[0]),
            "final_router_host_ns": final_router_ns,
            "forward_done_host_ns": forward_done_host_ns,
            "expert_done_host_ns": expert_done_host_ns,
            "return_start_host_ns": return_start_host_ns,
            "return_done_host_ns": return_done_host_ns,
            "first_expert_launch_host_ns": int(first_expert_launch_host_ns),
            "primary_done_host_ns": primary_done_host_ns,
            "full_reference_done_host_ns": full_reference_done_host_ns,
            "primary_makespan_us": (primary_done_host_ns - int(launch_ns[0])) / 1e3,
            "full_reference_makespan_us": (full_reference_done_host_ns - int(launch_ns[0])) / 1e3,
            "split_primary_timing": bool(split_primary_timing),
        },
        "diagnostics": {
            "router_us": (final_router_ns - router_start_ns) / 1e3,
            "router_chunk_cuda_us": [float(start.elapsed_time(end) * 1e3) for start, end in zip(starts, events, strict=True)],
            "data_prep": {
                "fast": bool(fast_data_prep),
                "static_precompute_us_outside_primary": float(packing.precompute_us) if fast_data_prep else 0.0,
                "mark_completed_total_us_inside_primary": float(packing.mark_completed_total_us) if fast_data_prep else 0.0,
                "count_h2d_overlap": bool(overlap_count_with_h2d),
            },
            "expert_h2d_gemm_us": (expert_gpu_done - expert_start) / 1e3,
            "expert_d2h_us": (expert_done - expert_gpu_done) / 1e3,
            "expert_progression": {
                "progressive": progressive_expert,
                "threshold": int(expert_batch_threshold) if progressive_expert else 0,
                "router_gpu_end_us": router_gpu_end_us,
                "forward_gpu_end_us": forward_gpu_end_us,
                "expert_gpu_end_us": expert_gpu_end_us,
                "expert_gpu_active_us": expert_gpu_active_us,
                "hidden_before_router_us": hidden_before_router_us,
                "hidden_before_forward_us": hidden_before_forward_us,
                "tail_after_forward_gpu_us": expert_tail_after_forward_gpu_us,
                "batch_sizes": batch_sizes,
                "tasks": task_rows,
                "full_batches": full_batch_rows,
            },
            "return_progression": {
                "progressive": bool(progressive_return),
                "descriptor_count": len(return_descriptors),
                "dependency_sizes": [len(value["expert_dependencies"]) for value in return_descriptors],
                "gpu_start_before_final_expert": return_gpu_start_before_expert,
                "gpu_complete_before_final_expert": return_gpu_complete_before_expert,
                "hidden_before_final_expert_us": return_hidden_before_expert_us,
                "tail_after_final_expert_gpu_us": max(
                    0.0,
                    max((value["gpu_end_us"] for value in return_descriptors), default=final_expert_gpu_us)
                    - final_expert_gpu_us,
                ),
            },
            "actual_combine_us": (actual_combine_done - actual_combine_start) / 1e3,
            "oracle_reconstruction_us": (combine_start - oracle_start) / 1e3,
            "combine_verification_us": (combine_done - combine_start) / 1e3,
        },
    }
    if retain_final_output:
        result["_final_output_array"] = combined.copy()
    return result


def _descriptor_equivalence(values: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    return [{key: value[key] for key in keys} for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASE_ORDER)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    cases = tuple(args.case or CASE_ORDER)
    if not args.allow_smoke and cases != CASE_ORDER:
        raise ValueError("canonical R4-A0/C0 requires all frozen correctness cases")
    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2: raise RuntimeError("R4 requires exactly two ranks")
    torch.cuda.set_device(rank); device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler(); plan = compiler.compile(topology)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    bridge = extension.IntegratedEventBridge(CHUNKS, allowed[-(rank + 1)], rank)
    router_stream = torch.cuda.Stream(device=device); comm_stream = torch.cuda.Stream(device=device)
    router_weight_cpu, router_bias_cpu = seed_router_params(D, EXPERTS, ROUTER_SEED)
    router_weight = router_weight_cpu.to(device)
    expert_weights = tuple(value.to(device) for value in seed_reference_experts(D, EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERTS, EXPERT_SEED))
    local_cases = []
    try:
        for case_name in cases:
            case_data = _case_inputs(case_name, rank)
            tokens = torch.from_numpy(case_data["tokens"]).to(device)
            bias = (router_bias_cpu + torch.from_numpy(case_data["bias_delta"])).to(device)
            mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
            mask[torch.arange(TOTAL_TOKENS, device=device), torch.from_numpy(case_data["topology_sources"]).to(device)] = True
            chunks = tuple(tokens.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
            masks = tuple(mask.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
            arms = {}
            for arm in ("C", "D"):
                dist.barrier()
                arms[arm] = _run_arm(
                    arm=arm, case_name=case_name, rank=rank, topology=topology, plan=plan,
                    bridge=bridge, case_data=case_data, tokens_device=tokens,
                    token_chunks=chunks, mask_chunks=masks, router_weight=router_weight,
                    router_bias=bias, expert_weights=expert_weights,
                    router_stream=router_stream, comm_stream=comm_stream,
                )
            local_cases.append({"case": case_name, "C": arms["C"], "D": arms["D"]})
            dist.barrier()
    finally:
        bridge.stop()
    local = {"rank": rank, "cases": local_cases}
    gathered: list[Any] | None = [None] * 2 if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        pairs, equivalence = [], []
        for row in gathered:
            for pair in row["cases"]:
                c, d = pair["C"], pair["D"]
                checks = {
                    "same_router_topk": c["topk_digests"] == d["topk_digests"],
                    "same_router_assignments": c["router_assignment_digest"] == d["router_assignment_digest"],
                    "same_forward_descriptors": _descriptor_equivalence(c["forward_descriptors"], ("chunk_ids", "sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest", "feature_digest")) == _descriptor_equivalence(d["forward_descriptors"], ("chunk_ids", "sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest", "feature_digest")),
                    "same_expert_batches_weights_outputs": c["expert"] == d["expert"],
                    "same_return_descriptors": _descriptor_equivalence(c["return_descriptors"], ("sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest", "output_digest")) == _descriptor_equivalence(d["return_descriptors"], ("sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest", "output_digest")),
                    "same_scheduler_actions": c["scheduler_actions"] == d["scheduler_actions"],
                    "same_final_outputs": c["final_output_digest"] == d["final_output_digest"],
                }
                equivalence.append({"rank": row["rank"], "case": pair["case"], **checks, "pass": all(checks.values())})
                pairs.extend((c, d))
        correctness_rows = [value["correctness"] for value in pairs]
        semantic_rows = [value["semantic"] for value in pairs]
        correctness = {
            "early_delayed_equivalence": all(value["pass"] for value in equivalence),
            "legality_100pct": all(value["legal"] == value["total"] for value in semantic_rows),
            "token_integrity_100pct": all(value["token_integrity"] for value in correctness_rows),
            "lost_zero": all(value["lost"] == 0 for value in correctness_rows),
            "duplicate_zero": all(value["duplicate"] == 0 and value["forward_duplicate"] == 0 and value["forward_cross_duplicate"] == 0 for value in correctness_rows),
            "wrong_source_zero": all(value["wrong_source"] == 0 for value in correctness_rows),
            "wrong_expert_zero": all(value["wrong_expert"] == 0 for value in correctness_rows),
            "wrong_destination_zero": all(value["wrong_destination"] == 0 for value in correctness_rows),
            "wrong_return_zero": all(value["wrong_return"] == 0 for value in correctness_rows),
            "wrong_position_zero": all(value["wrong_position"] == 0 for value in correctness_rows),
            "corruption_zero": all(value["corruption"] == 0 for value in correctness_rows),
            "expert_output_correct": all(value["expert_output_mismatch"] == 0 for value in correctness_rows),
            "combine_correct": all(value["final_combine_correct"] for value in correctness_rows),
            "semantic_divergence_zero": all(value[key] == 0 for value in semantic_rows for key in ("runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution", "future_access", "duplicate_dispatch", "stale_dispatch", "candidate_divergences", "action_divergences", "checker_divergences", "holder_divergences")),
        }
        forward = [descriptor for value in pairs for descriptor in value["forward_descriptors"]]
        returned = [descriptor for value in pairs for descriptor in value["return_descriptors"]]
        result = {
            "schema_version": 1, "study": "R4-A0/C0 reference full-MoE correctness",
            "status": "R4_A0_C0_PASS_PENDING_SUPERVISOR" if all(correctness.values()) else "R4_A0_C0_FAIL_PENDING_SUPERVISOR",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {"world_size": 2, "devices": [torch.cuda.get_device_name(i) for i in range(2)], "torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(), "python": platform.python_version()},
            "frozen_protocol": {"cases": list(cases), "tokens_per_rank_case": TOTAL_TOKENS, "router_dim": D, "expert_hidden": EXPERT_HIDDEN, "expert_output": EXPERT_OUTPUT, "expert_progressive": False, "forward_real_variable_a2av": True, "return_real_variable_a2av": True, "partial_shards_ratio": 0.75, "checkpoint8": True},
            "correctness": correctness, "pass": all(correctness.values()), "equivalence": equivalence,
            "diagnostics": {
                "router_us": distribution([value["diagnostics"]["router_us"] for value in pairs]),
                "router_chunk_cuda_us": distribution([item for value in pairs for item in value["diagnostics"]["router_chunk_cuda_us"]]),
                "forward_count_construction_us": distribution([value["count_construction_us"] for value in forward]),
                "forward_offset_construction_us": distribution([value["offset_construction_us"] for value in forward]),
                "forward_packing_us": distribution([value["packing_us"] for value in forward]),
                "forward_h2d_us": distribution([value["communication"]["h2d_us"] for value in forward]),
                "forward_count_exchange_us": distribution([value["communication"]["count_exchange_us"] for value in forward]),
                "forward_aiccl_control_us": distribution([value["aiccl_control_us"] for value in forward]),
                "forward_a2av_submit_us": distribution([value["communication"]["a2av_submit_us"] for value in forward]),
                "forward_a2av_completion_us": distribution([value["communication"]["a2av_completion_us"] for value in forward]),
                "expert_h2d_gemm_us": distribution([value["diagnostics"]["expert_h2d_gemm_us"] for value in pairs]),
                "expert_d2h_us": distribution([value["diagnostics"]["expert_d2h_us"] for value in pairs]),
                "return_packing_us": distribution([value["packing_us"] for value in returned]),
                "return_h2d_us": distribution([value["communication"]["h2d_us"] for value in returned]),
                "return_count_exchange_us": distribution([value["communication"]["count_exchange_us"] for value in returned]),
                "return_a2av_submit_us": distribution([value["communication"]["a2av_submit_us"] for value in returned]),
                "return_a2av_completion_us": distribution([value["communication"]["a2av_completion_us"] for value in returned]),
                "oracle_reconstruction_us": distribution([value["diagnostics"]["oracle_reconstruction_us"] for value in pairs]),
                "combine_verification_us": distribution([value["diagnostics"]["combine_verification_us"] for value in pairs]),
            },
            "rank_results": gathered,
        }
        if not result["pass"]: raise RuntimeError(f"R4 aggregate correctness failed: {correctness}")
        output = args.output_dir / "r4_a0_c0_results.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "sha256": sha256_file(output), "pass": result["pass"], "correctness": correctness}, indent=2))
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
