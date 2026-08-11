"""R3-P0 progressive early versus identical delayed real A2Av-T0 pilot.

This runner is restricted to the preregistered pilot seeds.  It deliberately
keeps reference packing/count exchange unchanged and defers receive D2H plus
verification until after the primary payload-completion timestamp.
"""

from __future__ import annotations

import argparse
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
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import (  # noqa: E402
    DynamicGuard, FastBinder, IncrementalState, StaticPlanCompiler, structural_signature,
)
from rlccl.transport.reference_a2av import (  # noqa: E402
    PAYLOAD_FIELDS, ProgressivePackingState, RouterAssignment, decode_records,
    pack_destination_layout, payload_multiset_digest, verify_received_records,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from scripts.run_r2_f0_integrated import (  # noqa: E402
    CHUNKS, D, EXPERTS, PARTIAL_CHUNKS, SCHEDULER_TRIGGERS, TOP_K,
    WAIT_TIMEOUT_NS, _action_signature, _load_bridge_extension,
)
from scripts.run_r3_a0_c0 import (  # noqa: E402
    CONTROL_LIMIT_PER_CHUNK, TOTAL_TOKENS, _control_tokens,
    _oracle_replay_variable, _warm_variable_alltoallv, distribution, sha256_file,
)


PILOT_SEEDS = (6042, 6142, 6242)
FORMAL_SEEDS = (5042, 5142, 5242)
FAMILIES = (
    "balanced", "skewed", "all_to_one_like", "zero_sized_pair",
    "multiple_progressive_shards",
)
JOBS_PER_FAMILY = 10
DEFAULT_CHUNKS = (512,) * CHUNKS
FAMILY_SPECS = {
    "balanced": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (0.0, 0.0, 0.0, 0.0)},
    "skewed": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (100.0, 0.0, 0.0, 0.0)},
    "all_to_one_like": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (100.0, 0.0, 50.0, 0.0)},
    "zero_sized_pair": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (0.0, 100.0, 0.0, 50.0)},
    "multiple_progressive_shards": {
        "chunk_sizes": (128, 256, 384, 512, 640, 768, 512, 896),
        "bias": (0.0, 0.0, 0.0, 0.0),
    },
}
PARAMETER_SEED = 20260805
A2AV_NAME = "A2Av-T0"


def _array_digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _label(kind: str, seed: int, family_index: int, job: int, arm: str, item: int) -> str:
    return (
        f"R3P0|kind={kind}|seed={seed}|family={family_index}|"
        f"job={job}|arm={arm}|item={item}"
    )


def _job_inputs(seed: int, family: str, job: int, rank: int) -> dict[str, Any]:
    if seed not in PILOT_SEEDS or seed in FORMAL_SEEDS:
        raise ValueError("only preregistered pilot seeds are permitted")
    family_index = FAMILIES.index(family)
    spec = FAMILY_SPECS[family]
    chunk_sizes = tuple(int(value) for value in spec["chunk_sizes"])
    if len(chunk_sizes) != CHUNKS or sum(chunk_sizes) != TOTAL_TOKENS:
        raise RuntimeError("invalid frozen chunk layout")
    rng_seed = seed * 100_000 + family_index * 1_000 + job * 10 + rank
    rng = np.random.default_rng(rng_seed)
    tokens = rng.standard_normal((TOTAL_TOKENS, D)).astype(np.float32)
    topology_sources = np.arange(TOTAL_TOKENS, dtype=np.int64) % EXPERTS
    token_base = seed * 10_000_000_000 + family_index * 100_000_000 + job * 10_000_000
    token_ids = token_base + rank * 1_000_000 + np.arange(TOTAL_TOKENS, dtype=np.int64)
    first = tokens[:, 0].view(np.uint32).astype(np.uint64)
    second = tokens[:, 1].view(np.uint32).astype(np.uint64)
    payload_words = ((first << np.uint64(30)) ^ second ^ token_ids.astype(np.uint64))
    payload_words &= np.uint64((1 << 62) - 1)
    offsets = np.cumsum((0,) + chunk_sizes)
    return {
        "case": family, "family_index": family_index, "job": job,
        "chunk_sizes": chunk_sizes,
        "chunk_offsets": tuple(int(value) for value in offsets),
        "tokens": tokens, "topology_sources": topology_sources,
        "token_ids": token_ids, "payload_words": payload_words.astype(np.int64),
        "bias_delta": np.asarray(spec["bias"], dtype=np.float32),
    }


def _make_assignments(
    case_data: dict[str, Any], rank: int, chunk: int, experts: np.ndarray,
) -> tuple[RouterAssignment, ...]:
    left, right = case_data["chunk_offsets"][chunk:chunk + 2]
    if len(experts) != right - left:
        raise ValueError("router output cardinality mismatch")
    return tuple(
        RouterAssignment(
            token_id=int(case_data["token_ids"][left + offset]), source_rank=rank,
            destination_rank=int(expert) % dist.get_world_size(), expert_id=int(expert),
            chunk_id=chunk, chunk_offset=offset,
            payload_word=int(case_data["payload_words"][left + offset]),
        )
        for offset, expert in enumerate(experts.tolist())
    )


def _run_arm(
    *, arm: str, seed: int, family: str, job: int, rank: int, topology: Any,
    plan: Any, bridge: Any, case_data: dict[str, Any], tokens_device: torch.Tensor,
    token_chunks: Sequence[torch.Tensor], mask_chunks: Sequence[torch.Tensor],
    weight: torch.Tensor, bias: torch.Tensor, router_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
) -> dict[str, Any]:
    delayed = arm == "D"
    if arm not in ("C", "D"):
        raise ValueError("invalid arm")
    family_index = FAMILIES.index(family)
    trial_id = f"r3-p0-{seed}-{family}-{job}-{arm}-rank{rank}"
    total_controls = sum(min(value, CONTROL_LIMIT_PER_CHUNK) for value in case_data["chunk_sizes"])
    state = IncrementalState(
        plan, max_tokens=max(1, total_controls), max_chunks=CHUNKS,
        sequence_id=trial_id, sequence_step=8,
    )
    binder, guard = FastBinder(plan), DynamicGuard(plan)
    packing = ProgressivePackingState(world_size=2, source_rank=rank, max_chunks=CHUNKS)
    bridge.reset_all()
    completion_events = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    host_indices = [
        torch.empty(size, dtype=torch.int64, pin_memory=True)
        for size in case_data["chunk_sizes"]
    ]
    host_numpy = [value.numpy() for value in host_indices]
    launch_host_ns = [0] * CHUNKS
    ready_host_ns = [0] * CHUNKS
    producer_error: list[BaseException | None] = [None]
    assignments_by_chunk: list[Any] = [None] * CHUNKS
    control_by_chunk: list[Any] = [None] * CHUNKS
    bounds: list[Any] = []
    decisions: list[Any] = []
    descriptors: list[dict[str, Any]] = []
    delayed_descriptors: list[tuple[Any, dict[str, Any]]] = []
    pending_payloads: list[dict[str, Any]] = []
    all_sent_records: list[tuple[int, ...]] = []

    def producer() -> None:
        try:
            torch.cuda.set_device(rank)
            with torch.inference_mode():
                for chunk in range(CHUNKS):
                    launch_host_ns[chunk] = time.monotonic_ns()
                    label = _label("router", seed, family_index, job, arm, chunk)
                    with record_function(label), torch.cuda.stream(router_stream):
                        start_events[chunk].record(router_stream)
                        indices, _ = router_topk(
                            token_chunks[chunk], weight, bias, TOP_K, mask=mask_chunks[chunk],
                        )
                        host_indices[chunk].copy_(indices, non_blocking=True)
                        completion_events[chunk].record(router_stream)
                    bridge.arm(chunk, completion_events[chunk].cuda_event)
        except BaseException as error:
            producer_error[0] = error

    origin_ns = time.monotonic_ns()
    thread = threading.Thread(target=producer, name=f"r3p0-router-rank{rank}")
    thread.start()

    def consume(chunk: int, reveal: bool) -> None:
        ready_host_ns[chunk] = int(bridge.wait_ready(chunk, WAIT_TIMEOUT_NS))
        assignments = _make_assignments(case_data, rank, chunk, host_numpy[chunk])
        controls = _control_tokens(
            case_name=f"p0-{seed}-{family}-{job}", rank=rank, chunk=chunk,
            assignments=assignments, case_data=case_data,
        )
        assignments_by_chunk[chunk], control_by_chunk[chunk] = assignments, controls
        packing.mark_completed(chunk, assignments)
        state.stage_ready_chunk(chunk, controls)
        if reveal:
            packing.reveal(chunk)
            state.consume_pending_chunk(chunk)

    def schedule(trigger: int) -> dict[str, float]:
        start = time.perf_counter_ns()
        with record_function(_label("aiccl", seed, family_index, job, arm, trigger)):
            bound = binder.step(state)
            action_ns = time.perf_counter_ns()
            decision = guard.apply(
                state, bound.proposal, require_scheduler_semantics=True,
                expected_state_version=bound.state_version,
            )
        done = time.perf_counter_ns()
        if not decision.accepted:
            raise RuntimeError(f"compiled scheduler failed closed: {decision}")
        bounds.append(bound)
        decisions.append(decision)
        return {
            "aiccl_action_us": (action_ns - start) / 1e3,
            "aiccl_control_us": (done - start) / 1e3,
        }

    def build_descriptor(chunk_ids: tuple[int, ...], trigger: int, control: dict[str, float]) -> tuple[Any, dict[str, Any]]:
        split_timing: dict[str, float] = {}
        layout = packing.build_delta_layout(chunk_ids, timing_sink=split_timing)
        pack_start = time.perf_counter_ns()
        with record_function(_label("packing", seed, family_index, job, arm, trigger)):
            packed = pack_destination_layout(layout)
        pack_done = time.perf_counter_ns()
        meta = {
            "descriptor_index": packing.descriptor_count - 1,
            "trigger_chunk": trigger, "chunk_ids": list(chunk_ids),
            "sendcounts_tokens": list(packed.sendcounts_tokens),
            "offsets_tokens": list(packed.offsets_tokens),
            "token_count": packed.total_tokens, "bytes": packed.total_bytes,
            **split_timing, "packing_us": (pack_done - pack_start) / 1e3, **control,
            "payload_multiset_digest": payload_multiset_digest(decode_records(packed.records)),
            "built_host_ns": time.monotonic_ns(),
        }
        return packed, meta

    def submit(packed: Any, meta: dict[str, Any]) -> None:
        trigger = int(meta["trigger_chunk"])
        h2d_start = time.perf_counter_ns()
        send_flat = np.array(packed.records.reshape(-1), dtype=np.int64, copy=True)
        send_device = torch.from_numpy(send_flat).to(tokens_device.device)
        send_counts = torch.tensor(packed.sendcounts_tokens, dtype=torch.int64, device=tokens_device.device)
        recv_counts_device = torch.empty(2, dtype=torch.int64, device=tokens_device.device)
        h2d_done = time.perf_counter_ns()
        count_call_host_ns = time.monotonic_ns()
        count_call = time.perf_counter_ns()
        with record_function(_label("count", seed, family_index, job, arm, trigger)), torch.cuda.stream(comm_stream):
            count_work = dist.all_to_all_single(recv_counts_device, send_counts, async_op=True)
        count_return = time.perf_counter_ns()
        count_work.wait()
        recv_counts = tuple(int(value) for value in recv_counts_device.cpu().tolist())
        count_done = time.perf_counter_ns()
        recv_device = torch.empty(sum(recv_counts) * PAYLOAD_FIELDS, dtype=torch.int64, device=tokens_device.device)
        payload_call_host_ns = time.monotonic_ns()
        payload_call = time.perf_counter_ns()
        with record_function(_label("payload", seed, family_index, job, arm, trigger)), torch.cuda.stream(comm_stream):
            payload_work = dist.all_to_all_single(
                recv_device, send_device,
                output_split_sizes=[value * PAYLOAD_FIELDS for value in recv_counts],
                input_split_sizes=list(packed.sendcounts_elements), async_op=True,
            )
        payload_return = time.perf_counter_ns()
        meta["communication"] = {
            "h2d_us": (h2d_done - h2d_start) / 1e3,
            "count_submit_us": (count_return - count_call) / 1e3,
            "count_completion_us": (count_done - count_call) / 1e3,
            "payload_submit_us": (payload_return - payload_call) / 1e3,
            "count_call_host_ns": count_call_host_ns,
            "payload_call_host_ns": payload_call_host_ns,
            "payload_submit_return_host_ns": time.monotonic_ns(),
        }
        descriptors.append(meta)
        all_sent_records.extend(decode_records(packed.records))
        pending_payloads.append({
            "work": payload_work, "recv_device": recv_device,
            "send_device": send_device, "meta": meta,
        })

    for chunk in range(PARTIAL_CHUNKS):
        consume(chunk, True)
        state.stage, state.ratio = chunk + 1, (chunk + 1) / CHUNKS
        control = schedule(chunk)
        packed, meta = build_descriptor((chunk,), chunk, control)
        if delayed:
            delayed_descriptors.append((packed, meta))
        else:
            submit(packed, meta)

    consume(6, False)
    consume(7, False)
    final_router_host_ns = int(ready_host_ns[7])
    for chunk in (6, 7):
        packing.reveal(chunk)
        state.consume_pending_chunk(chunk)
    state.stage, state.ratio = CHUNKS, 1.0
    control = schedule(7)
    packed, meta = build_descriptor((6, 7), 7, control)
    if delayed:
        delayed_descriptors.append((packed, meta))
        for one_packed, one_meta in delayed_descriptors:
            submit(one_packed, one_meta)
    else:
        submit(packed, meta)

    thread.join(timeout=120.0)
    if thread.is_alive() or producer_error[0] is not None:
        raise RuntimeError("router producer failed") from producer_error[0]
    for pending in pending_payloads:
        pending["work"].wait()
    torch.cuda.synchronize(tokens_device.device)
    primary_done_ns = time.monotonic_ns()
    router_chunk_us = [
        float(start.elapsed_time(end) * 1e3)
        for start, end in zip(start_events, completion_events, strict=True)
    ]

    all_received_arrays = []
    d2h_values = []
    for pending in pending_payloads:
        copy_start = time.perf_counter_ns()
        received = pending["recv_device"].cpu().numpy().reshape((-1, PAYLOAD_FIELDS)).copy()
        copy_done = time.perf_counter_ns()
        d2h_values.append((copy_done - copy_start) / 1e3)
        all_received_arrays.append(received)
    gathered_sent: list[Any] = [None, None]
    dist.all_gather_object(gathered_sent, all_sent_records)
    expected_by_token = {
        int(row[0]): tuple(int(value) for value in row)
        for rows in gathered_sent for row in rows if int(row[2]) == rank
    }
    received = np.concatenate(all_received_arrays, axis=0)
    verify_start = time.perf_counter_ns()
    verification = verify_received_records(received, destination_rank=rank, expected_by_token=expected_by_token)
    verify_done = time.perf_counter_ns()
    verification["d2h_us"] = sum(d2h_values)
    verification["unpack_verification_us"] = (verify_done - verify_start) / 1e3
    full_reference_done_ns = time.monotonic_ns()

    selected = [structural_signature(bound.selected) for bound in bounds]
    actions = [_action_signature(bound.proposal) for bound in bounds]
    oracle = _oracle_replay_variable(
        topology=topology, trial_id=trial_id,
        chunk_tokens=[tuple(value or ()) for value in control_by_chunk],
        selected_signatures=selected, action_signatures=actions,
        decisions=decisions, compiled_state=state,
    )
    assignment_rows = [item.record() for values in assignments_by_chunk for item in (values or ())]
    semantic = {
        "runtime_bfs_calls": binder.runtime_bfs_calls,
        "full_rebuild_count": state.full_rebuild_count,
        "unrevealed_execution": packing.unrevealed_execution,
        "future_access": packing.future_access_attempts,
        "duplicate_dispatch": packing.duplicate_dispatch,
        "stale_dispatch": packing.stale_dispatch,
        "candidate_divergences": oracle["candidate_divergences"],
        "action_divergences": oracle["action_divergences"],
        "checker_divergences": oracle["checker_divergences"],
        "holder_divergences": oracle["holder_divergences"],
        "legal": oracle["legal"], "total": oracle["total"],
        "token_integrity": bool(
            verification["pass"] and packing.dispatched_token_count == TOTAL_TOKENS
            and len(assignment_rows) == TOTAL_TOKENS
            and len({int(row[0]) for row in assignment_rows}) == TOTAL_TOKENS
        ),
    }
    zero_keys = (
        "runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution", "future_access",
        "duplicate_dispatch", "stale_dispatch", "candidate_divergences",
        "action_divergences", "checker_divergences", "holder_divergences",
    )
    if not (verification["pass"] and semantic["token_integrity"]
            and semantic["legal"] == semantic["total"]
            and all(semantic[key] == 0 for key in zero_keys)):
        raise RuntimeError(f"P0 correctness failure: {semantic}, {verification}")
    return {
        "arm": arm, "seed": seed, "family": family, "family_index": family_index,
        "job": job, "rank": rank, "transport": A2AV_NAME,
        "origin_host_ns": origin_ns, "first_router_launch_host_ns": int(launch_host_ns[0]),
        "primary_done_host_ns": primary_done_ns,
        "full_reference_done_host_ns": full_reference_done_ns,
        "final_router_host_ns": final_router_host_ns,
        "primary_makespan_us": (primary_done_ns - launch_host_ns[0]) / 1e3,
        "full_reference_makespan_us": (full_reference_done_ns - launch_host_ns[0]) / 1e3,
        "router_final_latency_us": (final_router_host_ns - launch_host_ns[0]) / 1e3,
        "router_chunk_cuda_us": router_chunk_us,
        "chunk_launch_host_ns": launch_host_ns, "chunk_ready_host_ns": ready_host_ns,
        "descriptors": descriptors, "verification": verification, "semantic": semantic,
        "router_assignment_digest": payload_multiset_digest(assignment_rows),
        "final_sent_payload_multiset_digest": payload_multiset_digest(all_sent_records),
        "final_received_payload_multiset_digest": verification["payload_multiset_digest"],
        "topk_by_chunk_digests": [_array_digest(np.asarray(value, dtype=np.int64)) for value in host_numpy],
        "scheduler_selected_signatures": [[list(item) for item in value] for value in selected],
        "scheduler_action_signatures": [[list(item) for item in value] for value in actions],
        "total_sent_tokens": len(all_sent_records),
        "total_sent_bytes": len(all_sent_records) * PAYLOAD_FIELDS * 8,
        "total_received_tokens": int(received.shape[0]),
    }


def _descriptor_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ("trigger_chunk", "chunk_ids", "sendcounts_tokens", "offsets_tokens", "token_count", "bytes", "payload_multiset_digest")
    return [{key: descriptor[key] for key in keys} for descriptor in result["descriptors"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=PILOT_SEEDS)
    parser.add_argument("--jobs-per-family", type=int, default=JOBS_PER_FAMILY)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not args.allow_smoke and args.jobs_per_family != JOBS_PER_FAMILY:
        raise ValueError("canonical P0 requires ten jobs per family")
    if args.seed in FORMAL_SEEDS:
        raise ValueError("formal seed access forbidden")

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R3-P0 requires exactly two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler()
    plan = compiler.compile(topology)
    if compiler.compile_bfs_sources <= 0 or not plan.proof.valid:
        raise RuntimeError("compiled static proof unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed[-(rank + 1)] if allowed != [-1] else -1
    bridge = extension.IntegratedEventBridge(CHUNKS, cpu_core, rank)
    router_stream = torch.cuda.Stream(device=device, priority=0)
    comm_stream = torch.cuda.Stream(device=device, priority=0)
    _warm_variable_alltoallv(device, rank)
    weight_cpu, base_bias_cpu = seed_router_params(D, EXPERTS, PARAMETER_SEED)
    weight = weight_cpu.to(device)
    trace_path = args.output_dir / f"r3_p0_seed{args.seed}_rank{rank}.trace.json"
    local_pairs: list[dict[str, Any]] = []
    seed_index = PILOT_SEEDS.index(args.seed)
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            for family_index, family in enumerate(FAMILIES):
                for job in range(args.jobs_per_family):
                    case_data = _job_inputs(args.seed, family, job, rank)
                    tokens_device = torch.from_numpy(case_data["tokens"]).to(device)
                    bias = (base_bias_cpu + torch.from_numpy(case_data["bias_delta"])).to(device)
                    mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
                    mask[torch.arange(TOTAL_TOKENS, device=device), torch.from_numpy(case_data["topology_sources"]).to(device)] = True
                    chunks = tuple(tokens_device.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
                    masks = tuple(mask.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
                    with torch.inference_mode(), torch.cuda.stream(router_stream):
                        router_topk(chunks[0], weight, bias, TOP_K, mask=masks[0])
                    torch.cuda.synchronize(device)
                    order = ("C", "D") if (seed_index + family_index + job) % 2 == 0 else ("D", "C")
                    arms: dict[str, Any] = {}
                    for arm in order:
                        dist.barrier()
                        arms[arm] = _run_arm(
                            arm=arm, seed=args.seed, family=family, job=job, rank=rank,
                            topology=topology, plan=plan, bridge=bridge, case_data=case_data,
                            tokens_device=tokens_device, token_chunks=chunks, mask_chunks=masks,
                            weight=weight, bias=bias, router_stream=router_stream,
                            comm_stream=comm_stream,
                        )
                    local_pairs.append({"family": family, "job": job, "order": list(order), "C": arms["C"], "D": arms["D"]})
                    dist.barrier()
            torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(trace_path))
    finally:
        bridge.stop()

    local = {
        "rank": rank, "seed": args.seed, "poller_cpu_core": cpu_core,
        "poller_pinned": bool(bridge.pinned), "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path), "trace_size_bytes": trace_path.stat().st_size,
        "pairs": local_pairs,
    }
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        rank_lookup = {(row["rank"], pair["family"], pair["job"]): pair for row in gathered for pair in row["pairs"]}
        pair_rows, equivalence = [], []
        for family in FAMILIES:
            for job in range(args.jobs_per_family):
                rank_pairs = [rank_lookup[(r, family, job)] for r in range(world_size)]
                c_rows, d_rows = [p["C"] for p in rank_pairs], [p["D"] for p in rank_pairs]
                c_primary = (max(r["primary_done_host_ns"] for r in c_rows) - min(r["first_router_launch_host_ns"] for r in c_rows)) / 1e3
                d_primary = (max(r["primary_done_host_ns"] for r in d_rows) - min(r["first_router_launch_host_ns"] for r in d_rows)) / 1e3
                c_full = (max(r["full_reference_done_host_ns"] for r in c_rows) - min(r["first_router_launch_host_ns"] for r in c_rows)) / 1e3
                d_full = (max(r["full_reference_done_host_ns"] for r in d_rows) - min(r["first_router_launch_host_ns"] for r in d_rows)) / 1e3
                pair_rows.append({
                    "seed": args.seed, "family": family, "job": job,
                    "C_primary_us": c_primary, "D_primary_us": d_primary,
                    "delta_us": d_primary - c_primary,
                    "C_full_reference_us": c_full, "D_full_reference_us": d_full,
                })
                for r, pair in enumerate(rank_pairs):
                    c, d = pair["C"], pair["D"]
                    checks = {
                        "same_router": c["router_assignment_digest"] == d["router_assignment_digest"],
                        "same_topk": c["topk_by_chunk_digests"] == d["topk_by_chunk_digests"],
                        "same_descriptors": _descriptor_signature(c) == _descriptor_signature(d),
                        "same_bytes": c["total_sent_bytes"] == d["total_sent_bytes"],
                        "same_calls": len(c["descriptors"]) == len(d["descriptors"]),
                        "same_actions": c["scheduler_action_signatures"] == d["scheduler_action_signatures"],
                        "same_sent_multiset": c["final_sent_payload_multiset_digest"] == d["final_sent_payload_multiset_digest"],
                        "same_received_multiset": c["final_received_payload_multiset_digest"] == d["final_received_payload_multiset_digest"],
                    }
                    equivalence.append({"family": family, "job": job, "rank": r, **checks, "pass": all(checks.values())})
        arms = [pair[arm] for row in gathered for pair in row["pairs"] for arm in ("C", "D")]
        semantic = [row["semantic"] for row in arms]
        verification = [row["verification"] for row in arms]
        correctness = {
            "runtime_bfs_zero": all(row["runtime_bfs_calls"] == 0 for row in semantic),
            "full_rebuild_zero": all(row["full_rebuild_count"] == 0 for row in semantic),
            "unrevealed_execution_zero": all(row["unrevealed_execution"] == 0 for row in semantic),
            "future_access_zero": all(row["future_access"] == 0 for row in semantic),
            "duplicate_dispatch_zero": all(row["duplicate_dispatch"] == 0 for row in semantic),
            "stale_dispatch_zero": all(row["stale_dispatch"] == 0 for row in semantic),
            "semantic_divergence_zero": all(row[key] == 0 for row in semantic for key in ("candidate_divergences", "action_divergences", "checker_divergences", "holder_divergences")),
            "legality_100pct": all(row["legal"] == row["total"] for row in semantic),
            "token_integrity_100pct": all(row["token_integrity"] for row in semantic),
            "lost_zero": all(row["lost"] == 0 for row in verification),
            "duplicate_zero": all(row["duplicate"] == 0 for row in verification),
            "wrong_destination_zero": all(row["wrong_destination"] == 0 for row in verification),
            "corruption_zero": all(row["corruption"] == 0 for row in verification),
            "cd_equivalence_zero": all(row["pass"] for row in equivalence),
        }
        payload = {
            "schema_version": 1, "study": "R3-P0 progressive early A2Av pilot",
            "status": "R3_P0_RAW_COMPLETE_PENDING_TRACE_ANALYSIS",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size, "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "python": platform.python_version(),
            },
            "frozen_protocol": {
                "seed": args.seed, "families": list(FAMILIES),
                "jobs_per_family": args.jobs_per_family, "parameter_seed": PARAMETER_SEED,
                "tokens_per_rank": TOTAL_TOKENS, "dimension": D, "experts": EXPERTS,
                "top_k": TOP_K, "chunks": CHUNKS, "partial_shards_ratio": 0.75,
                "checkpoint8": True, "transport": A2AV_NAME,
                "formal_seeds_touched": False,
            },
            "correctness": correctness, "pass": all(correctness.values()),
            "paired_rows": pair_rows, "equivalence": equivalence,
            "host_diagnostics": {
                "C_primary_us": distribution([row["C_primary_us"] for row in pair_rows]),
                "D_primary_us": distribution([row["D_primary_us"] for row in pair_rows]),
                "delta_us": distribution([row["delta_us"] for row in pair_rows]),
            },
            "rank_results": gathered,
        }
        if not payload["pass"]:
            raise RuntimeError(f"P0 fail-closed correctness gate failed: {correctness}")
        output = args.output_dir / f"r3_p0_seed{args.seed}_host.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "sha256": sha256_file(output), "pairs": len(pair_rows), "pass": payload["pass"]}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
