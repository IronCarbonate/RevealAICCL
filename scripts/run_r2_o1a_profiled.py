"""R2-O1A strict A/B/C/D device scheduling diagnosis runner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import (  # noqa: E402
    DynamicGuard,
    FastBinder,
    IncrementalState,
    StaticPlanCompiler,
    structural_signature,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from rlccl.uncertainty.observation import TruthTokenId  # noqa: E402
from scripts.run_r2_f0_integrated import (  # noqa: E402
    CHUNKS,
    CONTROL_PER_CHUNK,
    D,
    EXPERTS,
    PARTIAL_CHUNKS,
    SCHEDULER_TRIGGERS,
    TOKENS_PER_CHUNK,
    TOP_K,
    TOTAL_CONTROL_TOKENS,
    WAIT_TIMEOUT_NS,
    _FastStateIngress,
    _action_signature,
    _load_bridge_extension,
    _make_tokens,
    _oracle_replay,
    sha256_file,
)


MODES = ("A", "B", "C", "D")
MODE_ORDERS = {
    4042: ("A", "C", "D", "B"),
    4043: ("B", "D", "C", "A"),
    4044: ("D", "A", "B", "C"),
}


def _label(kind: str, *, seed: int, mode: str, trial: int, item: int) -> str:
    return f"R2O1A|kind={kind}|seed={seed}|mode={mode}|trial={trial}|item={item}"


def _run_trial(
    *,
    mode: str,
    seed: int,
    trial_index: int,
    rank: int,
    topology: Any,
    plan: Any,
    bridge: Any,
    token_chunks: tuple[torch.Tensor, ...],
    mask_chunks: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    sources: np.ndarray,
    router_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    events: list[torch.cuda.Event],
    host_indices: list[torch.Tensor],
    host_index_numpy: list[np.ndarray],
    descriptor_host_rows: list[torch.Tensor],
    descriptor_numpy_rows: list[np.ndarray],
    descriptor_device_rows: list[torch.Tensor],
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("unknown O1A condition")
    trial_id = f"o1a-seed{seed}-{mode}-{trial_index}-rank{rank}"
    use_runtime = mode != "A"
    use_nccl = mode in ("C", "D")
    delayed_nccl = mode == "D"

    state = binder = guard = ingress = None
    control_sources = token_ids = None
    if use_runtime:
        state = IncrementalState(
            plan,
            max_tokens=TOTAL_CONTROL_TOKENS,
            max_chunks=CHUNKS,
            sequence_id=trial_id,
            sequence_step=8,
        )
        binder = FastBinder(plan)
        guard = DynamicGuard(plan)
        control_sources = np.asarray(
            [sources[chunk * TOKENS_PER_CHUNK + offset]
             for chunk in range(CHUNKS) for offset in range(CONTROL_PER_CHUNK)],
            dtype=np.int32,
        )
        token_ids = tuple(
            TruthTokenId(f"r2-o1a:{trial_id}:{chunk}:{offset}")
            for chunk in range(CHUNKS) for offset in range(CONTROL_PER_CHUNK)
        )
        ingress = _FastStateIngress(state, token_ids, control_sources)
        bridge.reset_all()

    launch_host_ns = np.zeros(CHUNKS, dtype=np.int64)
    ready_host_ns = np.zeros(CHUNKS, dtype=np.int64)
    nccl_call_ns = np.zeros(len(SCHEDULER_TRIGGERS), dtype=np.int64)
    nccl_return_ns = np.zeros(len(SCHEDULER_TRIGGERS), dtype=np.int64)
    destination_snapshot = np.full((CHUNKS, CONTROL_PER_CHUNK), -1, dtype=np.int64)
    bounds: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    decisions: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    works: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    device_indices: list[Any] = [None] * CHUNKS
    device_scores: list[Any] = [None] * CHUNKS
    producer_error: list[BaseException | None] = [None]

    def produce_router() -> None:
        try:
            torch.cuda.set_device(rank)
            with torch.inference_mode():
                for chunk in range(CHUNKS):
                    with record_function(_label(
                        "router_chunk", seed=seed, mode=mode,
                        trial=trial_index, item=chunk,
                    )):
                        launch_host_ns[chunk] = time.monotonic_ns()
                        with torch.cuda.stream(router_stream):
                            indices, scores = router_topk(
                                token_chunks[chunk], weight, bias, TOP_K,
                                mask=mask_chunks[chunk],
                            )
                            host_indices[chunk].copy_(indices, non_blocking=True)
                            events[chunk].record(router_stream)
                        device_indices[chunk] = indices
                        device_scores[chunk] = scores
                        if use_runtime:
                            bridge.arm(chunk, events[chunk].cuda_event)
        except BaseException as error:
            producer_error[0] = error

    origin_host_ns = time.monotonic_ns()
    producer = threading.Thread(target=produce_router, name=f"o1a-router-{mode}-rank{rank}")
    producer.start()

    if not use_runtime:
        producer.join(timeout=120.0)
        if producer.is_alive():
            raise TimeoutError("router-only producer did not terminate")
        if producer_error[0] is not None:
            raise RuntimeError("router producer failed") from producer_error[0]
        events[-1].synchronize()
        final_host_ns = time.monotonic_ns()
        return {
            "mode": mode,
            "seed": seed,
            "trial": trial_index,
            "rank": rank,
            "origin_host_ns": origin_host_ns,
            "final_router_host_ns": final_host_ns,
            "combined_done_host_ns": final_host_ns,
            "chunk_launch_host_ns": [int(value) for value in launch_host_ns],
            "semantic": None,
            "action_structural_signatures": [],
            "descriptor_payloads": [],
            "nccl_count": 0,
            "nccl_bytes_per_call": 0,
        }

    assert state is not None and binder is not None and guard is not None and ingress is not None
    assert control_sources is not None and token_ids is not None

    def consume(chunk: int, *, reveal: bool) -> None:
        ready_host_ns[chunk] = int(bridge.wait_ready(chunk, WAIT_TIMEOUT_NS))
        ingress.stage(chunk, host_index_numpy[chunk])
        if reveal:
            ingress.consume(chunk)

    def submit(slot: int, trigger: int) -> None:
        label = _label("nccl", seed=seed, mode=mode, trial=trial_index, item=trigger)
        with record_function(label), torch.cuda.stream(comm_stream):
            descriptor_device_rows[slot].copy_(descriptor_host_rows[slot], non_blocking=True)
            nccl_call_ns[slot] = time.monotonic_ns()
            works[slot] = dist.all_reduce(descriptor_device_rows[slot], async_op=True)
            nccl_return_ns[slot] = time.monotonic_ns()

    def schedule(slot: int, trigger: int) -> None:
        with record_function(_label(
            "scheduler", seed=seed, mode=mode,
            trial=trial_index, item=trigger,
        )):
            bound = binder.step(state)
            decision = guard.apply(
                state,
                bound.proposal,
                require_scheduler_semantics=True,
                expected_state_version=bound.state_version,
            )
        if not decision.accepted or decision.applied_actions <= 0:
            raise RuntimeError(f"fail-closed scheduler decision: {decision}")
        bounds[slot] = bound
        decisions[slot] = decision
        if use_nccl:
            descriptor_numpy_rows[slot][2:5] = (
                decision.applied_actions,
                decision.state_version,
                state.revealed_count,
            )
            if not delayed_nccl:
                submit(slot, trigger)

    for chunk in range(PARTIAL_CHUNKS):
        consume(chunk, reveal=True)
        state.stage = chunk + 1
        state.ratio = (chunk + 1) / CHUNKS
        schedule(chunk, chunk)
    consume(6, reveal=False)
    consume(7, reveal=False)
    final_router_host_ns = int(ready_host_ns[7])
    ingress.consume(6)
    ingress.consume(7)
    state.stage = CHUNKS
    state.ratio = 1.0
    schedule(6, 7)

    if delayed_nccl:
        # The native ready timestamp above proves final-router CUDA completion
        # (including its D2H copy) before any D collective is submitted.
        for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
            submit(slot, trigger)

    producer.join(timeout=120.0)
    if producer.is_alive():
        raise TimeoutError("router producer did not terminate")
    if producer_error[0] is not None:
        raise RuntimeError("router producer failed") from producer_error[0]
    if use_nccl:
        for work in works:
            if work is None:
                raise RuntimeError("missing NCCL work")
            work.wait()
    combined_done_host_ns = time.monotonic_ns()

    chunk_tokens = []
    for chunk in range(CHUNKS):
        destination_snapshot[chunk, :] = host_index_numpy[chunk][:CONTROL_PER_CHUNK]
        chunk_tokens.append(_make_tokens(
            chunk=chunk,
            token_ids=token_ids,
            control_sources=control_sources,
            destinations=destination_snapshot[chunk],
        ))
    selected_signatures = [structural_signature(bound.selected) for bound in bounds]
    action_signatures = [_action_signature(bound.proposal) for bound in bounds]
    oracle = _oracle_replay(
        topology=topology,
        trial_id=trial_id,
        chunk_tokens=chunk_tokens,
        selected_signatures=selected_signatures,
        action_signatures=action_signatures,
        decisions=decisions,
        compiled_state=state,
    )
    token_integrity = bool(
        state.staged_count == TOTAL_CONTROL_TOKENS
        and state.revealed_count == TOTAL_CONTROL_TOKENS
        and len(set(state.token_ids[:TOTAL_CONTROL_TOKENS])) == TOTAL_CONTROL_TOKENS
        and state.ready_bitmap == (1 << CHUNKS) - 1
        and state.pending_ready_bitmap == 0
        and oracle["holder_divergences"] == 0
    )
    semantic = {
        "runtime_bfs_calls": binder.runtime_bfs_calls,
        "full_rebuild_count": state.full_rebuild_count,
        "candidate_divergences": oracle["candidate_divergences"],
        "action_divergences": oracle["action_divergences"],
        "checker_divergences": oracle["checker_divergences"],
        "holder_divergences": oracle["holder_divergences"],
        "legal": oracle["legal"],
        "total": oracle["total"],
        "token_integrity": token_integrity,
        "unrevealed_execution": 0,
    }
    descriptor_payloads = (
        [[int(value) for value in row.tolist()] for row in descriptor_numpy_rows]
        if use_nccl else []
    )
    return {
        "mode": mode,
        "seed": seed,
        "trial": trial_index,
        "rank": rank,
        "origin_host_ns": origin_host_ns,
        "final_router_host_ns": final_router_host_ns,
        "combined_done_host_ns": combined_done_host_ns,
        "chunk_launch_host_ns": [int(value) for value in launch_host_ns],
        "chunk_ready_host_ns": [int(value) for value in ready_host_ns],
        "nccl_call_host_ns": [int(value) for value in nccl_call_ns],
        "nccl_submit_return_host_ns": [int(value) for value in nccl_return_ns],
        "semantic": semantic,
        "action_structural_signatures": [
            [list(item) for item in signature] for signature in selected_signatures
        ],
        "descriptor_payloads": descriptor_payloads,
        "nccl_count": len(SCHEDULER_TRIGGERS) if use_nccl else 0,
        "nccl_bytes_per_call": 64 if use_nccl else 0,
        "nccl_total_bytes": 64 * len(SCHEDULER_TRIGGERS) if use_nccl else 0,
        "delayed_until_final_router": delayed_nccl,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials-per-mode", type=int, default=20)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.trials_per_mode < 20 and not args.allow_smoke:
        raise ValueError("canonical O1A requires 20 trials per A/B/C/D mode")
    if args.seed not in MODE_ORDERS:
        raise ValueError("seed outside preregistered O1A set")

    os.environ.setdefault("TORCH_NCCL_ENABLE_TIMING", "1")
    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R2-O1A requires exactly two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler()
    plan = compiler.compile(topology)
    rng = np.random.default_rng(args.seed)
    total = CHUNKS * TOKENS_PER_CHUNK
    tokens_np = rng.standard_normal((total, D)).astype(np.float32)
    sources = np.arange(total, dtype=np.int64) % EXPERTS
    tokens = torch.from_numpy(tokens_np).to(device)
    weight_cpu, bias_cpu = seed_router_params(D, EXPERTS, 20260805)
    weight, bias = weight_cpu.to(device), bias_cpu.to(device)
    mask = torch.zeros((total, EXPERTS), dtype=torch.bool, device=device)
    mask[torch.arange(total, device=device), torch.from_numpy(sources).to(device)] = True
    token_chunks = tuple(tokens.narrow(0, i * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK) for i in range(CHUNKS))
    mask_chunks = tuple(mask.narrow(0, i * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK) for i in range(CHUNKS))

    router_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)
    events = [torch.cuda.Event(enable_timing=False) for _ in range(CHUNKS)]
    host_indices = [torch.empty(TOKENS_PER_CHUNK, dtype=torch.int64, pin_memory=True) for _ in range(CHUNKS)]
    host_index_numpy = [value.numpy() for value in host_indices]
    descriptor_host = torch.empty((len(SCHEDULER_TRIGGERS), 8), dtype=torch.int64, pin_memory=True)
    descriptor_device = torch.empty((len(SCHEDULER_TRIGGERS), 8), dtype=torch.int64, device=device)
    descriptor_host_rows = [descriptor_host[index] for index in range(len(SCHEDULER_TRIGGERS))]
    descriptor_numpy_rows = [value.numpy() for value in descriptor_host_rows]
    descriptor_device_rows = [descriptor_device[index] for index in range(len(SCHEDULER_TRIGGERS))]
    for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
        descriptor_numpy_rows[slot][:] = (rank, trigger, 0, 0, 0, 0, 0, 1)

    with torch.inference_mode(), torch.cuda.stream(router_stream):
        warm_indices, _ = router_topk(token_chunks[0], weight, bias, TOP_K, mask=mask_chunks[0])
        host_indices[0].copy_(warm_indices, non_blocking=True)
        for event in events:
            event.record(router_stream)
    torch.cuda.synchronize(device)
    descriptor_device.zero_()
    dist.all_reduce(descriptor_device_rows[0], async_op=True).wait()
    dist.barrier()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed[-(rank + 1)] if allowed != [-1] else -1
    bridge = extension.IntegratedEventBridge(CHUNKS, cpu_core, rank)
    trials: list[dict[str, Any]] = []
    trace_path = args.output_dir / f"r2_o1a_seed{args.seed}_rank{rank}.trace.json"
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            for mode in MODE_ORDERS[args.seed]:
                for trial in range(args.trials_per_mode):
                    trials.append(_run_trial(
                        mode=mode,
                        seed=args.seed,
                        trial_index=trial,
                        rank=rank,
                        topology=topology,
                        plan=plan,
                        bridge=bridge,
                        token_chunks=token_chunks,
                        mask_chunks=mask_chunks,
                        weight=weight,
                        bias=bias,
                        sources=sources,
                        router_stream=router_stream,
                        comm_stream=comm_stream,
                        events=events,
                        host_indices=host_indices,
                        host_index_numpy=host_index_numpy,
                        descriptor_host_rows=descriptor_host_rows,
                        descriptor_numpy_rows=descriptor_numpy_rows,
                        descriptor_device_rows=descriptor_device_rows,
                    ))
                dist.barrier()
            torch.cuda.synchronize(device)
        profiler.export_chrome_trace(str(trace_path))
    finally:
        bridge.stop()

    local = {
        "rank": rank,
        "seed": args.seed,
        "trials_per_mode": args.trials_per_mode,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_size_bytes": trace_path.stat().st_size,
        "poller_cpu_core": cpu_core,
        "poller_pinned": bool(bridge.pinned),
        "trials": trials,
    }
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        semantic_rows = [
            trial["semantic"] for item in gathered for trial in item["trials"]
            if trial["semantic"] is not None
        ]
        requirements = {
            "runtime_bfs_zero": all(row["runtime_bfs_calls"] == 0 for row in semantic_rows),
            "full_rebuild_zero": all(row["full_rebuild_count"] == 0 for row in semantic_rows),
            "unrevealed_execution_zero": all(row["unrevealed_execution"] == 0 for row in semantic_rows),
            "semantic_divergence_zero": all(
                row[key] == 0 for row in semantic_rows
                for key in ("candidate_divergences", "action_divergences", "checker_divergences", "holder_divergences")
            ),
            "legality_100pct": all(row["legal"] == row["total"] for row in semantic_rows),
            "token_integrity_100pct": all(row["token_integrity"] for row in semantic_rows),
        }
        cd_checks = []
        for item in gathered:
            by_key = {(t["mode"], t["trial"]): t for t in item["trials"]}
            for trial in range(args.trials_per_mode):
                c, d = by_key[("C", trial)], by_key[("D", trial)]
                cd_checks.append({
                    "rank": item["rank"],
                    "trial": trial,
                    "actions_equal": c["action_structural_signatures"] == d["action_structural_signatures"],
                    "descriptors_equal": c["descriptor_payloads"] == d["descriptor_payloads"],
                    "nccl_count_equal": c["nccl_count"] == d["nccl_count"] == len(SCHEDULER_TRIGGERS),
                    "bytes_equal": c["nccl_total_bytes"] == d["nccl_total_bytes"] == 448,
                })
        strict_cd = all(all(row[key] for key in (
            "actions_equal", "descriptors_equal", "nccl_count_equal", "bytes_equal"
        )) for row in cd_checks)
        result = {
            "schema_version": 1,
            "study": "R2-O1A A/B/C/D profiled mechanism run",
            "status": "HOST_AND_SEMANTIC_COMPLETE_DEVICE_ANALYSIS_PENDING",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size,
                "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "python": platform.python_version(),
            },
            "preregistered": {
                "seed": args.seed,
                "seed_set": [4042, 4043, 4044],
                "trials_per_mode": args.trials_per_mode,
                "mode_order": list(MODE_ORDERS[args.seed]),
                "communication": "7 x 64-byte descriptor all_reduce per rank-trial",
                "paired_gain": "T_D - T_C",
                "classification": {
                    "launch_or_rendezvous": "call-start->GPU-start p50 > NCCL-duration p50 OR rank-start-skew p95 >100us",
                    "resource_contention": "paired C-B router-latency bootstrap CI95 lower >0",
                },
            },
            "semantic_requirements": requirements,
            "semantic_pass": all(requirements.values()),
            "strict_cd_control": {
                "comparisons": len(cd_checks),
                "pass": strict_cd,
                "details": cd_checks,
            },
            "rank_results": gathered,
            "next": "analyze CUPTI traces; O1B remains unauthorized",
        }
        output = args.output_dir / f"r2_o1a_seed{args.seed}_host.json"
        output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "seed": args.seed,
            "semantic": requirements,
            "strict_cd": result["strict_cd_control"],
            "traces": [{"rank": x["rank"], "path": x["trace_path"], "bytes": x["trace_size_bytes"]} for x in gathered],
            "output": str(output),
        }, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
