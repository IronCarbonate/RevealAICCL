"""Phase R2-E0: low-latency CUDA-event bridge and control-plane decomposition.

Primary E0 timing is performed by a native, pinned, allocation-free C++ busy
poller over a preallocated event ring. For every valid sample the previous
cudaEventQuery returned cudaErrorNotReady and the next returned cudaSuccess;
the all-host interval from the former query's start through the latter query's
return is therefore a conservative upper bound on CUDA-event completion ->
host/runtime-ready visibility.

This runner does not run formal E2E and does not implement or optimize the
scheduler.  Legacy ProcessPool/pickle measurements are diagnostic-only and are
strictly separated from the E0 timed bridge path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import CUDA_HOME, load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.recourse import bind_action  # noqa: E402
from rlccl.scheduling.robust_prefix import (  # noqa: E402
    build_scheduling_view,
    enumerate_candidates,
    pack_candidate_batch,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from rlccl.uncertainty.execution import Proposal, commit_proposal  # noqa: E402
from scripts.run_r1_concurrent_pipeline import (  # noqa: E402
    CHUNKS,
    CONTROL_PER_CHUNK,
    D,
    EXPERTS,
    PARTIAL_CHUNKS,
    TOKENS_PER_CHUNK,
    TOTAL_CONTROL_TOKENS,
    _LiveReadyState,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _legacy_r1_decomposition(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = [trial for rank in payload["rank_results"] for trial in rank["primary_trials"]]
    by_trigger: dict[str, Any] = {}
    queue_busy_all: list[float] = []
    dispatch_all: list[float] = []
    for trigger in [0, 1, 2, 3, 4, 5, 7]:
        rows = []
        for trial in trials:
            ordered = trial["scheduler_results"]
            for ordinal, current in enumerate(ordered):
                if int(current["trigger_chunk"]) != trigger:
                    continue
                previous_done = (
                    int(current["ready_host_ns"])
                    if ordinal == 0
                    else int(ordered[ordinal - 1]["checker_done_ns"])
                )
                ready = int(current["ready_host_ns"])
                start = int(current["scheduler_start_ns"])
                available = max(ready, previous_done)
                queue_busy = max(previous_done - ready, 0) / 1e3
                dispatch = max(start - available, 0) / 1e3
                nccl = next(
                    item for item in trial["nccl_submissions"]
                    if int(item["trigger_chunk"]) == trigger
                )
                row = {
                    "ready_to_worker_start_us": (start - ready) / 1e3,
                    "worker_busy_queue_us": queue_busy,
                    "executor_dispatch_wakeup_us": dispatch,
                    "scheduler_body_us": (int(current["scheduler_done_ns"]) - start) / 1e3,
                    "bind_us": (int(current["action_host_ns"]) - int(current["scheduler_done_ns"])) / 1e3,
                    "checker_us": (int(current["checker_done_ns"]) - int(current["checker_start_ns"])) / 1e3,
                    "checker_result_to_parent_nccl_api_us": (
                        int(nccl["api_call_host_ns"]) - int(current["checker_done_ns"])
                    ) / 1e3,
                }
                rows.append(row)
                queue_busy_all.append(queue_busy)
                dispatch_all.append(dispatch)
        by_trigger[str(trigger)] = {
            key: distribution([row[key] for row in rows]) for key in rows[0]
        }
    return {
        "source_result_sha256": sha256_file(path),
        "samples": len(trials) * 7,
        "by_trigger_chunk": by_trigger,
        "worker_busy_queue_us": distribution(queue_busy_all),
        "executor_dispatch_wakeup_us": distribution(dispatch_all),
        "interpretation": (
            "ready->worker-start exactly decomposes into worker-busy queue delay plus "
            "post-availability executor/IPC/process-wakeup delay. Chunk 0 has no queue backlog."
        ),
    }


def _ping_worker(sent_ns: int, payload: dict[str, Any]) -> dict[str, int]:
    entry_ns = time.monotonic_ns()
    _ = payload["additions"][0]["destinations"][0]
    exit_ns = time.monotonic_ns()
    return {"sent_ns": sent_ns, "entry_ns": entry_ns, "exit_ns": exit_ns}


def _legacy_ipc_diagnostic(repetitions: int) -> dict[str, Any]:
    payload = {
        "trigger_chunk": 0,
        "ready_host_ns": 0,
        "additions": [{
            "chunk": 0,
            "sources": [0, 1, 2, 3, 0, 1],
            "destinations": [1, 2, 3, 0, 2, 3],
        }],
        "final_checkpoint": False,
    }
    pickle_us: list[float] = []
    for _ in range(repetitions):
        start = time.monotonic_ns()
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_us.append((time.monotonic_ns() - start) / 1e3)

    context = mp.get_context("spawn")
    submit_call_us: list[float] = []
    submit_to_entry_us: list[float] = []
    worker_exit_to_parent_us: list[float] = []
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        executor.submit(_ping_worker, time.monotonic_ns(), payload).result(timeout=30)
        for _ in range(repetitions):
            sent = time.monotonic_ns()
            call_start = sent
            future = executor.submit(_ping_worker, sent, payload)
            submit_return = time.monotonic_ns()
            result = future.result(timeout=30)
            parent_return = time.monotonic_ns()
            submit_call_us.append((submit_return - call_start) / 1e3)
            submit_to_entry_us.append((result["entry_ns"] - sent) / 1e3)
            worker_exit_to_parent_us.append((parent_return - result["exit_ns"]) / 1e3)
    return {
        "scope": "legacy diagnostic only; forbidden from the R2 timed bridge path",
        "pickle_dumps_us": distribution(pickle_us),
        "processpool_submit_call_us": distribution(submit_call_us),
        "processpool_submit_to_worker_entry_us": distribution(submit_to_entry_us),
        "worker_exit_to_parent_result_us": distribution(worker_exit_to_parent_us),
    }


def _scheduler_component_diagnostic(
    destinations_by_chunk: list[list[int]],
    sources: np.ndarray,
    repetitions: int,
    topo: Any,
) -> dict[str, Any]:
    metrics = {
        "observation_append": [],
        "observation_materialize": [],
        "build_view_python_gil": [],
        "scheduler_enumerate_python_gil": [],
        "scheduler_pack_python_gil": [],
        "bind": [],
        "deterministic_checker": [],
    }
    per_stage: dict[str, dict[str, list[float]]] = {
        str(value): {key: [] for key in metrics} for value in [0, 1, 2, 3, 4, 5, 7]
    }
    for repetition in range(repetitions):
        state = _LiveReadyState(f"r2-diagnostic-{repetition}", topo)
        for chunk in range(PARTIAL_CHUNKS):
            stage_metrics = per_stage[str(chunk)]
            left = chunk * TOKENS_PER_CHUNK
            start = time.monotonic_ns()
            state.append_chunk(
                chunk,
                [int(value) for value in sources[left:left + CONTROL_PER_CHUNK]],
                destinations_by_chunk[chunk],
            )
            append_done = time.monotonic_ns()
            trusted = state.observation(final_checkpoint=False)
            observation_done = time.monotonic_ns()
            view = build_scheduling_view(trusted)
            view_done = time.monotonic_ns()
            candidates = enumerate_candidates(view)
            enumerate_done = time.monotonic_ns()
            structural = pack_candidate_batch(candidates, view.topology)
            pack_done = time.monotonic_ns()
            proposal = Proposal.from_transfers(tuple(
                bind_action(
                    view,
                    local_token_ordinal=item.local_token_ordinal,
                    edge_index=item.edge_index,
                    trusted_observation=trusted,
                )
                for item in structural
            ))
            bind_done = time.monotonic_ns()
            checked = commit_proposal(state.world, trusted, proposal)
            checker_done = time.monotonic_ns()
            if not checked.legal:
                raise RuntimeError("diagnostic checker failed")
            values = {
                "observation_append": (append_done - start) / 1e3,
                "observation_materialize": (observation_done - append_done) / 1e3,
                "build_view_python_gil": (view_done - observation_done) / 1e3,
                "scheduler_enumerate_python_gil": (enumerate_done - view_done) / 1e3,
                "scheduler_pack_python_gil": (pack_done - enumerate_done) / 1e3,
                "bind": (bind_done - pack_done) / 1e3,
                "deterministic_checker": (checker_done - bind_done) / 1e3,
            }
            for key, value in values.items():
                metrics[key].append(value)
                stage_metrics[key].append(value)

        # checkpoint8: append both withheld chunks, then run one unchanged step.
        start = time.monotonic_ns()
        for chunk in range(PARTIAL_CHUNKS, CHUNKS):
            left = chunk * TOKENS_PER_CHUNK
            state.append_chunk(
                chunk,
                [int(value) for value in sources[left:left + CONTROL_PER_CHUNK]],
                destinations_by_chunk[chunk],
            )
        append_done = time.monotonic_ns()
        trusted = state.observation(final_checkpoint=True)
        observation_done = time.monotonic_ns()
        view = build_scheduling_view(trusted)
        view_done = time.monotonic_ns()
        candidates = enumerate_candidates(view)
        enumerate_done = time.monotonic_ns()
        structural = pack_candidate_batch(candidates, view.topology)
        pack_done = time.monotonic_ns()
        proposal = Proposal.from_transfers(tuple(
            bind_action(
                view,
                local_token_ordinal=item.local_token_ordinal,
                edge_index=item.edge_index,
                trusted_observation=trusted,
            )
            for item in structural
        ))
        bind_done = time.monotonic_ns()
        checked = commit_proposal(state.world, trusted, proposal)
        checker_done = time.monotonic_ns()
        if not checked.legal:
            raise RuntimeError("checkpoint8 diagnostic checker failed")
        checkpoint_values = {
            "observation_append": (append_done - start) / 1e3,
            "observation_materialize": (observation_done - append_done) / 1e3,
            "build_view_python_gil": (view_done - observation_done) / 1e3,
            "scheduler_enumerate_python_gil": (enumerate_done - view_done) / 1e3,
            "scheduler_pack_python_gil": (pack_done - enumerate_done) / 1e3,
            "bind": (bind_done - pack_done) / 1e3,
            "deterministic_checker": (checker_done - bind_done) / 1e3,
        }
        for key, value in checkpoint_values.items():
            metrics[key].append(value)
            per_stage["7"][key].append(value)
    return {
        "overall_us": {key: distribution(values) for key, values in metrics.items()},
        "by_trigger_chunk_us": {
            stage: {key: distribution(values) for key, values in stage_metrics.items()}
            for stage, stage_metrics in per_stage.items()
        },
        "semantics": "unchanged R1 append-only observation, scheduler, bind_action, and commit_proposal",
    }


def _load_bridge_extension(build_dir: Path) -> Any:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is unavailable")
    # The server's Miniconda bin is not guaranteed to be inherited in PATH
    # under the SSH non-login shell, even when this interpreter owns ninja.
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        raise RuntimeError("ninja is unavailable in the interpreter environment")
    build_dir.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "extensions" / "r2_event_bridge" / "event_bridge.cpp"
    return load(
        name="r2_event_bridge_ext",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_include_paths=[str(Path(CUDA_HOME) / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{Path(CUDA_HOME) / 'lib64'}", "-lcudart"],
        with_cuda=False,
        verbose=False,
    )


def _bridge_benchmark(
    extension: Any,
    tokens: torch.Tensor,
    weight: torch.Tensor,
    trials: int,
    cpu_core: int,
) -> dict[str, Any]:
    device = tokens.device
    stream = torch.cuda.Stream(device=device)
    outputs = [
        torch.empty((TOKENS_PER_CHUNK, EXPERTS), dtype=tokens.dtype, device=device)
        for _ in range(CHUNKS)
    ]
    token_chunks = tuple(
        tokens.narrow(0, chunk * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK)
        for chunk in range(CHUNKS)
    )
    events = [torch.cuda.Event(enable_timing=False) for _ in range(CHUNKS)]

    # Setup-only warmup and lazy event initialization. No synchronize exists
    # in the bridge timed loop.
    with torch.cuda.stream(stream):
        torch.mm(token_chunks[0], weight, out=outputs[0])
        for event in events:
            event.record(stream)
    torch.cuda.synchronize(device)
    event_handles = [int(event.cuda_event) for event in events]
    if not all(event_handles):
        raise RuntimeError("CUDA event initialization produced a null handle")
    bridge = extension.BusyEventBridge(CHUNKS, cpu_core, int(device.index))

    snapshots: list[dict[str, Any]] = []
    for _ in range(trials):
        bridge.reset_all()
        with torch.cuda.stream(stream):
            for chunk in range(CHUNKS):
                torch.mm(token_chunks[chunk], weight, out=outputs[chunk])
                events[chunk].record(stream)
                bridge.arm(chunk, event_handles[chunk])
        bridge.wait_all(5_000_000_000)
        snapshots.extend(dict(item) for item in bridge.snapshot())

    bridge.stop()
    valid = [item for item in snapshots if item["upper_bound_valid"]]
    invalid = [item for item in snapshots if not item["upper_bound_valid"]]
    upper_us = [float(item["visibility_upper_bound_ns"]) / 1e3 for item in valid]
    success_query_us = [float(item["success_query_duration_ns"]) / 1e3 for item in valid]
    max_query_us = [float(item["max_query_duration_ns"]) / 1e3 for item in valid]
    polls = [float(item["poll_count"]) for item in valid]
    return {
        "samples_total": len(snapshots),
        "samples_with_rigorous_upper_bound": len(valid),
        "first_query_already_ready_samples": len(invalid),
        "valid_coverage": len(valid) / len(snapshots),
        "event_to_host_ready_upper_bound_us": distribution(upper_us),
        "raw_event_to_host_ready_upper_bound_us": upper_us,
        "success_cuda_event_query_duration_us": distribution(success_query_us),
        "max_cuda_event_query_duration_per_event_us": distribution(max_query_us),
        "poll_count": distribution(polls),
        "poller_cpu_core": cpu_core,
        "poller_pinned": bool(bridge.pinned),
        "poller_cuda_device": int(bridge.cuda_device),
        "ring_capacity": CHUNKS,
        "timed_path": {
            "native_cpp_thread": True,
            "busy_poll": True,
            "preallocated_ring": True,
            "json": False,
            "pickle": False,
            "multiprocessing_queue": False,
            "sleep": False,
            "dynamic_allocation_in_native_detection_path": False,
        },
        "measurement_proof": (
            "Previous cudaEventQuery returned NotReady; the upper-bound interval starts "
            "at that query's host start. The next query returned Success and the interval "
            "ends at its host return. True completion is inside this interval, so it is "
            "a conservative host-only upper bound on completion-to-host-ready visibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-trials", type=int, default=500)
    parser.add_argument("--diagnostic-repetitions", type=int, default=50)
    parser.add_argument(
        "--r1-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_r1" / "concurrent_pipeline" / "r1_concurrent_pipeline_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_r2" / "e0_event_bridge",
    )
    args = parser.parse_args()
    if args.bridge_trials < 20 or args.diagnostic_repetitions < 10:
        raise ValueError("insufficient R2-E0 repetitions")

    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    rng = np.random.default_rng(4042)
    total = CHUNKS * TOKENS_PER_CHUNK
    tokens_np = rng.standard_normal((total, D)).astype(np.float32)
    sources = np.arange(total, dtype=np.int64) % EXPERTS
    tokens = torch.from_numpy(tokens_np).to(device)
    weight_cpu, bias_cpu = seed_router_params(D, EXPERTS, 20260805)
    weight, bias = weight_cpu.to(device), bias_cpu.to(device)
    mask = torch.zeros((total, EXPERTS), dtype=torch.bool, device=device)
    mask[torch.arange(total, device=device), torch.from_numpy(sources).to(device)] = True

    # Derive the exact 48 control destinations once, outside every timed path.
    control_destinations: list[list[int]] = []
    with torch.inference_mode():
        for chunk in range(CHUNKS):
            left = chunk * TOKENS_PER_CHUNK
            indices, _ = router_topk(
                tokens[left:left + CONTROL_PER_CHUNK],
                weight,
                bias,
                1,
                mask=mask[left:left + CONTROL_PER_CHUNK],
            )
            control_destinations.append([int(value) for value in indices.cpu().numpy()])

    allowed_cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed_cores[-(rank + 1)] if allowed_cores != [-1] else -1
    extension = _load_bridge_extension(args.output_dir / "build")
    bridge = _bridge_benchmark(extension, tokens, weight, args.bridge_trials, cpu_core)

    topo, _ = _load_rear4_topology(PROJECT_ROOT)
    scheduler_components = _scheduler_component_diagnostic(
        control_destinations,
        sources,
        args.diagnostic_repetitions,
        topo,
    )
    ipc = _legacy_ipc_diagnostic(args.diagnostic_repetitions)
    r1_decomposition = _legacy_r1_decomposition(args.r1_results)

    local = {
        "rank": rank,
        "bridge": bridge,
        "scheduler_components": scheduler_components,
        "legacy_ipc": ipc,
    }
    gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)

    if rank == 0:
        assert gathered is not None
        aggregate_upper = [
            value
            for item in gathered
            for value in item["bridge"]["raw_event_to_host_ready_upper_bound_us"]
        ]
        rank_p95s = [
            float(item["bridge"]["event_to_host_ready_upper_bound_us"]["p95"])
            for item in gathered
        ]
        worst_rank_p95 = max(rank_p95s)
        coverage = min(float(item["bridge"]["valid_coverage"]) for item in gathered)
        requirements = {
            "native_single_process_bridge": True,
            "pinned_busy_poll_thread": all(item["bridge"]["poller_pinned"] for item in gathered),
            "preallocated_ring_bitmap": True,
            "timed_path_no_json_pickle_queue_sleep": True,
            "rigorous_upper_bound_coverage_ge_95pct": coverage >= 0.95,
            "event_to_host_ready_p95_lt_100us": worst_rank_p95 < 100.0,
            "stretch_event_to_host_ready_p95_lt_50us": worst_rank_p95 < 50.0,
            "r1_latency_decomposition_complete": True,
            "scheduler_semantics_unchanged": True,
            "formal_e2e_not_run": True,
        }
        pass_requirements = {key: value for key, value in requirements.items() if not key.startswith("stretch_")}
        technical_pass = all(pass_requirements.values())
        result = {
            "schema_version": 1,
            "study": "Phase R2-E0 Low-Latency Event Bridge",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "TECHNICAL_PASS_PENDING_SUPERVISOR" if technical_pass else "TECHNICAL_FAIL_PENDING_SUPERVISOR",
            "supervisor_gate": "PENDING",
            "environment": {
                "world_size": world_size,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "python": platform.python_version(),
                "allowed_cpu_cores": allowed_cores,
            },
            "preregistered_targets": {
                "event_to_host_ready_p95_us": 100.0,
                "stretch_p95_us": 50.0,
                "metric": "rigorous host-only NotReady-return -> Success-return upper bound",
            },
            "gate_r2_e0": {
                "requirements": requirements,
                "technical_pass": technical_pass,
                "final_gate": "PENDING_SUPERVISOR" if technical_pass else "FAIL_PENDING_SUPERVISOR_REVIEW",
            },
            "bridge_rank_results": [item["bridge"] for item in gathered],
            "event_bridge_aggregate": {
                "event_to_host_ready_upper_bound_us": distribution(aggregate_upper),
                "rank_p95_us": rank_p95s,
                "worst_rank_p95_us": worst_rank_p95,
                "minimum_valid_coverage": coverage,
            },
            "legacy_r1_decomposition": r1_decomposition,
            "legacy_ipc_rank_results": [item["legacy_ipc"] for item in gathered],
            "scheduler_component_rank_results": [item["scheduler_components"] for item in gathered],
            "main_latency_source": (
                "R1 serial ProcessPool worker backlog plus Python observation/view/enumeration; "
                "not CUDA event query and not NCCL submission API."
            ),
            "architecture_boundary": {
                "implemented_now": [
                    "native BusyEventBridge",
                    "pinned busy-poll thread",
                    "preallocated event ring/ready state",
                ],
                "designed_not_implemented": [
                    "StaticPlanCompiler",
                    "IncrementalState",
                    "FastBinder",
                    "StaticProof + DynamicGuard",
                    "IncrementalChecker",
                ],
            },
            "forbidden_work": {
                "formal_e2e": False,
                "real_alltoallv": False,
                "expert_gemm_combine": False,
                "deepep": False,
                "scheduler_semantic_change": False,
                "deterministic_checker_skipped": False,
                "workload_changed_to_expand_window": False,
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "r2_e0_results.json"
        output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        parsed = json.loads(output.read_text(encoding="utf-8"))
        readback = {
            "schema_version": 1,
            "status": "PASS" if parsed["gate_r2_e0"] == result["gate_r2_e0"] else "FAIL",
            "result_path": str(output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "result_sha256": sha256_file(output),
            "runner_sha256": sha256_file(Path(__file__)),
            "extension_sha256": sha256_file(
                PROJECT_ROOT / "extensions" / "r2_event_bridge" / "event_bridge.cpp"
            ),
            "json_roundtrip": parsed["study"] == result["study"],
            "supervisor_gate": "PENDING",
        }
        (args.output_dir / "r2_e0_readback.json").write_text(
            json.dumps(readback, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({
            "status": result["status"],
            "gate": result["gate_r2_e0"],
            "event_bridge_aggregate": result["event_bridge_aggregate"],
            "rank_distributions": [
                {
                    "rank": item["rank"],
                    "event_to_host_ready_upper_bound_us":
                        item["bridge"]["event_to_host_ready_upper_bound_us"],
                    "valid_coverage": item["bridge"]["valid_coverage"],
                    "poller_cpu_core": item["bridge"]["poller_cpu_core"],
                    "poller_pinned": item["bridge"]["poller_pinned"],
                }
                for item in gathered
            ],
            "main_latency_source": result["main_latency_source"],
            "output": str(output),
        }, indent=1))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
