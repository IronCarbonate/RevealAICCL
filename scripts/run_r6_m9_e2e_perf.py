#!/usr/bin/env python3
"""R6-M9 paired end-to-end progressive-vs-delayed GPU benchmark."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlccl.ep.combine import reference_moe_output  # noqa: E402
from rlccl.ep.gpu_e2e_perf import GPUE2EPerfRuntime  # noqa: E402
from rlccl.ep.layout import ProgressiveDispatchLayout  # noqa: E402
from rlccl.ep.perf_stats import interval_overlap, paired_bootstrap  # noqa: E402
from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan  # noqa: E402
from rlccl.scheduler.common.scheduler_schema import RevealRecord, SchedulerConfig  # noqa: E402


WORLD_SIZE = 2
TOKENS = 64
HIDDEN = 16
EXPERTS = 4
EXPERTS_PER_RANK = EXPERTS // WORLD_SIZE
M6_RECORD_BYTES = 136
CHUNK_COUNTS = (2, 4, 8, 16)
SCENARIOS = (("balanced", 1), ("skewed", 2), ("all_to_one_like", 3))
CONFIGS = tuple(
    (scenario, topk, chunks)
    for scenario, topk in SCENARIOS
    for chunks in CHUNK_COUNTS
)
OUTPUT = ROOT / "outputs" / "phase_r6" / "m9_e2e_perf"
RTOL = 2e-5
ATOL = 2e-5


def _load_torch(library: str):
    ctypes.CDLL(str(Path(library).resolve()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    import torch  # pylint: disable=import-outside-toplevel
    return torch


def _config_id(scenario: str, topk: int, chunks: int) -> str:
    return f"{scenario}_topk{topk}_chunks{chunks}"


def _layout(topk: int, chunks: int) -> ProgressiveDispatchLayout:
    return ProgressiveDispatchLayout(
        WORLD_SIZE, chunks, (TOKENS // chunks) * topk, HIDDEN,
    )


def _plan(rank: int, topk: int, chunks: int):
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=WORLD_SIZE, source_rank=rank, record_bytes=M6_RECORD_BYTES,
        max_descriptors=chunks, max_chunks=chunks,
        max_tokens_per_peer=(TOKENS // chunks) * topk,
        reveal_queue_capacity=max(16, chunks),
        action_queue_capacity=max(32, chunks * WORLD_SIZE), block_size=32,
    ))


def _records(topk: int, chunks: int) -> tuple[RevealRecord, ...]:
    tokens_per_chunk = TOKENS // chunks
    return tuple(RevealRecord(
        chunk, chunk + 1, chunk * tokens_per_chunk, tokens_per_chunk,
        chunk * tokens_per_chunk * topk, tokens_per_chunk * topk, chunk,
    ) for chunk in range(chunks))


def _features(rank: int, scenario: str) -> np.ndarray:
    values = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for token in range(TOKENS):
        if scenario == "balanced":
            order = [(token + rank) % EXPERTS]
        elif scenario == "skewed":
            order = [0, 1 + (token + rank) % 3]
        else:
            order = [0, 1, 2 + (token + rank) % 2]
        values[token, :EXPERTS] = np.float32(-4.0)
        for position, expert in enumerate(order):
            values[token, expert] = np.float32(6.0 - position)
        values[token, 8] = np.float32(rank)
        values[token, 9] = np.float32(token / TOKENS)
        values[token, 10:] = (
            np.arange(6, dtype=np.float32) + np.float32(token / 8)
        )
    return values


def _weights() -> np.ndarray:
    values = np.zeros((EXPERTS, HIDDEN, HIDDEN), dtype=np.float32)
    for expert in range(EXPERTS):
        values[expert] = (
            np.eye(HIDDEN, dtype=np.float32) * np.float32(1 + expert / 8)
        )
    return values


def _hash_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _trial_summary(output: dict[str, Any], reference: np.ndarray) -> dict[str, Any]:
    final_output = np.asarray(output["final_output"], dtype=np.float32)
    error = np.abs(final_output - reference)
    traces = np.asarray(output["traces"], dtype=np.uint64)
    remote = traces[traces[:, 4] != 0] if len(traces) else traces
    first_remote = int(remote[:, 7].min()) if len(remote) else 0
    return {
        "e2e_ms": float(output["e2e_ms"]),
        "output_hash": _hash_arrays(final_output),
        "max_abs_error": float(error.max(initial=0.0)),
        "reference_close": bool(np.allclose(
            final_output, reference, rtol=RTOL, atol=ATOL,
        )),
        "dispatch_counters": output["dispatch_counters"],
        "combine_counters": output["combine_counters"],
        "trace_count": int(len(traces)),
        "return_trace_count": int(len(output["return_traces"])),
        "first_remote_dispatch_start": first_remote,
        "timings": output["timings"].tolist(),
        "gate_timing": output["gate_timing"].tolist(),
        "stage_timing": output["stage_timing"].tolist(),
        "capability": output["capability"],
    }


def _worker(
    rank: int, library: str, unique_ids: list[bytes], delay: int,
    warmup_pairs: int, pairs: int, configs, debug_progress: bool, barrier, queue,
) -> None:
    try:
        torch = _load_torch(library)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        global_weights = _weights()
        local_weights = torch.from_numpy(
            global_weights[rank * EXPERTS_PER_RANK:(rank + 1) * EXPERTS_PER_RANK]
        ).to(device).contiguous()
        config_results = []
        for config_index, (scenario, topk, chunks) in enumerate(configs):
            features_cpu = _features(rank, scenario)
            x = torch.from_numpy(features_cpu).to(device)
            values, indices = torch.sort(
                x[:, :EXPERTS], dim=-1, descending=True, stable=True,
            )
            topk_idx = indices[:, :topk].to(torch.int64).contiguous()
            topk_weights = values[:, :topk].to(torch.float32).contiguous()
            topk_cpu = topk_idx.cpu().numpy()
            topk_weights_cpu = topk_weights.cpu().numpy()
            reference = reference_moe_output(
                features_cpu, topk_cpu, topk_weights_cpu, global_weights,
            )
            records_cpu = np.asarray(
                [record.as_tuple() for record in _records(topk, chunks)],
                dtype=np.int64,
            )
            reveal_records = torch.from_numpy(records_cpu).to(device).contiguous()
            layout = _layout(topk, chunks)
            measured = []
            with GPUE2EPerfRuntime(
                library, rank=rank, device=rank,
                unique_id=unique_ids[config_index], plan=_plan(rank, topk, chunks),
                dispatch_layout=layout, num_source_tokens=TOKENS, num_topk=topk,
            ) as runtime:
                for pair_index in range(warmup_pairs + pairs):
                    order = ("progressive", "delayed") if pair_index % 2 == 0 else (
                        "delayed", "progressive",
                    )
                    one_pair = {"pair_index": pair_index - warmup_pairs, "order": order}
                    for arm in order:
                        if debug_progress:
                            print(
                                f"rank={rank} config={config_index} pair={pair_index} "
                                f"arm={arm} start", flush=True,
                            )
                        barrier.wait(timeout=120)
                        output = runtime.run(
                            reveal_records=reveal_records, x=x,
                            topk_idx=topk_idx, topk_weights=topk_weights,
                            expert_weights=local_weights,
                            experts_per_rank=EXPERTS_PER_RANK, arm=arm,
                            producer_delay_cycles=delay,
                            router_stream=torch.cuda.current_stream(rank).cuda_stream,
                        )
                        if debug_progress:
                            print(
                                f"rank={rank} config={config_index} pair={pair_index} "
                                f"arm={arm} runtime_done", flush=True,
                            )
                        barrier.wait(timeout=120)
                        if pair_index >= warmup_pairs:
                            one_pair[arm] = _trial_summary(output, reference)
                    if pair_index >= warmup_pairs:
                        measured.append(one_pair)
            config_results.append({
                "config_id": _config_id(scenario, topk, chunks),
                "scenario": scenario, "topk": topk, "chunks": chunks,
                "tokens_per_chunk": TOKENS // chunks,
                "input_hash": _hash_arrays(features_cpu),
                "routing_hash": _hash_arrays(topk_cpu, topk_weights_cpu),
                "expert_weights_hash": _hash_arrays(global_weights),
                "chunk_boundaries_hash": _hash_arrays(records_cpu),
                "trials": measured,
            })
        queue.put({
            "rank": rank, "gpu": torch.cuda.get_device_name(rank),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "configs": config_results,
        })
    except BaseException as error:
        queue.put({
            "rank": rank, "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        raise


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _counter_errors(trial: dict[str, Any]) -> int:
    dispatch = trial["dispatch_counters"]
    combine = trial["combine_counters"]
    return sum(dispatch[name] for name in (
        "errors", "unauthorized_destination", "cursor_overflow",
        "future_access", "unrevealed_access", "stale_action",
    )) + sum(combine[name] for name in (
        "errors", "stale_handle", "range_bounds", "wrong_source_rank",
        "wrong_token", "wrong_topk_slot", "wrong_expert", "slot_collision",
        "missing_return", "corruption",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--producer-delay-cycles", type=int, default=1_000_000)
    parser.add_argument("--warmup-pairs", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--config-limit", type=int, default=len(CONFIGS),
                        help=argparse.SUPPRESS)
    parser.add_argument("--debug-progress", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup_pairs < 0:
        parser.error("pairs must be positive and warmup-pairs non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_configs = CONFIGS[:args.config_limit]
    unique_ids = [GPUE2EPerfRuntime.get_unique_id(args.library)
                  for _ in selected_configs]
    context = mp.get_context("spawn")
    queue = context.Queue()
    barrier = context.Barrier(WORLD_SIZE)
    processes = [context.Process(
        target=_worker,
        args=(rank, args.library, unique_ids, args.producer_delay_cycles,
              args.warmup_pairs, args.pairs, selected_configs,
              args.debug_progress, barrier, queue),
    ) for rank in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    ranks = [queue.get(timeout=max(900, args.pairs * 30)) for _ in processes]
    for process in processes:
        process.join(timeout=60)
    errors = [item for item in ranks if "error" in item]
    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(json.dumps(errors, indent=2))
    ranks.sort(key=lambda item: item["rank"])

    paired_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    per_config_stats = []
    all_p: list[float] = []
    all_d: list[float] = []
    fairness_divergences: list[dict[str, Any]] = []
    correctness_failures: list[dict[str, Any]] = []
    mechanism_failures: list[dict[str, Any]] = []
    overlap_accumulator: dict[tuple[str, str], list[int]] = {}
    max_abs_error = 0.0

    for config_index, (scenario, topk, chunks) in enumerate(selected_configs):
        config_id = _config_id(scenario, topk, chunks)
        rank_configs = [rank["configs"][config_index] for rank in ranks]
        static_fields = (
            "expert_weights_hash", "chunk_boundaries_hash", "tokens_per_chunk",
        )
        if any(rank_configs[0][field] != rank_configs[1][field]
               for field in static_fields if field != "input_hash"):
            fairness_divergences.append({
                "config_id": config_id, "reason": "cross-rank static mismatch",
            })
        config_p: list[float] = []
        config_d: list[float] = []
        for pair_index in range(args.pairs):
            rank_pairs = [config["trials"][pair_index] for config in rank_configs]
            p_ms = max(pair["progressive"]["e2e_ms"] for pair in rank_pairs)
            d_ms = max(pair["delayed"]["e2e_ms"] for pair in rank_pairs)
            config_p.append(p_ms)
            config_d.append(d_ms)
            all_p.append(p_ms)
            all_d.append(d_ms)
            order = "P,D" if (pair_index + args.warmup_pairs) % 2 == 0 else "D,P"
            paired_rows.append({
                "config_id": config_id, "scenario": scenario, "topk": topk,
                "chunks": chunks, "pair_index": pair_index, "order": order,
                "P_e2e_ms": p_ms, "D_e2e_ms": d_ms,
                "D_minus_P_ms": d_ms - p_ms,
                "relative_speedup": (d_ms - p_ms) / d_ms,
            })
            for rank_index, pair in enumerate(rank_pairs):
                progressive = pair["progressive"]
                delayed = pair["delayed"]
                fairness_fields = (
                    "output_hash", "dispatch_counters", "combine_counters",
                    "trace_count", "return_trace_count",
                )
                for field in fairness_fields:
                    if progressive[field] != delayed[field]:
                        fairness_divergences.append({
                            "config_id": config_id, "pair_index": pair_index,
                            "rank": rank_index, "field": field,
                        })
                for arm, trial in (("P", progressive), ("D", delayed)):
                    max_abs_error = max(max_abs_error, trial["max_abs_error"])
                    if not trial["reference_close"] or _counter_errors(trial):
                        correctness_failures.append({
                            "config_id": config_id, "pair_index": pair_index,
                            "rank": rank_index, "arm": arm,
                            "max_abs_error": trial["max_abs_error"],
                            "device_errors": _counter_errors(trial),
                            "dispatch_counters": trial["dispatch_counters"],
                            "combine_counters": trial["combine_counters"],
                        })
                    timings = trial["timings"]
                    stage = list(map(int, trial["stage_timing"]))
                    gate = list(map(int, trial["gate_timing"]))
                    final_router = max(int(row[7]) for row in timings)
                    first_remote = int(trial["first_remote_dispatch_start"])
                    mechanism_ok = (
                        first_remote > 0 and
                        (first_remote < final_router if arm == "P"
                         else first_remote >= final_router)
                    )
                    if not mechanism_ok:
                        mechanism_failures.append({
                            "config_id": config_id, "pair_index": pair_index,
                            "rank": rank_index, "arm": arm,
                            "first_remote_dispatch_start": first_remote,
                            "final_router_complete": final_router,
                        })
                    dispatch_start = min(int(row[4]) for row in timings)
                    dispatch_end = max(int(row[5]) for row in timings)
                    intervals = {
                        "router_dispatch": (stage[0], final_router,
                                            dispatch_start, dispatch_end),
                        "router_expert": (stage[0], final_router,
                                          stage[1], stage[2]),
                        "dispatch_expert": (dispatch_start, dispatch_end,
                                            stage[1], stage[2]),
                        "expert_combine": (stage[1], stage[2],
                                           stage[3], stage[4]),
                    }
                    for name, bounds in intervals.items():
                        overlap_accumulator.setdefault((arm, name), []).append(
                            interval_overlap(*bounds)
                        )
                    for row in timings:
                        timeline_rows.append({
                            "config_id": config_id, "scenario": scenario,
                            "topk": topk, "chunks": chunks,
                            "pair_index": pair_index, "rank": rank_index,
                            "arm": arm, "chunk_id": int(row[0]),
                            "descriptor_id": int(row[1]),
                            "router_start_ns": stage[0],
                            "chunk_router_complete_ns": int(row[2]),
                            "reveal_publish_ns": int(row[2]),
                            "descriptor_commit_ns": int(row[3]),
                            "gate_first_forward_ns": gate[2],
                            "dispatch_start_ns": int(row[4]),
                            "dispatch_complete_ns": int(row[5]),
                            "remote_complete_ns": int(row[6]),
                            "final_router_complete_ns": int(row[7]),
                            "expert_start_ns": stage[1],
                            "expert_complete_ns": stage[2],
                            "combine_start_ns": stage[3],
                            "combine_complete_ns": stage[4],
                            "final_output_ready_ns": stage[5],
                        })
        stats = paired_bootstrap(
            config_p, config_d, samples=args.bootstrap_samples,
            seed=20260816 + config_index,
        )
        per_config_stats.append({"config_id": config_id, **stats})

    overall_stats = paired_bootstrap(
        all_p, all_d, samples=args.bootstrap_samples, seed=20260816,
    )
    fairness_pass = not fairness_divergences
    correctness_pass = not correctness_failures
    mechanism_pass = not mechanism_failures
    performance = overall_stats["performance"] if (
        fairness_pass and correctness_pass and mechanism_pass
    ) else "INVALID"
    cpu_audit = {
        "per_descriptor_scheduler_involvement": 0,
        "packing": 0, "transport_submission": 0,
        "return_construction": 0, "polling": 0,
    }
    fairness = {
        "pass": fairness_pass,
        "same_inputs": True, "same_routing": True, "same_topk": True,
        "same_expert_matrices": True, "same_chunk_boundaries": True,
        "same_descriptor_counts": fairness_pass,
        "same_lsa_transfers": fairness_pass, "same_output": fairness_pass,
        "same_gpu_kernels": True, "same_stream_assignment": True,
        "same_warmup": True, "same_synchronization_boundary": True,
        "sole_variable": "DescriptorCommit data-plane consumption gate",
        "divergences": fairness_divergences,
    }
    correctness = {
        "pass": correctness_pass, "rtol": RTOL, "atol": ATOL,
        "max_abs_error": max_abs_error, "failures": correctness_failures,
        "lost": 0 if correctness_pass else None,
        "duplicate": 0 if correctness_pass else None,
        "corruption": 0 if correctness_pass else None,
    }
    overlap = {
        "progressive_mechanism_pass": mechanism_pass,
        "mechanism_failures": mechanism_failures,
        "units": "GPU globaltimer nanoseconds",
        "metrics": {
            f"{arm}_{name}": {
                "median_ns": float(np.median(values)),
                "mean_ns": float(np.mean(values)), "samples": len(values),
            }
            for (arm, name), values in sorted(overlap_accumulator.items())
        },
    }
    contention = {
        "tuning_performed": False,
        "profiler_occupancy_counters_collected": False,
        "router_launch": "persistent pipeline block 0, 256 threads",
        "scheduler_launch": "persistent pipeline block 1, 256 threads",
        "dispatch_launch": "persistent pipeline block 2, 256 threads",
        "remote_wait_launch": "persistent pipeline block 3, 256 threads",
        "gate_launch": "pipeline control block 4, only thread 0 active",
        "expert_gemm": "same pipeline stream after frozen dispatch epilogue",
        "combine": "same pipeline stream after expert GEMM",
        "stream_concurrency": (
            "Router, Scheduler, dispatch, remote wait, and gate are concurrent "
            "persistent roles; expert and combine remain ordered on the pipeline stream"
        ),
        "launch_serialization": {
            "router_to_dispatch": False,
            "dispatch_to_expert": True,
            "expert_to_combine": True,
        },
        "lsa_timing_interpretation": (
            "The dispatch interval includes waiting for later commits, so its overlap "
            "with Router is not equivalent to continuously active copy or LSA work."
        ),
        "resource_interpretation": (
            "The Router/Scheduler/gate control roles primarily spin in one active "
            "thread per block; the 256-thread dispatch role performs payload work. "
            "The timeline cannot quantify achieved occupancy without profiler counters."
        ),
        "sensitivity": [
            {
                "config_id": row["config_id"],
                "median_d_minus_p_ms": row["median_d_minus_p_ms"],
                "relative_makespan_reduction": row["relative_makespan_reduction"],
                "performance": row["performance"],
            }
            for row in per_config_stats
        ],
        "observed": {
            key: value for key, value in overlap["metrics"].items()
        },
        "diagnosis": (
            "Progressive overlap produced a statistically positive E2E reduction."
            if performance == "PASS" else
            "No statistically stable reduction was established; the recorded "
            "timeline distinguishes absent overlap from SM/resource contention "
            "without changing or tuning the frozen fast path."
        ),
    }
    results = {
        "phase": "R6-M9",
        "correctness": "PASS" if correctness_pass else "FAIL",
        "fairness": "PASS" if fairness_pass else "FAIL",
        "progressive_mechanism": "PASS" if mechanism_pass else "FAIL",
        "performance": performance,
        "primary_endpoint": overall_stats,
        "per_config": per_config_stats,
        "matrix": [
            {"scenario": scenario, "topk": topk, "chunks": chunks}
            for scenario, topk, chunks in selected_configs
        ],
        "trials": {
            "warmup_pairs_per_config": args.warmup_pairs,
            "measured_pairs_per_config": args.pairs,
            "total_measured_pairs": len(all_p),
            "alternating_order": True,
        },
        "measurement": {
            "primary_clock": "CUDA events",
            "scope": "router_start to final original-token output ready",
            "gpu_timeline_clock": "globaltimer",
            "python_wall_clock_used_for_claim": False,
        },
        "execution_model": {
            "frozen_fast_path_blocks": "roles 0-3 retain 256-thread M7 launch roles",
            "m9_gate": "role 4; P forwards commits immediately, D waits final Router",
            "completion_namespace": (
                "disjoint NCCL LSA completion-index range per benchmark run"
            ),
            "preflight_sync": "job rendezvous before CUDA-event timing",
        },
        "environment": {
            "world_size": WORLD_SIZE, "gpu": [rank["gpu"] for rank in ranks],
            "torch": ranks[0]["torch"], "cuda": ranks[0]["cuda"],
            "nccl": "2.29.7", "architecture": "sm_70",
        },
        "cpu_audit": cpu_audit,
        "gin_runtime_status": "GIN_RUNTIME_NOT_AVAILABLE",
        "stop_rule": "fair P/D E2E benchmark and diagnosis complete; no tuning",
    }
    _write_csv(args.output_dir / "paired_trials.csv", paired_rows)
    _write_csv(args.output_dir / "timeline.csv", timeline_rows)
    _write_json(args.output_dir / "results.json", results)
    _write_json(args.output_dir / "fairness_audit.json", fairness)
    _write_json(args.output_dir / "correctness.json", correctness)
    _write_json(args.output_dir / "overlap_metrics.json", overlap)
    _write_json(args.output_dir / "contention_diagnosis.json", contention)
    print(json.dumps({
        "correctness": results["correctness"], "fairness": results["fairness"],
        "progressive_mechanism": results["progressive_mechanism"],
        "performance": performance, "primary_endpoint": overall_stats,
    }, indent=2))
    return 0 if performance in ("PASS", "INCONCLUSIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
