"""Fail-closed Kineto/CUPTI trace analysis for R2-O0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np


TRIGGERS = (0, 1, 2, 3, 4, 5, 7)
CHUNKS = 8
PRIMARY_FRACTION = 0.75
LABEL = re.compile(
    r"R2O0\|kind=(?P<kind>[^|]+)\|seed=(?P<seed>\d+)\|"
    r"mode=(?P<mode>[ABC])\|trial=(?P<trial>\d+)\|item=(?P<item>\d+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(array.max()),
    }


def bootstrap_median_delta(values: list[float], seed: int) -> dict[str, float | bool]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10000, len(array)))
    estimates = np.median(array[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5], method="linear")
    return {
        "count": int(array.size),
        "median_delta_us": float(np.median(array)),
        "ci95_low_us": float(low),
        "ci95_high_us": float(high),
        "significant_positive_slowdown": bool(low > 0.0),
    }


def overlap_with_intervals(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals))


def _trace_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents missing from {path}")
    return events


def _rank_host(host: dict[str, Any], rank: int) -> dict[str, Any]:
    matches = [item for item in host["rank_results"] if int(item["rank"]) == rank]
    if len(matches) != 1:
        raise ValueError("rank host metadata is ambiguous")
    return matches[0]


def analyze_rank_trace(
    *, host: dict[str, Any], rank: int, trace_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events = _trace_events(trace_path)
    rank_host = _rank_host(host, rank)
    trial_meta = rank_host["trials"]
    expected_chunks = len(trial_meta) * CHUNKS

    d2h = sorted(
        (
            event for event in events
            if event.get("cat") == "gpu_memcpy"
            and event.get("name") == "Memcpy DtoH (Device -> Pinned)"
            and int(event.get("args", {}).get("bytes", -1)) == 32768
        ),
        key=lambda event: float(event["ts"]),
    )
    if len(d2h) != expected_chunks:
        raise ValueError(f"D2H chunk delimiter mismatch: {len(d2h)} != {expected_chunks}")
    router_streams = {int(event["args"]["stream"]) for event in d2h}
    if len(router_streams) != 1:
        raise ValueError(f"router stream is ambiguous: {router_streams}")
    router_stream = next(iter(router_streams))
    router_kernels = sorted(
        (
            event for event in events
            if event.get("cat") == "kernel"
            and int(event.get("args", {}).get("stream", -1)) == router_stream
            and "nccl" not in str(event.get("name", "")).lower()
        ),
        key=lambda event: float(event["ts"]),
    )

    router_rows: list[dict[str, Any]] = []
    previous_end = float("-inf")
    for group_index, marker in enumerate(d2h):
        marker_start = float(marker["ts"])
        kernels = [
            event for event in router_kernels
            if previous_end <= float(event["ts"]) < marker_start
        ]
        if len(kernels) < 5:
            raise ValueError(f"router chunk {group_index} has only {len(kernels)} kernels")
        meta = trial_meta[group_index // CHUNKS]
        chunk = group_index % CHUNKS
        start = min(float(event["ts"]) for event in kernels)
        end = max(float(event["ts"]) + float(event["dur"]) for event in kernels)
        router_rows.append({
            "seed": int(meta["seed"]),
            "rank": rank,
            "mode": str(meta["mode"]),
            "trial": int(meta["trial"]),
            "chunk": chunk,
            "gpu_start_us": start,
            "gpu_end_us": end,
            "gpu_duration_us": end - start,
            "kernel_count": len(kernels),
            "kernel_intervals_us": [
                [float(event["ts"]), float(event["ts"]) + float(event["dur"])]
                for event in kernels
            ],
            "router_stream": router_stream,
            "d2h_end_us": marker_start + float(marker["dur"]),
        })
        previous_end = marker_start + float(marker["dur"])

    annotations = sorted(
        (
            event for event in events
            if event.get("cat") == "user_annotation"
            and LABEL.fullmatch(str(event.get("name", "")))
            and "kind=nccl" in str(event.get("name", ""))
        ),
        key=lambda event: float(event["ts"]),
    )
    expected_nccl = sum(1 for meta in trial_meta if meta["mode"] == "C") * len(TRIGGERS)
    if len(annotations) != expected_nccl:
        raise ValueError(f"NCCL annotation mismatch: {len(annotations)} != {expected_nccl}")

    cpu_ops = [event for event in events if event.get("cat") == "cpu_op"]
    kernels_by_external: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        external = event.get("args", {}).get("External id")
        if external is not None:
            kernels_by_external.setdefault(int(external), []).append(event)

    router_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"], row["chunk"]): row
        for row in router_rows
    }
    host_c = {
        int(meta["trial"]): meta for meta in trial_meta if meta["mode"] == "C"
    }
    device_rows: list[dict[str, Any]] = []
    nccl_streams: set[int] = set()
    association_failures = 0
    for annotation in annotations:
        match = LABEL.fullmatch(str(annotation["name"]))
        assert match is not None
        seed = int(match.group("seed"))
        trial = int(match.group("trial"))
        trigger = int(match.group("item"))
        ann_start = float(annotation["ts"])
        ann_end = ann_start + float(annotation["dur"])
        comm_ops = [
            event for event in cpu_ops
            if ann_start <= float(event["ts"])
            and float(event["ts"]) + float(event["dur"]) <= ann_end
            and event.get("args", {}).get("Collective name") == "allreduce"
            and int(event.get("args", {}).get("In msg nelems", -1)) == 8
        ]
        if len(comm_ops) != 1:
            association_failures += 1
            continue
        external = int(comm_ops[0]["args"]["External id"])
        matched_kernels = [
            event for event in kernels_by_external.get(external, [])
            if event.get("args", {}).get("Collective name") == "allreduce"
            and int(event.get("args", {}).get("In msg nelems", -1)) == 8
        ]
        if len(matched_kernels) != 1:
            association_failures += 1
            continue
        kernel = matched_kernels[0]
        nccl_start = float(kernel["ts"])
        nccl_end = nccl_start + float(kernel["dur"])
        nccl_stream = int(kernel["args"]["stream"])
        nccl_streams.add(nccl_stream)
        chunk_rows = [
            router_lookup[(seed, rank, "C", trial, chunk)] for chunk in range(CHUNKS)
        ]
        final_end = float(chunk_rows[-1]["gpu_end_us"])
        future_intervals = [
            tuple(interval)
            for row in chunk_rows if int(row["chunk"]) > trigger
            for interval in row["kernel_intervals_us"]
        ]
        actual_overlap = overlap_with_intervals(nccl_start, nccl_end, future_intervals)
        slot = TRIGGERS.index(trigger)
        host_meta = host_c[trial]
        host_submit = int(host_meta["host_submit_return_ns"][slot])
        host_final = int(host_meta["final_router_host_ns"])
        device_rows.append({
            "seed": seed,
            "rank": rank,
            "trial": trial,
            "trigger_chunk": trigger,
            "host_margin_us": (host_final - host_submit) / 1e3,
            "host_submit_before_final": host_submit < host_final,
            "final_router_gpu_end_us": final_end,
            "nccl_gpu_start_us": nccl_start,
            "nccl_gpu_end_us": nccl_end,
            "nccl_gpu_duration_us": nccl_end - nccl_start,
            "gpu_margin_us": final_end - nccl_start,
            "gpu_start_before_final": nccl_start < final_end,
            "actual_overlap_us": actual_overlap,
            "positive_actual_overlap": actual_overlap > 0.0,
            "router_stream": router_stream,
            "nccl_stream": nccl_stream,
            "external_id": external,
            "nccl_kernel_name": str(kernel["name"]),
        })

    if association_failures or len(device_rows) != expected_nccl:
        raise ValueError(
            f"NCCL kernel association incomplete: rows={len(device_rows)}, "
            f"expected={expected_nccl}, failures={association_failures}"
        )
    if len(nccl_streams) != 1:
        raise ValueError(f"NCCL stream is ambiguous: {nccl_streams}")

    h2d_streams = sorted({
        int(event["args"]["stream"])
        for event in events
        if event.get("cat") == "gpu_memcpy"
        and event.get("name") == "Memcpy HtoD (Pinned -> Device)"
        and int(event.get("args", {}).get("bytes", -1)) == 64
    })
    audit = {
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_size_bytes": trace_path.stat().st_size,
        "router_chunk_delimiters": len(d2h),
        "router_chunks_reconstructed": len(router_rows),
        "nccl_annotations": len(annotations),
        "nccl_kernels_external_id_matched": len(device_rows),
        "association_failures": association_failures,
        "router_stream": router_stream,
        "descriptor_h2d_streams": h2d_streams,
        "nccl_stream": next(iter(nccl_streams)),
        "timeline_source": "Kineto/CUPTI kernel and gpu_memcpy events",
    }
    return router_rows, device_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3:
        raise ValueError("O0 requires exactly three preregistered seed run directories")

    all_router: list[dict[str, Any]] = []
    all_device: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    host_paths: list[Path] = []
    for run_dir in args.run_dir:
        matches = sorted(run_dir.glob("r2_o0_seed*_host.json"))
        if len(matches) != 1:
            raise ValueError(f"host artifact count in {run_dir}: {len(matches)}")
        host_path = matches[0]
        host = json.loads(host_path.read_text(encoding="utf-8"))
        hosts.append(host)
        host_paths.append(host_path)
        seed = int(host["preregistered"]["seed"])
        for rank in range(2):
            trace_path = run_dir / f"r2_o0_seed{seed}_rank{rank}.trace.json"
            router, device, audit = analyze_rank_trace(host=host, rank=rank, trace_path=trace_path)
            all_router.extend(router)
            all_device.extend(device)
            audits.append(audit)

    if sorted(int(host["preregistered"]["seed"]) for host in hosts) != [4042, 4043, 4044]:
        raise ValueError("preregistered seed set is incomplete")

    router_trials: list[dict[str, Any]] = []
    for seed in (4042, 4043, 4044):
        for rank in range(2):
            for mode in ("A", "B", "C"):
                trials = sorted({
                    int(row["trial"]) for row in all_router
                    if row["seed"] == seed and row["rank"] == rank and row["mode"] == mode
                })
                for trial in trials:
                    chunks = sorted(
                        (
                            row for row in all_router
                            if row["seed"] == seed and row["rank"] == rank
                            and row["mode"] == mode and row["trial"] == trial
                        ),
                        key=lambda row: row["chunk"],
                    )
                    if len(chunks) != CHUNKS:
                        raise ValueError("router trial has incomplete chunks")
                    router_trials.append({
                        "seed": seed,
                        "rank": rank,
                        "mode": mode,
                        "trial": trial,
                        "router_final_gpu_latency_us": chunks[-1]["gpu_end_us"] - chunks[0]["gpu_start_us"],
                        "chunk_gpu_duration_us": [row["gpu_duration_us"] for row in chunks],
                    })

    final_by_mode = {
        mode: distribution(
            row["router_final_gpu_latency_us"] for row in router_trials if row["mode"] == mode
        ) for mode in ("A", "B", "C")
    }
    per_chunk_by_mode = {
        mode: {
            str(chunk): distribution(
                row["chunk_gpu_duration_us"] [chunk]
                for row in router_trials if row["mode"] == mode
            ) for chunk in range(CHUNKS)
        } for mode in ("A", "B", "C")
    }
    trial_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"]): row
        for row in router_trials
    }
    keys = sorted({
        (row["seed"], row["rank"], row["trial"])
        for row in router_trials if row["mode"] == "A"
    })
    b_minus_a = [
        trial_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
        - trial_lookup[(seed, rank, "A", trial)]["router_final_gpu_latency_us"]
        for seed, rank, trial in keys
    ]
    c_minus_b = [
        trial_lookup[(seed, rank, "C", trial)]["router_final_gpu_latency_us"]
        - trial_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
        for seed, rank, trial in keys
    ]

    overlap_by_trigger = {}
    for trigger in TRIGGERS:
        rows = [row for row in all_device if row["trigger_chunk"] == trigger]
        overlap_by_trigger[str(trigger)] = {
            "eligible": len(rows),
            "gpu_start_before_final": sum(row["gpu_start_before_final"] for row in rows),
            "gpu_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in rows) / len(rows),
            "positive_actual_overlap": sum(row["positive_actual_overlap"] for row in rows),
            "positive_actual_overlap_fraction": sum(row["positive_actual_overlap"] for row in rows) / len(rows),
            "actual_overlap_us": distribution(row["actual_overlap_us"] for row in rows),
        }

    per_seed = {}
    for seed in (4042, 4043, 4044):
        rows = [row for row in all_device if row["seed"] == seed]
        early = [row for row in rows if row["trigger_chunk"] < 7]
        per_rank = {}
        for rank in range(2):
            rank_early = [row for row in early if row["rank"] == rank]
            positive_trials = sorted({
                row["trial"] for row in rank_early if row["positive_actual_overlap"]
            })
            per_rank[str(rank)] = {
                "early_eligible": len(rank_early),
                "positive_actual_overlap_samples": sum(row["positive_actual_overlap"] for row in rank_early),
                "positive_actual_overlap_fraction": sum(row["positive_actual_overlap"] for row in rank_early) / len(rank_early),
                "distinct_trials_with_positive_overlap": len(positive_trials),
            }
        per_seed[str(seed)] = {
            "eligible": len(rows),
            "host_submit_before_final_fraction": sum(row["host_submit_before_final"] for row in rows) / len(rows),
            "gpu_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in rows) / len(rows),
            "early_eligible": len(early),
            "early_gpu_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in early) / len(early),
            "positive_actual_overlap_fraction": sum(row["positive_actual_overlap"] for row in rows) / len(rows),
            "positive_actual_overlap_samples": sum(row["positive_actual_overlap"] for row in rows),
            "per_rank_stability": per_rank,
        }

    semantic_pass = all(host["semantic_pass"] for host in hosts)
    trace_association = all(
        audit["association_failures"] == 0
        and audit["nccl_annotations"] == audit["nccl_kernels_external_id_matched"]
        for audit in audits
    )
    overall_start_fraction = sum(row["gpu_start_before_final"] for row in all_device) / len(all_device)
    overall_overlap_fraction = sum(row["positive_actual_overlap"] for row in all_device) / len(all_device)
    early_device = [row for row in all_device if row["trigger_chunk"] < 7]
    overall_early_start_fraction = (
        sum(row["gpu_start_before_final"] for row in early_device) / len(early_device)
    )
    stable_three_runs = all(
        row["early_gpu_start_before_final_fraction"] >= PRIMARY_FRACTION
        and all(
            rank_row["distinct_trials_with_positive_overlap"] >= 3
            for rank_row in row["per_rank_stability"].values()
        )
        for row in per_seed.values()
    )
    requirements = {
        "cupTI_kernel_timeline_complete": trace_association,
        "not_inferred_from_nccl_api_return": True,
        "semantic_shadow_pass": semantic_pass,
        "early_gpu_start_before_final_fraction_ge_75pct": overall_early_start_fraction >= PRIMARY_FRACTION,
        "three_independent_seed_runs_stable_positive_overlap": stable_three_runs,
        "actual_overlap_observed": any(row["positive_actual_overlap"] for row in all_device),
        "runtime_bfs_full_rebuild_unrevealed_zero": all(
            all(host["semantic_requirements"][key] for key in (
                "runtime_bfs_zero", "full_rebuild_zero", "unrevealed_execution_zero"
            )) for host in hosts
        ),
    }
    technical_pass = all(requirements.values())
    scheduler_interference = bootstrap_median_delta(b_minus_a, 20260810)
    nccl_interference = bootstrap_median_delta(c_minus_b, 20260811)
    result = {
        "schema_version": 1,
        "study": "Phase R2-O0 True Router-Scheduler-NCCL Device Overlap",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "TECHNICAL_PASS_PENDING_SUPERVISOR" if technical_pass else "TECHNICAL_FAIL_PENDING_SUPERVISOR",
        "supervisor_gate": "PENDING",
        "preregistered_primary_fraction": PRIMARY_FRACTION,
        "seeds": [4042, 4043, 4044],
        "trials_per_mode_per_rank_per_seed": 20,
        "eligible_events": len(all_device),
        "gate_r2_o0": {"requirements": requirements, "technical_pass": technical_pass},
        "host_submit_before_final": {
            "count": sum(row["host_submit_before_final"] for row in all_device),
            "total": len(all_device),
            "fraction": sum(row["host_submit_before_final"] for row in all_device) / len(all_device),
            "host_margin_us": distribution(row["host_margin_us"] for row in all_device),
            "diagnostic_only": True,
        },
        "device_overlap": {
            "gpu_start_before_final_count": sum(row["gpu_start_before_final"] for row in all_device),
            "gpu_start_before_final_total": len(all_device),
            "gpu_start_before_final_fraction": overall_start_fraction,
            "early_gpu_start_before_final_count": sum(row["gpu_start_before_final"] for row in early_device),
            "early_gpu_start_before_final_total": len(early_device),
            "early_gpu_start_before_final_fraction": overall_early_start_fraction,
            "positive_actual_overlap_count": sum(row["positive_actual_overlap"] for row in all_device),
            "positive_actual_overlap_fraction": overall_overlap_fraction,
            "gpu_margin_us": distribution(row["gpu_margin_us"] for row in all_device),
            "early_gpu_margin_us": distribution(row["gpu_margin_us"] for row in early_device),
            "actual_overlap_duration_us": distribution(row["actual_overlap_us"] for row in all_device),
            "early_actual_overlap_duration_us": distribution(row["actual_overlap_us"] for row in early_device),
            "positive_only_actual_overlap_duration_us": distribution(
                row["actual_overlap_us"] for row in all_device if row["positive_actual_overlap"]
            ),
            "nccl_gpu_duration_us": distribution(row["nccl_gpu_duration_us"] for row in all_device),
            "per_trigger_chunk": overlap_by_trigger,
            "per_seed_run": per_seed,
        },
        "router_interference": {
            "router_final_gpu_latency_us": final_by_mode,
            "per_chunk_gpu_latency_us": per_chunk_by_mode,
            "scheduler_interference_B_minus_A": scheduler_interference,
            "nccl_induced_interference_C_minus_B": nccl_interference,
            "scheduler_interference_median_percent_of_A": (
                100.0 * scheduler_interference["median_delta_us"] / final_by_mode["A"]["p50"]
            ),
            "nccl_induced_interference_median_percent_of_B": (
                100.0 * nccl_interference["median_delta_us"] / final_by_mode["B"]["p50"]
            ),
            "nccl_significantly_slows_router": nccl_interference["significant_positive_slowdown"],
            "paired_trials": len(keys),
            "latin_square_order_control": {
                "4042": ["A", "B", "C"], "4043": ["B", "C", "A"], "4044": ["C", "A", "B"]
            },
        },
        "trace_audit": audits,
        "host_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in host_paths
        ],
        "raw_device_rows": all_device,
        "router_trial_rows": router_trials,
        "forbidden_work": {
            "formal_e2e": False,
            "real_alltoallv": False,
            "packing_gemm_combine": False,
            "deepep": False,
            "scheduler_optimized": False,
            "workload_or_chunk_changed_to_expand_window": False,
        },
        "next": "STOP_FOR_R2_O0_SUPERVISOR_REVIEW",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "gate": result["gate_r2_o0"],
        "host": result["host_submit_before_final"],
        "device": result["device_overlap"],
        "interference": result["router_interference"],
        "output": str(args.output),
    }, indent=1))


if __name__ == "__main__":
    main()
