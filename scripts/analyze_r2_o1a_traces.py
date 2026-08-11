"""Fail-closed Kineto/CUPTI analysis for R2-O1A A/B/C/D controls."""

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
SEEDS = (4042, 4043, 4044)
MODES = ("A", "B", "C", "D")
LABEL = re.compile(
    r"R2O1A\|kind=(?P<kind>[^|]+)\|seed=(?P<seed>\d+)\|"
    r"mode=(?P<mode>[ABCD])\|trial=(?P<trial>\d+)\|item=(?P<item>\d+)"
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


def bootstrap_median(values: list[float], seed: int) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10000, len(array)))
    estimates = np.median(array[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5], method="linear")
    return {
        "count": int(array.size),
        "median_us": float(np.median(array)),
        "ci95_low_us": float(low),
        "ci95_high_us": float(high),
        "ci95_lower_positive": bool(low > 0.0),
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
    expected_nccl = sum(meta["mode"] in ("C", "D") for meta in trial_meta) * len(TRIGGERS)
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
    host_trial_lookup = {
        (str(meta["mode"]), int(meta["trial"])): meta for meta in trial_meta
    }
    device_rows: list[dict[str, Any]] = []
    nccl_streams: set[int] = set()
    association_failures = 0
    for annotation in annotations:
        match = LABEL.fullmatch(str(annotation["name"]))
        assert match is not None
        seed = int(match.group("seed"))
        mode = str(match.group("mode"))
        trial = int(match.group("trial"))
        trigger = int(match.group("item"))
        ann_start = float(annotation["ts"])
        ann_end = ann_start + float(annotation["dur"])
        enclosed = [
            event for event in cpu_ops
            if ann_start <= float(event["ts"])
            and float(event["ts"]) + float(event["dur"]) <= ann_end
        ]
        comm_ops = [
            event for event in enclosed
            if event.get("args", {}).get("Collective name") == "allreduce"
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
        c10d_ops = [
            event for event in enclosed
            if str(event.get("name", "")).startswith("c10d::allreduce")
        ]
        call_event = min(c10d_ops, key=lambda event: float(event["ts"])) if c10d_ops else comm_ops[0]
        call_start = float(call_event["ts"])
        call_end = call_start + float(call_event["dur"])
        kernel = matched_kernels[0]
        nccl_start = float(kernel["ts"])
        nccl_end = nccl_start + float(kernel["dur"])
        nccl_stream = int(kernel["args"]["stream"])
        nccl_streams.add(nccl_stream)
        chunk_rows = [
            router_lookup[(seed, rank, mode, trial, chunk)] for chunk in range(CHUNKS)
        ]
        final_end = float(chunk_rows[-1]["gpu_end_us"])
        future_intervals = [
            tuple(interval)
            for row in chunk_rows if int(row["chunk"]) > trigger
            for interval in row["kernel_intervals_us"]
        ]
        actual_overlap = overlap_with_intervals(nccl_start, nccl_end, future_intervals)
        slot = TRIGGERS.index(trigger)
        explicit_host_call_us = (
            int(host_trial_lookup[(mode, trial)]["nccl_call_host_ns"][slot]) / 1e3
        )
        if actual_overlap > 0.0:
            execution_class = "concurrent_with_future_router"
        elif nccl_start < final_end:
            execution_class = "started_before_final_without_kernel_coexistence"
        else:
            execution_class = "post_final_or_queued"
        device_rows.append({
            "seed": seed,
            "rank": rank,
            "mode": mode,
            "trial": trial,
            "trigger_chunk": trigger,
            "annotation_start_us": ann_start,
            "annotation_end_us": ann_end,
            "explicit_monotonic_host_call_us": explicit_host_call_us,
            "trace_annotation_minus_monotonic_call_us": ann_start - explicit_host_call_us,
            "host_submit_call_start_us": call_start,
            "host_submit_call_end_us": call_end,
            "submit_call_start_to_gpu_start_us": nccl_start - call_start,
            "submit_call_end_to_gpu_start_us": nccl_start - call_end,
            "annotation_end_to_gpu_start_us": nccl_start - ann_end,
            "final_router_gpu_end_us": final_end,
            "nccl_gpu_start_us": nccl_start,
            "nccl_gpu_end_us": nccl_end,
            "nccl_gpu_duration_us": nccl_end - nccl_start,
            "gpu_start_before_final": nccl_start < final_end,
            "actual_overlap_us": actual_overlap,
            "positive_actual_overlap": actual_overlap > 0.0,
            "execution_class": execution_class,
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
    if expected_nccl and len(nccl_streams) != 1:
        raise ValueError(f"NCCL stream is ambiguous: {nccl_streams}")

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
        "nccl_stream": next(iter(nccl_streams)) if nccl_streams else None,
        "timeline_source": "unified Kineto/CUPTI CPU-op, kernel, and gpu_memcpy events",
    }
    return router_rows, device_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3:
        raise ValueError("O1A requires exactly three preregistered seed run directories")

    all_router: list[dict[str, Any]] = []
    all_device: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    host_paths: list[Path] = []
    for run_dir in args.run_dir:
        matches = sorted(run_dir.glob("r2_o1a_seed*_host.json"))
        if len(matches) != 1:
            raise ValueError(f"host artifact count in {run_dir}: {len(matches)}")
        host_path = matches[0]
        host = json.loads(host_path.read_text(encoding="utf-8"))
        hosts.append(host)
        host_paths.append(host_path)
        seed = int(host["preregistered"]["seed"])
        for rank in range(2):
            trace_path = run_dir / f"r2_o1a_seed{seed}_rank{rank}.trace.json"
            router, device, audit = analyze_rank_trace(host=host, rank=rank, trace_path=trace_path)
            all_router.extend(router)
            all_device.extend(device)
            audits.append(audit)

    if sorted(int(host["preregistered"]["seed"]) for host in hosts) != list(SEEDS):
        raise ValueError("preregistered seed set is incomplete")
    if not all(int(host["preregistered"]["trials_per_mode"]) == 20 for host in hosts):
        raise ValueError("canonical trial count is not 20 per mode")

    router_trials: list[dict[str, Any]] = []
    for seed in SEEDS:
        for rank in range(2):
            for mode in MODES:
                for trial in range(20):
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
                        "router_start_us": chunks[0]["gpu_start_us"],
                        "router_end_us": chunks[-1]["gpu_end_us"],
                        "router_final_gpu_latency_us": chunks[-1]["gpu_end_us"] - chunks[0]["gpu_start_us"],
                        "chunk_gpu_duration_us": [row["gpu_duration_us"] for row in chunks],
                    })

    trial_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"]): row
        for row in router_trials
    }
    device_by_trial: dict[tuple[int, int, str, int], list[dict[str, Any]]] = {}
    for row in all_device:
        device_by_trial.setdefault(
            (row["seed"], row["rank"], row["mode"], row["trial"]), []
        ).append(row)

    makespans: list[dict[str, Any]] = []
    pair_keys = [(seed, rank, trial) for seed in SEEDS for rank in range(2) for trial in range(20)]
    for seed, rank, trial in pair_keys:
        for mode in ("C", "D"):
            router = trial_lookup[(seed, rank, mode, trial)]
            comm = device_by_trial.get((seed, rank, mode, trial), [])
            if len(comm) != len(TRIGGERS):
                raise ValueError("C/D trial has incomplete communication")
            comm_start = min(row["nccl_gpu_start_us"] for row in comm)
            comm_end = max(row["nccl_gpu_end_us"] for row in comm)
            combined_end = max(router["router_end_us"], comm_end)
            makespans.append({
                "seed": seed,
                "rank": rank,
                "mode": mode,
                "trial": trial,
                "T_combined_us": combined_end - router["router_start_us"],
                "communication_first_gpu_start_from_router_start_us": comm_start - router["router_start_us"],
                "communication_completion_from_router_start_us": comm_end - router["router_start_us"],
                "router_latency_us": router["router_final_gpu_latency_us"],
                "actual_overlap_sum_us": sum(row["actual_overlap_us"] for row in comm),
            })
    makespan_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"]): row for row in makespans
    }
    rank_local_paired_rows = []
    for seed, rank, trial in pair_keys:
        c = makespan_lookup[(seed, rank, "C", trial)]
        d = makespan_lookup[(seed, rank, "D", trial)]
        rank_local_paired_rows.append({
            "seed": seed,
            "rank": rank,
            "trial": trial,
            "T_C_us": c["T_combined_us"],
            "T_D_us": d["T_combined_us"],
            "paired_overlap_gain_us": d["T_combined_us"] - c["T_combined_us"],
        })
    rank_local_gains = [row["paired_overlap_gain_us"] for row in rank_local_paired_rows]

    # NCCL is a two-rank operation.  The primary paired makespan therefore uses
    # one distributed-system observation per seed/trial: earliest router start
    # across the two ranks to the last router/NCCL GPU completion across ranks.
    system_makespans: list[dict[str, Any]] = []
    for seed in SEEDS:
        for trial in range(20):
            for mode in ("C", "D"):
                rank_routers = [trial_lookup[(seed, rank, mode, trial)] for rank in range(2)]
                rank_comms = [
                    device_by_trial[(seed, rank, mode, trial)] for rank in range(2)
                ]
                start = min(row["router_start_us"] for row in rank_routers)
                router_end = max(row["router_end_us"] for row in rank_routers)
                comm_end = max(
                    row["nccl_gpu_end_us"] for rows in rank_comms for row in rows
                )
                system_makespans.append({
                    "seed": seed,
                    "mode": mode,
                    "trial": trial,
                    "T_combined_us": max(router_end, comm_end) - start,
                    "communication_completion_from_router_start_us": comm_end - start,
                    "router_completion_from_router_start_us": router_end - start,
                })
    system_lookup = {
        (row["seed"], row["mode"], row["trial"]): row for row in system_makespans
    }
    paired_rows = []
    for seed in SEEDS:
        for trial in range(20):
            c = system_lookup[(seed, "C", trial)]
            d = system_lookup[(seed, "D", trial)]
            paired_rows.append({
                "seed": seed,
                "trial": trial,
                "T_C_us": c["T_combined_us"],
                "T_D_us": d["T_combined_us"],
                "paired_overlap_gain_us": d["T_combined_us"] - c["T_combined_us"],
            })
    gains = [row["paired_overlap_gain_us"] for row in paired_rows]

    final_by_mode = {
        mode: distribution(
            row["router_final_gpu_latency_us"] for row in router_trials if row["mode"] == mode
        ) for mode in MODES
    }
    per_chunk_by_mode = {
        mode: {
            str(chunk): distribution(
                row["chunk_gpu_duration_us"][chunk]
                for row in router_trials if row["mode"] == mode
            ) for chunk in range(CHUNKS)
        } for mode in MODES
    }
    b_minus_a = [
        trial_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
        - trial_lookup[(seed, rank, "A", trial)]["router_final_gpu_latency_us"]
        for seed, rank, trial in pair_keys
    ]
    c_minus_b = [
        trial_lookup[(seed, rank, "C", trial)]["router_final_gpu_latency_us"]
        - trial_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
        for seed, rank, trial in pair_keys
    ]
    d_minus_b = [
        trial_lookup[(seed, rank, "D", trial)]["router_final_gpu_latency_us"]
        - trial_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
        for seed, rank, trial in pair_keys
    ]

    c_rows = [row for row in all_device if row["mode"] == "C"]
    d_rows = [row for row in all_device if row["mode"] == "D"]
    by_rank_mode = {
        mode: {
            str(rank): {
                "submit_call_start_to_gpu_start_us": distribution(
                    row["submit_call_start_to_gpu_start_us"] for row in all_device
                    if row["mode"] == mode and row["rank"] == rank
                ),
                "submit_call_end_to_gpu_start_us": distribution(
                    row["submit_call_end_to_gpu_start_us"] for row in all_device
                    if row["mode"] == mode and row["rank"] == rank
                ),
                "nccl_kernel_duration_us": distribution(
                    row["nccl_gpu_duration_us"] for row in all_device
                    if row["mode"] == mode and row["rank"] == rank
                ),
                "router_final_gpu_latency_us": distribution(
                    row["router_final_gpu_latency_us"] for row in router_trials
                    if row["mode"] == mode and row["rank"] == rank
                ),
                "router_chunk_gpu_duration_us": distribution(
                    duration for row in router_trials
                    if row["mode"] == mode and row["rank"] == rank
                    for duration in row["chunk_gpu_duration_us"]
                ),
            } for rank in range(2)
        } for mode in ("C", "D")
    }

    row_lookup = {
        (row["seed"], row["mode"], row["trial"], row["trigger_chunk"], row["rank"]): row
        for row in all_device
    }
    rank_skews = []
    for seed in SEEDS:
        for mode in ("C", "D"):
            for trial in range(20):
                for trigger in TRIGGERS:
                    r0 = row_lookup[(seed, mode, trial, trigger, 0)]
                    r1 = row_lookup[(seed, mode, trial, trigger, 1)]
                    signed = r1["nccl_gpu_start_us"] - r0["nccl_gpu_start_us"]
                    rank_skews.append({
                        "seed": seed,
                        "mode": mode,
                        "trial": trial,
                        "trigger_chunk": trigger,
                        "rank1_minus_rank0_gpu_start_us": signed,
                        "absolute_gpu_start_skew_us": abs(signed),
                    })

    per_chunk = {}
    for trigger in TRIGGERS:
        rows = [row for row in c_rows if row["trigger_chunk"] == trigger]
        per_chunk[str(trigger)] = {
            "events": len(rows),
            "gpu_start_before_final": sum(row["gpu_start_before_final"] for row in rows),
            "positive_coexistence": sum(row["positive_actual_overlap"] for row in rows),
            "actual_overlap_us": distribution(row["actual_overlap_us"] for row in rows),
            "execution_class_counts": {
                label: sum(row["execution_class"] == label for row in rows)
                for label in (
                    "concurrent_with_future_router",
                    "started_before_final_without_kernel_coexistence",
                    "post_final_or_queued",
                )
            },
        }

    per_seed = {}
    for seed in SEEDS:
        seed_pairs = [row for row in paired_rows if row["seed"] == seed]
        seed_gains = [row["paired_overlap_gain_us"] for row in seed_pairs]
        seed_c = [row for row in c_rows if row["seed"] == seed]
        seed_c_early = [row for row in seed_c if row["trigger_chunk"] < 7]
        per_seed[str(seed)] = {
            "paired_gain_us": distribution(seed_gains),
            "paired_gain_median_bootstrap_ci95": bootstrap_median(seed_gains, seed + 10000),
            "positive_median_gain": bool(np.median(seed_gains) > 0.0),
            "positive_pair_fraction": sum(value > 0.0 for value in seed_gains) / len(seed_gains),
            "C_gpu_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in seed_c) / len(seed_c),
            "C_positive_actual_overlap_fraction": sum(row["positive_actual_overlap"] for row in seed_c) / len(seed_c),
            "C_early_gpu_start_before_final_fraction": sum(
                row["gpu_start_before_final"] for row in seed_c_early
            ) / len(seed_c_early),
            "C_early_positive_actual_overlap_fraction": sum(
                row["positive_actual_overlap"] for row in seed_c_early
            ) / len(seed_c_early),
        }

    semantic_pass = all(host["semantic_pass"] for host in hosts)
    strict_cd = all(host["strict_cd_control"]["pass"] for host in hosts)
    trace_association = all(
        audit["association_failures"] == 0
        and audit["nccl_annotations"] == audit["nccl_kernels_external_id_matched"]
        for audit in audits
    )
    d_pre_final = sum(row["gpu_start_before_final"] for row in d_rows)
    clock_correspondence: dict[str, Any] = {}
    for seed in SEEDS:
        medians = {}
        rank_rows = {}
        for rank in range(2):
            offsets = [
                row["trace_annotation_minus_monotonic_call_us"] for row in all_device
                if row["seed"] == seed and row["rank"] == rank
            ]
            medians[str(rank)] = float(np.median(offsets))
            centered = [value - medians[str(rank)] for value in offsets]
            rank_rows[str(rank)] = {
                "trace_annotation_minus_monotonic_call_us": distribution(offsets),
                "centered_absolute_residual_us": distribution(abs(value) for value in centered),
            }
        clock_correspondence[str(seed)] = {
            "per_rank": rank_rows,
            "rank_median_offset_difference_us": medians["1"] - medians["0"],
            "absolute_rank_median_offset_difference_us": abs(medians["1"] - medians["0"]),
        }
    controls = {
        "semantic_shadow_oracle_pass": semantic_pass,
        "strict_C_D_action_descriptor_count_bytes_equal": strict_cd,
        "trace_external_id_association_complete": trace_association,
        "D_has_zero_NCCL_GPU_start_before_final": d_pre_final == 0,
        "runtime_bfs_full_rebuild_unrevealed_zero": all(
            all(host["semantic_requirements"][key] for key in (
                "runtime_bfs_zero", "full_rebuild_zero", "unrevealed_execution_zero"
            )) for host in hosts
        ),
    }
    if not all(controls.values()):
        raise ValueError(f"O1A fail-closed control failure: {controls}")

    scheduler_interference = bootstrap_median(b_minus_a, 20260812)
    early_nccl_interference = bootstrap_median(c_minus_b, 20260813)
    delayed_nccl_interference = bootstrap_median(d_minus_b, 20260814)
    launch_p50 = distribution(row["submit_call_start_to_gpu_start_us"] for row in c_rows)["p50"]
    duration_p50 = distribution(row["nccl_gpu_duration_us"] for row in c_rows)["p50"]
    skew_p95 = distribution(row["absolute_gpu_start_skew_us"] for row in rank_skews if row["mode"] == "C")["p95"]
    assert isinstance(launch_p50, float) and isinstance(duration_p50, float) and isinstance(skew_p95, float)
    launch_dominated = bool(launch_p50 > duration_p50 or skew_p95 > 100.0)
    contention_dominated = bool(early_nccl_interference["ci95_lower_positive"])
    if launch_dominated and contention_dominated:
        diagnosis = "C_both_launch_rendezvous_and_resource_contention"
    elif launch_dominated:
        diagnosis = "A_launch_or_rendezvous_dominated"
    elif contention_dominated:
        diagnosis = "B_resource_contention_dominated"
    else:
        diagnosis = "neither_preregistered_signal_detected"

    gain_ci = bootstrap_median(gains, 20260815)
    all_three_positive = all(row["positive_median_gain"] for row in per_seed.values())
    result = {
        "schema_version": 1,
        "study": "Phase R2-O1A Device Scheduling Diagnosis",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "O1A_COMPLETE_PENDING_SUPERVISOR",
        "supervisor_gate": "PENDING",
        "seeds": list(SEEDS),
        "trials_per_mode_per_rank_per_seed": 20,
        "paired_distributed_system_trials": len(paired_rows),
        "paired_rank_local_diagnostics": len(rank_local_paired_rows),
        "controls": controls,
        "cross_rank_clock_correspondence": {
            "method": "same-host time.monotonic_ns call markers compared with Kineto CPU annotation timestamps",
            "per_seed": clock_correspondence,
            "max_absolute_rank_median_offset_difference_us": max(
                row["absolute_rank_median_offset_difference_us"]
                for row in clock_correspondence.values()
            ),
        },
        "paired_combined_makespan": {
            "definition": "paired_overlap_gain = T_D - T_C; each sample spans earliest router start to last router/NCCL GPU completion across both ranks",
            "sample_unit": "distributed seed/trial (both ranks), not rank-local",
            "T_C_us": distribution(row["T_combined_us"] for row in system_makespans if row["mode"] == "C"),
            "T_D_us": distribution(row["T_combined_us"] for row in system_makespans if row["mode"] == "D"),
            "paired_gain_us": distribution(gains),
            "paired_gain_median_bootstrap_ci95": gain_ci,
            "per_seed_independent_runs": per_seed,
            "three_of_three_runs_positive_median": all_three_positive,
            "rank_local_diagnostic": {
                "samples": len(rank_local_paired_rows),
                "paired_gain_us": distribution(rank_local_gains),
                "paired_gain_median_bootstrap_ci95": bootstrap_median(rank_local_gains, 20260816),
            },
            "diagnostic_not_O1B_gate": True,
        },
        "router_interference": {
            "router_final_gpu_latency_us": final_by_mode,
            "per_chunk_gpu_latency_us": per_chunk_by_mode,
            "B_minus_A_scheduler_runtime_us": scheduler_interference,
            "C_minus_B_early_NCCL_us": early_nccl_interference,
            "D_minus_B_delayed_NCCL_us": delayed_nccl_interference,
            "C_minus_B_median_percent_of_B": (
                100.0 * float(early_nccl_interference["median_us"]) / float(final_by_mode["B"]["p50"])
            ),
        },
        "communication": {
            "C_rank_local_completion_from_router_start_us": distribution(
                row["communication_completion_from_router_start_us"]
                for row in makespans if row["mode"] == "C"
            ),
            "D_rank_local_completion_from_router_start_us": distribution(
                row["communication_completion_from_router_start_us"]
                for row in makespans if row["mode"] == "D"
            ),
            "C_distributed_completion_from_router_start_us": distribution(
                row["communication_completion_from_router_start_us"]
                for row in system_makespans if row["mode"] == "C"
            ),
            "D_distributed_completion_from_router_start_us": distribution(
                row["communication_completion_from_router_start_us"]
                for row in system_makespans if row["mode"] == "D"
            ),
            "C_actual_overlap_sum_per_trial_us": distribution(
                row["actual_overlap_sum_us"] for row in makespans if row["mode"] == "C"
            ),
            "D_actual_overlap_sum_per_trial_us": distribution(
                row["actual_overlap_sum_us"] for row in makespans if row["mode"] == "D"
            ),
            "C_actual_overlap_per_collective_us": distribution(row["actual_overlap_us"] for row in c_rows),
            "C_positive_only_actual_overlap_per_collective_us": distribution(
                row["actual_overlap_us"] for row in c_rows if row["positive_actual_overlap"]
            ),
            "C_positive_overlap_fraction": sum(row["positive_actual_overlap"] for row in c_rows) / len(c_rows),
            "C_GPU_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in c_rows) / len(c_rows),
            "C_early_GPU_start_before_final_fraction": sum(
                row["gpu_start_before_final"] for row in c_rows if row["trigger_chunk"] < 7
            ) / sum(row["trigger_chunk"] < 7 for row in c_rows),
            "C_early_positive_overlap_fraction": sum(
                row["positive_actual_overlap"] for row in c_rows if row["trigger_chunk"] < 7
            ) / sum(row["trigger_chunk"] < 7 for row in c_rows),
            "D_GPU_start_before_final_count": d_pre_final,
            "per_trigger_chunk_C": per_chunk,
        },
        "launch_rendezvous": {
            "by_rank_and_mode": by_rank_mode,
            "C_submit_call_start_to_GPU_start_us": distribution(
                row["submit_call_start_to_gpu_start_us"] for row in c_rows
            ),
            "C_submit_call_end_to_GPU_start_us": distribution(
                row["submit_call_end_to_gpu_start_us"] for row in c_rows
            ),
            "C_NCCL_kernel_duration_us": distribution(row["nccl_gpu_duration_us"] for row in c_rows),
            "C_rank_start_absolute_skew_us": distribution(
                row["absolute_gpu_start_skew_us"] for row in rank_skews if row["mode"] == "C"
            ),
            "D_rank_start_absolute_skew_us": distribution(
                row["absolute_gpu_start_skew_us"] for row in rank_skews if row["mode"] == "D"
            ),
            "rank_start_skew_rows": rank_skews,
        },
        "diagnosis": {
            "classification": diagnosis,
            "launch_or_rendezvous_signal": launch_dominated,
            "resource_contention_signal": contention_dominated,
            "frozen_rule": {
                "launch_or_rendezvous": "C call-start->GPU-start p50 > C NCCL duration p50 OR C rank-start-skew p95 > 100us",
                "resource_contention": "paired C-B router latency median bootstrap CI95 lower > 0",
            },
            "supervisor_decision_required": True,
        },
        "trace_audit": audits,
        "host_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in host_paths
        ],
        "paired_rows": paired_rows,
        "rank_local_paired_rows": rank_local_paired_rows,
        "device_rows": all_device,
        "router_trial_rows": router_trials,
        "forbidden_work": {
            "O1B_intervention": False,
            "formal_e2e": False,
            "real_variable_alltoallv": False,
            "packing_gemm_combine": False,
            "deepep": False,
            "scheduler_semantics_changed": False,
            "workload_or_chunk_changed": False,
        },
        "next": "STOP_FOR_R2_O1A_SUPERVISOR_REVIEW; O1B remains unauthorized",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "controls": controls,
        "paired": result["paired_combined_makespan"],
        "router_final": result["router_interference"]["router_final_gpu_latency_us"],
        "router_deltas": {
            key: result["router_interference"][key] for key in (
                "B_minus_A_scheduler_runtime_us", "C_minus_B_early_NCCL_us",
                "D_minus_B_delayed_NCCL_us",
            )
        },
        "communication_summary": {
            key: result["communication"][key] for key in (
                "C_distributed_completion_from_router_start_us",
                "D_distributed_completion_from_router_start_us",
                "C_actual_overlap_sum_per_trial_us",
                "C_positive_overlap_fraction", "C_GPU_start_before_final_fraction",
                "D_GPU_start_before_final_count",
            )
        },
        "launch_summary": {
            key: result["launch_rendezvous"][key] for key in (
                "C_submit_call_start_to_GPU_start_us", "C_submit_call_end_to_GPU_start_us",
                "C_NCCL_kernel_duration_us", "C_rank_start_absolute_skew_us",
                "D_rank_start_absolute_skew_us",
            )
        },
        "diagnosis": result["diagnosis"],
        "output": str(args.output),
    }, indent=1))


if __name__ == "__main__":
    main()
