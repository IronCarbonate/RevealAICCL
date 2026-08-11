"""Fail-closed Kineto/CUPTI analysis for R2-O1B bounded transports."""

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
TRANSPORTS = ("T0", "T1", "T2", "T3")
CLEAR_IMPROVEMENT_RATIO = 0.80
LABEL = re.compile(
    r"R2O1B\|kind=(?P<kind>[^|]+)\|seed=(?P<seed>\d+)\|"
    r"mode=(?P<mode>[BCD][0-3]?)\|transport=(?P<transport>T[0-3]|B)\|"
    r"trial=(?P<trial>\d+)\|item=(?P<item>\d+)"
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
        return {
            "count": 0, "p50": None, "p95": None, "p99": None,
            "min": None, "max": None,
        }
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "min": float(array.min()),
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


def overlap_with_intervals(
    comm_intervals: list[tuple[float, float]],
    router_intervals: list[tuple[float, float]],
) -> float:
    return float(sum(
        max(0.0, min(c_end, r_end) - max(c_start, r_start))
        for c_start, c_end in comm_intervals
        for r_start, r_end in router_intervals
    ))


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
    router_streams = sorted({int(event["args"]["stream"]) for event in d2h})
    if len(router_streams) != 2:
        raise ValueError(f"O1B expected default and priority router streams: {router_streams}")
    kernels_by_stream: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        stream = int(event.get("args", {}).get("stream", -1))
        if stream in router_streams and "nccl" not in str(event.get("name", "")).lower():
            kernels_by_stream.setdefault(stream, []).append(event)
    for rows in kernels_by_stream.values():
        rows.sort(key=lambda event: float(event["ts"]))

    previous_end_by_stream = {stream: float("-inf") for stream in router_streams}
    router_rows: list[dict[str, Any]] = []
    for group_index, marker in enumerate(d2h):
        stream = int(marker["args"]["stream"])
        marker_start = float(marker["ts"])
        kernels = [
            event for event in kernels_by_stream[stream]
            if previous_end_by_stream[stream] <= float(event["ts"]) < marker_start
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
            "arm": str(meta["arm"]),
            "transport": meta["transport"],
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
            "router_stream": stream,
            "d2h_end_us": marker_start + float(marker["dur"]),
        })
        previous_end_by_stream[stream] = marker_start + float(marker["dur"])

    annotations = sorted(
        (
            event for event in events
            if event.get("cat") == "user_annotation"
            and LABEL.fullmatch(str(event.get("name", "")))
            and "kind=comm" in str(event.get("name", ""))
        ),
        key=lambda event: float(event["ts"]),
    )
    expected_annotations = sum(meta["arm"] in ("C", "D") for meta in trial_meta) * len(TRIGGERS)
    if len(annotations) != expected_annotations:
        raise ValueError(f"communication annotation mismatch: {len(annotations)} != {expected_annotations}")

    cpu_ops = [event for event in events if event.get("cat") == "cpu_op"]
    kernels_by_external: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") != "kernel":
            continue
        external = event.get("args", {}).get("External id")
        if external is not None:
            kernels_by_external.setdefault(int(external), []).append(event)
    p2p_kernels = sorted(
        (
            event for event in events
            if event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel_SendRecv")
        ),
        key=lambda event: float(event["ts"]),
    )
    p2p_annotations = [
        annotation for annotation in annotations if "transport=T3" in str(annotation["name"])
    ]
    if len(p2p_kernels) != len(p2p_annotations):
        raise ValueError(f"P2P kernel/order mismatch: {len(p2p_kernels)} != {len(p2p_annotations)}")
    p2p_kernel_by_annotation = {
        id(annotation): kernel for annotation, kernel in zip(p2p_annotations, p2p_kernels)
    }
    allreduce_param_sequence = sorted(
        (
            event for event in cpu_ops
            if event.get("args", {}).get("Collective name") == "allreduce"
            and int(event.get("args", {}).get("In msg nelems", -1)) in (2, 8)
        ),
        key=lambda event: float(event["ts"]),
    )
    allreduce_kernel_sequence = sorted(
        (
            event for event in events
            if event.get("cat") == "kernel"
            and str(event.get("name", "")).startswith("ncclDevKernel_AllReduce_Sum_u64")
        ),
        key=lambda event: float(event["ts"]),
    )
    if len(allreduce_param_sequence) != len(allreduce_kernel_sequence):
        raise ValueError(
            f"all-reduce global sequence mismatch: {len(allreduce_param_sequence)} != "
            f"{len(allreduce_kernel_sequence)}"
        )
    ordered_allreduce_by_external: dict[int, dict[str, Any]] = {}
    ordered_allreduce_missing_metadata = 0
    for param, kernel in zip(allreduce_param_sequence, allreduce_kernel_sequence):
        param_external = int(param["args"]["External id"])
        kernel_external = kernel.get("args", {}).get("External id")
        if kernel_external is not None and int(kernel_external) != param_external:
            raise ValueError(
                f"all-reduce global order/external mismatch: CPU={param_external}, "
                f"GPU={kernel_external}"
            )
        if kernel_external is None:
            ordered_allreduce_missing_metadata += 1
        ordered_allreduce_by_external[param_external] = kernel

    router_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"], row["chunk"]): row
        for row in router_rows
    }
    host_trial_lookup = {
        (str(meta["mode"]), int(meta["trial"])): meta for meta in trial_meta
    }
    device_rows: list[dict[str, Any]] = []
    comm_streams: set[int] = set()
    association_failures = 0
    association_failure_details: list[dict[str, Any]] = []
    external_matches = 0
    ordered_allreduce_fallbacks = 0
    ordered_p2p_matches = 0
    for annotation in annotations:
        match = LABEL.fullmatch(str(annotation["name"]))
        assert match is not None
        seed = int(match.group("seed"))
        mode = str(match.group("mode"))
        transport = str(match.group("transport"))
        arm = mode[0]
        trial = int(match.group("trial"))
        trigger = int(match.group("item"))
        ann_start = float(annotation["ts"])
        ann_end = ann_start + float(annotation["dur"])
        enclosed = [
            event for event in cpu_ops
            if ann_start <= float(event["ts"])
            and float(event["ts"]) + float(event["dur"]) <= ann_end
        ]
        comm_kernels: list[dict[str, Any]] = []
        if transport in ("T0", "T1", "T2"):
            expected_nelems = 2 if transport == "T2" else 8
            expected_ops = 4 if transport == "T2" else 1
            param_ops = [
                event for event in enclosed
                if event.get("args", {}).get("Collective name") == "allreduce"
                and int(event.get("args", {}).get("In msg nelems", -1)) == expected_nelems
            ]
            c10d_ops = [
                event for event in enclosed
                if str(event.get("name", "")).startswith("c10d::allreduce")
            ]
            if len(param_ops) != expected_ops or len(c10d_ops) != expected_ops:
                association_failures += 1
                association_failure_details.append({
                    "annotation": str(annotation["name"]),
                    "reason": "CPU communication op count",
                    "param_ops": len(param_ops),
                    "c10d_ops": len(c10d_ops),
                    "expected": expected_ops,
                })
                continue
            for param in param_ops:
                external = int(param["args"]["External id"])
                matches_for_external = [
                    event for event in kernels_by_external.get(external, [])
                    if event.get("args", {}).get("Collective name") == "allreduce"
                    and int(event.get("args", {}).get("In msg nelems", -1)) == expected_nelems
                ]
                if len(matches_for_external) != 1:
                    fallback = ordered_allreduce_by_external.get(external)
                    if (
                        len(matches_for_external) == 0
                        and fallback is not None
                        and fallback.get("args", {}).get("External id") is None
                    ):
                        comm_kernels.append(fallback)
                        ordered_allreduce_fallbacks += 1
                    else:
                        association_failures += 1
                        association_failure_details.append({
                            "annotation": str(annotation["name"]),
                            "reason": "external-id GPU kernel count",
                            "external_id": external,
                            "matched_kernels": len(matches_for_external),
                            "expected_nelems": expected_nelems,
                        })
                        comm_kernels = []
                        break
                else:
                    comm_kernels.append(matches_for_external[0])
                    external_matches += 1
            if not comm_kernels:
                continue
            call_start = min(float(event["ts"]) for event in c10d_ops)
            call_end = max(float(event["ts"]) + float(event["dur"]) for event in c10d_ops)
            association_method = (
                "global_order_missing_CUPTI_external_id_fallback"
                if any(kernel.get("args", {}).get("External id") is None for kernel in comm_kernels)
                else "record_param_comms_external_id"
            )
        elif transport == "T3":
            param_ops = [
                event for event in enclosed
                if event.get("args", {}).get("Collective name") in ("send", "recv")
                and int(event.get("args", {}).get("In msg nelems", -1)) == 8
            ]
            c10d_ops = [
                event for event in enclosed
                if str(event.get("name", "")).startswith(("c10d::send", "c10d::recv"))
            ]
            if len(param_ops) != 2 or len(c10d_ops) != 2:
                association_failures += 1
                association_failure_details.append({
                    "annotation": str(annotation["name"]),
                    "reason": "P2P CPU op count",
                    "param_ops": len(param_ops),
                    "c10d_ops": len(c10d_ops),
                })
                continue
            comm_kernels = [p2p_kernel_by_annotation[id(annotation)]]
            ordered_p2p_matches += 1
            call_start = min(float(event["ts"]) for event in c10d_ops)
            call_end = max(float(event["ts"]) + float(event["dur"]) for event in c10d_ops)
            association_method = "exact_count_NCCL_stream_submission_order"
        else:
            raise AssertionError("unexpected transport")

        comm_kernels.sort(key=lambda event: float(event["ts"]))
        kernel_intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
            for event in comm_kernels
        ]
        gpu_start = min(start for start, _ in kernel_intervals)
        gpu_end = max(end for _, end in kernel_intervals)
        for kernel in comm_kernels:
            comm_streams.add(int(kernel["args"]["stream"]))
        chunk_rows = [
            router_lookup[(seed, rank, mode, trial, chunk)] for chunk in range(CHUNKS)
        ]
        final_end = float(chunk_rows[-1]["gpu_end_us"])
        future_intervals = [
            tuple(interval)
            for row in chunk_rows if int(row["chunk"]) > trigger
            for interval in row["kernel_intervals_us"]
        ]
        actual_overlap = overlap_with_intervals(kernel_intervals, future_intervals)
        if actual_overlap > 0.0:
            execution_class = "concurrent_with_future_router"
        elif gpu_start < final_end:
            execution_class = "started_before_final_without_kernel_coexistence"
        else:
            execution_class = "post_final_or_queued"
        slot = TRIGGERS.index(trigger)
        explicit_host_call_us = (
            int(host_trial_lookup[(mode, trial)]["comm_call_host_ns"][slot]) / 1e3
        )
        device_rows.append({
            "seed": seed,
            "rank": rank,
            "mode": mode,
            "arm": arm,
            "transport": transport,
            "trial": trial,
            "trigger_chunk": trigger,
            "annotation_start_us": ann_start,
            "annotation_end_us": ann_end,
            "explicit_monotonic_host_call_us": explicit_host_call_us,
            "trace_annotation_minus_monotonic_call_us": ann_start - explicit_host_call_us,
            "host_submit_call_start_us": call_start,
            "host_submit_call_end_us": call_end,
            "submit_call_start_to_gpu_start_us": gpu_start - call_start,
            "submit_call_end_to_gpu_start_us": gpu_start - call_end,
            "submit_call_start_to_gpu_end_us": gpu_end - call_start,
            "final_router_gpu_end_us": final_end,
            "comm_gpu_start_us": gpu_start,
            "comm_gpu_end_us": gpu_end,
            "comm_gpu_kernel_sum_duration_us": sum(end - start for start, end in kernel_intervals),
            "comm_gpu_envelope_duration_us": gpu_end - gpu_start,
            "primitive_kernel_count": len(comm_kernels),
            "gpu_start_before_final": gpu_start < final_end,
            "actual_overlap_us": actual_overlap,
            "positive_actual_overlap": actual_overlap > 0.0,
            "execution_class": execution_class,
            "router_stream": int(chunk_rows[0]["router_stream"]),
            "comm_stream": int(comm_kernels[0]["args"]["stream"]),
            "association_method": association_method,
            "kernel_names": [str(event["name"]) for event in comm_kernels],
        })

    if association_failures or len(device_rows) != expected_annotations:
        raise ValueError(
            f"communication association incomplete: rows={len(device_rows)}, "
            f"expected={expected_annotations}, failures={association_failures}, "
            f"details={association_failure_details}"
        )
    if len(comm_streams) != 1:
        raise ValueError(f"NCCL stream is ambiguous: {comm_streams}")
    mode_streams: dict[str, list[int]] = {}
    for meta in trial_meta:
        mode = str(meta["mode"])
        mode_streams.setdefault(mode, sorted({
            int(row["router_stream"]) for row in router_rows if row["mode"] == mode
        }))
    t1_streams = {stream for mode, streams in mode_streams.items() if mode in ("C1", "D1") for stream in streams}
    other_streams = {stream for mode, streams in mode_streams.items() if mode not in ("C1", "D1") for stream in streams}
    if len(t1_streams) != 1 or len(other_streams) != 1 or t1_streams == other_streams:
        raise ValueError(f"T1 router priority stream separation failed: {mode_streams}")

    audit = {
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "trace_size_bytes": trace_path.stat().st_size,
        "router_chunk_delimiters": len(d2h),
        "router_chunks_reconstructed": len(router_rows),
        "communication_annotations": len(annotations),
        "communication_action_groups_matched": len(device_rows),
        "allreduce_kernels_external_id_matched": external_matches,
        "allreduce_global_sequence_length": len(allreduce_kernel_sequence),
        "allreduce_kernels_missing_external_metadata": ordered_allreduce_missing_metadata,
        "allreduce_global_order_fallbacks_used": ordered_allreduce_fallbacks,
        "p2p_kernels_order_matched": ordered_p2p_matches,
        "association_failures": association_failures,
        "association_failure_details": association_failure_details,
        "router_streams": router_streams,
        "T1_router_stream": next(iter(t1_streams)),
        "default_router_stream": next(iter(other_streams)),
        "nccl_stream": next(iter(comm_streams)),
        "timeline_source": "unified Kineto/CUPTI CPU-op, kernel, and gpu_memcpy events",
        "p2p_association_basis": "exact T3 annotation/kernel count and NCCL stream submission order",
    }
    return router_rows, device_rows, audit


def _router_trials(
    all_router: list[dict[str, Any]], seeds: list[int], trials_per_mode: int,
) -> list[dict[str, Any]]:
    output = []
    modes = ("B", "C0", "D0", "C1", "D1", "C2", "D2", "C3", "D3")
    for seed in seeds:
        for rank in range(2):
            for mode in modes:
                for trial in range(trials_per_mode):
                    chunks = sorted(
                        (
                            row for row in all_router
                            if row["seed"] == seed and row["rank"] == rank
                            and row["mode"] == mode and row["trial"] == trial
                        ),
                        key=lambda row: row["chunk"],
                    )
                    if len(chunks) != CHUNKS:
                        raise ValueError(f"incomplete router trial {seed}/{rank}/{mode}/{trial}")
                    output.append({
                        "seed": seed, "rank": rank, "mode": mode,
                        "arm": chunks[0]["arm"], "transport": chunks[0]["transport"],
                        "trial": trial,
                        "router_start_us": chunks[0]["gpu_start_us"],
                        "router_end_us": chunks[-1]["gpu_end_us"],
                        "router_final_gpu_latency_us": chunks[-1]["gpu_end_us"] - chunks[0]["gpu_start_us"],
                        "chunk_gpu_duration_us": [row["gpu_duration_us"] for row in chunks],
                        "router_stream": chunks[0]["router_stream"],
                    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not args.allow_smoke and len(args.run_dir) != 3:
        raise ValueError("canonical O1B requires exactly three seed run directories")

    all_router: list[dict[str, Any]] = []
    all_device: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    host_paths: list[Path] = []
    for run_dir in args.run_dir:
        matches = sorted(run_dir.glob("r2_o1b_seed*_host.json"))
        if len(matches) != 1:
            raise ValueError(f"host artifact count in {run_dir}: {len(matches)}")
        host_path = matches[0]
        host = json.loads(host_path.read_text(encoding="utf-8"))
        hosts.append(host)
        host_paths.append(host_path)
        seed = int(host["preregistered"]["seed"])
        for rank in range(2):
            trace_path = run_dir / f"r2_o1b_seed{seed}_rank{rank}.trace.json"
            router, device, audit = analyze_rank_trace(
                host=host, rank=rank, trace_path=trace_path,
            )
            all_router.extend(router)
            all_device.extend(device)
            audits.append(audit)

    seeds = sorted(int(host["preregistered"]["seed"]) for host in hosts)
    trials_per_mode_set = {int(host["preregistered"]["trials_per_mode"]) for host in hosts}
    if len(trials_per_mode_set) != 1:
        raise ValueError("trial count differs across runs")
    trials_per_mode = next(iter(trials_per_mode_set))
    canonical = seeds == list(SEEDS) and trials_per_mode == 20 and len(hosts) == 3
    if not args.allow_smoke and not canonical:
        raise ValueError("canonical seed/trial preregistration is incomplete")

    router_trials = _router_trials(all_router, seeds, trials_per_mode)
    router_lookup = {
        (row["seed"], row["rank"], row["mode"], row["trial"]): row
        for row in router_trials
    }
    device_by_trial: dict[tuple[int, int, str, int], list[dict[str, Any]]] = {}
    for row in all_device:
        device_by_trial.setdefault(
            (row["seed"], row["rank"], row["mode"], row["trial"]), []
        ).append(row)

    transport_results: dict[str, Any] = {}
    paired_rows_all: list[dict[str, Any]] = []
    for index, transport in enumerate(TRANSPORTS):
        c_mode, d_mode = f"C{index}", f"D{index}"
        system_rows = []
        for seed in seeds:
            for trial in range(trials_per_mode):
                for arm, mode in (("C", c_mode), ("D", d_mode)):
                    rank_routers = [router_lookup[(seed, rank, mode, trial)] for rank in range(2)]
                    rank_comms = [device_by_trial[(seed, rank, mode, trial)] for rank in range(2)]
                    if any(len(rows) != len(TRIGGERS) for rows in rank_comms):
                        raise ValueError("transport trial communication is incomplete")
                    start = min(row["router_start_us"] for row in rank_routers)
                    router_end = max(row["router_end_us"] for row in rank_routers)
                    comm_end = max(row["comm_gpu_end_us"] for rows in rank_comms for row in rows)
                    system_rows.append({
                        "seed": seed, "trial": trial, "transport": transport,
                        "arm": arm,
                        "T_combined_us": max(router_end, comm_end) - start,
                        "communication_completion_from_router_start_us": comm_end - start,
                        "router_completion_from_router_start_us": router_end - start,
                    })
        system_lookup = {
            (row["seed"], row["trial"], row["arm"]): row for row in system_rows
        }
        paired_rows = []
        for seed in seeds:
            for trial in range(trials_per_mode):
                c = system_lookup[(seed, trial, "C")]
                d = system_lookup[(seed, trial, "D")]
                paired_rows.append({
                    "seed": seed, "trial": trial, "transport": transport,
                    "T_C_us": c["T_combined_us"],
                    "T_D_us": d["T_combined_us"],
                    "Delta_us": d["T_combined_us"] - c["T_combined_us"],
                })
        paired_rows_all.extend(paired_rows)
        deltas = [row["Delta_us"] for row in paired_rows]
        per_seed = {}
        for seed in seeds:
            values = [row["Delta_us"] for row in paired_rows if row["seed"] == seed]
            per_seed[str(seed)] = {
                "Delta_us": distribution(values),
                "median_bootstrap_ci95": bootstrap_median(values, 20262000 + seed + index),
                "median_positive": bool(np.median(values) > 0.0),
                "positive_pair_fraction": sum(value > 0.0 for value in values) / len(values),
                "T_C_us": distribution(
                    row["T_combined_us"] for row in system_rows
                    if row["seed"] == seed and row["arm"] == "C"
                ),
                "T_D_us": distribution(
                    row["T_combined_us"] for row in system_rows
                    if row["seed"] == seed and row["arm"] == "D"
                ),
            }
        delta_bootstrap = bootstrap_median(deltas, 20263000 + index)
        primary = {
            "paired_median_positive": bool(np.median(deltas) > 0.0),
            "paired_bootstrap_ci95_lower_positive": bool(delta_bootstrap["ci95_lower_positive"]),
            "three_of_three_seed_medians_positive": bool(
                canonical and all(row["median_positive"] for row in per_seed.values())
            ),
        }
        primary["pass"] = all(primary.values())

        pair_keys = [
            (seed, rank, trial) for seed in seeds for rank in range(2)
            for trial in range(trials_per_mode)
        ]
        c_minus_b = [
            router_lookup[(seed, rank, c_mode, trial)]["router_final_gpu_latency_us"]
            - router_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
            for seed, rank, trial in pair_keys
        ]
        d_minus_b = [
            router_lookup[(seed, rank, d_mode, trial)]["router_final_gpu_latency_us"]
            - router_lookup[(seed, rank, "B", trial)]["router_final_gpu_latency_us"]
            for seed, rank, trial in pair_keys
        ]
        c_rows = [row for row in all_device if row["transport"] == transport and row["arm"] == "C"]
        d_rows = [row for row in all_device if row["transport"] == transport and row["arm"] == "D"]
        c_early = [row for row in c_rows if row["trigger_chunk"] < 7]
        row_lookup = {
            (row["seed"], row["arm"], row["trial"], row["trigger_chunk"], row["rank"]): row
            for row in all_device if row["transport"] == transport
        }
        skew_rows = []
        for seed in seeds:
            for arm in ("C", "D"):
                for trial in range(trials_per_mode):
                    for trigger in TRIGGERS:
                        r0 = row_lookup[(seed, arm, trial, trigger, 0)]
                        r1 = row_lookup[(seed, arm, trial, trigger, 1)]
                        signed = r1["comm_gpu_start_us"] - r0["comm_gpu_start_us"]
                        skew_rows.append({
                            "seed": seed, "arm": arm, "trial": trial,
                            "trigger_chunk": trigger,
                            "rank1_minus_rank0_gpu_start_us": signed,
                            "absolute_gpu_start_skew_us": abs(signed),
                        })
        c_skew = [row for row in skew_rows if row["arm"] == "C"]
        per_trigger = {}
        for trigger in TRIGGERS:
            rows = [row for row in c_rows if row["trigger_chunk"] == trigger]
            per_trigger[str(trigger)] = {
                "events": len(rows),
                "gpu_start_before_final": sum(row["gpu_start_before_final"] for row in rows),
                "positive_coexistence": sum(row["positive_actual_overlap"] for row in rows),
                "positive_overlap_duration_us": distribution(
                    row["actual_overlap_us"] for row in rows if row["positive_actual_overlap"]
                ),
                "execution_class_counts": {
                    label: sum(row["execution_class"] == label for row in rows)
                    for label in (
                        "concurrent_with_future_router",
                        "started_before_final_without_kernel_coexistence",
                        "post_final_or_queued",
                    )
                },
            }

        transport_results[transport] = {
            "primary": primary,
            "combined_makespan": {
                "T_C_us": distribution(row["T_combined_us"] for row in system_rows if row["arm"] == "C"),
                "T_D_us": distribution(row["T_combined_us"] for row in system_rows if row["arm"] == "D"),
                "Delta_us": distribution(deltas),
                "Delta_median_bootstrap_ci95": delta_bootstrap,
                "per_seed": per_seed,
            },
            "router_interference": {
                "C_minus_B_us": distribution(c_minus_b),
                "C_minus_B_median_bootstrap_ci95": bootstrap_median(c_minus_b, 20264000 + index),
                "D_minus_B_us": distribution(d_minus_b),
                "D_minus_B_median_bootstrap_ci95": bootstrap_median(d_minus_b, 20265000 + index),
                "B_router_final_us": distribution(
                    row["router_final_gpu_latency_us"] for row in router_trials if row["mode"] == "B"
                ),
                "C_router_final_us": distribution(
                    row["router_final_gpu_latency_us"] for row in router_trials if row["mode"] == c_mode
                ),
                "D_router_final_us": distribution(
                    row["router_final_gpu_latency_us"] for row in router_trials if row["mode"] == d_mode
                ),
                "C_per_chunk_us": {
                    str(chunk): distribution(
                        row["chunk_gpu_duration_us"][chunk]
                        for row in router_trials if row["mode"] == c_mode
                    ) for chunk in range(CHUNKS)
                },
            },
            "launch_rendezvous": {
                "C_submit_call_start_to_GPU_start_us": distribution(
                    row["submit_call_start_to_gpu_start_us"] for row in c_rows
                ),
                "C_submit_call_start_to_GPU_end_us": distribution(
                    row["submit_call_start_to_gpu_end_us"] for row in c_rows
                ),
                "C_rank_start_absolute_skew_us": distribution(
                    row["absolute_gpu_start_skew_us"] for row in c_skew
                ),
                "D_rank_start_absolute_skew_us": distribution(
                    row["absolute_gpu_start_skew_us"] for row in skew_rows if row["arm"] == "D"
                ),
                "C_kernel_sum_duration_us": distribution(
                    row["comm_gpu_kernel_sum_duration_us"] for row in c_rows
                ),
                "C_kernel_envelope_duration_us": distribution(
                    row["comm_gpu_envelope_duration_us"] for row in c_rows
                ),
                "C_by_rank": {
                    str(rank): {
                        "submit_call_start_to_GPU_start_us": distribution(
                            row["submit_call_start_to_gpu_start_us"] for row in c_rows if row["rank"] == rank
                        ),
                        "kernel_sum_duration_us": distribution(
                            row["comm_gpu_kernel_sum_duration_us"] for row in c_rows if row["rank"] == rank
                        ),
                    } for rank in range(2)
                },
                "rank_start_skew_rows": skew_rows,
            },
            "overlap": {
                "C_GPU_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in c_rows) / len(c_rows),
                "C_early_GPU_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in c_early) / len(c_early),
                "C_actual_coexistence_fraction": sum(row["positive_actual_overlap"] for row in c_rows) / len(c_rows),
                "C_early_actual_coexistence_fraction": sum(row["positive_actual_overlap"] for row in c_early) / len(c_early),
                "C_positive_overlap_duration_us": distribution(
                    row["actual_overlap_us"] for row in c_rows if row["positive_actual_overlap"]
                ),
                "D_GPU_start_before_final_count": sum(row["gpu_start_before_final"] for row in d_rows),
                "per_trigger_C": per_trigger,
            },
            "communication_completion": {
                "C_from_router_start_us": distribution(
                    row["communication_completion_from_router_start_us"]
                    for row in system_rows if row["arm"] == "C"
                ),
                "D_from_router_start_us": distribution(
                    row["communication_completion_from_router_start_us"]
                    for row in system_rows if row["arm"] == "D"
                ),
                "C_action_call_start_to_GPU_end_us": distribution(
                    row["submit_call_start_to_gpu_end_us"] for row in c_rows
                ),
                "D_action_call_start_to_GPU_end_us": distribution(
                    row["submit_call_start_to_gpu_end_us"] for row in d_rows
                ),
            },
            "seed4044_pathological_tail": (
                {
                    "T_C_us": per_seed["4044"]["T_C_us"],
                    "T_D_us": per_seed["4044"]["T_D_us"],
                    "Delta_us": per_seed["4044"]["Delta_us"],
                    "D_rank_start_absolute_skew_us": distribution(
                        row["absolute_gpu_start_skew_us"] for row in skew_rows
                        if row["arm"] == "D" and row["seed"] == 4044
                    ),
                } if 4044 in seeds else None
            ),
            "system_rows": system_rows,
            "paired_rows": paired_rows,
        }

    baseline = transport_results["T0"]
    improvement_fields = {
        "launch_stability": lambda row: row["launch_rendezvous"]["C_submit_call_start_to_GPU_start_us"]["p95"],
        "rank_skew": lambda row: row["launch_rendezvous"]["C_rank_start_absolute_skew_us"]["p95"],
        "router_interference": lambda row: row["router_interference"]["C_minus_B_median_bootstrap_ci95"]["median_us"],
        "seed4044_delayed_tail": lambda row: (
            row["seed4044_pathological_tail"]["T_D_us"]["p99"]
            if row["seed4044_pathological_tail"] is not None else None
        ),
    }
    improvement_summary: dict[str, Any] = {}
    for transport in TRANSPORTS:
        details = {}
        for name, getter in improvement_fields.items():
            base_value = getter(baseline)
            value = getter(transport_results[transport])
            if base_value is None or value is None or base_value <= 0:
                details[name] = {
                    "T0": base_value,
                    "value": value,
                    "ratio_to_T0": None,
                    "clear_ge_20pct_reduction": False,
                    "available": False,
                }
                continue
            details[name] = {
                "T0": float(base_value),
                "value": float(value),
                "ratio_to_T0": float(value / base_value) if base_value != 0 else None,
                "clear_ge_20pct_reduction": bool(value <= CLEAR_IMPROVEMENT_RATIO * base_value),
                "available": True,
            }
        improvement_summary[transport] = {
            "details": details,
            "clear_improvement_count": sum(
                row["clear_ge_20pct_reduction"] for row in details.values()
            ),
            "any_clear_improvement": any(
                row["clear_ge_20pct_reduction"] for row in details.values()
            ),
        }

    eligible = [
        transport for transport in ("T1", "T2", "T3")
        if transport_results[transport]["primary"]["pass"]
        and improvement_summary[transport]["any_clear_improvement"]
    ]
    priority_index = {"T1": 0, "T2": 1, "T3": 2}
    eligible.sort(key=lambda transport: (
        -improvement_summary[transport]["clear_improvement_count"],
        transport_results[transport]["combined_makespan"]["T_C_us"]["p50"],
        priority_index[transport],
    ))
    if eligible:
        recommendation = {
            "decision": "RECOMMEND_INTERVENTION_AS_R3_CANDIDATE_BACKEND_PENDING_SUPERVISOR",
            "transport": eligible[0],
            "eligible_transports": eligible,
        }
    elif transport_results["T0"]["primary"]["pass"]:
        recommendation = {
            "decision": "RETAIN_T0_WITH_LIMITATIONS_PENDING_SUPERVISOR",
            "transport": "T0",
            "eligible_transports": [],
        }
    else:
        recommendation = {
            "decision": "R3_CONTINUES_HOLD_T0_NOT_STABLY_POSITIVE",
            "transport": None,
            "eligible_transports": [],
        }

    semantic_pass = all(host["semantic_pass"] for host in hosts)
    strict_paired = all(host["strict_paired_controls"]["pass"] for host in hosts)
    cross_transport = all(host["cross_transport_semantic_controls"]["pass"] for host in hosts)
    trace_complete = all(
        audit["association_failures"] == 0
        and audit["communication_annotations"] == audit["communication_action_groups_matched"]
        for audit in audits
    )
    clock_rows = {}
    for seed in seeds:
        medians = {}
        for rank in range(2):
            offsets = [
                row["trace_annotation_minus_monotonic_call_us"] for row in all_device
                if row["seed"] == seed and row["rank"] == rank
            ]
            medians[str(rank)] = float(np.median(offsets))
        clock_rows[str(seed)] = {
            "rank0_median_offset_us": medians["0"],
            "rank1_median_offset_us": medians["1"],
            "absolute_rank_median_offset_difference_us": abs(medians["1"] - medians["0"]),
        }
    controls = {
        "semantic_shadow_oracle_pass": semantic_pass,
        "strict_Ck_Dk_controls_pass": strict_paired,
        "cross_transport_action_descriptor_semantics_pass": cross_transport,
        "trace_association_complete": trace_complete,
        "T2_slicing_factor_exactly_4": all(
            host["preregistered"]["slicing_factor"] == 4 for host in hosts
        ),
        "logical_bytes_448_per_rank_trial": all(
            host["preregistered"]["logical_bytes_per_rank_trial"] == 448 for host in hosts
        ),
        "runtime_bfs_full_rebuild_unrevealed_zero": all(
            all(host["semantic_requirements"][key] for key in (
                "runtime_bfs_zero", "full_rebuild_zero", "unrevealed_execution_zero"
            )) for host in hosts
        ),
    }
    if not all(controls.values()):
        raise ValueError(f"O1B fail-closed controls failed: {controls}")

    result = {
        "schema_version": 1,
        "study": "Phase R2-O1B Bounded Transport Interventions",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "O1B_COMPLETE_PENDING_SUPERVISOR" if canonical else "SMOKE_COMPLETE",
        "supervisor_gate": "PENDING",
        "canonical": canonical,
        "seeds": seeds,
        "trials_per_mode_per_rank_per_seed": trials_per_mode,
        "controls": controls,
        "cross_rank_clock_correspondence": {
            "method": "same-host monotonic call markers versus Kineto CPU annotations",
            "per_seed": clock_rows,
            "max_absolute_rank_median_offset_difference_us": max(
                row["absolute_rank_median_offset_difference_us"] for row in clock_rows.values()
            ),
        },
        "preregistered_selection": {
            "clear_improvement_ratio": CLEAR_IMPROVEMENT_RATIO,
            "clear_improvement_definition": "at least 20% reduction versus T0 in launch p95, rank-skew p95, C-B median, or seed4044 delayed T_D p99",
            "tie_break": "most clear improvements; then lowest C combined p50; then T1,T2,T3",
        },
        "transport_results": transport_results,
        "improvement_vs_T0": improvement_summary,
        "selection_recommendation": recommendation,
        "trace_audit": audits,
        "host_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in host_paths
        ],
        "paired_rows": paired_rows_all,
        "device_rows": all_device,
        "router_trial_rows": router_trials,
        "forbidden_work": {
            "fifth_transport": False,
            "real_variable_alltoallv": False,
            "token_packing": False,
            "expert_gemm_combine": False,
            "deepep": False,
            "scheduler_semantics_changed": False,
            "partial_shards_or_checkpoint_changed": False,
            "predictor_robust_adaptive": False,
            "router_artificially_extended": False,
            "communication_bytes_reduced": False,
            "formal_r3_e2e": False,
        },
        "next": "STOP_FOR_R2_O1B_SUPERVISOR_REVIEW; R3 remains unauthorized",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "controls": controls,
        "primary": {
            transport: transport_results[transport]["primary"] for transport in TRANSPORTS
        },
        "combined": {
            transport: transport_results[transport]["combined_makespan"]
            for transport in TRANSPORTS
        },
        "improvement": improvement_summary,
        "recommendation": recommendation,
        "output": str(args.output),
    }, indent=1))


if __name__ == "__main__":
    main()
