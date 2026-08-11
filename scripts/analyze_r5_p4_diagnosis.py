"""Fail-closed Kineto/CUPTI diagnosis for R5-P4 E1 versus D1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


DIAGNOSTIC_SEEDS = (13042, 13142, 13242)
FAMILIES = (
    "balanced", "skewed", "all_to_one_like", "zero_sized_pair",
    "multiple_progressive_shards",
)
ARMS = ("E1", "D1")
TRIGGERS = 7
FAMILY_CHUNK_SIZES = {
    "balanced": (512,) * 8,
    "skewed": (512,) * 8,
    "all_to_one_like": (512,) * 8,
    "zero_sized_pair": (512,) * 8,
    "multiple_progressive_shards": (128, 256, 384, 512, 640, 768, 512, 896),
}
LABEL = re.compile(
    r"R5P4\|kind=(?P<kind>router|count|payload)\|seed=(?P<seed>\d+)\|"
    r"family=(?P<family>\d+)\|job=(?P<job>\d+)\|arm=(?P<arm>E1|D1)\|item=(?P<item>\d+)"
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
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(array.size), "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "max": float(array.max()),
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents missing: {path}")
    return events


def _kernel_association(
    annotation: dict[str, Any], cpu_ops: list[dict[str, Any]],
    kernels_by_external: dict[int, list[dict[str, Any]]], *, nccl: bool,
) -> list[dict[str, Any]]:
    left = float(annotation["ts"])
    right = left + float(annotation["dur"])
    enclosed = [
        event for event in cpu_ops
        if left <= float(event.get("ts", -1))
        and float(event.get("ts", -1)) + float(event.get("dur", 0)) <= right
        and event.get("args", {}).get("External id") is not None
    ]
    external_ids = {int(event["args"]["External id"]) for event in enclosed}
    matched = [event for external in external_ids for event in kernels_by_external.get(external, [])]
    if nccl:
        matched = [event for event in matched if "nccl" in str(event.get("name", "")).lower()]
    else:
        matched = [event for event in matched if "nccl" not in str(event.get("name", "")).lower()]
    unique = {
        (float(event["ts"]), float(event["dur"]), str(event["name"]),
         int(event.get("args", {}).get("stream", -1))): event
        for event in matched
    }
    return sorted(unique.values(), key=lambda event: float(event["ts"]))


def analyze_rank(host: dict[str, Any], rank: int, trace_path: Path) -> dict[str, Any]:
    events = _load_events(trace_path)
    annotations = [
        event for event in events
        if event.get("cat") == "user_annotation" and LABEL.fullmatch(str(event.get("name", "")))
    ]
    jobs = int(host["frozen_protocol"]["jobs_per_family"])
    families = len(host["frozen_protocol"]["families"])
    expected = {
        "router": families * jobs * len(ARMS) * 8,
        "count": families * jobs * len(ARMS) * TRIGGERS,
        "payload": families * jobs * len(ARMS) * TRIGGERS,
    }
    cpu_ops = [event for event in events if event.get("cat") == "cpu_op"]
    kernels_by_external: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") == "kernel" and event.get("args", {}).get("External id") is not None:
            kernels_by_external.setdefault(int(event["args"]["External id"]), []).append(event)

    rank_host = next(value for value in host["rank_results"] if int(value["rank"]) == rank)
    arm_lookup = {
        (int(pair["family_index"]), int(pair["job"]), arm): pair["arms"][arm]
        for pair in rank_host["pairs"] for arm in ARMS
    }
    counts = {key: 0 for key in expected}
    router_rows, comm_rows = [], []
    failures = []
    seen_labels: set[str] = set()
    for annotation in annotations:
        label = str(annotation["name"])
        if label in seen_labels:
            failures.append(f"duplicate annotation {label}")
            continue
        seen_labels.add(label)
        match = LABEL.fullmatch(label)
        assert match is not None
        kind = match.group("kind")
        counts[kind] += 1
        family_index = int(match.group("family")); job = int(match.group("job"))
        arm = match.group("arm"); item = int(match.group("item"))
        kernels = _kernel_association(annotation, cpu_ops, kernels_by_external, nccl=kind != "router")
        if not kernels and kind == "router":
            # This PyTorch build does not propagate record_function External IDs
            # into the independently created router producer stream. The
            # fail-closed pinned-D2H reconstruction below handles this case.
            continue
        if not kernels:
            failures.append(f"no kernels for {label}")
            continue
        intervals = [
            (float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in kernels
        ]
        start = min(left for left, _ in intervals); end = max(right for _, right in intervals)
        row = {
            "seed": int(match.group("seed")), "rank": rank,
            "family": FAMILIES[family_index], "family_index": family_index,
            "job": job, "arm": arm, "item": item,
            "annotation_start_us": float(annotation["ts"]),
            "annotation_end_us": float(annotation["ts"]) + float(annotation["dur"]),
            "gpu_start_us": start, "gpu_end_us": end,
            "gpu_active_us": sum(right - left for left, right in intervals),
            "gpu_envelope_us": end - start,
            "streams": sorted({int(event.get("args", {}).get("stream", -1)) for event in kernels}),
            "kernel_count": len(kernels), "kernel_intervals_us": intervals,
            "kernel_names": sorted({str(event.get("name", "")) for event in kernels}),
        }
        if kind == "router":
            router_rows.append(row)
        else:
            descriptor = arm_lookup[(family_index, job, arm)]["forward_descriptors"][item]
            communication = descriptor["communication"]
            row.update({
                "kind": kind,
                "call_start_to_gpu_start_us": start - float(annotation["ts"]),
                "range_end_to_gpu_start_us": start - (
                    float(annotation["ts"]) + float(annotation["dur"])
                ),
                "descriptor_ready_host_ns": int(descriptor["descriptor_ready_host_ns"]),
                "communicate_enter_host_ns": int(descriptor["communicate_enter_host_ns"]),
                "count_start_host_ns": int(communication["count_start_host_ns"]),
                "count_visible_host_ns": int(communication["count_visible_host_ns"]),
                "count_wait_us": float(communication["count_wait_us"]),
                "count_event_gpu_us": float(communication["count_gpu_us"]),
                "payload_call_host_ns": int(communication["payload_call_host_ns"]),
                "payload_submit_return_host_ns": int(communication["payload_submit_return_host_ns"]),
                "payload_complete_host_ns": int(communication["payload_complete_host_ns"]),
            })
            comm_rows.append(row)

    router_association = "record_function_external_id"
    router_annotation_count = counts["router"]
    if not router_rows:
        execution_order = []
        for pair in rank_host["pairs"]:
            family = str(pair["family"])
            for arm in pair["arm_order"]:
                for chunk, size in enumerate(FAMILY_CHUNK_SIZES[family]):
                    execution_order.append((
                        int(pair["family_index"]), int(pair["job"]), str(arm),
                        chunk, int(size) * 8,
                    ))
        expected_router_bytes = {value[4] for value in execution_order}
        d2h = sorted(
            (
                event for event in events
                if event.get("cat") == "gpu_memcpy"
                and event.get("name") == "Memcpy DtoH (Device -> Pinned)"
                and int(event.get("args", {}).get("bytes", -1)) in expected_router_bytes
            ),
            key=lambda event: float(event["ts"]),
        )
        if len(d2h) != expected["router"]:
            raise ValueError(
                f"router D2H delimiter mismatch rank{rank}: {len(d2h)} != {expected['router']}"
            )
        streams = {int(event.get("args", {}).get("stream", -1)) for event in d2h}
        if len(streams) != 1:
            raise ValueError(f"router D2H stream ambiguity rank{rank}: {streams}")
        router_stream = next(iter(streams))
        candidate_kernels = sorted(
            (
                event for event in events
                if event.get("cat") == "kernel"
                and int(event.get("args", {}).get("stream", -1)) == router_stream
                and "nccl" not in str(event.get("name", "")).lower()
            ),
            key=lambda event: float(event["ts"]),
        )
        previous_end = float("-inf")
        seed = int(host["frozen_protocol"]["seed"])
        for marker, meta in zip(d2h, execution_order, strict=True):
            family_index, job, arm, chunk, expected_bytes = meta
            actual_bytes = int(marker.get("args", {}).get("bytes", -1))
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"router D2H byte mismatch rank{rank}: {actual_bytes} != {expected_bytes}"
                )
            marker_start = float(marker["ts"])
            kernels = [
                event for event in candidate_kernels
                if previous_end <= float(event["ts"]) < marker_start
            ]
            if not kernels:
                raise ValueError(f"router chunk has no kernels rank{rank}: {meta}")
            intervals = [
                (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
                for event in kernels
            ]
            start = min(left for left, _ in intervals)
            end = max(right for _, right in intervals)
            router_rows.append({
                "seed": seed, "rank": rank,
                "family": FAMILIES[family_index], "family_index": family_index,
                "job": job, "arm": arm, "item": chunk,
                "annotation_start_us": None, "annotation_end_us": None,
                "gpu_start_us": start, "gpu_end_us": end,
                "gpu_active_us": sum(right - left for left, right in intervals),
                "gpu_envelope_us": end - start, "streams": [router_stream],
                "kernel_count": len(kernels), "kernel_intervals_us": intervals,
                "kernel_names": sorted({str(event.get("name", "")) for event in kernels}),
                "association": "pinned_d2h_chunk_delimiter",
            })
            previous_end = marker_start + float(marker["dur"])
        router_association = "pinned_d2h_chunk_delimiter"
        counts["router"] = len(router_rows)

    if counts != expected or failures or len(router_rows) != expected["router"]:
        raise ValueError(
            f"fail-closed trace association rank{rank}: counts={counts}, expected={expected}, "
            f"router_rows={len(router_rows)}, failures={failures[:8]}"
        )
    return {
        "rank": rank, "router": router_rows, "communication": comm_rows,
        "audit": {
            "trace": str(trace_path), "sha256": sha256_file(trace_path),
            "size_bytes": trace_path.stat().st_size, "annotation_counts": counts,
            "association_failures": 0, "router_association": router_association,
            "router_annotation_count": router_annotation_count,
        },
    }


def _interval_overlap(intervals_a: list[tuple[float, float]], intervals_b: list[tuple[float, float]]) -> float:
    return float(sum(
        max(0.0, min(a1, b1) - max(a0, b0))
        for a0, a1 in intervals_a for b0, b1 in intervals_b
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not args.allow_smoke and len(args.run_dir) != 3:
        raise ValueError("canonical R5-P4 requires three seed directories")

    hosts, rank_analyses = [], []
    for run_dir in args.run_dir:
        host_paths = list(run_dir.glob("r5_p4_seed*_host.json"))
        if len(host_paths) != 1:
            raise ValueError(f"host artifact ambiguity: {run_dir}")
        host = json.loads(host_paths[0].read_text(encoding="utf-8"))
        hosts.append(host)
        seed = int(host["frozen_protocol"]["seed"])
        for rank in (0, 1):
            rank_analyses.append(analyze_rank(
                host, rank, run_dir / f"r5_p4_seed{seed}_rank{rank}.trace.json",
            ))
    seeds = tuple(sorted(int(host["frozen_protocol"]["seed"]) for host in hosts))
    if not args.allow_smoke and seeds != DIAGNOSTIC_SEEDS:
        raise ValueError("R5-P4 diagnostic seed mismatch")

    router_rows = [row for value in rank_analyses for row in value["router"]]
    comm_rows = [row for value in rank_analyses for row in value["communication"]]
    router_lookup = {
        (row["seed"], row["rank"], row["family_index"], row["job"], row["arm"], row["item"]): row
        for row in router_rows
    }
    comm_lookup = {
        (row["seed"], row["rank"], row["family_index"], row["job"], row["arm"], row["item"], row["kind"]): row
        for row in comm_rows
    }

    descriptor_rows, job_metrics = [], []
    paired_rows = [row for host in hosts for row in host["paired_rows"]]
    host_rank_lookup = {
        (int(host["frozen_protocol"]["seed"]), int(rank_result["rank"])): rank_result
        for host in hosts for rank_result in host["rank_results"]
    }
    for pair in paired_rows:
        seed = int(pair["seed"]); family_index = int(pair["family_index"]); job = int(pair["job"])
        arm_sums: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            sums = {
                "ready_skew_us": 0.0, "count_rendezvous_us": 0.0,
                "payload_launch_us": 0.0, "payload_gpu_execution_us": 0.0,
                "actual_router_payload_overlap_us": 0.0,
            }
            for item in range(TRIGGERS):
                count = [comm_lookup[(seed, rank, family_index, job, arm, item, "count")] for rank in (0, 1)]
                payload = [comm_lookup[(seed, rank, family_index, job, arm, item, "payload")] for rank in (0, 1)]
                ready_skew = abs(count[1]["descriptor_ready_host_ns"] - count[0]["descriptor_ready_host_ns"]) / 1e3
                count_issue_skew = abs(count[1]["count_start_host_ns"] - count[0]["count_start_host_ns"]) / 1e3
                count_envelope = (
                    max(value["count_visible_host_ns"] for value in count)
                    - min(value["count_start_host_ns"] for value in count)
                ) / 1e3
                payload_call_skew = abs(payload[1]["payload_call_host_ns"] - payload[0]["payload_call_host_ns"]) / 1e3
                gpu_start_skew = abs(payload[1]["gpu_start_us"] - payload[0]["gpu_start_us"])
                launch_delay = max(value["call_start_to_gpu_start_us"] for value in payload)
                gpu_execution = max(value["gpu_envelope_us"] for value in payload)
                overlap_by_rank = []
                per_chunk = []
                for rank in (0, 1):
                    future = [
                        interval
                        for chunk in range(8) if chunk > min(item, 7)
                        for interval in router_lookup[(seed, rank, family_index, job, arm, chunk)]["kernel_intervals_us"]
                    ]
                    overlap_value = _interval_overlap(payload[rank]["kernel_intervals_us"], future)
                    overlap_by_rank.append(overlap_value)
                    per_chunk.append({
                        str(chunk): _interval_overlap(
                            payload[rank]["kernel_intervals_us"],
                            router_lookup[(seed, rank, family_index, job, arm, chunk)]["kernel_intervals_us"],
                        ) for chunk in range(8)
                    })
                row = {
                    "seed": seed, "family": FAMILIES[family_index], "family_index": family_index,
                    "job": job, "arm": arm, "descriptor": item,
                    "ready_skew_us": ready_skew, "count_issue_skew_us": count_issue_skew,
                    "count_issue_to_both_complete_us": count_envelope,
                    "count_residual_wait_max_us": max(value["count_wait_us"] for value in count),
                    "count_event_gpu_max_us": max(value["count_event_gpu_us"] for value in count),
                    "payload_call_skew_us": payload_call_skew,
                    "payload_call_to_gpu_start_max_us": launch_delay,
                    "payload_gpu_start_skew_us": gpu_start_skew,
                    "payload_gpu_active_max_us": max(value["gpu_active_us"] for value in payload),
                    "payload_gpu_envelope_max_us": gpu_execution,
                    "actual_router_payload_overlap_us": sum(overlap_by_rank),
                    "per_rank_overlap_us": overlap_by_rank, "per_rank_per_chunk_overlap_us": per_chunk,
                }
                descriptor_rows.append(row)
                sums["ready_skew_us"] += ready_skew
                sums["count_rendezvous_us"] += count_envelope
                sums["payload_launch_us"] += launch_delay + gpu_start_skew
                sums["payload_gpu_execution_us"] += gpu_execution
                sums["actual_router_payload_overlap_us"] += sum(overlap_by_rank)

            router_rank_rows = [
                [router_lookup[(seed, rank, family_index, job, arm, chunk)] for chunk in range(8)]
                for rank in (0, 1)
            ]
            sums["router_gpu_envelope_us"] = max(
                rows[-1]["gpu_end_us"] - rows[0]["gpu_start_us"] for rows in router_rank_rows
            )
            sums["router_gpu_active_us"] = max(
                sum(row["gpu_active_us"] for row in rows) for rows in router_rank_rows
            )
            rank_hosts = [
                next(
                    pair_row["arms"][arm] for pair_row in host_rank_lookup[(seed, rank)]["pairs"]
                    if int(pair_row["family_index"]) == family_index and int(pair_row["job"]) == job
                ) for rank in (0, 1)
            ]
            sums["router_host_visibility_us"] = max(
                value["diagnostics"]["router_us"] for value in rank_hosts
            )
            arm_sums[arm] = sums

        e1_extra_makespan = -float(pair["delta_D1_minus_E1_us"])
        signed = {
            key: arm_sums["E1"][key] - arm_sums["D1"][key]
            for key in arm_sums["E1"]
        }
        job_metrics.append({
            "seed": seed, "family": FAMILIES[family_index], "family_index": family_index,
            "job": job, "E1_extra_makespan_us": e1_extra_makespan,
            "arms": arm_sums, "signed_E1_minus_D1": signed,
        })

    signed_categories = {
        "ready_skew": [value["signed_E1_minus_D1"]["ready_skew_us"] for value in job_metrics],
        "count_rendezvous": [value["signed_E1_minus_D1"]["count_rendezvous_us"] for value in job_metrics],
        "payload_launch_rank_skew": [value["signed_E1_minus_D1"]["payload_launch_us"] for value in job_metrics],
        "payload_gpu_execution": [value["signed_E1_minus_D1"]["payload_gpu_execution_us"] for value in job_metrics],
        "router_interference_envelope": [value["signed_E1_minus_D1"]["router_gpu_envelope_us"] for value in job_metrics],
        "router_interference_active": [value["signed_E1_minus_D1"]["router_gpu_active_us"] for value in job_metrics],
    }
    medians = {key: float(np.median(values)) for key, values in signed_categories.items()}
    positive_total = sum(max(value, 0.0) for value in medians.values())
    shares = {
        key: (max(value, 0.0) / positive_total if positive_total > 0 else 0.0)
        for key, value in medians.items()
    }
    collective_keys = ("ready_skew", "count_rendezvous", "payload_launch_rank_skew")
    resource_keys = ("payload_gpu_execution", "router_interference_envelope", "router_interference_active")
    collective_share = sum(shares[key] for key in collective_keys)
    resource_share = sum(shares[key] for key in resource_keys)

    def all_seed_group_positive(keys: tuple[str, ...]) -> bool:
        for seed in seeds:
            values = [
                sum(value["signed_E1_minus_D1"][{
                    "ready_skew": "ready_skew_us",
                    "count_rendezvous": "count_rendezvous_us",
                    "payload_launch_rank_skew": "payload_launch_us",
                    "payload_gpu_execution": "payload_gpu_execution_us",
                    "router_interference_envelope": "router_gpu_envelope_us",
                    "router_interference_active": "router_gpu_active_us",
                }[key]] for key in keys)
                for value in job_metrics if value["seed"] == seed
            ]
            if float(np.median(values)) <= 0:
                return False
        return True

    collective_seed_positive = all_seed_group_positive(collective_keys)
    resource_seed_positive = all_seed_group_positive(resource_keys)
    if collective_share >= 0.5 and collective_seed_positive:
        classification = "collective/rank-rendezvous dominated"
    elif resource_share >= 0.5 and resource_seed_positive:
        classification = "resource-contention dominated"
    else:
        classification = "both/mixed"

    by_arm = {}
    for arm in ARMS:
        desc = [value for value in descriptor_rows if value["arm"] == arm]
        routers = [value for value in job_metrics]
        by_arm[arm] = {
            "cross_rank_ready_skew_us": distribution(value["ready_skew_us"] for value in desc),
            "cross_rank_count_issue_skew_us": distribution(value["count_issue_skew_us"] for value in desc),
            "count_issue_to_both_complete_us": distribution(value["count_issue_to_both_complete_us"] for value in desc),
            "count_residual_wait_max_us": distribution(value["count_residual_wait_max_us"] for value in desc),
            "count_event_gpu_max_us": distribution(value["count_event_gpu_max_us"] for value in desc),
            "cross_rank_payload_call_skew_us": distribution(value["payload_call_skew_us"] for value in desc),
            "payload_call_to_gpu_start_max_us": distribution(value["payload_call_to_gpu_start_max_us"] for value in desc),
            "cross_rank_payload_gpu_start_skew_us": distribution(value["payload_gpu_start_skew_us"] for value in desc),
            "payload_gpu_active_max_us": distribution(value["payload_gpu_active_max_us"] for value in desc),
            "payload_gpu_envelope_max_us": distribution(value["payload_gpu_envelope_max_us"] for value in desc),
            "actual_router_payload_overlap_us": distribution(value["actual_router_payload_overlap_us"] for value in desc),
            "router_gpu_envelope_us": distribution(value["arms"][arm]["router_gpu_envelope_us"] for value in routers),
            "router_gpu_active_us": distribution(value["arms"][arm]["router_gpu_active_us"] for value in routers),
            "router_host_visibility_us": distribution(value["arms"][arm]["router_host_visibility_us"] for value in routers),
        }

    profiler_deltas = [float(value["delta_D1_minus_E1_us"]) for value in paired_rows]
    correctness = {
        "all_host_pairs_pass": all(value["pass"] for value in paired_rows),
        "all_trace_associations_pass": all(
            value["audit"]["association_failures"] == 0 for value in rank_analyses
        ),
        "max_output_difference_zero": max(
            value["max_abs_output_difference"]
            for host in hosts for rank in host["rank_results"] for value in (
                pair["equivalence"] for pair in rank["pairs"]
            )
        ) == 0.0,
    }
    result = {
        "schema_version": 1, "study": "R5-P4 optimized progressive diagnosis",
        "seeds": list(seeds), "pairs": len(paired_rows),
        "correctness": {**correctness, "pass": all(correctness.values())},
        "profiler_makespan_D1_minus_E1_us": distribution(profiler_deltas),
        "profiler_sign_reproduced": float(np.median(profiler_deltas)) < 0,
        "by_arm": by_arm,
        "signed_E1_minus_D1_category_deltas_us": {
            key: distribution(values) for key, values in signed_categories.items()
        },
        "descriptive_positive_cost_shares": {
            **shares, "collective_group": collective_share, "resource_group": resource_share,
            "non_causal_non_additive": True,
        },
        "classification": {
            "value": classification,
            "collective_share_ge_50pct": collective_share >= 0.5,
            "collective_three_seed_medians_positive": collective_seed_positive,
            "resource_share_ge_50pct": resource_share >= 0.5,
            "resource_three_seed_medians_positive": resource_seed_positive,
            "recommend_apply_for_msccl_backend_integration": classification == "collective/rank-rendezvous dominated",
        },
        "descriptor_rows": descriptor_rows, "job_metrics": job_metrics,
        "trace_audits": [value["audit"] for value in rank_analyses],
        "status": "R5_P4_DIAGNOSIS_COMPLETE_PENDING_SUPERVISOR",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        "r5_p4_smoke_results.json" if args.allow_smoke else "r5_p4_results.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output),
        "correctness": result["correctness"], "classification": result["classification"],
    }, indent=2))


if __name__ == "__main__":
    main()
