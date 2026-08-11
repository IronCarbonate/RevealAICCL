"""Fail-closed host/Kineto analysis for the preregistered R3-P0 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np


PILOT_SEEDS = (6042, 6142, 6242)
FAMILIES = (
    "balanced", "skewed", "all_to_one_like", "zero_sized_pair",
    "multiple_progressive_shards",
)
TRIGGERS = (0, 1, 2, 3, 4, 5, 7)
LABEL = re.compile(
    r"R3P0\|kind=(?P<kind>router|aiccl|packing|count|payload)\|"
    r"seed=(?P<seed>\d+)\|family=(?P<family>\d+)\|job=(?P<job>\d+)\|"
    r"arm=(?P<arm>[CD])\|item=(?P<item>\d+)"
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


def bootstrap_median(values: list[float]) -> dict[str, float | int | bool]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260810)
    samples = np.median(array[rng.integers(0, len(array), size=(10_000, len(array)))], axis=1)
    low, high = np.percentile(samples, (2.5, 97.5), method="linear")
    return {
        "count": len(values), "median_us": float(np.median(array)),
        "ci95_low_us": float(low), "ci95_high_us": float(high),
        "ci95_lower_positive": bool(low > 0.0),
    }


def overlap(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals))


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"traceEvents missing: {path}")
    return events


def _host_seed(host: dict[str, Any]) -> int:
    protocol = host["frozen_protocol"]
    if "seed" in protocol:
        return int(protocol["seed"])
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 1:
        raise ValueError("host artifact does not identify exactly one seed")
    return int(seeds[0])


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
    unique = {(float(event["ts"]), float(event["dur"]), str(event["name"]), int(event.get("args", {}).get("stream", -1))): event for event in matched}
    return sorted(unique.values(), key=lambda event: float(event["ts"]))


def analyze_trace(host: dict[str, Any], rank: int, trace_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    events = _load_events(trace_path)
    annotations = [
        event for event in events
        if event.get("cat") == "user_annotation" and LABEL.fullmatch(str(event.get("name", "")))
    ]
    expected_arms = len(FAMILIES) * int(host["frozen_protocol"]["jobs_per_family"]) * 2
    expected = {"router": expected_arms * 8, "count": expected_arms * 7, "payload": expected_arms * 7}
    cpu_ops = [event for event in events if event.get("cat") == "cpu_op"]
    kernels_by_external: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("cat") == "kernel" and event.get("args", {}).get("External id") is not None:
            kernels_by_external.setdefault(int(event["args"]["External id"]), []).append(event)
    rank_host = next(row for row in host["rank_results"] if int(row["rank"]) == rank)
    arm_lookup = {
        (FAMILIES.index(pair["family"]), int(pair["job"]), arm): pair[arm]
        for pair in rank_host["pairs"] for arm in ("C", "D")
    }
    router_rows, comm_rows = [], []
    failures = 0
    counts = {key: 0 for key in expected}
    for annotation in annotations:
        match = LABEL.fullmatch(str(annotation["name"]))
        assert match is not None
        kind = match.group("kind")
        if kind not in expected:
            continue
        counts[kind] += 1
        family_index, job, arm, item = int(match.group("family")), int(match.group("job")), match.group("arm"), int(match.group("item"))
        kernels = _kernel_association(annotation, cpu_ops, kernels_by_external, nccl=kind != "router")
        if not kernels:
            failures += 1
            continue
        start = min(float(event["ts"]) for event in kernels)
        end = max(float(event["ts"]) + float(event["dur"]) for event in kernels)
        intervals = [(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in kernels]
        base = {
            "seed": _host_seed(host), "rank": rank,
            "family": FAMILIES[family_index], "family_index": family_index,
            "job": job, "arm": arm, "item": item,
            "gpu_start_us": start, "gpu_end_us": end,
            "gpu_duration_us": sum(right - left for left, right in intervals),
            "gpu_envelope_us": end - start,
            "stream": sorted({int(event.get("args", {}).get("stream", -1)) for event in kernels}),
            "kernel_count": len(kernels), "kernel_intervals_us": intervals,
        }
        if kind == "router":
            router_rows.append(base)
        else:
            arm_host = arm_lookup[(family_index, job, arm)]
            descriptor = next(value for value in arm_host["descriptors"] if int(value["trigger_chunk"]) == item)
            call_ns = int(descriptor["communication"][f"{kind}_call_host_ns"])
            base.update({
                "kind": kind, "trace_annotation_start_us": float(annotation["ts"]),
                "trace_annotation_end_us": float(annotation["ts"]) + float(annotation["dur"]),
                "explicit_monotonic_call_us": call_ns / 1e3,
                "trace_submit_call_start_to_gpu_start_us": start - float(annotation["ts"]),
                "trace_api_return_to_gpu_start_us": (
                    start - (float(annotation["ts"]) + float(annotation["dur"]))
                ),
                "clock_domains_mixed": False,
            })
            comm_rows.append(base)
    # Kineto does not propagate record_function state into the independently
    # created router producer thread on this PyTorch build.  Reconstruct those
    # intervals from the pinned D2H chunk delimiters, as in the audited O1
    # analyzer, with fail-closed cardinality/byte/stream checks.
    if counts["router"] == 0:
        execution_order = []
        for pair in rank_host["pairs"]:
            for arm in pair["order"]:
                for chunk, size in enumerate(pair[arm]["chunk_sizes"] if "chunk_sizes" in pair[arm] else FAMILY_SPECS_FOR_TRACE(pair["family"])):
                    execution_order.append((FAMILIES.index(pair["family"]), int(pair["job"]), arm, chunk, int(size) * 8))
        expected_router_bytes = {meta[4] for meta in execution_order}
        d2h = sorted(
            (event for event in events if event.get("cat") == "gpu_memcpy"
             and event.get("name") == "Memcpy DtoH (Device -> Pinned)"
             and int(event.get("args", {}).get("bytes", -1)) in expected_router_bytes),
            key=lambda event: float(event["ts"]),
        )
        if len(d2h) != expected["router"]:
            raise ValueError(f"router D2H delimiter mismatch: {len(d2h)} != {expected['router']}")
        streams = {int(event.get("args", {}).get("stream", -1)) for event in d2h}
        if len(streams) != 1:
            raise ValueError(f"router stream ambiguity: {streams}")
        router_stream = next(iter(streams))
        candidate_kernels = sorted(
            (event for event in events if event.get("cat") == "kernel"
             and int(event.get("args", {}).get("stream", -1)) == router_stream
             and "nccl" not in str(event.get("name", "")).lower()),
            key=lambda event: float(event["ts"]),
        )
        previous_end = float("-inf")
        for marker, meta in zip(d2h, execution_order, strict=True):
            family_index, job, arm, chunk, expected_bytes = meta
            actual_bytes = int(marker.get("args", {}).get("bytes", -1))
            if actual_bytes != expected_bytes:
                raise ValueError(f"router D2H byte mismatch: {actual_bytes} != {expected_bytes}")
            marker_start = float(marker["ts"])
            kernels = [event for event in candidate_kernels if previous_end <= float(event["ts"]) < marker_start]
            if not kernels:
                raise ValueError("router chunk has no associated kernels")
            intervals = [(float(event["ts"]), float(event["ts"]) + float(event["dur"])) for event in kernels]
            start, end = min(left for left, _ in intervals), max(right for _, right in intervals)
            router_rows.append({
                "seed": _host_seed(host), "rank": rank,
                "family": FAMILIES[family_index], "family_index": family_index,
                "job": job, "arm": arm, "item": chunk, "gpu_start_us": start,
                "gpu_end_us": end, "gpu_duration_us": sum(right - left for left, right in intervals),
                "gpu_envelope_us": end - start, "stream": [router_stream],
                "kernel_count": len(kernels), "kernel_intervals_us": intervals,
                "association": "pinned_d2h_chunk_delimiter",
            })
            previous_end = marker_start + float(marker["dur"])
        counts["router"] = len(router_rows)
    if counts != expected or failures:
        raise ValueError(f"fail-closed trace association: counts={counts}, expected={expected}, failures={failures}")
    router_lookup = {(r["family_index"], r["job"], r["arm"], r["item"]): r for r in router_rows}
    for row in comm_rows:
        chunks = [router_lookup[(row["family_index"], row["job"], row["arm"], chunk)] for chunk in range(8)]
        final_end = chunks[-1]["gpu_end_us"]
        future = [tuple(interval) for chunk in chunks if chunk["item"] > row["item"] for interval in chunk["kernel_intervals_us"]]
        row["final_router_gpu_end_us"] = final_end
        row["gpu_start_before_final"] = row["gpu_start_us"] < final_end
        row["actual_overlap_us"] = overlap(row["gpu_start_us"], row["gpu_end_us"], future)
        row["positive_actual_overlap"] = row["actual_overlap_us"] > 0
    return router_rows, comm_rows, {
        "trace_path": str(trace_path), "trace_sha256": sha256_file(trace_path),
        "trace_size_bytes": trace_path.stat().st_size, "annotation_counts": counts,
        "association_failures": failures,
    }


def FAMILY_SPECS_FOR_TRACE(family: str) -> tuple[int, ...]:
    if family == "multiple_progressive_shards":
        return (128, 256, 384, 512, 640, 768, 512, 896)
    return (512,) * 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if not args.allow_smoke and len(args.run_dir) != 3:
        raise ValueError("canonical P0 analysis requires exactly three seed run directories")
    hosts, router_rows, comm_rows, trace_audits = [], [], [], []
    for run_dir in args.run_dir:
        host_paths = list(run_dir.glob("r3_p0_seed*_host.json"))
        if len(host_paths) != 1:
            raise ValueError(f"host artifact ambiguity: {run_dir}")
        host = json.loads(host_paths[0].read_text(encoding="utf-8"))
        hosts.append(host)
        seed = _host_seed(host)
        for rank in (0, 1):
            trace = run_dir / f"r3_p0_seed{seed}_rank{rank}.trace.json"
            routers, comms, audit = analyze_trace(host, rank, trace)
            router_rows.extend(routers)
            comm_rows.extend(comms)
            trace_audits.append(audit)
    observed_seeds = tuple(sorted(_host_seed(host) for host in hosts))
    if not args.allow_smoke and observed_seeds != PILOT_SEEDS:
        raise ValueError("pilot seed set mismatch")

    pairs = [row for host in hosts for row in host["paired_rows"]]
    deltas = [float(row["delta_us"]) for row in pairs]
    primary = bootstrap_median(deltas)
    per_seed = {}
    for seed in observed_seeds:
        seed_rows = [row for row in pairs if int(row["seed"]) == seed]
        values = [float(row["delta_us"]) for row in seed_rows]
        per_seed[str(seed)] = {
            "delta_us": distribution(values),
            "C_primary_us": distribution(float(row["C_primary_us"]) for row in seed_rows),
            "D_primary_us": distribution(float(row["D_primary_us"]) for row in seed_rows),
            "median_positive": bool(np.median(values) > 0),
        }
    per_family = {}
    for family in FAMILIES:
        rows = [row for row in pairs if row["family"] == family]
        per_family[family] = {
            "delta_us": distribution(float(row["delta_us"]) for row in rows),
            "C_primary_us": distribution(float(row["C_primary_us"]) for row in rows),
            "D_primary_us": distribution(float(row["D_primary_us"]) for row in rows),
            "positive_fraction": sum(float(row["delta_us"]) > 0 for row in rows) / len(rows),
        }

    all_arms = [pair[arm] for host in hosts for rank in host["rank_results"] for pair in rank["pairs"] for arm in ("C", "D")]
    descriptors = [descriptor for arm in all_arms for descriptor in arm["descriptors"]]
    by_arm = {}
    for arm_name in ("C", "D"):
        arms = [arm for arm in all_arms if arm["arm"] == arm_name]
        desc = [value for arm in arms for value in arm["descriptors"]]
        by_arm[arm_name] = {
            "router_us": distribution(arm["router_final_latency_us"] for arm in arms),
            "count_construction_us": distribution(value["count_construction_us"] for value in desc),
            "offset_construction_us": distribution(value["offset_construction_us"] for value in desc),
            "packing_us": distribution(value["packing_us"] for value in desc),
            "h2d_us": distribution(value["communication"]["h2d_us"] for value in desc),
            "delta_count_exchange_us": distribution(value["communication"]["count_completion_us"] for value in desc),
            "aiccl_control_us": distribution(value["aiccl_control_us"] for value in desc),
            "a2av_submit_us": distribution(value["communication"]["payload_submit_us"] for value in desc),
            "primary_combined_us": distribution(row[f"{arm_name}_primary_us"] for row in pairs),
            "full_reference_us": distribution(row[f"{arm_name}_full_reference_us"] for row in pairs),
        }
    payload_rows = [row for row in comm_rows if row["kind"] == "payload"]
    count_rows = [row for row in comm_rows if row["kind"] == "count"]
    for arm_name in ("C", "D"):
        rows = [row for row in payload_rows if row["arm"] == arm_name]
        by_arm[arm_name]["a2av_gpu_completion_us"] = distribution(row["gpu_duration_us"] for row in rows)

    count_detail: dict[str, Any] = {"overall": {}}
    for arm_name in ("C", "D"):
        count_detail["overall"][arm_name] = distribution(
            descriptor["communication"]["count_completion_us"]
            for host in hosts for rank in host["rank_results"] for pair in rank["pairs"]
            for descriptor in pair[arm_name]["descriptors"]
        )
    count_detail["per_seed"] = {
        str(seed): {arm: distribution(
            descriptor["communication"]["count_completion_us"]
            for host in hosts if _host_seed(host) == seed
            for rank in host["rank_results"] for pair in rank["pairs"]
            for descriptor in pair[arm]["descriptors"]
        ) for arm in ("C", "D")} for seed in observed_seeds
    }
    count_detail["per_chunk"] = {
        str(trigger): {arm: distribution(
            descriptor["communication"]["count_completion_us"]
            for host in hosts for rank in host["rank_results"] for pair in rank["pairs"]
            for descriptor in pair[arm]["descriptors"] if int(descriptor["trigger_chunk"]) == trigger
        ) for arm in ("C", "D")} for trigger in TRIGGERS
    }
    count_detail["per_rank"] = {
        str(rank_index): {arm: distribution(
            descriptor["communication"]["count_completion_us"]
            for host in hosts for rank in host["rank_results"] if int(rank["rank"]) == rank_index
            for pair in rank["pairs"] for descriptor in pair[arm]["descriptors"]
        ) for arm in ("C", "D")} for rank_index in (0, 1)
    }

    early_payload = [row for row in payload_rows if row["arm"] == "C"]
    router_slowdown_host = []
    for host in hosts:
        for rank in host["rank_results"]:
            for pair in rank["pairs"]:
                router_slowdown_host.append(pair["C"]["router_final_latency_us"] - pair["D"]["router_final_latency_us"])
    router_gpu_lookup = {
        (row["seed"], row["rank"], row["family"], row["job"], row["arm"], row["item"]): row
        for row in router_rows
    }
    jobs_per_family = int(hosts[0]["frozen_protocol"]["jobs_per_family"])
    router_slowdown_gpu = []
    for seed in observed_seeds:
        for rank_index in (0, 1):
            for family in FAMILIES:
                for job in range(jobs_per_family):
                    latency = {}
                    for arm in ("C", "D"):
                        first = router_gpu_lookup[(seed, rank_index, family, job, arm, 0)]["gpu_start_us"]
                        final = router_gpu_lookup[(seed, rank_index, family, job, arm, 7)]["gpu_end_us"]
                        latency[arm] = final - first
                    router_slowdown_gpu.append(latency["C"] - latency["D"])
    rank_skew = []
    lookup = {(r["seed"], r["family"], r["job"], r["arm"], r["item"], r["rank"]): r for r in payload_rows}
    for seed in observed_seeds:
        for family in FAMILIES:
            for job in range(jobs_per_family):
                for arm in ("C", "D"):
                    for trigger in TRIGGERS:
                        left = lookup[(seed, family, job, arm, trigger, 0)]["gpu_start_us"]
                        right = lookup[(seed, family, job, arm, trigger, 1)]["gpu_start_us"]
                        rank_skew.append({"arm": arm, "skew_us": abs(left - right)})
    built_before_final, built_total = 0, 0
    bytes_before_final, bytes_total = 0, 0
    for host in hosts:
        for rank in host["rank_results"]:
            for pair in rank["pairs"]:
                early = pair["C"]
                for descriptor in early["descriptors"]:
                    built_total += 1
                    bytes_total += int(descriptor["bytes"])
                    if int(descriptor["built_host_ns"]) < int(early["final_router_host_ns"]):
                        built_before_final += 1
                        bytes_before_final += int(descriptor["bytes"])

    correctness = {key: all(host["correctness"][key] for host in hosts) for key in hosts[0]["correctness"]}
    seed_pass = all(value["median_positive"] for value in per_seed.values())
    gate_pass = bool(primary["median_us"] > 0 and primary["ci95_lower_positive"] and seed_pass and all(correctness.values()))
    result = {
        "schema_version": 1, "study": "R3-P0 progressive early A2Av pilot",
        "status": "R3_P0_PASS_PENDING_SUPERVISOR" if gate_pass else "R3_P0_FAIL_PENDING_SUPERVISOR",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_gate": {
            **primary, "three_of_three_seed_medians_positive": seed_pass,
            "correctness_pass": all(correctness.values()), "pass": gate_pass,
        },
        "per_seed": per_seed, "per_family": per_family,
        "latency_breakdown": by_arm, "count_exchange_detail": count_detail,
        "correctness": correctness,
        "traffic": {
            "total_sent_bytes_all_rank_arms": sum(int(arm["total_sent_bytes"]) for arm in all_arms),
            "per_descriptor_bytes": distribution(int(value["bytes"]) for value in descriptors),
            "per_pair_token_counts": distribution(
                int(count) for value in descriptors for count in value["sendcounts_tokens"]
            ),
            "zero_sized_pairs": sum(
                int(count) == 0 for value in descriptors for count in value["sendcounts_tokens"]
            ),
        },
        "secondary": {
            "packing_descriptors_before_final": built_before_final,
            "packing_descriptor_fraction_before_final": built_before_final / built_total,
            "packing_bytes_fraction_before_final": bytes_before_final / bytes_total,
            "router_slowdown_C_minus_D_gpu_us": distribution(router_slowdown_gpu),
            "router_slowdown_C_minus_D_gpu_median_bootstrap": bootstrap_median(router_slowdown_gpu),
            "router_slowdown_C_minus_D_host_us": distribution(router_slowdown_host),
            "payload_submit_to_gpu_start_us": distribution(row["trace_submit_call_start_to_gpu_start_us"] for row in early_payload),
            "payload_api_return_to_gpu_start_us": distribution(row["trace_api_return_to_gpu_start_us"] for row in early_payload),
            "payload_rank_start_skew_us": distribution(row["skew_us"] for row in rank_skew if row["arm"] == "C"),
            "a2av_gpu_start_before_final_fraction": sum(row["gpu_start_before_final"] for row in early_payload) / len(early_payload),
            "actual_device_overlap_fraction": sum(row["positive_actual_overlap"] for row in early_payload) / len(early_payload),
            "positive_actual_overlap_us": distribution(row["actual_overlap_us"] for row in early_payload if row["actual_overlap_us"] > 0),
            "communication_completed_before_final_fraction": sum(row["gpu_end_us"] < row["final_router_gpu_end_us"] for row in early_payload) / len(early_payload),
            "count_submit_to_gpu_start_us": distribution(row["trace_submit_call_start_to_gpu_start_us"] for row in count_rows if row["arm"] == "C"),
            "count_api_return_to_gpu_start_us": distribution(row["trace_api_return_to_gpu_start_us"] for row in count_rows if row["arm"] == "C"),
            "host_device_clock_domains_mixed": False,
        },
        "trace_audit": trace_audits, "paired_rows": pairs,
        "source_artifacts": [{"path": str(path), "sha256": sha256_file(path)} for run in args.run_dir for path in run.glob("r3_p0_seed*_host.json")],
        "recommendation": "eligible_to_apply_for_R3_formal_validation" if gate_pass else "not_eligible_for_R3_formal_validation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "r3_p0_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "r3_p0_paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "pass": gate_pass, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
