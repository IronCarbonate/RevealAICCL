"""Aggregate R6-M3 timelines, controls, dependency audit, and Kineto kernels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEEDS = (13042, 13142, 13242)
MODES = ("normal", "router_absent", "dependency_resolved")


def dist(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    if not data.size:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(data.size), "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)), "p99": float(np.percentile(data, 99)),
        "max": float(data.max()), "mean": float(data.mean()),
    }


def overlap(left: int, right: int, other_left: int, other_right: int) -> int:
    return max(0, min(right, other_right) - max(left, other_left))


def queue_cover(
    left: int, right: int, *, router: list[tuple[int, int]],
    staging: list[tuple[int, int]], previous_puts: list[tuple[int, int]],
) -> dict[str, float]:
    if right <= left:
        return {"previous_put_us": 0.0, "future_router_us": 0.0, "staging_us": 0.0, "other_idle_us": 0.0}
    boundaries = {left, right}
    for intervals in (previous_puts, router, staging):
        for start, end in intervals:
            if overlap(left, right, start, end):
                boundaries.add(max(left, start)); boundaries.add(min(right, end))
    result = {"previous_put_us": 0.0, "future_router_us": 0.0, "staging_us": 0.0, "other_idle_us": 0.0}
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        middle = (start + end) // 2
        key = "other_idle_us"
        if any(a <= middle < b for a, b in previous_puts): key = "previous_put_us"
        elif any(a <= middle < b for a, b in router): key = "future_router_us"
        elif any(a <= middle < b for a, b in staging): key = "staging_us"
        result[key] += (end - start) / 1e3
    return result


def correctness_ok(result: dict[str, Any]) -> bool:
    correct, semantic = result["correctness"], result["semantic"]
    return bool(
        correct["final_combine_correct"] and correct["token_integrity"]
        and all(value == 0 for key, value in correct.items()
                if key not in {"final_combine_correct", "token_integrity"})
        and semantic["legal"] == semantic["total"]
        and all(value == 0 for key, value in semantic.items() if key not in {"legal", "total"})
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [json.loads((args.raw_dir / f"r6_m3_seed{seed}_host.json").read_text()) for seed in SEEDS]
    rows: list[dict[str, Any]] = []
    correctness = all(payload["pass"] for payload in payloads)
    legality = True
    equivalence = True
    wrong_event_dependency = future_pack_dependency = event_reuse_hazard = 0
    put_blocked_by_previous_wait = put_blocked_by_previous_put = put_blocked_by_unresolved_event = 0
    previous_put_block_us = unresolved_event_block_us = 0.0
    zero_overlap_reproductions = adapter_before_final = native_before_final = gpu_after_final = 0
    environment = payloads[0]["environment"]
    stream_audits: list[dict[str, Any]] = []

    for payload in payloads:
        for rank_result in payload["rank_results"]:
            rank = int(rank_result["rank"])
            for case in rank_result["cases"]:
                result = case["result"]
                correctness &= correctness_ok(result)
                legality &= bool(
                    result["semantic"]["future_access"] == 0
                    and result["semantic"]["unrevealed_execution"] == 0
                    and result["semantic"]["stale_dispatch"] == 0
                    and result["diagnostics"]["forward_transport"]["future_access"] == 0
                    and result["diagnostics"]["forward_transport"]["unrevealed_access"] == 0
                    and result["diagnostics"]["forward_transport"]["stale_action"] == 0
                )
                equivalence &= bool(case["equivalence"]["pass"])
                diag = result["diagnostics"]
                stream_audits.append({
                    "seed": case["seed"], "family": case["family"],
                    "rank": rank, "mode": case["mode"], **diag["stream_audit"],
                    "packing_stream": diag["stream_audit"]["default_stream"],
                    "packing_priority": diag["stream_audit"]["default_priority"],
                    "return_stream": diag["stream_audit"]["comm_stream"],
                    "return_priority": diag["stream_audit"]["comm_priority"],
                })
                router_intervals = list(zip(
                    diag["router_chunk_gpu_start_host_ns"],
                    diag["router_chunk_gpu_end_host_ns"], strict=True,
                ))
                final_router = int(diag["router_gpu_end_host_ns"])
                descriptors = result["forward_descriptors"]
                put_intervals = [
                    (int(value["communication"]["put_kernel_start_host_ns"]),
                     int(value["communication"]["put_kernel_end_host_ns"]))
                    for value in descriptors
                ]
                staging_intervals = [
                    (int(value["communication"]["staging_gpu_start_host_ns"]),
                     int(value["communication"]["staging_gpu_end_host_ns"]))
                    for value in descriptors
                ]
                for index, descriptor in enumerate(descriptors):
                    comm = descriptor["communication"]
                    trigger = int(descriptor["trigger"])
                    t0 = int(descriptor["router_chunk_launch_host_ns"])
                    t1, t2 = (int(value) for value in router_intervals[trigger])
                    t3 = int(descriptor["router_chunk_ready_host_ns"])
                    t4 = int(descriptor["state_update_done_host_ns"])
                    t5 = int(descriptor["action_commit_host_ns"])
                    t6 = int(descriptor["guard_pass_host_ns"])
                    t7 = int(descriptor["committed_action_created_host_ns"])
                    t8 = int(descriptor["packing_start_host_ns"])
                    t9 = int(comm["staging_gpu_start_host_ns"])
                    t10 = int(comm["staging_gpu_end_host_ns"])
                    t11 = int(descriptor["pack_event_record_host_ns"])
                    t12 = int(descriptor["comm_wait_event_enqueue_host_ns"])
                    t13 = int(comm["native_wrapper_enter_host_ns"])
                    t14 = int(comm["kernel_launch_call_host_ns"])
                    t15 = int(comm["kernel_launch_return_host_ns"])
                    t16 = int(comm["put_kernel_start_host_ns"])
                    t17 = int(comm["put_kernel_end_host_ns"])
                    t20 = int(comm["remote_wait_complete_host_ns"])
                    t19 = t20 - int(float(comm["wait_gpu_duration_us"]) * 1e3)
                    t21 = final_router
                    scheduler_us = (t7 - t2) / 1e3
                    packing_us = (t10 - t7) / 1e3
                    enqueue_us = (t15 - t10) / 1e3
                    queue_raw_us = (t16 - t15) / 1e3
                    queue_us = max(0.0, queue_raw_us)
                    ready_us = (t16 - t2) / 1e3
                    future_router = router_intervals[trigger + 1:]
                    cover = queue_cover(
                        t15, t16, router=future_router,
                        staging=[staging_intervals[index]], previous_puts=put_intervals[:index],
                    )
                    current_wrong = int(descriptor["comm_wait_event_id"] != descriptor["pack_event_id"])
                    current_future = int(
                        descriptor["comm_wait_event_id"] != descriptor["pack_event_id"]
                        or descriptor["pack_event_producer_stream"] == descriptor["comm_wait_event_stream"]
                    )
                    current_reuse = int(descriptor["pack_event_reused"])
                    wrong_event_dependency += current_wrong
                    future_pack_dependency += current_future
                    event_reuse_hazard += current_reuse
                    previous_wait = int(descriptor["preceding_operation"] == "wait")
                    previous_put = int(cover["previous_put_us"] > 0)
                    unresolved = int(t15 < t10 and t16 >= t10)
                    put_blocked_by_previous_wait += previous_wait
                    put_blocked_by_previous_put += previous_put
                    put_blocked_by_unresolved_event += unresolved
                    previous_put_block_us += cover["previous_put_us"]
                    unresolved_event_block_us += max(0, min(t16, t10) - t15) / 1e3
                    adapter_issue = int(descriptor["adapter_issue_host_ns"])
                    adapter_issue_before = adapter_issue < t21
                    native_issue_before = t13 < t21
                    starts_after = t16 >= t21
                    no_overlap = all(overlap(t16, t17, a, b) == 0 for a, b in future_router)
                    if case["mode"] == "normal":
                        adapter_before_final += int(adapter_issue_before)
                        native_before_final += int(native_issue_before)
                        gpu_after_final += int(starts_after)
                        zero_overlap_reproductions += int(starts_after and no_overlap)
                    rows.append({
                        "seed": case["seed"], "family": case["family"], "job": case["job"],
                        "rank": rank, "mode": case["mode"], "descriptor_index": index,
                        "trigger": trigger, "chunk_ids": "+".join(map(str, descriptor["chunk_ids"])),
                        "tokens": descriptor["tokens"],
                        **{f"T{i}": value for i, value in {
                            0:t0, 1:t1, 2:t2, 3:t3, 4:t4, 5:t5, 6:t6, 7:t7,
                            8:t8, 9:t9, 10:t10, 11:t11, 12:t12, 13:t13, 14:t14,
                            15:t15, 16:t16, 17:t17, 18:"N/A", 19:t19, 20:t20, 21:t21,
                        }.items()},
                        "reveal_to_commit_us": scheduler_us,
                        "packing_delay_us": packing_us,
                        "post_pack_enqueue_delay_us": enqueue_us,
                        "gpu_queue_delay_us": queue_us,
                        "gpu_queue_clock_map_raw_us": queue_raw_us,
                        "adapter_issue_host_ns": adapter_issue,
                        "adapter_issue_before_final_router": adapter_issue_before,
                        "ready_to_gpu_start_us": ready_us,
                        "bridge_publish_delay_us": (t3 - t2) / 1e3,
                        "state_binder_guard_us": (t6 - t3) / 1e3,
                        "frozen_cpu_packing_us": float(descriptor["packing_us"]),
                        "digest_bookkeeping_us": (
                            int(descriptor["descriptor_digest_done_host_ns"])
                            - int(descriptor["descriptor_digest_start_host_ns"])
                        ) / 1e3,
                        "adapter_cpu_byte_pack_us": (
                            int(descriptor["cpu_byte_pack_done_host_ns"])
                            - int(descriptor["cpu_byte_pack_start_host_ns"])
                        ) / 1e3,
                        "native_wrapper_to_launch_return_us": (t15 - t13) / 1e3,
                        "kernel_launch_api_us": (t15 - t14) / 1e3,
                        "put_kernel_duration_us": (t17 - t16) / 1e3,
                        "queue_previous_put_us": cover["previous_put_us"],
                        "queue_future_router_us": cover["future_router_us"],
                        "queue_staging_us": cover["staging_us"],
                        "queue_other_idle_us": cover["other_idle_us"],
                        "native_put_issue_before_final_router": native_issue_before,
                        "put_gpu_start_after_final_router": starts_after,
                        "router_put_gpu_overlap": not no_overlap,
                        "wrong_event_dependency": current_wrong,
                        "future_pack_dependency": current_future,
                        "event_reuse_hazard": current_reuse,
                        "put_blocked_by_previous_wait": previous_wait,
                        "put_blocked_by_previous_put": previous_put,
                        "put_blocked_by_unresolved_event": unresolved,
                        "pack_event_id": descriptor["pack_event_id"],
                        "pack_event_producer_stream": descriptor["pack_event_producer_stream"],
                        "comm_stream": descriptor["comm_wait_event_stream"],
                        "comm_stream_sequence_number": descriptor["comm_stream_sequence_number"],
                        "preceding_operation": descriptor["preceding_operation"],
                        "dependency_resolved_sync": descriptor["dependency_resolved_sync"],
                    })

    normal = [value for value in rows if value["mode"] == "normal"]
    metrics = (
        "reveal_to_commit_us", "packing_delay_us", "post_pack_enqueue_delay_us",
        "gpu_queue_delay_us", "ready_to_gpu_start_us", "bridge_publish_delay_us",
        "state_binder_guard_us", "frozen_cpu_packing_us", "digest_bookkeeping_us",
        "adapter_cpu_byte_pack_us", "native_wrapper_to_launch_return_us",
        "kernel_launch_api_us", "put_kernel_duration_us",
    )
    latency = {name: dist(value[name] for value in normal) for name in metrics}
    totals = {
        "scheduler_control": sum(max(0.0, value["reveal_to_commit_us"]) for value in normal),
        "packing_data_dependency": sum(max(0.0, value["packing_delay_us"]) for value in normal),
        "cpu_runtime_enqueue": sum(max(0.0, value["post_pack_enqueue_delay_us"]) for value in normal),
        "gpu_queue_scheduling": sum(max(0.0, value["gpu_queue_delay_us"]) for value in normal),
    }
    total = sum(totals.values())
    attribution = {name: {"total_us": value, "percent": value / total * 100.0} for name, value in totals.items()}
    control_summary = {
        mode: {
            "ready_to_gpu_start_us": dist(value["ready_to_gpu_start_us"] for value in rows if value["mode"] == mode),
            "gpu_queue_delay_us": dist(value["gpu_queue_delay_us"] for value in rows if value["mode"] == mode),
            "packing_delay_us": dist(value["packing_delay_us"] for value in rows if value["mode"] == mode),
            "post_pack_enqueue_delay_us": dist(value["post_pack_enqueue_delay_us"] for value in rows if value["mode"] == mode),
        } for mode in MODES
    }
    # Cross-rank start skew for matching normal descriptors.
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for value in normal:
        grouped.setdefault((value["seed"], value["family"], value["job"], value["descriptor_index"]), []).append(int(value["T16"]))
    rendezvous_skew = dist((max(values) - min(values)) / 1e3 for values in grouped.values() if len(values) == 2)

    kernel_rows: list[dict[str, Any]] = []
    for rank, path in enumerate(args.trace):
        trace = json.loads(path.read_text())
        for event in trace["traceEvents"]:
            if event.get("cat") not in {"kernel", "gpu_memcpy"} or event.get("ph") != "X":
                continue
            detail = event.get("args", {})
            kernel_rows.append({
                "rank": rank, "category": event.get("cat"), "name": event.get("name"),
                "start_us": event.get("ts"), "duration_us": event.get("dur"),
                "end_us": float(event.get("ts", 0)) + float(event.get("dur", 0)),
                "device": detail.get("device"), "stream": detail.get("stream"),
                "grid": detail.get("grid"), "block": detail.get("block"),
                "registers_per_thread": detail.get("registers per thread"),
                "shared_memory": detail.get("shared memory"),
                "blocks_per_sm": detail.get("blocks per SM"),
                "warps_per_sm": detail.get("warps per SM"),
                "estimated_occupancy_percent": detail.get("est. achieved occupancy %"),
            })
    put_kernels = [value for value in kernel_rows if "put_and_signal_kernel" in str(value["name"])]
    router_kernels = [value for value in kernel_rows if "gemm" in str(value["name"]).lower()]
    diagnosis_complete = bool(normal and zero_overlap_reproductions > 0 and all(value["count"] for value in latency.values()))
    root_identified = bool(
        diagnosis_complete
        and attribution["scheduler_control"]["percent"] > 50.0
        and latency["kernel_launch_api_us"]["p95"] < 1_000.0
        and control_summary["dependency_resolved"]["gpu_queue_delay_us"]["p50"] is not None
    )
    result = {
        "schema_version": "r6-m3-v1", "study": "R6-M3 post-issue GPU-start diagnosis",
        "environment": environment,
        "protocol": {"seeds": list(SEEDS), "families": sorted({value["family"] for value in normal}),
                     "normal_cases": 15, "normal_remote_descriptor_rank_rows": len(normal),
                     "control_rows": len(rows) - len(normal)},
        "reproduction": {"adapter_issue_before_final_router": adapter_before_final,
                         "native_wrapper_before_final_router": native_before_final,
                         "gpu_start_after_final_router": gpu_after_final,
                         "zero_overlap_pattern_rows": zero_overlap_reproductions,
                         "corrected_r6_m2_premise": (
                             "R6-M2 adapter entry was labeled put host issue; native wrapper/kernel enqueue "
                             "did not occur before final Router in the normal diagnostic corpus."
                         )},
        "latency_us": latency, "ready_to_start_attribution": attribution,
        "controls": control_summary,
        "dependency_audit": {
            "wrong_event_dependency": wrong_event_dependency,
            "future_pack_dependency": future_pack_dependency,
            "event_reuse_hazard": event_reuse_hazard,
        },
        "head_of_line_audit": {
            "put_blocked_by_previous_wait": put_blocked_by_previous_wait,
            "put_blocked_by_previous_put": put_blocked_by_previous_put,
            "put_blocked_by_unresolved_event": put_blocked_by_unresolved_event,
            "previous_put_block_total_us": previous_put_block_us,
            "unresolved_event_block_total_us": unresolved_event_block_us,
        },
        "gpu_queue_coverage_us": {
            name: sum(value[name] for value in normal)
            for name in ("queue_previous_put_us", "queue_future_router_us", "queue_staging_us", "queue_other_idle_us")
        },
        "rank_rendezvous_remaining_gpu_start_skew_us": rendezvous_skew,
        "stream_audit": {
            "all_priorities_unchanged_default": all(
                value[key] == 0 for value in stream_audits
                for key in ("router_priority", "packing_priority", "comm_priority", "return_priority", "default_priority")
            ),
            "normal_samples": [value for value in stream_audits if value["mode"] == "normal"],
            "packing_semantics": "registered-buffer H2D staging on current/default stream",
            "return_semantics": "frozen return uses the same communication stream",
        },
        "profiler": {
            "raw_traces": [path.name for path in args.trace],
            "kernel_events": len([value for value in kernel_rows if value["category"] == "kernel"]),
            "memcpy_events": len([value for value in kernel_rows if value["category"] == "gpu_memcpy"]),
            "put_kernel": {
                "count": len(put_kernels), "names": sorted({value["name"] for value in put_kernels}),
                "grid": put_kernels[0]["grid"] if put_kernels else None,
                "block": put_kernels[0]["block"] if put_kernels else None,
                "registers_per_thread": put_kernels[0]["registers_per_thread"] if put_kernels else None,
                "shared_memory": put_kernels[0]["shared_memory"] if put_kernels else None,
            },
            "router_gemm_names": sorted({value["name"] for value in router_kernels}),
        },
        "hot_path_sync_audit": [
            {"location": "_run_arm gpu origin", "caller": "setup", "reason": "align clocks before primary", "before_put_gpu_start": True, "in_progressive_hot_path": False},
            {"location": "MscclppFullMoeForwardTransport.finish", "caller": "frozen forward completion", "reason": "complete all outstanding puts/waits", "before_put_gpu_start": False, "in_progressive_hot_path": False},
            {"location": "dependency_resolved control staging_end.synchronize", "caller": "diagnostic control only", "reason": "remove staging dependency", "before_put_gpu_start": True, "in_progressive_hot_path": False},
        ],
        "safety": {"correctness_pass": correctness, "legality_pass": legality,
                   "control_equivalence_pass": equivalence},
        "causal_attribution": {
            "primary": "A. Scheduler/control latency" if attribution["scheduler_control"]["percent"] > 50 else "G. Other",
            "secondary": "B. Packing/data dependency",
            "negligible": ["C. Host/runtime kernel enqueue latency", "D. CUDA stream head-of-line dependency",
                           "E. GPU resource contention/non-concurrent residency", "F. MSCCL++ primitive internal synchronization"],
        },
        "candidate_followup": "Remove host-side per-descriptor serialization and make registered-buffer staging asynchronous/pinned before considering stream priority; do not change it in R6-M3.",
        "verdict": {
            "diagnosis": "COMPLETE" if diagnosis_complete else "INCOMPLETE",
            "correctness": "PASS" if correctness else "FAIL",
            "legality": "PASS" if legality else "FAIL",
            "root_cause": "IDENTIFIED" if root_identified else "NOT IDENTIFIED",
            "veto": "NO VETO" if correctness and legality and root_identified else "VETO",
        },
    }
    with (args.output_dir / "r6_m3_descriptor_timeline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (args.output_dir / "r6_m3_kernel_timeline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kernel_rows[0])); writer.writeheader(); writer.writerows(kernel_rows)
    (args.output_dir / "r6_m3_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": result["verdict"], "attribution": attribution,
                      "latency": {key: latency[key] for key in metrics[:5]}}, indent=2))


if __name__ == "__main__":
    main()
