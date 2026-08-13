"""Aggregate the preregistered R6-M2 paired pilot and descriptor traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np


SEEDS = (13042, 13142, 13242)
ARMS = ("NCCL-D", "NCCL-P", "MSCCLPP-D", "MSCCLPP-P")


def _med(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(median(materialized)) if materialized else 0.0


def _bootstrap_median_ci(values: list[float], seed: int = 6202) -> list[float]:
    rng = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    samples = rng.choice(source, size=(10_000, source.size), replace=True)
    estimates = np.median(samples, axis=1)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def _arm_timing(rank_arms: list[dict[str, Any]]) -> dict[str, float]:
    timing = [arm["timing"] for arm in rank_arms]
    origin = min(value["first_router_launch_host_ns"] for value in timing)
    def endpoint(name: str) -> float:
        return (max(value[name] for value in timing) - origin) / 1_000.0
    return {
        "primary_makespan_us": endpoint("primary_done_host_ns"),
        "router_final_us": endpoint("final_router_host_ns"),
        "forward_done_us": endpoint("forward_done_host_ns"),
        "expert_done_us": endpoint("expert_done_host_ns"),
        "return_done_us": endpoint("return_done_host_ns"),
    }


def _descriptor_rows(
    seed: int, family: str, job: int, arm_name: str,
    rank_arms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    backend = "mscclpp" if arm_name.startswith("MSCCLPP") else "nccl"
    by_rank = [value["forward_descriptors"] for value in rank_arms]
    rows: list[dict[str, Any]] = []
    for descriptor_index in range(len(by_rank[0])):
        descriptors = [values[descriptor_index] for values in by_rank]
        ready = [value["router_chunk_ready_host_ns"] for value in descriptors]
        issue = [value["communication"]["payload_call_host_ns"] for value in descriptors]
        if backend == "mscclpp":
            gpu_start = [value["communication"]["put_kernel_start_host_ns"] for value in descriptors]
            gpu_end = [value["communication"]["put_kernel_end_host_ns"] for value in descriptors]
        else:
            gpu_start = [value["communication"]["payload_gpu_start_host_ns"] for value in descriptors]
            gpu_end = [value["communication"]["payload_gpu_end_host_ns"] for value in descriptors]
        router_overlap_us = 0.0
        for rank, arm in enumerate(rank_arms):
            router_start = arm["diagnostics"]["router_gpu_start_host_ns"]
            router_end = arm["diagnostics"]["router_gpu_end_host_ns"]
            router_overlap_us += max(0, min(gpu_end[rank], router_end) - max(gpu_start[rank], router_start)) / 1_000.0
        final_router = max(value["timing"]["final_router_host_ns"] for value in rank_arms)
        rows.append({
            "seed": seed, "family": family, "job": job, "arm": arm_name,
            "backend": backend, "mode": arm_name.rsplit("-", 1)[1],
            "descriptor_index": descriptor_index,
            "chunk_ids": "+".join(str(value) for value in descriptors[0]["chunk_ids"]),
            "tokens": descriptors[0]["tokens"],
            "ready_skew_us": (max(ready) - min(ready)) / 1_000.0,
            "issue_skew_us": (max(issue) - min(issue)) / 1_000.0,
            "gpu_start_skew_us": (max(gpu_start) - min(gpu_start)) / 1_000.0,
            "ready_to_issue_us": _med((i - r) / 1_000.0 for i, r in zip(issue, ready)),
            "issue_to_gpu_start_us": _med((g - i) / 1_000.0 for g, i in zip(gpu_start, issue)),
            "ready_to_gpu_start_us": _med((g - r) / 1_000.0 for g, r in zip(gpu_start, ready)),
            "gpu_comm_envelope_us": (max(gpu_end) - min(gpu_start)) / 1_000.0,
            "gpu_kernel_duration_us": _med((e - s) / 1_000.0 for s, e in zip(gpu_start, gpu_end)),
            "router_overlap_us": router_overlap_us,
            "router_overlap": router_overlap_us > 0.0,
            "rank_issues_before_final_router": sum(value < final_router for value in issue),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [json.loads((args.raw_dir / f"r6_m2_seed{seed}_host.json").read_text()) for seed in SEEDS]
    rank_maps: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    environment = payloads[0]["environment"]
    source_pass = all(value["pass"] for value in payloads)
    for payload in payloads:
        for rank_result in payload["rank_results"]:
            for pair in rank_result["pairs"]:
                key = (pair["seed"], pair["family"], pair["job"])
                rank_maps.setdefault(key, []).append(pair)

    pair_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    complete = source_pass and len(rank_maps) == 45 and all(len(values) == 2 for values in rank_maps.values())
    correctness = complete
    multiple_outstanding = True
    for (seed, family, job), pairs in sorted(rank_maps.items()):
        rank_equivalence = all(value["equivalence"]["pass"] for value in pairs)
        arm_timings: dict[str, dict[str, float]] = {}
        arm_descriptors: dict[str, list[dict[str, Any]]] = {}
        for arm_name in ARMS:
            rank_arms = [pair["arms"][arm_name] for pair in pairs]
            arm_timings[arm_name] = _arm_timing(rank_arms)
            arm_descriptors[arm_name] = _descriptor_rows(seed, family, job, arm_name, rank_arms)
            descriptor_rows.extend(arm_descriptors[arm_name])
            correctness &= all(
                arm["correctness"]["final_combine_correct"]
                and arm["correctness"]["token_integrity"]
                and all(
                    count == 0 for name, count in arm["correctness"].items()
                    if name not in {"final_combine_correct", "token_integrity"}
                )
                and arm["semantic"]["legal"] == arm["semantic"]["total"]
                and all(
                    count == 0 for name, count in arm["semantic"].items()
                    if name not in {"legal", "total"}
                )
                for arm in rank_arms
            )
            if arm_name.startswith("MSCCLPP"):
                multiple_outstanding &= all(
                    arm["diagnostics"]["forward_transport"]["multiple_outstanding"]
                    and arm["diagnostics"]["forward_transport"]["mscclpp_put_calls"] > 1
                    and arm["diagnostics"]["forward_transport"]["future_access"] == 0
                    and arm["diagnostics"]["forward_transport"]["unrevealed_access"] == 0
                    and arm["diagnostics"]["forward_transport"]["stale_action"] == 0
                    for arm in rank_arms
                )
        correctness &= rank_equivalence
        row: dict[str, Any] = {
            "seed": seed, "family": family, "job": job,
            "equivalence_pass": rank_equivalence,
        }
        for arm_name in ARMS:
            prefix = arm_name.lower().replace("-", "_")
            row.update({f"{prefix}_{key}": value for key, value in arm_timings[arm_name].items()})
            traces = arm_descriptors[arm_name]
            for name in (
                "ready_skew_us", "issue_skew_us", "gpu_start_skew_us",
                "ready_to_issue_us", "issue_to_gpu_start_us", "ready_to_gpu_start_us",
                "gpu_comm_envelope_us", "gpu_kernel_duration_us", "router_overlap_us",
            ):
                row[f"{prefix}_median_{name}"] = _med(value[name] for value in traces)
            row[f"{prefix}_router_overlap_descriptors"] = sum(value["router_overlap"] for value in traces)
            row[f"{prefix}_rank_issues_before_final_router"] = sum(
                value["rank_issues_before_final_router"] for value in traces
            )
        row["gain_nccl_us"] = (
            arm_timings["NCCL-D"]["primary_makespan_us"]
            - arm_timings["NCCL-P"]["primary_makespan_us"]
        )
        row["gain_mscclpp_us"] = (
            arm_timings["MSCCLPP-D"]["primary_makespan_us"]
            - arm_timings["MSCCLPP-P"]["primary_makespan_us"]
        )
        row["gain_delta_mscclpp_minus_nccl_us"] = row["gain_mscclpp_us"] - row["gain_nccl_us"]
        pair_rows.append(row)

    gain_nccl = [value["gain_nccl_us"] for value in pair_rows]
    gain_mscclpp = [value["gain_mscclpp_us"] for value in pair_rows]
    gain_delta = [value["gain_delta_mscclpp_minus_nccl_us"] for value in pair_rows]
    seed_medians = {
        str(seed): {
            "gain_nccl_us": _med(value["gain_nccl_us"] for value in pair_rows if value["seed"] == seed),
            "gain_mscclpp_us": _med(value["gain_mscclpp_us"] for value in pair_rows if value["seed"] == seed),
            "gain_delta_mscclpp_minus_nccl_us": _med(
                value["gain_delta_mscclpp_minus_nccl_us"] for value in pair_rows if value["seed"] == seed
            ),
        } for seed in SEEDS
    }
    families = sorted({value["family"] for value in pair_rows})
    family_medians = {
        family: {
            "gain_nccl_us": _med(value["gain_nccl_us"] for value in pair_rows if value["family"] == family),
            "gain_mscclpp_us": _med(value["gain_mscclpp_us"] for value in pair_rows if value["family"] == family),
            "gain_delta_mscclpp_minus_nccl_us": _med(
                value["gain_delta_mscclpp_minus_nccl_us"] for value in pair_rows if value["family"] == family
            ),
        } for family in families
    }
    arm_summary = {}
    for arm_name in ARMS:
        prefix = arm_name.lower().replace("-", "_")
        arm_summary[arm_name] = {
            "median_primary_makespan_us": _med(value[f"{prefix}_primary_makespan_us"] for value in pair_rows),
            "median_ready_skew_us": _med(value[f"{prefix}_median_ready_skew_us"] for value in pair_rows),
            "median_issue_skew_us": _med(value[f"{prefix}_median_issue_skew_us"] for value in pair_rows),
            "median_gpu_start_skew_us": _med(value[f"{prefix}_median_gpu_start_skew_us"] for value in pair_rows),
            "median_ready_to_gpu_start_us": _med(value[f"{prefix}_median_ready_to_gpu_start_us"] for value in pair_rows),
            "median_gpu_comm_envelope_us": _med(value[f"{prefix}_median_gpu_comm_envelope_us"] for value in pair_rows),
            "total_router_overlap_descriptors": sum(value[f"{prefix}_router_overlap_descriptors"] for value in pair_rows),
            "total_rank_issues_before_final_router": sum(value[f"{prefix}_rank_issues_before_final_router"] for value in pair_rows),
        }

    mechanism_pass = (
        _med(gain_delta) > 0.0
        and arm_summary["MSCCLPP-P"]["total_rank_issues_before_final_router"] > 0
        and arm_summary["MSCCLPP-P"]["median_ready_to_gpu_start_us"]
            < arm_summary["NCCL-P"]["median_ready_to_gpu_start_us"]
        and multiple_outstanding
    )
    performance_pass = _med(gain_mscclpp) > 0.0 and all(
        value["gain_mscclpp_us"] > 0.0 for value in seed_medians.values()
    )
    result = {
        "schema_version": "r6-m2-v1",
        "study": "R6-M2 MSCCL++ Progressive Pipeline Pilot",
        "environment": environment,
        "protocol": {
            "seeds": list(SEEDS), "families": families, "jobs_per_family": 3,
            "paired_cases": len(pair_rows), "full_moe_arm_runs": len(pair_rows) * len(ARMS),
            "primary_endpoint": "earliest Router launch through return/combine completion",
            "bootstrap_resamples": 10_000,
        },
        "paired_summary": {
            "median_gain_nccl_us": _med(gain_nccl),
            "median_gain_nccl_95ci_us": _bootstrap_median_ci(gain_nccl, 6202),
            "median_gain_mscclpp_us": _med(gain_mscclpp),
            "median_gain_mscclpp_95ci_us": _bootstrap_median_ci(gain_mscclpp, 6203),
            "median_gain_delta_mscclpp_minus_nccl_us": _med(gain_delta),
            "median_gain_delta_95ci_us": _bootstrap_median_ci(gain_delta, 6204),
            "positive_case_directions": {
                "NCCL": sum(value > 0.0 for value in gain_nccl),
                "MSCCLPP": sum(value > 0.0 for value in gain_mscclpp),
            },
        },
        "seed_medians": seed_medians,
        "family_medians": family_medians,
        "arm_diagnostics": arm_summary,
        "safety": {
            "complete": complete, "correctness_pass": correctness,
            "multiple_outstanding_pass": multiple_outstanding,
        },
        "verdict": {
            "correctness": "PASS" if correctness else "FAIL",
            "mechanism": "PASS" if mechanism_pass else "FAIL",
            "performance": "PASS" if performance_pass else "FAIL",
            "veto": "NO VETO" if correctness else "VETO",
        },
    }
    with (args.output_dir / "r6_m2_raw_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    with (args.output_dir / "r6_m2_descriptor_trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(descriptor_rows[0]))
        writer.writeheader(); writer.writerows(descriptor_rows)
    (args.output_dir / "r6_m2_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["verdict"], indent=2))


if __name__ == "__main__":
    main()
