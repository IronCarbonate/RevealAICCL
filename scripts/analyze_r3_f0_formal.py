"""Canonical analysis for profiler-off R3-F0 primary plus CUPTI subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np


FORMAL_SEEDS = (5042, 5142, 5242)
FAMILIES = (
    "balanced", "skewed", "all_to_one_like", "zero_sized_pair",
    "multiple_progressive_shards",
)
TRIGGERS = (0, 1, 2, 3, 4, 5, 7)


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
        "count": int(array.size), "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def bootstrap(values: list[float], seed: int = 20260811) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.median(array[rng.integers(0, len(array), size=(10_000, len(array)))], axis=1)
    low, high = np.percentile(estimates, (2.5, 97.5))
    return {
        "count": len(values), "median_us": float(np.median(array)),
        "ci95_low_us": float(low), "ci95_high_us": float(high),
        "ci95_lower_positive": bool(low > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    protocol = primary["frozen_protocol"]
    if tuple(protocol["seeds"]) != FORMAL_SEEDS:
        raise ValueError("formal seed mismatch")
    if int(protocol["jobs_per_family"]) != 20 or bool(protocol["profiler_enabled"]):
        raise ValueError("formal primary methodology mismatch")
    if diagnostic["trace_audit"] and any(
        int(row["association_failures"]) != 0 for row in diagnostic["trace_audit"]
    ):
        raise ValueError("diagnostic trace association failure")

    pairs = primary["paired_rows"]
    if len(pairs) != 300:
        raise ValueError("formal pair count mismatch")
    deltas = [float(row["delta_us"]) for row in pairs]
    primary_gate = bootstrap(deltas)
    per_seed = {}
    for seed in FORMAL_SEEDS:
        rows = [row for row in pairs if int(row["seed"]) == seed]
        values = [float(row["delta_us"]) for row in rows]
        per_seed[str(seed)] = {
            "delta_us": distribution(values),
            "C_primary_us": distribution(float(row["C_primary_us"]) for row in rows),
            "D_primary_us": distribution(float(row["D_primary_us"]) for row in rows),
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

    all_arms = [
        pair[arm] for rank in primary["rank_results"] for pair in rank["pairs"]
        for arm in ("C", "D")
    ]
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
    count_detail: dict[str, Any] = {
        "overall": {arm: by_arm[arm]["delta_count_exchange_us"] for arm in ("C", "D")},
        "per_seed": {}, "per_chunk": {}, "per_rank": {},
    }
    for seed in FORMAL_SEEDS:
        count_detail["per_seed"][str(seed)] = {
            arm: distribution(
                descriptor["communication"]["count_completion_us"]
                for rank in primary["rank_results"] for pair in rank["pairs"]
                if int(pair["seed"]) == seed for descriptor in pair[arm]["descriptors"]
            ) for arm in ("C", "D")
        }
    for trigger in TRIGGERS:
        count_detail["per_chunk"][str(trigger)] = {
            arm: distribution(
                descriptor["communication"]["count_completion_us"]
                for rank in primary["rank_results"] for pair in rank["pairs"]
                for descriptor in pair[arm]["descriptors"]
                if int(descriptor["trigger_chunk"]) == trigger
            ) for arm in ("C", "D")
        }
    for rank_index in (0, 1):
        count_detail["per_rank"][str(rank_index)] = {
            arm: distribution(
                descriptor["communication"]["count_completion_us"]
                for rank in primary["rank_results"] if int(rank["rank"]) == rank_index
                for pair in rank["pairs"] for descriptor in pair[arm]["descriptors"]
            ) for arm in ("C", "D")
        }

    built, total, built_bytes, total_bytes = 0, 0, 0, 0
    for rank in primary["rank_results"]:
        for pair in rank["pairs"]:
            arm = pair["C"]
            for descriptor in arm["descriptors"]:
                total += 1; total_bytes += int(descriptor["bytes"])
                if int(descriptor["built_host_ns"]) < int(arm["final_router_host_ns"]):
                    built += 1; built_bytes += int(descriptor["bytes"])

    correctness = {
        **{f"primary_{key}": bool(value) for key, value in primary["correctness"].items()},
        **{f"diagnostic_{key}": bool(value) for key, value in diagnostic["correctness"].items()},
    }
    three_positive = all(value["median_positive"] for value in per_seed.values())
    gate_pass = bool(
        primary_gate["median_us"] > 0 and primary_gate["ci95_lower_positive"]
        and three_positive and all(correctness.values())
    )
    result = {
        "schema_version": 1, "study": "R3-F0 formal real variable-size A2Av validation",
        "status": "R3_F0_PASS_PENDING_SUPERVISOR" if gate_pass else "R3_F0_FAIL_PENDING_SUPERVISOR",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formal_primary": {
            **primary_gate, "three_of_three_seed_medians_positive": three_positive,
            "correctness_pass": all(correctness.values()), "pass": gate_pass,
            "paired_jobs": len(pairs), "profiler_enabled": False,
        },
        "per_seed": per_seed, "per_family": per_family,
        "latency_breakdown": by_arm, "count_exchange_detail": count_detail,
        "packing_hidden": {
            "descriptors_before_final": built,
            "descriptor_fraction_before_final": built / total,
            "bytes_fraction_before_final": built_bytes / total_bytes,
        },
        "diagnostic_subset": {
            "corpus_jobs": [0, 10], "paired_jobs": len(diagnostic["paired_rows"]),
            "excluded_from_primary": True,
            "a2av_gpu_completion": {
                arm: diagnostic["latency_breakdown"][arm]["a2av_gpu_completion_us"]
                for arm in ("C", "D")
            },
            "device": diagnostic["secondary"],
            "trace_audit": diagnostic["trace_audit"],
        },
        "correctness": correctness,
        "traffic": {
            "sent_received_token_records": sum(int(arm["total_sent_tokens"]) for arm in all_arms),
            "total_bytes": sum(int(arm["total_sent_bytes"]) for arm in all_arms),
        },
        "pilot_comparison": {
            "pilot_median_us": 958.144,
            "pilot_ci95_us": [49.412, 1889.688],
            "formal_reproduced_positive_gate": gate_pass,
        },
        "source_artifacts": {
            "primary": {"path": str(args.primary), "sha256": sha256_file(args.primary)},
            "diagnostic": {"path": str(args.diagnostic), "sha256": sha256_file(args.diagnostic)},
        },
        "recommendation": (
            "formal_evidence_for_real_uncertain_variable_size_A2Av_improvement"
            if gate_pass else "formal_improvement_not_established"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "r3_f0_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "r3_f0_paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output),
        "pass": gate_pass, "formal_primary": result["formal_primary"],
    }, indent=2))


if __name__ == "__main__":
    main()
