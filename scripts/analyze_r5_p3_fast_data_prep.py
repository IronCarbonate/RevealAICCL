"""Analyze the preregistered R5-P3 fast data-preparation pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.analyze_r4_p0_full_moe import _stage_summary, distribution  # noqa: E402
from scripts.run_r3_a0_c0 import sha256_file  # noqa: E402
from scripts.run_r3_p0_profiled import FAMILIES  # noqa: E402
from scripts.run_r5_p3_fast_data_prep import ARMS, JOBS_PER_FAMILY, PILOT_SEEDS  # noqa: E402


BOOTSTRAP_REPLICATES = 10_000


def bootstrap(values: Iterable[float], seed: int, *, suffix: str = "us") -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True), axis=1,
    )
    return {
        "n": int(data.size), f"median_{suffix}": float(np.median(data)),
        f"ci95_lower_{suffix}": float(np.percentile(medians, 2.5)),
        f"ci95_upper_{suffix}": float(np.percentile(medians, 97.5)),
        "positive": int(np.count_nonzero(data > 0)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
    }


def grouped(rows: list[dict[str, Any]], key: str, value: Any, metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row[key] == value]
    return {
        **distribution(values), "positive": sum(item > 0 for item in values),
        "median_positive": float(np.median(values)) > 0,
    }


def _all_rank_arms(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [rank_arms[arm] for row in rows for rank_arms in row["ranks"].values()]


def _correct(arm: dict[str, Any]) -> bool:
    correctness, semantic = arm["correctness"], arm["semantic"]
    zero_correctness = tuple(
        key for key in correctness if key not in ("final_combine_correct", "token_integrity")
    )
    zero_semantic = (
        "runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution", "future_access",
        "duplicate_dispatch", "stale_dispatch", "candidate_divergences", "action_divergences",
        "checker_divergences", "holder_divergences",
    )
    return bool(
        all(correctness[key] == 0 for key in zero_correctness)
        and correctness["final_combine_correct"] and correctness["token_integrity"]
        and semantic["legal"] == semantic["total"]
        and all(semantic[key] == 0 for key in zero_semantic)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    host = json.loads(args.input.read_text(encoding="utf-8"))
    rows, protocol = host["pairs"], host["frozen_protocol"]
    if not args.allow_smoke:
        if tuple(protocol["seeds"]) != PILOT_SEEDS or tuple(protocol["families"]) != FAMILIES:
            raise RuntimeError("R5-P3 corpus mismatch")
        if protocol["jobs_per_family"] != JOBS_PER_FAMILY or len(rows) != 150:
            raise RuntimeError("R5-P3 corpus cardinality mismatch")
        if tuple(protocol["fast_arms"]) != ("E1", "D1"):
            raise RuntimeError("R5-P3 arm mismatch")

    contrasts = {
        "fast_E0_minus_E1": ("delta_fast_us", 20260818, "E0", "E1"),
        "optimized_progressive_D1_minus_E1": ("delta_progressive_us", 20260819, "D1", "E1"),
    }
    contrast_results = {}
    for name, (metric, seed, denominator, _) in contrasts.items():
        values = [float(row[metric]) for row in rows]
        relative = [
            100.0 * float(row[metric]) / float(row[denominator]["primary_makespan_us"])
            for row in rows
        ]
        contrast_results[name] = {
            "primary": bootstrap(values, seed),
            "paired_relative_makespan_reduction_pct": bootstrap(relative, seed, suffix="pct"),
            "per_seed": {
                str(value): grouped(rows, "seed", value, metric) for value in protocol["seeds"]
            },
            "per_family": {
                value: grouped(rows, "family", value, metric) for value in protocol["families"]
            },
        }

    precompute_inclusive = [
        float(row["E0"]["primary_makespan_us"])
        - float(row["E1"]["precompute_inclusive_makespan_us"])
        for row in rows
    ]
    all_arms = [arm for name in ARMS for arm in _all_rank_arms(rows, name)]
    equivalences = [value for row in rows for value in row["rank_equivalence"].values()]
    correctness = {
        "all_rank_arms_correct": all(_correct(arm) for arm in all_arms),
        "all_pair_equivalence": all(row["pass"] for row in rows),
        "byte_exact_fast_descriptors": all(
            value["byte_exact_forward_descriptors"] for value in equivalences
        ),
        "same_scheduler_expert_return": all(
            value["same_scheduler_actions"] and value["same_expert_batches_weights_outputs"]
            and value["same_return_descriptors"] for value in equivalences
        ),
        "final_outputs_equivalent": all(value["final_outputs_equivalent"] for value in equivalences),
        "runtime_bfs_full_rebuild_zero": all(
            arm["semantic"]["runtime_bfs_calls"] == 0
            and arm["semantic"]["full_rebuild_count"] == 0 for arm in all_arms
        ),
    }

    arms = {}
    for name in ARMS:
        arm_rows = _all_rank_arms(rows, name)
        descriptors = [value for arm in arm_rows for value in arm["forward_descriptors"]]
        communications = [value["communication"] for value in descriptors]
        arms[name] = {
            "primary_makespan_us": distribution(row[name]["primary_makespan_us"] for row in rows),
            "precompute_inclusive_makespan_us": distribution(
                row[name]["precompute_inclusive_makespan_us"] for row in rows
            ),
            "stages": _stage_summary(arm_rows),
            "data_prep": {
                "count_construction_us": distribution(value["count_construction_us"] for value in descriptors),
                "offset_construction_us": distribution(value["offset_construction_us"] for value in descriptors),
                "packing_us": distribution(value["packing_us"] for value in descriptors),
                "static_precompute_us_outside_primary": distribution(
                    value["diagnostics"]["data_prep"]["static_precompute_us_outside_primary"]
                    for value in arm_rows
                ),
                "mark_completed_total_us_inside_primary": distribution(
                    value["diagnostics"]["data_prep"]["mark_completed_total_us_inside_primary"]
                    for value in arm_rows
                ),
                "count_exchange_host_us": distribution(value["count_exchange_us"] for value in communications),
                "count_exchange_gpu_us": distribution(value["count_gpu_us"] for value in communications),
                "count_residual_wait_us": distribution(value["count_wait_us"] for value in communications),
                "prestarted_count_descriptors": sum(value["count_prestarted_before_packing"] for value in communications),
                "pinned_metadata_descriptors": sum(value["metadata_host_pinned"] for value in communications),
                "pinned_feature_descriptors": sum(value["values_host_pinned"] for value in communications),
                "descriptors": len(descriptors),
            },
        }

    packing_deltas, packing_speedups = [], []
    hidden_count_during_packing = []
    hidden_fraction = []
    for row in rows:
        for rank_arms in row["ranks"].values():
            for e0_descriptor, e1_descriptor in zip(
                rank_arms["E0"]["forward_descriptors"],
                rank_arms["E1"]["forward_descriptors"], strict=True,
            ):
                reference = float(e0_descriptor["packing_us"])
                fast = float(e1_descriptor["packing_us"])
                packing_deltas.append(reference - fast)
                packing_speedups.append(reference / max(fast, 1e-9))
                communication = e1_descriptor["communication"]
                overlap_ns = max(
                    0,
                    min(int(e1_descriptor["packing_done_host_ns"]), int(communication["count_visible_host_ns"]))
                    - max(int(e1_descriptor["packing_start_host_ns"]), int(communication["count_start_host_ns"])),
                )
                overlap_us = overlap_ns / 1e3
                hidden_count_during_packing.append(overlap_us)
                hidden_fraction.append(100.0 * overlap_us / max(fast, 1e-9))

    fast_primary = contrast_results["fast_E0_minus_E1"]["primary"]
    progressive_primary = contrast_results["optimized_progressive_D1_minus_E1"]["primary"]
    fast_seed_positive = all(
        value["median_positive"]
        for value in contrast_results["fast_E0_minus_E1"]["per_seed"].values()
    )
    progressive_seed_positive = all(
        value["median_positive"]
        for value in contrast_results["optimized_progressive_D1_minus_E1"]["per_seed"].values()
    )
    packing_distribution = distribution(packing_deltas)
    gates = {
        "correctness": {**correctness, "pass": all(correctness.values())},
        "packing_mechanism": {
            "median_reference_minus_fast_positive": packing_distribution["p50"] > 0,
            "byte_exact_descriptors": correctness["byte_exact_fast_descriptors"],
        },
        "fast_full_moe": {
            "median_positive": fast_primary["median_us"] > 0,
            "ci95_lower_positive": fast_primary["ci95_lower_us"] > 0,
            "three_of_three_seed_medians_positive": len(protocol["seeds"]) == 3 and fast_seed_positive,
        },
        "optimized_progressive": {
            "median_positive": progressive_primary["median_us"] > 0,
            "ci95_lower_positive": progressive_primary["ci95_lower_us"] > 0,
            "three_of_three_seed_medians_positive": len(protocol["seeds"]) == 3 and progressive_seed_positive,
        },
    }
    for value in (gates["packing_mechanism"], gates["fast_full_moe"], gates["optimized_progressive"]):
        value["pass"] = all(value.values())

    result = {
        "schema_version": 1, "study": "R5-P3 fast progressive data-preparation pilot analysis",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "frozen_protocol": protocol, "correctness": correctness,
        "contrasts": contrast_results,
        "precompute_inclusive_E0_minus_E1": bootstrap(precompute_inclusive, 20260820),
        "packing": {
            "paired_reference_minus_fast_us": packing_distribution,
            "paired_speedup_x": distribution(packing_speedups),
            "count_exchange_overlap_with_packing_us": distribution(hidden_count_during_packing),
            "packing_interval_overlapped_pct": distribution(hidden_fraction),
        },
        "arms": arms, "gates": gates,
        "max_abs_final_output_difference": max(
            difference for value in equivalences
            for difference in value["max_abs_output_difference"].values()
        ),
        "status": (
            "R5_P3_PASS_PENDING_SUPERVISOR"
            if gates["correctness"]["pass"] and gates["packing_mechanism"]["pass"]
            and gates["fast_full_moe"]["pass"] and gates["optimized_progressive"]["pass"]
            else "R5_P3_FAIL_PENDING_SUPERVISOR"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        "r5_p3_smoke_results.json" if args.allow_smoke else "r5_p3_results.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output),
        "status": result["status"], "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
