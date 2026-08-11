"""Analyze the preregistered R5-P1 progressive-expert pilot."""

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
from scripts.run_r5_p1_progressive_expert import (  # noqa: E402
    ARMS, EXPERT_BATCH_THRESHOLD, JOBS_PER_FAMILY, PILOT_SEEDS,
)


BOOTSTRAP_REPLICATES = 10_000


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True), axis=1)
    return {
        "n": int(data.size), "median_us": float(np.median(data)),
        "ci95_lower_us": float(np.percentile(medians, 2.5)),
        "ci95_upper_us": float(np.percentile(medians, 97.5)),
        "positive": int(np.count_nonzero(data > 0)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
    }


def grouped(rows: list[dict[str, Any]], key: str, value: Any, metric: str) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows if row[key] == value]
    return {**distribution(values), "positive": sum(item > 0 for item in values), "median_positive": float(np.median(values)) > 0}


def _all_rank_arms(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [rank_arms[arm] for row in rows for rank_arms in row["ranks"].values()]


def _correct(arm: dict[str, Any]) -> bool:
    correctness, semantic = arm["correctness"], arm["semantic"]
    zero_correctness = tuple(key for key in correctness if key not in ("final_combine_correct", "token_integrity"))
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
            raise RuntimeError("R5-P1 corpus mismatch")
        if protocol["jobs_per_family"] != JOBS_PER_FAMILY or len(rows) != 150:
            raise RuntimeError("R5-P1 corpus cardinality mismatch")
        if protocol["expert_batch_threshold"] != EXPERT_BATCH_THRESHOLD:
            raise RuntimeError("R5-P1 threshold mismatch")

    contrasts = {
        "expert_incremental_E0_minus_P": ("delta_expert_us", 20260815),
        "total_pipeline_D_minus_P": ("delta_pipeline_us", 20260816),
        "forward_control_D_minus_E0": ("delta_forward_us", 20260817),
    }
    contrast_results = {}
    for name, (metric, seed) in contrasts.items():
        values = [float(row[metric]) for row in rows]
        contrast_results[name] = {
            "primary": bootstrap(values, seed),
            "per_seed": {str(value): grouped(rows, "seed", value, metric) for value in protocol["seeds"]},
            "per_family": {value: grouped(rows, "family", value, metric) for value in protocol["families"]},
        }

    p_arms = _all_rank_arms(rows, "P")
    e0_arms = _all_rank_arms(rows, "E0")
    all_arms = [arm for name in ARMS for arm in _all_rank_arms(rows, name)]
    p_tasks = [task for arm in p_arms for task in arm["diagnostics"]["expert_progression"]["tasks"]]
    active_degradation = []
    for row in rows:
        for rank_arms in row["ranks"].values():
            p_active = rank_arms["P"]["diagnostics"]["expert_progression"]["expert_gpu_active_us"]
            e0_active = rank_arms["E0"]["diagnostics"]["expert_progression"]["expert_gpu_active_us"]
            active_degradation.append((p_active - e0_active) / e0_active * 100.0)

    hidden_by_seed = {}
    for seed in protocol["seeds"]:
        seed_arms = [
            rank_arms["P"] for row in rows if row["seed"] == seed
            for rank_arms in row["ranks"].values()
        ]
        hidden_by_seed[str(seed)] = {
            "pre_forward_batches": sum(
                task["start_us"] < arm["diagnostics"]["expert_progression"]["forward_gpu_end_us"]
                for arm in seed_arms for task in arm["diagnostics"]["expert_progression"]["tasks"]
            ),
            "completed_pre_forward_batches": sum(
                task["completed_before_forward"] for arm in seed_arms
                for task in arm["diagnostics"]["expert_progression"]["tasks"]
            ),
        }

    correctness = {
        "all_rank_arms_correct": all(_correct(arm) for arm in all_arms),
        "all_pair_equivalence": all(row["pass"] for row in rows),
        "expert_loss_duplicate_future_zero": all(
            arm["correctness"][key] == 0 for arm in all_arms
            for key in ("expert_execution_loss", "expert_execution_duplicate", "expert_future_access")
        ),
        "final_outputs_equivalent": all(
            value["final_outputs_equivalent"] for row in rows for value in row["rank_equivalence"].values()
        ),
        "runtime_bfs_full_rebuild_zero": all(
            arm["semantic"]["runtime_bfs_calls"] == 0 and arm["semantic"]["full_rebuild_count"] == 0
            for arm in all_arms
        ),
    }
    mechanism = {
        "pre_forward_batch_each_seed": all(value["pre_forward_batches"] > 0 for value in hidden_by_seed.values()),
        "positive_hidden_gpu_time": sum(
            arm["diagnostics"]["expert_progression"]["hidden_before_forward_us"] for arm in p_arms
        ) > 0,
    }
    expert_primary = contrast_results["expert_incremental_E0_minus_P"]["primary"]
    pipeline_primary = contrast_results["total_pipeline_D_minus_P"]["primary"]
    expert_seed_positive = all(
        value["median_positive"] for value in contrast_results["expert_incremental_E0_minus_P"]["per_seed"].values()
    )
    pipeline_seed_positive = all(
        value["median_positive"] for value in contrast_results["total_pipeline_D_minus_P"]["per_seed"].values()
    )
    gates = {
        "correctness_mechanism": {
            **correctness, **mechanism,
            "pass": all(correctness.values()) and all(mechanism.values()),
        },
        "incremental_expert_performance": {
            "median_positive": expert_primary["median_us"] > 0,
            "ci95_lower_positive": expert_primary["ci95_lower_us"] > 0,
            "three_of_three_seed_medians_positive": len(protocol["seeds"]) == 3 and expert_seed_positive,
        },
        "total_pipeline_performance": {
            "median_positive": pipeline_primary["median_us"] > 0,
            "ci95_lower_positive": pipeline_primary["ci95_lower_us"] > 0,
            "three_of_three_seed_medians_positive": len(protocol["seeds"]) == 3 and pipeline_seed_positive,
        },
    }
    for value in (gates["incremental_expert_performance"], gates["total_pipeline_performance"]):
        value["pass"] = all(value.values())

    arms = {}
    for arm in ARMS:
        arm_rows = _all_rank_arms(rows, arm)
        arms[arm] = {
            "primary_makespan_us": distribution(row[arm]["primary_makespan_us"] for row in rows),
            "stages": _stage_summary(arm_rows),
            "expert_gpu_active_us": distribution(
                value["diagnostics"]["expert_progression"]["expert_gpu_active_us"] for value in arm_rows
            ),
            "expert_tail_after_forward_gpu_us": distribution(
                value["diagnostics"]["expert_progression"]["tail_after_forward_gpu_us"] for value in arm_rows
            ),
        }

    progression = {
        "threshold": EXPERT_BATCH_THRESHOLD,
        "batch_size_distribution": distribution(task["batch_size"] for task in p_tasks),
        "batches": len(p_tasks), "gemm_launches": len(p_tasks) * 2,
        "full_threshold_batches": sum(task["batch_size"] == EXPERT_BATCH_THRESHOLD for task in p_tasks),
        "remainder_flush_batches": sum(task["batch_size"] < EXPERT_BATCH_THRESHOLD for task in p_tasks),
        "per_expert_batches": {
            str(expert): sum(task["expert"] == expert for task in p_tasks) for expert in range(4)
        },
        "hidden_before_router_gpu_us": distribution(
            arm["diagnostics"]["expert_progression"]["hidden_before_router_us"] for arm in p_arms
        ),
        "hidden_before_forward_gpu_us": distribution(
            arm["diagnostics"]["expert_progression"]["hidden_before_forward_us"] for arm in p_arms
        ),
        "hidden_fraction_of_expert_gpu_active": distribution(
            100.0 * arm["diagnostics"]["expert_progression"]["hidden_before_forward_us"]
            / max(arm["diagnostics"]["expert_progression"]["expert_gpu_active_us"], 1e-9)
            for arm in p_arms
        ),
        "pre_forward_batches": sum(task["start_us"] < arm["diagnostics"]["expert_progression"]["forward_gpu_end_us"] for arm in p_arms for task in arm["diagnostics"]["expert_progression"]["tasks"]),
        "completed_pre_forward_batches": sum(task["completed_before_forward"] for task in p_tasks),
        "completed_pre_forward_tokens": sum(task["batch_size"] for task in p_tasks if task["completed_before_forward"]),
        "pre_router_batches": sum(task["start_us"] < arm["diagnostics"]["expert_progression"]["router_gpu_end_us"] for arm in p_arms for task in arm["diagnostics"]["expert_progression"]["tasks"]),
        "completed_pre_router_tokens": sum(task["batch_size"] for task in p_tasks if task["completed_before_router"]),
        "by_seed": hidden_by_seed,
        "expert_efficiency_degradation_pct_P_vs_E0": distribution(active_degradation),
        "max_abs_final_output_difference": max(
            difference for row in rows for value in row["rank_equivalence"].values()
            for difference in value["max_abs_output_difference"].values()
        ),
    }

    result = {
        "schema_version": 1, "study": "R5-P1 progressive expert execution pilot analysis",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "frozen_protocol": protocol, "correctness": correctness, "mechanism": mechanism,
        "contrasts": contrast_results, "gates": gates, "arms": arms,
        "progressive_expert": progression,
        "status": (
            "R5_P1_PASS_PENDING_SUPERVISOR"
            if gates["correctness_mechanism"]["pass"] and gates["incremental_expert_performance"]["pass"]
            else "R5_P1_FAIL_PENDING_SUPERVISOR"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("r5_p1_smoke_results.json" if args.allow_smoke else "r5_p1_results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "status": result["status"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
