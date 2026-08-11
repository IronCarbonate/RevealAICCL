"""Analyze the preregistered R5-P2 progressive-return pilot."""

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
from scripts.run_r5_p2_progressive_return import ARMS, JOBS_PER_FAMILY, PILOT_SEEDS  # noqa: E402


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260817


def bootstrap(values: Iterable[float], seed: int) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True), axis=1,
    )
    return {
        "n": int(data.size), "median_us": float(np.median(data)),
        "ci95_lower_us": float(np.percentile(medians, 2.5)),
        "ci95_upper_us": float(np.percentile(medians, 97.5)),
        "positive": int(np.count_nonzero(data > 0)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
    }


def bootstrap_percent(values: Iterable[float], seed: int) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(
        rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True), axis=1,
    )
    return {
        "n": int(data.size), "median_pct": float(np.median(data)),
        "ci95_lower_pct": float(np.percentile(medians, 2.5)),
        "ci95_upper_pct": float(np.percentile(medians, 97.5)),
        "positive": int(np.count_nonzero(data > 0)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
    }


def grouped(rows: list[dict[str, Any]], key: str, value: Any) -> dict[str, Any]:
    values = [float(row["delta_return_us"]) for row in rows if row[key] == value]
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
            raise RuntimeError("R5-P2 corpus mismatch")
        if protocol["jobs_per_family"] != JOBS_PER_FAMILY or len(rows) != 150:
            raise RuntimeError("R5-P2 corpus cardinality mismatch")
        if not protocol["full_size_expert_batches"] or protocol["return_descriptor_split_merge_reorder"]:
            raise RuntimeError("R5-P2 frozen execution mismatch")

    deltas = [float(row["delta_return_us"]) for row in rows]
    relative = [
        100.0 * float(row["delta_return_us"]) / float(row["E0"]["primary_makespan_us"])
        for row in rows
    ]
    primary = bootstrap(deltas, BOOTSTRAP_SEED)
    relative_primary = bootstrap_percent(relative, BOOTSTRAP_SEED)
    per_seed = {str(seed): grouped(rows, "seed", seed) for seed in protocol["seeds"]}
    per_family = {family: grouped(rows, "family", family) for family in protocol["families"]}

    all_arms = [arm for name in ARMS for arm in _all_rank_arms(rows, name)]
    p2_arms = _all_rank_arms(rows, "P2")
    e0_arms = _all_rank_arms(rows, "E0")
    equivalences = [value for row in rows for value in row["rank_equivalence"].values()]
    correctness = {
        "all_rank_arms_correct": all(_correct(arm) for arm in all_arms),
        "all_pair_equivalence": all(row["pass"] for row in rows),
        "same_full_expert_batches_shapes_count": all(
            value["same_expert_batches_shapes_count_weights_outputs"] for value in equivalences
        ),
        "same_return_descriptors_count_bytes": all(
            value["same_return_descriptors"] and value["same_return_descriptor_count"]
            and value["same_return_bytes"] for value in equivalences
        ),
        "final_outputs_equivalent": all(value["final_outputs_equivalent"] for value in equivalences),
        "no_future_return_access": all(value["no_future_return_access"] for value in equivalences),
        "runtime_bfs_full_rebuild_zero": all(
            arm["semantic"]["runtime_bfs_calls"] == 0
            and arm["semantic"]["full_rebuild_count"] == 0 for arm in all_arms
        ),
    }

    mechanism_by_seed = {}
    for seed in protocol["seeds"]:
        seed_arms = [
            rank_arms["P2"] for row in rows if row["seed"] == seed
            for rank_arms in row["ranks"].values()
        ]
        mechanism_by_seed[str(seed)] = {
            "gpu_start_before_final_expert": sum(
                arm["diagnostics"]["return_progression"]["gpu_start_before_final_expert"]
                for arm in seed_arms
            ),
            "gpu_complete_before_final_expert": sum(
                arm["diagnostics"]["return_progression"]["gpu_complete_before_final_expert"]
                for arm in seed_arms
            ),
            "hidden_before_final_expert_us": sum(
                arm["diagnostics"]["return_progression"]["hidden_before_final_expert_us"]
                for arm in seed_arms
            ),
        }
    mechanism = {
        "by_seed": mechanism_by_seed,
        "early_gpu_start_each_seed": all(
            value["gpu_start_before_final_expert"] > 0 for value in mechanism_by_seed.values()
        ),
        "positive_hidden_return_gpu_time": sum(
            arm["diagnostics"]["return_progression"]["hidden_before_final_expert_us"]
            for arm in p2_arms
        ) > 0,
    }

    expert_interval_diff = []
    expert_interval_relative = []
    for row in rows:
        for rank_arms in row["ranks"].values():
            p2_value = rank_arms["P2"]["diagnostics"]["expert_progression"]["expert_gpu_active_us"]
            e0_value = rank_arms["E0"]["diagnostics"]["expert_progression"]["expert_gpu_active_us"]
            expert_interval_diff.append(p2_value - e0_value)
            expert_interval_relative.append(100.0 * (p2_value - e0_value) / max(e0_value, 1e-9))

    gates = {
        "correctness": {**correctness, "pass": all(correctness.values())},
        "mechanism": {**mechanism, "pass": all(
            value for key, value in mechanism.items() if key != "by_seed"
        )},
        "performance": {
            "median_positive": primary["median_us"] > 0,
            "ci95_lower_positive": primary["ci95_lower_us"] > 0,
            "three_of_three_seed_medians_positive": len(protocol["seeds"]) == 3 and all(
                value["median_positive"] for value in per_seed.values()
            ),
        },
    }
    gates["performance"]["pass"] = all(gates["performance"].values())

    arms = {}
    for name in ARMS:
        arm_rows = _all_rank_arms(rows, name)
        descriptors = [value for arm in arm_rows for value in arm["return_descriptors"]]
        arms[name] = {
            "primary_makespan_us": distribution(row[name]["primary_makespan_us"] for row in rows),
            "stages": _stage_summary(arm_rows),
            "expert_gpu_interval_us": distribution(
                arm["diagnostics"]["expert_progression"]["expert_gpu_active_us"] for arm in arm_rows
            ),
            "return_gpu_duration_us": distribution(value["gpu_duration_us"] for value in descriptors),
            "return_dependency_wait_us": distribution(value["dependency_wait_us"] for value in descriptors),
            "return_tail_after_expert_gpu_us": distribution(
                arm["diagnostics"]["return_progression"]["tail_after_final_expert_gpu_us"]
                for arm in arm_rows
            ),
        }

    p2_descriptors = [value for arm in p2_arms for value in arm["return_descriptors"]]
    result = {
        "schema_version": 1, "study": "R5-P2 progressive return pilot analysis",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "frozen_protocol": protocol,
        "primary": {
            "E0_minus_P2": primary,
            "paired_relative_makespan_reduction_pct": relative_primary,
            "per_seed": per_seed, "per_family": per_family,
        },
        "correctness": correctness, "mechanism": mechanism, "gates": gates,
        "arms": arms,
        "progressive_return": {
            "descriptors": len(p2_descriptors),
            "dependency_size_distribution": distribution(
                len(value["expert_dependencies"]) for value in p2_descriptors
            ),
            "gpu_start_before_final_expert": sum(
                value["gpu_start_before_final_expert"] for value in p2_descriptors
            ),
            "gpu_complete_before_final_expert": sum(
                value["gpu_complete_before_final_expert"] for value in p2_descriptors
            ),
            "positive_hidden_descriptors": sum(
                value["hidden_before_final_expert_us"] > 0 for value in p2_descriptors
            ),
            "hidden_before_final_expert_us": distribution(
                arm["diagnostics"]["return_progression"]["hidden_before_final_expert_us"]
                for arm in p2_arms
            ),
            "hidden_fraction_of_return_gpu_duration_pct": distribution(
                100.0 * arm["diagnostics"]["return_progression"]["hidden_before_final_expert_us"]
                / max(sum(value["gpu_duration_us"] for value in arm["return_descriptors"]), 1e-9)
                for arm in p2_arms
            ),
            "expert_interval_paired_difference_us_P2_minus_E0": distribution(expert_interval_diff),
            "expert_interval_paired_change_pct_P2_vs_E0": distribution(expert_interval_relative),
            "max_abs_final_output_difference": max(
                value["max_abs_output_difference"] for value in equivalences
            ),
        },
        "status": (
            "R5_P2_PASS_PENDING_SUPERVISOR"
            if gates["correctness"]["pass"] and gates["mechanism"]["pass"]
            and gates["performance"]["pass"]
            else "R5_P2_FAIL_PENDING_SUPERVISOR"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        "r5_p2_smoke_results.json" if args.allow_smoke else "r5_p2_results.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "sha256": sha256_file(output),
        "status": result["status"], "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
