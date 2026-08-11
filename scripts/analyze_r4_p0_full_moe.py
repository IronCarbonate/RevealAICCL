"""Analyze the preregistered R4-P0 primary host-timing corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_r3_a0_c0 import sha256_file  # noqa: E402
from scripts.run_r4_p0_full_moe import FAMILIES, JOBS_PER_FAMILY, PILOT_SEEDS  # noqa: E402


BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 10_000


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": int(data.size), "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)), "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def bootstrap_median(values: list[float]) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(data, size=(BOOTSTRAP_REPLICATES, data.size), replace=True)
    medians = np.median(samples, axis=1)
    return {
        "n": int(data.size), "median_us": float(np.median(data)),
        "ci95_lower_us": float(np.percentile(medians, 2.5)),
        "ci95_upper_us": float(np.percentile(medians, 97.5)),
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
    }


def _arm_results(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [rank_arm[arm] for row in rows for rank_arm in row["ranks"].values()]


def _descriptor_values(arms: list[dict[str, Any]], direction: str, field: str) -> list[float]:
    descriptors = "forward_descriptors" if direction == "forward" else "return_descriptors"
    values = []
    for arm in arms:
        for descriptor in arm[descriptors]:
            source = descriptor["communication"] if field in descriptor.get("communication", {}) else descriptor
            values.append(float(source[field]))
    return values


def _semantic_pass(arm: dict[str, Any]) -> bool:
    correctness = arm["correctness"]
    semantic = arm["semantic"]
    zero_correctness = (
        "forward_duplicate", "forward_cross_duplicate", "wrong_source", "wrong_expert",
        "wrong_destination", "wrong_return", "wrong_position", "corruption", "lost",
        "duplicate", "expert_output_mismatch",
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


def _stage_summary(arms: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "router_us": distribution(value["diagnostics"]["router_us"] for value in arms),
        "forward_stage_us": distribution(
            (value["timing"]["forward_done_host_ns"] - value["timing"]["first_router_launch_host_ns"]) / 1e3
            for value in arms
        ),
        "expert_compute_and_d2h_us": distribution(
            (value["timing"]["expert_done_host_ns"] - value["timing"]["forward_done_host_ns"]) / 1e3
            for value in arms
        ),
        "return_stage_us": distribution(
            (value["timing"]["return_done_host_ns"] - value["timing"]["return_start_host_ns"]) / 1e3
            for value in arms
        ),
        "actual_combine_us": distribution(value["diagnostics"]["actual_combine_us"] for value in arms),
        "forward_packing_us": distribution(_descriptor_values(arms, "forward", "packing_us")),
        "forward_count_exchange_us": distribution(_descriptor_values(arms, "forward", "count_exchange_us")),
        "return_packing_us": distribution(_descriptor_values(arms, "return", "packing_us")),
        "return_count_exchange_us": distribution(_descriptor_values(arms, "return", "count_exchange_us")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    host = json.loads(args.input.read_text(encoding="utf-8"))
    rows = host["pairs"]
    protocol = host["frozen_protocol"]
    if not args.allow_smoke:
        if tuple(protocol["seeds"]) != PILOT_SEEDS or tuple(protocol["families"]) != FAMILIES:
            raise RuntimeError("R4-P0 corpus does not match preregistration")
        if int(protocol["jobs_per_family"]) != JOBS_PER_FAMILY or len(rows) != 150:
            raise RuntimeError("R4-P0 corpus cardinality mismatch")
    deltas = [float(row["delta_us"]) for row in rows]
    primary = bootstrap_median(deltas)
    per_seed = {
        str(seed): {
            **distribution(row["delta_us"] for row in rows if row["seed"] == seed),
            "median_positive": float(np.median([row["delta_us"] for row in rows if row["seed"] == seed])) > 0,
        }
        for seed in protocol["seeds"]
    }
    per_family = {
        family: distribution(row["delta_us"] for row in rows if row["family"] == family)
        for family in protocol["families"]
    }
    all_arms = [arm for name in ("C", "D") for arm in _arm_results(rows, name)]
    correctness = {
        "paired_equivalence_100pct": all(row["pass"] for row in rows),
        "legality_and_token_integrity_100pct": all(_semantic_pass(arm) for arm in all_arms),
        "runtime_bfs_zero": all(arm["semantic"]["runtime_bfs_calls"] == 0 for arm in all_arms),
        "full_rebuild_zero": all(arm["semantic"]["full_rebuild_count"] == 0 for arm in all_arms),
        "future_access_zero": all(arm["semantic"]["future_access"] == 0 and arm["semantic"]["unrevealed_execution"] == 0 for arm in all_arms),
    }
    seed_gate = all(value["median_positive"] for value in per_seed.values())
    gate = {
        "paired_median_positive": primary["median_us"] > 0,
        "bootstrap_ci95_lower_positive": primary["ci95_lower_us"] > 0,
        "three_of_three_seed_medians_positive": seed_gate and len(per_seed) == 3,
        "correctness": all(correctness.values()),
    }
    gate["pass"] = all(gate.values())
    arms = {}
    for arm in ("C", "D"):
        arm_rows = _arm_results(rows, arm)
        arms[arm] = {
            "primary_full_moe_makespan_us": distribution(row[arm]["primary_makespan_us"] for row in rows),
            "full_reference_makespan_us": distribution(row[arm]["full_reference_makespan_us"] for row in rows),
            "stages": _stage_summary(arm_rows),
        }
    result = {
        "schema_version": 1, "study": "R4-P0 full-reference-MoE pilot analysis",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "frozen_protocol": protocol, "correctness": correctness,
        "primary_delta_D_minus_C_us": primary, "per_seed_delta_us": per_seed,
        "per_family_delta_us": per_family, "arms": arms, "gate": gate,
        "status": "R4_P0_PASS_PENDING_SUPERVISOR" if gate["pass"] else "R4_P0_FAIL_PENDING_SUPERVISOR",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "r4_p0_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = args.output_dir / "r4_p0_summary.csv"
    lines = ["scope,name,n,p50_us,p95_us,p99_us,max_us"]
    for scope, values in (("seed", per_seed), ("family", per_family)):
        for name, value in values.items():
            lines.append(f'{scope},{name},{value["n"]},{value["p50"]:.6f},{value["p95"]:.6f},{value["p99"]:.6f},{value["max"]:.6f}')
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "gate": gate, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
