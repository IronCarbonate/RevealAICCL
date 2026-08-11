"""Analyze the preregistered R4-F0 formal corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_r3_a0_c0 import sha256_file  # noqa: E402
from scripts.run_r3_p0_profiled import FAMILIES  # noqa: E402
from scripts.run_r4_f0_full_moe import FORMAL_JOBS_PER_FAMILY, FORMAL_SEEDS  # noqa: E402
from scripts.analyze_r4_p0_full_moe import (  # noqa: E402
    _arm_results, _semantic_pass, _stage_summary, distribution,
)


BOOTSTRAP_SEED = 20260814
BOOTSTRAP_REPLICATES = 10_000


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


def grouped(values: list[float]) -> dict[str, Any]:
    result = distribution(values)
    result["positive"] = sum(value > 0 for value in values)
    result["non_positive"] = sum(value <= 0 for value in values)
    result["positive_fraction"] = result["positive"] / len(values)
    result["median_positive"] = float(np.median(values)) > 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    host = json.loads(args.input.read_text(encoding="utf-8"))
    rows, protocol = host["pairs"], host["frozen_protocol"]
    if not args.allow_smoke:
        if tuple(protocol["seeds"]) != FORMAL_SEEDS or tuple(protocol["families"]) != FAMILIES:
            raise RuntimeError("R4-F0 corpus does not match preregistration")
        if int(protocol["jobs_per_family"]) != FORMAL_JOBS_PER_FAMILY or len(rows) != 300:
            raise RuntimeError("R4-F0 corpus cardinality mismatch")
        if not protocol["formal"]:
            raise RuntimeError("formal marker missing")
    deltas = [float(row["delta_us"]) for row in rows]
    primary = bootstrap_median(deltas)
    per_seed = {
        str(seed): grouped([float(row["delta_us"]) for row in rows if row["seed"] == seed])
        for seed in protocol["seeds"]
    }
    per_family = {
        family: grouped([float(row["delta_us"]) for row in rows if row["family"] == family])
        for family in protocol["families"]
    }
    all_arms = [arm for name in ("C", "D") for arm in _arm_results(rows, name)]
    correctness = {
        "paired_equivalence_100pct": all(row["pass"] for row in rows),
        "legality_and_token_integrity_100pct": all(_semantic_pass(arm) for arm in all_arms),
        "runtime_bfs_zero": all(arm["semantic"]["runtime_bfs_calls"] == 0 for arm in all_arms),
        "full_rebuild_zero": all(arm["semantic"]["full_rebuild_count"] == 0 for arm in all_arms),
        "future_access_zero": all(
            arm["semantic"]["future_access"] == 0 and arm["semantic"]["unrevealed_execution"] == 0
            for arm in all_arms
        ),
    }
    gate = {
        "paired_median_positive": primary["median_us"] > 0,
        "bootstrap_ci95_lower_positive": primary["ci95_lower_us"] > 0,
        "three_of_three_seed_medians_positive": len(per_seed) == 3 and all(value["median_positive"] for value in per_seed.values()),
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
        "schema_version": 1, "study": "R4-F0 formal full-reference-MoE analysis",
        "source": {"path": str(args.input), "sha256": sha256_file(args.input)},
        "frozen_protocol": protocol, "correctness": correctness,
        "primary_delta_D_minus_C_us": primary,
        "corpus_delta_distribution_us": grouped(deltas),
        "per_seed_delta_us": per_seed, "per_family_delta_us": per_family,
        "arms": arms, "gate": gate,
        "status": "R4_F0_PASS_PENDING_SUPERVISOR" if gate["pass"] else "R4_F0_FAIL_PENDING_SUPERVISOR",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("r4_f0_smoke_results.json" if args.allow_smoke else "r4_f0_results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = args.output_dir / ("r4_f0_smoke_summary.csv" if args.allow_smoke else "r4_f0_summary.csv")
    lines = ["scope,name,n,positive,non_positive,p50_us,p95_us,p99_us,max_us"]
    for scope, values in (("seed", per_seed), ("family", per_family)):
        for name, value in values.items():
            lines.append(f'{scope},{name},{value["n"]},{value["positive"]},{value["non_positive"]},{value["p50"]:.6f},{value["p95"]:.6f},{value["p99"]:.6f},{value["max"]:.6f}')
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "gate": gate, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
