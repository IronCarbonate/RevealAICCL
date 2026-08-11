#!/usr/bin/env python3
"""Train three V1 seeds, run paired ablations, and apply the V1 stage gate."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


TAIL_METRICS = (
    "completion_steps_p95",
    "completion_steps_cvar90",
    "completion_steps_cvar95",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Formal multi-seed V1 ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--train-families", nargs="+", default=[
        "smooth_ar", "alternating_burst", "moving_hotspot", "sparse_switching"
    ])
    parser.add_argument("--heldout-families", nargs="+", default=[
        "bimodal", "heavy_tail_clipped"
    ])
    parser.add_argument("--num-train-sequences", type=int, default=10)
    parser.add_argument("--num-validation-sequences", type=int, default=2)
    parser.add_argument("--num-eval-sequences", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.0)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--epsilon-mean", type=float, default=0.20)
    parser.add_argument("--epsilon-var", type=float, default=0.30)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-target", type=int, default=500)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-seed", type=int, default=1_000_042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-mean-degradation", type=float, default=0.02)
    parser.add_argument("--output-dir", default="outputs/moment_v1/formal")
    parser.add_argument("--checkpoint-dir", default="checkpoints/moment_v1/formal")
    return parser.parse_args()


def run(command, log_path):
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def confidence_interval(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "num_seeds": len(array)}
    half_width = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "num_seeds": len(array),
    }


def aggregate_seed_metrics(seed_summaries):
    result = {"overall": {}, "by_family": {}}
    methods = sorted(seed_summaries[0]["overall"])
    families = sorted(seed_summaries[0]["by_family"])
    metric_names = sorted(seed_summaries[0]["overall"][methods[0]])
    for method in methods:
        result["overall"][method] = {
            metric: confidence_interval([
                summary["overall"][method][metric] for summary in seed_summaries
            ])
            for metric in metric_names
            if metric != "num_collectives"
        }
    for family in families:
        result["by_family"][family] = {}
        for method in methods:
            result["by_family"][family][method] = {
                metric: confidence_interval([
                    summary["by_family"][family][method][metric]
                    for summary in seed_summaries
                ])
                for metric in metric_names
                if metric != "num_collectives"
            }
    return result


def stable_improvement(seed_summaries, scope, family, lhs, rhs, metric):
    deltas = []
    for summary in seed_summaries:
        group = summary[scope] if family is None else summary[scope][family]
        deltas.append(float(group[lhs][metric] - group[rhs][metric]))
    return {
        "deltas": deltas,
        "median_delta": float(np.median(deltas)),
        "positive_seeds": int(sum(delta > 0 for delta in deltas)),
        "stable": sum(delta > 0 for delta in deltas) >= math.ceil(2 * len(deltas) / 3)
        and float(np.median(deltas)) > 0,
    }


def formal_gate(seed_summaries, families, max_mean_degradation):
    family_evidence = {}
    all_families_tail_improve = True
    all_families_context_sensitive = True
    for family in families:
        tail = {
            metric: stable_improvement(
                seed_summaries, "by_family", family, "baseline", "full", metric
            )
            for metric in TAIL_METRICS
        }
        context = {
            metric: stable_improvement(
                seed_summaries, "by_family", family, "shuffled", "full", metric
            )
            for metric in ("completion_steps_mean",) + TAIL_METRICS
        }
        tail_ok = any(item["stable"] for item in tail.values())
        context_ok = any(item["stable"] for item in context.values())
        all_families_tail_improve &= tail_ok
        all_families_context_sensitive &= context_ok
        family_evidence[family] = {
            "tail_vs_baseline": tail,
            "correct_context_vs_shuffled": context,
            "tail_ok": tail_ok,
            "context_ok": context_ok,
        }

    baseline_means = [s["overall"]["baseline"]["completion_steps_mean"] for s in seed_summaries]
    full_means = [s["overall"]["full"]["completion_steps_mean"] for s in seed_summaries]
    mean_degradation = float(np.mean(full_means) / np.mean(baseline_means) - 1.0)
    legality_ok = all(
        method["legality_rate"] == 1.0
        for summary in seed_summaries
        for method in summary["overall"].values()
    )
    go = (
        all_families_tail_improve
        and all_families_context_sensitive
        and mean_degradation <= max_mean_degradation
        and legality_ok
    )
    return {
        "status": "GO" if go else "NO_GO",
        "formal": len(seed_summaries) >= 3,
        "family_evidence": family_evidence,
        "mean_degradation_relative": mean_degradation,
        "mean_degradation_limit": max_mean_degradation,
        "legality_ok": legality_ok,
        "criteria": (
            "Every held-out family must show a positive tail delta in at least "
            "two-thirds of seeds, correct context must stably beat shuffled "
            "context, mean degradation must be within the limit, and legality must be 100%."
        ),
    }


def main():
    args = parse_args()
    if len(args.seeds) < 3:
        print("WARNING: fewer than three training seeds; result is preliminary", flush=True)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = []
    combined_rows = []

    shared_train = [
        "--topology", args.topology,
        "--train-families", *args.train_families,
        # Checkpoint selection uses disjoint sequences from training families.
        # Distribution-held-out families are consumed only by final evaluation.
        "--validation-families", *args.train_families,
        "--num-train-sequences", str(args.num_train_sequences),
        "--num-validation-sequences", str(args.num_validation_sequences),
        "--sequence-length", str(args.sequence_length),
        "--window-size", str(args.window_size),
        "--min-history", str(args.min_history),
        "--mean-level", str(args.mean_level),
        "--std-level", str(args.std_level),
        "--max-entry", str(args.max_entry),
        "--epsilon-mean", str(args.epsilon_mean),
        "--epsilon-var", str(args.epsilon_var),
        "--time-limit", str(args.time_limit),
        "--hidden-dim", str(args.hidden_dim),
        "--epochs", str(args.epochs),
        "--batch-target", str(args.batch_target),
        "--ppo-epochs", str(args.ppo_epochs),
        "--mini-batch-size", str(args.mini_batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
    ]

    for seed in args.seeds:
        seed_output = output_dir / f"seed_{seed}"
        seed_checkpoints = checkpoint_dir / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        for mode in ("baseline", "moment"):
            mode_checkpoint_dir = seed_checkpoints / mode
            command = [
                sys.executable, "scripts/train_moment_policy.py",
                "--policy-mode", mode,
                *shared_train,
                "--seed", str(seed),
                "--output-dir", str(mode_checkpoint_dir),
            ]
            run(command, seed_output / f"train_{mode}.log")

        eval_command = [
            sys.executable, "scripts/evaluate_moment_policy.py",
            "--baseline-checkpoint", str(seed_checkpoints / "baseline" / "baseline_best.pth"),
            "--moment-checkpoint", str(seed_checkpoints / "moment" / "moment_best.pth"),
            "--topology", args.topology,
            "--families", *args.heldout_families,
            "--num-sequences", str(args.num_eval_sequences),
            "--sequence-length", str(args.sequence_length),
            "--window-size", str(args.window_size),
            "--min-history", str(args.min_history),
            "--mean-level", str(args.mean_level),
            "--std-level", str(args.std_level),
            "--max-entry", str(args.max_entry),
            "--epsilon-mean", str(args.epsilon_mean),
            "--epsilon-var", str(args.epsilon_var),
            "--time-limit", str(args.time_limit),
            "--seed", str(args.eval_seed),
            "--device", args.device,
            "--max-mean-degradation", str(args.max_mean_degradation),
            "--output-dir", str(seed_output),
        ]
        run(eval_command, seed_output / "evaluate.log")
        summary = json.loads((seed_output / "v1_ablation_summary.json").read_text(encoding="utf-8"))
        summary["training_seed"] = seed
        seed_summaries.append(summary)
        with (seed_output / "v1_ablation_detail.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                combined_rows.append({"training_seed": seed, **row})

    aggregate = {
        "schema_version": 1,
        "config": vars(args),
        "per_seed": seed_summaries,
        "metrics_across_training_seeds": aggregate_seed_metrics(seed_summaries),
        "gate": formal_gate(seed_summaries, args.heldout_families, args.max_mean_degradation),
    }
    (output_dir / "v1_formal_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    with (output_dir / "v1_formal_detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    print(json.dumps({
        "metrics_across_training_seeds": aggregate["metrics_across_training_seeds"]["overall"],
        "gate": aggregate["gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
