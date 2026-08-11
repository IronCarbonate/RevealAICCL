#!/usr/bin/env python3
"""Run paired V1 baseline/moment held-out sequence evaluation."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.envs.evaluator import load_topology_info
from rlccl.evaluation import (
    build_shuffled_context_map,
    evaluate_sequence_policy,
    summarize_rows,
)
from rlccl.models import SlotLevelPolicy
from rlccl.training import SequenceDatasetConfig, build_sequence_problems


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V1 moment ablations")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--moment-checkpoint", required=True)
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument(
        "--families", nargs="+", default=["bimodal", "heavy_tail_clipped"]
    )
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.0)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--epsilon-mean", type=float, default=0.20)
    parser.add_argument("--epsilon-var", type=float, default=0.30)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1_000_042)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/moment_v1")
    parser.add_argument(
        "--max-mean-degradation",
        type=float,
        default=0.02,
        help="Maximum relative mean completion degradation allowed by the provisional gate",
    )
    return parser.parse_args()


def load_policy(path, expected_mode, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mode = checkpoint.get("policy_mode", expected_mode)
    if mode != expected_mode:
        raise ValueError(f"Checkpoint {path} has policy_mode={mode}, expected {expected_mode}")
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 64))
    moment = expected_mode == "moment"
    model = SlotLevelPolicy(
        node_feat_dim=12 if moment else 5,
        edge_feat_dim=2,
        cand_feat_dim=9 if moment else 5,
        chunk_feat_dim=2,
        hidden_dim=hidden_dim,
        global_moment_feat_dim=8 if moment else 0,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, checkpoint


def grouped_summary(rows):
    result = {"overall": {}}
    methods = sorted({row["method"] for row in rows})
    families = sorted({row["family"] for row in rows})
    for method in methods:
        result["overall"][method] = summarize_rows(
            row for row in rows if row["method"] == method
        )
    result["by_family"] = {}
    for family in families:
        result["by_family"][family] = {}
        for method in methods:
            result["by_family"][family][method] = summarize_rows(
                row
                for row in rows
                if row["method"] == method and row["family"] == family
            )
    return result


def provisional_gate(summary, max_mean_degradation):
    baseline = summary["overall"]["baseline"]
    full = summary["overall"]["full"]
    shuffled = summary["overall"]["shuffled"]
    tail_keys = (
        "completion_steps_p95",
        "completion_steps_cvar90",
        "completion_steps_cvar95",
    )
    tail_improvements = {
        key: baseline[key] - full[key] for key in tail_keys
    }
    mean_limit = baseline["completion_steps_mean"] * (1.0 + max_mean_degradation)
    legality_ok = all(
        method["legality_rate"] == 1.0 for method in summary["overall"].values()
    )
    shuffled_not_equal = any(
        abs(shuffled[key] - full[key]) > 1e-12
        for key in ("completion_steps_mean",) + tail_keys
    )
    return {
        "status": "PROVISIONAL_GO"
        if any(value > 0 for value in tail_improvements.values())
        and full["completion_steps_mean"] <= mean_limit
        and legality_ok
        and shuffled_not_equal
        else "NO_GO",
        "note": "A formal stable-improvement claim still requires at least three training seeds.",
        "tail_improvements_vs_baseline": tail_improvements,
        "mean_degradation_relative": (
            full["completion_steps_mean"] / baseline["completion_steps_mean"] - 1.0
        ),
        "legality_ok": legality_ok,
        "shuffled_not_equal": shuffled_not_equal,
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.num_sequences < 2:
        raise ValueError("At least two sequences are required for shuffled-context evaluation")
    device = torch.device(args.device)
    topology = load_topology_info(args.topology)
    dataset = SequenceDatasetConfig(
        families=tuple(args.families),
        num_sequences_per_family=args.num_sequences,
        sequence_length=args.sequence_length,
        window_size=args.window_size,
        min_history=args.min_history,
        mean_level=args.mean_level,
        std_level=args.std_level,
        max_entry=args.max_entry,
        epsilon_mean=args.epsilon_mean,
        epsilon_var=args.epsilon_var,
        seed=args.seed,
        time_limit=args.time_limit,
    )
    problems, _, records = build_sequence_problems(topology, dataset)
    shuffled = build_shuffled_context_map(problems)
    baseline, baseline_checkpoint = load_policy(
        args.baseline_checkpoint, "baseline", device
    )
    moment, moment_checkpoint = load_policy(args.moment_checkpoint, "moment", device)

    rows = evaluate_sequence_policy(
        baseline, problems, device, context_mode="baseline", moment_max_entry=args.max_entry
    )
    for context_mode in ("mean_only", "full", "shuffled"):
        rows.extend(
            evaluate_sequence_policy(
                moment,
                problems,
                device,
                context_mode=context_mode,
                moment_max_entry=args.max_entry,
                shuffled_contexts=shuffled,
            )
        )

    summary = grouped_summary(rows)
    summary["gate"] = provisional_gate(summary, args.max_mean_degradation)
    summary["evaluation_config"] = vars(args)
    summary["dataset_records"] = records
    summary["checkpoint_epochs"] = {
        "baseline": baseline_checkpoint.get("epoch"),
        "moment": moment_checkpoint.get("epoch"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_dir / "v1_ablation_detail.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "v1_ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"overall": summary["overall"], "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
