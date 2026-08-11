#!/usr/bin/env python3
"""Train baseline or V1 moment-conditioned policy on traffic sequences."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.config import get_config
from rlccl.envs.evaluator import load_topology_info
from rlccl.models import SlotLevelPolicy
from rlccl.training import (
    SequenceDatasetConfig,
    build_sequence_problems,
    evaluate_model,
    train_epoch,
)


SCHEMA_VERSION = 1


def load_trusted_checkpoint(path, device):
    """Load a checkpoint produced by this training script.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
    while our trusted checkpoint intentionally also stores optimizer/config
    state.  The fallback keeps compatibility with older PyTorch releases.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train V1 moment-conditioned policy")
    parser.add_argument("--policy-mode", choices=["baseline", "moment"], default="moment")
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument(
        "--train-families",
        nargs="+",
        default=["smooth_ar", "alternating_burst", "moving_hotspot", "sparse_switching"],
    )
    parser.add_argument(
        "--validation-families",
        nargs="+",
        default=["smooth_ar", "alternating_burst", "moving_hotspot", "sparse_switching"],
    )
    parser.add_argument(
        "--num-train-sequences",
        "--num-sequences",
        dest="num_train_sequences",
        type=int,
        default=8,
    )
    parser.add_argument("--num-validation-sequences", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.0)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--epsilon-mean", type=float, default=0.20)
    parser.add_argument("--epsilon-var", type=float, default=0.30)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-target", type=int, default=500)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="checkpoints/moment_v1")
    parser.add_argument("--resume")
    return parser.parse_args()


def dataset_config(args, families, count, seed):
    return SequenceDatasetConfig(
        families=tuple(families),
        num_sequences_per_family=count,
        sequence_length=args.sequence_length,
        window_size=args.window_size,
        min_history=args.min_history,
        mean_level=args.mean_level,
        std_level=args.std_level,
        max_entry=args.max_entry,
        epsilon_mean=args.epsilon_mean,
        epsilon_var=args.epsilon_var,
        seed=seed,
        time_limit=args.time_limit,
    )


def make_model(args, device):
    moment = args.policy_mode == "moment"
    return SlotLevelPolicy(
        node_feat_dim=12 if moment else 5,
        edge_feat_dim=2,
        cand_feat_dim=9 if moment else 5,
        chunk_feat_dim=2,
        hidden_dim=args.hidden_dim,
        global_moment_feat_dim=8 if moment else 0,
    ).to(device)


def checkpoint_payload(args, model, optimizer, epoch, best_score, train_config):
    return {
        "schema_version": SCHEMA_VERSION,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_score": best_score,
        "config": train_config,
        "policy_mode": args.policy_mode,
        "training_families": args.train_families,
        "validation_families": args.validation_families,
        "traffic_config": {
            "sequence_length": args.sequence_length,
            "window_size": args.window_size,
            "min_history": args.min_history,
            "mean_level": args.mean_level,
            "std_level": args.std_level,
            "max_entry": args.max_entry,
            "epsilon_mean": args.epsilon_mean,
            "epsilon_var": args.epsilon_var,
        },
        "moment_feature_normalization": {
            "z_clip": 10.0,
            "max_entry": args.max_entry,
            "node_mean_scale": "context_mean_node_load",
            "node_std_scale": "context_mean_node_std",
        },
        "command_args": vars(args),
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topology = load_topology_info(args.topology)
    print("Building temporally ordered train sequences...", flush=True)
    train_problems, _, train_records = build_sequence_problems(
        topology,
        dataset_config(args, args.train_families, args.num_train_sequences, args.seed),
    )
    print("Building disjoint held-out validation sequences...", flush=True)
    validation_problems, _, validation_records = build_sequence_problems(
        topology,
        dataset_config(
            args,
            args.validation_families,
            args.num_validation_sequences,
            args.seed + 1_000_000,
        ),
    )

    train_config = get_config()
    train_config.update(
        {
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "batch_target": args.batch_target,
            "ppo_epochs": args.ppo_epochs,
            "mini_batch_size": args.mini_batch_size,
            "policy_mode": args.policy_mode,
            "moment_max_entry": args.max_entry,
        }
    )
    model = make_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    best_score = -float("inf")

    if args.resume:
        checkpoint = load_trusted_checkpoint(args.resume, device)
        if checkpoint.get("policy_mode", args.policy_mode) != args.policy_mode:
            raise ValueError("Resume checkpoint policy mode mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", -float("inf")))

    print(
        f"mode={args.policy_mode} train_collectives={len(train_problems)} "
        f"validation_collectives={len(validation_problems)} "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {"train": train_records, "validation": validation_records}, indent=2
        ),
        encoding="utf-8",
    )

    for epoch in range(start_epoch, args.epochs):
        metrics = train_epoch(
            model,
            optimizer,
            train_problems,
            device,
            train_config,
            epoch,
            args.epochs,
        )
        policy_loss, value_loss, entropy, reward = metrics
        print(
            f"epoch={epoch + 1} policy={policy_loss:.6f} value={value_loss:.6f} "
            f"entropy={entropy:.6f} reward={reward:.6f}",
            flush=True,
        )
        if (epoch + 1) % args.eval_interval == 0:
            score, steps = evaluate_model(
                model, validation_problems, device, args.max_entry
            )
            score = float(score)
            steps = float(steps)
            print(
                f"validation epoch={epoch + 1} score={score:.6f} steps={steps:.6f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save(
                    checkpoint_payload(
                        args, model, optimizer, epoch, best_score, train_config
                    ),
                    output_dir / f"{args.policy_mode}_best.pth",
                )
        torch.save(
            checkpoint_payload(args, model, optimizer, epoch, best_score, train_config),
            output_dir / f"{args.policy_mode}_latest.pth",
        )

    torch.save(
        checkpoint_payload(
            args, model, optimizer, args.epochs - 1, best_score, train_config
        ),
        output_dir / f"{args.policy_mode}_final.pth",
    )
    print(f"DONE best_score={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
