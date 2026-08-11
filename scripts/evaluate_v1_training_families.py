#!/usr/bin/env python3
"""Evaluate rebuilt V1 checkpoints on disjoint sequences from training families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument(
        "--families",
        nargs="+",
        default=["smooth_ar", "alternating_burst", "moving_hotspot", "sparse_switching"],
    )
    parser.add_argument("--num-sequences", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.0)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--epsilon-mean", type=float, default=0.20)
    parser.add_argument("--epsilon-var", type=float, default=0.30)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=2_000_042)
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heldout-eval-dir")
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis/training_family_eval")
    return parser.parse_args()


def _run(command: list[str], log_path: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def _read_rows(path: Path, training_seed: int) -> list[dict[str, str | int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {"training_seed": training_seed, **row}
            for row in csv.DictReader(handle)
        ]


def _write_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_sequences <= 0:
        raise ValueError("num-sequences must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows: list[dict[str, str | int]] = []
    for training_seed in args.training_seeds:
        seed_dir = output_dir / f"seed_{training_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(args.checkpoint_dir) / f"seed_{training_seed}"
        command = [
            sys.executable,
            "scripts/evaluate_moment_policy.py",
            "--baseline-checkpoint",
            str(checkpoint_dir / "baseline" / "baseline_best.pth"),
            "--moment-checkpoint",
            str(checkpoint_dir / "moment" / "moment_best.pth"),
            "--topology",
            args.topology,
            "--families",
            *args.families,
            "--num-sequences",
            str(args.num_sequences),
            "--sequence-length",
            str(args.sequence_length),
            "--window-size",
            str(args.window_size),
            "--min-history",
            str(args.min_history),
            "--mean-level",
            str(args.mean_level),
            "--std-level",
            str(args.std_level),
            "--max-entry",
            str(args.max_entry),
            "--epsilon-mean",
            str(args.epsilon_mean),
            "--epsilon-var",
            str(args.epsilon_var),
            "--time-limit",
            str(args.time_limit),
            "--seed",
            str(args.eval_seed),
            "--device",
            args.device,
            "--output-dir",
            str(seed_dir),
        ]
        _run(command, seed_dir / "evaluate.log")
        training_rows.extend(
            _read_rows(seed_dir / "v1_ablation_detail.csv", training_seed)
        )
    _write_rows(output_dir / "training_family_detail.csv", training_rows)

    combined = list(training_rows)
    if args.heldout_eval_dir:
        heldout_dir = Path(args.heldout_eval_dir)
        for training_seed in args.training_seeds:
            combined.extend(
                _read_rows(
                    heldout_dir / f"seed_{training_seed}" / "v1_ablation_detail.csv",
                    training_seed,
                )
            )
        _write_rows(output_dir / "combined_bucket_detail.csv", combined)
    manifest = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "command": sys.argv,
        "config": vars(args),
        "split_unit": "complete sequence",
        "evaluation_seed": args.eval_seed,
        "training_family_rows": len(training_rows),
        "combined_rows": len(combined),
        "note": (
            "Evaluation sequences use base seed 2,000,042 and are disjoint from "
            "training, validation, and formal held-out evaluation sequences."
        ),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
