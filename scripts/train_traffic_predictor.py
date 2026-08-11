#!/usr/bin/env python3
"""Train dependency-free history-only traffic-summary predictors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.envs.evaluator import load_topology_info
from rlccl.models import TrafficPredictorSuite
from rlccl.models.traffic_predictor import (
    build_history_examples,
    deterministic_group_coefficients,
)
from rlccl.traffic import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--families", nargs="+", default=list(LONG_HORIZON_FAMILIES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument("--sequences-per-seed", type=int, default=4)
    parser.add_argument("--train-sequences-per-seed", type=int, default=3)
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--history-window", type=int, default=16)
    parser.add_argument("--recent-steps", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.5)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--calibration-candidates", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis/predictor")
    return parser.parse_args()


def _specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sequences_per_seed < 2:
        raise ValueError("sequences-per-seed must be at least 2")
    if not 0 < args.train_sequences_per_seed < args.sequences_per_seed:
        raise ValueError("train-sequences-per-seed must leave at least one test sequence")
    unknown = sorted(set(args.families) - set(LONG_HORIZON_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown families: {unknown}")
    specs: list[dict[str, Any]] = []
    for family_index, family in enumerate(args.families):
        for seed_index, base_seed in enumerate(args.seeds):
            for sequence_index in range(args.sequences_per_seed):
                variant = None
                if family == "same_moments_different_dynamics":
                    variant = SAME_MOMENT_VARIANTS[
                        (seed_index + sequence_index) % len(SAME_MOMENT_VARIANTS)
                    ]
                actual_seed = (
                    int(base_seed)
                    + family_index * 1_000_000
                    + sequence_index * 10_000
                )
                specs.append(
                    {
                        "family": family,
                        "base_seed": int(base_seed),
                        "sequence_index": sequence_index,
                        "actual_seed": actual_seed,
                        "dynamics_variant": variant,
                        "split": (
                            "train"
                            if sequence_index < args.train_sequences_per_seed
                            else "test"
                        ),
                    }
                )
    return specs


def _generate(spec: dict[str, Any], config: dict[str, Any]) -> Any:
    return generate_long_horizon_sequence(
        LongHorizonTrafficConfig(
            num_nodes=int(config["num_nodes"]),
            sequence_length=int(config["sequence_length"]),
            family=spec["family"],
            seed=int(spec["actual_seed"]),
            mean_level=float(config["mean_level"]),
            std_level=float(config["std_level"]),
            max_entry=int(config["max_entry"]),
            dynamics_variant=spec.get("dynamics_variant"),
            calibration_candidates=int(config["calibration_candidates"]),
            topology_name=str(config["topology"]),
        )
    )


def main() -> None:
    args = parse_args()
    if min(args.sequence_length, args.history_window, args.recent_steps, args.min_history) <= 0:
        raise ValueError("Sequence/history arguments must be positive")
    if args.sequence_length <= max(args.recent_steps, args.min_history):
        raise ValueError("sequence-length is too short for the history settings")
    topology = load_topology_info(args.topology)
    group_coefficients = deterministic_group_coefficients(topology)
    config = {
        **vars(args),
        "num_nodes": int(topology.V),
        "group_count": int(group_coefficients.shape[0]),
        "bandwidth_group_target": (
            "offered load under deterministic shortest-path routing; not learned schedule load"
        ),
    }
    specs = _specs(args)
    train_specs = [item for item in specs if item["split"] == "train"]
    test_specs = [item for item in specs if item["split"] == "test"]
    train_ids = {(item["family"], item["actual_seed"]) for item in train_specs}
    test_ids = {(item["family"], item["actual_seed"]) for item in test_specs}
    if train_ids & test_ids:
        raise AssertionError("Train/test split overlaps complete sequences")

    print(f"Generating {len(train_specs)} complete training sequences...", flush=True)
    train_sequences = [_generate(spec, config) for spec in train_specs]
    train_examples = build_history_examples(
        train_sequences,
        group_coefficients=group_coefficients,
        history_window=args.history_window,
        recent_steps=args.recent_steps,
        min_history=args.min_history,
    )
    print(
        f"Fitting predictors on {len(train_examples)} history-only examples...",
        flush=True,
    )
    suite = TrafficPredictorSuite(
        num_nodes=topology.V,
        group_count=group_coefficients.shape[0],
        alpha=args.ridge_alpha,
    ).fit(train_examples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite.save(str(output_dir / "traffic_predictor.npz"))
    np.save(output_dir / "bandwidth_group_coefficients.npy", group_coefficients)
    manifest = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "numpy": np.__version__,
        "command": sys.argv,
        "config": config,
        "train_sequences": train_specs,
        "test_sequences": test_specs,
        "train_example_count": len(train_examples),
        "split_unit": "complete traffic sequence",
        "history_semantics": (
            "Every feature for target X_t is built only from X_0..X_(t-1)."
        ),
    }
    (output_dir / "predictor_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": str(output_dir / "traffic_predictor.npz"),
                "train_sequences": len(train_specs),
                "test_sequences_reserved": len(test_specs),
                "train_examples": len(train_examples),
                "target_dim": suite.target_dim,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
