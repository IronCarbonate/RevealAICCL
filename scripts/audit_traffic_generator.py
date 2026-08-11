#!/usr/bin/env python3
"""Run Phase-A diagnostics against the existing traffic generator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.evaluation.traffic_audit import audit_sequence, summarize_audits
from rlccl.traffic.long_horizon_generator import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
)
from rlccl.traffic.process_generator import TrafficProcessConfig, generate_traffic_sequence


ALL_FAMILIES = (*TrafficProcessConfig.FAMILIES, *LONG_HORIZON_FAMILIES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the existing moment-bounded traffic generator")
    parser.add_argument("--families", nargs="+", choices=ALL_FAMILIES, default=list(TrafficProcessConfig.FAMILIES))
    parser.add_argument("--generator", choices=("auto", "legacy", "long"), default="auto")
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64, 1024, 4096])
    parser.add_argument("--num-sequences", type=int, default=20, help="Sequences per family, length, and base seed")
    parser.add_argument("--short-window", type=int, default=16)
    parser.add_argument("--medium-window", type=int, default=128)
    parser.add_argument("--long-window", type=int, default=512)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242], help="Independent base seeds")
    parser.add_argument("--output-dir", default="outputs/traffic_audit")
    parser.add_argument("--device", default="cpu", help="Recorded for reproducibility; this NumPy audit runs on CPU")
    parser.add_argument("--save-matrices", action="store_true")
    parser.add_argument("--max-generation-attempts", type=int, default=100)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--mean-level", type=float, default=2.0)
    parser.add_argument("--std-level", type=float, default=1.0)
    parser.add_argument("--long-std-level", type=float, default=1.5)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--epsilon-mean", type=float, default=0.20)
    parser.add_argument("--epsilon-var", type=float, default=0.30)
    parser.add_argument("--topology-name", default="Rear4GPU")
    parser.add_argument("--max-period-lag", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU worker processes")
    parser.add_argument("--calibration-candidates", type=int, default=3)
    return parser.parse_args()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "versions": {"numpy": _version("numpy"), "torch": _version("torch"), "pytest": _version("pytest")},
        "git_commit": _git_commit(root),
        "git_commit_note": None if _git_commit(root) else "workspace is not a valid Git worktree",
        "command": [sys.executable, *sys.argv],
        "device_requested": args.device,
        "device_used": "cpu (NumPy)",
        "seed_semantics": (
            "--num-sequences is generated independently for every family, sequence length, "
            "and --seeds base; actual_seed = seed_base + sequence_index"
        ),
    }


def _detail_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "status": record["status"],
        "family": record["family"],
        "sequence_length": record["sequence_length"],
        "seed_base": record["seed_base"],
        "sequence_index": record["sequence_index"],
        "actual_seed": record["actual_seed"],
        "dynamics_variant": record.get("dynamics_variant"),
    }
    if record["status"] == "failure":
        row["error"] = record["error"]
        return row
    row.update(
        {
            "generation_seconds": record["generation"]["wall_time_seconds"],
            "generation_attempts": record["generation"]["attempts"],
            "total_mean": record["total_traffic"]["mean"],
            "total_std": record["total_traffic"]["std"],
            "total_p99": record["total_traffic"]["p99"],
            "total_max": record["total_traffic"]["max"],
            "exact_duplicate_ratio": record["temporal"]["exact_duplicate_ratio"],
            "detected_exact_period": record["temporal"]["periodicity"]["detected_exact_period"],
            "hotspot_migrations": record["temporal"]["hotspot_destination_migrations"],
            "mean_sparsity": record["spatial"]["off_diagonal_sparsity"]["mean"],
            "metrics_json": json.dumps(record, separators=(",", ":")),
        }
    )
    return row


def _audit_task(task: dict[str, Any]) -> dict[str, Any]:
    long_horizon = task["family"] in LONG_HORIZON_FAMILIES
    if task["generator"] == "legacy" and long_horizon:
        raise ValueError("long-horizon family requested with --generator legacy")
    if task["generator"] == "long" and not long_horizon:
        raise ValueError("legacy family requested with --generator long")
    if long_horizon:
        variant = (
            SAME_MOMENT_VARIANTS[task["sequence_index"] % len(SAME_MOMENT_VARIANTS)]
            if task["family"] == "same_moments_different_dynamics"
            else None
        )
        config = LongHorizonTrafficConfig(
            num_nodes=task["num_nodes"],
            sequence_length=task["sequence_length"],
            family=task["family"],
            seed=task["actual_seed"],
            mean_level=task["mean_level"],
            std_level=task["long_std_level"],
            max_entry=task["max_entry"],
            short_window=task["short_window"],
            medium_window=task["medium_window"],
            long_window=task["long_window"],
            dynamics_variant=variant,
            calibration_candidates=min(task["calibration_candidates"], task["max_generation_attempts"]),
            topology_name=task["topology_name"],
        )
    else:
        config = TrafficProcessConfig(
            num_nodes=task["num_nodes"],
            sequence_length=task["sequence_length"],
            window_size=task["short_window"],
            mean_level=task["mean_level"],
            std_level=task["std_level"],
            max_entry=task["max_entry"],
            epsilon_mean=task["epsilon_mean"],
            epsilon_var=task["epsilon_var"],
            family=task["family"],
            seed=task["actual_seed"],
            max_generation_attempts=task["max_generation_attempts"],
            topology_name=task["topology_name"],
        )
    started = time.perf_counter()
    try:
        sequence = (
            generate_long_horizon_sequence(config)
            if long_horizon
            else generate_traffic_sequence(config)
        )
        elapsed = time.perf_counter() - started
        record = audit_sequence(
            sequence,
            short_window=task["short_window"],
            medium_window=task["medium_window"],
            long_window=task["long_window"],
            max_period_lag=task["max_period_lag"],
            generation_seconds=elapsed,
        )
        record.update(
            {
                "status": "success",
                "seed_base": task["seed_base"],
                "sequence_index": task["sequence_index"],
                "actual_seed": task["actual_seed"],
                "dynamics_variant": sequence.metadata.get("dynamics_variant"),
            }
        )
        if task["save_matrices"]:
            matrix_dir = Path(task["output_dir"]) / "matrices"
            matrix_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = matrix_dir / (
                f"{task['family']}_L{task['sequence_length']}_base{task['seed_base']}_i{task['sequence_index']}.json"
            )
            matrix_path.write_text(json.dumps(sequence.to_dict()), encoding="utf-8")
            record["saved_matrix_path"] = str(matrix_path)
        return record
    except (RuntimeError, ValueError) as error:
        return {
            "status": "failure",
            "family": task["family"],
            "sequence_length": task["sequence_length"],
            "seed_base": task["seed_base"],
            "sequence_index": task["sequence_index"],
            "actual_seed": task["actual_seed"],
            "dynamics_variant": (
                SAME_MOMENT_VARIANTS[task["sequence_index"] % len(SAME_MOMENT_VARIANTS)]
                if task["family"] == "same_moments_different_dynamics"
                else None
            ),
            "generation_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    args = parse_args()
    if args.num_sequences <= 0:
        raise ValueError("--num-sequences must be positive")
    if args.max_generation_attempts <= 0:
        raise ValueError("--max-generation-attempts must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.calibration_candidates <= 0:
        raise ValueError("--calibration-candidates must be positive")
    if any(window <= 1 for window in (args.short_window, args.medium_window, args.long_window)):
        raise ValueError("audit windows must be greater than one")
    if any(length < args.short_window for length in args.sequence_lengths):
        raise ValueError("every sequence length must be at least --short-window")

    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for family in args.families:
        for length in args.sequence_lengths:
            for seed_base in args.seeds:
                for sequence_index in range(args.num_sequences):
                    actual_seed = seed_base + sequence_index
                    tasks.append(
                        {
                            **vars(args),
                            "family": family,
                            "sequence_length": length,
                            "seed_base": seed_base,
                            "sequence_index": sequence_index,
                            "actual_seed": actual_seed,
                        }
                    )

    if args.workers == 1:
        iterator = map(_audit_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(_audit_task, tasks, chunksize=1)
    records = []
    try:
        for record in iterator:
            records.append(record)
            print(
                f"{record['status']} family={record['family']} length={record['sequence_length']} "
                f"seed_base={record['seed_base']} actual_seed={record['actual_seed']}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown()

    metadata = _metadata(args, root)
    summary = {
        "schema_version": 1,
        "metadata": metadata,
        "audit_config": vars(args),
        **summarize_audits(records),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = [_detail_row(record) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "audit_detail.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for family in args.families:
        payload = {
            "schema_version": 1,
            "metadata": metadata,
            "family": family,
            "records": [record for record in records if record["family"] == family],
        }
        (output_dir / f"family_{family}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
