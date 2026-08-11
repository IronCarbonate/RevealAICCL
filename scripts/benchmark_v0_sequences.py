#!/usr/bin/env python3
"""Benchmark the current-demand-only V0 policy on moment-bounded sequences.

This script does not feed moment context into SlotLevelPolicy.  It measures the
baseline policy on paired traffic matrices while separately timing the
history-only SlidingMomentEstimator, which is the implemented pre-V1 boundary.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.envs.decoder import SlotDecoder
from rlccl.envs.evaluator import build_problem_from_scenario, evaluate_schedule, load_topology_info
from rlccl.envs.problem import compute_received_chunks
from rlccl.models import SlotLevelPolicy
from rlccl.traffic.matrix_utils import traffic_matrix_to_scenario
from rlccl.traffic.moment_estimator import SlidingMomentEstimator
from rlccl.traffic.moment_validation import validate_sequence_moment_bounds
from rlccl.traffic.process_generator import TrafficProcessConfig, generate_traffic_sequence


DEFAULT_FAMILIES = list(TrafficProcessConfig.FAMILIES)


def parse_args():
    parser = argparse.ArgumentParser(description="V0 paired sequence performance benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
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
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output-dir", default="outputs/performance")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cvar(values, alpha):
    array = np.asarray(values, dtype=np.float64)
    threshold = np.quantile(array, alpha)
    tail = array[array >= threshold]
    return float(tail.mean())


def summarize_rows(rows):
    completion = np.asarray([row["completion_steps"] for row in rows], dtype=np.float64)
    synthesis = np.asarray([row["synthesis_ms"] for row in rows], dtype=np.float64)
    estimator = np.asarray([row["estimator_ms"] for row in rows], dtype=np.float64)
    total_seconds = max(float(synthesis.sum()) / 1000.0, 1e-12)
    return {
        "num_collectives": len(rows),
        "completion_steps_mean": float(completion.mean()),
        "completion_steps_median": float(np.median(completion)),
        "completion_steps_p95": float(np.percentile(completion, 95)),
        "completion_steps_p99": float(np.percentile(completion, 99)),
        "completion_steps_cvar90": cvar(completion, 0.90),
        "completion_steps_cvar95": cvar(completion, 0.95),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "legality_rate": float(np.mean([row["legal"] for row in rows])),
        "synthesis_ms_mean": float(synthesis.mean()),
        "synthesis_ms_median": float(np.median(synthesis)),
        "synthesis_ms_p95": float(np.percentile(synthesis, 95)),
        "synthesis_ms_p99": float(np.percentile(synthesis, 99)),
        "estimator_ms_mean": float(estimator.mean()),
        "estimator_ms_p95": float(np.percentile(estimator, 95)),
        "synthesis_throughput_collectives_per_second": len(rows) / total_seconds,
    }


def build_problem(matrix, topology_info, time_limit):
    scenario = traffic_matrix_to_scenario(matrix)
    problem = build_problem_from_scenario(
        V=topology_info.V,
        E=topology_info.E,
        edges=topology_info.edges,
        capacities=topology_info.capacities,
        shared_constraints=topology_info.shared_constraints,
        scenario=scenario,
        T=time_limit,
    )
    problem.topology_info = topology_info
    return problem


def run_collective(model, decoder, problem, device):
    state = problem.initial_state.copy()
    demands = problem.demands.copy()
    schedule = []
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(problem.T):
            slot, _, _, _, _, _ = decoder.decode_slot(
                model, state, demands, step, problem.T, train=False
            )
            schedule.append(slot)
            received = compute_received_chunks(
                slot, problem.topology_info.edge_dst, problem.V
            )
            state = np.maximum(state, received)
            demands = demands * (1 - received)
            if not np.any(demands):
                break
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    score, error = evaluate_schedule(schedule, problem)
    timeout = bool(np.any(demands))
    completion_steps = problem.T if timeout else len(schedule)
    return {
        "completion_steps": int(completion_steps),
        "score": float(score),
        "synthesis_ms": float(elapsed_ms),
        "timeout": timeout,
        "legal": error == "",
        "error": error,
    }


def generate_sequences(args, num_nodes):
    sequences = []
    generation_rows = []
    for family_index, family in enumerate(args.families):
        for sequence_index in range(args.num_sequences):
            sequence_seed = args.seed + family_index * 10000 + sequence_index
            config = TrafficProcessConfig(
                num_nodes=num_nodes,
                sequence_length=args.sequence_length,
                window_size=args.window_size,
                mean_level=args.mean_level,
                std_level=args.std_level,
                max_entry=args.max_entry,
                epsilon_mean=args.epsilon_mean,
                epsilon_var=args.epsilon_var,
                family=family,
                seed=sequence_seed,
                topology_name=args.topology,
            )
            start = time.perf_counter()
            sequence = generate_traffic_sequence(config)
            generation_ms = (time.perf_counter() - start) * 1000.0
            diagnostics = validate_sequence_moment_bounds(sequence)
            if not diagnostics["passed"]:
                raise RuntimeError(f"Generated sequence failed validation: {sequence.sequence_id}")
            sequences.append(sequence)
            generation_rows.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "family": family,
                    "generation_ms": generation_ms,
                    "max_mean_error": diagnostics["max_mean_error"],
                    "max_var_error": diagnostics["max_var_error"],
                }
            )
            print(
                f"GENERATED family={family} sequence={sequence_index + 1}/{args.num_sequences} "
                f"mean_err={diagnostics['max_mean_error']:.4f} "
                f"var_err={diagnostics['max_var_error']:.4f}",
                flush=True,
            )
    return sequences, generation_rows


def load_model(checkpoint, device):
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    hidden_dim = int(config.get("hidden_dim", 128))
    model = SlotLevelPolicy(
        node_feat_dim=5,
        edge_feat_dim=2,
        cand_feat_dim=5,
        chunk_feat_dim=2,
        hidden_dim=hidden_dim,
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse_args()
    if args.num_sequences <= 0 or args.sequence_length <= 0:
        raise ValueError("num-sequences and sequence-length must be positive")
    if args.min_history <= 0 or args.min_history > args.window_size:
        raise ValueError("min-history must be in [1, window-size]")
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    invalid_families = sorted(set(args.families) - set(DEFAULT_FAMILIES))
    if invalid_families:
        raise ValueError(f"Unsupported families: {invalid_families}")
    if "cuda" in args.devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    torch.set_num_threads(args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topology_info = load_topology_info(args.topology)
    sequences, generation_rows = generate_sequences(args, topology_info.V)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    detail_rows = []

    for device_name in args.devices:
        device = torch.device(device_name)
        model = load_model(checkpoint, device)
        decoder = SlotDecoder(topology_info)

        # Warm up model kernels and decoder device caches outside timed samples.
        warm_problem = build_problem(sequences[0].matrices[0], topology_info, args.time_limit)
        run_collective(model, decoder, warm_problem, device)

        for sequence in sequences:
            estimator = SlidingMomentEstimator(
                topology_info.V, args.window_size, args.min_history
            )
            for step, matrix in enumerate(sequence.matrices):
                estimator_start = time.perf_counter()
                context = estimator.get_context(matrix, sequence.mean_ref, sequence.var_ref)
                estimator_ms = (time.perf_counter() - estimator_start) * 1000.0
                problem = build_problem(matrix, topology_info, args.time_limit)
                result = run_collective(model, decoder, problem, device)
                estimator.update(matrix)
                row = {
                    "device": device_name,
                    "sequence_id": sequence.sequence_id,
                    "family": sequence.family,
                    "sequence_seed": sequence.seed,
                    "step": step,
                    "history_length": context.history_length,
                    "confidence": context.confidence,
                    "mean_drift": context.mean_drift,
                    "var_drift": context.var_drift,
                    "estimator_ms": estimator_ms,
                    **result,
                }
                detail_rows.append(row)
            print(
                f"BENCHMARKED device={device_name} sequence={sequence.sequence_id} "
                f"collectives={len(sequence.matrices)}",
                flush=True,
            )

    detail_path = output_dir / "v0_sequence_benchmark_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    grouped = {}
    for device_name in args.devices:
        grouped[device_name] = {}
        device_rows = [row for row in detail_rows if row["device"] == device_name]
        grouped[device_name]["overall"] = summarize_rows(device_rows)
        for family in args.families:
            family_rows = [row for row in device_rows if row["family"] == family]
            grouped[device_name][family] = summarize_rows(family_rows)

    paired = {}
    if set(args.devices) >= {"cpu", "cuda"}:
        cpu = {
            (row["sequence_id"], row["step"]): row
            for row in detail_rows
            if row["device"] == "cpu"
        }
        cuda = {
            (row["sequence_id"], row["step"]): row
            for row in detail_rows
            if row["device"] == "cuda"
        }
        keys = sorted(set(cpu) & set(cuda))
        paired = {
            "num_pairs": len(keys),
            "completion_match_rate": float(
                np.mean([cpu[key]["completion_steps"] == cuda[key]["completion_steps"] for key in keys])
            ),
            "legality_match_rate": float(
                np.mean([cpu[key]["legal"] == cuda[key]["legal"] for key in keys])
            ),
            "cuda_synthesis_speedup_vs_cpu": float(
                np.mean([cpu[key]["synthesis_ms"] for key in keys])
                / np.mean([cuda[key]["synthesis_ms"] for key in keys])
            ),
        }

    summary = {
        "benchmark": "v0_current_demand_only",
        "preliminary": True,
        "policy_uses_moment_context": False,
        "model_path": str(Path(args.model_path).resolve()),
        "model_sha256": file_sha256(args.model_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "config": vars(args),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "generation": {
            "num_sequences": len(sequences),
            "mean_generation_ms": float(np.mean([row["generation_ms"] for row in generation_rows])),
            "max_mean_error": float(max(row["max_mean_error"] for row in generation_rows)),
            "max_var_error": float(max(row["max_var_error"] for row in generation_rows)),
        },
        "results": grouped,
        "paired_cpu_cuda": paired,
    }
    summary_path = output_dir / "v0_sequence_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"DETAIL_CSV={detail_path}")
    print(f"SUMMARY_JSON={summary_path}")


if __name__ == "__main__":
    main()
