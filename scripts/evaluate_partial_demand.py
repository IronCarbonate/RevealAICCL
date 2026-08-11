#!/usr/bin/env python3
"""Paired full/partial-demand information-value experiment for V1 policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.envs.decoder import SlotDecoder
from rlccl.envs.evaluator import evaluate_schedule, load_topology_info
from rlccl.envs.problem import ProblemInstance, compute_received_chunks
from rlccl.evaluation.partial_demand import (
    OBSERVATION_MODES,
    PartialDemandObservation,
    build_partial_observation,
)
from rlccl.models import SlotLevelPolicy
from rlccl.traffic import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    SlidingMomentEstimator,
    generate_long_horizon_sequence,
    traffic_matrix_to_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--hide-ratios", nargs="+", type=float, default=[0.25, 0.50])
    parser.add_argument(
        "--observation-modes", nargs="+", default=list(OBSERVATION_MODES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument(
        "--training-seeds", nargs="+", type=int, default=[42, 142, 242]
    )
    parser.add_argument(
        "--families", nargs="+", default=list(LONG_HORIZON_FAMILIES)
    )
    parser.add_argument("--num-sequences-per-family", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis")
    parser.add_argument("--report", default="docs/PARTIAL_DEMAND_EXPERIMENT.md")
    return parser.parse_args()


def _load_policy(path: Path, mode: str, device: torch.device) -> tuple[Any, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("policy_mode", mode) != mode:
        raise ValueError(f"Checkpoint mode mismatch: {path}")
    hidden_dim = int(checkpoint.get("config", {}).get("hidden_dim", 64))
    moment = mode == "moment"
    model = SlotLevelPolicy(
        node_feat_dim=12 if moment else 5,
        edge_feat_dim=2,
        cand_feat_dim=9 if moment else 5,
        chunk_feat_dim=2,
        hidden_dim=hidden_dim,
        global_moment_feat_dim=8 if moment else 0,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequences(args: argparse.Namespace) -> list[Any]:
    if args.num_sequences_per_family <= 0:
        raise ValueError("num-sequences-per-family must be positive")
    result = []
    for family_index, family in enumerate(args.families):
        for seed_index, base_seed in enumerate(args.seeds):
            for sequence_index in range(args.num_sequences_per_family):
                variant = None
                if family == "same_moments_different_dynamics":
                    variant = SAME_MOMENT_VARIANTS[
                        (seed_index + sequence_index) % len(SAME_MOMENT_VARIANTS)
                    ]
                actual_seed = (
                    int(base_seed)
                    + 3_000_000
                    + family_index * 100_000
                    + sequence_index * 1_000
                )
                result.append(
                    generate_long_horizon_sequence(
                        LongHorizonTrafficConfig(
                            sequence_length=args.sequence_length,
                            family=family,
                            seed=actual_seed,
                            mean_level=2.0,
                            std_level=1.5,
                            max_entry=args.max_entry,
                            dynamics_variant=variant,
                            calibration_candidates=1,
                            topology_name=args.topology,
                        )
                    )
                )
    return result


def _problem(matrix: np.ndarray, sequence: Any, step: int, topology: Any, time_limit: int) -> Any:
    scenario = traffic_matrix_to_scenario(
        matrix,
        sequence_id=sequence.sequence_id,
        sequence_step=step,
        family=sequence.family,
    )
    return ProblemInstance(
        num_nodes=topology.V,
        num_chunks=scenario["C"],
        num_edges=topology.E,
        time_limit=time_limit,
        capacities=topology.capacities,
        topology=topology.edges,
        demands=np.asarray(scenario["demands"], dtype=np.int64),
        initial_state=np.asarray(scenario["initial_state"], dtype=np.int64),
        shared_constraints=topology.shared_constraints,
        topology_info=topology,
        traffic_matrix=matrix,
        scenario_type="all_to_all_v",
        sequence_id=sequence.sequence_id,
        sequence_step=step,
        metadata={"family": sequence.family},
    )


def _run(
    model: Any,
    problem: Any,
    device: torch.device,
    *,
    context: Any,
    observation: PartialDemandObservation | None,
    max_entry: int,
) -> dict[str, Any]:
    decoder = SlotDecoder(problem.topology_info)
    state = problem.initial_state.copy()
    true_demands = problem.demands.copy()
    observed_demands = (
        true_demands.astype(np.float32).copy()
        if observation is None
        else observation.observation_demands.copy()
    )
    observed_matrix = (
        problem.traffic_matrix if observation is None else observation.observed_matrix
    )
    schedule = []
    completion = problem.T
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for slot in range(problem.T):
            slot_matrix, _, _, _, _, _ = decoder.decode_slot(
                model,
                state,
                true_demands,
                slot,
                problem.T,
                train=False,
                moment_context=context,
                current_matrix=observed_matrix,
                moment_max_entry=max_entry,
                observation_demands=(
                    None if observation is None else observed_demands
                ),
            )
            schedule.append(slot_matrix)
            received = compute_received_chunks(
                slot_matrix, problem.topology_info.edge_dst, problem.topology_info.V
            )
            state = np.maximum(state, received)
            true_demands = true_demands * (1 - received)
            observed_demands = observed_demands * (1 - received)
            if not np.any(true_demands):
                completion = slot + 1
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    synthesis_ms = (time.perf_counter() - started) * 1000.0
    timeout = bool(np.any(true_demands))
    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=np.int64))
    score, error = evaluate_schedule(schedule, problem)
    return {
        "completion_steps": int(completion),
        "timeout": timeout,
        "legal": error == "",
        "evaluation_error": error,
        "score": float(score),
        "synthesis_ms": float(synthesis_ms),
    }


def _observation_conditions(
    args: argparse.Namespace,
    matrix: np.ndarray,
    true_demands: np.ndarray,
    actual_seed: int,
    step: int,
) -> list[tuple[str, float | None, PartialDemandObservation | None]]:
    result: list[tuple[str, float | None, PartialDemandObservation | None]] = [
        ("full", None, None)
    ]
    for mode_index, mode in enumerate(args.observation_modes):
        if mode not in OBSERVATION_MODES:
            raise ValueError(f"Unsupported observation mode: {mode}")
        ratios = args.hide_ratios if mode in {"random_entries", "partial_shards"} else [None]
        for ratio_index, ratio in enumerate(ratios):
            observation_seed = (
                actual_seed * 1_000_003 + step * 101 + mode_index * 17 + ratio_index
            ) % (2**32)
            observation = build_partial_observation(
                matrix,
                true_demands,
                mode=mode,
                hide_ratio=ratio,
                seed=observation_seed,
            )
            result.append((mode, observation.hide_ratio, observation))
    return result


def _bootstrap_ci(by_sequence: dict[str, list[float]], samples: int, seed: int) -> list[float]:
    values = np.asarray(
        [np.mean(by_sequence[name]) for name in sorted(by_sequence)], dtype=np.float64
    )
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        ratio = "none" if row["hide_ratio"] is None else f"{row['hide_ratio']:.2f}"
        grouped.setdefault((row["observation_mode"], ratio, row["method"]), []).append(row)
    conditions = []
    for (mode, ratio, method), items in sorted(grouped.items()):
        completion = np.asarray([item["completion_steps"] for item in items], dtype=np.float64)
        synthesis = np.asarray([item["synthesis_ms"] for item in items], dtype=np.float64)
        deltas = np.asarray([item["paired_completion_delta"] for item in items])
        by_sequence: dict[str, list[float]] = {}
        by_seed: dict[int, list[float]] = {}
        for item in items:
            by_sequence.setdefault(item["sequence_id"], []).append(
                item["paired_completion_delta"]
            )
            by_seed.setdefault(item["training_seed"], []).append(
                item["paired_completion_delta"]
            )
        seed_means = {str(seed): float(np.mean(value)) for seed, value in by_seed.items()}
        positive_seeds = sum(value > 0 for value in seed_means.values())
        ci = _bootstrap_ci(by_sequence, args.bootstrap_samples, 20260727 + len(conditions))
        conditions.append(
            {
                "observation_mode": mode,
                "hide_ratio": None if ratio == "none" else float(ratio),
                "method": method,
                "raw_sample_count": len(items),
                "independent_sequence_count": len(by_sequence),
                "training_seed_count": len(by_seed),
                "completion_mean": float(completion.mean()),
                "completion_median": float(np.median(completion)),
                "completion_p95": float(np.quantile(completion, 0.95)),
                "completion_p99": float(np.quantile(completion, 0.99)),
                "completion_cvar95": float(
                    completion[completion >= np.quantile(completion, 0.95)].mean()
                ),
                "synthesis_ms_mean": float(synthesis.mean()),
                "synthesis_ms_p95": float(np.quantile(synthesis, 0.95)),
                "legality_rate": float(np.mean([item["legal"] for item in items])),
                "timeout_rate": float(np.mean([item["timeout"] for item in items])),
                "paired_completion_delta_mean": float(deltas.mean()),
                "paired_completion_delta_ci95": ci,
                "positive_training_seeds": int(positive_seeds),
                "paired_delta_by_training_seed": seed_means,
                "stable_moment_benefit": bool(
                    method == "moment"
                    and float(deltas.mean()) > 0
                    and ci[0] > 0
                    and positive_seeds >= math.ceil(2 * len(by_seed) / 3)
                ),
            }
        )
    full_moment = next(
        item
        for item in conditions
        if item["observation_mode"] == "full" and item["method"] == "moment"
    )
    partial_moment = [
        item
        for item in conditions
        if item["observation_mode"] != "full" and item["method"] == "moment"
    ]
    return {
        "conditions": conditions,
        "full_moment_benefit": full_moment["stable_moment_benefit"],
        "partial_stable_benefit_conditions": [
            {
                "observation_mode": item["observation_mode"],
                "hide_ratio": item["hide_ratio"],
            }
            for item in partial_moment
            if item["stable_moment_benefit"]
        ],
        "overall_legality_rate": float(np.mean([row["legal"] for row in rows])),
        "overall_timeout_rate": float(np.mean([row["timeout"] for row in rows])),
    }


def _write_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    moments = [item for item in summary["conditions"] if item["method"] == "moment"]
    lines = [
        "# Partial-demand 信息价值实验",
        "",
        "## 语义",
        "",
        "策略 observation 与 ground-truth execution 明确分离：解码特征使用 partial observation，物理 state transition、真实 demand 清除、completion、timeout 和 legality 始终使用完整真实 X_t。partial observation 不能修改或新增真实 demand；imputed destination 只影响策略特征，任何实际传输仍必须通过原确定性 topology/capacity/shared-group 可行性约束。",
        "历史 context 在调度 X_t 前只由 X_0...X_{t-1} 更新；partial moment 的当前 z/global 特征使用 partial/proxy current matrix，而不是完整 X_t。",
        "random-entry 设置为保持既有 chunk action space 会暴露 chunk 数和 source ownership，但隐藏 destination entry；这是当前 V1 chunk 表示的明确局限。",
        "",
        "## Paired 结果",
        "",
        "正的 delta 表示 Moment 比相同 observation 的 baseline 少用 completion slot。",
        "",
        "| observation | hidden | sequences | Moment mean | paired delta | sequence bootstrap 95% CI | positive seeds | stable | legality | timeout | synthesis ms |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in moments:
        ci = item["paired_completion_delta_ci95"]
        hidden = "-" if item["hide_ratio"] is None else f"{item['hide_ratio']:.0%}"
        lines.append(
            f"| {item['observation_mode']} | {hidden} | {item['independent_sequence_count']} | "
            f"{item['completion_mean']:.4f} | {item['paired_completion_delta_mean']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {item['positive_training_seeds']}/{item['training_seed_count']} | "
            f"{'yes' if item['stable_moment_benefit'] else 'no'} | {item['legality_rate']:.2%} | "
            f"{item['timeout_rate']:.2%} | {item['synthesis_ms_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Full moment 稳定受益：{'是' if summary['full_moment_benefit'] else '否'}。",
            f"Partial moment 稳定受益条件：`{json.dumps(summary['partial_stable_benefit_conditions'], ensure_ascii=False)}`。",
            f"所有运行的整体 legality：{summary['overall_legality_rate']:.2%}；timeout：{summary['overall_timeout_rate']:.2%}。",
            "",
            "判断规则：只有 Full moment 不优于 Full baseline、且至少一个 partial 条件跨 seed 且 bootstrap CI 稳定为正时，才支持把 moments 转向 partial-observation action conditioning。",
            "",
            "## 输出与复现",
            "",
            f"- summary：`{Path(args.output_dir) / 'partial_demand_summary.json'}`",
            f"- paired detail：`{Path(args.output_dir) / 'partial_demand_detail.csv'}`",
            "",
            "```bash",
            "python scripts/evaluate_partial_demand.py \\",
            f"  --checkpoint-dir {args.checkpoint_dir} \\",
            "  --hide-ratios 0.25 0.50 \\",
            "  --observation-modes random_entries source_totals source_destination_totals partial_shards \\",
            "  --seeds 42 142 242 \\",
            f"  --output-dir {args.output_dir}",
            "```",
            "",
        ]
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.sequence_length < args.window_size or args.window_size < args.min_history:
        raise ValueError("Require sequence-length >= window-size >= min-history")
    if any(not 0.0 <= ratio < 1.0 for ratio in args.hide_ratios):
        raise ValueError("hide-ratios must be in [0,1)")
    device = torch.device(args.device)
    topology = load_topology_info(args.topology)
    sequences = _sequences(args)
    rows: list[dict[str, Any]] = []
    checkpoint_metadata = {}

    for training_seed in args.training_seeds:
        seed_dir = Path(args.checkpoint_dir) / f"seed_{training_seed}"
        baseline_path = seed_dir / "baseline" / "baseline_best.pth"
        moment_path = seed_dir / "moment" / "moment_best.pth"
        if not baseline_path.is_file() or not moment_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint pair under {seed_dir}")
        baseline, baseline_checkpoint = _load_policy(baseline_path, "baseline", device)
        moment, moment_checkpoint = _load_policy(moment_path, "moment", device)
        checkpoint_metadata[str(training_seed)] = {
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "baseline_epoch": baseline_checkpoint.get("epoch"),
            "moment": str(moment_path),
            "moment_sha256": _sha256(moment_path),
            "moment_epoch": moment_checkpoint.get("epoch"),
        }
        for sequence in sequences:
            estimator = SlidingMomentEstimator(
                num_nodes=topology.V,
                window_size=args.window_size,
                min_history=args.min_history,
            )
            for step, matrix in enumerate(sequence.matrices):
                problem = _problem(matrix, sequence, step, topology, args.time_limit)
                conditions = _observation_conditions(
                    args, matrix, problem.demands, sequence.seed, step
                )
                for mode, hide_ratio, observation in conditions:
                    observed_matrix = matrix if observation is None else observation.observed_matrix
                    context = estimator.get_context(
                        observed_matrix, sequence.mean_ref, sequence.var_ref
                    )
                    baseline_result = _run(
                        baseline,
                        problem,
                        device,
                        context=None,
                        observation=observation,
                        max_entry=args.max_entry,
                    )
                    moment_result = _run(
                        moment,
                        problem,
                        device,
                        context=context,
                        observation=observation,
                        max_entry=args.max_entry,
                    )
                    for method, result in (
                        ("baseline", baseline_result),
                        ("moment", moment_result),
                    ):
                        rows.append(
                            {
                                "training_seed": training_seed,
                                "sequence_id": sequence.sequence_id,
                                "sequence_seed": sequence.seed,
                                "sequence_step": step,
                                "family": sequence.family,
                                "observation_mode": mode,
                                "hide_ratio": hide_ratio,
                                "method": method,
                                **result,
                                "paired_completion_delta": baseline_result[
                                    "completion_steps"
                                ]
                                - result["completion_steps"],
                                "paired_synthesis_delta_ms": baseline_result[
                                    "synthesis_ms"
                                ]
                                - result["synthesis_ms"],
                                "observed_total": int(observed_matrix.sum()),
                                "ground_truth_total": int(np.asarray(matrix).sum()),
                                "history_length": int(context.history_length),
                                "observation_metadata_json": json.dumps(
                                    {} if observation is None else observation.metadata,
                                    separators=(",", ":"),
                                ),
                            }
                        )
                estimator.update(matrix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "partial_demand_detail.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, args)
    summary.update(
        {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "command": sys.argv,
            "config": vars(args),
            "raw_row_count": len(rows),
            "independent_sequence_count": len(sequences),
            "checkpoint_metadata": checkpoint_metadata,
            "execution_semantics": (
                "Policy features use observation_demands/observed_matrix. Physical state, "
                "true demand clearing, completion, timeout, and legality use full ground truth."
            ),
            "history_only_verified": True,
        }
    )
    (output_dir / "partial_demand_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(summary, args)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sequences": len(sequences),
                "legality_rate": summary["overall_legality_rate"],
                "timeout_rate": summary["overall_timeout_rate"],
                "partial_stable_benefit_conditions": summary[
                    "partial_stable_benefit_conditions"
                ],
                "output": str(output_dir / "partial_demand_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
