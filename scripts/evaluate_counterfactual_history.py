#!/usr/bin/env python3
"""Evaluate identical current traffic under deliberately different prior histories."""

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

from rlccl.envs.decoder import SlotDecoder, get_candidate_moment_features
from rlccl.envs.evaluator import evaluate_schedule, load_topology_info
from rlccl.envs.problem import ProblemInstance, compute_received_chunks
from rlccl.evaluation.counterfactual import (
    CounterfactualHistoryPair,
    action_edit_distance,
    context_distance,
    context_from_prior_history,
    edge_use_l1,
    json_ready_context,
    sparse_schedule,
)
from rlccl.models import SlotLevelPolicy
from rlccl.traffic import (
    LONG_HORIZON_FAMILIES,
    SAME_MOMENT_VARIANTS,
    LongHorizonTrafficConfig,
    generate_long_horizon_sequence,
    traffic_matrix_to_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument(
        "--training-seeds", nargs="+", type=int, default=[42, 142, 242]
    )
    parser.add_argument("--topology", default="Rear4GPU")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=20)
    parser.add_argument("--max-entry", type=int, default=8)
    parser.add_argument("--min-context-distance", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis")
    parser.add_argument("--report", default="docs/COUNTERFACTUAL_HISTORY.md")
    return parser.parse_args()


def _load_policy(path: Path, mode: str, device: torch.device) -> tuple[Any, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    checkpoint_mode = checkpoint.get("policy_mode", mode)
    if checkpoint_mode != mode:
        raise ValueError(f"{path} has mode={checkpoint_mode}, expected {mode}")
    config = checkpoint.get("config", {})
    hidden_dim = int(config.get("hidden_dim", 64))
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


def _history_bank(seed: int, length: int, max_entry: int) -> tuple[Any, Any]:
    low = generate_long_horizon_sequence(
        LongHorizonTrafficConfig(
            sequence_length=length,
            family="hotspot_random_walk",
            seed=seed + 500_000,
            mean_level=1.0,
            std_level=0.75,
            max_entry=max_entry,
            calibration_candidates=1,
        )
    )
    high = generate_long_horizon_sequence(
        LongHorizonTrafficConfig(
            sequence_length=length,
            family="rare_shock_recovery",
            seed=seed + 700_000,
            mean_level=3.0,
            std_level=1.5,
            max_entry=max_entry,
            shock_probability=0.02,
            calibration_candidates=1,
        )
    )
    return low, high


def build_pairs(args: argparse.Namespace) -> list[CounterfactualHistoryPair]:
    if args.num_pairs <= 0 or not args.seeds:
        raise ValueError("num-pairs and seeds must be positive/non-empty")
    family_counts = {
        (seed, family): 0 for seed in args.seeds for family in LONG_HORIZON_FAMILIES
    }
    assignments: list[tuple[int, str, int]] = []
    for index in range(args.num_pairs):
        seed = args.seeds[index % len(args.seeds)]
        family = LONG_HORIZON_FAMILIES[(index // len(args.seeds)) % len(LONG_HORIZON_FAMILIES)]
        sequence_index = family_counts[(seed, family)]
        family_counts[(seed, family)] += 1
        assignments.append((seed, family, sequence_index))

    current_sequences: dict[tuple[int, str], Any] = {}
    for (seed, family), count in family_counts.items():
        if count == 0:
            continue
        variant = None
        if family == "same_moments_different_dynamics":
            variant = SAME_MOMENT_VARIANTS[seed % len(SAME_MOMENT_VARIANTS)]
        current_sequences[(seed, family)] = generate_long_horizon_sequence(
            LongHorizonTrafficConfig(
                sequence_length=max(64, count),
                family=family,
                seed=seed + 900_000 + LONG_HORIZON_FAMILIES.index(family) * 10_000,
                mean_level=2.0,
                std_level=1.5,
                max_entry=args.max_entry,
                dynamics_variant=variant,
                calibration_candidates=1,
            )
        )

    per_seed_count = math.ceil(args.num_pairs / len(args.seeds))
    bank_length = max(128, args.window_size + per_seed_count * 3)
    banks = {seed: _history_bank(seed, bank_length, args.max_entry) for seed in args.seeds}
    pairs: list[CounterfactualHistoryPair] = []
    for pair_index, (seed, family, sequence_index) in enumerate(assignments):
        current_sequence = current_sequences[(seed, family)]
        current = np.asarray(current_sequence.matrices[sequence_index], dtype=np.int64).copy()
        low, high = banks[seed]
        available = bank_length - args.window_size + 1
        start_a = (pair_index * 7) % available
        start_b = (pair_index * 11 + args.window_size) % available
        history_a = tuple(
            np.asarray(item, dtype=np.int64).copy()
            for item in low.matrices[start_a : start_a + args.window_size]
        )
        history_b = tuple(
            np.asarray(item, dtype=np.int64).copy()
            for item in high.matrices[start_b : start_b + args.window_size]
        )
        pairs.append(
            CounterfactualHistoryPair(
                pair_id=f"pair-{pair_index:04d}-seed{seed}-{family}",
                family=family,
                seed=seed,
                current_matrix=current,
                history_a=history_a,
                history_b=history_b,
                mean_ref=current_sequence.mean_ref,
                var_ref=current_sequence.var_ref,
            )
        )
    return pairs


def _problem_from_matrix(pair: CounterfactualHistoryPair, topology: Any, time_limit: int) -> Any:
    scenario = traffic_matrix_to_scenario(
        pair.current_matrix, sequence_id=pair.pair_id, sequence_step=0, family=pair.family
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
        traffic_matrix=pair.current_matrix,
        scenario_type="all_to_all_v",
        sequence_id=pair.pair_id,
        sequence_step=0,
        metadata={"family": pair.family, "counterfactual_seed": pair.seed},
    )


def _first_candidate_trace(
    model: Any,
    decoder: SlotDecoder,
    state_info: dict[str, Any],
    micro_actions: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    if not micro_actions:
        return {"candidates": [], "logits": [], "selected": None}
    action = micro_actions[0]
    cand_c = np.asarray(action["cand_c"], dtype=np.int64)
    cand_e = np.asarray(action["cand_e"], dtype=np.int64)
    node_feats = state_info["node_feats"].to(device)
    edge_feats = state_info["edge_feats"].to(device)
    chunk_feats = state_info["chunk_feats"].to(device)
    global_features = state_info.get("global_moment_feats")
    if global_features is not None:
        global_features = global_features.to(device)
    with torch.no_grad():
        h_v, h_e, h_c, g_ctx = model.encode_state(
            node_feats,
            edge_feats,
            decoder.edge_src_t.to(device),
            decoder.edge_dst_t.to(device),
            chunk_feats,
            global_moment_feats=global_features,
        )
        dynamic = decoder.get_candidate_dynamic_features(
            cand_c,
            cand_e,
            state_info["demands"],
            state_info["dist_to_demand"],
            np.zeros(decoder.E, dtype=np.float32),
            np.zeros(decoder.num_groups, dtype=np.float32),
            0,
            decoder.max_steps,
        )
        if state_info.get("moment_enabled"):
            dynamic = np.concatenate(
                [
                    dynamic,
                    get_candidate_moment_features(
                        cand_e,
                        decoder.edge_src,
                        decoder.edge_dst,
                        state_info["candidate_moment_node_arrays"],
                    ),
                ],
                axis=1,
            )
        logits = model.get_candidate_logits(
            h_v,
            h_e,
            h_c,
            g_ctx,
            decoder.edge_src_t.to(device),
            decoder.edge_dst_t.to(device),
            torch.tensor(cand_e, dtype=torch.long, device=device),
            torch.tensor(cand_c, dtype=torch.long, device=device),
            torch.tensor(dynamic, dtype=torch.float32, device=device),
        )
    selected_index = int(action["action_idx"])
    return {
        "candidates": [[int(c), int(e)] for c, e in zip(cand_c, cand_e)],
        "logits": logits.detach().cpu().numpy().astype(float).tolist(),
        "selected": [int(cand_c[selected_index]), int(cand_e[selected_index])],
    }


def _run_policy(
    model: Any,
    problem: Any,
    device: torch.device,
    *,
    context: Any,
    current_matrix: np.ndarray,
    max_entry: int,
) -> dict[str, Any]:
    decoder = SlotDecoder(problem.topology_info)
    state = problem.initial_state.copy()
    demands = problem.demands.copy()
    schedule: list[np.ndarray] = []
    actions: list[tuple[int, int, int]] = []
    first_trace: dict[str, Any] | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    completion = problem.T
    with torch.no_grad():
        for slot in range(problem.T):
            slot_matrix, _, _, _, state_info, micro_actions = decoder.decode_slot(
                model,
                state,
                demands,
                slot,
                problem.T,
                train=False,
                moment_context=context,
                current_matrix=current_matrix,
                moment_max_entry=max_entry,
            )
            if first_trace is None:
                first_trace = _first_candidate_trace(
                    model, decoder, state_info, micro_actions, device
                )
            schedule.append(slot_matrix)
            for action in micro_actions:
                selected = int(action["action_idx"])
                actions.append(
                    (
                        int(slot),
                        int(action["cand_c"][selected]),
                        int(action["cand_e"][selected]),
                    )
                )
            received = compute_received_chunks(
                slot_matrix, problem.topology_info.edge_dst, problem.topology_info.V
            )
            state = np.maximum(state, received)
            demands = demands * (1 - received)
            if not np.any(demands):
                completion = slot + 1
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    synthesis_ms = (time.perf_counter() - started) * 1000.0
    timeout = bool(np.any(demands))
    while len(schedule) < problem.T:
        schedule.append(np.zeros((problem.C, problem.E), dtype=np.int64))
    score, error = evaluate_schedule(schedule, problem)
    digest = hashlib.sha256()
    for matrix in schedule:
        digest.update(np.ascontiguousarray(matrix, dtype=np.int8).tobytes())
    return {
        "completion": int(completion),
        "timeout": timeout,
        "legal": error == "",
        "evaluation_error": error,
        "score": float(score),
        "synthesis_ms": float(synthesis_ms),
        "schedule": schedule,
        "schedule_sparse": sparse_schedule(schedule),
        "schedule_sha256": digest.hexdigest(),
        "actions": actions,
        "first_trace": first_trace or {"candidates": [], "logits": [], "selected": None},
    }


def _bootstrap_ci(values: dict[str, list[float]], samples: int, seed: int) -> list[float]:
    cluster = np.asarray([np.mean(values[key]) for key in sorted(values)], dtype=np.float64)
    if cluster.size == 1:
        return [float(cluster[0]), float(cluster[0])]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, cluster.size, size=(samples, cluster.size))
    means = cluster[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _aggregate(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_pair: dict[str, list[float]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row["paired_delta_average_moment"])
    methods: dict[str, dict[str, Any]] = {}
    for name, key in (
        ("baseline", "baseline_completion"),
        ("moment_history_a", "moment_a_completion"),
        ("moment_history_b", "moment_b_completion"),
    ):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        synth = np.asarray([row[key.replace("completion", "synthesis_ms")] for row in rows])
        methods[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "cvar95": float(values[values >= np.quantile(values, 0.95)].mean()),
            "synthesis_ms_mean": float(synth.mean()),
        }
    return {
        "num_rows": len(rows),
        "num_pairs": len({row["pair_id"] for row in rows}),
        "num_training_seeds": len({row["training_seed"] for row in rows}),
        "methods": methods,
        "baseline_history_equivalence_rate": float(
            np.mean([row["baseline_histories_equivalent"] for row in rows])
        ),
        "moment_schedule_change_rate": float(
            np.mean([row["moment_schedule_changed"] for row in rows])
        ),
        "harmful_context_interference_rate": float(
            np.mean([row["action_level_context_interference"] for row in rows])
        ),
        "beneficial_context_change_rate": float(
            np.mean([row["beneficial_context_change"] for row in rows])
        ),
        "mean_action_edit_distance": float(
            np.mean([row["moment_action_edit_distance"] for row in rows])
        ),
        "mean_edge_use_l1": float(np.mean([row["moment_edge_use_l1"] for row in rows])),
        "mean_first_logit_l2": float(
            np.mean([row["moment_first_logit_l2"] for row in rows])
        ),
        "paired_delta_average_moment_mean": float(
            np.mean([row["paired_delta_average_moment"] for row in rows])
        ),
        "paired_delta_average_moment_bootstrap_ci95": _bootstrap_ci(
            by_pair, args.bootstrap_samples, 20260727
        ),
        "legality_rate": float(
            np.mean(
                [
                    row["baseline_a_legal"]
                    and row["baseline_b_legal"]
                    and row["moment_a_legal"]
                    and row["moment_b_legal"]
                    for row in rows
                ]
            )
        ),
        "timeout_rate": float(
            np.mean(
                [
                    row["baseline_a_timeout"]
                    or row["baseline_b_timeout"]
                    or row["moment_a_timeout"]
                    or row["moment_b_timeout"]
                    for row in rows
                ]
            )
        ),
    }


def _write_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    methods = summary["methods"]
    ci = summary["paired_delta_average_moment_bootstrap_ci95"]
    lines = [
        "# 相同当前 X、不同历史的反事实诊断",
        "",
        "## 设计",
        "",
        f"构造 {summary['num_pairs']} 个相同当前 traffic matrix、相同 topology、相同初始 schedule state 的 pair，并在 {summary['num_training_seeds']} 个独立 V1 训练 seed 上评测。",
        "History A/B 都只由当前 X 之前的独立矩阵组成；`SlidingMomentEstimator.get_context` 后才会在真实时序中更新当前矩阵，本实验没有把未来 X 放入 estimator。",
        "baseline 对两个历史分别实际运行；Moment-full 使用显著不同的两个 history-only context。",
        "",
        "## 结果",
        "",
        "| method | mean completion | median | p95 | p99 | CVaR95 | mean synthesis ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "moment_history_a", "moment_history_b"):
        item = methods[name]
        lines.append(
            f"| {name} | {item['mean']:.4f} | {item['median']:.4f} | {item['p95']:.4f} | "
            f"{item['p99']:.4f} | {item['cvar95']:.4f} | {item['synthesis_ms_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- baseline 两历史等价率：{summary['baseline_history_equivalence_rate']:.2%}",
            f"- Moment-full 因历史改变完整 schedule 的比例：{summary['moment_schedule_change_rate']:.2%}",
            f"- action-level context interference（变化且至少一个历史结果劣于 baseline）：{summary['harmful_context_interference_rate']:.2%}",
            f"- 变化且至少一个历史优于 baseline：{summary['beneficial_context_change_rate']:.2%}",
            f"- baseline completion - 两个 Moment completion 平均值：{summary['paired_delta_average_moment_mean']:.4f}，按 pair cluster bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]；正数才表示 Moment 更好。",
            f"- 首 slot logits L2 差异均值：{summary['mean_first_logit_l2']:.4f}；动作 edit distance 均值：{summary['mean_action_edit_distance']:.3f}；edge-use L1 均值：{summary['mean_edge_use_l1']:.3f}。",
            f"- schedule legality：{summary['legality_rate']:.2%}；timeout：{summary['timeout_rate']:.2%}。",
            "",
            "若 `action-level context interference` 非零，则它是直接证据：完整当前 demand 与初始状态相同，仅历史 moments 就能改变 action-level schedule，并且至少一个变化方向恶化 completion。",
            "",
            "## 输出与复现",
            "",
            f"- paired 明细：`{Path(args.output_dir) / 'counterfactual_detail.csv'}`",
            f"- 汇总与运行元数据：`{Path(args.output_dir) / 'counterfactual_summary.json'}`",
            "",
            "```bash",
            "python scripts/evaluate_counterfactual_history.py \\",
            f"  --checkpoint-dir {args.checkpoint_dir} \\",
            f"  --num-pairs {args.num_pairs} \\",
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
    if args.window_size < args.min_history or args.min_history <= 0:
        raise ValueError("Require window-size >= min-history > 0")
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    device = torch.device(args.device)
    topology = load_topology_info(args.topology)
    pairs = build_pairs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for training_seed in args.training_seeds:
        seed_dir = Path(args.checkpoint_dir) / f"seed_{training_seed}"
        baseline_path = seed_dir / "baseline" / "baseline_best.pth"
        moment_path = seed_dir / "moment" / "moment_best.pth"
        if not baseline_path.is_file() or not moment_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint pair under {seed_dir}")
        baseline, baseline_checkpoint = _load_policy(baseline_path, "baseline", device)
        moment, moment_checkpoint = _load_policy(moment_path, "moment", device)
        checkpoints[str(training_seed)] = {
            "baseline": str(baseline_path),
            "baseline_sha256": _sha256(baseline_path),
            "baseline_epoch": baseline_checkpoint.get("epoch"),
            "moment": str(moment_path),
            "moment_sha256": _sha256(moment_path),
            "moment_epoch": moment_checkpoint.get("epoch"),
        }

        for pair in pairs:
            context_a = context_from_prior_history(
                pair.history_a,
                pair.current_matrix,
                pair.mean_ref,
                pair.var_ref,
                window_size=args.window_size,
                min_history=args.min_history,
            )
            context_b = context_from_prior_history(
                pair.history_b,
                pair.current_matrix,
                pair.mean_ref,
                pair.var_ref,
                window_size=args.window_size,
                min_history=args.min_history,
            )
            distance = context_distance(context_a, context_b)
            if distance["combined"] < args.min_context_distance:
                raise RuntimeError(
                    f"Counterfactual histories too similar for {pair.pair_id}: {distance}"
                )
            problem = _problem_from_matrix(pair, topology, args.time_limit)
            baseline_a = _run_policy(
                baseline,
                problem,
                device,
                context=None,
                current_matrix=pair.current_matrix,
                max_entry=args.max_entry,
            )
            baseline_b = _run_policy(
                baseline,
                problem,
                device,
                context=None,
                current_matrix=pair.current_matrix,
                max_entry=args.max_entry,
            )
            moment_a = _run_policy(
                moment,
                problem,
                device,
                context=context_a,
                current_matrix=pair.current_matrix,
                max_entry=args.max_entry,
            )
            moment_b = _run_policy(
                moment,
                problem,
                device,
                context=context_b,
                current_matrix=pair.current_matrix,
                max_entry=args.max_entry,
            )
            baseline_equivalent = baseline_a["schedule_sha256"] == baseline_b["schedule_sha256"]
            moment_changed = moment_a["schedule_sha256"] != moment_b["schedule_sha256"]
            baseline_completion = baseline_a["completion"]
            harmful = moment_changed and max(
                moment_a["completion"], moment_b["completion"]
            ) > baseline_completion
            beneficial = moment_changed and min(
                moment_a["completion"], moment_b["completion"]
            ) < baseline_completion
            logits_a = np.asarray(moment_a["first_trace"]["logits"], dtype=np.float64)
            logits_b = np.asarray(moment_b["first_trace"]["logits"], dtype=np.float64)
            if logits_a.shape != logits_b.shape:
                raise AssertionError("First-slot candidate sets changed before the first action")
            row = {
                "pair_id": pair.pair_id,
                "family": pair.family,
                "counterfactual_seed": pair.seed,
                "topology": args.topology,
                "training_seed": training_seed,
                "current_matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(pair.current_matrix).tobytes()
                ).hexdigest(),
                "initial_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(problem.initial_state).tobytes()
                ).hexdigest(),
                "ground_truth_demands_sha256": hashlib.sha256(
                    np.ascontiguousarray(problem.demands).tobytes()
                ).hexdigest(),
                "current_total_traffic": int(pair.current_matrix.sum()),
                **{f"context_{key}": value for key, value in distance.items()},
                "context_a_json": json.dumps(json_ready_context(context_a), separators=(",", ":")),
                "context_b_json": json.dumps(json_ready_context(context_b), separators=(",", ":")),
                "baseline_a_completion": baseline_a["completion"],
                "baseline_b_completion": baseline_b["completion"],
                "baseline_completion": baseline_completion,
                "moment_a_completion": moment_a["completion"],
                "moment_b_completion": moment_b["completion"],
                "baseline_a_synthesis_ms": baseline_a["synthesis_ms"],
                "baseline_b_synthesis_ms": baseline_b["synthesis_ms"],
                "baseline_synthesis_ms": 0.5 * (
                    baseline_a["synthesis_ms"] + baseline_b["synthesis_ms"]
                ),
                "moment_a_synthesis_ms": moment_a["synthesis_ms"],
                "moment_b_synthesis_ms": moment_b["synthesis_ms"],
                "baseline_a_timeout": baseline_a["timeout"],
                "baseline_b_timeout": baseline_b["timeout"],
                "moment_a_timeout": moment_a["timeout"],
                "moment_b_timeout": moment_b["timeout"],
                "baseline_a_legal": baseline_a["legal"],
                "baseline_b_legal": baseline_b["legal"],
                "moment_a_legal": moment_a["legal"],
                "moment_b_legal": moment_b["legal"],
                "baseline_histories_equivalent": baseline_equivalent,
                "moment_schedule_changed": moment_changed,
                "action_level_context_interference": harmful,
                "beneficial_context_change": beneficial,
                "paired_delta_moment_a": baseline_completion - moment_a["completion"],
                "paired_delta_moment_b": baseline_completion - moment_b["completion"],
                "paired_delta_average_moment": baseline_completion
                - 0.5 * (moment_a["completion"] + moment_b["completion"]),
                "moment_action_edit_distance": action_edit_distance(
                    moment_a["actions"], moment_b["actions"]
                ),
                "moment_edge_use_l1": edge_use_l1(
                    moment_a["schedule"], moment_b["schedule"]
                ),
                "moment_first_logit_l2": float(np.linalg.norm(logits_a - logits_b)),
                "baseline_a_first_action_json": json.dumps(
                    baseline_a["first_trace"]["selected"]
                ),
                "baseline_b_first_action_json": json.dumps(
                    baseline_b["first_trace"]["selected"]
                ),
                "moment_a_first_action_json": json.dumps(moment_a["first_trace"]["selected"]),
                "moment_b_first_action_json": json.dumps(moment_b["first_trace"]["selected"]),
                "baseline_first_candidates_json": json.dumps(
                    baseline_a["first_trace"]["candidates"], separators=(",", ":")
                ),
                "baseline_a_first_logits_json": json.dumps(
                    baseline_a["first_trace"]["logits"], separators=(",", ":")
                ),
                "baseline_b_first_logits_json": json.dumps(
                    baseline_b["first_trace"]["logits"], separators=(",", ":")
                ),
                "moment_first_candidates_json": json.dumps(
                    moment_a["first_trace"]["candidates"], separators=(",", ":")
                ),
                "moment_a_first_logits_json": json.dumps(
                    moment_a["first_trace"]["logits"], separators=(",", ":")
                ),
                "moment_b_first_logits_json": json.dumps(
                    moment_b["first_trace"]["logits"], separators=(",", ":")
                ),
                "baseline_a_schedule_sha256": baseline_a["schedule_sha256"],
                "baseline_b_schedule_sha256": baseline_b["schedule_sha256"],
                "moment_a_schedule_sha256": moment_a["schedule_sha256"],
                "moment_b_schedule_sha256": moment_b["schedule_sha256"],
                "baseline_a_schedule_json": json.dumps(
                    baseline_a["schedule_sparse"], separators=(",", ":")
                ),
                "baseline_b_schedule_json": json.dumps(
                    baseline_b["schedule_sparse"], separators=(",", ":")
                ),
                "moment_a_schedule_json": json.dumps(
                    moment_a["schedule_sparse"], separators=(",", ":")
                ),
                "moment_b_schedule_json": json.dumps(
                    moment_b["schedule_sparse"], separators=(",", ":")
                ),
            }
            rows.append(row)

    detail_path = output_dir / "counterfactual_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _aggregate(rows, args)
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
            "checkpoints": checkpoints,
            "history_semantics": (
                "Both histories are independent prior matrices. Context is read before "
                "the identical current X and the current X is never inserted into history."
            ),
        }
    )
    (output_dir / "counterfactual_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(summary, args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
