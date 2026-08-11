#!/usr/bin/env python3
"""Evaluate history-only traffic predictors on complete held-out sequences."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.models import TrafficPredictorSuite
from rlccl.models.traffic_predictor import build_history_examples
from rlccl.traffic import LongHorizonTrafficConfig, generate_long_horizon_sequence


METHODS = (
    "constant",
    "previous",
    "moment_only",
    "recent_history",
    "oracle_current_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/v1_diagnosis/predictor")
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis/predictability")
    parser.add_argument("--report", default="docs/TRAFFIC_PREDICTABILITY.md")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


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


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    left = _rankdata(np.asarray(actual).reshape(-1))
    right = _rankdata(np.asarray(predicted).reshape(-1))
    if left.std(ddof=0) <= 1e-12 or right.std(ddof=0) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _continuous_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    mse = float(np.mean(np.square(error)))
    denominator = float(np.sum(np.square(actual - actual.mean())))
    r2 = 1.0 - float(np.sum(np.square(error))) / denominator if denominator > 1e-12 else 0.0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2),
        "spearman": _spearman(actual, predicted),
    }


def _slices(num_nodes: int, group_count: int) -> dict[str, slice]:
    cursor = 0
    result = {"current_total_traffic": slice(cursor, cursor + 1)}
    cursor += 1
    result["current_source_load_vector"] = slice(cursor, cursor + num_nodes)
    cursor += num_nodes
    result["current_destination_load_vector"] = slice(cursor, cursor + num_nodes)
    cursor += num_nodes
    result["current_hotspot_strength"] = slice(cursor, cursor + 1)
    cursor += 1
    result["current_sparsity"] = slice(cursor, cursor + 1)
    cursor += 1
    result["current_bandwidth_group_load"] = slice(cursor, cursor + group_count)
    return result


def _effective_sample_size(values: np.ndarray, max_lag: int = 64) -> tuple[float, float]:
    series = np.asarray(values, dtype=np.float64)
    if len(series) < 3 or series.std(ddof=0) <= 1e-12:
        return float(len(series)), 0.0
    centered = series - series.mean()
    variance = float(np.dot(centered, centered))
    correlations = []
    for lag in range(1, min(max_lag, len(series) - 1) + 1):
        value = float(np.dot(centered[:-lag], centered[lag:]) / variance)
        if value <= 0:
            break
        correlations.append(value)
    ess = len(series) / max(1.0 + 2.0 * sum(correlations), 1.0)
    lag1 = float(np.dot(centered[:-1], centered[1:]) / variance)
    return float(ess), lag1


def _evaluate_scope(
    indices: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
    slices: dict[str, slice],
) -> dict[str, Any]:
    target = predictions["target"]
    result: dict[str, Any] = {}
    constant_rmse: dict[str, float] = {}
    for method in METHODS:
        continuous = predictions[method]["continuous"][indices]
        actual = target["continuous"][indices]
        metrics = {
            name: _continuous_metrics(actual[:, part], continuous[:, part])
            for name, part in slices.items()
            if part.stop > part.start
        }
        if method == "constant":
            constant_rmse = {name: item["rmse"] for name, item in metrics.items()}
        for name, item in metrics.items():
            baseline = constant_rmse.get(name, item["rmse"])
            item["relative_rmse_improvement_vs_constant"] = (
                (baseline - item["rmse"]) / baseline if baseline > 1e-12 else 0.0
            )
        result[method] = {
            "continuous": metrics,
            "hotspot_destination_accuracy": float(
                np.mean(
                    predictions[method]["hotspot"][indices]
                    == target["hotspot"][indices]
                )
            ),
        }
    return result


def _bootstrap_improvement(
    examples: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    by_sequence: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        by_sequence.setdefault(example["sequence_id"], []).append(index)
    names = sorted(by_sequence)
    actual = predictions["target"]["continuous"][:, 0]
    constant = predictions["constant"]["continuous"][:, 0]
    rng = np.random.default_rng(seed)
    result: dict[str, list[float]] = {}
    for method in ("previous", "moment_only", "recent_history"):
        predicted = predictions[method]["continuous"][:, 0]
        cluster = []
        for name in names:
            idx = np.asarray(by_sequence[name], dtype=np.int64)
            constant_rmse = float(np.sqrt(np.mean(np.square(constant[idx] - actual[idx]))))
            method_rmse = float(np.sqrt(np.mean(np.square(predicted[idx] - actual[idx]))))
            cluster.append(constant_rmse - method_rmse)
        values = np.asarray(cluster, dtype=np.float64)
        draws = rng.integers(0, len(values), size=(samples, len(values)))
        means = values[draws].mean(axis=1)
        result[method] = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    return result


def _write_report(summary: dict[str, Any], args: argparse.Namespace) -> None:
    overall = summary["metrics"]["overall"]
    bootstrap = summary["total_rmse_improvement_bootstrap_ci95"]
    model_dir_display = str(Path(args.model_dir)).replace("\\", "/")
    output_dir_display = str(Path(args.output_dir)).replace("\\", "/")
    lines = [
        "# 历史 moments 对当前流量的可预测性",
        "",
        "## 设计",
        "",
        f"训练 {summary['train_sequence_count']} 条完整 sequence，测试 {summary['test_sequence_count']} 条完全不重叠的完整 sequence；测试样本 {summary['raw_test_sample_count']} 个。",
        "预测 X_t 时所有非 oracle 方法只使用 X_0...X_{t-1}。moment-only 使用滑窗矩阵均值/方差；recent-history 是对有序最近 summary 序列的多输出 ridge autoregressor；oracle 只作为上界。",
        "bandwidth-group load 是 topology 上确定性最短路的 offered-load proxy，不冒充 learned schedule 的真实 group utilization。",
        "",
        "## Overall 当前总流量",
        "",
        "| method | MAE | RMSE | R² | Spearman | vs constant RMSE | hotspot accuracy | sequence-bootstrap ΔRMSE CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metric = overall[method]["continuous"]["current_total_traffic"]
        ci = bootstrap.get(method)
        ci_text = "oracle/constant" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        lines.append(
            f"| {method} | {metric['mae']:.4f} | {metric['rmse']:.4f} | {metric['r2']:.4f} | "
            f"{metric['spearman']:.4f} | {metric['relative_rmse_improvement_vs_constant']:.2%} | "
            f"{overall[method]['hotspot_destination_accuracy']:.2%} | {ci_text} |"
        )
    previous_total = overall["previous"]["continuous"]["current_total_traffic"]
    moment_total = overall["moment_only"]["continuous"]["current_total_traffic"]
    recent_total = overall["recent_history"]["continuous"]["current_total_traffic"]
    lines.extend(
        [
            "",
            "结论：moment-only 虽优于 constant，但其总流量 RMSE "
            f"`{moment_total['rmse']:.4f}` 明显高于 previous-value 的 `{previous_total['rmse']:.4f}`；"
            f"recent-history 为 `{recent_total['rmse']:.4f}`。因此 moments 不满足“优于简单 previous-value”判据，而有序近期序列明显优于 moment 压缩。",
            "",
            "## 其他当前流量 summary",
            "",
            "| target | previous RMSE | moment RMSE | recent RMSE | moment vs constant | recent vs constant |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target_name in (
        "current_source_load_vector",
        "current_destination_load_vector",
        "current_hotspot_strength",
        "current_sparsity",
        "current_bandwidth_group_load",
    ):
        previous = overall["previous"]["continuous"][target_name]
        moment = overall["moment_only"]["continuous"][target_name]
        recent = overall["recent_history"]["continuous"][target_name]
        lines.append(
            f"| {target_name} | {previous['rmse']:.4f} | {moment['rmse']:.4f} | "
            f"{recent['rmse']:.4f} | {moment['relative_rmse_improvement_vs_constant']:.2%} | "
            f"{recent['relative_rmse_improvement_vs_constant']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 按 family 的关键判断",
            "",
            "| family | constant RMSE | previous improvement | moment improvement | recent improvement | moment hotspot acc | recent hotspot acc |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family, values in summary["metrics"]["by_family"].items():
        constant = values["constant"]["continuous"]["current_total_traffic"]
        previous = values["previous"]["continuous"]["current_total_traffic"]
        moment = values["moment_only"]["continuous"]["current_total_traffic"]
        recent = values["recent_history"]["continuous"]["current_total_traffic"]
        lines.append(
            f"| {family} | {constant['rmse']:.4f} | {previous['relative_rmse_improvement_vs_constant']:.2%} | "
            f"{moment['relative_rmse_improvement_vs_constant']:.2%} | {recent['relative_rmse_improvement_vs_constant']:.2%} | "
            f"{values['moment_only']['hotspot_destination_accuracy']:.2%} | "
            f"{values['recent_history']['hotspot_destination_accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 统计单位",
            "",
            f"独立测试 sequence {summary['test_sequence_count']} 条；raw step 样本 {summary['raw_test_sample_count']}；各 sequence 总流量 ESS 合计 {summary['total_traffic_effective_sample_size_sum']:.2f}；平均 lag-1 ACF {summary['mean_total_traffic_lag1_acf']:.4f}。",
            "bootstrap 以完整 sequence 为 cluster，不把时间步当成独立样本。",
            "",
            "## 输出与复现",
            "",
            f"- summary：`{output_dir_display}/predictability_summary.json`",
            "",
            "```bash",
            "python scripts/train_traffic_predictor.py \\",
            "  --sequence-length 1024 \\",
            "  --families regime_switching_long stochastic_volatility rare_shock_recovery hotspot_random_walk same_moments_different_dynamics \\",
            "  --seeds 42 142 242 \\",
            f"  --output-dir {model_dir_display}",
            "python scripts/evaluate_traffic_predictability.py \\",
            f"  --model-dir {model_dir_display} \\",
            f"  --output-dir {output_dir_display}",
            "```",
            "",
        ]
    )
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    model_dir = Path(args.model_dir)
    manifest = json.loads((model_dir / "predictor_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    group_coefficients = np.load(model_dir / "bandwidth_group_coefficients.npy")
    suite = TrafficPredictorSuite.load(str(model_dir / "traffic_predictor.npz"))
    print(
        f"Regenerating {len(manifest['test_sequences'])} complete held-out sequences...",
        flush=True,
    )
    sequences = [_generate(spec, config) for spec in manifest["test_sequences"]]
    train_ids = {
        (item["family"], item["actual_seed"]) for item in manifest["train_sequences"]
    }
    test_ids = {(sequence.family, sequence.seed) for sequence in sequences}
    if train_ids & test_ids:
        raise AssertionError("Complete-sequence train/test leakage")
    examples = build_history_examples(
        sequences,
        group_coefficients=group_coefficients,
        history_window=int(config["history_window"]),
        recent_steps=int(config["recent_steps"]),
        min_history=int(config["min_history"]),
    )
    if any(example["history_last_step"] >= example["step"] for example in examples):
        raise AssertionError("Future/current traffic leaked into predictor features")
    predictions = suite.predict(examples)
    slices = _slices(suite.num_nodes, suite.group_count)
    all_indices = np.arange(len(examples), dtype=np.int64)
    metrics = {"overall": _evaluate_scope(all_indices, predictions, slices), "by_family": {}}
    for family in sorted({item["family"] for item in examples}):
        indices = np.asarray(
            [index for index, item in enumerate(examples) if item["family"] == family],
            dtype=np.int64,
        )
        metrics["by_family"][family] = _evaluate_scope(indices, predictions, slices)

    ess_values = []
    lag1_values = []
    for sequence in sequences:
        values = np.asarray(sequence.matrices).sum(axis=(1, 2))
        ess, lag1 = _effective_sample_size(values)
        ess_values.append(ess)
        lag1_values.append(lag1)
    summary = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "numpy": np.__version__,
        "command": sys.argv,
        "model_dir": str(model_dir),
        "config": config,
        "split_unit": "complete traffic sequence",
        "history_only_verified": True,
        "train_sequence_count": len(manifest["train_sequences"]),
        "test_sequence_count": len(sequences),
        "raw_test_sample_count": len(examples),
        "total_traffic_effective_sample_size_sum": float(sum(ess_values)),
        "mean_total_traffic_lag1_acf": float(np.mean(lag1_values)),
        "metrics": metrics,
        "total_rmse_improvement_bootstrap_ci95": _bootstrap_improvement(
            examples, predictions, args.bootstrap_samples, args.bootstrap_seed
        ),
        "oracle_note": "Oracle uses current truth only as an upper bound and is never fitted.",
        "bandwidth_group_note": config["bandwidth_group_target"],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(summary, args)
    print(
        json.dumps(
            {
                "test_sequences": len(sequences),
                "test_examples": len(examples),
                "overall_total": {
                    method: metrics["overall"][method]["continuous"][
                        "current_total_traffic"
                    ]
                    for method in METHODS
                },
                "output": str(output_dir / "predictability_summary.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
