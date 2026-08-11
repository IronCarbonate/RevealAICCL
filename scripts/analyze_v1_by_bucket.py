#!/usr/bin/env python3
"""Paired, sequence-clustered bucket analysis for the formal V1 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from pathlib import Path
import re
import socket
import sys
from typing import Any, Iterable

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rlccl.traffic import TrafficProcessConfig, generate_traffic_sequence
from rlccl.traffic.moment_validation import relative_l2_error


METHOD_ORDER = ("baseline", "mean_only", "full", "shuffled")
NUMERIC_BUCKETS = (
    "current_total_traffic",
    "current_vs_history_mean_deviation",
    "current_vs_history_variance_deviation",
    "traffic_sparsity",
    "source_hotspot_strength",
    "destination_hotspot_strength",
    "regime_duration",
    "sequence_total_acf1",
    "baseline_completion_time",
    "baseline_synthesis_time",
)
CATEGORICAL_BUCKETS = (
    "family",
    "training_distribution",
    "hotspot_migration",
    "burst_flag",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail", default="outputs/moment_v1/formal/v1_formal_detail.csv"
    )
    parser.add_argument(
        "--formal-summary", default="outputs/moment_v1/formal/v1_formal_summary.json"
    )
    parser.add_argument("--output-dir", default="outputs/v1_diagnosis")
    parser.add_argument("--report", default="docs/V1_BUCKET_ANALYSIS.md")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def _float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value not in (None, "") else math.nan


def _bool(row: dict[str, str], name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"1", "true", "yes"}


def _sequence_seed(sequence_id: str) -> int:
    match = re.search(r"-seed(-?\d+)$", sequence_id)
    if match is None:
        raise ValueError(f"Cannot recover seed from sequence_id={sequence_id!r}")
    return int(match.group(1))


def _acf1(values: np.ndarray) -> float:
    if values.size < 3 or float(values.std(ddof=0)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _sequence_features(sequence: Any, min_history: int) -> dict[int, dict[str, Any]]:
    matrices = np.asarray(sequence.matrices, dtype=np.float64)
    totals = matrices.sum(axis=(1, 2))
    acf1 = _acf1(totals)
    off_diagonal = ~np.eye(matrices.shape[1], dtype=bool)
    reference_total = float(sequence.mean_ref.sum())
    reference_total_std = float(np.sqrt(np.maximum(sequence.var_ref, 0.0).sum()))
    history: deque[np.ndarray] = deque(maxlen=int(sequence.metadata["window_size"]))
    history_totals: deque[float] = deque(maxlen=int(sequence.metadata["window_size"]))
    previous_destination: int | None = None
    previous_regime: str | None = None
    regime_duration = 0
    result: dict[int, dict[str, Any]] = {}

    for step, current in enumerate(matrices):
        if history:
            stacked = np.stack(tuple(history), axis=0)
            history_mean = stacked.mean(axis=0)
            history_variance = stacked.var(axis=0, ddof=0)
        else:
            history_mean = np.asarray(sequence.mean_ref, dtype=np.float64)
            history_variance = np.asarray(sequence.var_ref, dtype=np.float64)

        mean_deviation = relative_l2_error(current, history_mean)
        squared_residual = np.square(current - history_mean)
        variance_deviation = relative_l2_error(squared_residual, history_variance)
        source_load = current.sum(axis=1)
        destination_load = current.sum(axis=0)
        current_total = float(current.sum())
        hotspot_destination = int(np.argmax(destination_load))
        migration = previous_destination is not None and hotspot_destination != previous_destination

        if current_total < reference_total - 0.5 * reference_total_std:
            regime = "low"
        elif current_total > reference_total + 0.5 * reference_total_std:
            regime = "high"
        else:
            regime = "normal"
        regime_duration = regime_duration + 1 if regime == previous_regime else 1

        if len(history_totals) >= min_history:
            hist_total = np.asarray(history_totals, dtype=np.float64)
            burst = current_total > float(hist_total.mean() + 2.0 * hist_total.std(ddof=0))
        else:
            burst = False

        result[step] = {
            "current_total_traffic": current_total,
            "current_vs_history_mean_deviation": float(mean_deviation),
            "current_vs_history_variance_deviation": float(variance_deviation),
            "traffic_sparsity": float(np.mean(current[off_diagonal] == 0)),
            "source_hotspot_strength": float(
                source_load.max() / max(float(source_load.mean()), 1e-12)
            ),
            "destination_hotspot_strength": float(
                destination_load.max() / max(float(destination_load.mean()), 1e-12)
            ),
            "hotspot_migration": bool(migration),
            "burst_flag": bool(burst),
            "regime_duration": int(regime_duration),
            "sequence_total_acf1": acf1,
        }
        history.append(current.copy())
        history_totals.append(current_total)
        previous_destination = hotspot_destination
        previous_regime = regime
    return result


def reconstruct_feature_map(
    detail_rows: list[dict[str, str]], config: dict[str, Any]
) -> dict[tuple[str, int], dict[str, Any]]:
    unique = sorted({(row["family"], row["sequence_id"]) for row in detail_rows})
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for family, sequence_id in unique:
        sequence = generate_traffic_sequence(
            TrafficProcessConfig(
                num_nodes=4,
                sequence_length=int(config["sequence_length"]),
                window_size=int(config["window_size"]),
                mean_level=float(config["mean_level"]),
                std_level=float(config["std_level"]),
                max_entry=int(config["max_entry"]),
                epsilon_mean=float(config["epsilon_mean"]),
                epsilon_var=float(config["epsilon_var"]),
                family=family,
                seed=_sequence_seed(sequence_id),
                topology_name=str(config.get("topology", "Rear4GPU")),
            )
        )
        if sequence.sequence_id != sequence_id:
            raise AssertionError(f"Reconstructed {sequence.sequence_id}, expected {sequence_id}")
        for step, features in _sequence_features(
            sequence, min_history=int(config["min_history"])
        ).items():
            result[(sequence_id, step)] = features
    return result


def _quantile_labels(values: Iterable[float]) -> tuple[np.ndarray, list[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.asarray(["unavailable"] * array.size, dtype=object), []
    edges = np.unique(np.quantile(finite, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if edges.size == 1:
        return np.asarray(["all"] * array.size, dtype=object), edges.tolist()
    labels = np.empty(array.size, dtype=object)
    for index, value in enumerate(array):
        if not np.isfinite(value):
            labels[index] = "unavailable"
            continue
        bucket = min(int(np.searchsorted(edges[1:-1], value, side="right")), edges.size - 2)
        labels[index] = f"q{bucket + 1}:[{edges[bucket]:.6g},{edges[bucket + 1]:.6g}]"
    return labels, edges.tolist()


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else math.nan


def _cvar95(values: np.ndarray) -> float:
    if not values.size:
        return math.nan
    threshold = np.quantile(values, 0.95)
    tail = values[values >= threshold]
    return float(tail.mean())


def _cluster_bootstrap_ci(
    deltas_by_sequence: dict[str, list[float]], samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    names = sorted(deltas_by_sequence)
    if not names:
        return math.nan, math.nan
    cluster_means = np.asarray(
        [np.mean(deltas_by_sequence[name]) for name in names], dtype=np.float64
    )
    if cluster_means.size == 1:
        value = float(cluster_means[0])
        return value, value
    draws = rng.integers(0, cluster_means.size, size=(samples, cluster_means.size))
    means = cluster_means[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_group(
    rows: list[dict[str, Any]], samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    completion = np.asarray([row["completion_steps"] for row in rows], dtype=np.float64)
    synthesis = np.asarray([row["synthesis_ms"] for row in rows], dtype=np.float64)
    deltas = np.asarray([row["paired_completion_delta"] for row in rows], dtype=np.float64)
    synthesis_deltas = np.asarray(
        [row["paired_synthesis_delta_ms"] for row in rows], dtype=np.float64
    )
    by_sequence: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_sequence[row["sequence_id"]].append(row["paired_completion_delta"])
        by_seed[row["training_seed"]].append(row["paired_completion_delta"])
    ci_low, ci_high = _cluster_bootstrap_ci(by_sequence, samples, rng)
    seed_means = {str(seed): float(np.mean(values)) for seed, values in sorted(by_seed.items())}
    positive_seeds = int(sum(value > 0 for value in seed_means.values()))
    return {
        "raw_sample_count": len(rows),
        "independent_sequence_count": len(by_sequence),
        "training_seed_count": len(by_seed),
        "completion_mean": float(completion.mean()),
        "completion_median": float(np.median(completion)),
        "completion_p95": _percentile(completion, 0.95),
        "completion_p99": _percentile(completion, 0.99),
        "completion_cvar95": _cvar95(completion),
        "synthesis_ms_mean": float(synthesis.mean()),
        "synthesis_ms_p95": _percentile(synthesis, 0.95),
        "legality_rate": float(np.mean([row["legal"] for row in rows])),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])),
        "paired_completion_delta_mean": float(deltas.mean()),
        "paired_completion_delta_median": float(np.median(deltas)),
        "paired_completion_delta_ci95_low": ci_low,
        "paired_completion_delta_ci95_high": ci_high,
        "paired_synthesis_delta_ms_mean": float(synthesis_deltas.mean()),
        "positive_training_seeds": positive_seeds,
        "paired_delta_by_training_seed": json.dumps(seed_means, sort_keys=True),
        "stable_benefit": bool(
            float(deltas.mean()) > 0.0
            and ci_low > 0.0
            and positive_seeds >= math.ceil(2 * len(by_seed) / 3)
            and len(by_sequence) >= 3
        ),
    }


def analyze(
    detail_rows: list[dict[str, str]],
    feature_map: dict[tuple[str, int], dict[str, Any]],
    config: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    training_families = set(config.get("train_families", ()))
    baseline_by_key = {
        (int(row["training_seed"]), row["problem_id"]): row
        for row in detail_rows
        if row["method"] == "baseline"
    }
    enriched: list[dict[str, Any]] = []
    for row in detail_rows:
        key = (int(row["training_seed"]), row["problem_id"])
        baseline = baseline_by_key.get(key)
        if baseline is None:
            raise ValueError(f"Missing paired baseline for {key}")
        feature_key = (row["sequence_id"], int(row["sequence_step"]))
        if feature_key not in feature_map:
            raise ValueError(f"Missing reconstructed traffic features for {feature_key}")
        completion = int(float(row["completion_steps"]))
        synthesis = _float(row, "synthesis_ms")
        item: dict[str, Any] = {
            **row,
            **feature_map[feature_key],
            "training_seed": int(row["training_seed"]),
            "sequence_step": int(row["sequence_step"]),
            "completion_steps": completion,
            "synthesis_ms": synthesis,
            "timeout": _bool(row, "timeout"),
            "legal": _bool(row, "legal"),
            "training_distribution": (
                "training_family" if row["family"] in training_families else "heldout_family"
            ),
            "baseline_completion_time": int(float(baseline["completion_steps"])),
            "baseline_synthesis_time": _float(baseline, "synthesis_ms"),
            "paired_completion_delta": int(float(baseline["completion_steps"])) - completion,
            "paired_synthesis_delta_ms": _float(baseline, "synthesis_ms") - synthesis,
        }
        enriched.append(item)

    bucket_edges: dict[str, list[float]] = {}
    for dimension in NUMERIC_BUCKETS:
        labels, edges = _quantile_labels(item[dimension] for item in enriched)
        bucket_edges[dimension] = edges
        for item, label in zip(enriched, labels):
            item[f"bucket_{dimension}"] = str(label)
    for dimension in CATEGORICAL_BUCKETS:
        for item in enriched:
            value = item[dimension]
            item[f"bucket_{dimension}"] = str(value).lower() if isinstance(value, bool) else str(value)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for dimension in CATEGORICAL_BUCKETS + NUMERIC_BUCKETS:
        bucket_name = f"bucket_{dimension}"
        for item in enriched:
            grouped[(dimension, item[bucket_name], item["method"])].append(item)

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for (dimension, bucket, method), rows in sorted(grouped.items()):
        summaries.append(
            {
                "dimension": dimension,
                "bucket": bucket,
                "method": method,
                **_summarize_group(rows, bootstrap_samples, rng),
            }
        )
    return enriched, summaries, bucket_edges


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(
    summaries: list[dict[str, Any]],
    detail_rows: list[dict[str, str]],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    output_dir_display = str(Path(args.output_dir)).replace("\\", "/")
    detail_display = str(Path(args.detail)).replace("\\", "/")
    formal_display = str(Path(args.formal_summary)).replace("\\", "/")
    full = [row for row in summaries if row["method"] == "full"]
    stable = [row for row in full if row["stable_benefit"]]
    family_rows = [row for row in full if row["dimension"] == "family"]
    correlation_rows = [row for row in full if row["dimension"] == "sequence_total_acf1"]
    legal = all(str(row.get("legal", "")).lower() == "true" for row in detail_rows)
    timeout = sum(str(row.get("timeout", "")).lower() == "true" for row in detail_rows)
    independent_sequences = len({row["sequence_id"] for row in detail_rows})
    training_seed_count = len({row["training_seed"] for row in detail_rows})
    present_families = sorted({row["family"] for row in detail_rows})
    configured_training_families = set(config.get("train_families", ()))
    present_training_families = [
        family for family in present_families if family in configured_training_families
    ]
    present_heldout_families = [
        family for family in present_families if family not in configured_training_families
    ]
    stable_family_rows = [row for row in family_rows if row["stable_benefit"]]
    stable_family_names = [str(row["bucket"]) for row in stable_family_rows]
    stable_family_scope = (
        "仅出现在训练 family"
        if stable_family_names
        and all(name in configured_training_families for name in stable_family_names)
        else "并非仅出现在训练 family"
    )
    high_acf_row = next(
        (row for row in correlation_rows if str(row["bucket"]).startswith("q4:")), None
    )
    scope_line = (
        "本报告分析按原正式配置重建 checkpoint 后生成的训练/held-out 合并 paired detail；"
        "C1 的第一步已先直接分析旧正式 detail，确认缺少训练 family 覆盖后才进行重建复评。"
        "分桶分析本身不更新模型参数，V1 模型结构未修改。"
        if "training_family_eval" in detail_display
        else "本报告直接分析既有正式 V1 paired detail；没有重新训练，也没有修改 V1 模型。"
    )
    coverage_line = (
        "- 当前合并 detail 覆盖训练 family "
        f"{', '.join(f'`{name}`' for name in present_training_families) or '（无）'}；"
        "覆盖 held-out family "
        f"{', '.join(f'`{name}`' for name in present_heldout_families) or '（无）'}。"
    )

    lines = [
        "# V1 逐桶诊断",
        "",
        "## 执行范围",
        "",
        scope_line,
        f"原始记录 {len(detail_rows)} 条，独立 traffic sequence {independent_sequences} 条，训练 seed {training_seed_count} 个。",
        "bootstrap 以完整 `sequence_id` 为 cluster，不把重叠时间步当作独立样本。正的 paired delta 表示该方法比 baseline 少用 completion slot。",
        "",
        "## 结论",
        "",
        (
            f"- 稳定受益桶：{len(stable)} 个（判据：mean delta > 0、sequence-cluster bootstrap 95% CI 下界 > 0、至少三分之二训练 seed 为正、至少 3 条独立 sequence）。"
        ),
        f"- 全部输入 schedule 合法：{'是' if legal else '否'}；timeout 记录：{timeout}。",
        coverage_line,
        (
            "- family 级稳定受益："
            f"{', '.join(f'`{name}`' for name in stable_family_names) or '无'}；"
            f"{stable_family_scope}。稳定判据要求至少 2/3 seed 为正，"
            "不等同于三个 seed 方向全部一致。"
        ),
        (
            "- 最高时间相关（lag-1 ACF 的 q4）桶："
            f"mean delta={high_acf_row['paired_completion_delta_mean']:.4f}，"
            f"95% CI=[{high_acf_row['paired_completion_delta_ci95_low']:.4f}, "
            f"{high_acf_row['paired_completion_delta_ci95_high']:.4f}]，"
            f"稳定受益={'是' if high_acf_row['stable_benefit'] else '否'}；"
            "没有证据表明高相关场景比低相关场景更适合 moments。"
            if high_acf_row is not None
            else "- 没有可用的 sequence lag-1 ACF 分桶。"
        ),
        "- 高时间相关性是否更适合 moments 由下表直接给出；不能仅用 family 标签推断。",
        "",
        "## Moment-full family 结果",
        "",
        "| family | raw n | sequences | mean delta | bootstrap 95% CI | positive seeds | stable | synthesis delta ms |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['bucket']} | {row['raw_sample_count']} | {row['independent_sequence_count']} | "
            f"{row['paired_completion_delta_mean']:.4f} | "
            f"[{row['paired_completion_delta_ci95_low']:.4f}, {row['paired_completion_delta_ci95_high']:.4f}] | "
            f"{row['positive_training_seeds']}/{row['training_seed_count']} | "
            f"{'yes' if row['stable_benefit'] else 'no'} | {row['paired_synthesis_delta_ms_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 按 sequence lag-1 ACF 分桶",
            "",
            "| ACF bucket | sequences | mean delta | bootstrap 95% CI | stable |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in correlation_rows:
        lines.append(
            f"| {row['bucket']} | {row['independent_sequence_count']} | "
            f"{row['paired_completion_delta_mean']:.4f} | "
            f"[{row['paired_completion_delta_ci95_low']:.4f}, {row['paired_completion_delta_ci95_high']:.4f}] | "
            f"{'yes' if row['stable_benefit'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 输出与复现",
            "",
            f"- 汇总：`{output_dir_display}/bucket_summary.csv`",
            f"- 带重建流量特征的 paired 明细：`{output_dir_display}/bucket_enriched_detail.csv`",
            f"- 分桶边界与环境元数据：`{output_dir_display}/bucket_metadata.json`",
            "",
            "```bash",
            "python scripts/analyze_v1_by_bucket.py \\",
            f"  --detail {detail_display} \\",
            f"  --formal-summary {formal_display} \\",
            f"  --output-dir {output_dir_display}",
            "```",
            "",
            "限制：当前流量特征通过 formal summary 中记录的配置与 `sequence_id` seed 确定性重建；分析脚本会校验重建后的 sequence ID。`regime_duration` 是相对配置参考总量的 low/normal/high 在线驻留长度，不使用未来状态。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    detail_path = Path(args.detail)
    summary_path = Path(args.formal_summary)
    with detail_path.open(encoding="utf-8", newline="") as handle:
        detail_rows = list(csv.DictReader(handle))
    formal = json.loads(summary_path.read_text(encoding="utf-8"))
    config = formal["config"]
    feature_map = reconstruct_feature_map(detail_rows, config)
    enriched, summaries, bucket_edges = analyze(
        detail_rows, feature_map, config, args.bootstrap_samples, args.seed
    )

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "bucket_summary.csv", summaries)
    _write_csv(output_dir / "bucket_enriched_detail.csv", enriched)
    metadata = {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "command": sys.argv,
        "source_detail": str(detail_path),
        "source_formal_summary": str(summary_path),
        "config": config,
        "bootstrap": {
            "unit": "complete sequence_id cluster",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "bucket_edges": bucket_edges,
    }
    (output_dir / "bucket_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(summaries, detail_rows, config, args), encoding="utf-8")
    stable = sum(row["method"] == "full" and row["stable_benefit"] for row in summaries)
    print(
        json.dumps(
            {
                "raw_rows": len(detail_rows),
                "feature_rows": len(feature_map),
                "summary_rows": len(summaries),
                "stable_full_benefit_buckets": stable,
                "output_dir": str(output_dir),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
