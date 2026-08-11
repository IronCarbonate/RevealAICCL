# Phase 4.11：结果表 / 图 / Artifact 索引

更新日期：2026-08-06

## 1. 图（Figures）

**本阶段及前序阶段未生成任何图片文件**（无 PNG/SVG/PDF 图表）。全部结果为 Markdown 表格 +
JSON/CSV 数值 + 文本型证据链。论文如需图表，须在 Phase 4.12（若获批）由既有数值重新绘制，且不得改变数值。

## 2. 结果表索引（文档 → 关键表）

| 文档 | 关键结果表 |
|---|---|
| `docs/uncertainty_aiccl/PHASE3B_AMBIGUITY_RESULTS.md` | 条件 1–6 判定表；paired delta 分解；LOFO 表 |
| `docs/uncertainty_aiccl/H1_PREDICTABILITY_RESULTS.md` | 七条件判定表；关键点预测对比表；跨 family 表 |
| `docs/uncertainty_aiccl/H2_EARLY_PLANNING_RESULTS.md` | H2 条件 1–8 判定表；方法均值表 |
| `docs/phase4_5/H2A_COMPUTE_FEASIBILITY.md` | 组件耗时明细表；理想化下界表；加速要求表 |
| `docs/phase4_5/H2B_ALGORITHMIC_VALUE.md` | scheduling-only 对比表；动作/prefix 表；分桶表 |
| `docs/phase4_6/W1W2_EVALUATION.md`、`W3_RISK_GATE.md` | W1 微基准表；W2 策略对比表；W3 分桶/留出表 |
| `docs/phase4_6/ROUTE_A_REVEAL_RESULTS.md` | 六档 reveal 主结果表；配对 bootstrap 表；四假设表 |
| `docs/phase4_7/H5_RESULTS.md` | 七臂 J 表；PASS 判据表 |
| `docs/phase4_7/H6_RESULTS.md` | 五选择器 × 三预算表；配对统计表 |
| `docs/phase4_7/H7_RESULTS.md` | 方案对比表；配对统计表 |
| `docs/phase4_8/PHASE4_8_2_MICROBENCH.md` | 15 项微基准表；成本校准表 |
| `docs/phase4_8/PHASE4_8_3_PILOT.md` | pilot 结果表；十问表；P0 检查表 |
| `docs/phase4_8/PHASE4_8_5_FORMAL_RESULTS.md` | L1 正式三臂表；配对统计表；D1 判据 14 项表 |
| `docs/phase4_8/PHASE4_8_6_L2_RESULTS.md`、`docs/phase4_9/L2_FINAL_REPORT.md` | L2 正式三臂表；配对统计表；NCCL 微基准表 |
| `docs/phase4_10/P10_I1_RESULTS.md` | T1–T7 17 项检查表 |
| `docs/phase4_10/P10_1C_PILOT_RESULTS.md` | pilot 两臂表；配对/边界表 |
| `docs/phase4_10/P10_1D_TIMING_RESULTS.md` | 三臂结果表；effects 表；P10-T0 检查表 |
| `docs/phase4_10/P10_1E_FORMAL_ADMISSIBILITY.md` | P1–P4 证明表；指标冻结表 |
| `docs/phase4_10/SCHEDULER_LATENCY_BREAKDOWN.md` | 单步组件分解表；每 stage 表；first-commit 表；分类表 |
| `docs/phase4_10/SCHEDULER_FAST_PATH_LOWER_BOUND.md` | 探针表；下界汇总表；可行性对比表 |
| `docs/phase4_10/PHASE4_10_FINAL_REPORT.md` | 门链表；三类结论表；hotspot 边界表 |

## 3. Artifact 索引（outputs/ 权威副本）

### 3.1 `outputs/h1_predictability/`

`manifest.json`（C702D8CE…）、`raw_sequence_metrics.csv`（1,590 行）、`raw_probability_metrics.csv`（180 行）、`summary.json`。

### 3.2 `outputs/phase3b_ambiguity/`

10 文件（schema-v2）：`manifest.json`（DF821805…）、raw_calibration_scores（4,800）、raw_validation_metrics（19,200）、
raw_case_metrics（120,000）、raw_sequence_metrics（300）、raw_lofo_*（19,200/76,800/9,600）、raw_dependence_metrics（240）、
`summary.json`（0628310C…）。八张 raw 表合计 250,140 行。

### 3.3 `phase4_formal_artifacts/`（H2 正式只读副本）

`manifest.json`、`raw_validation_metrics.csv`（9,600）、`raw_test_episode_metrics.csv`（2,700）、
`raw_test_sequence_metrics.csv`（135）、`raw_test_execution_events.csv`（147,690）、`raw_timing_metrics.csv`（21,600）、
`raw_timing_metrics.csv`、`summary.json`、`h1_best_point_model.json`。正式 canonical 位于远程
`outputs/phase4_early_planning/`（exit 0，八产物原子发布）。

### 3.4 `outputs/phase4_5/`

`h2a_profile/`：analyze_h2a.py、a1_timing_by_method_component.json、a1_robust_episode_profile.csv/json、
a1_robust_aggregates.json、a1_robust_relations.json、a1_all_methods_e2e.json；
`h2b_analysis/`：analyze_h2b.py、h2b_analysis.json、h2b_per_sequence.csv。

### 3.5 `outputs/phase4_6/`

`route_a_reveal/`：route_a_runner.py、route_a_results.json；
`w1_static_precompute/`：w1_static_precompute.py、w1_precomputed.json；
`w2_scheduler/`：w2_scheduler.py、w2_diagnostic.py、w2_diagnostic_subset.json、w2_diagnostic_full.json；
`w3_risk_gate/`：w3_risk_gate.py、w3_risk_gate.json。

### 3.6 `outputs/phase4_7/`

`h5_realizable_reveal/`：corpus_h5.py、cost_calibration.py、h5_runner.py、h5_test.json；
`h6_selective_reveal/`：h6_runner.py、h6_test.json；
`h7_adaptive_reveal/`：h7_runner.py、h7_test.json。

### 3.7 `outputs/phase4_8/deployment_validation/`

L1：formal_test.py、real_exec.py、microbench.py/results.json、pilot.py/results.json、final_summary.json、
condition_summary.json、timing_breakdown.json、throughput_results.json、job_sequence_results.json、i1_equivalence.json、
raw_jobs.json、hashes.json、integrity_manifest.json、protocol_manifest.json、run_command.txt、environment_manifest.json；
L2：l2_final_summary.json、l2_condition_summary.json、l2_timing_breakdown.json、l2_throughput_results.json、
l2_job_sequence_results.json、l2_environment_manifest.json、l2_collective_results.json、l2_collective_bench.py、
hashes_l2.json、integrity_manifest_l2.json、read_back_report_l2.json、readback_l2.py。

### 3.8 `outputs/phase4_10/`

`p10_1a_substrate/`：reference_router.py、p10_i1_tests.py、p10_i1_results.json；
`p10_1c_pilot/`：p10_1c_pilot.py、p10_1c_pilot_results.json；
`p10_1d_timing/`：p10_1d_timing.py、p10_1d_timing_results.json；
`p10_1e_admissibility/`：p10_1e_readiness_test.py、p10_1e_readiness_test.json；
`p10_1f_audit/`：p10_1f_scheduler_breakdown.py、p10_1f_scheduler_breakdown.json。

### 3.9 其余（背景/审计）

`outputs/traffic_audit/`、`outputs/moment_v1/`、`outputs/performance/`、`outputs/strategies/`、`outputs/xml/`、
`outputs/v1_diagnosis/`（早期 Phase 0–2 证据，仅背景引用）。

## 4. 一致性说明

- Phase 4.10 全部 11 项输出 artifacts 本地/远程 md5 一致（见 `PHASE4_10_REPRODUCIBILITY_MANIFEST.md`）；
- L2 产物 10 项与 `hashes_l2.json` 匹配（`read_back_report_l2.json`）；Phase 3B 十文件 final read-back PASS；
- 所有统计以文档 + JSON 双记录为准；论文写作引用本文档索引即可回溯到原始 artifact。
