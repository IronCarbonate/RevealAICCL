# Phase 4.6 W1 + W2 评估报告（象限 2 首批工作流）

更新日期：2026-08-04
执行环境：`autodl-container-36da11a152-db2cf032`（冻结 venv，PYTHONHASHSEED=0）
数据：冻结正式 artifacts（只读）+ 冻结代码原语；全部新代码位于 `outputs/phase4_6/`，未修改 production 文件

## W1：静态预计算与候选准备

脚本：`outputs/phase4_6/w1_static_precompute/w1_static_precompute.py`
产物：`w1_precomputed.json`

结果：

| 项 | 值 |
|---|---|
| topology digest | `12e2a329...`（与正式 manifest 一致） |
| 预计算对象 | 4 节点拓扑全部 12 个 OD 对的 canonical 最短路径、距离、edge/group 单位容量、usable 边 |
| 等价性 | 12/12 OD 对与冻结 `canonical_shortest_path` / `_shortest_path_edges` 逐位一致，0 差异 |
| 微基准 | 单次路径/距离重算 ≈ 57.7 µs；预计算查表 ≈ 0.19 µs；**加速 ≈ 302×** |

评估：预计算与冻结语义 bit 级等价，可安全用于查询加速；但 Partial 的每 episode wall 仅 ≈45 ms，路径/距离重算占比约 3–6 ms（估计 ~10%），因此对 E2E 的收益有限、对 completion **无影响**（语义不变）。W1 的价值是**为未来所有调度器提供零成本静态结构**，而非直接改善完成时间。

## W2：当前观测确定性调度改进

脚本：`outputs/phase4_6/w2_scheduler/w2_scheduler.py` + `w2_diagnostic.py`
产物：`w2_diagnostic_subset.json`、`w2_diagnostic_full.json`

等价性门（诊断运行器对 frozen `partial_current_only` 的忠实性）：

- subset 100/100、full 300/300：completion、first_action、legality、executed_actions 与正式 artifacts **完全一致**（0 差异）。

全量结果（15 sequence × 20 coordinates = 300 episode/策略）：

| 策略 | completion | legality | first_action | actions | wall_ms |
|---|---:|---:|---:|---:|---:|
| baseline（=Partial） | 20.607 | 1.000 | 4.107 | 23.9 | 45.2 |
| distance（按剩余跳数排序） | 20.607 | 1.000 | 4.107 | 23.9 | 45.3 |
| headroom（按容量余量排序） | 20.607 | 1.000 | 4.107 | 23.9 | 45.3 |
| lookahead（min-blocking 贪心） | 20.577 | 1.000 | 4.107 | 23.9 | 74.5 |

sequence-level 配对（lookahead vs baseline）：

- 15 条 sequence：5 正 / 3 负 / 7 平；按 family：4/5 正但量级 0.017–0.133 slots；按 seed：+0.08 / 0.00 / +0.01；
- bootstrap（10,000，seed 20260801）：mean +0.030，**CI [−0.0067, +0.0767] 跨 0，不显著**；
- lookahead 计算成本 +65%（74.5 vs 45.2 ms）。

## 结论

1. **W1 = PASS（等价性）**：静态预计算与冻结语义逐位一致，提供 ~302× 查询加速；对 completion 无影响，对 Partial E2E 的节省估计 ~10%。
2. **W2 = 无有效收益**：distance/headroom 与 baseline 完全相同；lookahead 提升 +0.03 slots 且 CI 跨 0，不显著，并增加计算成本。改进候选排序**无法**压缩 completion。
3. 含义：completion 差距（Partial 20.6 vs full-info executable 9.9 vs LB 3.35）主要受**信息揭示节奏与容量**约束，而非候选排序/打包次序；当前观测确定性调度已接近其可行前沿。与 H2b 结论一致：多场景评分与候选排序都无调度价值增量。
4. 对后续：W3（风险检测与有限启用 gate）仍值得做——其价值不在于调度器排序，而在于**判断哪些 episode 值得提前行动/等待**（利用 mode/checkpoint/已揭示 demand 特征），以及把 W1 静态结构用于成本侧。

## 约束审计

- 未修改 production planner / checker / reveal；正式 artifacts 只读；legality 100%；timeout 0；未开启 Phase 5；新代码全部在 `outputs/phase4_6/`。
