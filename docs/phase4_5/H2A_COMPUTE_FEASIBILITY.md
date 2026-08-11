# H2a：计算可行性评估（Phase 4.5-A）

更新日期：2026-08-04
执行者：Systems Performance Agent（用户批准创建；子代理工具连续 4 次未执行任务，由主 Agent 在相同约束下接管完成，见 TASK_LEDGER）
数据来源：正式 artifacts（只读）`outputs/phase4_early_planning/`（本地只读副本 `phase4_formal_artifacts/`）
判定：**H2a = PASS（条件性）**

## 1. 结论摘要

robust 的 E2E ≈ 1042 ms 中，约 **92.3% 的在线耗时已被正式 frozen 计时组件解释**，瓶颈高度集中在两个组件：

| 组件 | 每 episode 平均 | 占 E2E |
|---|---:|---:|
| ambiguity_construction | 479.7 ms | 46.0% |
| prefix_synthesis | 450.9 ms | 43.3% |
| unattributed | 76.5 ms | 7.3% |
| support_selection | 7.7 ms | 0.7% |
| recourse_repair / fallback / checker_commit / h1_inference | ≈ 1.2–4.5 ms | <1% |

两组件合计 ≈ 931 ms（89.3% E2E / 91.1% online），是唯一需要针对性优化的对象。二者均为**单线程 Python/numpy 在 4 节点微拓扑上的重复计算**（scenario 重建 + 逐 slot 评分），不存在不可省略的算法性下限：理想化“只保留 checker/commit”下界 E2E ≈ 21.7 ms，远低于 104–116 ms baseline。

进入 baseline 1.5 倍（≈164.8 ms）需要在线耗时约 **7.1×** 加速；该加速对上述两类组件（向量化、协议允许的同 stage 缓存、候选/场景并行、固定历史离线预计算）是可信的。因此 H2a **PASS（条件性）**：必须由 flag 控制的等价性原型实证达到该加速，且 H2b 必须另行证明算法价值。

## 2. 方法与数据

- 只读分析正式 `raw_timing_metrics.csv`（21,600 行 = 2,700 episode × 8 组件）、`raw_test_episode_metrics.csv`（2,700 行）、`raw_test_sequence_metrics.csv`、`summary.json`。
- robust 方法 episode 数 = 300（15 sequence × 20 coordinates），与协议一致。
- 分析脚本：`outputs/phase4_5/h2a_profile/analyze_h2a.py`；产物：`a1_timing_by_method_component.json`、`a1_robust_episode_profile.csv/json`、`a1_robust_aggregates.json`、`a1_robust_relations.json`、`a1_all_methods_e2e.json`。

## 3. A1：精确耗时分解

### 3.1 robust 总体

| 指标 | 值 |
|---|---:|
| episode 数 | 300 |
| E2E mean | 1042.46 ms |
| E2E median / p95 / p99 | 1015.6 / 1588.1 / 2462.3 ms |
| total_online mean | 1021.96 ms |
| completion mean | 20.49 slots（≈20.5 ms） |
| 7 exclusive 组件和 mean | 945.4 ms |
| unattributed mean | 76.5 ms |
| **解释比例（exclusive/online）** | **92.3%** |

一致性检查：exclusive 和 + unattributed = 1021.9 ms ≈ total_online 1021.96 ms ✓。

### 3.2 组件明细（mean ms/episode，n=300）

| 组件 | total_s | mean | median | p95 | p99 | 占 E2E | 分类 |
|---|---:|---:|---:|---:|---:|---:|---|
| ambiguity_construction | 143.9 | 479.7 | 457.2 | 898.4 | 1060.6 | 46.0% | 在线必需，但部分可离线预计算/可缓存 |
| prefix_synthesis | 135.3 | 450.9 | 342.4 | 823.6 | 1526.8 | 43.3% | 在线必需，可向量化/并行/缓存 |
| unattributed | 23.0 | 76.5 | 73.7 | 100.1 | 147.3 | 7.3% | 实验记录+编排开销，可压缩 |
| support_selection | 2.3 | 7.7 | 7.3 | 10.5 | 11.5 | 0.7% | 在线必需（K=8 选择），开销小 |
| recourse_repair | 1.3 | 4.5 | 4.3 | 6.9 | 12.7 | 0.4% | 在线必需 |
| fallback | 0.4 | 1.4 | 1.3 | 2.5 | 2.9 | 0.1% | 在线必需（低频） |
| checker_commit | 0.4 | 1.2 | 1.2 | 1.7 | 2.2 | 0.1% | **不可省略（安全检查）** |
| h1_inference | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | robust 不使用 |

### 3.3 与 K/H/P/reveal/replan 的关系

- K：300/300 episode `actual_k_max=8`（stage 0–3 为 8，stage 4 为 1），与协议一致；K 不引入跨 episode 差异。
- H/P/λ：全部使用 frozen validation winner（manifest `selected_config`），无变化。
- replan：`true_replan_events` 7–15/episode（均值约 10），对应 stage change + exhaustion + invalidation；每次 replan 重建 support+prefix，是 ambiguity/prefix 高耗时的直接原因。
- 稳定性：5/5 family（967–1068 ms）、3/3 seed（964–1088 ms）、4 checkpoint（988–1070 ms）下 mean online 均约 1 s，结论跨 seed/family 稳定。

## 4. A2：Profiling 约束审计

现状：正式 frozen 计时已覆盖 8 组件，解释 92.3% 在线耗时，无需新增插桩即可完成 H2a 主结论。

若后续需要更细粒度分解（逐 slot、逐 candidate、逐 scenario），约束为：

1. instrumentation 必须 flag 控制，默认关闭；
2. profiling on/off 的事件与 artifact hash 必须一致（须有 equivalence 测试）；
3. 不得改变 RNG 序列、方法顺序、fail-closed 或 deterministic checker 路径；
4. 必须量化 profiler 自身开销（当前 unattributed 7.5% 已隐含此类开销上界）；
5. wall-clock 与子阶段累计时间必须做一致性检查（当前 92.3% 解释率即该检查的通过证据）；
6. GPU 计时不适用（本实验无 GPU 参与），但如引入需正确同步。

本轮**未**在生产代码中添加任何 profiler。

## 5. A3：理想化下界

### 5.1 单项 counterfactual（每 episode E2E 均值，若该组件免费）

| 组件免费 | E2E mean（ms） | 相对 baseline 109.8 ms |
|---|---:|---:|
| h1_inference | 1042.5 | 9.5× |
| ambiguity_construction | 562.7 | 5.1× |
| support_selection | 1034.7 | 9.4× |
| prefix_synthesis | 591.6 | 5.4× |
| recourse_repair | 1038.0 | 9.5× |
| fallback | 1041.1 | 9.5× |
| checker_commit | 1041.3 | 9.5× |
| unattributed | 965.9 | 8.8× |
| ambiguity + prefix 同时免费 | ≈111.8 | ≈1.0× |
| 只保留 checker_commit | 21.7 | 0.20× |
| 全部在线免费（仅 completion） | 20.5 | 0.19× |

### 5.2 加速要求

- baseline 参考 = (115.80 + 103.88)/2 = 109.84 ms；1.5× = 164.8 ms。
- 达到 1.5× baseline：允许在线 ≈144.3 ms → 需要 **7.1×** 在线加速。
- 达到 1× baseline：允许在线 ≈89.4 ms → 需要 **11.4×**。
- 单项 free 无法达标（562/592 ms）；必须同时压降 ambiguity 与 prefix。

### 5.3 可信实现路径（不改变语义）

1. **ambiguity_construction（480 ms）**：
   - 每 coordinate 的 32 个 recent-history 矩阵固定，其 descriptor/summary 可**离线预计算**（协议禁止跨 coordinate 复用，但同 coordinate 的固定历史预计算不改变语义）；
   - reconciliation 向量化（32 候选批量 numpy）；
   - 协议 §7 已允许同 stage 同 digest 的 immutable support 复用——核对并落实该缓存；
   - 候选间并行（12 核）。目标 8–15×。
2. **prefix_synthesis（451 ms）**：
   - 静态 scenario load 每 plan 只投影一次；逐 (scenario, candidate) 评分结果缓存；向量化 scoring；
   - 4 节点拓扑全距离线预计算，消除反复 BFS/distance；
   - 场景间/候选间并行。目标 5–10×。
3. **unattributed（77 ms）**：编排与事件记录开销，轻量化简 2–3×。
4. **checker_commit 保持**（1.2 ms，必要安全检查）。

预计 E2E 区间：ambiguity 30–60 + prefix 45–90 + unattributed 25–40 + 其余 ≈15 → online 115–205 ms → **E2E ≈136–226 ms**；中位估计 ≈150–165 ms（1.4–1.5× baseline）。必须由 flag 控制的等价性原型实证。

## 6. H2a 判据核对

| 判据 | 结果 |
|---|---|
| 解释 ≥90% 总时间 | PASS（92.3%） |
| 存在不关闭安全检查的可信优化路径 | PASS（向量化/缓存/并行/离线预计算；checker 语义不变） |
| 预计优化后 E2E 进入 baseline 1.5× 内或强条件收益 | PASS（中位估计 1.4–1.5×；敏感性 1.2–2.1×，见风险） |
| 不依赖“所有计算免费” | PASS（仅 checker 下界 21.7 ms 是界，不是方案） |
| 结论跨 seed/family | PASS（5/5 family、3/3 seed 稳定） |
| Supervisor 认可估计方法 | 见 `SUPERVISOR_REVIEW_H2A.md` |

## 7. 风险与条件

- **条件 1**：7× 加速必须由 flag 控制的原型实证；若原型仅达到 <5×，H2a 降级为 FAIL（进入四象限 3/4）。
- **条件 2**：H2a PASS 不代表 H2 可重判；H2b 必须另行证明算法价值。
- 敏感性：本估计的乐观端依赖并行效率；悲观端（串行、无缓存收益）E2E ≈226 ms（2.1×），不满足 1.5×，此时需更强条件收益支撑。
- 未做：未修改 production planner、未添加 profiler、未重跑正式实验、未开启 Phase 5。

## 8. 结论

**H2a = PASS（条件性）**：robust E2E 的高耗时是**可实现性**问题而非算法性下限问题；存在不触碰 checker/reveal/paired semantics 的 7× 级加速路径，预计可将 E2E 压入 baseline 1.5 倍以内。最终成立与否取决于等价性原型实测与 H2b 的算法价值判定。
