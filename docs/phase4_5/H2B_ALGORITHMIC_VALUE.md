# H2b：算法价值评估（Phase 4.5-B）

更新日期：2026-08-04
数据来源：正式 artifacts（只读）`outputs/phase4_early_planning/`（本地只读副本）
判定：**H2b = FAIL**

## 1. 结论摘要

robust prefix 相对 `partial_current_only` 的 scheduling-only（completion）收益为 **+0.11 slots（≈0.11 ms）**，统计上稳定为正（sequence-level paired 95% CI [0.087, 0.140]，13/15 sequence 正向、3/3 seed 正向、5/5 family 正向），但**量级过小**：

- 收益 ≈ 0.11 ms，连 H2a 的“只保留必要 checker”预算（≈1.2 ms）都覆盖不了，更无法覆盖任何可信实现的在线规划预算（≈140 ms，1.5× baseline 允许值）；
- robust 的已执行动作与 Partial 动作集合**98.5% 重合**，首动作一致 71.0%，同 slot 动作一致 62.6%——K=8 多场景评分基本没有带来决策差异；
- 不存在收益足够大的预定义工作区间（所有分桶收益 0.05–0.27 slots，mode 0 为 −0.05）。

结论：robust prefix **缺乏独立于 Partial 的算法价值**；相对 Wait 的优势完全来自“提前处理已揭示 demand”（与 Partial 相同），而非多场景稳健规划。

## 2. 数据与方法

- 正式 `raw_test_episode_metrics.csv`（2,700 行）、`raw_test_sequence_metrics.csv`（135 行）、`raw_test_execution_events.csv`（147,690 行）、`summary.json`、`manifest.json`。
- 主比较：`scenario_robust_prefix` vs `partial_current_only`；对照：`wait_until_known`。
- 统计：sequence-level paired delta（15 sequence），family-stratified bootstrap 10,000 次（seed 20260801，同正式协议）；positive-sequence ESS；事件账本动作级对齐。
- 脚本：`outputs/phase4_5/h2b_analysis/analyze_h2b.py`；产物：`h2b_analysis.json`、`h2b_per_sequence.csv`。

## 3. B1：量化结果

### 3.1 scheduling-only（completion）与尾部

| 指标 | robust | partial | delta（partial−robust，正=robust 优） |
|---|---:|---:|---:|
| completion mean（15 seq 等权） | 20.49 | 20.61 | **+0.113**（bootstrap CI [0.087, 0.140]） |
| completion CVaR95 | 更低 | 更高 | +0.134（CI [−0.133, +0.533]，跨 0） |
| E2E mean | 1042.5 ms | 103.9 ms | −938.6（实现开销，见 H2a） |

- 符号：13/15 sequence 正向、0 负、2 平。
- ESS：75 series（15 seq × 5 mode × 4 checkpoint）mean 3.85、sum 288.6、lag1 ACF −0.23。

### 3.2 动作与 prefix

| 指标 | 值 |
|---|---:|
| robust 动作出现在 Partial 动作集合的比例 | **98.5%** |
| 首动作一致率 | 71.0%（213/300） |
| 同 slot 动作一致率（mean） | 62.6% |
| 完整动作序列一致率 | 11.7%（35/300） |
| proposed prefix actions（mean） | 23.95 |
| committed prefix actions（mean） | 23.95 |
| commit/proposal 比 | 1.0 |
| discarded unexecuted actions（mean） | **0.0** |
| true replan（mean） | 9.37（reveal 4.0 + exhaustion 5.37 + invalidation 0） |
| no_common_action / fallback episode 比例 | 300/300（100%） |
| residual repair actions（mean） | 7.23 |

### 3.3 reveal 相关

| 指标 | robust | partial | wait |
|---|---:|---:|---:|
| first_action_slot（mean） | 4.11 | 4.11 | 16.0 |
| reveal 前完成率 | 0% | 0% | — |
| reveal 前已执行 actions（mean） | 16.72 | 16.73 | 0 |
| completion（mean） | 20.49 | 20.61 | 25.88 |

### 3.4 分桶（completion delta）

| 分桶 | 各桶 delta（slots） | 说明 |
|---|---|---|
| family（5 桶 ×3 seq） | +0.067 ~ +0.167，5/5 正 | 桶样本不足 5 seq，仅参考 |
| base_seed（3 桶 ×5 seq） | +0.090 ~ +0.140，3/3 正 | 样本充分，均微小 |
| mode_index（5 桶 ×3 seq） | +0.050 ~ +0.267，4/5 正（mode 0 = −0.05） | 无单调 reveal-latency 趋势 |
| checkpoint_index（4 桶 ×3.75 seq 均值） | +0.053 ~ +0.160，4/4 正 | 历史长度无单调趋势 |
| true_replan（9 桶） | replan 7→15 时 completion 19.5→26.0 | replan 更多是困境症状，非收益来源 |

## 4. B2：十个问题与回答

1. **robust 是否多数时候与 Partial 相同动作？** 是。动作集合重合 98.5%，首动作一致 71%，同 slot 一致 62.6%。
2. **公共 prefix 是否过短？** 完整动作序列精确一致仅 11.7%——公共**序列**短；但动作**集合**几乎相同，差异主要在时序而非选择。
3. **prefix 是否经常被 discard？** 否。discarded=0、commit/proposal=1.0；replan 全部发生在 prefix 已完整执行后（stage change / exhaustion），无 invalidation。
4. **CVaR 优势来自哪些 sequence/family？** 无稳定来源：CVaR95 paired CI [−0.133, +0.533] 跨 0，优势不显著。
5. **robust 优于 Wait 是否只因提前处理已揭示 demand？** 是。首动作 4.11（=Partial）vs Wait 16.0；reveal 前动作数 16.72（≈Partial 16.73）；+5.4 slots 的优势全部来自“不等待”。
6. **相对 Partial 的 +0.11 是否稳定？** 统计稳定（CI>0、13/15 seq、3/3 seed、5/5 family），但量级 ≈0.5% completion、≈0.11 ms。
7. **是否存在样本充分的正向区间？** 无。所有分桶均 ≤0.27 slots，无足够大的正向工作区间；mode 0 为负。
8. **reveal delay 增大时收益是否增加？** 无证据。reveal 进度固定（full reveal slot 16）；mode/checkpoint 分桶无单调关系。
9. **K=8 是否只有覆盖价值没有决策价值？** 是。K=8 多场景评分产生的动作与仅看当前 observation 的 Partial 几乎相同（98.5% 重合），覆盖未转化为决策差异。
10. **coverage 与 scheduling benefit 是否脱钩？** 是。300/300 episode actual_k_max=8（全覆盖），收益仍仅 +0.11 slots，覆盖与调度收益脱钩。

## 5. H2b 判据核对

| 判据 | 结果 |
|---|---|
| 存在预定义且样本充分的工作区间 | **FAIL**（全部分桶收益 ≤0.27 slots，无足够大区间） |
| robust scheduling-only 相对 Partial 稳定正向 | PASS（+0.11，CI [0.087, 0.140]） |
| ≥3 seed 中满足预注册多数 | PASS（3/3 正） |
| sequence-level paired CI lower > 0 | PASS（0.087） |
| ≥4/5 family 正向或明确适用边界 | PASS（5/5 正） |
| 收益足以覆盖可信实现的规划预算 | **FAIL**（0.11 ms vs 必要 checker ≈1.2 ms、vs 优化预算 ≈140 ms） |
| 非仅改善 CVaR 而损害 mean/regret | PASS（completion mean 亦正） |
| Supervisor 认可 | 见 `SUPERVISOR_REVIEW_H2B.md` |

FAIL 触发条件：**“robust 与 Partial 基本相同”（98.5% 动作重合、+0.11 slots）成立**；“K=8 只有覆盖价值”（问题 9/10）成立；“只相对 Wait 有优势，对 Partial 无优势”（问题 5/6）成立。

## 6. 结论与四象限

**H2b = FAIL。**

结合 H2a = PASS（条件性）：落入**象限 2（H2a PASS / H2b FAIL）**——可以加速，但 robust prefix 缺乏独立决策价值。按指令，停止大规模优化，转向：

- anticipatory preparation / candidate preparation；
- risk detection 与高风险 episode 的有限启用 gate；
- current-observation-only online scheduling（Partial 路线）；
- topology / static precomputation、relay/path ranking。

不得在 H2b FAIL 下实施多场景 robust prefix 优化或重评 H2。
