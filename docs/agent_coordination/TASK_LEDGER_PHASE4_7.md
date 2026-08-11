# Task Ledger — Phase 4.7

更新日期：2026-08-04

## Phase 4.7-0：真实 Reveal 语义映射（本轮）

| # | 任务 | 负责人 | 状态 |
|---|---|---|---|
| 1 | 读取执行指令与冻结结果（Route A/H2/Phase 4.6） | 主 Agent | 完成 |
| 2 | 冻结 H1 FAIL / H2 FAIL / Phase 5 CLOSED / Route A PASS | 主 Agent | 完成（未改判） |
| 3 | 固定 `partial_current_only` 为 fast-layer baseline | 主 Agent | 完成（协议声明） |
| 4 | 审计 router/top-k/token aggregation/histogram/dispatch/sync/traffic matrix 数据流 | Core（主 Agent 执行） | 完成 → `REALIZABLE_REVEAL_SEMANTICS.md` |
| 5 | 建立 reveal capability table | Core | 完成 → `REVEAL_CAPABILITY_TABLE.md` |
| 6 | 建立 reveal cost model | Core | 完成 → `REVEAL_COST_MODEL.md` |
| 7 | 区分真实可实现与仅 proxy 可实现 reveal | Core | 完成（语义/能力表 §4） |
| 8 | R0 判定 | Supervisor | 完成 → `SUPERVISOR_REVIEW_PHASE4_7_0.md` |
| 9 | 输出 H5 draft protocol | 主 Agent | 完成 → `H5_DRAFT_PROTOCOL.md` |
| 10 | 停止等待用户审核 | 主 Agent | 本轮停止点 |

## H5（R0 PASS 后，待用户批准）

- 固定 Partial fast runner；新 corpus（seeds 2042/2142/2242 草案）；7 比较臂；成本计入 E2E J；sequence-level 统计；H5 PASS/FAIL 预注册。

状态更新（2026-08-04）：用户批准；H5 协议正式化（`H5_PROTOCOL.md`）；新 corpus 生成（2042/2142/2242，45 条，与 H2/Route A 零重合）；成本校准（histogram 336ns、msg 8.8µs 实测；collective 假设值标注）；test 运行 7 臂×300 coords。结果：**A2 +6.06 / A3 +5.98 / A4 +9.22 ms（CI lower>0、15/15 seq、5/5 family、3/3 seed）；A5 −0.13 ms（全负）**。判定：**H5 = PASS**（A2/A3/A4）。报告：`H5_RESULTS.md`、`SUPERVISOR_REVIEW_H5.md`（PASS/NO VETO）；产物：`outputs/phase4_7/h5_realizable_reveal/`。下一步：H6（固定预算选择性 reveal），需用户批准。

状态更新（2026-08-04）：用户批准 H6。5 预注册选择器 × 3 预算（25/50/75%）× 300 coords：**partial_shards 在所有预算下显著优于 random**（+0.60/+0.81/+0.57 ms，CI lower>0、14-15/15 seq、5/5 family、3/3 seed）；entry 级选择器与 random 无差异（source_totals Δ=0）。判定：**H6 = PASS**（最佳固定 profile = partial_shards；预算单位差异已标注）。报告：`H6_RESULTS.md`、`SUPERVISOR_REVIEW_H6.md`（PASS/NO VETO）；产物：`outputs/phase4_7/h6_selective_reveal/`。下一步：H7（自适应 reveal controller），需用户批准。

状态更新（2026-08-04）：用户批准 H7。规则控制器（validation 拟合 5 特征桶 → test 评估）：**退化选择 0.75（300/300），与固定 B75 完全等价（ΔJ=0.000 ms）**；oracle 每 episode 最优相对 B75 仅好 0.0014 ms（异质性存在但价值可忽略）。判定：**H7 = FAIL**——保留最佳固定 profile = **partial_shards @ 75%（full reveal slot 8）**；不保留自适应 controller。Phase 4.7 收敛结论：可实现早期揭示带来约 8–14% E2E 改善（计入成本），自适应无价值。报告：`H7_RESULTS.md`、`SUPERVISOR_REVIEW_H7.md`（FAIL/NO VETO）；产物：`outputs/phase4_7/h7_adaptive_reveal/`。

## Phase 4.7-F：Formal Closure（本轮，用户批准）

| # | 任务 | 状态 |
|---|---|---|
| 1 | PHASE4_7_FINAL_REPORT.md | 完成 |
| 2 | FINAL_DEPLOYMENT_RECOMMENDATION.md | 完成 |
| 3 | EVIDENCE_CHAIN.md | 完成 |
| 4 | NEGATIVE_RESULTS_SUMMARY.md | 完成 |
| 5 | REPRODUCIBILITY_MANIFEST.md | 完成 |
| 6 | TASK_LEDGER + DECISION_LOG 更新 | 完成 |
| 7 | Supervisor 最终独立审查 | 完成（PASS/NO VETO） |
| 8 | 可声称/不可声称结论清单 | 完成（见 FINAL_REPORT §4） |
| 9 | Phase 4.8 Real-Deployment Validation 草案协议 | 完成（DRAFT） |
| 10 | 停止等待用户审核 | 本轮停止点 |

最终冻结：R0 PASS / H5 PASS / H6 PASS / H7 FAIL / H2 FAIL / Phase 5 CLOSED；固定方案 = partial_shards @ 75% / full@slot 8 / partial_current_only / 其余全关。

## H6（H5 PASS 后）

- 固定预算 B 下选择性 reveal（random/fixed/priority/oracle VOI 上界）。

## H7（H6 PASS 后）

- 自适应 reveal controller（规则/contextual bandit/simple supervised；禁止直接大型 RL）。
