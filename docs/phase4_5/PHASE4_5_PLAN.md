# Phase 4.5 Plan：H2 Failure Decomposition

更新日期：2026-08-04
状态：Phase 4.5-0 执行中（只读审计）；4.5-A / 4.5-B 待用户批准与 Systems Performance Agent 决策。

## 1. 目标与正式状态

项目主目标不变：提升 AICCL 在流量尚未完全揭示时的调度效果。

正式结论冻结（不可覆盖）：

- Phase 4 正式运行成功（exit 0），八产物原子发布于 `outputs/phase4_early_planning/`，staging 清零；
- Gate H2 = **FAIL**（conditions 1/3/6 明确失败），Phase 5 Gate = **CLOSED**；
- robust − Partial E2E Δ = −938.6（95% CI [−992.6, −896.7]）；robust − Wait E2E Δ = −926.7（95% CI [−977.0, −888.0]）；
- robust E2E ≈ 1042 ms，passive baselines ≈ 104–116 ms；scheduling-only Δ ≈ +0.11；
- robust completion ≈ 20.49，Partial ≈ 20.61，Wait ≈ 25.88；seed 正向 0/3、family 正向 0/5；legality 100%、timeout 0、CVaR95 优于 passive；
- 归因：主因 C（在线 ambiguity/prefix synthesis/replan overhead 过高），伴随 D/G（当前 reveal 设置下提前信息未形成足够可执行收益）。

## 2. 本阶段要回答的两个问题

- H2a（计算可行性）：约 1042 ms 的 robust E2E 能否在不改变调度语义、reveal 语义与安全约束的前提下，可信地降到接近 104–116 ms baseline 区间？
- H2b（算法价值）：忽略在线实现开销后，robust prefix 相对 `partial_current_only` 是否在预注册且样本充分的条件下有稳定且足够大的 scheduling-only 收益？

两者都通过，才允许提出“优化后重新评估 H2”。

## 3. 执行顺序

1. Phase 4.5-0：正式结果冻结与只读审计（本轮）
2. Phase 4.5-A：H2a 计算可行性（待批准）
3. Phase 4.5-B：H2b 算法价值（待批准）
4. Supervisor 独立审查
5. 四象限路线决策（H2a/H2b PASS/FAIL 组合）
6. 等待用户批准后才允许实施优化或新算法

不得跳过 H2a/H2b 直接重跑正式 Phase 4。

## 4. 角色与职责

| 角色 | 职责 |
|---|---|
| 主 Agent（Orchestrator / Lead） | 执行顺序控制、read-only 审计、文档、向用户申请 |
| Supervisor（Project Director） | 项目总监，独立审查，拥有暂停/否决/要求返工权力 |
| Core Research Engineer | 只读 artifact 审计与代码入口定位（本轮），后续执行 H2a/H2b 分析 |
| Systems Performance Agent（可选） | 唯一职责：H2a 性能分解与计算可行性评估；不改变调度语义、不关闭 checker、不自行大规模重构；完成 H2a 后退出 |

Systems Performance Agent **未经用户明确批准不得创建**。

## 5. 本轮（4.5-0）范围

- 读取 H2 正式结果、协议、Supervisor 报告；
- 独立 read-back 正式 artifacts，验证 raw→episode→sequence→conditions→summary 重算链；
- 记录 git commit（非 Git 工作区则写 unavailable）；
- 记录 Python / PyTorch / CUDA / hostname / CPU；
- 定位 ambiguity build、observation reconciliation、scenario selection、prefix proposal、multi-scenario scoring、feasibility check、state clone、commit、reveal、discard、replan、event ledger、artifact serialization 的真实代码入口；
- 创建：`FORMAL_RESULT_FREEZE.md`、`TASK_LEDGER_PHASE4_5.md`、`RISK_REGISTER_PHASE4_5.md`；
- 运行现有测试并记录真实结果；
- Supervisor 输出 Phase 4.5-0 审查；
- 向用户申请添加临时 Systems Performance Agent，然后停止。

## 6. 本轮禁止

- 创建 Systems Performance Agent / 新增其他 Subagent；
- 修改 production planner；
- 添加 profiler / 开始性能优化；
- 重跑 Phase 4 正式实验；
- 开启 Phase 5；
- 修改 H2 正式结论或正式 artifacts。

## 7. 输出文件

- `docs/phase4_5/PHASE4_5_PLAN.md`（本文件）
- `docs/phase4_5/FORMAL_RESULT_FREEZE.md`
- `docs/agent_coordination/TASK_LEDGER_PHASE4_5.md`
- `docs/agent_coordination/RISK_REGISTER_PHASE4_5.md`
- `docs/agent_coordination/SUPERVISOR_REVIEW_PHASE4_5_0.md`

后续阶段（批准后）：`docs/phase4_5/H2A_COMPUTE_FEASIBILITY.md`、`outputs/phase4_5/h2a_profile/`、`docs/agent_coordination/SUPERVISOR_REVIEW_H2A.md`、`docs/phase4_5/H2B_ALGORITHMIC_VALUE.md`、`outputs/phase4_5/h2b_analysis/`、`docs/agent_coordination/SUPERVISOR_REVIEW_H2B.md`。
