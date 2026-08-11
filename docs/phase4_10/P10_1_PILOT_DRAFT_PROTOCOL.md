# P10-1 Pilot 草案协议（DRAFT）

更新日期：2026-08-05
状态：**DRAFT（P10-I1=PASS 后，用户批准后实施）**

## 1. 目标

在 L2-R reference router 数据流上运行 development/validation pilot：确认 D1 vs D0 的 E2E 方向与 L2 结论一致，且 router 路径计时为 M 级。

## 2. 数据流（固定）

token 流 → reference router（gating Linear → top-k）→ router 派生 traffic → partial_shards reveal（D1 75%@8 / D0 full@16）→ partial_current_only 调度 →（collective 为真实 NCCL）。

## 3. 固定与禁止

- 固定：frozen profile、router 权重（冻结）、scheduler、checker、fail-closed；
- 禁止：调参、恢复被冻结机制、实现真实 GEMM/combine（P10-2 前）、Triton 优化、DeepEP、L3、3042 作为正式 test；
- 命名：L2-R reference，不称生产 router。

## 4. Pilot 内容

- 新开发/validation workload（不得用 3042 作为正式 test；用全新种子）；
- 指标：ΔE2E（D1 vs D0）、router/top-k/shard 时延（M）、completion、吞吐、legality、timeout；
- 等价性：D0/D1 相同 token/权重/top-k（逐 token 断言）；token 一致性检查；
- 统计：sequence-level paired bootstrap。

## 5. P10-1 Pilot PASS（草案）

1. router 路径可运行且计时 M 级；
2. ΔE2E > 0 且 CI lower > 0（若方向与 L2 一致）；
3. legality 100%、timeout 不增；
4. 无泄漏、token 一致；
5. Supervisor PASS。

## 6. 输出

- `outputs/phase4_10/p10_1_pilot/`；`docs/phase4_10/P10_1_PILOT_RESULTS.md`；`docs/agent_coordination/SUPERVISOR_REVIEW_P10_1_PILOT.md`。
