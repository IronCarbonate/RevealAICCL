# Phase 4.10-1F：P10-SF0 协议（Scheduler Fast-Path 有界可行性）

更新日期：2026-08-06

> **Phase R0 superseding note（2026-08-10）**：本文件是历史预注册/审计记录。
> 下文“理论最低/下界”措辞已被 R0 纠正：证据只支持 step-only 1,043.1µs、
> bind/checker-inclusive 1,139.5µs、digest-inclusive 2,047.2µs 的实现估算。
> 419.8µs 是 replay/quantized candidate window，不是 concurrent window。

## 1. 范围

只审计，不实施大规模优化。本阶段冻结 router、workload、75% budget、checkpoint 8、`partial_current_only` 与确定性 checker；
只回答：12.29ms 单步 p95 能否被精确解释、冻结实现的 fast-path estimate 是多少、是否可能满足
`scheduler p95 + first-commit p95 < 419.8µs`，并把实际目标预注册为 `<336µs`。

## 2. 判定门

### SF0-A：审计完整性

通过标准（全部满足）：

| # | 标准 | 本阶段证据 |
|---:|---|---|
| A1 | 冻结项未变（router/workload/75%/ckpt8/partial_current_only/checker） | 冻结声明 + 关键文件 md5 本地/远程一致 |
| A2 | 复现并精确分解单步 12.29ms p95 | p95 复现 11.29–12.93ms；组件分解 p95 合计 92.4%（≥90%）、均值合计 99.2% |
| A3 | 主导子成本归因 | BFS 距离重算 = enumerate 均值 90.1%；静态可缓存 |
| A4 | 静态/动态/可缓存/可增量/不可消除分类 | 完成（见 SCHEDULER_LATENCY_BREAKDOWN §7） |
| A5 | first-commit preparation latency 实测 | prep_full p95 = 8,673.8µs（含 checker） |
| A6 | 实现 fast-path estimates | step-only 1,043.1µs；含 bind/checker 1,139.5µs；含 digest 2,047.2µs；非理论下界 |
| A7 | 未触碰禁止项 | 未运行 formal、未生成/查看 formal test、未换 workload、未改生产路径、未跳过 checker、未改候选语义、未实现 GEMM/combine、无人工 delay、未改 75%/ckpt8、未恢复冻结机制、未进 DeepEP/L3、未创建 Subagent |

### SF0-B：优化计划可准入性（是否可向用户申请实施）

通过标准（全部满足）：

| # | 标准 | 本阶段证据 |
|---:|---|---|
| B1 | 计划只含纯实现层变换（等价性约束 §3） | 计划只含 E1–E7（预计算距离/增量 loads/融合 gate/惰性 digest/记忆化） |
| B2 | 计划不改变候选/排序/打包/checker 语义 | 等价性约束 §2 全保 |
| B3 | 计划在冻结 workload 上实测可达到预注册目标 L_total < 336µs | **FAIL**：step-only estimate 1,043.1µs > 336µs（3.1×）；含 bind/checker 为 1,139.5µs，无法认证 |
| B4 | 不实施大规模优化、不触碰生产路径（本阶段） | 仅测量探针；生产路径零改动 |

## 3. 预注册目标（正式 P10-1A 重检时冻结）

```
P4_pass := L_sched_p95 + L_commit_p95 < 336µs
  L_sched_p95 ≤ 200µs（scheduler 单步：enumerate+gate+pack）
  L_commit_p95 ≤ 135µs（bind + Proposal + commit_proposal checker）
```

测量口径：冻结 workload（P10-1E 输入）、真实计时、profiling OFF、含确定性 checker、无人工 delay、无预计算后延迟显示。

## 4. 判定规则与停止条件

1. SF0-A 全 PASS → 审计完成，进入 SF0-B 判定；
2. SF0-B 全 PASS → 向用户申请是否开始实施（仅纯实现层计划）；
3. **SF0-B 任一 FAIL → 不向用户申请实施；记录判定、P10-1 formal 保持 HOLD，停止并等待用户审核**；
4. 任何一步触碰禁止项 → 立即停止并如实上报。

## 5. 本阶段结论

- SF0-A = **PASS**（审计完整）；
- SF0-B = **FAIL**（预注册目标在当前有界纯实现层下不可认证）；
- P10-1 formal = **HOLD**；P10-F0 = FAIL/NO VETO 维持；
- 下一合法步骤：用户审核后决定方向（批准向量化/记忆化实现阶段、调整窗口/工作负载语义、或终止该路径）。
