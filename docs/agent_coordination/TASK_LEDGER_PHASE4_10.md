# Task Ledger — Phase 4.10

> **Phase R0 superseding note（2026-08-10）**：本 ledger 保留历史执行措辞。
> “419.8µs actionable window / 1.045ms theoretical lower bound”已纠正为 replay
> candidate window 与 implementation estimates（1,043.1/1,139.5/2,047.2µs）。
> 历史 formal CLOSED 不禁止新 concurrent/event-driven architecture。

更新日期：2026-08-05

## Phase 4.10-0：Production-Path Admissibility Audit（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 冻结既有结论与 deployment profile | 完成 |
| 2 | 审计 router/top-k/token arrival/histogram/shard/expert packing/GEMM/combine/DeepEP 路径 | 完成 |
| 3 | 四态分类（存在/可编译/可运行/进 critical path） | 完成 |
| 4 | D0/D1 真实 router 路径公平映射 | 完成 |
| 5 | measurement capability table | 完成 |
| 6 | P10-1 draft protocol | 完成（DRAFT） |
| 7 | Supervisor P10-R0 判定 | 完成（CONDITIONAL PASS/NO VETO） |
| 8 | 停止等待用户审核 | 本轮停止点 |

## 后续

- P10-1：真实 router/top-k/shard readiness（需用户批准；新建实现 + 等价性门）；
- P10-2：真实 expert GEMM/packing/combine；
- DeepEP：V100 不支持，需 Ampere/Hopper 硬件后另行评估。

## Phase 4.10-1A：Executable MoE Substrate Selection（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 冻结结论与 profile（L2-R 命名边界） | 完成 |
| 2 | 搜索仓库/环境 MoE reference（a/b 不存在，c 选定） | 完成 |
| 3 | 数据流定义（tensor→logits→topk→histogram→shard-ready） | 完成 |
| 4 | D0/D1 相同 token/权重/top-k 证明设计 | 完成 |
| 5 | correctness oracle 与 token 一致性检查定义 | 完成 |
| 6 | P10-S0 protocol | 完成（FROZEN） |
| 7 | P10-I1 draft protocol | 完成（DRAFT） |
| 8 | Supervisor P10-S0 判定 | 完成（PASS/NO VETO） |
| 9 | 停止等待用户审核 | 本轮停止点 |

## Phase 4.10-1B：Reference Router Bridge Implementation and Equivalence（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 修订实施：router 派生 traffic 为 ground truth、真实 CUDA shard-ready、lexicographic tie-break、L2-R 命名 | 完成 |
| 2 | 实现最小 PyTorch reference router | 完成 |
| 3 | 实现 CPU correctness oracle | 完成 |
| 4 | 实现 router-derived histogram/traffic | 完成 |
| 5 | 实现真实 shard-ready event（CUDA 完成事件） | 完成 |
| 6 | D0/D1 接入相同 router 数据流 | 完成（共享 stream 断言） |
| 7 | no-leak 与 token consistency 检查 | 完成 |
| 8 | P10-I1 全部测试 | 完成（17/17 PASS） |
| 9 | Supervisor P10-I1 判定 | 完成（PASS/NO VETO） |
| 10 | P10-1 pilot draft protocol | 完成（DRAFT） |
| 11 | 停止等待用户审核 | 本轮停止点 |

## Phase 4.10-1C：Reference Router Bridge Pilot（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 冻结 reference router / 权重 / token corpus split / profile | 完成 |
| 2 | D0/D1 相同 token arrival/logits/top-k/histogram/traffic | 完成（20/20 断言） |
| 3 | shard-ready 无 per-shard sync（异步 CUDA 事件） | 完成 |
| 4 | profiling OFF/ON 分别测量 | 完成 |
| 5 | CUDA event/query/instrumentation overhead 量化 | 完成（噪声已标注） |
| 6 | RR-D0/RR-D1 paired pilot | 完成 |
| 7 | D1 75%/ckpt8 来自真实 router completion | 完成 |
| 8 | 记录 router/shard/scheduler/NCCL/completion/E2E/throughput/legality/timeout | 完成 |
| 9 | hotspot_random_walk 保留并分析 | 完成（−32.8ms，如实） |
| 10 | 独立 read-back pilot artifacts | 完成（0 差异） |
| 11 | Supervisor P10-P0 判定 | 完成（CONDITIONAL PASS/NO VETO） |
| 12 | P10-1 formal draft protocol | 完成（DRAFT） |
| 13 | 停止等待用户审核 | 本轮停止点 |

## Phase 4.10-1D：Timing Semantics and Measurement Stabilization（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 三臂（B0/C0/C1） | 完成 |
| 2 | 真实 chunked token arrival + 逐 chunk router | 完成 |
| 3 | shard-ready = 每 chunk CUDA 完成 | 完成 |
| 4 | 无全量 top-k 后切片 | 完成 |
| 5 | C0/C1 共享 token/arrival/router/top-k/traffic | 完成（20/20） |
| 6 | reveal effect = C0−C1；deployment = B0−C1 | 完成（completion +5.2/+5.2；E2E ≈0/+11.2ms amortized） |
| 7 | cold/steady/amortized 分别报告 | 完成 |
| 8 | 交替顺序 overhead 测量 | 完成（~11ms 稳定） |
| 9 | 主结果 profiling OFF | 完成 |
| 10 | hotspot_random_walk 保留分解 | 完成 |
| 11 | P10-T0 测试 | 完成（全 PASS） |
| 12 | Supervisor P10-T0 判定 | 完成（PASS/NO VETO） |
| 13 | 修订 P10-1 formal protocol | 完成（DRAFT） |
| 14 | 停止等待用户审核 | 本轮停止点 |

## Phase 4.10-1E：Formal Protocol Admissibility and Workload Freeze（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | steady-state B0−C1 唯一部署主指标 / C0−C1 reveal 次级主指标 | 完成 |
| 2 | cold/amortized 降级 | 完成 |
| 3 | 不对称消除（warmup/世界/NCCL）与 Latin-square 顺序 | 完成 |
| 4 | workload scale matrix 预注册（360 格点） | 完成 |
| 5 | replay/quantized candidate window 定义 | 完成；R0 明确非 concurrent window |
| 6 | 证明测试（P1–P4） | 完成（P4 FAIL） |
| 7 | 无 artificial sleep / 预计算延迟 | 完成 |
| 8 | 修订 P10-1 formal protocol | 完成（HOLD） |
| 9 | 预注册 P10-1A/P10-1B Gate | 完成 |
| 10 | hotspot_random_walk 保留 | 完成 |
| 11 | Supervisor P10-F0 判定 | 完成（FAIL/NO VETO） |
| 12 | 停止等待用户审核 | 本轮停止点 |

## Phase 4.10-1F：Scheduler Fast-Path Feasibility Audit（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 冻结 router/workload/75%/ckpt8/partial_current_only/checker | 完成（文件 md5 本地/远程一致） |
| 2 | 精确分解 scheduler 单步 12.29ms p95 | 完成（复现 11.29–12.93ms；enumerate 83.5% / pack 8.9% / gate 0.1%） |
| 3 | 至少解释 90% wall time | 完成（p95 口径 92.4%、均值口径 99.2%；BFS 占 enumerate 均值 90.1%） |
| 4 | 静态/动态/可缓存/可增量/不可消除分类 | 完成 |
| 5 | 测量 first-commit preparation latency | 完成（p95 = 8,673.8µs，含 checker；slot 4、2 动作） |
| 6 | 实现 fast-path estimates | 完成（1,043.1/1,139.5/2,047.2µs；非理论下界） |
| 7 | 判断旧 replay 配置是否达到预注册目标 | 完成（否；不外推到新 concurrent architecture） |
| 8 | 实际目标预注册 <336µs | 完成（L_total < 336µs；L_sched ≤ 200µs、L_commit ≤ 135µs） |
| 9 | 输出允许的纯实现层优化计划 | 完成（DRAFT E1–E7，未实施） |
| 10 | 不实施大规模优化，Supervisor 判定 SF0-A/SF0-B | 完成（SF0-A = PASS；SF0-B = FAIL，均 NO VETO） |
| 11 | 仅 SF0-B PASS 才向用户申请实施 | 完成（SF0-B FAIL → 不申请） |
| 12 | 停止等待用户审核 | 本轮停止点 |

产物：`docs/phase4_10/SCHEDULER_LATENCY_BREAKDOWN.md`、`SCHEDULER_FAST_PATH_LOWER_BOUND.md`、
`SCHEDULER_EQUIVALENCE_CONSTRAINTS.md`、`P10_SF0_PROTOCOL.md`、`SCHEDULER_OPTIMIZATION_DRAFT_PLAN.md`、
`docs/agent_coordination/SUPERVISOR_REVIEW_P10_SF0_AB.md`；测量脚本与 JSON：
`outputs/phase4_10/p10_1f_audit/`。

## Phase 4.10-F：Final Closure（本轮）

| # | 任务 | 状态 |
|---|---|---|
| 1 | 独立 read-back Phase 4.10 全部 artifacts | 完成（11 项输出本地/远程 md5 一致；文档 vs JSON 0 差异） |
| 2 | 固化门链 P10-R0/S0/I1/P0/T0/F0/SF0 | 完成（含 P10-1 formal = CLOSED） |
| 3 | 如实记录 P10-F0-v1 FAIL 与 SF0-B FAIL | 完成 |
| 4 | 固化历史 replay candidate（419.8µs）与实现 estimates | 完成；R0 已修正口径 |
| 5 | 区分 L2-S / L2-R router 正确性 / L2-R E2E infeasibility | 完成 |
| 6 | 保留 pilot 负结果与 hotspot_random_walk 边界 | 完成（−32.8ms / −0.59ms / −3.6ms） |
| 7 | final report / negative result / evidence chain / reproducibility manifest / future scope | 完成 |
| 8 | Supervisor 独立判定 Phase 4.10-F | 完成（PASS/NO VETO） |
| 9 | 停止等待用户审核 | 本轮停止点 |

产物：`docs/phase4_10/PHASE4_10_FINAL_REPORT.md`、`PHASE4_10_NEGATIVE_RESULTS.md`、
`PHASE4_10_EVIDENCE_CHAIN.md`、`PHASE4_10_REPRODUCIBILITY_MANIFEST.md`、`PHASE4_10_FUTURE_SCOPE.md`、
`docs/agent_coordination/SUPERVISOR_REVIEW_PHASE4_10_F.md`。
