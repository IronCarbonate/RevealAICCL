# Phase 4.11：Claims-to-Evidence Matrix（论文主张 ↔ 证据 ↔ artifact ↔ 统计）

更新日期：2026-08-10（Phase R0 修正）
规则：每条论文 claim 必须同时满足 (a) 有来源文档；(b) 有 artifact 文件；(c) 有统计结果。不满足的表述一律 fail closed。

## 1. 四层结论框架（严格区分）

| 层 | 含义 | 结论状态 |
|---|---|---|
| L1 | 单机（RTX 2080 Ti）高保真、合成 shim、单 rank | 派生汇总支持 D1 = PASS（ΔE2E +10.95ms）；原始 raw jobs 已丢失，provenance 降级 |
| L2-S | 单机 2×V100、真实 2-rank NCCL、合成 shim/合成 GEMM | D1 = PASS（ΔE2E +6.46ms） |
| L2-R | reference router substrate（真实 CUDA top-k/shard-ready，非生产 MoE） | 历史 17/17；R0 强化 19/19；旧 replay 路径不可准入，真实 concurrency 未测 |
| L3 | 多节点 RDMA/NVSHMEM/DeepEP | **未验证**（V100 sm_70 不支持 DeepEP） |

## 2. Claims 矩阵（支持型）

| ID | Claim（论文可用表述） | 层 | 判定 | 来源文档 | Artifact | 统计结果 |
|---|---|---|---|---|---|---|
| C1 | 历史流量预测（MLP）无稳定收益，previous-value 更稳 | proxy | 支持（负结果） | `docs/uncertainty_aiccl/H1_PREDICTABILITY_RESULTS.md` | `outputs/h1_predictability/summary.json` | total paired Δ −0.0790，CI [−0.1133,−0.0478]；1/5 family 正；LOFO 0/5 |
| C2 | 多场景 robust prefix 相对被动基线无 E2E 收益（overhead 主导） | proxy | 支持（负结果） | `docs/uncertainty_aiccl/H2_EARLY_PLANNING_RESULTS.md` | `phase4_formal_artifacts/summary.json`（正式只读副本） | E2E Δ −938.58ms（CI [−992.59,−896.65]）；robust 1042ms vs 基线 ~104–116ms |
| C3 | robust prefix 相对 partial_current_only 无独立调度价值 | proxy | 支持（负结果） | `docs/phase4_5/H2B_ALGORITHMIC_VALUE.md` | `outputs/phase4_5/h2b_analysis/h2b_analysis.json` | scheduling-only +0.11 slots（CI [0.087,0.140]）；动作重合 98.5%；无工作区间 |
| C4 | 信息揭示节奏是 completion regret 主导瓶颈 | proxy | 支持 | `docs/phase4_6/ROUTE_A_REVEAL_RESULTS.md` | `outputs/phase4_6/route_a_reveal/route_a_results.json` | S3 11.80 vs S0 20.95 vs fullinfo 10.80；S1 +6.03/S2 +7.56/S3 +9.16（CI>0）；S4 −15.22 |
| C5 | 可实现早期揭示有 E2E 价值（成本计入） | proxy | 支持 | `docs/phase4_7/H5_RESULTS.md` | `outputs/phase4_7/h5_realizable_reveal/h5_test.json` | A4 +9.22ms（CI [+8.26,+10.13]）、A2 +6.06ms；15/15 seq、5/5 family、3/3 seed |
| C6 | 全局聚合信息无调度价值 | proxy | 支持（负结果） | `docs/phase4_7/H5_RESULTS.md` | `outputs/phase4_7/h5_realizable_reveal/h5_test.json` | A5 ΔJ −0.13ms（0/15 seq 正） |
| C7 | 固定预算下 partial_shards 是唯一稳定优于 random 的选择器 | proxy | 支持 | `docs/phase4_7/H6_RESULTS.md` | `outputs/phase4_7/h6_selective_reveal/h6_test.json` | +0.604/+0.807/+0.573ms（B25/B50/B75，CI>0，5/5 family、3/3 seed） |
| C8 | 自适应 reveal 控制无价值 | proxy | 支持（负结果） | `docs/phase4_7/H7_RESULTS.md` | `outputs/phase4_7/h7_adaptive_reveal/h7_test.json` | controller ≡ B75（Δ=0.0000ms）；oracle 上界 0.0014ms |
| C9 | 调度侧候选排序/风险门控/静态预计算无 completion 收益 | proxy | 支持（负结果） | `docs/phase4_6/W1W2_EVALUATION.md`、`W3_RISK_GATE.md` | `outputs/phase4_6/w2_scheduler/w2_diagnostic_full.json`、`w3_risk_gate/w3_risk_gate.json` | lookahead +0.030（CI 跨 0）；W3 规则 100/100 选 act，gate 空转；W1 302× 查询加速但 completion 不变 |
| C10 | L1 单机真实时间下固定 profile 的派生汇总为正；不得声称 raw-level provenance 完整 | L1 | 降级支持 | `docs/phase4_8/PHASE4_8_5_FORMAL_RESULTS.md`、`outputs/phase_r0/evidence_repair/l1_provenance_status.json` | `final_summary.json`、`job_sequence_results.json`；原始 `l1_* raw jobs` 未找到 | 派生 ΔE2E +10,953µs；L1 raw artifact = LOST，禁止重造冒充历史 raw |
| C11 | 单机多 GPU 真实 NCCL 下固定 profile 仍有部署收益 | L2-S | 支持 | `docs/phase4_8/PHASE4_8_6_L2_RESULTS.md`、`docs/phase4_9/L2_FINAL_REPORT.md` | `outputs/phase4_8/deployment_validation/l2_final_summary.json`、`l2_collective_results.json`、`read_back_report_l2.json` | ΔE2E +6,458µs（CI [+3,409,+9,385]）；completion +6.43；吞吐 +13.7%；真实 NCCL allreduce 62–87µs / allgather 122–136µs |
| C12 | reference router（L2-R）实现正确且确定 | L2-R | 强化支持 | `docs/phase4_10/P10_I1_RESULTS.md`、`docs/phase_r0/EVIDENCE_REPAIR_REPORT.md` | 历史 `p10_i1_results.json`、R0 `p10_i1_strengthened_results.json` | 历史 17/17；R0 19/19（actual 75% view、独立 traffic oracle、hidden perturbation/no-leak、conservation、ties） |
| C13 | D0/D1 在 reference router 路径共享同一 token/权重/top-k 流 | L2-R | 支持 | `docs/phase4_10/P10_I1_RESULTS.md`、`P10_1C_PILOT_RESULTS.md` | `p10_i1_results.json`、`p10_1c_pilot_results.json` | T3/T3b PASS；pilot same_stream_all/same_traffic_all = true（20/20） |
| C14 | L2-R pilot 机制可准入但 E2E 收益未确立 | L2-R | 支持 | `docs/phase4_10/P10_1C_PILOT_RESULTS.md` | `outputs/phase4_10/p10_1c_pilot/p10_1c_pilot_results.json` | completion Δ +1.95；E2E Δ −19,688.8µs（setup 主导）；legality 100%、timeout 0 |
| C15 | 三臂测量下 C1 completion 收益稳健但 E2E 稳态≈0 | L2-R | 支持 | `docs/phase4_10/P10_1D_TIMING_RESULTS.md` | `outputs/phase4_10/p10_1d_timing/p10_1d_timing_results.json` | C1 22.9 vs 28.1（+5.2 slots）；稳态 E2E C1≈B0（151.4 vs 150.0ms）；overhead ~11ms |
| C16 | replay/quantized candidate window（419.8µs）小于 scheduler 单步 p95（~12.3ms），旧 formal 不可准入；不构成 concurrent window 测量 | L2-R | 支持 | `docs/phase4_10/P10_1E_FORMAL_ADMISSIBILITY.md` | `outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.json` | P1/P2/P3 PASS、P4 FAIL；candidate 419.84µs < p95 12,290.03µs |
| C17 | scheduler 单步延迟主要由静态可缓存 BFS 距离重算构成 | L2-R | 支持 | `docs/phase4_10/SCHEDULER_LATENCY_BREAKDOWN.md` | `outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.json` | p95 口径分解合计 92.4%（enumerate 83.5%+pack 8.9%+gate 0.1%）；BFS 占 enumerate 均值 90.1% |
| C18 | 冻结 Python 实现的 fast-path estimates 为 step-only 1,043.1µs、含 bind/checker 1,139.5µs、含 digest 2,047.2µs；不是严格理论下界 | L2-R | 支持 | `docs/phase4_10/SCHEDULER_FAST_PATH_LOWER_BOUND.md` | `p10_1f_scheduler_breakdown.json` | 测量锚定估算；enumerate-min 48/48 候选恒等，但未证明跨实现下界 |
| C19 | hotspot_random_walk family 在 L1/L2/pilot/1D 均为负或近零 | 全层 | 支持（边界） | `PHASE4_8_5_FORMAL_RESULTS.md`、`PHASE4_8_6_L2_RESULTS.md`、`P10_1C_PILOT_RESULTS.md`、`P10_1D_TIMING_RESULTS.md` | 对应 JSON | L1 −1.1ms；L2 −1.7ms；pilot −32,817µs；1D reveal −590µs / deployment −3,553µs |
| C20 | 负结果集合（H1/H2/H2b/H7/W2/W3/A5/pilot/P10-F0-v1/SF0-B）完整保留 | 全层 | 支持 | `docs/phase4_10/PHASE4_10_NEGATIVE_RESULTS.md` | 各阶段 artifacts | 见各条统计 |

## 3. Completeness 检查（claim → artifact → statistic）

- C1–C20 每条均映射到 ≥1 个文档、≥1 个 artifact 文件、≥1 个统计量；
- 所有统计量均为已冻结协议下的正式/预注册结果（paired bootstrap CI、sequence-level 配对、真实计时、profiling OFF 主指标）；
- 跨阶段 corpus 零重合已冻结（H1/H2: 642/742/842；Route A: 1042/1142/1242；H5–H7: 2042/2142/2242；L1/L2 formal: 3042/3142/3242；P10-1: 4042 dev/val；formal corpus 5042/5142/5242 从未生成）；
- provenance 缺失项：L1 原始 raw jobs 已丢失；仅保留派生汇总，已在 C10 明示降级。

## 4. Fail-closed 清单（禁止进入论文的表述）

| 表述 | 原因 |
|---|---|
| “生产 MoE router” / “L2-R 是生产路径” | reference substrate ≠ 生产 MoE；P10-R0 已确认仓库无生产 router |
| “L3/DeepEP/RDMA 已验证” | 未运行；V100 sm_70 不支持 DeepEP；无多节点验证 |
| “L2-R production-path E2E 有收益” | P10-F0-v1 FAIL；E2E 稳态≈0（1D）；pilot E2E 为负 |
| “1.045ms 是 strict/theoretical lower bound” | 证据只支持 step-only 1,043.1µs；含 bind/checker 为 1,139.5µs；均为实现 estimate |
| “419.8µs 是真实 concurrent actionable window” | readiness 先量化/replay；未运行 router↔scheduler concurrent pipeline |
| “P10-1 formal 通过或其 CLOSED 禁止新并发架构” | 历史 replay-based formal = CLOSED，未运行；新 concurrent/event-driven architecture 可另立 Gate |
| “自适应/robust 预测有价值” | H1/H2/H7 全 FAIL；oracle 上界 0.0014ms |
| “调度排序/门控有 completion 收益” | W2/W3 无价值；H2b 动作重合 98.5% |
| “L1/L2 结论可外推至多节点或生产 SLA” | 结论限定 L1/L2-S；L3 未验证 |
| “completion 收益等于 E2E 收益” | completion +6.43 slots 与 E2E Δ 分别报告；1D 稳态 E2E≈0 如实 |
| “pilot E2E 为负不说明任何问题” | 负结果如实保留并归因 setup 主导；不得选择性删除 |

## 5. 结论

Claims 矩阵已纳入 R0 provenance 降级与口径修正；L1 raw 丢失不再被表述为完整 provenance。
矩阵供 Supervisor 独立审核（见 `SUPERVISOR_REVIEW_PHASE4_11_CLAIMS.md`）。
