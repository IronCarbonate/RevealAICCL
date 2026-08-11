# Phase 4.10 证据链（EVIDENCE_CHAIN）

更新日期：2026-08-06

> **R0 evidence correction（2026-08-10）**：P10-1D 是 CUDA router timing
> 后的 quantized readiness replay，不是 concurrent pipeline。419.8µs 是 replay
> candidate window；fast-path estimates 为 1,043.1/1,139.5/2,047.2µs，不是
> 理论下界。历史 P10-1 formal CLOSED 不禁止另立 concurrent/event-driven architecture。

## 1. 门链

```text
L2-S（合成路径部署收益）= PASS（Phase 4.9-F；ΔE2E +6.46ms，CI [+3.41,+9.38]ms）
  └─► P10-R0 = CONDITIONAL PASS（无真实 MoE router/GEMM；NCCL 真实可用；DeepEP 不可行）
        └─► P10-S0 = PASS（substrate = 最小 PyTorch reference，L2-R）
              └─► P10-I1 = PASS（历史 17/17；R0 强化 19/19）
                    └─► P10-P0 = CONDITIONAL PASS（pilot：completion +1.95；E2E −19.7ms setup 主导）
                          └─► P10-T0 = PASS（三臂测量稳定：C1 completion +5.2、E2E 稳态≈0）
                                └─► P10-F0-v1 = FAIL（P4：replay candidate 419.8µs < scheduler p95 12,290µs）
                                      └─► P10-SF0-A = PASS（分解 ≥90%；first-commit 8.67ms；实现 estimates）
                                            └─► P10-SF0-B = FAIL（step-only estimate 1,043.1µs > 336µs）
                                                  └─► 历史 replay-based P10-1 formal = CLOSED
```

## 2. 各环节关键文档与数字

| Gate | 文档 | 关键数字 / 证据 |
|---|---|---|
| P10-R0 | `docs/phase4_10/PHASE4_10_0_PRODUCTION_PATH_AUDIT.md`、`SUPERVISOR_REVIEW_PHASE4_10_R0.md` | 组件四态分类；DeepEP sm_70 不支持；NCCL 真实 |
| P10-S0 | `docs/phase4_10/EXECUTABLE_MOE_SUBSTRATE.md`、`P10_S0_PROTOCOL.md` | (a)/(b) 不存在，(c) 选定；L2-R 命名 |
| P10-I1 | `docs/phase4_10/P10_I1_RESULTS.md`、R0 `p10_i1_strengthened_results.json` | 历史 17/17；R0 19/19（actual view、independent oracle、counterfactual no-leak、conservation、ties） |
| P10-P0 | `docs/phase4_10/P10_1C_PILOT_RESULTS.md`、`outputs/.../p10_1c_pilot_results.json` | Δcompletion +1.95；ΔE2E −19,688.8µs；hotspot −32,817.2µs；legality 100% |
| P10-T0 | `docs/phase4_10/P10_1D_TIMING_RESULTS.md`、`outputs/.../p10_1d_timing_results.json` | C1 22.9 vs C0/B0 28.1；router timing 后 readiness replay；非 concurrency |
| P10-F0 | `docs/phase4_10/P10_1E_FORMAL_ADMISSIBILITY.md`、`SUPERVISOR_REVIEW_P10_F0.md`、`outputs/.../p10_1e_readiness_test.json` | P1/P2/P3 PASS、P4 FAIL；replay candidate 419.84µs；p95 12,290.03µs |
| P10-SF0-A/B | `docs/phase4_10/SCHEDULER_LATENCY_BREAKDOWN.md`、`SCHEDULER_FAST_PATH_LOWER_BOUND.md`、`SUPERVISOR_REVIEW_P10_SF0_AB.md`、`outputs/.../p10_1f_scheduler_breakdown.json` | fast-path estimates 1,043.1/1,139.5/2,047.2µs；非理论下界 |
| P10-1 formal | `docs/phase4_10/P10_1_FORMAL_PROTOCOL.md` | HOLD → CLOSED；P4 判定式预注册 `L_sched + L_commit < 336µs` |

## 3. 一致性说明

- 11 项 Phase 4.10 输出 artifacts 本地/远程 md5 逐项一致（独立 read-back 0 差异）；
- 所有 JSON 与文档数字逐项核对一致（仅 1E 文档“~30 倍”口径按窗口比校正，见 FINAL_REPORT §3）；
- P10-1D/1E/1F 使用同一冻结 workload 派生（seed 4042）；formal corpus（5042/5142/5242）从未生成/运行；
- P10-1D router timing 真实，但 scheduler 消费是事后 replay；现有证据不含真实 concurrent wall-clock correspondence。
