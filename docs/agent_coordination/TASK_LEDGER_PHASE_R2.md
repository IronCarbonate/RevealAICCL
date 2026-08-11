# Task Ledger — Phase R2

更新日期：2026-08-10

| # | 任务 | 状态 |
|---:|---|---|
| 1 | 冻结 R1 Supervisor 裁决与 R2-E0 边界 | 完成 |
| 2 | R1 ready→scheduler timestamp 分解 | 完成 |
| 3 | idle IPC/wakeup 与 scheduler component 诊断 | 完成 |
| 4 | single-process native EventBridge | 完成 |
| 5 | pinned busy polling + preallocated ring/bitmap | 完成 |
| 6 | timed path 禁止 JSON/pickle/queue/sleep/allocation | 完成 |
| 7 | 双 V100、500 trials/rank、8,000 event samples | 完成 |
| 8 | event→host-ready p95 <100µs | PASS；worst-rank 5.019µs |
| 9 | stretch p95 <50µs | PASS；worst-rank 5.019µs |
| 10 | artifacts independent read-back | PASS |
| 11 | StaticPlanCompiler/IncrementalState/FastBinder/StaticProof+DynamicGuard | R2-C0 已实现并通过 Supervisor |
| 12 | old/new checker equivalence protocol | R2-C0 已执行；E1/E2/E3 exact equivalence |
| 13 | R2-E0 | **Supervisor PASS / NO VETO；实现冻结为 runtime substrate** |
| 14 | StaticPlanCompiler + StaticProof | R2-C0 真正实现；runtime BFS=0 |
| 15 | IncrementalState | R2-C0 真正实现；trajectory fast path full rebuild=0 |
| 16 | FastBinder + DynamicGuard | R2-C0 真正实现；旧 scheduler/checker 保持 oracle |
| 17 | E1/E2/E3 | 360/212/36 tests，全部 0 mismatch |
| 18 | R2-C0 | **Supervisor PASS / NO VETO** |
| 19 | integrated CUDA-ready→compiled commit→real NCCL async submit | R2-F0 已完成；700 events |
| 20 | F0-A semantic/safety | **PASS**；700 candidate/action/checker comparisons，0 divergence |
| 21 | F0-B hard p95 <655.551µs | **PASS**；578.891µs |
| 22 | F0 stretch p95 <300µs | **FAIL**；578.891µs |
| 23 | submit-return-before-final diagnostic | 600/700 eligible shards；非 R2-O0 结论 |
| 24 | R2-F0 | **Supervisor PASS / NO VETO** |
| 25 | O0 canonical A/B/C CUPTI runs | 3 seeds × 2 ranks × 20 trials/mode；完成 |
| 26 | CUPTI trace reconstruction | 2,880 router chunks + 840 NCCL kernels；0 association failure |
| 27 | host submit-before-final | 720/840 = 85.714%；复现，仅 diagnostic |
| 28 | early NCCL GPU-start-before-final ≥75% | **FAIL**；447/720 = 62.083% |
| 29 | actual device kernel overlap | 324/840 = 38.571%；存在但不稳定 |
| 30 | three-seed stability | **FAIL**；early start 71.67%/64.17%/50.42% |
| 31 | NCCL-induced router interference | **显著**；median +808.299µs，95% CI [783.418,839.482]µs |
| 32 | R2-O0 | **TECHNICAL FAIL / PENDING SUPERVISOR** |
| 33 | R3 real variable-size AlltoAllv | 不建议；未获授权、未启动 |
| 34 | R2-O0 Supervisor review | **FAIL / NO VETO**；历史结论冻结，R3 未授权 |
| 35 | O1A strict A/B/C/D control | 3 seeds × 2 ranks × 20 trials/mode；C/D 120/120 controls 全等 |
| 36 | O1A CUPTI reconstruction | 3,840 router chunks + 1,680 NCCL kernels；0 association failure |
| 37 | distributed paired T_D−T_C | p50 3,449.724µs；p95 167,634.769µs；median 95% CI [2,718.223,5,165.632]µs |
| 38 | three independent runs | median gain 3/3 positive；4044 delayed-rendezvous tail 原样保留 |
| 39 | O1A router contention | C−B median +775.659µs；95% CI [743.516,815.579]µs；约 +19.14% |
| 40 | O1A launch/rendezvous | C submit→GPU-start p50/p95 105.770/22,242.998µs；rank skew p50/p95 715.747/14,519.292µs |
| 41 | O1A diagnosis | **Supervisor PASS / NO VETO；C: launch/rendezvous + resource contention both** |
| 42 | O1B authorization | Supervisor 授权固定 T0/T1/T2/T3；R3 继续未授权 |
| 43 | O1B strict paired controls | 3 seeds × 2 ranks × 20 trials；Ck/Dk 480/480，cross-transport 480/480 PASS |
| 44 | O1B semantic/safety | 1,080 rank-trials、7,560/7,560 legal；BFS/full rebuild/unrevealed/divergence 全 0；token 1,080/1,080 |
| 45 | O1B trace reconstruction | 8,640 router chunks、6,720 action groups、10,080 allreduce primitive kernels、1,680 P2P kernels；0 association failure |
| 46 | T0 primary | **PASS**；Delta median 845.335µs，95% CI [661.897,1,687.218]µs，3/3 seeds positive |
| 47 | T1 primary | **FAIL**；总体 CI 正但 seed4044 median −3,675.363µs，非 3/3 positive |
| 48 | T2 primary | **FAIL**；CI lower −629.628µs，seed4044 median −91,269.019µs |
| 49 | T3 primary | **PASS**；Delta median 2,095.778µs，95% CI [1,251.219,4,390.917]µs，3/3 seeds positive |
| 50 | intervention selection | 无 intervention 同时满足 primary PASS 与预注册 ≥20% improvement；建议保留 T0 with limitations |
| 51 | R2-O1B | **Supervisor PASS / NO VETO**；selection=RETAIN T0 WITH LIMITATIONS；T3 不入主线 |

禁止项核对：未运行 formal E2E；未实现 real AlltoAllv、expert GEMM/combine、
DeepEP；未改 75%/checkpoint8 或 partial_current_only；未跳过 deterministic checker；
未优化 scheduler；未恢复 predictor/robust/adaptive；未改变 workload/chunk；未创建 Subagent。
