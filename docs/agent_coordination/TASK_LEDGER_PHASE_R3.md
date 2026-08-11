# Task Ledger — Phase R3

更新日期：2026-08-10

| # | 任务 | 状态 |
|---:|---|---|
| 1 | 冻结 O1B Supervisor PASS 与 T0-with-limitations selection | 完成 |
| 2 | R3-A0 A2Av-T0 reference substrate | 完成；真实 NCCL uneven-split all_to_all_single |
| 3 | Router assignment → destination lists/sendcounts/offsets | 完成；无 externally supplied counts |
| 4 | Deterministic contiguous reference packing | 完成；8×int64 verifiable record |
| 5 | Incremental progressive descriptors | 完成；chunks 0–5 delta，checkpoint8=[6,7] |
| 6 | R3-C0 C/D semantic equivalence | 14/14 PASS |
| 7 | Correctness | 114,688 tokens；lost/duplicate/wrong-destination/corruption 全 0 |
| 8 | Scheduler semantics | 196/196 legal；BFS/full rebuild/divergence 全 0 |
| 9 | Coverage cases | balanced/skewed/all-to-one/zero/empty/single/multiple 全 PASS |
| 10 | Variable traffic | pair sizes 0–1,024；95 distinct；34 zero-sized pairs |
| 11 | Independent read-back | PASS；196 split-transpose + 196 descriptor checks |
| 12 | R3-A0/C0 | **COMPLETE / PENDING SUPERVISOR** |
| 13 | R3-P0 / formal E2E | 未授权、未运行；立即停止 |

禁止项核对：未实现 expert GEMM、return combine、production MoE、DeepEP/PCCL production；
未增加 transport；未改 scheduler/75%/checkpoint8；未运行 R3-P0/formal E2E；未创建 Subagent。
