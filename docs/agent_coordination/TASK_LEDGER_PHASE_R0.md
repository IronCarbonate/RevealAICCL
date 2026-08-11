# Task Ledger — Phase R0 Evidence Repair

更新日期：2026-08-10

| # | 任务 | 状态 |
|---:|---|---|
| 1 | 搜索本地 L1 raw/备份/归档 | 完成；未找到 |
| 2 | 搜索服务器 L1 raw/备份/归档 | 完成；仅找到与本地一致的 L2 raw |
| 3 | 标记 L1 raw lost，禁止重造 | 完成 |
| 4 | 保存旧 L2 environment manifest | 完成；`pre_r0_SUPERSEDED` |
| 5 | 从 V100/NCCL evidence 重建 L2 manifest | 完成 |
| 6 | actual 75% partial-view test | 完成；PASS |
| 7 | independent token→traffic oracle | 完成；PASS |
| 8 | unrevealed counterfactual + no-leak | 完成；49/64 suffix 改变，prefix/traffic 不变 |
| 9 | token loss/duplication + deterministic ties | 完成；PASS |
| 10 | P10-I1 strengthened CUDA run | 完成；19/19 PASS |
| 11 | 修正文档 replay/concurrency/window/lower-bound 口径 | 完成 |
| 12 | 生成 R0 hashes/read-back | 完成；JSON/source/hash read-back PASS |
| 13 | Supervisor R0 Gate | **PASS / NO VETO** |
| 14 | Phase R1 | **AUTHORIZED** |

禁止项核对：未改 scheduler 算法；未改 75%/checkpoint8/checker；未运行 formal
E2E；未实现 DeepEP/GEMM/combine；未恢复 predictor/robust/adaptive；未人工延长
router；未挑 workload。
