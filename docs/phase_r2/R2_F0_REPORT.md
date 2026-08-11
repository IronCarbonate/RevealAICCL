# Phase R2-F0：Integrated Fast Ready-to-Commit Feasibility

更新日期：2026-08-10  
状态：**TECHNICAL PASS / PENDING SUPERVISOR**  
边界：本报告不是 R2-O0，不作正式三路 overlap 结论。

## 1. Gate 结论

在 2× Tesla V100-SXM2-32GB、2-rank real NCCL、50 trials/rank 上，首次串联：

`CUDA shard-ready → native EventBridge → IncrementalState → FastBinder → DynamicGuard → descriptor binding → real NCCL all_reduce(async_op=True)`。

共 100 trials、700 个 eligible scheduler events。结果：

- **F0-A Semantic/Safety：PASS**；
- **F0-B hard gate：PASS**，ready→NCCL-submit-return p95 = **578.891µs < 655.551µs**；
- **<300µs stretch：FAIL**；
- R2-F0 总体：**TECHNICAL PASS / PENDING SUPERVISOR**。

## 2. Integrated cumulative latency

所有 timestamp 均为 host monotonic time；CUDA 与 CPU timestamp 未直接混算。

| cumulative metric | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| ready→state | 700 | 58.420 | 70.235 | 107.884 | 1019.273 |
| ready→action | 700 | 167.902 | 237.218 | 360.987 | 1273.196 |
| ready→guard | 700 | 271.752 | 386.068 | 585.441 | 1513.609 |
| ready→NCCL-call | 700 | 336.853 | 470.499 | 727.229 | 1674.622 |
| ready→NCCL-submit-return | 700 | **426.230** | **578.891** | **959.454** | **2267.696** |

`t_action` 是 compiled ordered candidate selection + proposal binding 完成；
`t_guard_done` 是 DynamicGuard fail-closed apply 完成；`t_nccl_call` 位于预分配
descriptor host→device enqueue 之后；`t_nccl_submit_return` 是真实
`dist.all_reduce(..., async_op=True)` API 返回。

## 3. Incremental stage breakdown

以下由 700 行 raw host timestamp 独立重算，不是 cumulative 差值的近似：

| incremental stage | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|
| EventBridge-ready→state done | 58.420 | 70.235 | 107.884 | 1019.273 |
| state done→action | 108.986 | 177.762 | 274.913 | 457.368 |
| action→guard done | 109.845 | 146.678 | 215.883 | 656.250 |
| guard done→NCCL call | 63.858 | 92.251 | 122.190 | 319.378 |
| NCCL API call→submit return | 85.743 | 131.257 | 222.952 | 707.027 |

两 rank 的 ready→submit p95 分别为 490.695µs 与 612.802µs；aggregate
p95 使用全部 700 个 eligible event，而不是挑选较快 rank。

## 4. F0-A semantic/safety

| requirement | result |
|---|---:|
| runtime BFS | **0** |
| fast-path full rebuild | **0** |
| unrevealed execution | **0** |
| old/new ordered candidate comparisons | 700，**0 divergence** |
| old/new action comparisons | 700，**0 divergence** |
| old/new checker comparisons | 700，**0 divergence** |
| old/new holder-state divergence | **0** |
| legality | **700/700 = 100%** |
| token integrity | **100/100 trials = 100%** |
| real async NCCL submit | **700/700 eligible events** |

旧 `build_scheduling_view → enumerate_candidates → pack_candidate_batch →
bind_action → commit_proposal` 仅在每个 trial 的 final NCCL waits 之后回放，
不进入任何 ready→submit timestamp。计时路径不使用 ProcessPool、
`multiprocessing.Queue`、pickle、JSON、sleep polling、runtime BFS、旧 Python
candidate enumeration 或 full-state rebuild。

75%/checkpoint8 保持不变：chunk 0–5 逐 chunk reveal；chunk 6 虽已完成并进入
pending-ready，但不可见；直到 chunk 7 完成才按 checkpoint8 一次消费 chunk 6/7。
FastBinder 与 DynamicGuard 只遍历 `ordinal < revealed_count`，hidden top-k 不可执行。

## 5. submit-before-final diagnostic

使用保守的 `t_nccl_submit_return` 计算：

`margin_i = t_final_router_completion - t_nccl_submit_return_i`。

- eligible shards：700；
- `margin_i > 0`：**600/700（85.714%）**；
- positive margin p50/p95/p99/max：
  **1756.050 / 2818.363 / 3608.239 / 5276.617µs**。

每个 trial 的前 6 个 progressive event 均观察到 submit-return-before-final；
checkpoint8 event 本身以 final router completion 为 ready 时刻，因此其 100 个样本
不可能在 final completion 前提交。

这证明 F0 样本中已经出现真实 submit-before-final，但只作为 diagnostic。
**未运行 R2-O0，也不在此宣称正式 router || scheduler || NCCL overlap Gate 通过。**

## 6. 实现与证据

- 独立 CUDA router stream，每 chunk 独立 reference-router forward；
- 每 chunk D2H top-k copy 后记录独立 CUDA completion event；
- native pinned busy EventBridge 使用 `cudaEventQuery`，无 sleep；
- timed region 无 per-chunk `torch.cuda.synchronize()`、`event.synchronize()`；
- 单进程 per-rank，router producer thread 与 native EventBridge/control thread 并行；
- preallocated CUDA events、pinned top-k buffers、state arrays、descriptor buffers；
- 独立 communication stream，真实 NCCL `all_reduce(async_op=True)`；
- API call、submit return 与 trial-final work waits 分离记录。

Canonical artifacts：

- `outputs/phase_r2/f0_integrated_ready_commit/r2_f0_results.json`
- `outputs/phase_r2/f0_integrated_ready_commit/r2_f0_readback.json`
- `outputs/phase_r2/f0_integrated_ready_commit/r2_f0_independent_readback.json`
- `scripts/run_r2_f0_integrated.py`
- `extensions/r2_event_bridge/integrated_event_bridge.cpp`

未运行 formal E2E、real AlltoAllv、expert packing/GEMM/combine、DeepEP、
predictor/robust/adaptive；未修改 workload/chunk、75% 或 checkpoint8；未进入 R2-O0。

## 7. 建议

建议 Supervisor 审核 R2-F0。若裁决 PASS / NO VETO，再申请正式进入 R2-O0，
以 per-shard timestamps 对 `t_NCCL_submit_i < t_final_router_completion` 做正式
router || scheduler || NCCL overlap Gate；在此之前停止。
