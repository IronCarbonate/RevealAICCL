# Phase R1：Real Concurrent Router–Scheduler Pipeline（R1-C0 / R1-T0）

更新日期：2026-08-10  
状态：**Supervisor accepted：R1-C0 = TECHNICAL FAIL；R1-T0 = COMPLETE；Phase R2-E0 = AUTHORIZED**

## 1. 结论先行

本轮实现了真实的 **router 与 scheduler overlap**，并在 checker 通过后执行真实
`NCCL all_reduce(async_op=True)`。但是 unchanged scheduler 没有在 final router
completion 前产生 action/commit，因此 NCCL 也没有在 router 尚未结束时提交。

所以：

- router || scheduler：**真正实现并实测成立**；
- scheduler → checker → real async NCCL：**真正实现并实测成立**；
- router || scheduler || NCCL 三者同时重叠：**未成立**；
- commit-before-final-router-completion：**不存在**；
- R1-C0：**FAIL**，不得称为完整三路 concurrent pipeline PASS。

## 2. 实现边界

- 独立 CUDA router stream；8 个 chunk 独立 forward；每 chunk 独立 CUDA events；
- timed path 无 per-chunk `torch.cuda.synchronize()` 或 `event.synchronize()`；
- host runtime 仅用非阻塞 `event.query()` 消费完成 chunk；
- scheduler 在独立 CPU process 上运行 unchanged
  `build_scheduling_view → enumerate_candidates → pack_candidate_batch`；
- append-only ready state 只加入已完成 chunk，future top-k 不传给 scheduler；
- 前 6/8 chunk 按 partial_shards 75% 逐步进入 ready state；最后 2/8 直到
  checkpoint 8 才一起进入 full state；
- unchanged `bind_action` + deterministic `commit_proposal`，异常 fail closed；
- checker legal 后才调用 real NCCL `dist.all_reduce(..., async_op=True)`；
- API-call、submit-return 和最终 `work.wait()` 分开计时；
- 控制路径全部使用 `time.monotonic_ns()`；CUDA event 只计算 CUDA duration，
  未与 host timestamp 相减。

冻结 workload 为 8×4096×D2048 reference-router forward。为保持历史 48-token
scheduler world，每个 4096-token chunk 的前 6 个实际 top-k 输出作为 control demand，
共 48 token；它们不是预计算或 replay，但本结果也不代表全 32,768 token 的生产
MoE dispatch scheduling。

## 3. Gate R1-C0

| 要求 | 结果 |
|---|---|
| no per-chunk global/event sync | PASS |
| ≥3 progressive readiness events | PASS（8/8） |
| scheduler 在 final router completion 前运行 | PASS |
| legal action 在 final completion 前产生 | **FAIL** |
| legal checker commit 在 final completion 前完成 | **FAIL** |
| real NCCL submit 在 final completion 前发生 | **FAIL** |
| no future top-k access | PASS |
| hidden suffix perturbation 不改变当前 action | PASS（两 rank） |
| suffix perturbation 实际改变 top-k | PASS（2/2 suffix chunks） |
| token integrity | PASS |
| legality | PASS（280/280，100%） |
| partial_shards 75% / checkpoint8 | PASS |
| artifact read-back | PASS |

R1-C0 的必要条件并未全部满足，因此 technical verdict = **FAIL**。

## 4. Gate R1-T0 timing

40 个主 trial（20 trials × 2 ranks），单位均为 host-monotonic µs：

| 指标 | p50 | p95 | p99 |
|---|---:|---:|---:|
| **W_host** | **655.551** | **847.062** | **895.198** |
| ready→scheduler | 10,631.999 | 33,707.881 | 46,154.854 |
| ready→action | 15,578.771 | 42,964.009 | 53,008.673 |
| ready→checker / commit | **15,651.339** | **43,050.968** | **53,092.082** |
| ready→NCCL submit-return | 42,628.778 | 56,830.956 | 63,634.671 |
| per-chunk remaining actionable window | 261.130 | 686.098 | 820.008 |
| NCCL API-call→submit-return | 66.602 | 114.117 | 256.652 |
| final NCCL wait / work | 3.721 | 8.108 | 13.592 |

`ready→commit p95 = 43.051ms`，约为 `W_host p50 = 0.656ms` 的 65.7 倍。
最短 observed ready→commit 仍为 2.395ms，晚于所有主 trial 的 W_host 上界
0.914ms，因此没有 commit-before-final。

## 5. Router/runtime/NCCL interference

| 指标 | baseline p50 | concurrent-runtime p50 | 差值/比例 |
|---|---:|---:|---:|
| router total host | 2,152.512µs | 2,746.406µs | +593.894µs / 1.276× |
| router CUDA duration sum | 1,411.616µs | 1,425.344µs | +13.728µs / 1.0097× |

本轮 **0 个 NCCL submit 与 router execution 重叠**。因此上述 host/CUDA 差值只能
称为 event polling、IPC 和 scheduler worker 引入的 runtime interference；不能归因
为 NCCL interference，也不能估计 router↔NCCL GPU contention。

## 6. Counterfactual / no-leak

在两个 rank 上分别对未揭示 chunks 6–7 做真实输入扰动：

- prefix chunks 0–5 router assignment hashes 完全不变；
- prefix scheduler actions 完全不变；
- suffix 2/2 chunk assignment hashes 改变；
- scheduler payload 只从 `event.query()==True` 的 chunk 构造。

因此当前 action 未读取 future top-k。

## 7. 分类与停止点

`W_host p50 = 655.551µs`，属于 **B：数百微秒级**。

按预注册规则，下一候选是申请 **Compiled Event-Driven AICCL**。本轮没有实现
StaticPlanCompiler、FastBinder 或 IncrementalChecker，也没有做 scheduler 优化。

Supervisor 已接受本结论并授权 Phase R2-E0；历史 replay-based P10-1 仍 CLOSED。

## 8. Artifacts

- `scripts/run_r1_concurrent_pipeline.py`
- `outputs/phase_r1/concurrent_pipeline/r1_concurrent_pipeline_results.json`
- `outputs/phase_r1/concurrent_pipeline/r1_readback.json`
- `outputs/phase_r1/concurrent_pipeline/r1_artifact_manifest.json`
