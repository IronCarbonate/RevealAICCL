# P10-1D 结果：Timing Semantics and Measurement Stabilization

> **Phase R0 correction（2026-08-10）**：本阶段使用真实逐 chunk CUDA
> router timing，但 router 完成后才把 readiness 量化/replay 给 scheduler。
> 因此它不是 router↔scheduler progressive concurrent pipeline，也没有直接
> 测量 concurrent actionable window。

更新日期：2026-08-05
判定：**P10-T0 = PASS（测量方法学稳定）/ NO VETO；科学上：completion 收益稳健、E2E 收益未在本规模确立**

## 1. 执行

- 2× V100 / torch 2.8.0+cu128 / NCCL 2.27.3；token corpus seed 4042（20 序列，dev 12/val 8；未用 3042）；
- 三臂：RR-B0（batched router + full@16）、RR-C0（chunked router + full@16）、RR-C1（相同 chunked router + partial_shards 75%@8，chunk 门控）；
- 逐 chunk router 独立执行（**无完整 top-k 后切片**）；shard-ready = 每 chunk 的 CUDA 完成事件；
- C0/C1 共享同一 chunked router 输出（token/arrival/top-k/traffic）；
- kernel warmup + 代表性规模（2048 token）timing probe 推导 arrival slots（SLOT_US=稳态中位 chunk 时间）；主指标 profiling OFF；overhead 用交替顺序测量。

## 2. 结果（20 jobs/臂）

| 臂 | completion | E2E cold (ms) | E2E steady (ms) | E2E amortized (ms) | router (µs) | overhead (ms) | legality | timeout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 28.1 | 378.7 | 150.0 | 159.7 | 257.6 | 11.0 | 100% | 0 |
| C0 | 28.1 | 148.9 | 148.9 | 148.3 | 934.8 | 11.2 | 100% | 0 |
| C1 | **22.9** | 145.7 | 151.4 | 148.5 | 934.8 | 11.7 | 100% | 0 |

## 3. Effects（主指标 = profiling OFF）

| 效应 | completion Δ | E2E Δ |
|---|---:|---:|
| reveal（C0 − C1） | **+5.2** | −0.24 ms（≈中性） |
| deployment（B0 − C1） | **+5.2** | +11.2 ms（amortized；稳态下 C1≈B0） |

## 4. P10-T0 测试

| 检查 | 结果 |
|---|---|
| 逐 chunk router 执行（无全量 top-k 后切片） | PASS（按构造 + 逐 chunk forward） |
| shard-ready = 每 chunk CUDA 完成事件 | PASS（真实事件；无 per-shard sync 依赖） |
| C0/C1 共享 token/arrival/router/top-k/traffic | PASS（20/20，traffic 逐 job 相等） |
| router vs CPU oracle（masked） | PASS |
| D1 75%@8 门控来自真实 arrival slots | PASS（arrival slots [1,1,0,0]——本规模 chunk 在 slot≤1 完成，门控近 no-op；如实） |
| no-leak / token 一致性 | PASS |
| profiling overhead（交替顺序） | PASS（~11ms 稳定、正） |
| cold/steady/amortized 分别报告 | PASS |
| hotspot_random_walk 保留并分解 | PASS（reveal −0.59ms、deployment −3.6ms） |
| 独立 read-back | PASS（0 差异） |

## 5. 诚实结论

1. **completion 收益稳健**：C1（chunked + 75%@8）比 C0 与 B0 均好 **+5.2 slots**（22.9 vs 28.1），方向与既往一致；
2. **E2E 收益未在本规模确立**：稳态下 C1（151.4ms）≈ B0（150.0ms）；amortized 的 +11.2ms 主要来自 B0 冷启动（378.7ms）噪声；
3. **chunk arrival 本规模为子 slot**（slots 0-1）：chunk 门控对调度近 no-op，C1 收益来自 reveal profile 而非 arrival 延迟——如实声明；
4. router 成本：chunked（934.8µs）≈ batched（257.6µs）×3.6，但相对调度窗口可忽略。

## 6. 限制

- E2E 仍含 ~150ms 调度窗口噪声（20 jobs 规模）；formal 需更大规模 + 稳态聚焦；
- arrival slots 子 slot，未能在本规模展示"arrival 延迟影响调度"的语义（需更大 chunk 或真实 MoE 规模）。
