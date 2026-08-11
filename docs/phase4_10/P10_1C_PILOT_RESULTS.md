# P10-1C Pilot 结果（Reference Router Bridge）

更新日期：2026-08-05
判定：**P10-P0 = CONDITIONAL PASS（pilot 机制可准入；formal 需修正测量方法）**

## 1. 执行

- 环境：2× V100，torch 2.8.0+cu128 / NCCL 2.27.3，真实 2-rank NCCL；
- token corpus：seed 4042，20 序列（5 family × 4），dev 12 / val 8；**未使用 3042/3142/3242**；
- reference router（冻结权重）每 job 运行一次，RR-D0/RR-D1 **共享同一 router 输出**；
- shard-ready：异步 CUDA 事件（**无 per-shard cudaDeviceSynchronize**）；
- profiling OFF（wall E2E）与 ON（分项）分别测量。

## 2. 主结果（20 jobs/臂）

| 指标 | RR-D0 | RR-D1 |
|---|---:|---:|
| completion | 22.20 | **20.25** |
| E2E off（µs） | 113,994 | 133,682 |
| throughput（jobs/s） | 8.77 | 7.48 |
| legality | 100% | 100% |
| timeout | 0 | 0 |
| scheduler（µs/job） | 7,752 | 8,738 |
| GPU/NCCL（µs/job） | 7,966 | 7,487 |

配对（RR-D1 vs RR-D0）：completion Δ **+1.95 slots**；E2E Δ **−19.7ms**（pilot 规模下 D1 墙钟更差）；same_stream=True、same_traffic=True（20/20）。

## 3. 关键验证（items 2–9）

1. **D0/D1 相同 token arrival / logits / top-k / histogram / traffic**：same_stream_all=True、same_traffic_all=True（逐 job 断言）✓；
2. **shard-ready 不依赖 per-shard sync**：异步 CUDA 事件，单次最终 sync；实测 shard-ready（首 shard 含编译 warmup 166ms、后续 ~0.4ms）✓；
3. **profiling OFF/ON 分别测量**：off=wall、on=分项；overhead 量化（D0 −10.8ms、D1 +5.1ms，受顺序/负载噪声影响，见限制）✓；
4. **D1 75%/ckpt8 来自真实 router completion**：router 在调度前完成（shard-ready 全部早于 slot 0），reveal 75% 由已完成路由的 token 组成 ✓；
5. **记录项**：router/shard/scheduler/NCCL/completion/E2E/throughput/legality/timeout 全部记录 ✓；
6. **hotspot_random_walk 保留并分析**：e2e Δ −32.8ms（D1 在该 family 更差），与既往一致，如实报告 ✓；
7. **独立 read-back**：重算与官方一致（completion Δ 1.95、E2E Δ −19688.8µs，0 差异）✓。

## 4. 限制（如实）

1. **E2E 被固定 setup 主导**：每 job 的 world 构建 ~80–100ms，两臂相同，墙钟噪声吞没调度差；pilot 规模下 E2E 配对不显著（D1 甚至更差）——**formal 必须摊销 setup 或显式测量调度窗口**；
2. **overhead 测量噪声**：ON−OFF 顺序/负载敏感（D0 为负）；formal 需交替顺序 + warmup + 更多重复；
3. completion 收益（+1.95 slots）在 reference-router 路径方向与既往一致，但 E2E 收益未在 pilot 规模确认。

## 5. 结论

Pilot 机制（router 运行、D0/D1 共享流、真实 shard-ready、legality 100%、无 timeout、read-back 一致）**可准入**；但 E2E 测量方法需在 formal 协议中修正（setup 摊销 + 有序 warmup）。
