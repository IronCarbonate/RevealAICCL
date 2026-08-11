# Phase 4.8 L2：单机多 GPU 真实 NCCL 验证

更新日期：2026-08-05
判定：**D1 = PASS（L2：2× V100 真实 NCCL）**

## 1. 环境（新服务器）

- 2× Tesla V100-SXM2-32GB（driver 580.105.08 / CUDA 13.0）；
- torch 2.8.0+cu128、NCCL 2.27.3、numpy 2.3.2；
- 单机多 GPU（L2）；无多节点 RDMA（L3 未验证）；
- 仓库（172 文件 hash 校验通过）与 Phase 4.8 代码已迁移。

## 2. 真实 NCCL 2-rank 微基准（证据 M）

| 操作 | mean | median | p95 | p99 |
|---|---:|---:|---:|---:|
| allreduce 64B（expert histogram） | 63.8 µs | 62.5 | 68.6 | 76.8 |
| allreduce 32B（group） | 68.2 µs | 63.5 | 98.3 | 105.5 |
| allreduce 128B（matrix） | 61.7 µs | 58.4 | 74.8 | 159.7 |
| allreduce 1KB | 86.6 µs | 65.5 | 92.2 | 132.1 |
| allgather 128B | 121.6 µs | 121.9 | 146.4 | 169.0 |
| allgather 1KB | 136.2 µs | 120.8 | 140.3 | 268.3 |

此前 cost model 的假设值（~15µs）被实测取代：真实 2-rank allreduce ≈ 60–90µs（贵 4–6 倍）。

## 3. 正式 test（3042 corpus，300 jobs/臂，真实 NCCL allreduce 计入每 commit slot）

| 指标 | D0 | D1 | D2（上界） |
|---|---:|---:|---:|
| completion | 20.34 | **13.91** | 9.65 |
| E2E wall（µs） | 53729 | **47272** | 50175 |
| throughput（jobs/s） | 18.61 | **21.15** | 19.93 |
| GPU busy（含真实 collective，µs/job） | 4375 | **2239** | 2026 |
| legality / timeout | 100% / 0 | 100% / 0 | 100% / 0 |

## 4. 配对统计（D1 vs D0）

| 指标 | mean | 95% CI |
|---|---:|---:|
| ΔE2E | **+6,458 µs（≈6.46 ms）** | [+3,409, +9,385] µs |
| completion Δ | +6.43 slots | [+5.68, +7.12] |

- seed：3/3 正（3042 +7.4ms、3142 +5.6ms、3242 +6.4ms）；
- family：4/5 正（hotspot_random_walk −1.7ms，适用边界注明）；
- 吞吐 +13.7%；GPU busy（含真实 collective）下降 49%。

## 5. 结论

**D1 = PASS（L2）**：把 collective 从 L1 合成换成 L2 真实 NCCL 后，`partial_shards @ 75%、checkpoint 8` 仍相对 D0 带来约 **6.5 ms/job（约 12%）E2E 改善**。真实同步成本（~64µs/allreduce）按 commit slot 计入（D0 ~20 次、D1 ~14 次），仍远小于调度 CPU 节省（~1.3ms/slot × 6 slots），结论在 L2 稳健。

## 6. 边界

1. L2 = 单机多 GPU 真实 NCCL + 真实 GPU kernel；**router 仍为合成 shim、expert GEMM 为合成**；
2. **L3（多节点 RDMA/NVSHMEM/DeepEP）未验证**；
3. hotspot_random_walk family 为负；
4. H2=FAIL、Phase 5 CLOSED 维持。

## 7. 产物

- `outputs/phase4_8/deployment_validation/l2_collective_results.json`、`l2_final_summary.json`、`l2_*`（已备份本地）
- 脚本：`l2_collective_bench.py`、`formal_test.py`（PH4_8_L2=1 模式）
