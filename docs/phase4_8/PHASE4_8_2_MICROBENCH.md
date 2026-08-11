# Phase 4.8-2：Microbenchmark 与成本校准

更新日期：2026-08-04
环境：RTX 2080 Ti 11GB / 12 核 cgroup / torch 2.8.0+cu128 / numpy 2.3.2；GPU warm-up 50 次；每指标 200–500 次重复；bootstrap 95% CI；证据等级 M（除非标注）。

## 1. 实测结果

| 指标 | mean | median | p95 | p99 | 单位 | 说明 |
|---|---:|---:|---:|---:|---|---|
| local shard accounting / token | 302 | 299 | 309 | 370 | ns | CPU |
| shard-ready event | 1.52 | 1.51 | 1.54 | 1.55 | µs | CPU |
| reveal metadata construction | 4.20 | 4.19 | 4.23 | 4.28 | µs | CPU |
| control-message RTT | 4.95 | 4.85 | 5.00 | 7.15 | µs | localhost UDP；真实 fabric 不同（E/S） |
| control-message bytes | 64 | — | — | — | B | D（协议固定） |
| histogram update / token | 302 | 300 | 316 | 403 | ns | CPU |
| scheduler step（enumerate+pack） | 1308 | 1301 | 1352 | 1766 | µs | CPU，代表 stage-1 视图 |
| host/device sync | 17.5 | 17.4 | 18.3 | 23.2 | µs | CUDA sync |
| GPU kernel launch（tiny matmul） | 36.4 | 30.7 | 32.0 | 38.9 | µs | |
| dispatch+GEMM（synthetic, 1/8/32 tokens） | 46/44/39 | 37/35/37 | 39/95/39 | 51/176/54 | µs | 合成 expert GEMM |
| GEMM overlap（2 ops, seq vs streams） | 100 / 138 | 53 / 98 | 74 / 573 | 90 / 588 | µs | 微小 op 下 streams 反而更慢（S 级结论） |
| NCCL/DeepEP contention | — | — | — | — | — | **单 rank 不可测**（L2/L3 必需，S） |
| worker RSS | 721,508 | — | — | — | kB | 含 torch CUDA 上下文 |
| CPU 利用率（调度器） | 1.00 | — | — | — | ratio | CPU-bound |

完整数据：`outputs/phase4_8/deployment_validation/microbench_results.json`

## 2. 对成本模型的校准（更新 cost_params）

| 参数 | 旧值 | 新实测（M） |
|---|---:|---:|
| control message RTT | 8.8 µs | **4.95 µs** |
| histogram/token | 336 ns | **302 ns** |
| host/device sync | 假设 | **17.5 µs** |
| kernel launch / dispatch+GEMM | 假设 | **~36–46 µs** |
| scheduler step（proxy） | — | **1.31 ms** |

## 3. 对 D1 的初步含义

1. **调度器 CPU 是主导成本**：proxy 每 slot 约 1.31 ms；D1 相对 D0 少约 6 个 completion slot，预估节省调度 CPU ~8 ms/job——比控制消息（~5µs×事件数）与 GPU 开销（~50µs/slot）高两个数量级；
2. **reveal/control/sync 成本可忽略**（µs 级 vs ms 级调度成本），与 H5 结论一致；
3. **微小 op 的 stream 并行无收益**（overlap 反而更慢，S 级）——pilot 应验证真实规模；
4. **NCCL/DeepEP 竞争无法在本实例测量**——D1 正式判定只能基于 L1 单机（明确限制）。

## 4. 约束

- 全部为 L1 单机测量；合成 GEMM 不代表生产 MoE；未运行 pilot/正式实验；未改 production 代码。
