# Phase 4.8-0：Measurement Capability Table

更新日期：2026-08-04

| 待测项 | 当前可测 | 需最小构建后 | 工具/来源 | 备注 |
|---|---|---|---|---|
| router start/end | 否 | 是 | router shim + perf_counter | 需实现 |
| top-k ready | 否 | 是 | router shim + CUDA event/CPU clock | 需实现 |
| local shard ready | 部分（proxy 事件） | 是（真实时间戳） | reveal 事件 + perf_counter | proxy 有事件，无真实时钟 |
| reveal event | 部分 | 是 | 同上 | — |
| control message send/receive | 否（仅 localhost RTT 8.8µs 实测） | 是 | socket/协调器计时 | H5 已有 localhost 测量 |
| synchronization start/end | 否 | 是 | NCCL 单 rank / 模拟 + 计时 | 多 rank 需 L2 |
| scheduler start/end | 是（proxy 壁钟） | 是 | perf_counter | proxy 已有 |
| dispatch start/end | 否 | 是 | dispatch shim | 需实现 |
| expert GEMM start/end | 否 | 是 | CUDA event（torch） | 需最小 kernel |
| all-to-all start/end | 否 | 是 | 模拟/占位 + 计时 | L2 前为模拟 |
| allreduce/allgather start/end | 否 | 是 | NCCL 单 rank 或模拟 | 多 rank 需 L2 |
| CPU wait | 否 | 是 | CUDA event + CPU clock 对齐 | 需实现 |
| GPU idle | 否 | 是 | CUDA event 间隙 | 需实现 |
| kernel launch | 否 | 是 | torch profiler / CUDA event | torch profiler 可用 |
| job/microbatch completion | 是（proxy completion） | 是（真实墙钟） | 事件账本 + perf_counter | — |
| NCCL contention | 否 | 部分 | NCCL trace（L2+） | 单 rank 不可测 |
| memory footprint | 是（cgroup） | 是 | /proc、nvidia-smi | — |
| CPU utilization | 是 | 是 | /proc/stat | — |

## 可测与不可测成本

- **可测（L1 构建后）**：router、top-k、shard/reveal、scheduler、dispatch、GEMM、控制消息、CPU/GPU 时间、job 完成；
- **不可测（本环境）**：多 rank allreduce/allgather 真实竞争、NCCL/DeepEP contention、多节点 RDMA、compute-comm overlap 的真实网络行为——需 L2/L3 硬件；
- **禁止**：把不可测成本设为 0；必须标注证据等级（M 实测 / E 外推 / D 推导 / S 假设 / O oracle）。
