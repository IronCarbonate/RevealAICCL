# Production-Path Measurement Capability Table

更新日期：2026-08-05

| 组件 | 状态 | 可测项（当前） | 需构建后测 | 工具 |
|---|---|---|---|---|
| router（真实） | 缺失 | — | token→expert top-k 时延、路由决策 | 需实现（P10-1） |
| top-k（真实） | 缺失 | — | top-k kernel 时延 | 需实现（P10-1） |
| token arrival（真实） | 缺失 | — | 到达事件时延/吞吐 | 需实现（P10-1） |
| shard readiness（真实） | 缺失 | — | shard 完成事件 | 需实现（P10-1） |
| histogram（真实聚合） | 缺失 | — | 聚合时延 | 需实现（P10-1）+ NCCL |
| expert packing/GEMM/combine | 缺失 | — | kernel 时延、吞吐 | 需实现（P10-2） |
| NCCL allreduce/allgather | **可用** | 已测（62–87µs / 122–136µs，M） | 更高 rank 数 | torch.distributed |
| MSCCL 工具 | 代码存在、不可编译 | — | 安装 msccl 后 | xml_converter |
| DeepEP | 缺失 + 硬件不支持（V100 sm_70） | — | 需 sm_80+ 硬件 | DeepEP |
| triton kernel | 工具可用 | — | 自定义 kernel 时延 | triton 3.4.0 |
| 调度器（partial_current_only） | 可用（proxy） | 1.31ms/step（M） | 真实路径适配 | CPU perf_counter |

## 结论

当前可直接测量的生产路径仅 NCCL collectives；router/GEMM/DeepEP 均需构建或新硬件。所有新组件测量须标注证据等级（M/E/D/S/O）。
