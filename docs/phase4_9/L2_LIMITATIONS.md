# L2 限制（Limitations）

更新日期：2026-08-05

1. **L3 未验证**：无多节点 RDMA/NVSHMEM/DeepEP；collective contention 在高 rank 数下未测；
2. **router 为合成 shim**：非真实 MoE router/top-k；shard readiness 为 proxy 事件；
3. **expert GEMM 为合成 kernel**：非真实 expert/packing/combine；
4. **hotspot_random_walk family 为负**（ΔE2E −1.7ms）：预注册适用边界，不得在未复测前声称该 family 收益；
5. **调度器为单一协调器视图**：真实多 rank 并行调度语义未建模；
6. **NCCL 为 2-rank 实测**：更高 rank 数的延迟/带宽未测（需 L2+ 硬件）；
7. 控制消息为 localhost 实测，真实 fabric 不同；
8. E2E wall 受机器负载影响（CI 较宽）；结论基于配对，非绝对时延。

以上限制不改变 L2-D1 PASS；但任何生产声明必须经由 Phase 4.10 真实路径替换后重新验证。
