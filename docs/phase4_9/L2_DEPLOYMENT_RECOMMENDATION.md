# L2 部署建议（Deployment Recommendation）

更新日期：2026-08-05

## 1. 推荐配置（冻结）

| 项 | 值 |
|---|---|
| reveal mode | partial_shards |
| reveal budget | 75% |
| full reveal checkpoint | 8 |
| fast scheduler | partial_current_only |
| adaptive / robust / predictor / risk-gate / lookahead | disabled |

## 2. L2 证据（2× V100 真实 NCCL）

- ΔE2E D1 vs D0 = +6.46ms（CI [+3.41, +9.38]ms），completion +6.43 slots；
- 吞吐 +13.7%；legality 100%；timeout 0；
- 真实 NCCL allreduce 62–87µs、allgather 122–136µs 已计入。

## 3. 实现映射（L2 级）

1. 每 rank 维护 token/shard 到达计数（本地流式），75% 在 checkpoint 8 前揭示；
2. 调度决策由单一协调器按 partial_current_only 语义进行（冻结 checker）；
3. 全局聚合（expert histogram 等）用真实 NCCL allreduce（每 commit slot 一次）；
4. 控制消息：每 reveal 事件一次（实测 ~5µs）。

## 4. 部署前置

- 真实 router/top-k 替换合成 shim（Phase 4.10）；
- 真实 expert GEMM/packing/combine 替换合成 kernel（Phase 4.10）；
- L3（多节点）需 RDMA/NVSHMEM/DeepEP 验证。

## 5. 禁止

- 不重新调参、不恢复 adaptive/robust/predictor/risk-gate/lookahead；
- 不把 L2 结论外推为 L3 或生产 SLA。
