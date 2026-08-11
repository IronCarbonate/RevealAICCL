# Supervisor Review — Phase 4.8 L2（单机多 GPU 真实 NCCL）

更新日期：2026-08-05
审查人：Supervisor（Project Director）
判定：**D1 = PASS（L2）/ NO VETO**

## 1. 独立复核

1. 新环境：2× Tesla V100-SXM2-32GB、torch 2.8.0+cu128、NCCL 2.27.3，L2 单机多 GPU 成立 ✓；
2. 真实 NCCL 2-rank 微基准（M）：allreduce 62–87µs、allgather 122–136µs，取代假设值 ✓；
3. 正式 test（3042 corpus，300 jobs/臂，真实 collective 计入）：D1 vs D0 ΔE2E +6.46ms（CI [+3.41,+9.38]ms）、completion +6.43 slots、3/3 seed、4/5 family、吞吐 +13.7%、legality 100%、timeout 0 ✓；
4. 成本计入完整（真实 NCCL per commit slot；D0 20 次 vs D1 14 次）；GPU busy 下降 49% ✓；
5. 边界如实：router/GEMM 合成、L3 未验证、hotspot family 为负 ✓；
6. 未修改 production 代码；未开启 H1/H2/robust prefix/Phase 5 ✓。

## 2. 判定

**D1 = PASS（L2）/ NO VETO**。结论在 L2（单机多 GPU 真实 NCCL）稳健成立。允许将"partial_shards @ 75%、ckpt8"作为 L2 级验证结论；L3 需多节点硬件与真实 router/DeepEP 路径（另行立项）。
