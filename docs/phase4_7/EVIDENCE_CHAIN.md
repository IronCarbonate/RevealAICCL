# 证据链（EVIDENCE_CHAIN）

更新日期：2026-08-04

```text
H2 = FAIL (正式, conditions 1/3/6; E2E overhead 10x)
  └─► Route A = PASS (reveal 越早 completion 越低; S3 partial 11.80 vs fullinfo 10.80;
                     新种子 1042/1142/1242, 与正式零重合, 5400 episodes)
        └─► Phase 4.7-0 = R0 PASS (rank-local 流式信息真实可实现且便宜;
                                   capability table + cost model)
              └─► H5 = PASS (可实现早期揭示计入成本后 E2E +6~9 ms / 8–15%;
                             A2/A3/A4 CI lower>0, 15/15 seq, 5/5 family, 3/3 seed;
                             A5 全局聚合 = 负)
                    └─► H6 = PASS (固定预算下 partial_shards 最佳;
                                   vs random +0.60/+0.81/+0.57 ms, CI>0)
                          └─► H7 = FAIL (自适应 controller 退化;
                                         oracle 上界仅 0.0014 ms)
                                └─► 最终固定方案: partial_shards @ 75%, full@slot 8,
                                     fast=partial_current_only, 其余全关
```

## 各环节关键文档

| 环节 | 文档 | 关键数字 |
|---|---|---|
| H2 FAIL | `docs/uncertainty_aiccl/H2_EARLY_PLANNING_RESULTS.md` | robust−Partial E2E Δ=−938.6；conditions 1/3/6 FAIL |
| Route A PASS | `docs/phase4_6/ROUTE_A_REVEAL_RESULTS.md` | S0 20.95→S1 14.92→S2 13.40→S3 11.80；fullinfo 10.80；LB 4.25 |
| R0 PASS | `docs/agent_coordination/SUPERVISOR_REVIEW_PHASE4_7_0.md` | 五条件逐条通过（rank-local 计数等） |
| H5 PASS | `docs/phase4_7/H5_RESULTS.md` | A4 +9.22 [8.26,10.13]；A2 +6.06；A5 −0.13 |
| H6 PASS | `docs/phase4_7/H6_RESULTS.md` | partial_shards +0.60/+0.81/+0.57（CI>0） |
| H7 FAIL | `docs/phase4_7/H7_RESULTS.md` | controller Δ=0 vs B75；oracle Δ=0.0014ms |
| 可复现 | `docs/phase4_7/REPRODUCIBILITY_MANIFEST.md` | corpus/hash/seeds/scripts/cost params |

## 一致性说明

- 跨环节 corpus 零重合：H2 正式（642/742/842）、Route A（1042/1142/1242）、H5/H6/H7（2042/2142/2242）；
- 每个环节先验证运行器等价性（H5 gate 300/300、W2 300/300、Route A gate 300/300）；
- 跨 run 的 scheduler wall 中位数有波动（proxy Python 开销），故跨 run J 绝对值不直接比较；所有结论基于**run 内配对**。
