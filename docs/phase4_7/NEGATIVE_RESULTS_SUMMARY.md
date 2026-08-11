# 负面结果汇总（NEGATIVE_RESULTS_SUMMARY）

更新日期：2026-08-04

## 1. 研究路线级负面结论

| 结论 | 关键证据 | 状态 |
|---|---|---|
| H1 历史预测 FAIL | MLP 无跨 seed 稳定增益；LOFO 0/5 family 正向；不恢复 point prediction | 冻结 |
| H2 robust prefix FAIL | completion/CVaR 局部不差，但 E2E 慢 ~10 倍；conditions 1/3/6 FAIL | 冻结 |
| Phase 5 CLOSED | 无 Gate 依据开放 | 冻结 |
| W2 调度排序改进无价值 | distance/headroom 与 partial 完全相同；lookahead +0.03 不显著且更慢 | 冻结 |
| W3 风险 gate 空转 | 提前行动 99% 坐标更优、wasted=0；不存在"等待更优"子群 | 冻结 |
| A5 全局聚合信息负收益 | bandwidth-group 全局揭示不改变 completion 且增加同步 → J 全序列为负 | 冻结 |
| H7 自适应 controller FAIL | 规则控制器退化为固定 B75（Δ=0）；oracle 上界仅 0.0014ms | 冻结 |

## 2. 单点负面证据

- entry 级揭示顺序无调度价值（H6：source_totals vs random Δ=0.000）；
- 更粗揭示粒度（S5）劣于更细（S1）：同 slot 8 差 1.38 slots；
- W1 静态预计算 302× 查询加速但不改变 completion（E2E 节省 ~10%）；
- cost-free reveal（A7）与可实现（A4）仅差 0.08ms：成本不是主要矛盾，但仍是必须计入的约束。

## 3. 意义

负面结果共同收敛到同一结论：**该 proxy 语义下，调度决策侧（排序、选择、自适应、多场景）没有剩余价值；瓶颈是信息揭示时机与粒度，且最优解是简单固定的 partial_shards @ 75% / slot 8。** 这是可复用的负面证据，防止未来路线重复探索。
