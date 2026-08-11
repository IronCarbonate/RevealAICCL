# 最终部署建议（FINAL_DEPLOYMENT_RECOMMENDATION）

更新日期：2026-08-04
状态：**FROZEN（待 Phase 4.8 真实部署验证后升级为部署规范）**

## 1. 推荐配置

| 项 | 值 | 真实系统映射 |
|---|---|---|
| reveal mode | `partial_shards` | 按 shard/token 到达逐片揭示，而非整 entry 掩码 |
| reveal budget（wave-1） | 75% | 第一批揭示 75% 已到达 token，其余在 slot 8 前补齐 |
| full reveal checkpoint | slot 8 | 全局聚合/同步最晚在 slot 8 完成 |
| fast scheduler | `partial_current_only` | 每 slot 用当前已揭示 token 直接贪心调度，无预测/无多场景 |
| adaptive controller | disabled | 固定 75% 分片，不做自适应 |
| robust prefix / predictor / risk gate | disabled | 全部关闭 |

## 2. 预期收益（proxy 证据）

- H5：A4（流式揭示）J 改善 +9.2ms（约 15%）、A2（粗粒度） +6.1ms（约 10%），相对 A1 现状；
- H6：partial_shards @ 75% 是固定预算最优（相对 random +0.57~0.81ms，相对现状 full-reveal-16 约 12% J 改善）；
- 综合：**在计入可实现成本后，预期 E2E 改善约 8–14%**（以 completion 与 J 的 proxy 测量为据）。

## 3. 实现要点（真实系统）

1. 在 router/输入队列维护 rank-local per-token 计数（已到达即知），按 shard 粒度向调度器暴露已揭示 token 集（source/destination/当前 holder 绑定）；
2. 首次揭示 wave-1 覆盖约 75% 已到达 token；不等待全局聚合即可开始调度（本地信息即可执行）；
3. 全局聚合（expert histogram、bandwidth-group）**不作为调度决策输入**（A5 证明无价值且增加成本）；仅用于事后统计；
4. 调度器 = 现有 partial（看到什么做什么）语义，不改 checker；
5. 控制消息预算：每 episode 2 次 reveal 事件（wave-1 + 全量），成本 ~0.02ms。

## 4. 成本参数与不确定性

- 实测（本机）：histogram 336 ns/token；matrix 613 ns/entry；控制消息 RTT 8.8 µs；
- 假设（须 Phase 4.8 实测）：collective 延迟 α=10µs·log2(P)、带宽 β=10 GB/s、阻塞系数 1.0、pipeline 干扰 0.1、P=4；
- 敏感性：A2/A4 无同步，对 collective 假设不敏感；A5 为负对假设不敏感。

## 5. 部署前置（Phase 4.8）

1. 在多 rank 环境实测 router/top-k/histogram/allreduce 真实时延与带宽；
2. 验证 partial_shards @ 75% / slot 8 在真实流水线中的可实现性与 E2E；
3. 校准 cost model 后重算 J；若真实同步成本改变结论，以实测为准。

## 6. 禁止

- 不在部署中启用 robust prefix、历史预测、risk gate、自适应 controller；
- 不把本建议外推为 legacy Torch decoder 或未经验证的部署性能；
- 不把 proxy J 数字当作生产 SLA。
