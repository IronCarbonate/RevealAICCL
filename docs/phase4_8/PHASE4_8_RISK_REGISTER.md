# Phase 4.8 风险登记

更新日期：2026-08-04

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R-4.8-1 | 把 proxy slot 结果称为真实部署结果 | 高 | L0–L3 分级明确；L1 结论限定高保真 |
| R-4.8-2 | 未测量同步/控制成本被设 0 | 高 | 证据等级强制标注；D1 主要依赖 M |
| R-4.8-3 | L1 结果冒充 L2/L3（多节点） | 高 | 单 GPU 环境明确不可达 L2/L3；不得外推 |
| R-4.8-4 | baseline 与候选 profile 不公平比较（同时改 scheduler） | 高 | 唯一区别 = reveal 参数 |
| R-4.8-5 | CPU/GPU 时钟直接相减 | 高 | CUDA event 同步；critical-path wall-clock 为主指标 |
| R-4.8-6 | 真实 router/DeepEP 语义缺失导致高保真层失真 | 中 | R1 条件性通过；如语义不成立申请 Real-System Semantics Agent（需用户批准） |
| R-4.8-7 | 插桩改变行为（profiling on/off hash 不一致） | 高 | I1 equivalence 门强制 |
| R-4.8-8 | 复用 Phase 4.7 正式 test 作为 Phase 4.8 test | 高 | 新 workload/corpus（新种子） |
| R-4.8-9 | 单 GPU 无法测真实 collective contention | 中 | 如实报告为 L1 限制；L2/L3 需新硬件 |
| R-4.8-10 | 服务器不稳定（本会话多次失联） | 中 | 结果本地+服务器双备份；正式运行前稳定性检查 |
| R-4.8-11 | 检查点 8 profile 被提前实现 | 高 | 本轮禁止；Phase 4.8-1 才允许最小实现 |
