# P10-S0 协议：Substrate Selection & Correctness

更新日期：2026-08-05
状态：**FROZEN（待用户批准后进入 P10-1A 实现或 P10-I1）**

## 1. Substrate（冻结）

- 最小 PyTorch reference（标准 torch ops）：gating Linear → topk → bincount → shard-ready 事件；
- E=4 experts、k=1、fp32、GPU（2×V100）；权重固定 seed 初始化；
- 命名：**L2-R reference substrate**（不称生产 router）。

## 2. 正确性与一致性检查（冻结）

- correctness oracle = 确定性朴素 reference（上述）；优化路径须与之逐位一致；
- token 无丢失/无重复/最终 traffic 一致性检查（见 ROUTER_CORRECTNESS_ORACLE）；
- D0/D1 相同 token/权重/top-k 断言；
- profiling on/off 等价（P10-I1）。

## 3. P10-S0 PASS 判据（预注册草案）

1. 选型证据完整（a/b 不存在，c 选定）；
2. 数据流定义完整；
3. correctness oracle 与检查定义完整；
4. D0/D1 公平性证明设计完整；
5. 不称 production router；L2-R 边界明确；
6. Supervisor PASS。

## 4. 输出

- `outputs/phase4_10/p10_1a_substrate/`（实现，P10-1A 批准后）
- `docs/phase4_10/P10_S0_RESULTS.md`、`P10_I1_DRAFT_PROTOCOL.md`（下一步）
