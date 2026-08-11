# Phase 4.10 草案：Production-Path Bridge Validation（DRAFT）

更新日期：2026-08-05
状态：**DRAFT（L2-F0 后提交用户审核）**

## 1. 目标

在不改变 frozen profile（partial_shards @ 75%、ckpt8、partial_current_only）的前提下，逐步把 L2 验证中的合成组件替换为真实生产路径组件，确认 E2E 收益在真实路径上保持。

## 2. 替换顺序（逐步、每步独立 Gate）

1. **P10-1 真实 router/top-k/shard readiness**：合成 shim → 真实 router 实现（token→expert top-k、shard 完成事件）；
2. **P10-2 真实 expert GEMM/packing/combine**：合成 matmul → 真实 expert kernel（packing/dispatch/combine）；
3. **P10-3 DeepEP dispatch/combine**（可用时）：接入真实 DeepEP 通信路径；
4. **P10-D 正式复核**：全真实路径下 D1 vs D0 的 ΔE2E。

每步保持：reveal mode/budget/checkpoint、scheduler、checker、fail-closed 不变；不调参。

## 3. 指标与判据

- 主指标：ΔE2E（D1 vs D0），critical-path wall-clock；
- 每步 Gate：合法性 100%、timeout 不增、等价性（替换前后 completion/动作 hash 一致，除非替换本身改变语义并另行预注册）、成本 M 级；
- 最终 D10 PASS：真实路径下 ΔE2E > 0 且 CI lower > 0、≥3 seed、≥4/5 family、legality 100%、Supervisor PASS。

## 4. 约束

- 禁止重新调参、恢复 adaptive/robust/predictor/risk-gate/lookahead；
- 禁止直接进入 L3（多节点需新协议与新硬件）；
- 禁止修改 production 代码（只新增 bridge 组件与测试）；
- 不创建额外 Subagent（如需 Real-System Semantics Agent，另行申请）。

## 5. 输出

- `outputs/phase4_10/production_bridge/`；`docs/phase4_10/PHASE4_10_PROTOCOL.md`、`P10_RESULTS.md`；`docs/agent_coordination/SUPERVISOR_REVIEW_P10.md`。
