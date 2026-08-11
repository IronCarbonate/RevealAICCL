# Phase 4.8 草案协议：Real-Deployment Validation

更新日期：2026-08-04
状态：**DRAFT（等待用户审核）**

## 1. 目标

验证 Phase 4.7 最终固定方案（partial_shards @ 75%、full reveal slot 8、fast=partial_current_only、其余关闭）在**真实多 rank 部署语义**下的可实现性与 E2E，校准 Phase 4.7 成本模型中的假设项。

## 2. 验证对象

1. router：真实 top-k 计算时延与本地直方图更新成本；
2. token aggregation：per-source/per-expert 计数的真实到达时序；
3. 全局聚合：allreduce/allgather 真实延迟、带宽、阻塞与 pipeline 干扰（P≥2 实测）；
4. 控制消息：真实协调器往返时延；
5. E2E：在真实流水线（或高保真仿真）中测量 J 全部分项；
6. legality/timeout：与 proxy 一致（100% / 不增）。

## 3. 预注册设计

- 比较臂：A1 现状（full-reveal-16）、A4 流式揭示、A6 full-info 上界（对应 H5 臂）；
- 主指标：J = T_completion + T_reveal_wait + T_reveal_control + T_sync + T_scheduler + T_execution（全部实测或校准）；
- 成功判据（对应 H5 PASS 条款）：计入真实成本后 A4 相对 A1 的 J 改善 > 0 且 CI lower > 0；legality 100%；timeout 不增；
- 失败处置：若真实同步/阻塞成本推翻 proxy 结论，以实测为准并重新评估部署建议。

## 4. 数据与语义约束

- 不得泄漏未来真值；普通调度只接收已揭示 token/允许 aggregate；
- 调度语义与 checker 保持 partial_current_only 不变；
- 若需新 corpus，须新种子并做 zero-overlap；优先复用 Phase 4.7 corpus 做语义对照。

## 5. 产出

- `docs/phase4_8/PHASE4_8_PROTOCOL.md`（正式化）、`PHASE4_8_RESULTS.md`
- `outputs/phase4_8/deployment_validation/`（实测成本、E2E、结果）
- `docs/agent_coordination/SUPERVISOR_REVIEW_PHASE4_8.md`

## 6. 前置条件

1. 用户批准本草案并明确部署/仿真环境；
2. 提供多 rank 环境（或高保真仿真）与真实 router/dispatch 实现；
3. 必要时申请 Real-System Semantics Agent（说明职责/生命周期，等待批准）。
