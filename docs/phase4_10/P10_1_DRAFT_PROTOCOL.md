# Phase 4.10-1 草案协议：真实 Router / Top-k / Shard Readiness 替换

更新日期：2026-08-05
状态：**DRAFT（P10-R0 CONDITIONAL PASS 后、用户批准后正式化）**

## 1. 目标

把 L2 验证中的合成 router shim 替换为**真实可执行**的 router/top-k/token-arrival/shard-readiness 实现，在保持 frozen profile（partial_shards @ 75%、ckpt8、partial_current_only）不变的前提下，确认 E2E 收益保持。

## 2. 实现范围（P10-1 只做 router 层）

1. token 到达事件（真实计时）；
2. router top-k（每 token 的 expert 选择，真实 kernel/计时）；
3. shard readiness（shard 完成事件）；
4. 按 profile 在 checkpoint 8/16 揭示对应比例已到达 token（D1/D0 唯一差异）。

**不替换**：expert GEMM/combine（P10-2）、DeepEP（硬件不支持，另行）。合成 GEMM 在本阶段保留并标注。

## 3. 公平映射与等价性门

- D0/D1 使用**相同 token 到达流**，仅揭示时机/粒度不同；
- 等价性门：真实 router 路径下，D0 的 completion/动作 hash 与 proxy/L2 语义一致（在 router 语义明确映射后）；
- 无未来信息泄漏；legality 100%；timeout 不增。

## 4. 指标

ΔE2E（D1 vs D0）、router/top-k/shard 时延（M）、completion、吞吐、legality、timeout；sequence-level paired bootstrap。

## 5. P10-1 PASS（预注册草案）

1. 真实 router 路径可运行且计时为 M 级；
2. D1 vs D0 ΔE2E > 0 且 CI lower > 0；
3. ≥3 seed、≥4/5 family；
4. legality 100%、timeout 不增；
5. 无泄漏；等价性门通过；
6. Supervisor PASS。

## 6. 约束

- 不调参；不恢复 adaptive/robust/predictor/risk-gate/lookahead；
- 不修改 production 代码（只新增 bridge 组件与测试）；
- DeepEP 步骤明确**排除**（V100 不支持；需 Ampere/Hopper 硬件）；
- 输出：`outputs/phase4_10/p10_1_router/`、`docs/phase4_10/P10_1_RESULTS.md`、`docs/agent_coordination/SUPERVISOR_REVIEW_P10_1.md`。
