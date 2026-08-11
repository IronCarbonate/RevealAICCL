# P10-1 Formal 协议（修订 v2）

> **Phase R0 correction（2026-08-10）**：本协议是历史 replay-based P10-1
> formal 记录并保持 CLOSED。419.8µs 是 replay/quantized candidate window，
> 不是真实 concurrent window；该 CLOSED 不禁止新 concurrent/event-driven architecture。

更新日期：2026-08-06
状态：**HOLD（P10-F0 = FAIL；待调度器延迟问题解决并经 P10-1A 后解除）**

## 1. 主指标（冻结）

- 唯一部署主指标：steady-state E2E（B0 − C1），profiling OFF；
- reveal 次级主指标：steady-state E2E（C0 − C1）；
- cold/amortized 仅次级；不得作为部署收益依据。

## 2. 前置 Gate（预注册）

- **P10-1A**：readiness 证明测试 P1–P4 全 PASS（window > scheduler p95）；
- **P10-1B**：P10-1A 后冻结并运行 formal test。

## 3. 阻塞项（P10-F0 FAIL 依据）

replay/quantized candidate window（419.8µs）< 调度器单步 p95（12,290µs）——旧 replay 配置不可准入；这不是对真实 concurrent window 的测量。

## 4. Corpus / 测量 / 统计 / 约束

- 新 corpus（5042/5142/5242）；正式 test 冻结前不查看；
- Latin-square 三臂顺序；warmup ≥10；交替 OFF/ON overhead；NCCL 真实 2-rank；
- 判据：steady-state E2E Δ > 0 且 CI lower > 0（若声明 E2E 收益）；completion Δ > 0；≥3 seed；≥4/5 family 或预注册边界；legality 100%；timeout 不增；read-back；Supervisor PASS；
- 不调参、不恢复被冻结机制、不实现 GEMM/combine、不用 Triton、不进 DeepEP/L3、不创建额外 Subagent；L2-R 命名。

## 5. 输出

- `outputs/phase4_10/p10_1_formal/`；`docs/phase4_10/P10_1_FORMAL_RESULTS.md`；`docs/agent_coordination/SUPERVISOR_REVIEW_P10_1_FORMAL.md`。
