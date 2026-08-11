# P10-1E：Formal Protocol Admissibility and Workload Freeze

> **Phase R0 修正（2026-08-10）**：本测试先完成逐 chunk CUDA timing，再把
> completion times 量化为 slot 并 replay 给 scheduler。419.84µs 只称
> replay/quantized candidate actionable window，不是直接测得的 concurrent
> router↔scheduler pipeline window。

更新日期：2026-08-06
判定：**P10-F0 = FAIL（正式协议在本配置下不可准入）**

## 1. 指标冻结

| 角色 | 指标 |
|---|---|
| 唯一部署主指标 | **steady-state E2E（B0 − C1）**，profiling OFF |
| reveal 次级主指标 | **steady-state E2E（C0 − C1）**，profiling OFF |
| 次级指标 | cold-start、amortized E2E、completion、throughput、legality、timeout、router/shard/scheduler/NCCL 分项 |

amortized +11.2ms（来自 B0 冷启动）**不得**作为部署收益；以 steady-state 为准（P10-1D 实测 C0−C1 ≈ −2.5ms、B0−C1 ≈ −1.4ms，均非正收益）。

## 2. 不对称消除（冻结）

1. B0/C0/C1 三种路径在计时前全部 warmup（编译对称）；
2. world 构建共用同一 `from_traffic_matrix` 流程（traffic 相同）；
3. NCCL init 单次、进程组唯一；collective 计数两 rank 一致；
4. 计时从统一起点（warmup 后）开始。

## 3. 执行顺序（冻结：Latin-square）

6 个 job 为一轮，三臂顺序按 Latin square 轮转：`[B0,C0,C1] / [C1,B0,C0] / [C0,C1,B0] / ...`，抵消漂移；每轮前 warmup 1 job。

## 4. Workload scale matrix（预注册，禁止按收益挑配置）

| 维度 | 冻结范围 |
|---|---|
| N tokens/job | {48, 96, 192} |
| D features | {64, 128, 256} |
| chunks | {4, 8} |
| families | 5（含 hotspot_random_walk） |
| seeds | {4042, 4142, 4242, 4342} |
| 总格点 | 3×3×2×5×4 = 360（开发/验证用；正式 test 用新 corpus 5042/5142/5242） |

全部格点预注册；**不得**根据 validation 收益只选最优格点。

## 5. Actionable readiness window（冻结定义）

- window = [75% readiness slot, final router completion slot]（真实 CUDA 完成时间推导，SLOT_US=稳态中位 chunk 时间）；
- 禁止 artificial sleep 或预计算后延迟显示。

## 6. 证明测试结果（真实计时，8 chunks × 4096 tokens × D=2048）

| 性质 | 结果 |
|---|---|
| P1 ≥3 active-window readiness events | PASS（8 个 chunk CUDA 完成事件） |
| P2 first commit < final router completion | PASS（commit @slot4 < final @slot8） |
| P3 75% readiness 不在 scheduler 启动前完成 | PASS（p75 @slot6 ≥ 启动 slot4） |
| P4 replay/quantized candidate window > scheduler p95 | **FAIL（419.8µs << scheduler step p95 12,290µs）** |

结论：**历史 P4 失败**——replay/quantized candidate window（~0.42ms）远小于调度器单步 p95（~12.3ms）。该实验不能证明真实 concurrent actionable window，也不能据此排除新的 concurrent/event-driven architecture；它只说明旧 replay-based P10-1 formal 不可准入。

## 7. P10-1A / P10-1B Gate（预注册）

- **P10-1A（Admissibility Re-check）**：在调度器延迟问题解决后重跑本证明测试，P1–P4 全 PASS 才准入；
- **P10-1B（Formal Run）**：P10-1A PASS 后，冻结正式协议、运行 formal test（新 corpus 5042/5142/5242，稳态主指标）。

## 8. hotspot_random_walk

保留并单列（既往 reveal/deployment E2E 均负），不得掩盖。

## 9. 约束

- 未运行 formal；未生成/查看 formal test 结果；未根据 validation 挑 workload；未实现 GEMM/combine；未用 Triton；未改 75%/ckpt8/router/scheduler；未恢复被冻结机制；未进 DeepEP/L3；未创建额外 Subagent。
