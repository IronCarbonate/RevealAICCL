# P10-1 Formal 草案协议（DRAFT）

更新日期：2026-08-05
状态：**DRAFT（P10-P0=CONDITIONAL PASS 后，用户批准后冻结）**

## 1. 目标

在 L2-R reference router 路径上做正式 D1 vs D0 判定：确认计入真实 router/shard/NCCL 成本后，frozen profile 的 E2E 收益成立。

## 2. Corpus（新，冻结）

- 禁止：3042/3142/3242（保留为 Phase 4.8 formal）与任何已用 corpus；
- 草案种子：`(5042, 5142, 5242)`（正式化时冻结）；
- 划分：development / validation / **formal test**（正式 test 冻结前不查看）；按独立 job/sequence。

## 3. 测量方法（修正 pilot 限制）

- **E2E 主指标 = 调度窗口 critical-path**（world 构建摊销：每进程预热 + 每 job 复测 setup 成本并从 E2E 中剔除或单列）；
- overhead 测量：交替 OFF/ON 顺序 + warmup ≥50 + 重复 ≥5，报 mean/p95/CI；
- NCCL：真实 2-rank allreduce 计入 critical path；
- router/shard：真实 CUDA 事件（异步，无 per-shard sync）。

## 4. 统计与判据（预注册草案）

- 主比较：RR-D1 vs RR-D0 的 sequence-level paired ΔE2E，bootstrap 10,000（seed 20260801）；
- PASS：ΔE2E > 0 且 CI lower > 0；≥3 seed；≥4/5 family 正向或有预注册边界；completion 改善仍存在；legality 100%；timeout 不增；router/shard/NCCL 成本计入；read-back 一致；Supervisor PASS；
- hotspot_random_walk 必须单列报告（既往为负，不得掩盖）。

## 5. 约束

- 不调参、不恢复被冻结机制、不实现真实 GEMM/combine（P10-2）、不用 Triton、不进 DeepEP/L3、不创建额外 Subagent；
- 命名：L2-R reference，不称生产 router。

## 6. 输出

- `outputs/phase4_10/p10_1_formal/`（15 项 artifact + read-back）；`docs/phase4_10/P10_1_FORMAL_RESULTS.md`；`docs/agent_coordination/SUPERVISOR_REVIEW_P10_1_FORMAL.md`。
