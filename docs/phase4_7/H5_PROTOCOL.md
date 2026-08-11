# H5 正式协议：可实现早期信息是否有 E2E 价值

更新日期：2026-08-04
状态：**FROZEN（用户批准；R0=PASS）**
前序：`docs/phase4_7/H5_DRAFT_PROTOCOL.md`、`SUPERVISOR_REVIEW_PHASE4_7_0.md`（R0=PASS 有条件）

## 1. 固定内容

- fast layer scheduler = **`partial_current_only`**（不改语义）；
- 不使用 robust prefix、历史预测、risk gate；
- 不修改 deterministic checker；
- 所有 reveal 成本计入 E2E `J`；禁止只报 completion；
- 新 corpus；不复用 H2 正式 test、Phase 4.6 W1–W3 test、Route A seeds。

## 2. 新 corpus（冻结）

- 生成器与 Phase 4 / Route A 一致（length 256、4 nodes、mean 2.0、std 1.5、max 8、Rear4GPU、5 family、同一 variant 规则）；
- **base seeds = `(2042, 2142, 2242)`**；
- 45 sequence（5 family × 3 seed × 3 split），按完整 sequence 划分，同一 sequence 不跨 split；
- 正式 test 冻结前不查看；validation 先行；
- 必须记录：generator version、topology/config、manifest hash、zero-overlap（对 H2 45 条 + Route A 45 条 digest 均零交集）。

## 3. Reveal profiles（冻结，来自 capability table 的可实现项）

| 臂 | 名称 | 信息流 | 成本计入 |
|---|---|---|---|
| A1 | current fixed full-reveal baseline | 冻结揭示节奏（full slot 16） | 无额外 reveal 成本 |
| A2 | coarse early reveal | 本地流式：per-source 计数 + 已到达 token 精确揭示（更早 cadence） | C_compute + C_control（无同步） |
| A3 | coarse early + progressive refinement | A2 + 每 8 slot 一次全局 expert histogram | + C_sync（allreduce） |
| A4 | rank-local streaming reveal | 每 token 到达即精确揭示（最早 cadence） | C_compute + C_control |
| A5 | group-level reveal | 全局 bandwidth-group aggregate 在固定 slot 揭示 | + C_sync |
| A6 | full-information reference | slot 0 起全矩阵 | 不计（仅上界） |
| A7 | cost-free reveal | Route A S3 语义（免费提前） | 不计（仅 oracle 分析） |

## 4. 成本模型参数（冻结默认值；校准后替换）

```text
C_reveal = C_compute + C_control_message + C_sync + C_blocking + C_memory + C_pipeline_interference
```

参数（默认值，标注来源；校准后冻结新值）：

| 参数 | 默认 | 说明/来源 |
|---|---:|---|
| c_tok（histogram/top-k 每 token 计算） | 0.1 µs | proxy 实测校准 |
| c_msg（控制消息往返） | 10 µs | 文献级小消息 RTT（待校准） |
| alpha（allreduce 延迟系数） | 10 µs × log2(P) | 文献级（待校准） |
| beta（带宽系数） | 10 GB/s | 保守假设（待校准） |
| c_block（阻塞） | 1.0 × 同步耗时 | 默认 1:1（保守） |
| c_mem | 0（不进入 J，单独报告） | 内存仅报告 |
| c_int（pipeline 干扰） | 0.1 × 同步耗时 | 默认（待校准） |
| P（rank 数） | 4 | 与 4 节点拓扑一致 |

校准结果写入 `outputs/phase4_7/h5_realizable_reveal/cost_params.json`，并标注"measured / assumed"。

## 5. 端到端目标

```text
J = T_completion + T_reveal_wait + T_reveal_control + T_sync + T_scheduler + T_execution
```

- T_completion = completion_slots × 1 ms（proxy）；
- T_scheduler = proxy 内 partial 调度耗时（已测量 ≈104 ms/sequence）；
- T_execution = 动作执行时间（proxy 内计 0，真实系统另报）；
- T_reveal_wait / T_reveal_control / T_sync 由 profile 与成本模型计算。

## 6. 统计与记录

- 每臂：completion mean/median/p95/p99/CVaR95、full-info regret、J 及各分项、reveal volume/event count、marginal benefit per unit、legality、timeout；
- 配对：A2–A5 vs A1（baseline）sequence-level paired bootstrap 10,000（seed 20260801），95% CI；
- 分桶：family（5）、seed（3）、mode（5）；结论桶 ≥5 条独立 sequence；
- ESS/ACF 按协议报告。

## 7. H5 PASS / FAIL

PASS（全部满足）：1) 相对 A1，J 改善 > 0；2) paired CI lower > 0；3) ≥3 seed；4) ≥4/5 family 正向或有预注册适用边界；5) legality 100%；6) timeout 不增；7) 收益非免费 oracle 信息（A7 不计入，A2–A5 成本已计入）；8) 存在非平凡预算区间；9) Supervisor PASS。

FAIL（任一）：计入成本后 J 无改善；仅 completion 改善而 J 恶化；收益仅来自免费 oracle；legality/timeout 违反。

## 8. 输出

- `outputs/phase4_7/h5_realizable_reveal/`（corpus manifest、cost_params、runner、结果 CSV/JSON）
- `docs/phase4_7/H5_RESULTS.md`
- `docs/agent_coordination/SUPERVISOR_REVIEW_H5.md`
