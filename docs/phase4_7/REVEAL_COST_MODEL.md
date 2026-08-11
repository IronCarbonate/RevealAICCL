# Phase 4.7-0：Reveal Cost Model

更新日期：2026-08-04

## 1. 总成本

```text
C_reveal = C_compute + C_control_message + C_sync + C_blocking + C_memory + C_pipeline_interference
```

## 2. 各分项定义与映射

| 分项 | 定义 | 映射到真实系统 | 参数化估计 |
|---|---|---|---|
| C_compute | 产生/更新 reveal 信息的计算 | router top-k（已存在）、直方图更新、矩阵构造 | `c_tok * n_tokens`；top-k 为 router 既有成本（不重复计）；直方图 O(1)/token |
| C_control_message | reveal 决策/请求的控制消息 | 协调器↔rank 的 reveal 请求/应答 | `c_msg * n_reveal_events`（每事件 2 条消息） |
| C_sync | collective 同步 | allreduce/allgather 全局聚合 | 延迟 `alpha * log(P)` + 带宽 `beta * bytes`；bytes = 桶数 × 8B × P（allreduce）/ P×桶数（allgather） |
| C_blocking | 等待同步导致的流水线停顿 | reveal 决策等待全局量时阻塞调度/后续 collective | `c_block * t_block`；与 overlap 能力负相关 |
| C_memory | 缓冲/存储开销 | 直方图、矩阵、shard 缓冲 | `c_mem * (桶数×8B + 矩阵字节)` |
| C_pipeline_interference | 对计算/通信的干扰 | 同步占用带宽、dispatch 排队 | `c_int * (同步字节/总带宽)` |

## 3. 端到端目标

```text
J = T_completion + T_reveal_wait + T_reveal_control + T_sync + T_scheduler + T_execution
```

- T_completion：调度完成时间（proxy completion × slot 时长）——Route A 已量化对 reveal 时刻的敏感性；
- T_reveal_wait：因信息晚到而等待的时间（对应 Route A 的 wait 劣势）；
- T_reveal_control：控制消息时间 = C_control_message；
- T_sync：全局聚合时间 = C_sync；
- T_scheduler：`partial_current_only` 的调度开销（≈104 ms/sequence 量级，已测量）；
- T_execution：动作执行/GEMM/dispatch 时间。

**只报告 completion 是禁止的**；H5 必须报告 J 及所有分项。

## 4. 预算与约束

- reveal 预算 B（H6 定义）= revealed_units + α·control_messages + β·sync_events + γ·blocking_time；
- 任何 reveal profile 的成本必须计入 J，**不允许把未测量同步成本置 0**；
- cost-free reveal 只作为 oracle 分析（上界），不作为普通方法；
- 参数（c_tok、c_msg、alpha、beta、c_block、c_mem、c_int、P、带宽）在 H5 先做 profile 校准（server 端 micro-benchmark 或文献值并明确标注），再用于 J。

## 5. 当前可用测量

- proxy 内调度器开销：`partial_current_only` E2E ≈ 103.9–115.8 ms/sequence（正式 artifacts `raw_test_episode_metrics.csv`）；
- reveal 敏感性：Route A（full reveal slot 16→8→4→1 的 completion 变化）；
- 未测量（H5 需校准）：真实同步延迟/带宽、控制消息时延、阻塞与 pipeline interference。
