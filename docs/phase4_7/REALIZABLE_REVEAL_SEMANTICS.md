# Phase 4.7-0：真实 Reveal 语义映射（数据流审计）

更新日期：2026-08-04
状态：Phase 4.7-0 只读审计完成；未实现任何新 reveal policy；未生成新 corpus

## 1. 审计对象与映射原则

本审计把 AICCL 代理中的"reveal"映射到真实 MoE 风格系统的信息产生点。代理中的 `X_t`（traffic matrix，source→destination token 计数）在真实系统中不是一份即时可得的表，而是由以下组件在运行中逐步累积的信息：

1. **token 到达/输入队列**：谁到达、何时到达（rank-local 计数）；
2. **router**：为每个 token 计算 top-k expert 选择（本地计算）；
3. **token aggregation**：按 source/expert 的本地计数（histogram）；
4. **dispatch**：把 token 发送到目标 rank（alltoall/转发）；
5. **expert/GEMM 执行**：真正的计算发生在信息揭示之后；
6. **collective 同步**：allreduce/allgather 聚合全局统计；
7. **traffic matrix construction**：由各 rank 本地计数聚合出全局 source→destination 矩阵。

## 2. 数据流图

```text
tokens ──► 输入队列(rank-local) ──► router: top-k 计算 ──► 本地 per-source/per-expert 计数
                                        │                        │
                                        │ (本地、无同步)          │ 本地 histogram（可流式）
                                        ▼                        ▼
                                    dispatch (alltoall) ◄── 控制消息(何时 reveal/同步)
                                        │
                                        ▼
                                    expert/GEMM 执行
                                        │
                                        ▼
                    全局聚合: allreduce(总和) / allgather(矩阵) ──► 全局 expert histogram
                                        │                              │
                                        ▼                              ▼
                    全局 bandwidth-group aggregate ──────────► 完整 traffic matrix
```

关键结论：**信息沿流水线从"本地即时"逐步变为"全局延迟"**。所有"更早"的信息都是 rank-local 或流式的；任何全局量（全局 expert histogram、完整 matrix）都以同步为代价。

## 3. 十四个问题的回答

1. **router 何时产生 rank-local topk_idx**：每个 token 在 router 前向时即时产生（token 处理时点），rank-local，无同步；这是最早可用的语义信息之一。
2. **每个 rank 何时知道本地 token 分配**：token 到达输入队列时即可计数（到达即知），早于 router。
3. **source totals 何时可得**：本 rank 的 source totals 在 token 到达时可得（本地计数）；**全局** source totals 需要 allreduce。
4. **destination/expert histogram 何时可得**：本地版本在 router top-k 后可得（每 token O(1) 更新）；全局版本需要 allreduce（秩×桶规模）。
5. **bandwidth-group aggregate 何时可得**：本地 top-k 经公开拓扑带宽组投影可得本地版本；全局版本需要同步。
6. **shard-level demand 何时可得**：每个 shard 处理完成时本地可得；全局需要跨 rank 聚合。
7. **完整 traffic matrix 何时可得**：仅在所有 rank 的本地 source×destination 计数全部聚合后（allgather/allreduce），且必须等全部 token 处理/到达。
8. **哪些信息需要全局同步**：全局 expert histogram、全局 source/destination totals、全局 bandwidth-group、完整 traffic matrix。
9. **哪些信息可以流式产生**：所有 rank-local 计数（到达计数、top-k 计数、本地 histogram、本地 shard 计数）。
10. **哪些信息会阻塞 GEMM/dispatch/collective**：同步（allreduce/allgather）可能阻塞后续 dispatch/collective；控制消息本身不阻塞计算，但 reveal 决策若等待同步会阻塞调度。
11. **reveal 是否可以与计算/通信 overlap**：可以——本地流式计数与计算天然 overlap；全局聚合可以与 GEMM overlap（异步 collective），但会占用带宽；阻塞成本需显式建模，不能默认 0。
12. **更细 reveal 增加多少控制消息/内存/同步**：控制消息 ≈ reveal 事件次数；内存 ≈ 桶数与秩数的乘积；同步 ≈ 每 reveal 一次 collective（粒度越细、频次越高，同步成本越高）。
13. **提前 reveal 是精确值还是估计值**：本地到达/路由计数是**精确的已发生事实**（不预测未来）；但"全局总量/全局直方图"在同步前只是**本地子集**，不是全局真值；任何由本地外推全局的量为估计。
14. **真实系统中 reveal 的物理来源**：输入队列计数器、router top-k 输出、本地 histogram、shard 完成事件、collective 聚合结果。

## 4. 真实可实现 vs 仅 proxy 可实现

| 信息 | 真实系统可实现 | 说明 |
|---|---|---|
| total token count（全局） | 部分 | 单一入口点可知；多入口需 allreduce |
| per-source token count（本 rank） | **是（流式、即时）** | 到达计数，无同步 |
| per-destination/expert total（本 rank） | **是（router 后即时）** | top-k 计数，无同步 |
| local top-k | **是（router 内）** | 计算副产物，零额外成本 |
| 全局 expert histogram | 是（有同步成本） | allreduce 求和 |
| bandwidth-group aggregate（全局） | 是（有同步成本） | 全局直方图经拓扑投影 |
| partial shard demand（本 rank） | **是（流式）** | shard 完成计数 |
| 完整 traffic matrix | 是（最贵、最晚） | 全局聚合，最后可得 |
| proxy 的"免费提前精确 entry" | **否** | 代理假设，真实系统不存在 |

## 5. 对 Phase 4.7 的含义

- 真实可实现的"提前信息"主要是 **rank-local、流式、粗粒度**（source 计数、top-k、本地直方图）；
- 全局量（expert histogram、bandwidth-group、完整 matrix）都有同步成本且更晚；
- H5 的 reveal profiles 必须基于上表：coarse early reveal（本地计数）、progressive refinement（本地→全局）、rank-local streaming reveal、group-level reveal（全局带宽组，计同步）。
