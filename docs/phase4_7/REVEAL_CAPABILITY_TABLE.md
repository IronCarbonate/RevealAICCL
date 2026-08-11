# Phase 4.7-0：Reveal Capability Table

更新日期：2026-08-04

| 信息类型 | 最早可用时刻 | 粒度 | 精确/估计 | 本地/全局 | 同步需求 | 数据量 | 估计开销 | 是否可流式 |
|---|---|---|---|---|---|---|---|---|
| total token count | token 到达（批内） | 全局整数 | 精确（到达后） | 全局 | 单入口无；多入口 allreduce | O(1) | 计算≈0；同步≈log(P) 延迟 | 是 |
| per-source token count（本 rank） | token 到达 | rank×source | 精确（已到达） | 本地 | 无 | O(S) | O(1)/token | 是 |
| per-source token count（全局） | 首次 allreduce 后 | 全局向量 | 精确（同步时点） | 全局 | allreduce | O(S) | 同步×1 | 是 |
| per-destination/expert total（本 rank） | router top-k 后 | rank×expert | 精确（已路由） | 本地 | 无 | O(E) | O(1)/token（top-k 副产物） | 是 |
| local top-k | router 前向时 | token×k | 精确 | 本地 | 无 | O(k)/token | 已计入 router 计算 | 是 |
| 全局 expert histogram | 首次 allreduce 后 | 全局向量 | 精确（同步时点） | 全局 | allreduce | O(E) | 同步：latency+带宽 O(P·E) | 是（低频） |
| bandwidth-group aggregate（全局） | 全局直方图后 | 组向量 | 精确（同步时点） | 全局 | allreduce+投影 | O(G) | 同步 + O(G) 投影 | 是（低频） |
| partial shard demand（本 rank） | shard 完成 | 本地分片 | 精确（已完成） | 本地 | 无 | O(shard) | O(1)/token | 是 |
| partial shard demand（全局） | 跨 rank 聚合后 | 全局分片 | 精确（聚合时点） | 全局 | allgather | O(P·shard) | 同步 ×1 | 否（离散事件） |
| 完整 traffic matrix | 全部 token 处理+全局聚合后 | S×D | 精确（最终） | 全局 | allreduce/allgather | O(S·D) | 同步 + O(S·D) 内存/带宽 | 否 |
| （proxy 假设）免费提前精确 entry | — | — | — | — | — | — | 真实系统**不存在** | — |

记法：S=source 数（=4 节点）、D=destination/expert 数（=4）、E=expert 数、G=带宽组数、P=rank 数、k=top-k。

**时序顺序**（真实系统）：token 到达计数 → router top-k → 本地直方图（以上均流式、无同步）→（可选同步）全局直方图/总量 →（再同步）完整 matrix。任何全局信息都不早于一次 collective。

**成本说明**：表中"估计开销"为结构估计；具体数值（µs/消息、带宽、阻塞）必须在 H5 用 profile 校准，**未测量成本不得设为 0**。
