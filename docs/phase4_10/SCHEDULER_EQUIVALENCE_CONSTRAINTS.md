# Phase 4.10-1F：Scheduler Fast-Path 等价性约束

更新日期：2026-08-06

## 1. 目的

本文件固定“纯实现层优化”与“语义变更”的边界。任何未来优化计划都必须逐条满足本约束，否则视为改变冻结语义，需重新走用户/监督准入。

## 2. 必须保持的冻结语义（可观测行为）

### 2.1 调度决策（partial_current_only 结构路径）
- 候选集：`enumerate_candidates(view)` 输出与生产完全相同的 `(local_token_ordinal, edge_index, before_distance, after_distance)` 集合；
- 候选排序：按 `(local_token_ordinal, edge_index)` 升序，不得改变；
- 门控：`gate` 谓词（chunk ordinal 与 `arrival_slots[chunk] ≤ slot`）不变，使用冻结的 arrival_slots（P10-1E：1..8）；
- 打包：`pack_candidate_batch` 的 first-fit 顺序与容量/共享组判定不变（边容量与组带宽按 1 单位归一化后逐候选判定）；
- 每槽提交动作序列（含顺序）与生产一致；first-commit slot 与动作数一致（本冻结世界：slot 4、2 动作）。

### 2.2 观察与视图
- `SchedulingObservationView` 的所有字段值（stage/ratio/state_version/observed_matrix/entry_mask/totals/revealed_tokens/topology/digests）必须与生产一致；
- digest（observation_digest / residual_state_digest）的值必须一致；若惰性计算，任何消费者（事件账本、plan/recourse 校验）读到的值必须与立即计算完全相等；
- 数组只读性（`readonly_array`）与深拷贝语义不得破坏（纯实现层可复用已经只读的观察数组，但不得暴露可写别名）。

### 2.3 确定性 checker（不得跳过/弱化）
- `commit_proposal` 的完整校验序列必须保留：state_version/sequence 匹配、token 已揭示且可执行、无重复 token、边存在性、source possession、destination 未持有、边容量、共享组带宽、全部通过后才应用、`_state_version += 1`；
- 拒绝路径（TypeError/ValueError → illegal 终止）与 `checker_rejected` 事件语义不变；
- checker 版本标识 `phase1-atomic-v1` 语义不变。

### 2.4 world 状态与事件
- `_possession` 更新、`state_version` 递增、事件账本（proposal_bound / action_committed / batch_committed / wait_latch / plan_built / checker_rejected / episode_end）的字段与顺序不变；
- no-leak / token 一致性 / traffic 一致性不变。

## 3. 允许的纯实现层变换（不改变 2.1–2.4 的任何可观测值）

| # | 变换 | 等价性论证 |
|---|---|---|
| E1 | 全对最短路径预计算（静态拓扑），距离 O(1) 查找替代逐次 BFS | 距离是拓扑纯函数；对同一 (topology, source, destination) 值恒等 |
| E2 | `_usable` 静态标志预计算 | 容量/组限制在单个 episode 内静态；判定值恒等 |
| E3 | 增量 batch loads（运行累计边/组用量） | `can_add_candidate` 的数值结果与从零重建完全一致；first-fit 顺序不变 |
| E4 | 融合 gate 到 enumerate 循环 | 谓词不变，仅减少一次元组重建 |
| E5 | digest 惰性计算 / 零拷贝视图 | digest 值不变；消费者读取时点后移不影响语义；数组保持只读 |
| E6 | 按 (holders, destination) 记忆化候选边列表 | 候选判定是 (holders, destination, topology) 的纯函数；输出集合与排序不变（需按 ordinal 排序） |
| E7 | 绑定/检查器内部 O(1) 数据结构（如按 edge 的负载数组预分配） | 校验数值与副作用不变 |

## 4. 禁止的变换（超出“纯实现层”，须重新准入）

| 类别 | 禁止项 |
|---|---|
| 语义改变 | 改变候选动作集/排序、pack 顺序、门控谓词、commit 顺序 |
| checker | 跳过、缓存复用（跨槽）、弱化校验、延迟应用、批量提交绕过原子性 |
| 冻结参数 | 修改 75% budget、checkpoint 8、partial_current_only、router、workload、arrival/槽位定义、time_limit |
| 计时诚信 | 人工 sleep、预计算后延迟显示、profiling 后补偿、cache 未来 reveal、oracle 泄漏 |
| 硬件/算子替换 | 真实 expert GEMM/combine、Triton kernel 替换、DeepEP、L3 多节点；V100 不支持 DeepEP（sm_70）事实不变 |
| 恢复冻结机制 | 恢复任何被冻结/废除的机制（如自适应控制器、W3 门控空转优化） |

## 5. 等价性验证要求（未来优化阶段必须满足）

1. **候选恒等**：优化实现与生产实现在冻结 workload 全部 80 slots 上，候选集与排序逐项相等（本阶段探针已示范 48/48 恒等校验）；
2. **动作序列恒等**：episode 级 committed action 序列（ordinal, edge_index, slot）与生产逐项相等；
3. **checker 行为恒等**：合法/拒绝行为与 state_version 序列相等；
4. **digest 恒等**：observation/residual digest 与生产逐槽相等（若惰性，读取时相等）；
5. **计时口径**：profiling OFF、真实计时、p95 在冻结 workload 上认证；目标 L_total < 336µs（L_sched ≤ 200µs、L_commit ≤ 135µs）。

本阶段仅完成测量与探针，未实施任何 E1–E7 变换，未触碰生产路径。
