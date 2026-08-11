# Compiled Event-Driven AICCL Architecture（Design Only）

更新日期：2026-08-10  
状态：EventBridge（R2-E0）与 compiled semantic path（R2-C0）已实现；R2-C0 待 Supervisor

## 1. 不变量与边界

后续架构必须保持：`partial_current_only`、`partial_shards=75%`、checkpoint8、
deterministic checker、fail-closed。future chunk/top-k 在 completion event 对 host
可见之前不得进入 ready state。历史 replay-based P10-1 保持 CLOSED。

目标流水线仍是：

`router future chunks || revealed scheduler || legal NCCL submission`，最终成功条件为
至少一个真实 shard `i` 满足：

`t_NCCL_submit_i < t_final_router_completion`。

## 2. Runtime 数据流

1. Router stream 独立执行 chunk forward 并 record completion event。
2. EventBridge 原生 pinned thread busy-query 预注册 event handle。
3. bridge 只把完成 slot 的 atomic state 从 ARMED 发布为 READY。
4. runtime 读取固定 ready bitmap，以 release/acquire 顺序将新 chunk 合并到
   IncrementalState；未完成 slot 不提供 traffic 指针。
5. StaticPlanCompiler 预生成的 immutable plan 只由 ready bits 和当前容量状态索引。
6. FastBinder 将 structural action 中的 local ordinal 绑定为已经 revealed 的 opaque
   token ID。
7. StaticProof 元数据先做常数时间静态约束检查，DynamicGuard 检查当前版本、ready、
   holder/capacity/shared-group/token-integrity；任何不一致 fail closed。
8. equivalence 阶段旧 deterministic checker 仍为权威；通过 R2-C0 后才能申请改变运行模式。
9. checker commit 后才允许真实 NCCL async API call；API call、return、最终 wait 分开计时。

## 3. 固定内存布局

### EventBridge（已实现）

- `Slot[8]`：`state/event_handle/armed/last_notready_start/ready/error/poll_count`；
- cache-line aligned atomic storage；
- poll loop 无 allocator、锁、sleep、Python callback；
- host monotonic timestamp only；CUDA event 只作完成条件，不与 host clock 相减。

### StaticPlanCompiler（R2-C0 已实现）

输入仅含冻结 topology、partial_current_only semantics、capacity/shared-group 规则、
deterministic tie order 和 checkpoint schedule。输出为不可变数组：

- canonical edge table 与 shared-group membership；
- 每个 revealed-count/state-class 的 candidate template offsets；
- stable action IDs 与 deterministic sort keys；
- binder ordinal ranges；
- StaticProof obligations/metadata；
- plan version、topology digest、semantic digest。

编译器不得读取 future traffic，也不得根据收益/运行结果选择 workload。

### IncrementalState（R2-C0 已实现）

预分配结构：

- `ready_bitmap[8]`、`consumed_bitmap[8]`；
- revealed token/source/destination arrays；
- per-expert holder bitsets；
- per-edge 与 shared-group remaining capacity；
- token seen/committed bitsets；
- monotonic `state_version` 与 checkpoint marker。

每次 transition 只处理新 ready chunk 的 delta，不重建 Python observation/world。

### FastBinder（R2-C0 已实现）

输入是 `(plan_action_id, state_version)`，输出固定大小 proposal buffer。必须检查：

- ordinal 对应 chunk 已 ready；
- token ID 已 revealed 且未使用；
- source/holder 与 structural edge 一致；
- tie order 与旧 binder 相同；
- 无内存分配；错误立即 fail closed。

### StaticProof + DynamicGuard（R2-C0 已实现）

StaticProof 对固定 plan 证明 topology/edge/group/ordinal 范围与确定性排序；它不能证明
运行时 traffic。DynamicGuard 检查所有 traffic-dependent 条件：ready、holder、edge/group
capacity、same-slot forwarding、duplicate/loss、state version。二者共同过滤 proposal，但在
R2-C0 equivalence 通过前不得取代旧 checker。

## 4. 线程与调度模型

- 每 rank 单进程；router launch thread、native event thread、scheduler runtime thread
  共享预分配内存；
- event thread 固定 CPU，scheduler thread 使用不同 CPU；
- event→ready 由 atomic release/acquire 发布，不用 queue/pipe/condition sleep；
- runtime 只处理 bitmap delta；没有 JSON/pickle/ProcessPool；
- Python 仅在 setup、artifact serialization 和非 timed diagnostics 中使用；
- 若 Python runtime 仍无法达到 R2-F0，才申请把 plan lookup/guard/commit path 下沉为
  C++ extension；本轮不实现该路径。

## 5. 时间戳

所有控制路径 timestamp 使用同一 host monotonic domain：

- event bridge `ready_ns`；
- scheduler start；
- legal action；
- checker done/commit；
- NCCL API call 与 submit return；
- final router completion 对 host 可见时刻。

CUDA duration 单独使用 CUDA event elapsed time，只作 GPU duration；不得与 host timestamp
直接相减。R2-O0 的核心不等式只比较 host monotonic timestamps。

## 6. 后续 Gate（未授权实施）

- R2-C0：compiled path 与旧 scheduler/checker semantic equivalence 已完成技术测试，待 Supervisor；
- R2-F0：测量真实 ready→commit feasibility，禁止借助 workload/window 变化；
- R2-O0：真实证明至少一个 `t_NCCL_submit_i < t_final_router_completion`。

本设计不构成上述 Gate 的完成或实施授权。
