# Phase 4.10-1F：Scheduler 纯实现层优化计划（草稿，未实施）

更新日期：2026-08-06
状态：**DRAFT / 未实施 / 未触碰生产路径**（等价性约束见 `SCHEDULER_EQUIVALENCE_CONSTRAINTS.md`）

> R0 口径：下表是冻结 Python 实现 estimate，不是 strict/theoretical lower bound；
> 419.8µs 也是 replay/quantized candidate window，不是 concurrent measurement。

## 1. 目标

在冻结语义下把 `L_total = L_sched + L_commit` 压到预注册目标 `<336µs`（L_sched ≤ 200µs、L_commit ≤ 135µs）。

## 2. 现状与差距（实测）

| 项 | 现值 p95 | 目标 | 差距 |
|---|---:|---:|---:|
| L_sched（enumerate+gate+pack） | 11.29–12.93ms | ≤ 200µs | 56–65× |
| L_commit（bind+checker） | 95.5µs | ≤ 135µs | 已达标 |
| step-only optimized-path estimate | 1,043.1µs（含 bind/checker 1,139.5µs） | < 336µs | 3.1×（step-only） |

## 3. 允许的优化项（按收益/风险排序，全部属 E1–E7）

| 序 | 优化 | 对应探针实测 | 预估后 p95 | 等价性 | 风险 |
|---:|---|---:|---:|---|---|
| 1 | 全对最短路径预计算（BFS→O(1) 查找） | enumerate 9,425µs → enumerate-min 798.7µs | step ≈ 950µs | E1（距离纯函数） | 低 |
| 2 | `_usable` 静态标志预计算 | 含在 enumerate 内 | — | E2 | 低 |
| 3 | pack 增量 loads（运行累计） | pack 1,005.7µs → 138.3µs | step ≈ 950µs | E3（数值恒等） | 低 |
| 4 | 融合 gate 到枚举循环 | gate 11.5µs → ~0 | step ≈ 940µs | E4（谓词不变） | 低 |
| 5 | digest 惰性/移出决策关键路径 + 零拷贝视图 | view 1,104µs → ~106µs（digest 907.7µs 延迟到事件账本消费时） | 首提交总延迟再降 ~0.9ms | E5（值不变） | 中（需 digest 读取恒等测试） |
| 6 | 按 (holders, destination) 记忆化候选列表 | 未实施未测量（乐观 300–600µs 总区间） | L_total 乐观 300–600µs | E6（输出集合+排序不变） | 中（holders 变化需正确失效） |
| 7 | bind/checker 内部 O(1) 数据结构 | bind 14.1µs / checker 82.4µs 已接近最小 | L_commit ≈ 90–100µs | E7 | 低 |

纯实现层落地后的预期：L_sched ≈ 940–950µs、L_commit ≈ 90–100µs、L_total ≈ 1,030–1,050µs——
**仍超出 336µs 目标约 3.1×**。只有进一步启用未测量的记忆化（E6）或 numpy 向量化（超出本阶段“纯 Python 层”测量范围）才可能接近目标，且无法在本阶段认证。

## 4. 明确不实施 / 不包含项

- 真实 expert GEMM / combine；Triton kernel；DeepEP；L3 多节点（V100 sm_70 不支持 DeepEP）；
- 修改 workload、75% budget、checkpoint 8、partial_current_only、router、arrival/槽位定义、time_limit；
- 跳过或弱化 `commit_proposal` checker；人工 delay；预计算后延迟显示；恢复任何被冻结机制；
- 本阶段不实施任何 E1–E7 变换（仅探针测量）。

## 5. 若未来用户批准实施（新阶段）的建议顺序

1. Phase O-1：实现 E1+E2+E3+E4（纯 Python，低风险），在冻结 workload 上做候选恒等（80 slots 逐项）与 L_total p95 实测；
2. Phase O-2：实现 E5（惰性 digest + 零拷贝视图），做 digest/事件账本恒等校验；
3. Phase O-3：实现 E6（记忆化），做 holders 失效测试与动作序列恒等；
4. 每步必须通过：等价性校验（候选/动作序列/checker/digest）+ 冻结 workload p95 认证；
5. 仅当实测 L_total < 336µs 且 P1–P4 全 PASS，才可申请 P10-1A 重检 → P10-1B formal。

## 6. 结论

允许的纯实现层优化能把 12.9ms 降到约 1.0–1.1ms（≈12×），但**无法在本阶段证明达到 336µs 目标**；
该计划作为草稿保留，供用户审核后在明确批准下进入实施与测量阶段。
