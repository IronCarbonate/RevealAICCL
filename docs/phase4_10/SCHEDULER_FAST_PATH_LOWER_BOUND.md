# Phase 4.10-1F：Scheduler Implementation Fast-Path Estimate

更新日期：2026-08-06

口径修正：2026-08-10（Phase R0）。文件名为历史路径，本文不再声称严格或
理论下界。

## 1. 口径定义（预注册）

- `L_sched` = scheduler 单步 p95（冻结定义：enumerate + gate + pack_candidate_batch）；
- `L_commit` = 决策后首提交 p95（bind + Proposal + commit_proposal checker）；
- `L_total` = L_sched + L_commit（“调度决策 + 提交”的实现估算）；
- replay/quantized candidate actionable window（P10-1E 历史冻结值）= 419.84µs；该值不是 concurrent pipeline window；
- 实际目标（预注册）：**L_total < 336µs**（= 0.8 × 419.84），子目标 L_sched ≤ 200µs、L_commit ≤ 135µs。

## 2. 方法：测量锚定的 implementation fast-path estimate

用纯测量探针（不改生产代码）估计“在同一语义、同一 Python 实现层内”的
fast path 成本，全部基于远程服务器实测。分离探针 p95 的相加不是联合 p95，
也不能证明其他等价实现无法更快：

| 探针 | p95 (µs) | 语义等价校验 |
|---|---:|---|
| enumerate-min（预计算全对距离 + 静态 usable 标志） | 798.7 | 48/48 候选与生产 enumerate 相等 |
| pack-incremental（运行累计 loads，同一 first-fit 顺序） | 138.3 | 同一候选顺序、同一批选择规则 |
| gate 融合（并入枚举） | ~0（原 11.5） | 谓词不变 |
| view-min（零拷贝复用只读数组、不计算 digest） | 106.1 | 字段值一致（digest 值由惰性计算保持） |
| digest 对（observation+residual，若仍在关键路径） | 907.7 | 值不变，仅计算时点后移 |
| bind（序数→truth token） | 14.1 | 绑定结果不变 |
| checker（commit_proposal） | 82.4 | 完全保留，不跳过不弱化 |

## 3. Fast-path estimate 汇总

| 指标 | 数值 (µs) |
|---|---:|
| step-only estimate（enumerate-min + pack-inc + view-min） | **1,043.1** |
| bind/checker estimate | 96.4 |
| **bind/checker-inclusive estimate** | **1,139.5** |
| digest-inclusive estimate | **2,047.2** |

## 4. 可行性判定

| 场景 | L_total p95 | vs 419.84µs | vs 336µs |
|---|---:|---:|---:|
| 当前实现（实测） | ≈ 11.39–13.03ms | 27–31× 超 | 34–39× 超 |
| step-only optimized-path estimate | ≈ 1,043µs | 2.48× 超 | 3.10× 超 |
| bind/checker-inclusive estimate | ≈ 1,140µs | 2.71× 超 | 3.39× 超 |
| digest 留在关键路径 | ≈ 2,047µs | 4.88× 超 | 6.09× 超 |
| 乐观未测量估计（记忆化/向量化，未实施未测量） | 300–600µs | 边界 | 边界/超 |

**历史判定：旧 replay-based P10-1 未达到准入目标。**

冻结 Python fast-path 探针给出 step-only ~1,043µs、含 bind/checker
~1,140µs、含 digest ~2,047µs。它们均高于 419.8µs replay/quantized
candidate window 与历史 336µs 目标，但只是 implementation estimates；不能据此
断言新的 concurrent/event-driven、compiled 或 GPU-resident architecture 不可能进入
其真实直接测量窗口。

因此：历史 P10-1E P4 在冻结 replay 配置下 FAIL；历史 P10-1 formal 后续已由
用户裁定 CLOSED。该 CLOSED 不覆盖新立项的 concurrent/event-driven architecture。

## 5. 诚实的边界声明

1. 这些数值是独立 microbenchmark 组成的 implementation estimates，不是数学、信息论或跨架构下界。
2. 记忆化、向量化、compiled、event-driven 或 GPU-resident 版本未在本阶段实现或测量，不得据此宣称其性能。
3. bind/checker-inclusive estimate 明确为 1,139.5µs；1,043.1µs 仅是 step-only，不能标成“含 checker”。
4. digest-inclusive estimate 为 2,047.2µs。
5. 未来 concurrent pipeline 必须直接记录 event completion→host visibility→scheduler start→legal action→NCCL submit，不得沿用 replay window 冒充并发窗口。

## 6. 预注册目标（正式 P10-1A 重检时的 P4 判定式）

```
P4_pass := L_sched_p95 + L_commit_p95 < 336µs
  with  L_sched_p95 ≤ 200µs  and  L_commit_p95 ≤ 135µs（子目标，合计 335µs < 336µs）
```

测量要求：冻结 workload、真实计时、profiling OFF、含确定性 checker、无人工 delay、无预计算后延迟显示。
