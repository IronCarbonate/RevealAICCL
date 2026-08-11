# H6 结果：固定预算下的选择性 Reveal

更新日期：2026-08-04
判定：**H6 = PASS（partial_shards 为胜出选择器）**

## 1. 执行

- 数据：H5 冻结新 corpus（2042/2142/2242）test split，15 sequence × 20 coordinates = 300/臂；
- 选择器（预注册，冻结 `DemandRevealProcess` 模式）：random_entries / source_totals_first / source_destination_totals_first / partial_shards / time_based_arrival；
- 预算：wave-1 reveal ratio 25% / 50% / 75%，full reveal 固定 slot 8；
- 成本：control（实测 8.8µs/消息）；sync 只计 source_destination_totals_first（全局 dest 直方图）；scheduler 固定中位数；
- 参照：full_at_8（无提前信息）、full_at_16（当前 baseline）、A6 fullinfo（上界，H5 已测 50.61ms）。

## 2. 主结果（J，ms）

| 预算 | random | source_totals | source+dest | **partial_shards** | time_based |
|---|---:|---:|---:|---:|---:|
| B25 | 58.41 | 58.41 | 58.43 | **57.81** | 58.44 |
| B50 | 57.40 | 57.40 | 57.42 | **56.59** | 57.48 |
| B75 | 56.59 | 56.59 | 56.61 | **56.02** | 56.67 |

参照：full_at_8 = 59.82；full_at_16（现状）= 67.82。

## 3. 配对统计（partial_shards vs random_entries）

| 预算 | ΔJ | 95% CI | seq 正 | family 正 | seed 正 |
|---|---:|---:|---:|---:|---:|
| B25 | +0.604 ms | [+0.383, +0.877] | 14/15 | 5/5 | 3/3 |
| B50 | +0.807 ms | [+0.683, +0.957] | 15/15 | 5/5 | 3/3 |
| B75 | +0.573 ms | [+0.453, +0.703] | 15/15 | 5/5 | 3/3 |

entry 级选择器 vs random（B50）：source_totals +0.000（完全相同）；source+dest −0.020（同步成本）；time_based −0.084（CI 跨 0，不显著）。

## 4. 解释

1. **token 级分片揭示（partial_shards）是唯一稳定优于 random 的选择器**：同一配置预算下 J 改善 0.57–0.81 ms，跨 seed/family/序列稳定；其成本（控制 ~0.02ms + 计算 ~0.002ms）仅为收益的 ~1/30。
2. **entry 级揭示顺序不影响 partial 调度**：random / source_totals / source_destination 在相同预算下 completion 完全相同（16.76/15.75/14.94）——与 H2b/W2 结论一致：在"看到什么做什么"下，先揭示哪个完整 entry 没有调度价值。
3. partial_shards 的价值在于**把预算花在更多 entry 的部分 token 上**（更宽的 entry 覆盖），使 partial 更早获得可执行 token 的多样性。
4. **单位说明（诚实标注）**：partial_shards 的预算单位是 token 比例，entry 模式是 entry 比例，两者并非严格同单位；"配置相同 reveal ratio 下 partial_shards 更优"的结论成立，"严格相同 token 预算下"的排序需后续同单位实验确认。

## 5. H6 PASS 判据核对（partial_shards）

| 判据 | 结果 |
|---|---|
| 1. 相同预算下优于 random/fixed | PASS（3/3 预算，CI lower>0） |
| 2. E2E 稳定改善 | PASS（+0.57~+0.81 ms） |
| 3. sequence-level CI lower > 0 | PASS |
| 4. ≥3 seed | PASS（3/3） |
| 5. 多 family 稳定 | PASS（5/5） |
| 6. selector 开销低于收益 | PASS（~1/30） |
| 7. legality 100% | PASS |
| 8. timeout 不增 | PASS |
| 9. Supervisor PASS | 见 `SUPERVISOR_REVIEW_H6.md` |

## 6. 结论

**H6 = PASS**：在固定 reveal 预算下，token 级分片选择器（partial_shards）稳定优于 random；entry 级选择器无增益（与既有发现一致）。最佳固定 reveal profile = **partial_shards**。允许进入 H7（自适应 reveal controller），但：

- H7 必须以 partial_shards 为固定基础 profile；
- 严格同单位预算对比列为后续补充实验；
- H2=FAIL、Phase 5 CLOSED 维持。
