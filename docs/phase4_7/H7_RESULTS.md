# H7 结果：自适应 Reveal Controller（规则版）

更新日期：2026-08-04
判定：**H7 = FAIL**（保留最佳固定 profile = partial_shards @ 75%）

## 1. 执行

- 数据：H5 冻结新 corpus（2042/2142/2242）validation 拟合、test 评估（300 coords）；
- 控制器（规则版，预注册）：输入 = 历史热点强度 / source 负载失衡 / 带宽组压力（各按 validation 分位分 3 桶）+ checkpoint + mode；输出 = wave-1 预算 b ∈ {25%, 50%, 75%}（partial_shards）；
- 拟合：validation 上各特征桶取 argmin-J 预算；选 validation J 最优的规则变体；
- 评估：test 上对比固定 B25/B50/B75 与 oracle（每 episode 最优预算，上界）。

## 2. 主结果（test，J ms）

| 方案 | J | 说明 |
|---|---:|---|
| 固定 B25 | 55.18 | |
| 固定 B50 | 53.97 | |
| 固定 B75（最佳固定） | **53.395** | H6 胜出 profile |
| 规则控制器（选定 b_hotspot_strength） | **53.395** | 300/300 选择 0.75（退化） |
| oracle 每 episode 最优 | 53.394 | 上界 |

## 3. 配对统计

| 对比 | ΔJ | 95% CI | seq 正 |
|---|---:|---:|---:|
| controller vs B75 | +0.0000 ms | [0.000, 0.000] | 0/15 |
| controller vs B50 | +0.574 ms | [+0.361, +0.791] | 15/15 |
| controller vs oracle | −0.0014 ms | [−0.0019, −0.0009] | 15/15（oracle 更好） |

## 4. 解释

1. **规则控制器退化**：validation 拟合的规则在每个特征桶都选择 0.75，test 上 300/300 全部选择 0.75——与最佳固定 profile（B75）完全等价（Δ=0.000 ms）。
2. **自适应的理论价值极小**：oracle（每 episode 最优预算，分布为 135×0.75 / 119×0.50 / 46×0.25）相对固定 B75 仅好 **0.0014 ms（0.003%）**——存在 episode 异质性，但可榨取价值可忽略，且低于任何实际控制开销。
3. 结论：当前设置下**没有值得自适应控制的 reveal 异质性**；控制器负收益（若计入自身开销），应保留固定 profile。

## 5. H7 PASS 判据核对

| 判据 | 结果 |
|---|---|
| 1. 相对最佳固定 profile，E2E 改善 | **FAIL**（Δ=0） |
| 2. CI lower > 0 | FAIL（0） |
| 3. ≥3 seed | 无意义（与 B75 相同） |
| 4. 多 family | 无意义 |
| 5. 不发生频繁震荡 | PASS（从不切换） |
| 6. 控制开销低于收益 | FAIL（收益=0） |
| 7. legality 100% | PASS |
| 8. Supervisor PASS | FAIL（见 `SUPERVISOR_REVIEW_H7.md`） |

## 6. 结论

**H7 = FAIL**。按协议：**保留最佳固定 reveal profile = `partial_shards` @ 75% 预算（full reveal slot 8）**；不强行保留复杂中层 controller；不进入进一步自适应训练。当前研究链收敛为：

- H5 PASS：可实现早期揭示有 E2E 价值（粗粒度/流式）；
- H6 PASS：固定预算下 partial_shards 是最佳选择器；
- H7 FAIL：无需自适应——固定 75% 分片揭示即为该语义下的实用解。

## 7. 约束

- 未修改 production 代码；H2=FAIL、Phase 5 CLOSED 维持；
- 若未来系统语义改变（更大的 rank 数、同步成本结构变化、更强异质性），H7 可重新评估，但需新协议。
