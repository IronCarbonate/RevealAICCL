# P10-I1 草案协议：Substrate ↔ Proxy 等价性

更新日期：2026-08-05
状态：**DRAFT**

## 1. 目标

证明 L2-R reference substrate 的 router 语义与 L2 proxy reveal 语义在 D0/D1 下等价（除 profile 外无差异）。

## 2. 等价性检查

1. **router 输出一致性**：相同 token 序列下，substrate 的 expert_idx/score 与 proxy 的 destination 映射逐 token 一致；
2. **D0/D1 相同输入**：两臂 token 序列、权重、top-k 相同（逐 token 断言）；
3. **profile 唯一性**：除 reveal 时机/粒度外，调度器/checker 输入相同；
4. **profiling on/off**：动作/事件 hash、completion、legality、timeout、RNG 一致；
5. **token 检查**：无丢失/无重复/最终 traffic 一致性。

## 3. P10-I1 PASS（草案）

- 全部检查通过（0 差异）；开销已量化；profiling 默认关闭；
- Supervisor PASS。

## 4. 输出

- `outputs/phase4_10/p10_i1_equivalence/`；`docs/phase4_10/P10_I1_RESULTS.md`；`docs/agent_coordination/SUPERVISOR_REVIEW_P10_I1.md`。
