# P10-I1 结果：Reference Router 等价性

更新日期：2026-08-05
判定：**P10-I1 = PASS（17/17）**

## 1. 实现内容（L2-R reference substrate，`outputs/phase4_10/p10_1a_substrate/`）

- `reference_router.py`：gating Linear → 确定性 top-k（stable descending sort，lexicographic tie-break：高分优先、平局取小 expert index）→ bincount histogram（router 派生 traffic 即 ground truth）→ 分片 CUDA shard-ready 事件（每 shard kernel 完成后同步计时）；
- `p10_i1_tests.py`：17 项等价性测试。

## 2. 测试结果（2× V100，seed 20260805，B=256、D=16、E=4、k=1）

| # | 检查 | 结果 |
|---|---|---|
| T1 | router vs CPU oracle（index 完全一致、score 容差内） | PASS |
| T2 | 确定性（重复运行一致） | PASS |
| T3 | D0/D1 共享同一 router 流（相同 token/权重/top-k） | PASS |
| T3b | D0 full view / D1 75% view / 同一 stream digest | PASS |
| T4 | 无丢失（sum(hist)=B）、每 token 一个输出、专家 index 合法、traffic=router ground truth | PASS |
| T5 | shard-ready 为真实 CUDA 完成事件（非负、可测、累计有序） | PASS |
| T6 | no-leak 反事实（未揭示 token 扰动不改变已揭示视图） | PASS |
| T7 | profiling on/off 等价 | PASS |

修正过程（如实记录）：T4/T5 初版断言语义错误（唯一 expert 数 ≠ B；微小 kernel elapsed 为 0；elapsed_time 参数顺序反），已修正为正确语义（每 token 一个输出、累计完成时间有序、start.elapsed_time(end)）。

## 3. 关键结论

1. **traffic ground truth 由 router top-k 派生**（不强制匹配旧 proxy mapping）✓；
2. **D0/D1 使用相同 token、相同冻结权重、相同确定性 top-k**，唯一差异 = reveal 时机/粒度 ✓；
3. **shard readiness 来自真实 CUDA kernel 完成事件**（非 CPU 预标签）✓；
4. **确定性 lexicographic tie-break** 已实现并被 oracle 验证 ✓；
5. 命名保持 **L2-R reference**，不称生产 router ✓。

## 4. 约束

- 未运行 pilot/formal test；未生成/查看正式 test 结果；未实现真实 GEMM/combine；未用 Triton；未改 profile；未进 DeepEP/L3；未创建额外 Subagent。
