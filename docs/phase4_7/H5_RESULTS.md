# H5 结果：可实现早期信息有 E2E 价值

更新日期：2026-08-04
协议：`docs/phase4_7/H5_PROTOCOL.md`（FROZEN）
判定：**H5 = PASS**（A2/A3/A4；A5 单独 FAIL）

## 1. 执行

- 新 corpus：base seeds `(2042, 2142, 2242)`，45 sequence，与 H2 正式 45 条、Route A 45 条 **digest 零重合**（`universe_digest=3d69637a...`）；
- 运行：validation 15 sequence 先行（未用于选择），test 15 sequence × 20 coordinates = 300/臂，7 臂，legality 100%、无 timeout；
- 成本：`cost_params.json`——histogram 336 ns/token、matrix 613 ns/entry、控制消息 8.77 µs RTT（均本机实测）；collective 延迟/带宽为文献级假设（标注 assumed）；
- J（E2E）= completion×1ms + scheduler（固定中位数） + compute + control + sync + blocking + pipeline。

## 2. 主结果（test，15 sequence × 20 coords）

| 臂 | completion | J（ms） | ΔJ vs A1 | 95% CI | seq 正 | family 正 | seed 正 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 当前 full-reveal baseline | 20.46 | 60.92 | — | — | — | — | — |
| A2 coarse early reveal | 14.34 | 54.85 | **+6.06** | [+5.50, +6.59] | 15/15 | 5/5 | 3/3 |
| A3 A2 + 全局直方图同步 | 14.34 | 54.94 | **+5.98** | [+5.42, +6.51] | 15/15 | 5/5 | 3/3 |
| A4 rank-local streaming | 11.16 | 51.69 | **+9.22** | [+8.26, +10.13] | 15/15 | 5/5 | 3/3 |
| A5 group-level reveal | 20.46 | 61.04 | **−0.13** | [−0.13, −0.13] | 0/15 | 0/5 | 0/3 |
| A6 full-information reference | 10.16 | 50.61 | +10.31 | 上界 | — | — | — |
| A7 cost-free reveal | 11.16 | 51.61 | +9.31 | 仅 oracle 分析 | — | — | — |

## 3. 关键证据

1. **可实现早期信息有真实 E2E 价值**：rank-local 流式揭示（A4）比当前 full-reveal baseline 的 J 改善 **9.2 ms（约 15%）**，coarse early（A2）改善 6.1 ms；全部 CI lower>0、15/15 sequence、5/5 family、3/3 seed。
2. **成本几乎可忽略**：A4 的可实现成本（compute 0.01 ms + control 0.07 ms）仅占收益的 ~1%；A4 vs A7（cost-free）只差 0.08 ms——即使成本假设放大 10 倍，A2/A4 仍显著为正。
3. **全局同步信息不值得**：A5（bandwidth-group 全局聚合）不改变 completion（aggregate 不解锁动作），仅增加同步成本 → J 变差（−0.13 ms，所有序列为负）。这验证了 Phase 4.7-0 的语义判断：**本地流式精确信息有价值，全局聚合信息在当前语义下无调度价值**。
4. A3（A2+全局直方图）比 A2 略差（−0.08 ms）：同步成本 > 直方图带来的任何额外价值（当前 partial 不使用直方图做决策）。

## 4. H5 PASS 判据核对

| 判据 | A2/A3/A4 |
|---|---|
| 1. 相对 A1，E2E J 改善 > 0 | PASS（+6.0/+6.0/+9.2 ms） |
| 2. sequence-level paired CI lower > 0 | PASS |
| 3. ≥3 seed | PASS（3/3） |
| 4. ≥4/5 family 正向 | PASS（5/5） |
| 5. legality 100% | PASS |
| 6. timeout 不增加 | PASS（max completion 22/22/19 vs 30） |
| 7. 收益非免费 oracle | PASS（A4 成本已计入；A7 仅 oracle） |
| 8. 存在非平凡预算区间 | PASS（3–4 次 reveal 事件、~0.07 ms 控制成本 vs 6–9 ms 收益） |
| 9. Supervisor PASS | 见 `SUPERVISOR_REVIEW_H5.md` |

## 5. 结论

**H5 = PASS**：在真实可实现的信息获取语义下（rank-local 流式/粗粒度 reveal、成本计入 E2E），提前揭示已到达 token 的信息带来显著且稳定的 E2E 收益；全局聚合信息（group-level）无价值。允许进入 H6（固定预算下的选择性 reveal）。Route A 的 proxy 结论在此获得可实现性验证。

## 6. 约束

- 未修改 production 代码/checker；正式 artifacts 只读；新 corpus/新输出目录；
- 成本参数中 collective 项为假设值（A5 为负对假设不敏感；A2/A4 无同步，稳健）；
- H2=FAIL、Phase 5 CLOSED 维持。
