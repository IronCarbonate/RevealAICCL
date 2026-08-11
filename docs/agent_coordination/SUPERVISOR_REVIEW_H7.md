# Supervisor Review — H7（自适应 Reveal Controller）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**H7 = FAIL / NO VETO**

## 1. 独立复核

1. 规则控制器在 validation 拟合、test 评估（300 coords），无 test 泄漏 ✓；
2. 控制器 300/300 选择 0.75，与固定 B75 完全相同（ΔJ=0.000 ms）✓；
3. oracle 每 episode 最优预算分布（135/119/46）显示异质性存在，但总价值仅 0.0014 ms（0.003%）✓；
4. legality 100%；未修改 production 代码；未强行训练复杂 controller ✓；
5. H2=FAIL、Phase 5 CLOSED 维持 ✓。

## 2. 判定

**H7 = FAIL / NO VETO**。自适应控制器相对最佳固定 profile 无任何 E2E 改善，且理论可榨取价值（oracle 上界）可忽略。按协议：

- **保留最佳固定 reveal profile = `partial_shards` @ 75%（full reveal slot 8）**；
- 不进入 H7 后续（无自适应训练）；
- 该 profile 作为 Phase 4.7 的可部署结论输出；
- 未来重开自适应需新协议与新语义证据。

## 3. Phase 4.7 总体结论

H5 PASS → H6 PASS（partial_shards）→ H7 FAIL（保留固定 profile）。研究链收敛：**在真实可实现语义下，以 partial_shards @ 75% 提前揭示（full reveal slot 8）替代当前 full-reveal-at-16，可在计入成本后带来约 8–14% 的 E2E 改善（H5 A4/A2 与 H6 B75 综合）**；自适应无额外价值。
