# Supervisor Review — H6（固定预算选择性 Reveal）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**H6 = PASS / NO VETO**（partial_shards）

## 1. 独立复核

1. 使用 H5 冻结新 corpus test split（300 coords/臂），无新数据选择 ✓；
2. 5 个预注册选择器 × 3 预算，成本计入 J（control 实测、sync 仅对全局 dest 模式）✓；
3. partial_shards vs random：B25 +0.604（CI [0.383, 0.877]）、B50 +0.807（[0.683, 0.957]）、B75 +0.573（[0.453, 0.703]），14-15/15 seq、5/5 family、3/3 seed ✓；
4. entry 级选择器与 random 完全相同（source_totals Δ=0），说明 entry 揭示顺序无调度价值，与 H2b/W2 一致 ✓；
5. legality 100%、无 timeout；selector 成本 ~1/30 收益 ✓；
6. 预算单位差异（token vs entry）已如实标注，未掩盖 ✓；
7. 未修改 production 代码；H2=FAIL、Phase 5 CLOSED 维持 ✓。

## 2. 判定

**H6 = PASS / NO VETO**。最佳固定 reveal profile = **partial_shards**（token 级分片）。允许进入 H7（自适应 reveal controller），前提：

- H7 以 partial_shards 为固定基础；
- 先规则/contextual bandit，禁止直接大型 RL；
- 控制开销必须低于收益；不得频繁震荡；
- 同单位预算对比作为补充实验在 H7 前或并行完成。
