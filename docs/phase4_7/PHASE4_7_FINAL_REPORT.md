# Phase 4.7 最终报告（Formal Closure）

更新日期：2026-08-04
状态：**FROZEN**

## 1. 正式冻结结论

| Gate | 判定 |
|---|---|
| R0（真实可实现 reveal） | **PASS**（有条件） |
| H5（可实现早期信息 E2E 价值） | **PASS** |
| H6（固定预算选择性 reveal） | **PASS** |
| H7（自适应 reveal controller） | **FAIL** |
| H2（早期规划） | **FAIL**（维持） |
| Phase 5 | **CLOSED**（维持） |

## 2. 最终固定方案（部署建议）

| 项 | 固定值 |
|---|---|
| reveal mode | `partial_shards`（token 级分片） |
| reveal budget（wave-1） | 75% |
| full reveal checkpoint | slot 8（原 slot 16） |
| fast scheduler | `partial_current_only` |
| adaptive reveal controller | disabled |
| robust prefix | disabled |
| historical predictor | disabled |
| risk gate | disabled |

## 3. 各 Gate 关键证据

### R0 = PASS（`SUPERVISOR_REVIEW_PHASE4_7_0.md`）

rank-local per-source 计数、local top-k、本地直方图、shard 计数满足五条件（比完整矩阵早、比全局同步便宜、有调度语义、无未来真值、成本可建模）；全局量计同步成本；Route A 的"免费提前精确 entry"为 proxy 语义，不是部署事实。

### H5 = PASS（`H5_RESULTS.md`）

新 corpus（2042/2142/2242）7 臂 × 300 coords，成本计入 E2E：

| 臂 | completion | J（ms） | ΔJ vs A1（CI） |
|---|---:|---:|---:|
| A1 现状 full-reveal-16 | 20.46 | 60.92 | — |
| A2 coarse early | 14.34 | 54.85 | +6.06 [5.50, 6.59] |
| A3 A2+全局直方图同步 | 14.34 | 54.94 | +5.98 [5.42, 6.51] |
| A4 rank-local streaming | 11.16 | 51.69 | +9.22 [8.26, 10.13] |
| A5 group-level 全局聚合 | 20.46 | 61.04 | −0.13（全负） |
| A6 full-information reference | 10.16 | 50.61 | 上界 |
| A7 cost-free reveal | 11.16 | 51.61 | oracle 分析 |

15/15 seq、5/5 family、3/3 seed 正向；legality 100%、无 timeout；可实现成本 ~0.08ms（收益的 ~1%）。

### H6 = PASS（`H6_RESULTS.md`）

5 预注册选择器 × 3 预算 × 300 coords：**partial_shards 在所有预算下显著优于 random**（+0.60/+0.81/+0.57 ms，CI lower>0、14–15/15 seq、5/5 family、3/3 seed）；entry 级选择器与 random 无差异（source_totals Δ=0.000）。预算单位（token vs entry）差异已如实标注。

### H7 = FAIL（`H7_RESULTS.md`）

规则控制器（validation 拟合 → test 评估）退化为 300/300 选择 0.75，与固定 B75 完全等价（ΔJ=0.000 ms）；oracle 每 episode 最优相对 B75 仅好 **0.0014 ms**（0.003%）——自适应无可榨取价值。**保留固定 profile = partial_shards @ 75%。**

## 4. 可声称 / 不可声称

### 可声称（证据支持）

1. 在冻结 proxy 语义与可实现信息流下，把 full reveal 从 slot 16 提前到 slot 8、并以 partial_shards 分片揭示 75%，计入可实现成本后带来约 **8–14% E2E 改善**（H5 A2/A4 + H6 B75 综合）。
2. rank-local 流式/粗粒度揭示的成本（本机实测：直方图 336ns/token、控制消息 8.8µs）远低于收益；该结论对 collective 成本假设不敏感（A2/A4 无同步）。
3. 全局聚合信息（expert histogram / bandwidth-group）在当前调度语义下无调度价值（A5 为负）。
4. 自适应 reveal controller 在当前设置下无价值（oracle 上界 0.0014ms）。

### 不可声称（禁止）

1. 不得声称 H2/robust prefix 有效，或 Phase 5 可开放。
2. 不得把 Route A 或 H5 的 proxy 数字直接外推为真实部署性能（collective 同步/阻塞/带宽为假设值，须 Phase 4.8 实测）。
3. 不得声称"更早真值一定有用"是通用贡献——仅在本 proxy/语义下成立。
4. 不得把 completion-only 收益冒充 E2E 收益（所有结论基于计入成本的 J）。
5. 不得声称 partial_shards 的选择器优势在"严格相同 token 预算"下已证明（单位差异待补充实验）。
6. 不得外推到 legacy Torch decoder / 真实生产 AICCL，除非通过 Phase 4.8 部署验证。

## 5. 交付物

- `docs/phase4_7/`：PHASE4_7_FINAL_REPORT（本文件）、FINAL_DEPLOYMENT_RECOMMENDATION、EVIDENCE_CHAIN、NEGATIVE_RESULTS_SUMMARY、REPRODUCIBILITY_MANIFEST、H5/H6/H7 协议与结果、REALIZABLE_REVEAL_SEMANTICS、REVEAL_CAPABILITY_TABLE、REVEAL_COST_MODEL
- `docs/phase4_8/PHASE4_8_DRAFT_PROTOCOL.md`（Real-Deployment Validation 草案）
- `docs/agent_coordination/`：TASK_LEDGER_PHASE4_7、RISK_REGISTER_PHASE4_7、SUPERVISOR_REVIEW_PHASE4_7_0/H5/H6/H7/F、DECISION_LOG（追加）
- `outputs/phase4_7/`：h5_realizable_reveal/、h6_selective_reveal/、h7_adaptive_reveal/
