# H5 草案协议：可实现早期信息是否有 E2E 价值（DRAFT）

更新日期：2026-08-04
状态：**DRAFT（等待用户审核；R0 PASS 后正式化）**

## 1. 固定内容

- fast layer scheduler = **`partial_current_only`**（冻结为 fast-layer baseline，不改语义）；
- 不使用 robust prefix、历史预测、risk gate；
- 不修改 deterministic checker；
- 全部 reveal 收益计入成本（`docs/phase4_7/REVEAL_COST_MODEL.md`）；
- 新 corpus；不复用 H2 正式 test、Phase 4.6 W1–W3 test、Route A seeds 1042/1142/1242。

## 2. 新 corpus 计划（仅草案，R0 后按第 8 节正式生成）

- 生成器与 Phase 4/Route A 一致（length 256、4 nodes、mean 2.0、std 1.5、max 8、Rear4GPU、5 family）；
- 新 base seeds：**`(2042, 2142, 2242)`**（草案建议，正式化时冻结）；
- 45 sequence（5 family × 3 seed × 3 split），按完整 sequence 划分，同一 sequence 不跨 split；
- 正式 test 冻结前不查看；validation 先行；
- 必须记录：generator version、topology/config、manifest hash、zero-overlap check（对 H2 45 条 + Route A 45 条 digest）。

## 3. 比较臂（pre-registered）

1. current fixed full-reveal baseline（现状：full reveal 固定时点）；
2. coarse early reveal（rank-local source/top-k 计数，流式、无同步）；
3. coarse early + progressive refinement（本地→全局逐步细化，计同步）；
4. rank-local streaming reveal（每 token 计数即用）；
5. group-level reveal（全局 bandwidth-group 聚合，计同步）；
6. full-information reference（仅上界）；
7. cost-free reveal（仅 oracle 分析，不计成本）。

## 4. 记录指标

- completion mean/median/p95/p99/CVaR95；
- full-information regret；
- total E2E J（= T_completion + T_reveal_wait + T_reveal_control + T_sync + T_scheduler + T_execution）；
- reveal wait、control-message time、synchronization time、scheduler time、execution time；
- reveal information volume、reveal event count、marginal benefit per information unit；
- legality、timeout、pipeline interference。

## 5. H5 PASS 判据（草案）

1. 相对 current reveal baseline，**E2E（J）改善 > 0**（非 completion-only）；
2. sequence-level paired CI lower > 0；
3. ≥3 seed；
4. ≥4/5 family 正向或有预注册适用边界；
5. legality 100%；
6. timeout 不增加；
7. 收益不是免费 oracle 信息（成本已计入且收益仍正）；
8. 存在非平凡预算区间（B 在可行范围内）；
9. Supervisor PASS。

## 6. H5 FAIL 判定（草案）

- 计入成本后 E2E 无改善；或仅 completion 改善而 E2E 恶化；
- 收益仅来自免费 oracle 信息；
- legality/timeout 违反；
- 则停止 reveal-policy 扩展，在真实部署验证 / full-information scheduler efficiency 中重新选择。

## 7. 输出

- `docs/phase4_7/H5_PROTOCOL.md`（正式化）、`H5_RESULTS.md`；
- `outputs/phase4_7/h5_realizable_reveal/`（fixed Partial runner、capability/cost 模块、结果）；
- `docs/agent_coordination/SUPERVISOR_REVIEW_H5.md`。

## 8. 前置条件（R0 PASS 后）

1. 校准成本模型参数（profile 或文献值，标注来源）；
2. 冻结 reveal profiles（来自 capability table 的可实现项）；
3. 用户批准正式 corpus/seeds/protocol；
4. 生成新 corpus 并做 zero-overlap + read-back 校验；
5. validation 选择/确认，再冻结 test。
