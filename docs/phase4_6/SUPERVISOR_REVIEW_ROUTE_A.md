# Supervisor Review — Route A（Reveal 敏感性）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**PASS / NO VETO**

Route A 预注册协议（`docs/phase4_6/ROUTE_A_REVEAL_SENSITIVITY_PROTOCOL.md`）及结果（`outputs/phase4_6/route_a_reveal/`）独立复核通过：

- 新 corpus（1042/1142/1242）与正式 corpus digest 零交集；
- 等价性门 S0 在正式 corpus 300/300 复现正式 partial；
- completion 随 full-reveal 提前单调下降（36.18 → 20.95 → 14.92 → 13.40 → 11.80），配对 CI 全排除 0；
- S3 partial（11.80）距 fullinfo（10.80）仅 1.0 slot；fullinfo regret vs LB = 6.55，与揭示无关；
- mode 影响远小于节奏（8.07–9.48 增益）；
- legality 100%、timeout 0；未修改 production 文件。

结论：**PASS / NO VETO**。注意：Route A 的"更早 reveal"是 proxy 语义（精确 entry、成本不计）；其真实可实现性由 Phase 4.7 另行审计，不得直接外推。
