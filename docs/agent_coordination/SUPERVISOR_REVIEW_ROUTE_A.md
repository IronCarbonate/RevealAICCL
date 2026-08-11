# Supervisor Review — Route A（Reveal 敏感性）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**PASS / NO VETO**

## 1. 审查内容

`docs/phase4_6/ROUTE_A_REVEAL_SENSITIVITY_PROTOCOL.md` 的预注册设计及其结果 `outputs/phase4_6/route_a_reveal/route_a_results.json`、`docs/phase4_6/ROUTE_A_REVEAL_RESULTS.md`。

## 2. 独立复核

1. 新 corpus 与正式 corpus digest 零交集 ✓（脚本断言，未冲突）；
2. 等价性门：S0 在正式 corpus 上 300/300 复现正式 partial ✓（completion/first_action/legality/actions 0 差异）；
3. 主结果单调性：S4(32)=36.18、S0(16)=20.95、S1(8)=14.92、S2(4)=13.40、S3(1)=11.80，与 paired bootstrap CI（全部排除 0）一致 ✓；
4. 信息差距归零：S3 partial 11.80 vs fullinfo 10.80（+1.0）✓；
5. 调度效率差距：fullinfo regret vs LB = 6.55，跨 corpus 一致，与揭示无关 ✓；
6. mode 分桶：S0→S3 增益 8.07–9.48，模式间差异小 ✓；
7. 合法性：全部 episode legality=100%、timeout=0；未修改 production 文件；正式 artifacts 只读 ✓。

## 3. 判定

**PASS / NO VETO**。路线 A 的四项预注册假设全部成立，证据充分（300 coordinates、序列级配对 CI、跨 mode/family/seed 稳定）。结论可信：信息揭示节奏是 completion 的主导瓶颈；简单调度器在信息可用时接近最优；剩余差距为信息无关的调度效率问题。

## 4. 后续约束

- 任何基于该结论的研究（例如更优 reveal 设计、调度效率改进）需另立协议并经用户批准；
- H2=FAIL、Phase 5 CLOSED 维持；不得在现有协议上改判。
