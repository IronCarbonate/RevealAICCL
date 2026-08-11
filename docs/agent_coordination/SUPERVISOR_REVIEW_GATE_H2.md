# Supervisor Review — Gate H2（正式复核）

更新日期：2026-08-04

## 复核范围

对 `outputs/phase4_early_planning` 正式八产物做独立复核，不读取任何 intermediate 缓存，仅使用冻结代码与正式产物。

## 独立验证动作

1. 进程/运行通道：`formal.exit=0`，`formal.start`/`formal.end` 记录完整，无 staging 残留，destination 由流水线原子发布。
2. 八产物精确文件集：8/8，无额外文件。
3. exact 行数核对：validation 9,600 / episode 2,700 / sequence 135 / events 147,690 / timing 21,600，与 manifest `artifact_row_counts` 一致。
4. manifest 标志：`integrity_complete=True`、`evidence_complete=True`、`data_status=FAIL`、`gate_status=PENDING_SUPERVISOR`。
5. 独立 `read_back_artifacts(require_final=True)`：artifact universe、schema、logical/scientific hash、summary hash、H1 digest、row digest、raw→episode→sequence→conditions→summary 重算链全部一致（函数对任何不一致抛异常，本次无异常）。
6. 环境与代码 hash：Python `0c05a22b...`、pip-freeze `6f27b26b...`、源码 `696E75BD...`、协议 `4246D661...`、launcher `032a702c...` 均与准入记录一致。

## 条件复核结果

- conditions 1–8：`1=F, 2=T, 3=F, 4=T, 5=T, 6=F, 7=T, 8=T`。
- `failed_conditions=[1,3,6]`，`insufficient_conditions=[]`。
- 关键数字复核：E2E Δ（robust−comparator）Partial −938.58（CI [−992.59, −896.65]）、Wait −926.66（CI [−976.98, −887.96]）；CVaR95 Δ 均 < 0；seed 0/3 正、family 5/5 负；legality 全 1.0；timeout 率全 0.0；ESS=15。

## Gate H2 裁决

**FAIL**（非 HOLD、非 VETO）。

- conditions 1/3/6 有明确反例证据，按协议"任一明确 FAIL 则 data Gate FAIL 优先于 tail/environment HOLD"。
- 未发现信息隔离、legality、checker、artifact 完整性或流程边界违规；因此不升级为 Supervisor VETO，也不存在"宣称 PASS 绕过边界"问题。
- Condition 9（Supervisor 独立复核）：**FAIL / NO VETO**——复核结论与 data Gate 一致。

## 对 Phase 5 的影响

Phase 5 Gate 保持 **CLOSED**。禁止进入 rolling configuration / Pareto frontier；robust prefix 的 E2E 表现当前显著劣于被动基线（在线 overhead 主导），任何后续研究须先解决 synthesis/replan overhead 并重新冻结配置后另行评估。
