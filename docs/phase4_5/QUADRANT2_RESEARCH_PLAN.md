# Phase 4.6 立项：象限 2 —— 准备型与风险门控的当前观测调度

更新日期：2026-08-04
状态：**PLAN（待用户批准工作流后实施）**；Phase 5 保持 CLOSED

## 1. 立项依据（H2a / H2b 结论）

- H2a = PASS（条件性）：robust E2E 高耗时是实现性问题，存在不改变语义的 7× 级加速路径。
- H2b = FAIL：robust prefix 相对 `partial_current_only` 的 scheduling-only 收益仅 +0.11 slots（CI [0.087, 0.140]），动作集合重合 98.5%，K=8 无决策价值，收益无法覆盖任何可信规划预算。
- 四象限裁决：**H2a PASS / H2b FAIL → 象限 2**：停止多场景 robust prefix 优化，转向 anticipatory preparation、risk detection、current-observation-only online scheduling、topology/static precomputation、candidate preparation、relay/path ranking、高风险 episode 有限启用 gate。

## 2. W0 只读 regret 审计（已完成，正式 artifacts）

| 方法 | completion | regret vs LB | first_action | residual repair |
|---|---:|---:|---:|---:|
| full_information_lower_bound | 3.35 | 0.0 | 0.0 | 0.0 |
| full_information_executable_reference | 9.88 | 6.53 | 0.0 | 0.23 |
| partial_current_only | 20.61 | 17.26 | 4.11 | 7.22 |
| scenario_robust_prefix | 20.49 | 17.14 | 4.11 | 7.23 |
| wait_until_known | 25.88 | 22.53 | 16.0 | 23.95 |
| 三类 point plans | ≈20.48–20.49 | ≈17.13 | 4.11 | 7.21–7.23 |

分解：

- 信息延迟差距 ≈ 20.6 − 9.9 = **10.7 slots**（部分揭示下所有可实现方法都无法达到全信息可执行参照）；
- 调度效率差距 ≈ 9.9 − 3.35 = **6.5 slots**（即使全信息，当前 direct scheduler 也离 provable 下界有距离）；
- 所有 ordinary 方法首动作均在第 4 slot、reveal 后修复动作 ≈7.2，说明**可行收益在 reveal 后的调度质量与信息利用，不在多场景前缀**。

## 3. 目标与边界

目标：在不预测未来、不做在线多场景重复求解的前提下，缩小当前观测调度相对 full-information 下界的 regret，并给出 bounded-regret 证据。

硬边界（不变）：

- 不修改 feasibility checker / commit_proposal；legality 必须 100%；
- 不修改 reveal semantics、observation API、seeds、corpus；
- 不关闭 fail-closed；不增加 timeout；
- 不打开 Phase 5；不改正式 artifacts；新实验一律新目录。

## 4. 研究问题

- RQ1（准备价值）：固定拓扑的静态预计算（canonical paths、edge/group units、全距离、候选池）能否在不改变动作语义的前提下降低每 slot 调度开销并改善 completion？
- RQ2（调度质量）：只使用当前已揭示 observation 的确定性调度（改进 tie-break / lookahead / packing）能否把 completion 从 ≈20.6 向 full-info executable（9.9）方向压缩？
- RQ3（风险门控）：能否用正式 artifacts 可观测特征（mode、checkpoint、已揭示 demand 比例、组/热点负载不确定性代理）识别"提前行动有价值"或"等待更优"的 episode 子群，并用有限 gate 只在这些子群启用准备型动作？
- RQ4（bounded regret）：任何新方法在冻结 test corpus 上的 completion regret 与 reveal 恢复时间是否可测、稳定且跨 seed/family。

## 5. 工作流（每项实施前需用户批准）

### W1：静态预计算与候选准备（无语义变化）

- 对 Rear4GPU 拓扑离线预计算 canonical shortest-path、edge/group atomic units、all-pairs distance、候选枚举池；
- 评估对 `partial_current_only`/direct scheduler 的替换成本与 completion/E2E 影响；
- 产物：`outputs/phase4_6/w1_static_precompute/`；评估：同一 corpus、同 seed、legality 100%、completion 不劣化。

### W2：当前观测确定性调度改进

- 在保持 (ordinal, edge) 语义与 checker 不变下，改进 direct scheduler 的确定性 tie-break / 单步 lookahead / batch packing；
- 评估目标：completion 向 9.9（full-info executable）方向压缩，且 E2E 保持在 baseline 量级（≤1.5×）；
- 产物：`outputs/phase4_6/w2_scheduler/`；配对比较 vs `partial_current_only`（sequence-level CI，seed/family 稳定）。

### W3：风险检测与有限启用 gate

- 用正式 episode/sequence 数据建立 episode 特征→收益的映射（分桶 ≥5 条独立 sequence）；
- 定义高风险/高价值启用条件（如已揭示 demand 密度、组负载不确定性、预期等待）；
- 只在高价值子群启用准备型动作，其余保持 partial 行为；
- 产物：`outputs/phase4_6/w3_risk_gate/`；预注册启用规则，禁止事后挑桶。

### W4：bounded-regret 评估框架

- 统一报告：completion regret vs LB、regret vs full-info executable、reveal 恢复时间（reveal 后完成所需 slots）、legality、timeout、E2E；
- 主 corpus 为冻结 Phase 4 的 15 条 test sequences（45 条 corpus 划分不变），新输出目录；
- 产物：`outputs/phase4_6/w4_regret/`、`docs/phase4_6/BOUNDED_REGRET_EVIDENCE.md`。

## 6. 预注册评估标准（任一工作流的 Gate）

1. legality = 100%，checker 未修改；
2. discrete/wall timeout 率不高于 `partial_current_only`；
3. sequence-level paired completion 不劣化（CI upper < 0 则 FAIL），改善需跨 ≥3 seed 与 ≥4/5 family；
4. E2E ≤ 1.5× `partial_current_only` baseline（除非证明条件性收益）；
5. 结论基于 ≥5 条独立 sequence 的桶；禁止行级当独立样本、禁止只挑正向桶；
6. Supervisor 独立复核。

## 7. 冻结与新增

- 冻结不变：corpus（45 sequence / 15 test）、checker、reveal、seeds、Phase 4 正式 artifacts（只读）；
- 新增：`outputs/phase4_6/`、`docs/phase4_6/`、脚本与评估代码（新文件，不改 production planner 语义）；
- 若涉及生产文件修改，必须单独 RED→最小实现→全量回归→Supervisor 准入，且先获用户批准。

## 8. 审批点与停止条件

- 本计划为立项文件；**每个工作流（W1–W4）实施前需用户单独批准**；
- 任一工作流违反第 6 节标准即停止并回退；
- 完成 W4 后由 Supervisor 输出 bounded-regret 证据审查，再决定是否形成新 H2 协议。
