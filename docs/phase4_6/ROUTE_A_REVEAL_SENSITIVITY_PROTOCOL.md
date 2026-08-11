# Route A 预注册协议：Reveal 机制敏感性（信息价值量化）

更新日期：2026-08-04
状态：**PROTOCOL（待用户批准后执行）**；Phase 5 保持 CLOSED；正式 artifacts 只读

## 1. 动机与假设

Phase 4 正式实验与 Phase 4.5/4.6 分析一致表明：在当前揭示节奏（full reveal 固定 slot 16）下，所有可实现方法的 completion 相对 full-information lower bound 的 regret ≈17 slots，其中约 10.7 slots 是信息延迟、6.5 slots 是全信息下的调度效率差距。调度改进（W2）、风险门控（W3）、多场景前缀（H2b）均无增益。

路线 A 直接量化"信息揭示节奏"这一唯一被证明的瓶颈：

- H-A1（单调性）：completion 随 full-reveal slot 提前而单调下降；
- H-A2（信息差距归零）：当 full-reveal 足够早（≤8 slots），`partial_current_only` 的 completion 逼近该 corpus 自己的 full-information executable reference；
- H-A3（剩余差距为调度效率）：信息差距消失后，剩余 regret 与 full-info executable 相对 LB 的差距一致（≈6.5 slots 量级），与揭示节奏无关；
- H-A4（mode 交互）：揭示模式（随机/总量/分片/时间）对收益的影响远小于揭示节奏。

## 2. 新 corpus（不复用正式 test 集）

- 生成器与正式 Phase 4 完全一致（length 256、4 nodes、mean 2.0、std 1.5、max entry 8、Rear4GPU、同一 family 顺序与 variant 规则）；
- **新 base seeds：`(1042, 1142, 1242)`**（与正式 `(642, 742, 842)` 不同）；
- 45 条 sequence（5 family × 3 seed × 3 split），sequence_id/actual_seed 规则沿用协议（`actual_seed = base_seed + family*1_000_000 + split*10_000`）；
- 必须通过 `validate_fresh_sequence_digests`（与正式 45 条 digest 零交集）；
- 主评估只用 **test split（15 sequence = 5 family × 3 seed）**；每 sequence 20 coordinates（4 checkpoints × 5 modes），共 300 test coordinates。

## 3. Reveal 调度（预注册，6 档）

slot→stage 映射由新运行器控制：`stage(s) = min(s // stage_len, len(ratios)-1)`；ratio 序列严格递增、首 0 末 1（`DemandRevealProcess` 校验）。

| 档位 | stage_len | ratios | full-reveal slot | 含义 |
|---|---:|---|---:|---|
| S0（对照=冻结） | 4 | (0, .25, .5, .75, 1) | 16 | 正式节奏（复现 partial ≈20.6 即验证运行器） |
| S1 | 2 | (0, .25, .5, .75, 1) | 8 | 揭示提前一倍 |
| S2 | 2 | (0, .5, 1) | 4 | 更早更粗 |
| S3 | 1 | (0, 1) | 1 | 近乎即时 |
| S4 | 8 | (0, .25, .5, .75, 1) | 32 | 揭示推迟一倍（对照） |
| S5 | 4 | (0, .5, 1) | 8 | 同 slot 8 但阶段更粗 |

## 4. 方法（复用冻结原语，新运行器）

- `partial_current_only`：每 slot 用当前揭示 observation 直接调度（复用 W2 已验证的 direct 循环，等价性门 300/300 通过）；
- `wait_until_known`：full reveal 前等待，之后直接调度；
- `full_information_executable_reference`：slot 0 起使用 full observation 直接调度；
- `full_information_lower_bound`：每 coordinate 计算一次，与揭示无关；
- 候选/容量/checker 全部复用冻结代码（enumerate_candidates / pack_candidate_batch / commit_proposal）；不改 reveal 语义本身（只改调度节奏与 ratios）。

## 5. 统计与报告

- 主指标：completion mean（300 coordinates / 15 sequence 等权）；每 sequence 20 coordinates 等权聚合；
- 配对比较：S_i vs S0（partial），sequence-level paired bootstrap 10,000（seed 20260801），报告 95% CI；
- 副指标：regret vs LB、与 full-info executable 的差距、legality、timeout、E2E（partial 行）；
- 分桶：family（5 桶）、seed（3 桶）、mode（5 桶）、checkpoint（4 桶）；结论桶 ≥5 条独立 sequence；
- 合法性要求：legality=100%、timeout=0，否则该档 fail closed。

## 6. 预期判定

- H-A1/H-A2/H-A3 均成立 → 确认信息延迟是唯一瓶颈，量化"提前揭示到 slot X 能节省多少 completion"；
- 若 H-A3 不成立（提前揭示后 partial 仍明显差于 full-info executable）→ 存在新的信息利用缺口，值得后续研究；
- 无论结果如何，不改变 H2=FAIL / Phase 5 CLOSED；不触碰正式 artifacts。

## 7. 产出

- `outputs/phase4_6/route_a_reveal/`（新运行器、新 corpus 物化、逐档结果 CSV/JSON）
- `docs/phase4_6/ROUTE_A_REVEAL_RESULTS.md`
- `docs/agent_coordination/SUPERVISOR_REVIEW_ROUTE_A.md`

## 8. 待用户批准

本协议为新预注册实验协议（新 corpus、新调度、新输出目录）。批准后执行：新运行器 → 等价性门（S0 复现正式 partial）→ 6 档调度全量运行 → 统计分析 → Supervisor 复核。
