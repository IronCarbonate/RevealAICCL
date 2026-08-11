# Phase 4.6 W3：风险检测与有限启用 gate 评估

更新日期：2026-08-04
数据：冻结正式 artifacts（只读），`partial_current_only` vs `wait_until_known` 逐坐标配对
判定：**gate 无操作价值（空转）**——该 corpus 中不存在"等待更优"的 episode 子群

## 1. 预注册设计

- 规则族（先验定义）：`mode` / `checkpoint` / `family` / `mode×checkpoint`；
- 拟合：base seeds {642, 742}（10 条 sequence，200 coordinates）上各桶 `benefit = wait_completion − partial_completion`，桶均值 > 0 → 该桶启用"提前行动（act）"，否则 "wait"；
- 评估：留出 seed 842（5 条 sequence，100 coordinates），完全无样本内选择。

## 2. 结果

### 2.1 提前行动 vs 等待（全 corpus，300 coordinates）

| 指标 | 值 |
|---|---:|
| benefit mean（wait − partial） | +5.27 slots |
| benefit median | +4.00 |
| benefit > 0 比例 | **99.0%** |
| partial wasted_executed_actions | mean 0.00，max 0，**300/300 episode 为 0** |

### 2.2 分桶（benefit，slots）

| 分桶 | 值 |
|---|---|
| mode | random_entries 5.03 / source_totals_first 5.27 / source_dest_totals_first 4.65 / partial_shards 6.25 / time_based_arrival 5.15（5/5 正） |
| checkpoint | 4.72 / 5.59 / 5.27 / 5.51（4/4 正） |
| family | 4.02–7.95（5/5 正） |
| seed | 4.97 / 5.00 / 5.84（3/3 正） |

### 2.3 留出 seed 842 上的 gate 表现

| 规则 | act 桶数 | 选择 act 的 coordinates | completion mean |
|---|---|---:|---:|
| always-act（=Partial） | — | — | 20.27 |
| always-wait | — | — | 25.24 |
| rule[mode] | 5/5 | 100/100 | 20.27 |
| rule[checkpoint] | 4/4 | 100/100 | 20.27 |
| rule[family] | 5/5 | 100/100 | 20.27 |
| rule[mode×checkpoint] | 20/20 | 100/100 | 20.27 |

所有预注册规则都在留出集上选择"提前行动"（100/100），gate 退化为 always-act = `partial_current_only`。

## 3. 结论

1. **该冻结 corpus 中不存在"等待更优"的 episode 子群**：提前行动在 99% 坐标上严格更好，且从不产生 wasted actions（合法动作全为严格 progress）。
2. 风险检测/有限启用 gate **无操作价值**——不存在需要 gate 关闭的"高风险"子群。
3. 与 W2 一致：当前观测调度（Partial）已是该 reveal/容量语义下可实现的前沿；剩余 regret（vs LB 17.3 slots）由信息延迟（~10.7）与全信息下的调度效率差距（~6.5）构成，后者需要完整信息才可触及，超出"部分揭示 + 冻结语义"的边界。
4. 象限 2 路线在本 corpus 上的系统性结论：**静态预计算（W1）等价可用但无 completion 收益；调度改进（W2）与风险门控（W3）均无价值**。进一步的 completion 收益只能来自改变 reveal 机制或评估目标（超出 Phase 4 冻结协议范围），或转向其他平台/语义验证。

## 4. 产物与约束

- `outputs/phase4_6/w3_risk_gate/w3_risk_gate.py`、`w3_risk_gate.json`
- 未修改 production 代码；正式 artifacts 只读；legality 100%；timeout 0；Phase 5 CLOSED。
