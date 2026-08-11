# Phase 4.6 总结（象限 2 首批工作流 W1–W3）

更新日期：2026-08-04

## 背景

H2a = PASS（条件性）；H2b = FAIL；四象限 = **H2a PASS / H2b FAIL → 象限 2**（准备型 + 风险门控 + 当前观测调度）。用户批准 W1+W2，随后批准 W3。

## 结果汇总

| 工作流 | 目标 | 结果 | 判定 |
|---|---|---|---|
| W1 静态预计算 | 拓扑路径/容量/距离预计算 | 12/12 OD 与冻结代码逐位等价；查询加速 ~302× | PASS（等价性）；对 completion 无影响，E2E 节省有限 |
| W2 调度改进 | 排序/lookahead 压缩 completion | distance/headroom 与 Partial 完全相同；lookahead +0.030 slots（CI 跨 0 不显著）且慢 65%；等价性门 300/300 | 无有效收益 |
| W3 风险 gate | 只在有价值子群启用提前行动 | 提前行动 99% 坐标严格更优（+5.27 slots），wasted=0；全部预注册规则在留出 seed 上 100% 选择 act | gate 空转，无操作价值 |

## 总体结论

在冻结的 Phase 4 语义（candidate 合法性、checker、reveal 节奏、容量）下：

1. **当前观测调度（`partial_current_only`）已是可实现前沿**——它 ≈ robust prefix 的 completion（20.61 vs 20.49），且 E2E 便宜 10 倍；
2. 调度改进、静态预计算、风险门控均无法带来 completion 增益；
3. 剩余 regret（vs LB ≈17.3 slots）中 ~10.7 slots 是信息延迟（full reveal 前无法行动到前沿），~6.5 slots 是全信息下 direct scheduler 与 provable LB 的效率差距——两者都要求改变 reveal 或评估语义，超出本协议边界；
4. 正式结论维持：H2 = FAIL；Phase 5 = CLOSED；不实施任何优化重评。

## 建议的后续研究路线（供用户决策，均需新协议）

- 路线 A：研究 reveal 机制本身（更早/更密集揭示）对 completion 的敏感度——需新协议与新 corpus 划分，不得复用正式 test 集；
- 路线 B：在保持当前 reveal 下，以"E2E/成本"为目标（而非 completion）评估 Partial 路线的部署化收益；
- 路线 C：停止本 proxy 的进一步调度研究，转向真实 AICCL 部署语义验证（legacy 外推禁止，需全新验证协议）。

## 产物清单

- `docs/phase4_6/W1W2_EVALUATION.md`、`docs/phase4_6/W3_RISK_GATE.md`（本总结）
- `outputs/phase4_6/w1_static_precompute/`、`w2_scheduler/`、`w3_risk_gate/`
- 账本/风险登记更新；全部同步服务器。
