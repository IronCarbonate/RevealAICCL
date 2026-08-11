# Supervisor Review — Phase 4.11 Claims Matrix

更新日期：2026-08-06
审查人：Supervisor（Project Director）
判定：**Phase 4.11 Claims Matrix = PASS / NO VETO**

## 1. 独立复核

1. **汇总完整性**：Phase 3B → Phase 4.10 共 20 个 Gate/实验条目，判定与关键数字与原始文档逐一一致（H1/H2/H2a/H2b、Route A、H5–H7、W1–W3、L1/L2 formal、P10-R0→SF0-B、Phase 4.10-F）✓；
2. **Claims-to-Evidence Matrix**：C1–C20 每条均映射到 ≥1 文档 + ≥1 artifact + ≥1 统计量；corpus 隔离、配对统计、真实计时口径一致 ✓；
3. **四层区分**：L1 / L2-S / L2-R 正确性 / L2-R infeasibility 严格分立；无跨层混用 ✓；
4. **负结果保留**：H1、H2、H2b、H7、W2、W3、A5、hotspot_random_walk、pilot E2E、P10-F0-v1、SF0-B 全部列入且未被删除/弱化 ✓；
5. **Fail-closed 清单**：10 类过度/生产化/L3 表述全部排除（生产 MoE、L3/DeepEP/RDMA 已验证、L2-R E2E 收益、scheduler <336µs 可认证、formal 通过、自适应/预测有价值、排序/门控有收益、跨层外推、completion=E2E、选择性删负结果）✓；
6. **Paper draft**：outline/abstract/introduction/evaluation/discussion/limitations 齐备；每条数字可回溯到矩阵；limitations 覆盖 L3、reference router、scheduler floor、workload 特定性、控制消息 fabric、completion/E2E 分离 ✓；
7. **Artifact overview/reproduction/hardware/expected results**：分阶段命令、硬件矩阵（L1/L2-S/L2-R/L3）、期望值表完整；重跑纪律与禁止项明确 ✓；
8. **禁止项复核**：未运行新实验；未修改 scheduler/router/reveal profile；未实现 memoization/vectorization；未重开 P10-1；未选择性删除负结果；未称 L2-R 为生产 MoE；未声称 L3/DeepEP/RDMA 已验证；未创建额外 Subagent ✓。

## 2. 判定

**Phase 4.11 Claims Matrix = PASS / NO VETO**：

1. 全部论文可用 claim 具备 artifact + 统计支持；
2. 四层结论框架与负结果完整；
3. 过度/生产化/L3 表述全部 fail closed；
4. 论文草案与 artifact 文档可直接作为后续成文基础（需保持本矩阵与禁止项不变）。

## 3. 结论

允许提交用户审核 Phase 4.11。任何后续论文写作必须继续遵守本矩阵：不得新增未经 artifact 支持的 claim；
不得恢复被 fail-closed 的表述；论文图表只能由既有数值重绘。
