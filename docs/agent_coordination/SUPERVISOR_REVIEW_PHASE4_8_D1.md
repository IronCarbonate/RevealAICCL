# Supervisor Review — Phase 4.8 D1（Deployment Validity）

更新日期：2026-08-04
审查人：Supervisor（Project Director）
判定：**D1 = PASS（L1）/ NO VETO**

## 1. 独立复核

1. 正式 corpus（3042/3142/3242）与全部先前 corpus digest 零重合；正式 test 冻结后一次运行 ✓；
2. D1 vs D0：ΔE2E +10.95ms（CI [+3.60, +23.15]ms lower>0）、completion +6.43 slots（CI [+5.68,+7.12]）、3/3 seed、4/5 family（hotspot 负，已注明适用边界）、吞吐 +24.7%、legality 100%、timeout 0 ✓；
3. 成本计入：control 实测（M）、GPU/调度实测、sync 单 rank N/A（S 标注）✓；
4. read-back：300 jobs/臂完整，integrity/hash 一致 ✓；
5. 未修改 production 代码；未开启 H1/H2/robust prefix/Phase 5；结论限定 L1 单机，未声称 L2/L3 ✓；
6. 正式 artifact 集（environment/protocol/final/job_sequence/timing/throughput/condition/hashes/integrity/run_command）存在且可重算 ✓。

## 2. 判定

**D1 = PASS（L1 单机高保真）/ NO VETO**。Phase 4.8 证据链闭合：proxy（H5 PASS、H6 PASS、H7 FAIL）→ L1 pilot（P0 PASS）→ L1 正式 test（D1 PASS）。最终可部署 profile = **partial_shards @ 75%、full reveal checkpoint 8、fast scheduler = partial_current_only（其余全关）**。

## 3. 边界（必须遵守）

1. 本 PASS 仅对 L1 单机有效；多节点 L2/L3 验证需新硬件与新协议；
2. hotspot_random_walk family 为负，作为适用边界记录；
3. 真实生产 MoE/router/DeepEP 路径验证需另行立项；
4. H2=FAIL、Phase 5 CLOSED 维持。
