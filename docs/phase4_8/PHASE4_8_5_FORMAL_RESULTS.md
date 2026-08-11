# Phase 4.8-5：正式 Real-Deployment Test（Gate D1）

更新日期：2026-08-04
判定：**D1 = PASS（L1 单机高保真）**

> **Phase R0 provenance note（2026-08-10）**：本文件记录的派生 L1 汇总仍在，
> 但当前仓库、服务器与上传归档均未找到原始 L1 raw jobs。L1 raw artifact
> 已标 LOST，禁止重造冒充；因此本结果不再具有完整 raw-level provenance。

## 1. 执行

- 正式 test corpus：`(3042, 3142, 3242)`（与 H2/Route A/H5-H7 全部 digest 零重合；universe `d1daf2fa...`）；
- 15 test sequences × 20 coordinates = 300 jobs/臂；D0/D1/D2 全量运行（PROFILE_ON，warm-up 后）；
- 环境：1× RTX 2080 Ti / torch 2.8.0+cu128 / NCCL 2.27.3 / numpy 2.3.2 / L1 单机。

## 2. 主结果（300 jobs/臂）

| 指标 | D0 | D1 | D2（上界） |
|---|---:|---:|---:|
| completion mean | 20.34 | **13.91** | 9.65 |
| E2E wall mean（µs） | 55195 | **44271** | 47238 |
| throughput（jobs/s） | 18.12 | **22.59** | 21.17 |
| scheduler（µs/job） | 4307 | 3335 | 3166 |
| GPU busy（µs/job） | 1650 | 1053 | 1031 |
| control cost（µs/job） | 100.7 | 68.8 | 47.8 |
| legality | 100% | 100% | 100% |
| timeout | 0 | 0 | 0 |

## 3. 配对统计（D1 vs D0）

| 指标 | mean | 95% CI | 说明 |
|---|---:|---:|---|
| ΔE2E（µs） | **+10,953（≈10.95 ms）** | [+3,598, +23,148] | 正值 = D1 更好 |
| completion Δ | +6.43 slots | [+5.68, +7.12] | |

- seed 方向：3/3 正（3042 +21.9ms、3142 +6.0ms、3242 +4.9ms）；
- family 方向：**4/5 正**（rare_shock +6.1、regime +30.6、same_moments +10.1、stochastic +8.9；hotspot_random_walk −1.1ms 为负）；
- read-back：3 臂各 300 jobs 完整，hash/integrity 一致。

## 4. D1 PASS 判据核对

| 判据 | 结果 |
|---|---|
| 1. ΔE2E > 0 | PASS（+10.95 ms） |
| 2. paired 95% CI lower > 0 | PASS（+3.60 ms） |
| 3. ≥3 seed 或等价 workload group | PASS（3/3） |
| 4. ≥4/5 family 正向或有预注册边界 | PASS（4/5；hotspot_random_walk 为负，作为适用边界注明） |
| 5. completion 改善仍存在 | PASS（+6.43 slots） |
| 6. 全部 reveal/control/sync 成本已计入 | PASS（control 实测计入；sync 单 rank N/A） |
| 7. 不降低吞吐 | PASS（+24.7%） |
| 8. GEMM stall 不显著增 | PASS（GPU busy 下降 36%） |
| 9. collective contention 不显著增 | 单 rank 不可测（S；不构成增加证据） |
| 10. memory 在预算内 | PASS（worker RSS 721MB，远低于 40GB） |
| 11. legality 100% | PASS |
| 12. timeout 不增 | PASS（0） |
| 13. 可 read-back 重算 | PASS |
| 14. Supervisor PASS | 见 `SUPERVISOR_REVIEW_PHASE4_8_D1.md` |

## 5. 结论

**D1 = PASS（L1 单机高保真）**：`partial_shards @ 75%、full reveal checkpoint 8` 相对当前部署 baseline（默认 reveal、checkpoint 16），在计入实测 reveal/control/GPU 成本后，E2E 改善约 **10.95 ms/job（约 20%）**、completion +6.43 slots、吞吐 +24.7%，legality 100%、无 timeout。proxy（H5–H7）→ L1 真实时间（pilot + formal）证据链完整闭合。

## 6. 边界与限制（如实声明）

1. 本结论限于 **L1 单机高保真**（合成 router/GEMM、单 rank）；**不得声称 L2/L3 多节点部署收益**；
2. `hotspot_random_walk` family 的 E2E 为负（−1.1ms）——适用边界注明；
3. NCCL/DeepEP contention 未测（S），正式多节点验证需 L2/L3 硬件；
4. 合成 GEMM 不代表生产 MoE；真实模型/路由验证需生产路径（另行立项）。
