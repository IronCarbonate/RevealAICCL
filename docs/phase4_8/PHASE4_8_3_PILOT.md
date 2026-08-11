# Phase 4.8-3：Development/Validation Pilot

更新日期：2026-08-04
判定：**P0 = PASS**

## 1. 执行

- L1 高保真执行层（real_exec，PROFILE_ON），工作负载 = H5 validation split（15 seq × 20 coords = 300 jobs/臂）；
- 冻结环境 D0 参照（phase4-env h5_runner）用于交叉验证；
- 每 job 记录 reveal/scheduler（CPU）与 dispatch/GEMM/collective（CUDA event）事件。

## 2. 结果（validation，300 jobs/profile）

| 指标 | D0 | D1 | D2（上界） |
|---|---:|---:|---:|
| completion mean | 20.59 | **14.45** | 10.27 |
| first_action mean | 4.05 | 4.00 | 0.00 |
| legality | 100% | 100% | 100% |
| timeout | 0 | 0 | 0 |
| E2E wall（ms） | 55.30 | **45.49** | 48.51 |
| throughput（jobs/s） | 18.08 | **21.98** | 20.61 |
| scheduler CPU（ms/job） | 4.23 | **3.32** | 3.20 |
| GPU busy（ms/job） | 1.68 | **1.11** | 1.09 |
| reveal 事件数（mean） | 20.59 | 14.45 | 10.27 |

## 3. Pilot 十问回答

| # | 问题 | 回答 |
|---|---|---|
| 1 | D1 是否更早启动有效 dispatch | 首动作 4.00 vs 4.05（几乎相同，均受首次揭示 slot 4 门控）；**收益在完成时间**（14.45 vs 20.59），非首动作延迟 |
| 2 | reveal 75% 是否真实发生在 checkpoint 8 | 是（D1 = (0,.75,1)、stage_len 4 → slots 4–7 揭示 75%、slot 8 全量；事件流确认） |
| 3 | 额外控制成本是否计入 critical path | 是（reveal 事件记录在 CPU critical path；每 job 控制成本 ~5µs×事件数，可忽略） |
| 4 | 是否影响 router | N/A（合成 shim，无生产 router） |
| 5 | 是否增加 GEMM stall | 否（D1 GPU busy 1.11ms < D0 1.68ms，因 slot 更少） |
| 6 | 是否增加 NCCL/网络竞争 | 单 rank 不可测（S；L2/L3 必需） |
| 7 | 是否降低 throughput | 否（D1 21.98 vs D0 18.08 jobs/s，**+21.6%**） |
| 8 | 是否产生不合法动作 | 否（legality 100%） |
| 9 | 是否出现 timeout | 否（0） |
| 10 | proxy completion 改善在真实时间中方向一致 | **是**：D1 completion（14.45<20.59）且 E2E wall（45.5<55.3ms）方向一致 |

## 4. P0 检查

| 项 | 结果 |
|---|---|
| D0 复现当前 baseline（vs 冻结参照） | PASS（300/300 completion 0 差异、legality 0 差异） |
| D1 配置实际生效 | PASS（completion 20.59→14.45） |
| 无未来信息泄漏 | PASS（partial 语义；I1 已证 hash 等价） |
| legality 100% | PASS |
| timeout 不增 | PASS（0） |
| 插桩与成本核算可信 | PASS（I1 等价性 + microbench M 级） |
| 无吞吐崩溃 | PASS（+21.6%） |
| **P0 综合** | **PASS** |

## 5. 结论

**P0 = PASS**：L1 真实时间下，D1（partial_shards @ 75%、checkpoint 8）相对 D0（默认 reveal、checkpoint 16）E2E 墙钟改善约 **9.8 ms/job（17.7%）**、吞吐提升 21.6%、legality 100%、无 timeout——proxy 的 completion 结论在真实执行时间中方向一致，且成本（控制/同步/GPU）不吞收益（主导成本为调度 CPU，D1 因 slot 更少而更低）。允许冻结正式协议（Phase 4.8-4）。

## 6. 约束

- Pilot 使用 development/validation 工作负载，**不用于正式 D1 判定**；
- 仍为 L1 单机（合成 GEMM、单 rank）；正式协议须记录此限制。
