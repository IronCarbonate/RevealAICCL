# L2 最终报告（Phase 4.9-F）

更新日期：2026-08-05
判定：**L2-D1 = PASS / L2-F0 = PASS（read-back 通过）**

## 1. 冻结状态

H1=FAIL、Phase 3B=PASS、H2=FAIL、Phase 5=CLOSED、H5=PASS、H6=PASS、H7=FAIL、**L1-D1=PASS、L2-D1=PASS**；frozen profile = partial_shards @ 75%、checkpoint 8；fast scheduler = partial_current_only。

## 2. L2 正式结果（2× Tesla V100-SXM2-32GB，真实 NCCL 2-rank）

| 指标 | D0 | D1 | D2（上界） |
|---|---:|---:|---:|
| completion | 20.34 | **13.91** | 9.65 |
| E2E wall（µs） | 53,729 | **47,272** | 50,175 |
| 吞吐（jobs/s） | 18.61 | **21.15** | 19.93 |
| legality / timeout | 100% / 0 | 100% / 0 | 100% / 0 |

配对（D1 vs D0）：ΔE2E **+6,458 µs（CI [+3,409, +9,385]）**；completion Δ +6.43 slots（CI [+5.68, +7.12]）；seed 3/3 正；family 4/5 正。

## 3. 独立 read-back（通过）

- 10 项 L2 产物与 `hashes_l2.json` 全部匹配（hash_ok=true）；
- 从 `raw_jobs.json` 独立重算 job→sequence→condition→summary，与 `l2_final_summary.json`/`l2_condition_summary.json` **完全一致（0 差异）**；
- 注：原 `hashes.json` 因 torchrun 双 rank 并发写文件存在竞态；已从磁盘实际文件重新生成 `hashes_l2.json`/`integrity_manifest_l2.json`，科学数值不受影响。

## 4. 真实 NCCL 微基准（固化，证据 M）

allreduce：64B 63.8µs、128B 61.7µs、1KB 86.6µs；allgather：128B 121.6µs、1KB 136.2µs（mean；p95/p99/CI 见 `l2_collective_results.json`）。

## 5. hotspot_random_walk 负结果（如实报告）

family 级 ΔE2E：hotspot_random_walk **−1,741 µs（负）**；rare_shock +6,335、regime +6,320、same_moments +11,452、stochastic +9,917 µs（正）。即：在 hotspot_random_walk 上 D1 的 E2E 比 D0 差约 1.7ms。该 family 作为**预注册适用边界**明确记录，不掩盖、不剔除。

## 6. 结论

L2 单机多 GPU（真实 NCCL）下，frozen profile 相对 baseline 的 E2E 改善约 **6.5ms/job（约 12%）**，统计显著、跨 seed、4/5 family；结论稳健。允许将本结论作为 L2 级正式结果冻结；L3 与真实生产路径另行验证。

## 7. 产物

- `outputs/phase4_8/deployment_validation/`：l2_final_summary.json、l2_condition_summary.json、l2_collective_results.json、raw_jobs.json、hashes_l2.json、integrity_manifest_l2.json、read_back_report_l2.json
- 报告：本文件、L2_DEPLOYMENT_RECOMMENDATION、L2_PORTABILITY_ANALYSIS、L2_LIMITATIONS、L2_REPRODUCIBILITY_MANIFEST、PRODUCTION_PATH_BRIDGE_DRAFT_PROTOCOL
