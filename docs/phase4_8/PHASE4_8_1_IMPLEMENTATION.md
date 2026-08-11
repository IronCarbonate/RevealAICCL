# Phase 4.8-1：最小实现与计时插桩（Gate I1）

更新日期：2026-08-04
判定：**I1 = PASS**

## 1. 实现内容

`outputs/phase4_8/deployment_validation/`：

- `real_exec.py`：L1 高保真执行层
  - D0（当前部署 baseline）：坐标自身冻结 reveal mode、full reveal checkpoint 16；
  - D1（候选 profile）：partial_shards、75% budget、full reveal checkpoint 8；
  - D2（full-information reference）：slot 0 全揭示，仅上界；
  - scheduler = 冻结 partial_current_only 语义（未改）；
  - flag 控制计时（`PH4_8_PROFILE=1`，默认关闭）；CPU perf_counter + CUDA event；合成 dispatch/GEMM/collective 为真实 GPU kernel（标注 synthetic）。
- `i1_equivalence.py`：I1 等价性验证（开发集 = H5 corpus 前 3 条 test sequence）。

## 2. I1 结果（180 jobs = 3 seq × 20 coords × 3 profiles）

| 检查项 | 结果 |
|---|---|
| completion（PROFILE OFF vs ON） | 0 差异 |
| legality | 0 差异，100% |
| action_digest | 0 差异 |
| executed action 数 | 0 差异 |
| D0 vs 冻结 H5 A1（60 coords 交叉验证） | **0 差异**（运行器忠实复现冻结 baseline） |
| 插桩开销（ON vs OFF wall） | mean 16.1 ms/job、median 4.6 ms/job（含合成 GPU kernel + 同步；纯 recorder 开销远小于此） |
| 记录事件数（ON） | 9,679 |

## 3. 说明与约束

- 插桩默认关闭；ON/OFF 的行为 hash 完全一致（等价性已证）；
- 合成 dispatch/GEMM/collective 是真实 GPU kernel（torch 2.8.0+cu128 / RTX 2080 Ti），标注为 synthetic 执行负载，不代表生产 MoE 模型；
- D0 使用坐标自身冻结 mode（与 H5 A1 一致）——这是"当前部署 baseline"的忠实语义；
- 未修改 production 代码；未实现 checkpoint 8 之外的任何 profile；未运行 microbenchmark/pilot/正式实验。

## 4. 结论

**I1 = PASS**：插桩不改变行为、开销已量化、运行器与冻结 baseline 逐坐标一致。允许进入 Phase 4.8-2（microbenchmark 与成本校准），需用户批准。
