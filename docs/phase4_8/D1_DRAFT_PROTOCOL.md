# D1 草案协议：Real-Deployment Validation（DRAFT）

更新日期：2026-08-04
状态：**DRAFT（R1=CONDITIONAL PASS 后、用户批准后正式化）**

## 1. 比较臂

| 臂 | reveal | checkpoint | scheduler | 角色 |
|---|---|---|---|---|
| D0 | current/default | 16 | partial_current_only | 当前部署 baseline |
| D1 | partial_shards, 75% | 8 | partial_current_only | 候选 profile |
| D2 | full-information | 0 | partial_current_only | 仅上界 |

唯一区别：reveal profile（D0↔D1）；D2 仅 regret/upper-bound。

## 2. 主指标与成本

- 主指标：`ΔE2E = E2E_D0 − E2E_D1`（正值 = 候选更好）；critical-path wall-clock；
- E2E 分项：router、top-k、shard/reveal、control message、sync、scheduler、dispatch、GEMM、all-to-all、allreduce/allgather、CPU wait、GPU idle、kernel launch、job 完成；
- CPU/GPU 时钟经 CUDA event 同步；并行区间不重复计入；累计工作量仅作次级指标；
- 所有成本带证据等级（M/E/D/S/O），正式 D1 主要依赖 M。

## 3. 新 workload/corpus（草案）

- 禁止复用：H2 正式 test、Phase 4.6 test、Route A、Phase 4.7 H5/H6/H7 test；
- 草案种子：**`(3042, 3142, 3242)`**（正式化时冻结）；
- 划分：development / validation / calibration / formal test，按独立 job/sequence；
- 正式 test 在协议冻结前不查看；须 zero-overlap digest 校验与 read-back。

## 4. 执行环境（L1 高保真，本机）

- 1× RTX 2080 Ti、torch 2.8.0+cu128、NCCL 2.27.3、CUPTI；
- 通信/同步为单 rank 实测或高保真模拟；**不声称 L2/L3**；
- 若需 L2/L3，须另行提供 ≥2 GPU / 多节点硬件。

## 5. D1 PASS / FAIL（预注册，来自指令文件）

PASS（全部）：E2E 改善 > 0；job/sequence paired 95% CI lower > 0；≥3 seed 或等价 workload group；≥4/5 family 正向或有预注册边界；completion 改善仍存在；全部 reveal/control/sync 成本已计入；吞吐不降；GEMM stall 不显著增；collective contention 不显著增；memory 在预算内；legality 100%；timeout 不增；可 read-back 重算；Supervisor PASS。

FAIL（任一）：E2E CI 跨 0 或为负；completion 收益被成本吞没；吞吐明显下降；GEMM/collective 干扰抵消收益；仅单 seed/family；需关安全检查；实际环境低于声明等级；artifact 不完整。

## 6. 正式 artifact（原子发布，15 项）

`environment_manifest.json`、`protocol_manifest.json`、`raw_events.*`、`microbatch_results.*`、`job_sequence_results.*`、`timing_breakdown.*`、`throughput_results.*`、`resource_utilization.*`、`condition_summary.*`、`final_summary.json`、`integrity_manifest.json`、`read_back_report.json`、`run_command.txt`、`git_commit.txt`、`hashes.json`；支持 raw→microbatch→job/sequence→condition→final 全链重算。

## 7. 前置（R1 CONDITIONAL PASS 后）

1. Phase 4.8-1 构建高保真执行层（router shim、dispatch/GEMM、reveal/sync 计时）并通过 I1 equivalence；
2. Phase 4.8-2 microbenchmark 校准（证据等级 M）；
3. Phase 4.8-3 pilot 通过 P0；
4. 用户批准正式协议与 corpus 后冻结、运行 D1。
