# Phase 4.8-4：正式协议冻结（Real-Deployment Validation）

更新日期：2026-08-04
状态：**FROZEN（待用户批准后进入 Phase 4.8-5 正式 test）**

## 1. 比较臂与主指标

| 臂 | reveal | checkpoint | scheduler |
|---|---|---|---|
| D0 | 坐标自身冻结 mode | 16 | partial_current_only |
| D1 | partial_shards, 75% | 8 | partial_current_only |
| D2 | full-information | 0 | partial_current_only（仅上界） |

主指标：`ΔE2E = E2E_D0 − E2E_D1`（正值 = D1 更好；critical-path wall-clock，CUDA event 同步）。

## 2. 正式 test corpus（冻结）

- base seeds = **`(3042, 3142, 3242)`**；
- 45 sequence（5 family × 3 seed × 3 split），生成器与 Phase 4 一致；
- 与 H2（642/742/842）、Route A（1042/1142/1242）、H5-H7（2042/2142/2242）**digest 零重合**（corpus_phase4_8_manifest.json）；
- 划分：fit 15 / validation 15 / **formal test 15**（按独立 sequence；正式 test 冻结前不查看结果）；
- universe digest：见 corpus manifest。

## 3. 执行环境（冻结，L1 单机）

| 项 | 值 |
|---|---|
| GPU | 1× RTX 2080 Ti 11GB |
| 驱动 / CUDA | 580.76.05 / 13.0（driver） |
| PyTorch / NCCL | 2.8.0+cu128 / 2.27.3 |
| numpy | 2.3.2 |
| CPU / 内存 | 12 核 cgroup / 40GB |
| 网络 | 单节点（无多节点；L2/L3 不声明） |
| 频率/电源模式 | 未固定（GPU 为默认；记录于 run_command） |
| 环境变量 | PYTHONHASHSEED=0、PYTHONDONTWRITEBYTECODE=1、LC_ALL=C、PH4_8_PROFILE=1 |
| warm-up | 每臂正式运行前 GPU warm-up ≥50 次 + 1 个全 job warm-up |

## 4. 统计

- 每臂：15 test sequences × 20 coordinates = 300 jobs；
- 重复：每 job 单次（确定性调度）；为 CI 采用 job/sequence 级配对 bootstrap 10,000（seed 20260801）；
- 主比较：D1 vs D0 的 sequence-level paired ΔE2E，95% CI lower > 0；
- 次级指标：completion、throughput（jobs/s）、tokens/s、p95/p99、CVaR95、router/scheduler/dispatch latency、GEMM stall、communication overlap、control-message cost、GPU 利用率、legality、timeout、memory overhead；
- 单位：独立 job/sequence；禁止行级当独立样本。

## 5. D1 PASS / FAIL（预注册）

PASS（全部）：ΔE2E > 0；paired 95% CI lower > 0；≥3 seed（corpus 内 3 base seeds 或等价 workload group）；≥4/5 family 正向或有预注册边界；completion 改善仍存在；全部 reveal/control/sync 成本已计入；吞吐不降；GEMM stall 不显著增；collective contention 不显著增；memory 在预算内；legality 100%；timeout 不增；可 read-back 重算；Supervisor PASS。

FAIL（任一）：CI 跨 0 或为负；收益被成本吞没；吞吐明显下降；GEMM/collective 干扰抵消；仅单 seed/family；需关安全检查；环境低于声明等级；artifact 不完整。

## 6. 正式 artifact（原子发布，15 项）

`outputs/phase4_8/deployment_validation/`：environment_manifest.json、protocol_manifest.json、raw_events.*、microbatch_results.*、job_sequence_results.*、timing_breakdown.*、throughput_results.*、resource_utilization.*、condition_summary.*、final_summary.json、integrity_manifest.json、read_back_report.json、run_command.txt、git_commit.txt、hashes.json；支持 raw→microbatch→job/sequence→condition→final 全链重算。

## 7. 冻结声明

- 本协议为 L1 单机高保真验证；不得声称 L2/L3；
- 正式 test 结果在 Phase 4.8-5 运行前不查看；
- 未修改 production 代码；合成 GEMM 不代表生产 MoE；
- H2=FAIL、Phase 5 CLOSED 维持。
