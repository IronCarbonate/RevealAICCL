# Phase 4.10 可复现清单（REPRODUCIBILITY_MANIFEST）

更新日期：2026-08-06

## 1. 环境（实测运行环境）

| 项 | 值 |
|---|---|
| 主机 | region-41.seetacloud.com（root；SSH 端口 50973） |
| GPU | 2× Tesla V100-SXM2-32GB |
| Python | 3.12.3（/root/miniconda3/bin/python） |
| torch / NCCL | 2.8.0+cu128 / 2.27.3 |
| 仓库 | /root/autodl-tmp/RLCCL-main（172 文件与本地 0 差异） |
| 运行前置 | `cd /root/autodl-tmp/RLCCL-main && export PYTHONPATH=/root/autodl-tmp/RLCCL-main`（脚本直接运行时须含仓库根） |

## 2. 冻结 workload

- P10-1C/1D：token corpus seed 4042（20 序列，5 family × 4；dev 12 / val 8；**未用 3042/3142/3242**）；
- P10-1D/1E/1F：router 权重 seed 20260805、K=1、E=4；chunk 计时 8×4096 tokens × D=2048；world 48 tokens × 16 features（traffic 由 router top-k 派生，seed 4042）；ratios (0.0, 0.75, 1.0)；stage_len 4；slots 80；time_limit 80；Rear4GPU 拓扑（容量/组带宽归一化为 1 单位）；
- formal corpus（5042/5142/5242）：**未生成、未运行**；
- frozen profile：partial_shards @ 75%、checkpoint 8、partial_current_only。

## 3. 脚本与命令

| 阶段 | 脚本 | 命令（仓库根） |
|---|---|---|
| P10-I1 | `outputs/phase4_10/p10_1a_substrate/p10_i1_tests.py` | `PYTHONPATH=$PWD python outputs/phase4_10/p10_1a_substrate/p10_i1_tests.py` |
| P10-1C pilot | `outputs/phase4_10/p10_1c_pilot/p10_1c_pilot.py` | `PYTHONPATH=$PWD python outputs/phase4_10/p10_1c_pilot/p10_1c_pilot.py` |
| P10-1D timing | `outputs/phase4_10/p10_1d_timing/p10_1d_timing.py` | `PYTHONPATH=$PWD python outputs/phase4_10/p10_1d_timing/p10_1d_timing.py` |
| P10-1E readiness | `outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.py` | `PYTHONPATH=$PWD python outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.py` |
| P10-1F breakdown | `outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.py` | `PYTHONPATH=$PWD python outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.py` |

## 4. Artifacts（本地 = 远程，md5 逐项一致）

| 文件 | md5 |
|---|---|
| outputs/phase4_10/p10_1a_substrate/reference_router.py | 75883f1fc26be6d968a4d1a9179c4d81 |
| outputs/phase4_10/p10_1a_substrate/p10_i1_tests.py | ccb5f6f9f750be62c82a106239fd4e07 |
| outputs/phase4_10/p10_1a_substrate/p10_i1_results.json | 80367d20b3466ffcf413f4604cddb638 |
| outputs/phase4_10/p10_1c_pilot/p10_1c_pilot.py | e23c52e58d3d2a1ba1dba1a695f59c84 |
| outputs/phase4_10/p10_1c_pilot/p10_1c_pilot_results.json | 47e2f5318e783eeb632e776e1688e721 |
| outputs/phase4_10/p10_1d_timing/p10_1d_timing.py | 88d92990eb253e18042baa260da7a420 |
| outputs/phase4_10/p10_1d_timing/p10_1d_timing_results.json | 463800b121c7a8f932ff8902d17743fc |
| outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.py | 7d2ba682903069bca039233f0050db5f |
| outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.json | 56ebf62bbce698d9746030d9af926676 |
| outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.py | 5957e921b354e3e10fb65587881df4d5 |
| outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.json | edba099879a47c83e21eba71771dbf90 |

关键冻结源文件（本地=远程）：robust_prefix.py `e04055e4…`、execution.py `6e546007…`、reveal.py `9a696214…`、
problem.py `d7ba3375…`、observation.py `8ff41709…`、recourse.py `c0c22378…`、phase4_experiment.py `32a24a78…`、
ambiguity_experiment.py `558370ec…`。

## 5. 统计口径

- 主指标 profiling OFF；P10-1F 组件分解 p95/均值双口径（p95 合计 92.4%、均值合计 99.2%）；
- p95 = 排序后 0.95 分位（与 P10-1E 同定义）；
- 配对结论基于 run 内配对（pilot/1D）；1E/1F 为单世界真实计时（无人工 delay、无预计算后延迟显示）；
- 证据等级：M（实测）、E（可执行/等价性）、D（文档）、S（主观/推断）、O（oracle）已在各文档标注。

## 6. 复现注意

1. P10-1F 探针（enumerate-min / incremental-pack / fast-view）为测量代码，不改生产路径；重跑前须校验生产文件 md5 无漂移；
2. scheduler p95 有运行间波动（11.3–12.9ms），复现以中位数/均值+区间报告为准；
3. 服务器 SSH 凭据仅用于本清单记录，不在任何 artifact 中输出。
