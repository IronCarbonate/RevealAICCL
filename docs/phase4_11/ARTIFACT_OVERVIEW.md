# Phase 4.11：Artifact Overview / Reproduction / Hardware / Expected Results

更新日期：2026-08-10（Phase R0 修正）

## 1. Artifact Overview（按阶段）

| 阶段 | 关键 artifact | 类型 |
|---|---|---|
| Phase 3B | `outputs/phase3b_ambiguity/*`（10 文件，250,140 raw 行） | 正式（schema-v2，read-back PASS） |
| H1 | `outputs/h1_predictability/*`（4 文件） | 正式（schema-2，read-back PASS） |
| H2 | `phase4_formal_artifacts/*`（本地只读副本；canonical 在远程 `outputs/phase4_early_planning/`） | 正式（八产物，exit 0，read-back PASS） |
| H2a/H2b | `outputs/phase4_5/h2a_profile/*`、`h2b_analysis/*` | 只读分析（正式 artifacts 派生） |
| 象限 2 | `outputs/phase4_6/route_a_reveal/`、`w1_static_precompute/`、`w2_scheduler/`、`w3_risk_gate/` | 预注册实验（新 corpus，等价性门 300/300） |
| H5–H7 | `outputs/phase4_7/h5_realizable_reveal/`、`h6_selective_reveal/`、`h7_adaptive_reveal/` | 预注册实验（新 corpus 2042/2142/2242） |
| L1 部署 | `final_summary.json`、`job_sequence_results.json` + R0 `l1_provenance_status.json` | 派生汇总保留；原始 L1 raw jobs 已丢失，禁止重造冒充 |
| L2-S 部署 | `l2_*` + R0 重建 `l2_environment_manifest.json` + `L2_PROVENANCE_R0.md` | 2×V100 真实 NCCL；旧错误 manifest 保留并标 SUPERSEDED |
| P10-1 | `outputs/phase4_10/p10_1a_substrate/`、`p10_1c_pilot/`、`p10_1d_timing/`、`p10_1e_admissibility/`、`p10_1f_audit/` | 参考路径实验（11 项 artifact 本地/远程 md5 一致） |
| Phase R0 | `outputs/phase_r0/evidence_repair/`、`scripts/run_r0_i1_strengthening.py` | evidence repair；P10-I1-strengthened 19/19；R1 未开始 |

## 2. Reproduction Instructions

### 2.1 环境准备

```bash
cd /root/autodl-tmp/RLCCL-main   # 或本地 RLCCL-main
export PYTHONPATH=$PWD PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C
/root/miniconda3/bin/python -V   # 3.12.3；torch 2.8.0+cu128；NCCL 2.27.3
```

### 2.2 分阶段重跑（顺序按文档，勿混用 corpus）

| 阶段 | 命令（仓库根） | 耗时参考 |
|---|---|---|
| Phase 3B | `python -B scripts/run_phase3b_ambiguity.py --formal --output-dir outputs/phase3b_ambiguity` | ~58 min |
| H1 | `python -B scripts/run_h1_predictability.py --formal` | ~2 min |
| H2（远程正式） | 冻结 launcher（admission-003；仅限授权副本） | ~4.5 h |
| H2a/H2b | `python outputs/phase4_5/h2a_profile/analyze_h2a.py`、`python outputs/phase4_5/h2b_analysis/analyze_h2b.py` | 分钟级 |
| Route A / W1–W3 | `python outputs/phase4_6/route_a_reveal/route_a_runner.py`、`w1_static_precompute/w1_static_precompute.py`、`w2_scheduler/w2_scheduler.py`、`w3_risk_gate/w3_risk_gate.py` | 分钟–小时级 |
| H5/H6/H7 | `python outputs/phase4_7/h5_realizable_reveal/h5_runner.py`、`h6_selective_reveal/h6_runner.py`、`h7_adaptive_reveal/h7_runner.py` | 小时级 |
| L1 formal | `python -m torch.distributed.run --nproc_per_node=1 outputs/phase4_8/deployment_validation/formal_test.py`（PH4_8_PROFILE=1） | 小时级 |
| L2-S formal | `python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 outputs/phase4_8/deployment_validation/formal_test.py`（PH4_8_L2=1） | 小时级 |
| P10-I1 / R0 strengthen | `python outputs/phase4_10/p10_1a_substrate/p10_i1_tests.py`、`python scripts/run_r0_i1_strengthening.py` | 分钟级；R0 strengthen 需要 CUDA |
| P10-1C/1D/1E/1F | `python outputs/phase4_10/p10_1c_pilot/p10_1c_pilot.py`、`p10_1d_timing/p10_1d_timing.py`、`p10_1e_admissibility/p10_1e_readiness_test.py`、`p10_1f_audit/p10_1f_scheduler_breakdown.py` | 1–5 分钟/阶段 |

### 2.3 复现纪律

1. 每个阶段使用其冻结 corpus，禁止混用/挑选；
2. 重跑前校验关键源文件 md5（`PHASE4_10_REPRODUCIBILITY_MANIFEST.md` §4）；
3. 正式/预注册实验必须先冻结协议与 artifact schema，再运行；
4. P10-1 formal 已 CLOSED，禁止运行 formal test（5042/5142/5242 从未生成）。

## 3. Hardware Requirements

| 层 | 硬件 | 用途 |
|---|---|---|
| L1 | 1× RTX 2080 Ti（11GB）或等价 | 单机高保真部署验证（单 rank） |
| L2-S / L2-R | 2× Tesla V100-SXM2-32GB（或等价），torch 2.8.0+cu128、NCCL 2.27.3 | 真实 2-rank NCCL；reference router 路径 |
| L3 | 多节点 RDMA/NVSHMEM；DeepEP 需 sm_80+（Ampere/Hopper） | **未验证；V100 sm_70 不支持 DeepEP** |
| CPU/内存 | 12 核 / 40 GiB（H2 正式实测） | scheduler/ambiguity 计算 |

## 4. Expected Results（复现后应得到的关键值）

| 阶段 | 期望结果 |
|---|---|
| Phase 3B | conditions 1–6 PASS；boundary_scenarios K=8；radius 0.34327919716983946 |
| H1 | total paired Δ −0.0790（CI [−0.1133,−0.0478]）；H1 FAIL |
| H2 | robust E2E 1042.46ms vs Wait 115.80 / Partial 103.88ms；H2 FAIL |
| H2a/H2b | 92.3% 解释率；+0.11 slots；98.5% 动作重合 |
| Route A | S3 11.80 vs S0 20.95 vs fullinfo 10.80 |
| H5/H6/H7 | A4 +9.22ms；partial_shards +0.57–0.81ms；controller ≡ B75 |
| L1 formal | ΔE2E +10,953µs（CI [+3,598,+23,148]）；completion +6.43；吞吐 +24.7% |
| L2-S formal | ΔE2E +6,458µs（CI [+3,409,+9,385]）；completion +6.43；吞吐 +13.7% |
| P10-I1 | 历史 17/17；R0 strengthen 19/19（actual 75% view、独立 traffic oracle、hidden perturbation/no-leak、token conservation、ties） |
| P10-1C | completion Δ +1.95；E2E Δ −19,688.8µs；hotspot −32,817.2µs |
| P10-1D | C1 22.9 vs 28.1（+5.2 slots）；稳态 E2E C1≈B0；readiness replay，非 concurrency |
| P10-1E | P1/P2/P3 PASS、P4 FAIL；replay/quantized candidate window 419.84µs；p95 12,290.03µs |
| P10-1F | 单步 p95 11.29–12.93ms；first-commit 8,673.8µs；fast-path estimates 1,043.1/1,139.5/2,047.2µs |

注：scheduler p95 有运行间波动（11.3–12.9ms），复现以中位数 + 区间报告；其余统计量应在协议统计精度内复现。

## 5. Fail-closed 检查

- 不声称生产 MoE（reference ≠ production）；不声称 L3/DeepEP/RDMA 已验证；不声称 L2-R E2E 收益；
- 不声称 419.8µs 为 concurrent window，不声称 1.045ms 为理论下界；历史 P10-1 CLOSED 不禁止另立新架构；
- hotspot_random_walk 与全部负结果不得删除或弱化。
