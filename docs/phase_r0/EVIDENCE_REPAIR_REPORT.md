# Phase R0：Evidence Repair Report

更新日期：2026-08-10  
状态：**Supervisor R0 = PASS / NO VETO（2026-08-10）；Phase R1 = AUTHORIZED**

## 1. 范围与不变量

本阶段只修复证据和表述，不修改 scheduler、router/reveal 算法、75% budget、
checkpoint 8、`partial_current_only` 或 checker 语义；未运行新的 formal E2E，
未接 DeepEP，未实现 expert GEMM/combine，也未开始 concurrent pipeline。

## 2. A — L1 provenance

本地工作区、服务器 `/root/autodl-tmp` 和 `phase4_remote_payload.tar.gz` 均已做
只读文件名、内容特征、大小、archive member 与 SHA-256 搜索。未找到独立的
历史 L1 raw jobs。

- 判定：`L1_RAW_ARTIFACT_LOST`；
- 禁止：重新生成数据并冒充历史 raw；
- 当前无后缀 `raw_jobs.json` 的 SHA-256 为
  `3923ebe87a0391f43122c092d16635715c63df19588836336631709ebdcb71ed`，
  本地/服务器一致，但内容是 L2（D0 53,729.038µs；D1 47,272.295µs），不得误认作 L1；
- 仍存的 `final_summary.json` 与 `job_sequence_results.json` 只能支持“历史派生汇总
  报告正收益”，不能支持“当前仓库具有可独立重算的 L1 raw evidence chain”。

机器可读状态见 `outputs/phase_r0/evidence_repair/l1_provenance_status.json`。

## 3. B — L2 provenance

旧 `l2_environment_manifest.json` 实际是误放入 L2 集合的 L1/RTX 2080 Ti
manifest。R0 已：

1. 原样保留为 `l2_environment_manifest.pre_r0_SUPERSEDED.json`；
2. 从 `l2_collective_results.json`、`l2_final_summary.json`、
   `read_back_report_l2.json` 和真实 NCCL 执行代码重建 canonical manifest；
3. 用当前服务器的 2× Tesla V100-SXM2-32GB、torch 2.8.0+cu128、CUDA 12.8、
   NCCL 2.27.3 环境做 corroboration；
4. 明示缺少历史时点的完整 `nvidia-smi`/launch log，当前服务器观测不是其替代品。

修复后的 manifest 只支持 **L2-S synthetic substrate + real two-rank NCCL**，
不支持 production MoE 或 L2-R 声明。旧 `hashes_l2.json` 与旧 integrity manifest
作为历史文件保留；R0 之后以新的 R0 artifact manifest 为准。

## 4. C — P10-I1 strengthening

在服务器 V100 CUDA 上运行 `scripts/run_r0_i1_strengthening.py`，结果 **19/19 PASS**：

- actual 75% prefix-only view：192/256 token，未计算/存储 future token view；
- CPU 显式循环独立重建 token→traffic oracle，与 CUDA 向量化结果一致；
- 真实修改未揭示 suffix：49/64 assignments 改变，但 revealed prefix indices、scores
  与 partial traffic 均不变；
- 跨 shard 256/256 token 无丢失、无重复，sharded/batched assignments 一致；
- all-equal、masked、重复运行以及 CPU oracle deterministic tie tests 全通过。

结果 artifact SHA-256：
`54bf6e25b10b5a921bee846dbd9de44295c3bc7c8ff1640f2c3a7ef54b890f24`。
其证据范围仅为 reference-router correctness；不构成 E2E 或 concurrency 证据。

## 5. D — 文档口径修复

权威汇总、证据矩阵、论文草稿、Phase 4.10 收尾文档和历史 Supervisor 记录均已
修正或加 superseding note：

- P10-1D = 真实逐 chunk CUDA router timing，随后 readiness quantized/replayed；
- 419.84µs = replay/quantized candidate actionable window，不是直接测量的
  concurrent pipeline window；
- implementation fast-path estimates：step-only 1,043.1µs、含 bind/checker
  1,139.5µs、含 digest 2,047.2µs；不是 strict/theoretical lower bound；
- 历史 replay-based P10-1 formal 继续 CLOSED，但不禁止新 concurrent/event-driven
  architecture。

## 6. R0 Gate 提交

| 项 | 状态 |
|---|---|
| A. L1 provenance | 完成：raw 标 LOST，未重造 |
| B. L2 provenance | 完成：manifest 重建，旧文件 SUPERSEDED 保留 |
| C. P10-I1 strengthening | 完成：V100 CUDA 19/19 PASS |
| D. 文档修正 | 完成：权威文档更新，历史记录保留并标 superseded in part |
| Supervisor R0 判定 | **PASS / NO VETO** |
| Phase R1 | **AUTHORIZED** |

Supervisor 已完成独立审查并判定 R0 PASS / NO VETO；以上事实自此冻结。
