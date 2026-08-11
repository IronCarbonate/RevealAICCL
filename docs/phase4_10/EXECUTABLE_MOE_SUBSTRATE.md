# Phase 4.10-1A：Executable MoE Substrate Selection

更新日期：2026-08-05
判定：**substrate = (c) 最小 PyTorch reference**（L2-R 级）

## 1. 冻结结论与 deployment profile

H1=FAIL、Phase 3B=PASS、H2=FAIL、Phase 5=CLOSED、H5=PASS、H6=PASS、H7=FAIL、L1-D1=PASS、L2-D1=PASS、L2-F0=PASS、P10-R0=CONDITIONAL PASS。frozen profile = partial_shards @ 75%、checkpoint 8、partial_current_only。**本阶段目标等级 = L2-R（可执行 reference MoE operator path + 真实双 GPU NCCL），不称"生产 router"**。

## 2. 搜索结论（仓库 + 环境）

| 候选 | 结果 |
|---|---|
| (a) 已有可运行 MoE 实现 | **无**（仓库无 MoE 代码；环境无 megablocks/tutel/fastermoe 等包） |
| (b) 已有 PyTorch MoE reference | **无**（torch.nn 无 MoE 模块） |
| (c) 最小 PyTorch reference | **选定**（需从零构建，作为 L2-R substrate） |
| (d) Triton | 仅作后续优化，不作第一正确性 oracle（triton 3.4.0 可用） |

## 3. 选定的 substrate（最小 PyTorch reference）

组件（全部为标准 PyTorch，确定性、可编译、可运行于 2×V100）：

1. **router gating**：`Linear(D → E)` 产生 logits；
2. **top-k**：`torch.topk`（E=4 experts，k=1，确定性 tie-break：logits 相等时取小 index）；
3. **histogram**：`torch.bincount` 统计每 expert 的 token 数；
4. **shard readiness**：按 token shard 批次产生"已路由"事件；
5. expert 线性层（可选最小 `Linear(D→D)` 用于算子闭环，P10-2 才做真实 GEMM）。

权重：**冻结**（固定 seed 初始化，D0/D1 共用同一权重）。实现位置：`outputs/phase4_10/p10_1a_substrate/`（新文件，不改 production）。

## 4. 约束

- 本 substrate 是 **reference**，不得称为"生产 router"；L2-P 需接入已有生产 MoE runtime 后才可命名；
- 本轮只选型与协议，**不实现** router bridge、不运行实验、不生成 corpus、不实现真实 GEMM、不用 Triton 优化；
- 不调整 75%/ckpt8；不恢复被冻结机制；不进 DeepEP/L3。
