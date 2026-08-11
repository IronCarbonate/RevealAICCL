# Phase 4.8-0：真实部署环境审计

更新日期：2026-08-04
状态：只读审计完成；未修改 production 代码、未实现 profile、未运行实验

## 1. 冻结结论（Phase 4.7，不可修改）

R0 PASS / H5 PASS / H6 PASS / H7 FAIL / H2 FAIL / Phase 5 CLOSED；候选 profile 冻结为 partial_shards @ 75%、full reveal checkpoint 8、fast=partial_current_only、adaptive/robust/predictor/risk-gate/lookahead 全关、deterministic checker 与 fail-closed 保持。

## 2. 当前硬件与软件环境（实测 2026-08-04）

| 项 | 实测值 |
|---|---|
| 主机 | autodl-container-36da11a152-db2cf032 |
| GPU | 1× NVIDIA GeForce RTX 2080 Ti（11,264 MiB，空闲） |
| 驱动 / CUDA | 580.76.05 / 13.0（driver 级）；无 nvcc（无 CUDA toolkit CLI） |
| PyTorch | 2.8.0+cu128（CUDA 可用；device_count=1） |
| NCCL | 2.27.3（torch 内置） |
| CUPTI | `/usr/local/cuda-12.8/extras/CUPTI` 存在 |
| Nsight Systems/Compute | 无 nsys / ncu CLI |
| CPU / 内存 | 96 核可见（cgroup 配额 12 核）/ 40 GiB cgroup 上限 |
| 网络 | 单节点容器；无多节点 NIC/IB/RDMA 证据 |
| 冻结 venv | phase4-env（无 torch）；base miniconda 有 torch 2.8.0+cu128 |

## 3. L0–L3 等级判定

| 等级 | 定义 | 当前仓库 | 当前硬件 | 可达 |
|---|---|---|---|---|
| L0 | 纯 proxy | **是**（NumPy 调度，无真实 router/GPU 通信） | — | 当前状态 |
| L1 | 高保真软件执行（真实 scheduler + 真实 CPU/GPU kernel/单机通信，部分同步模拟） | 需构建 | **可支持**（1× GPU + torch/CUDA event + 模拟 comm） | **可达到** |
| L2 | 单机多 GPU（真实 router/kernel/NCCL/IPC） | 需构建 | **否**（仅 1 GPU） | 需 ≥2 GPU 实例 |
| L3 | 多节点真实部署（NCCL/NVSHMEM/RDMA、compute-comm overlap） | 需构建 | **否** | 需多节点硬件 |

**结论：当前仓库为 L0（proxy）；当前硬件最高支持 L1（高保真单机执行）。L2/L3 需新硬件，不在本实例可达范围内。**

## 4. 关键差距

1. 无真实 router/top-k（MoE 语义）路径；仅冻结的 legacy decoder 启发式候选剪枝（`rlccl/envs/decoder.py:431`，非专家路由）；
2. 无 dispatch/all-to-all/GEMM 执行路径（`rlccl/config.py` 仅存 AllToAll/AllGather 成本常数 0.2/0.3，非真实通信）；
3. 无 NVSHMEM/DeepEP；NCCL 仅单 rank 可用；
4. 无 Nsight CLI；CUPTI 存在但需通过 torch profiler 或安装工具使用；
5. 无多 GPU、无多节点、无 RDMA。

## 5. 审计结论

可达最高等级：**L1**（高保真软件执行）。据此 R1 判定为 **CONDITIONAL PASS**（见 Supervisor 复核），条件：

1. Phase 4.8-1 必须先构建高保真执行层（router shim、dispatch/GEMM kernel、reveal/sync 计时）后再测量；
2. 结论限定为高保真/单机验证，不得声称 L2/L3；
3. reveal/control/sync 成本必须实测（M），不得置 0；
4. 用户批准后进入 Phase 4.8-1。
