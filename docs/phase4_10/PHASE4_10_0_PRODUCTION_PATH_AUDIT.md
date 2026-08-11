# Phase 4.10-0：Production-Path Admissibility Audit

更新日期：2026-08-05
状态：只读审计完成；未修改 production 代码、未替换组件、未运行实验

## 1. 冻结结论与 deployment profile

H1=FAIL、Phase 3B=PASS、H2=FAIL、Phase 5=CLOSED、H5=PASS、H6=PASS、H7=FAIL、L1-D1=PASS、L2-D1=PASS、L2-F0=PASS。frozen profile = partial_shards @ 75%、checkpoint 8、fast scheduler = partial_current_only（adaptive/robust/predictor/risk-gate/lookahead 全关）。不可调参。

## 2. 组件审计（仓库 + 服务器实测）

| 组件 | 仓库代码 | 实测状态 |
|---|---|---|
| router（MoE token→expert） | **无** | 仅冻结 legacy decoder 的启发式候选剪枝（`decoder.py:431`），非 MoE router |
| top-k（到 expert） | **无** | 同上 |
| token arrival | 仅 proxy `DemandRevealProcess` | 非真实到达事件 |
| histogram | 仅 proxy 计数 | 非真实聚合 |
| shard readiness | 仅 proxy 事件 | 非真实 shard 完成事件 |
| expert packing | **无** | — |
| expert GEMM | **无**（仅冻结 V1 policy torch 模型，非 expert kernel） | — |
| combine | **无** | — |
| AllToAll/AllGather 工具 | `xml_converter.py`（MSCCL 元数据/XML） | msccl **未安装** → 当前环境不可编译 |
| NCCL collectives | torch.distributed | **真实、可运行、已在 L2 critical path** |
| DeepEP | **无** | **未安装**；V100=sm_70，DeepEP（需 sm_80+）**不支持** |
| triton | — | 3.4.0 可用（可编译自定义 kernel） |

## 3. 四态分类（代码存在 / 可编译 / 可运行 / 进 critical path）

| 组件 | 代码存在 | 可编译 | 可运行 | 进 critical path |
|---|---|---|---|---|
| router/top-k（MoE） | 否 | 否 | 否 | 否 |
| token arrival / shard readiness | 否（仅 proxy） | 否 | 否 | 否 |
| histogram | 否（仅 proxy） | 否 | 否 | 否 |
| expert packing / GEMM / combine | 否 | 否 | 否 | 否 |
| MSCCL AllToAll/AllGather 工具 | 是（xml_converter） | 否（缺 msccl） | 否 | 否 |
| NCCL（torch.distributed） | 是 | 是 | **是** | **是（L2 已用）** |
| DeepEP | 否 | 否 | 否（且硬件不支持） | 否 |
| triton kernel 编译 | 工具可用 | 是（需自定义） | 待实现 | 待实现 |

## 4. D0/D1 在真实 router 路径中的公平映射

- 唯一区别 = reveal profile：D0 = 默认 reveal（full@16）；D1 = partial_shards 75%（full@8）；scheduler/checker/fail-closed 相同；
- 真实 router 路径下的映射：router 产生 token 到达/top-k/shard 事件 → 按 profile 在 checkpoint 8/16 之前向调度器揭示对应比例的已到达 token；
- 公平性要求：真实 router 必须对 D0/D1 产生**相同的 token 到达流**（只改变揭示时机/粒度），否则比较不公平；
- 调度器仍为 partial_current_only（协调器视图）。

## 5. 审计结论

当前仓库**不存在**真实 MoE router / expert GEMM / DeepEP 生产路径；真实可用的只有 NCCL collectives（torch.distributed，L2 已验证）。因此生产路径桥接需要**新建**真实组件（P10-1 router、P10-2 GEMM），且 DeepEP 步骤在本硬件（V100 sm_70）上**不可行**，需 sm_80+（Ampere/Hopper）硬件。

据此 P10-R0 判定为 **CONDITIONAL PASS**（见 Supervisor 复核）。

## 6. 约束

- 未修改 production 代码；未替换 router/GEMM；未运行 pilot/formal；未生成正式 corpus；未调参；未恢复被冻结机制；未进 L3；未创建额外 Subagent。
