# Phase 4.10 未来工程范围（FUTURE_SCOPE）

更新日期：2026-08-06
状态：**仅记录可选方向；P10-1 formal 已 CLOSED，任何方向需用户重新授权**

## 1. 已关闭 / 不开放

- 历史 replay-based P10-1 formal：CLOSED（用户裁定），不得把它改名后重开或把 replay 结果冒充 concurrency；
- 允许另立新的 concurrent/event-driven router pipeline，但必须使用新的 Gate、直接 timestamp 与独立证据链；
- Phase 5：CLOSED；DeepEP/L3：不在当前硬件（V100 sm_70）范围。

## 2. 可选未来方向（仅作记录，不构成申请）

| # | 方向 | 前置条件 | 备注 |
|---:|---|---|---|
| F1 | 真实生产 MoE router（L2-P 命名） | 具备真实 MoE runtime 的硬件/仓库环境；新的等价性门 | 当前仓库无生产 router；需全新立项 |
| F2 | 调度器快速路径（向量化/记忆化） | 用户明确批准新实现阶段；实测认证 `L_sched + L_commit < 336µs` 并通过等价性校验（候选/动作序列/checker/digest 恒等） | 本阶段 SF0-B FAIL；未测量技术不得作为证据 |
| F3 | 真实 expert GEMM/packing/combine（P10-2） | 新协议与硬件 | 本阶段明确未实现 |
| F4 | DeepEP / L3 多节点 | Ampere/Hopper+ 硬件；RDMA/NVSHMEM 验证 | V100 不可行（sm_70） |
| F5 | MSCCL AllToAll/AllGather 工具 | 安装 msccl 后编译 | 当前不可编译 |
| F6 | 新 workload/窗口语义下的 reveal 研究 | 用户方向；须重新证明 P4 | 窗口/预算/checkpoint 冻结不得擅自变更 |

## 3. 已冻结可复用资产

- L2-S 部署收益（ΔE2E +6.46ms）与 frozen profile（partial_shards @ 75% / ckpt8 / partial_current_only）；
- L2-R reference router 正确性（P10-I1 17/17）与等价性测试基座（`p10_i1_tests.py`）；
- replay/quantized candidate window（419.8µs）、implementation fast-path estimates（step-only 1,043.1µs；含 bind/checker 1,139.5µs；含 digest 2,047.2µs）、历史目标（<336µs）；
- hotspot_random_walk 负结果边界（−32.8ms / −0.59ms / −3.6ms）；
- 测量脚本与 read-back 方法（`outputs/phase4_10/p10_1f_audit/`、本清单 §3）。

## 4. 规则

1. 任何未来方向必须先获得用户/Supervisor 明确授权，不得沿用已失效的准入；
2. 不得恢复被冻结机制、不得改动 75%/ckpt8/router/workload 冻结定义；
3. 新结论必须在新协议下实测认证并通过 read-back，方可与既有冻结结论并列。
