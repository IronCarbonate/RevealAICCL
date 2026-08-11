# L2 可移植性分析（Portability Analysis）

更新日期：2026-08-05

## 1. 组件可移植性

| 组件 | 跨硬件可移植性 | 说明 |
|---|---|---|
| 调度 CPU 节省（~1.3ms/slot × 6 slots） | **高** | 纯 CPU 逻辑，与 GPU/网络无关；是 ΔE2E 主导项 |
| reveal/control 成本（~5µs/事件） | 高 | 协议固定，与硬件弱相关 |
| completion 改善（+6.43 slots） | 高 | 调度语义决定，硬件无关 |
| NCCL allreduce 延迟 | 中 | 随硬件/NCCL 版本变化（V100 62–87µs） |
| GPU kernel（合成 GEMM） | 低（合成） | 不代表生产 MoE，需真实路径 |

## 2. L1→L2 的幅度变化

- L1（RTX 2080 Ti，合成 collective）：ΔE2E +10.95ms（约 20%）；
- L2（2×V100，真实 NCCL）：ΔE2E +6.46ms（约 12%）；
- 差异主要来自真实 collective 成本（两臂都增加，D0 更多）与机器负载；
- **方向与显著性在两级一致**（CI lower>0），结论可移植；幅度需按目标硬件校准。

## 3. 结论

frozen profile 的 E2E 优势是**机制性**的（调度 CPU 节省 + 信息更早可用），不是特定硬件偶然结果；具体幅度应在目标部署硬件上复测。NCCL/DeepEP 相关成本须用目标环境的 M 级测量更新。
