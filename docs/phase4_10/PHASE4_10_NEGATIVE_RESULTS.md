# Phase 4.10 负面结果汇总（NEGATIVE_RESULTS）

更新日期：2026-08-06

## 1. 路径级负面结论（冻结）

| 结论 | 关键证据 | 状态 |
|---|---|---|
| P10-F0-v1 FAIL：历史 formal 不可准入 | replay/quantized candidate window 419.8µs < scheduler step p95 12,290µs（复现 11.29–12.93ms）；未测 concurrent window | 冻结并由 R0 修正口径 |
| P10-SF0-B FAIL：旧实现优化目标不可认证 | implementation fast-path estimate：step-only 1,043.1µs；含 bind/checker 1,139.5µs；含 digest 2,047.2µs；不是严格理论下界 | 冻结并由 R0 修正口径 |
| P10-1 formal CLOSED | 用户裁定：不批准向量化/记忆化解锁、不换 workload/窗口、不重开 | 冻结 |
| L2-R production-path E2E 不可证明 | 窗口/调度器/下界三重差距；pilot 与 1D 均未确立 E2E 收益 | 冻结 |

## 2. 测量级负面证据

- **P10-1C pilot E2E 为负**：RR-D1 vs RR-D0 ΔE2E = **−19.7ms**（配对不显著，被 ~80–100ms 固定 setup 主导）；completion +1.95 slots 方向为正但 E2E 未确认；
- **P10-1D E2E 收益未在本规模确立**：稳态 C1（151.4ms）≈ B0（150.0ms）；reveal C0−C1 ≈ −2.5ms、deployment B0−C1 ≈ −1.4ms（均非正）；amortized +11.2ms 来自 B0 冷启动噪声，不得作收益；
- **chunk arrival 子 slot**：1D 规模下 chunk 在 slot ≤1 完成，arrival 门控对调度近 no-op；C1 收益来自 reveal profile 而非 arrival 延迟；
- **scheduler 单步延迟主导**：12.29ms p95 中 ~99% 为静态/可缓存/可增量开销（BFS 重算 83.5% + pack 8.9%），但纯实现层消减后下界仍超窗口；
- **DeepEP 硬件级不可行**：V100（sm_70）不满足 DeepEP 的 Ampere/Hopper（sm_80+）要求，未安装即不可行；
- **MSCCL 不可编译**：xml_converter 代码存在，但 msccl 未安装，当前环境不可编译 AllToAll/AllGather 工具。

## 3. hotspot_random_walk 适用边界（保留）

| 阶段 | hotspot_random_walk E2E Δ |
|---|---:|
| P10-1C pilot（RR-D1 vs D0） | **−32,817µs（−32.8ms）** |
| P10-1D reveal（C0−C1） | −590µs（−0.59ms） |
| P10-1D deployment（B0−C1） | **−3,553µs（−3.6ms）** |

该 family 在 P10-1 全部阶段均为负，作为预注册适用边界单列，不掩盖、不剔除。

## 4. 意义

负面结果共同支持的收窄结论是：**旧 replay-based L2-R reference 路径没有在 419.8µs 量化候选窗口内完成调度；
瓶颈不是 router 正确性（已 PASS），而是当前 CPU scheduler 单步延迟。现有 fast-path 数值是实现估算，不排除新的 concurrent/event-driven architecture。**
这防止未来在同一冻结语义下重复探索；任何新出路必须来自用户方向（新实现层授权、新 workload/窗口定义或新硬件）。
