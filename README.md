# RevealAICCL

> Safe progressive collective scheduling for Router-derived, initially unknown MoE traffic.

## 摘要

RevealAICCL 研究一个现实的 MoE 通信问题：AlltoAllv 流量只有在 Router 运行后才逐步产生，而传统 collective scheduler 通常假设完整 traffic 已知。系统不预测未来流量，也不允许未揭示 demand 进入执行路径；它在 Router 逐 chunk 产生 `token → expert` assignment 时，增量更新调度状态并提交已合法揭示的通信。

当前 reference system 在 **2× Tesla V100-SXM2-32GB** 上使用 PyTorch Router、Compiled Event-Driven AICCL 和 PyTorch distributed NCCL。正式实验中，progressive forward AlltoAllv 相对 identical delayed control 的 paired median 改善为 **+0.829 ms**；完整 reference MoE 的改善为 **+2.801 ms**，95% CI **[+0.967, +3.714] ms**。R5 fast data-prep 将 packing p50 降低 **13.214×**、count construction 降低约 **21.71×**，并作为当前工程 baseline。

## 1. 问题定义

设 Router 按 chunk 逐步揭示 token 的 expert destination。RevealAICCL 在任意时刻只允许已完成 chunk 贡献 demand：

```text
unknown future traffic
        ↓
completed Router chunks
        ↓
revealed demand only
        ↓
compiled scheduling and legal communication
```

系统保持以下语义：

- `partial_current_only`：当前动作只依赖当前已揭示 demand；
- `partial_shards@75%` 与 `checkpoint8`：保持冻结的 partial-view 配置；
- unrevealed token/top-k 不可进入 ready state、packing 或通信；
- deterministic ordering、dynamic legality guard 与 fail-closed checker；
- runtime BFS = 0，fast-path full-state rebuild = 0。

## 2. 系统方法

### 2.1 Revealed-Only Router Traffic

传统 AICCL 接收完整 traffic matrix 后生成 schedule；真实 MoE 则只能在 Router 运行过程中逐 chunk 得知 token 去向。RevealAICCL 因此将 token 划分为多个 Router chunk，并只对已经完成的 chunk 执行 deterministic top-k：

```text
Router chunk 0 → 揭示第一部分 token 去向
Router chunk 1 → 揭示下一部分 token 去向
...
Router final   → 完整 traffic 才全部已知
```

例如，8 个 token 被分为两个 chunk：

```text
chunk 0: t0 t1 t2 t3
chunk 1: t4 t5 t6 t7
```

若 `chunk 0` 的 top-k assignment 为：

```text
t0 → Expert 2 → GPU1
t1 → Expert 0 → GPU0
t2 → Expert 3 → GPU1
t3 → Expert 1 → GPU0
```

那么此时 scheduler 只能观察并处理这四个 token；`chunk 1` 的 assignment 仍属于 hidden future。每个 chunk 在独立 CUDA Router stream 上记录 completion event，EventBridge 通过非阻塞 query 将已完成 chunk 加入 ready bitmap，未完成 chunk 不可进入 scheduler、packing 或通信。

### 2.2 Incremental AICCL Scheduling

每次有新 chunk 完成时，scheduler 只把该 chunk 产生的 delta demand 加入现有状态。例如 `chunk 0` 新增两个 `GPU0 → GPU1` token，而此前仍有 `x` 个已揭示 token 未处理，则 residual demand 更新为：

```text
previous revealed demand = x
new revealed delta       = 2
current residual demand  = x + 2
```

为避免每次 reveal 都重新搜索拓扑，系统预先编译静态 route 与通信模板，在运行时仅做增量绑定：

| 模块 | 作用 |
|---|---|
| `StaticPlanCompiler` | 预计算 source/destination route、topology legality、resource-group mapping 和固定模板顺序。 |
| `IncrementalState` | 维护 ready/committed bitmap、residual demand、pending-ready state 和 resource credits。 |
| `FastBinder` | 将已揭示 demand 绑定到预编译模板，并确定性选择 action。 |
| `DynamicGuard` | 再次检查 revealed-only、duplicate commit、residual demand、capacity 与 bandwidth-group conflict。 |

通过检查后产生 committed action；运行时不重新 BFS，也不重建完整状态。Compiled path 与原 scheduler/checker 保持 static、single-step 和 trajectory 三级 exact equivalence。

### 2.3 Progressive Variable-Size Communication

Committed action 被转换为按 destination 分组的 token list。对 rank `i`，`sendcounts[i]` 表示当前 descriptor 中发往该 rank 的 token 数；offset 和 contiguous payload 由已揭示 assignment 直接构造：

```text
Action:
GPU0 → GPU1
[t0, t2]

sendcounts[i] = 当前发往 GPU i 的 token 数
```

当前工程 baseline 使用 preallocated per-destination buffer、incremental counter/offset 和 tensor/native packing，避免为每个 descriptor 重建 Python token list。通信由 PyTorch distributed NCCL 执行真实 uneven-split AlltoAllv：

```python
torch.distributed.all_to_all_single(
    ...,
    input_split_sizes=sendcounts,
    output_split_sizes=recvcounts,
    async_op=True,
)
```

每个 descriptor 只携带已完成 chunk 的 delta sendcounts，不需要等待或读取最终完整 sendcounts。

### 2.4 Progressive Execution Pipeline

当后续 Router chunk 仍在计算时，先前 chunk 已可完成 reveal、schedule 和 communication submission：

```text
Router chunk 0  ███
                   ↓ reveal
                   schedule 0
                   communication 0 ──────

Router chunk 1      ███
                       ↓ reveal
                       schedule 1
                       communication 1 ──────

Router chunk 2          ███
                          ↓ reveal
                          schedule 2
                          communication 2 ──────
```

完整 reference MoE 在 forward dispatch 后执行相同的 non-progressive per-expert FP32 MLP，再通过真实 variable-size NCCL return path 将结果送回 source rank，并按 original token position combine。Early 与 Delayed 对照使用完全相同的 Router/top-k、descriptor、expert batch、GEMM shape、return payload 和最终输出，唯一实验变量是 forward descriptor 的启动时机。

### 2.5 Safety and Correctness

Payload 携带可验证的 token identity、source、destination 和内容。执行路径检查：

- future/unrevealed token 不可被调度、打包或发送；
- 每个 token 恰好 dispatch 和 return 一次；
- token→expert、return source/destination 与 original position 正确；
- lost、duplicate、wrong-expert、wrong-destination、wrong-return 和 corruption 均为 0；
- legality、token integrity、hidden-future perturbation 与 Early/Delayed final-output equivalence 全部通过。

## 3. 实验设置

| 项目 | 配置 |
|---|---|
| Hardware | 2× Tesla V100-SXM2-32GB，single node，2 NCCL ranks |
| Router | Minimal/reference PyTorch deterministic top-k Router |
| Scheduler | Compiled Event-Driven AICCL |
| Backend | PyTorch distributed NCCL；不是 MSCCL/MSCCL++、DeepEP 或 PCCL |
| Communication | Uneven-split `all_to_all_single(async_op=True)` |
| Traffic | balanced、skewed、all-to-one-like、zero-sized-pair、multiple-progressive-shards |
| Primary comparison | Progressive Early vs identical Delayed；唯一变量为 descriptor 启动时机 |
| Primary metric | `Delta = T_delayed − T_early`，正值表示 progressive 更快 |

Formal primary timing 从 first Router launch 到目标输出完成，包含对应路径中的 Router、data preparation、count exchange、AICCL、communication，以及 full-MoE 实验中的 expert、return 和 actual combine；correctness-only oracle 不进入 primary。

## 4. 结果

### 4.1 控制面与语义

| 项目 | 结果 |
|---|---:|
| Event completion → host-ready p95 | **4.743 µs** |
| Integrated ready → NCCL submit-return p50 / p95 / p99 | **426.230 / 578.891 / 959.454 µs** |
| Static equivalence | **360/360**, 0 mismatch |
| Single-step equivalence | **212/212**, 0 mismatch |
| Trajectory equivalence | **36/36**, 524 per-step comparisons, 0 mismatch |
| Checker comparisons | **736**, 0 mismatch |
| Hidden pending / suffix checks | **192/192** unaffected；**12/12** prefix-equivalent |

### 4.2 Formal Performance

| Experiment | Corpus | Paired median Delta | 95% CI | Seeds |
|---|---:|---:|---:|---:|
| Progressive forward variable-size AlltoAllv | 300 pairs | **+0.829 ms** | **[+0.242, +1.439] ms** | **3/3 positive** |
| Full reference MoE | 300 pairs / 1,200 rank-arms | **+2.801 ms** | **[+0.967, +3.714] ms** | **3/3 positive** |

Full-MoE paired relative makespan reduction median 为 **+0.608%**，95% CI **[+0.225%, +0.960%]**。这是 corpus-wide paired result，不表示每个 traffic family 或每个 tail sample 都更快；R3 与 R4 的数值对应不同 primary boundary，不能相加。

### 4.3 Fast Data-Prep Engineering Result

| Metric | Reference E0 | Fast E1 | Improvement |
|---|---:|---:|---:|
| Packing p50 / descriptor | 10.844 ms | 0.828 ms | **13.214×** paired median speedup |
| Count construction p50 | 503.584 µs | 23.193 µs | **≈21.71×** |
| Full-MoE `E0 − E1` | — | — | **+78.872 ms**, CI **[+76.626, +80.810] ms** |
| Paired relative makespan reduction | — | — | **+16.423%**, CI **[+15.722%, +18.253%]** |

`E0 − E1` 沿用 first-Router-launch primary boundary，static route-independent precompute 位于边界之前。将 precompute 加回后的保守 diagnostic 为 **+3.498 ms**，95% CI **[+1.400, +5.549] ms**；因此 `16.423%` 不是包含全部新增准备成本的 production/E2E 加速。

在相同 fast data-prep 下，optimized progressive E1 尚未优于 optimized delayed D1：`D1 − E1 = −3.093 ms`，0/3 seeds、0/5 families positive。当前诊断指向 cross-rank collective/rendezvous，而不是 data-prep 本身；fast data-prep 保留为 baseline，但 optimized progressive timing value 仍待解决。

### 4.4 Correctness

- R3 formal：4,915,200 token records，legality/token integrity 100%；
- R4 formal：300/300 paired comparisons、1,200/1,200 rank-arms PASS；
- forward/return lost、duplicate、wrong destination、wrong return、wrong position、corruption 均为 0；
- unrevealed execution、future access、action/checker divergence 均为 0；
- Early/Delayed final outputs equivalent。

## 5. 实现结构

```text
RevealAICCL/
├── rlccl/
│   ├── uncertainty/      # revealed-only state and fail-closed semantics
│   ├── scheduling/       # compiled scheduler and incremental state
│   ├── transport/        # variable A2Av, full-MoE and fast data-prep
│   ├── traffic/          # traffic generators and views
│   └── utils/            # utilities; legacy XML exporter is not a runtime backend
├── extensions/r2_event_bridge/
├── scripts/              # experiment runners and analyzers
├── tests/                # semantic, equivalence and correctness tests
└── docs/                 # preregistrations, reports and evidence chain
```

内部 Python package 暂时保留名称 `rlccl` 以维持兼容性。

## 6. 快速检查

项目目前没有锁定的通用 environment 文件。CPU semantic tests 需要 Python、PyTorch、NumPy 和 pytest：

```bash
python -m pytest -q \
  tests/test_no_future_leakage.py \
  tests/test_uncertainty_environment.py \
  tests/test_r2_compiled_scheduler.py \
  tests/test_r3_reference_a2av.py \
  tests/test_r4_reference_full_moe.py
```

GPU reference smoke 需要 Linux、2×CUDA GPU、NCCL 和 PyTorch C++ extension toolchain：

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/run_r4_a0_c0_full_moe.py \
  --output-dir outputs/r4_a0_c0_smoke \
  --case balanced \
  --allow-smoke
```

Smoke test 不能替代冻结 corpus、seed、hash/read-back 和 paired statistics 的 formal evidence。

## 7. 局限

- 结果来自 2×V100、single-node、双 rank reference environment；尚未验证 multi-node RDMA 或 production deployment。
- Router、packing、expert MLP 与 combine 均为 reference implementation。
- 当前 backend 是 PyTorch distributed NCCL；R6 MSCCL integration 已完整回退，仓库中没有 active MSCCL incremental runtime。
- R3/R4 存在 traffic-family heterogeneity，结论是 corpus-wide paired median，而非 universal speedup。
- NCCL count exchange、rank skew 和 collective rendezvous 仍有 heavy tail。
- Expert compute 和 return/combine 当前不 progressive。
- `outputs/` 默认不进入 Git；大型 raw traces 由报告中的 hash/read-back provenance 管理。
- L1 original raw artifacts 已丢失，只保留历史 derived summary，不允许重造 raw 冒充原证据。

## 8. 文档

- [完整中英双语项目报告](docs/UNKNOWN_TRAFFIC_PROGRESSIVE_AICCL_PROJECT_REPORT_BILINGUAL.md)
- [Evidence Repair](docs/phase_r0/EVIDENCE_REPAIR_REPORT.md)
- [Compiled Scheduler Equivalence](docs/phase_r2/R2_C0_REPORT.md)
- [Integrated Fast Path](docs/phase_r2/R2_F0_REPORT.md)
- [Variable-Size AlltoAllv Formal Report](docs/phase_r3/R3_F0_REPORT.md)
- [Full-MoE Formal Report](docs/phase_r4/R4_F0_REPORT.md)
- [Fast Data-Prep Report](docs/phase_r5/R5_P3_REPORT.md)

## English Abstract

RevealAICCL enables safe progressive scheduling when MoE AlltoAllv traffic is produced incrementally by the Router. Only completed Router chunks may update scheduler state or enter communication. The system combines a native CUDA-event bridge, compiled route templates, incremental state, deterministic binding and guards, Router-derived packing, real uneven-split NCCL forward/return communication, a reference expert MLP, and token-position combine.

On two V100 GPUs, formal paired evaluation reports a **+0.829 ms** median improvement for progressive forward AlltoAllv and **+2.801 ms** for the complete reference MoE path, with 3/3 formal seeds positive. Fast data preparation reduces median per-descriptor packing by **13.214×** and count construction by approximately **21.71×**. Under the inherited first-Router-launch boundary, E0→E1 improves by **78.872 ms**; a precompute-inclusive diagnostic is **+3.498 ms**. The active backend is PyTorch distributed NCCL, not MSCCL/MSCCL++ or a production MoE runtime.
