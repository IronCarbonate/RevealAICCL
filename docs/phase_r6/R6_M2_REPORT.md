# R6-M2：MSCCL++ Progressive Pipeline Pilot 报告

## 结论

**R6-M2 = Correctness PASS / Mechanism PASS / Performance FAIL / NO VETO**

这次试验说明两件事。第一，MSCCL++ 已经能安全接进完整 MoE 前向流水线：四个实验臂的 Router、调度动作、前向分片、专家结果、NCCL 返回和最终合并结果全部一致，没有丢失、重复、发错位置、读取未揭示数据或过期动作。第二，MSCCL++ 确实减轻了渐进发送时的跨 rank 等待，但这批 pilot 数据还不能证明它稳定缩短了完整 MoE 时间。

按预注册规则，正确性通过就没有 VETO；性能失败不否定实现，只表示当前证据不够稳定，不能宣称提速。

## 实验范围

- 3 个固定种子：13042、13142、13242。
- 5 类负载：balanced、skewed、all-to-one-like、zero-sized-pair、multiple-progressive-shards。
- 每类每种子 3 个任务，共 45 组严格配对样本。
- 每组运行 NCCL-D、NCCL-P、MSCCLPP-D、MSCCLPP-P 四臂，共 180 次完整 MoE。
- 主计时从两个 rank 中最早一次 Router 启动开始，到最晚一次返回并完成 combine 为止。
- 四臂使用相同 token、Router 参数、chunk、scheduler、packing、descriptor、专家计算和返回路径；只改变前向通信后端以及 delayed/progressive 发起边界。

## 配对性能结果

这里的 `Gain = Delayed - Progressive`，正数表示 progressive 更快。

| 指标 | 配对中位数 | 95% bootstrap CI | 正方向样本 |
|---|---:|---:|---:|
| Gain_NCCL | -1.932 ms | [-8.027, 2.439] ms | 18 / 45 |
| Gain_MSCCLPP | +1.498 ms | [-1.647, 9.244] ms | 25 / 45 |
| Gain_MSCCLPP - Gain_NCCL | +4.802 ms | [-4.681, 13.327] ms | — |

MSCCL++ 的点估计比 NCCL 好，但两个关键置信区间都跨过 0，说明波动仍足以改变结论。三个种子的 `Gain_MSCCLPP` 中位数分别是 +5.340 ms、+0.822 ms、-1.257 ms。预注册要求三个种子方向都为正，因此 **Performance FAIL**。

五类负载的 `Gain_MSCCLPP` 中位数也不一致：all-to-one-like 为 +6.916 ms，balanced 为 +0.382 ms，multiple-progressive-shards 为 -0.006 ms，skewed 为 -1.905 ms，zero-sized-pair 为 +11.629 ms。这进一步说明当前效果依赖负载形状，尚未形成稳定的普遍提速。

## 代码方面遇到的问题与解决办法

### 1. 旧脚本偷偷依赖历史输出目录

新的 R6-M2 入口要复用 R2/R3 中的常量、CUDA event bridge 和负载定义。第一次上服务器运行时，Python 报 `No module named 'reference_router'`。原因不是服务器代码旧，而是被复用的旧脚本把 `outputs/phase4_10/.../reference_router.py` 当成了代码依赖。本地恰好留着这份历史产物，所以本地导入不报错；干净的 R6 服务器工作副本没有它，问题才暴露出来。

解决办法是把 Router 固定实现正式放进 `rlccl.transport.reference_router`，并把 R2、R3 和 R4 的复用入口都改为从正式包导入。这样运行代码不再依赖某次旧实验留下的输出文件，服务器和本地使用的是同一份 Router 实现。

### 2. MSCCL++ 原来的 buffer 布局只能传很小的测试记录

R6-M1 的桥接层按固定 32-byte record 设计，足够验证 primitive，却装不下完整 MoE 的 token 元数据和 2048 维 FP32 feature。若直接套用，要么截断数据，要么改变原来的 descriptor 边界，都会破坏四臂公平比较。

解决办法是让 registered-buffer layout 支持可配置、8-byte 对齐的 record 大小。R6-M2 把每条记录编码成 72-byte 元数据加 8192-byte feature，共 8264 bytes；descriptor、token 顺序、目标 rank 和字节内容保持不变。每个 rank 只预分配并注册一块大 buffer，后续所有分片都复用它，没有在热路径反复注册内存。

### 3. 需要允许多个分片在途，不能每发一个就立刻等

如果 `put + signal` 后马上 `wait`，代码虽然能跑，却会把 MSCCL++ 又做成逐分片同步，正好失去 R6-M2 想验证的能力。

解决办法是把提交和完成拆开：descriptor 通过 reveal 与 guard 后立即提交对应的远端 put/signal；同一轮可以连续提交 7 个 descriptor；只在原来冻结的 forward completion 边界排入匹配 wait 并统一同步。正式 45 组样本中，每个 MSCCL++ arm 都记录了多个真实 put 和 wait，且 future、unrevealed、stale 三类违规计数全部为 0。

### 4. 只有 CPU 时间看不出“发起了”和“GPU 真开始了”的差别

R5-P4 已经发现 NCCL-P 的 host 调用很早，但 GPU 通信可能因为跨 rank rendezvous 晚很久才开始。若 R6-M2 只记录 Python 函数进入时间，就无法回答 MSCCL++ 是否解决了真正的等待。

解决办法是在 C++/CUDA bridge 内为每次 put 和 wait 记录 CUDA event，并把 event 时间映射到统一的 host steady-clock 时间轴；NCCL 路径也在 payload collective 前后记录 CUDA event。这样每个 descriptor 都有 ready、host issue、GPU start/end、signal 和 wait complete，可以直接算 ready skew、issue skew、GPU-start skew及通信持续时间。

### 5. 聚合脚本最初假定了不存在的 `pass` 子字段

正式运行本身已经完成且全部正确，但第一次聚合时，脚本把明细结构误记成 `correctness.pass` 和 `semantic.pass`。实际数据保留的是逐项错误计数、`final_combine_correct`、`token_integrity` 以及 `legal/total`，因此聚合时报了 `KeyError`。

解决办法不是修改原始结果，而是让聚合器按真实语义逐项判断：所有错误计数必须为 0，最终 combine 与 token 完整性必须为真，`legal == total`。修正后重新聚合同一批原始文件，最终正确性为 PASS。

## 机制诊断

MSCCLPP-P 的 descriptor 级中位 ready-to-GPU-start 为 67.417 ms，NCCL-P 为 125.171 ms；GPU-start skew 分别为 1.168 ms 和 12.072 ms。也就是说，MSCCL++ 的一侧 put 明显减少了“本 rank 已经准备好，却在等另一 rank 一起进入 collective”的时间。这与 `Gain_MSCCLPP - Gain_NCCL` 的 +4.802 ms 中位点估计方向一致，所以按预注册规则 **Mechanism PASS**。

不过，trace 中四臂的 Router/通信 GPU overlap descriptor 数都是 0。少量 progressive host issue 发生在 final Router 之前，但 GPU kernel 没有真正与 Router kernel 时间重叠。说人话就是：MSCCL++ 让通信更容易早点排进 GPU 队列，却还没让它和 Router 计算真正同时执行。当前完整 MoE 端到端只有小幅且不稳定的变化，这是最直接的解释。

## 产物

- `outputs/phase_r6/m2/r6_m2_results.json`：最终统计、环境和 verdict。
- `outputs/phase_r6/m2/r6_m2_raw_pairs.csv`：45 组配对结果。
- `outputs/phase_r6/m2/r6_m2_descriptor_trace.csv`：1260 条 descriptor 级时序记录。
- `scripts/run_r6_m2_pipeline.py`：四臂完整 MoE runner。
- `scripts/analyze_r6_m2_pipeline.py`：固定统计和 bootstrap 聚合器。

按照授权边界，本阶段在 pilot、正确性、性能和 rendezvous 诊断完成后停止；没有做 scheduler/packing/Router 的额外优化，也没有因为结果不够好而事后改阈值。
