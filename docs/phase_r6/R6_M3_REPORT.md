# R6-M3：MSCCL++ Post-Issue GPU-Start Diagnosis 报告

## 最终判定

**R6-M3 = Diagnosis COMPLETE / Correctness PASS / Legality PASS / Root Cause IDENTIFIED / NO VETO**

最重要的结论不是“MSCCL++ kernel 在 GPU 队列里被 Router 堵了几十毫秒”，而是恰好相反：R6-M2 把 Python transport adapter 入口误叫成了 put host issue。R6-M3 把 adapter、C++ wrapper、kernel launch call、launch return 和 GPU start 分开以后发现，正常路径里 put 根本没有在 future Router 运行期间进入 C++ 或完成 kernel enqueue。几十毫秒主要耗在进入 transport 之前的单线程 host 控制 backlog。

## 67.417 ms 到底耗在哪里

R6-M3 的 15 个 normal case、两个 rank、每 rank 7 个 descriptor，共得到 210 条完整时间线。新的 `ready → GPU-start` 中位数为 68.597 ms，与 R6-M2 的 67.417 ms 稳定复现。

| 分段 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| reveal/Router GPU end → CommittedAction | 67.336 ms | 140.482 ms | 149.259 ms | 152.830 ms |
| CommittedAction → staging GPU end | 1.334 ms | 2.341 ms | 3.147 ms | 3.210 ms |
| staging GPU end → kernel launch return | 0.041 ms | 0.138 ms | 0.301 ms | 0.333 ms |
| kernel launch return → put GPU start | 0.000235 ms | 0.002168 ms | 0.002778 ms | 0.003088 ms |
| Router GPU end → put GPU start | 68.597 ms | 143.686 ms | 151.544 ms | 155.305 ms |

按所有 descriptor 累计时间分解：

- Scheduler/control：97.888%。
- Packing/data dependency：2.008%。
- CPU/runtime kernel enqueue：0.103%。
- GPU queue/scheduling：0.001%。

这里的 scheduler/control 是一个时间段名称，不表示 `FastBinder` 本身跑了 67 ms。进一步拆开可以看到，EventBridge publish p50 只有 28.4 µs，冻结的 CPU packing p50 为 0.831 ms，descriptor digest/bookkeeping p50 为 11.190 ms，adapter 内 CPU byte pack p50 为 0.590 ms。最大的一段是 `state/binder/guard` 所在的 host 串行窗口，p50 54.453 ms；其中大量时间其实是后续 descriptor 已 ready，但主 Python 线程仍在依次完成前面 descriptor 的调度、digest、staging 和提交。

这个 backlog 很有规律：descriptor 0 的 ready→start p50 是 18.949 ms，随后逐个累积到 descriptor 6 的 142.409 ms。这不是 GPU 排队形状，而是单个 host consumer 顺序处理 7 个已经很快 ready 的 descriptor 的形状。

## 对十个必答问题的回答

### 1. 67.417 ms 主要耗在哪里？

主要耗在 **A. Scheduler/control latency**，更准确地说是 transport 之前的单线程 host descriptor backlog，占 97.888%。它包含当前 descriptor 等待主控制循环轮到自己，以及 scheduler、digest 和 bookkeeping；不是 FastBinder 单项耗时，也不是 GPU queue。

### 2. packing 是否阻塞 put？

有少量影响，但不是主因。CommittedAction 到 staging GPU end 的 p50 为 1.334 ms，只占约 2.008%。dependency-resolved control 提前完成同一个 staging event 后，ready→start p50 从 normal 的 68.597 ms 变为 70.438 ms，没有改善。因此 packing dependency 不是几十毫秒延迟的来源。

### 3. comm stream 是否错误等待未来工作？

否。每个 descriptor 创建独立 event，event 在该 descriptor 自己的 registered-buffer staging 后记录，并由 comm stream 等待同一个 event：

- `wrong_event_dependency = 0`
- `future_pack_dependency = 0`
- `event_reuse_hazard = 0`

不存在等待整个 packing stream tail、后续 descriptor、final Router 或复用旧 event 的证据。

### 4. kernel launch API 本身是否慢？

否。`cudaLaunchKernel` 调用到返回的 p50 为 7.21 µs，p95 为 12.51 µs。C++ wrapper 进入到 launch return 的 p50 为 13.35 µs。即使 p99 受少数 host 抖动影响达到约 181 µs，也远小于 68 ms。

### 5. kernel enqueue 后还在 GPU queue 等多久？

按 CUDA event 映射后的非负值，p50 为 0.235 µs，p95 为 2.168 µs，均值约 0.604 µs。host steady-clock 与 CUDA event 映射在亚微秒附近会出现轻微负偏差，结果文件保留了 raw 值，并在统计时只把它视作时钟映射误差。结论是没有几十毫秒 GPU queue delay。

### 6. queue delay 时 GPU 主要运行什么？

几乎没有可归因的活动。210 条 normal queue 窗口累计只有约 126.8 µs，时间线与 Kineto 都没有发现这些窗口被 future Router、staging 或前一个 put 覆盖。这里不是“GPU 正忙”，而是 launch return 后几乎立即开始。

### 7. put 与 Router 为什么没有 concurrent residency？

因为两者没有同时进入 GPU 可调度状态。210/210 个 normal descriptor 的 put GPU start 都在 final Router GPU end 之后；更关键的是 0/210 个 native wrapper 入口早于 final Router，0/210 个 adapter 入口早于 final Router。没有并发驻留不是 V100 资源不足，而是 host 提交时序根本没有制造并发窗口。

Kineto 显示 `put_and_signal_kernel` 只有 grid `[1,1,1]`、block `[256,1,1]`、28 registers/thread、0 shared memory；Router 使用 Volta SGEMM kernels。这些数据也不支持 put 因资源太大而无法驻留的解释。

### 8. 是否存在 single-comm-stream head-of-line blocking？

否：

- `put_blocked_by_previous_wait = 0`
- `put_blocked_by_previous_put = 0`
- `put_blocked_by_unresolved_event = 0`

所有 remote wait 仍只在冻结的 forward completion 边界统一入队，put 前面没有 wait。正常 queue 窗口也没有被前一个 put 覆盖。

### 9. rank rendezvous 在剩余 delay 中还有多少？

MemoryChannel 已经没有 collective rendezvous。匹配 descriptor 的跨 rank GPU-start skew p50 为 1.767 ms、p95 为 9.398 ms。这是两个 rank 各自 host backlog 和 staging 进度不同造成的剩余异步偏差，不是 collective 必须同时进入的等待。相对于 ready→start p50 68.597 ms，它不是主因。

### 10. 最可能恢复真实 overlap 的优化点是什么？

候选后续是：先消除 host 端逐 descriptor 串行 backlog，使 ready descriptor 能及时进入 adapter/native enqueue；同时把 registered-buffer staging 做成真正的 pinned/asynchronous pipeline，并把 digest/诊断 bookkeeping 移出关键提交路径。只有在 put 确实能于 future Router 结束前 enqueue 以后，才有理由研究 stream priority 或 kernel residency。

本阶段只记录这个 `candidate_followup`，没有实施。

## 两个因果 control

Router-absent control 让 producer 在当前 descriptor 已提交后才发后续 Router，因而不让多个 descriptor 一次性在 host 侧形成 backlog。它把 ready→start p50 从 68.597 ms 降到 18.087 ms，而 GPU queue p50 仍只有 0.033 µs。这证明 future Router 的作用是让 descriptor 产生速度远快于单线程 consumer，而不是让已 enqueue 的 put kernel占不到 GPU。

Dependency-resolved control 在 real put enqueue 前显式完成当前 descriptor 自己的 staging event。ready→start p50 为 70.438 ms，GPU queue p50 为 0.053 µs，与 normal 无实质改善。这排除了 pack-event dependency 是主因。

## 同步和 stream 审计

Router、packing/default、communication、return、expert 和 count stream 的实际句柄均已记录，priority 全部保持默认值 0。正常 progressive submit 热路径中没有 `cudaDeviceSynchronize`、`torch.cuda.synchronize`、stream synchronize 或 event synchronize。

存在的同步点是：primary 之前用于统一时钟原点的 origin event；所有 outstanding put/wait 完成后的冻结 forward boundary；以及只存在于 dependency-resolved control 的 staging event synchronize。它们都没有被伪装成 normal 热路径行为。

## 正确性、合法性与产物

15 个 normal case 和 6 个 control case 全部通过完整 MoE 正确性。lost、duplicate、wrong-destination、corruption、future、unrevealed、stale 和 scheduler divergence 均为 0，control 的 Router、scheduler、descriptor、expert identity 和最终输出保持冻结等价。

产物：

- `outputs/phase_r6/m3/r6_m3_results.json`
- `outputs/phase_r6/m3/r6_m3_descriptor_timeline.csv`
- `outputs/phase_r6/m3/r6_m3_kernel_timeline.csv`
- `outputs/phase_r6/m3/r6_m3_kineto_rank0.json`
- `outputs/phase_r6/m3/r6_m3_kineto_rank1.json`

归因总结：A dominant（97.888%）；B secondary（2.008%）；C negligible（0.103%）；D、E、F 均 negligible/unsupported。R6-M3 在 root cause 和 controls 闭环后停止，没有进入 R6-M4，也没有修改 priority、kernel、descriptor 或 scheduler。
