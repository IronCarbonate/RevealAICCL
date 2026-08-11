# Phase R2-E0：Low-Latency Event Bridge

更新日期：2026-08-10  
状态：**Supervisor PASS / NO VETO；EventBridge frozen as R2 runtime substrate**

## 1. 结论

R2-E0 的工程目标已经达到：在 2× Tesla V100-SXM2-32GB 上，原生 C++
busy-poll event bridge 对 8,000 个真实 CUDA chunk completion event 的
`event completion → host/runtime ready visible` 保守 host-only 上界为：

| 指标 | 数值 |
|---|---:|
| aggregate p50 | 2.949µs |
| aggregate p95 | **4.743µs** |
| aggregate p99 | 5.322µs |
| aggregate max | 60.863µs |
| rank 0 p95 | 4.174µs |
| rank 1 p95 | 5.019µs |
| worst-rank p95 | **5.019µs** |
| 有效上界覆盖率 | **8,000/8,000（100%）** |

因此预注册的 `p95 < 100µs` 与 stretch `p95 < 50µs` 均通过。
本结论只说明 event bridge 可行，不表示 R2-C0/R2-F0/R2-O0 已完成。

## 2. 测量口径

每个有效样本满足：前一次 `cudaEventQuery` 返回 `cudaErrorNotReady`，下一次
返回 `cudaSuccess`。区间从前一次查询的 host monotonic **开始**时刻，到下一次
成功查询的 host monotonic **返回**时刻。真实完成时刻必在该区间内，因此它是
completion→host-ready 的保守上界；没有把 CUDA clock 与 CPU clock 相减。

每 rank 500 trials，每 trial 8 个独立 router chunk forward/event。setup 区域仅有
一次全局 synchronize，用于 warmup 和 event 句柄初始化；event bridge timed loop
没有 per-chunk `torch.cuda.synchronize()` 或 `event.synchronize()`。

timed detection path 使用：

- 单进程原生 C++ thread；
- CPU affinity 固定的 busy poller（rank 0/1 分别 core 95/94）；
- 每个 poller thread 显式 `cudaSetDevice(rank)`；
- 预分配、cache-line aligned 的固定 8-slot atomic ring/bitmap；
- timed path 无 JSON、pickle、multiprocessing queue、sleep 和 Python 动态分配。

成功查询本身的 p95 为 rank 0 `0.669µs`、rank 1 `0.890µs`。aggregate max
`60.863µs` 是单个尾部样本；预注册判定量为 p95，aggregate p99 仍为 `5.322µs`。

正式环境：2× Tesla V100-SXM2-32GB、world size 2、PyTorch 2.8.0+cu128、
CUDA runtime 12.8、Python 3.12.3；C++ extension build dependency 为
`ninja==1.11.1.4`。正式命令：

```bash
cd /root/autodl-tmp/RLCCL-main
PYTHONPATH=$PWD PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C \
  /root/miniconda3/bin/python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29514 scripts/run_r2_e0_event_bridge.py
```

## 3. R1 ready→scheduler 精确边界分解

R1 的 `ready_host_ns` 是 Python `event.query()==True` 返回之后记录的，因此
CUDA event query/polling 位于 `ready→scheduler_start` 区间之前，在该指标中的
贡献按定义为 0；旧 R1 没有记录真实 CUDA completion 对应的 host 时刻，不能
事后伪造这段数值。R2-E0 的 event bridge 微基准是对这一上游路径的独立替代测量。

R1 原始 timestamp 可将 `ready→scheduler_start` 精确拆成：

`worker_busy_queue + executor_dispatch/wakeup`。

其中 `worker_busy_queue = max(previous_checker_done - ready, 0)`，其余为 worker
可用后到 `scheduler_start` 的 ProcessPool/executor/IPC/wakeup 合并边界。不同分位数
不能直接相加。

| trigger chunk | ready→start p50/p95 | worker-busy p50/p95 | dispatch+wakeup p50/p95 | scheduler body p50/p95 |
|---:|---:|---:|---:|---:|
| 0 | 0.249 / 0.376ms | 0 / 0ms | 0.249 / 0.376ms | 2.136 / 2.206ms |
| 1 | 2.477 / 2.615ms | 2.399 / 2.531ms | 0.080 / 0.091ms | 3.274 / 3.344ms |
| 2 | 5.807 / 6.281ms | 5.723 / 6.093ms | 0.083 / 0.113ms | 4.084 / 4.172ms |
| 3 | 10.630 / 20.757ms | 9.806 / 10.411ms | 0.820 / 1.774ms | 4.936 / 5.474ms |
| 4 | 15.656 / 25.801ms | 15.569 / 25.717ms | 0.086 / 0.542ms | 5.738 / 5.999ms |
| 5 | 21.640 / 36.350ms | 21.394 / 31.571ms | 0.088 / 9.593ms | 6.708 / 6.993ms |
| 7 | 28.367 / 43.064ms | 28.278 / 42.972ms | 0.089 / 0.121ms | 9.204 / 9.797ms |

跨全部 280 records：worker-busy p50/p95 为 `9.816/32.561ms`；worker 可用后的
dispatch/wakeup p50/p95 为 `0.088/3.012ms`。chunk 0 没有 backlog，其 p95 仅
`0.376ms`；后续 chunk 的 10–34ms 主要是单 worker 串行处理造成的累计排队。

## 4. IPC、wakeup、Python/GIL 与 scheduler component 诊断

ProcessPool 内部把 feeder、serialization、pipe IPC、OS wakeup 和 worker dispatch
封装在一起，R1 timestamp 不能诚实地将它们继续拆成可相加的独立项。为避免伪精度，
本轮报告可观测边界，并另做 50 次 idle microdiagnostic：

| 项目 | rank 0 p95 | rank 1 p95 |
|---|---:|---:|
| pickle.dumps（diagnostic only） | 1.861µs | 1.869µs |
| ProcessPool submit call | 16.075µs | 16.914µs |
| submit→worker entry（IPC+wakeup 合并） | 108.592µs | 111.036µs |
| worker exit→parent result | 71.348µs | 78.075µs |

这些数据解释 idle path，不用于抵消或重写 R1 原始延迟。

对 unchanged observation/scheduler/checker 运行 50 repetitions、每 rank 350 个
stage records，得到：

| component | rank 0 p95 | rank 1 p95 |
|---|---:|---:|
| observation append | 59.559µs | 59.994µs |
| observation materialize | 841.272µs | 858.133µs |
| build view（Python/GIL） | 1.125ms | 1.126ms |
| enumerate（Python/GIL） | 6.529ms | 6.797ms |
| pack（Python/GIL） | 550.660µs | 557.751µs |
| bind | 17.573µs | 16.826µs |
| deterministic checker | 83.498µs | 84.308µs |

因此有两个层次的主要 latency source：

1. 对 `ready→scheduler_start`，主因是旧 R1 单 ProcessPool worker 的串行 backlog；
2. 对 `scheduler_start→commit`，主因是 Python observation/view/enumerate/pack，尤其
   enumerate，而不是 event query 或 NCCL submission API。

## 5. Gate R2-E0

| 要求 | 结果 |
|---|---|
| single-process native event bridge | PASS |
| pinned busy-poll thread | PASS，2/2 rank |
| preallocated ring/bitmap | PASS |
| timed path 禁止 JSON/pickle/queue/sleep | PASS |
| 有效保守上界覆盖率 ≥95% | PASS，100% |
| event→host-ready worst-rank p95 <100µs | PASS，5.019µs |
| stretch p95 <50µs | PASS，5.019µs |
| R1 latency decomposition | COMPLETE |
| scheduler semantics unchanged | PASS |
| formal E2E 未运行 | PASS |

Supervisor 已裁决 **R2-E0 = PASS / NO VETO**，并要求停止继续优化
EventBridge；当前实现冻结为 R2 runtime substrate。

## 6. 明确未完成与停止点

本轮只实现 EventBridge。以下组件仅完成架构设计，没有实现：

- StaticPlanCompiler；
- IncrementalState；
- FastBinder；
- StaticProof + DynamicGuard；
- IncrementalChecker。

未运行 formal E2E，未实现 real AlltoAllv、expert packing/GEMM/combine、DeepEP；
未改 75%/checkpoint8、partial_current_only 动作语义或 deterministic checker；未恢复
predictor/robust/adaptive；未更换/延长 workload。

R2-C0 已获授权并另行实施；R2-F0/R2-O0 未由 E0 自动启动。

## 7. Artifacts

- `extensions/r2_event_bridge/event_bridge.cpp`
- `scripts/run_r2_e0_event_bridge.py`
- `outputs/phase_r2/e0_event_bridge/r2_e0_results.json`
- `outputs/phase_r2/e0_event_bridge/r2_e0_readback.json`
- `outputs/phase_r2/e0_event_bridge/r2_e0_independent_readback.json`
- `docs/phase_r2/COMPILED_EVENT_DRIVEN_ARCHITECTURE.md`
- `docs/phase_r2/CHECKER_EQUIVALENCE_PROTOCOL.md`
