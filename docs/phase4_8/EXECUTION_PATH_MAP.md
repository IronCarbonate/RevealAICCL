# Phase 4.8-0：真实执行路径图

更新日期：2026-08-04

## 1. 目标路径 vs 当前仓库

| 路径 | 真实/高保真定义 | 当前仓库状态 | Phase 4.8-1 落点 |
|---|---|---|---|
| router | token→expert top-k 选择 | 无（仅 legacy decoder 启发式剪枝 `decoder.py:431`，冻结） | 新建最小高保真 router shim |
| top-k | 每 token 的 k 个 expert 选择 | 无 MoE 语义 | router shim 内实现 |
| token/shard readiness | shard 完成事件 | proxy `DemandRevealProcess`/`partial_shards`（`uncertainty/reveal.py`） | 加真实时间戳事件 |
| reveal | 已揭示 token 集 | `DemandRevealProcess.observation_for_stage`（proxy，存在） | 映射到真实事件流 |
| scheduler | fast-layer partial | `partial_current_only`（`phase4_experiment.py` run_public_episode；冻结） | 复用（不改语义） |
| dispatch | token→rank 发送 | 无（`config.py` 仅 AllToAll 常数 0.2） | 新建高保真 dispatch 计时 |
| GEMM | expert 计算 | 仅 legacy V1 torch 模型（冻结，非执行语义） | 新建最小 GEMM kernel 计时 |
| all-to-all | rank 间交换 | 无 | 模拟或单 rank 占位 + 计时 |
| allreduce/allgather | 全局聚合 | 无（`config.py` 常数 0.3） | NCCL 单 rank 或模拟 + 实测 |
| NCCL | 通信库 | 2.27.3（torch 内置）可用 | 单 rank 可用；多 rank 需 L2 |
| NVSHMEM/DeepEP | 专用通信 | **缺失** | 需安装/需 L2+ |
| event/timing | 事件账本与时间戳 | proxy `_event_row`/read-back 存在；真实时间戳无 | CUDA event + torch profiler + CPU perf_counter |

## 2. 数据流（目标 L1 高保真）

```text
token 到达(计数, CPU) ─► shard ready(事件) ─► reveal(partial_shards @75%, ckpt8)
        │                                    │
        ▼                                    ▼
router top-k(CPU/GPU) ─► scheduler(partial_current_only, 冻结)
        │                                    │
        ▼                                    ▼
dispatch(计时) ─► expert GEMM(CUDA event) ─► allreduce/allgather(计时, 模拟或单 rank)
        │
        ▼
job/microbatch 完成(事件)
```

## 3. baseline 与候选 profile 的映射（唯一区别 = reveal profile）

| | D0（current baseline） | D1（候选 profile） |
|---|---|---|
| reveal mode | current/default | partial_shards |
| full reveal checkpoint | 16 | 8 |
| budget | 默认 | 75% |
| scheduler | partial_current_only | partial_current_only（相同） |
| checker/fail-closed | 开 | 开 |

两 profile 运行在同一高保真执行层；除 reveal 参数外无任何差异（公平比较要求）。

## 4. 最小代码改动位置（仅 Phase 4.8-1 实施，本轮不改）

- 新建 `outputs/phase4_8/deployment_validation/`（或独立模块）承载 router shim、dispatch/GEMM 计时、reveal profile 开关；
- 复用：`uncertainty/reveal.py`（partial_shards 语义）、`scheduling/robust_prefix.py` 原语、`partial_current_only` 循环；
- 不改 production：`phase4_experiment.py`、`execution.py`（checker）、frozen artifacts。
