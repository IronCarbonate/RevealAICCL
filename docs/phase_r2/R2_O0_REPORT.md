# Phase R2-O0：True Router–Scheduler–NCCL Device Overlap

更新日期：2026-08-10  
状态：**TECHNICAL FAIL / PENDING SUPERVISOR**  
结论边界：mechanism-only；未运行 formal E2E，未实现 AlltoAllv。

## 1. 正式结论

Host-side submit-before-final 在三组 canonical runs 中复现，但 CUPTI GPU timeline
推翻了“API return 即 GPU overlap”的隐含推断：

- host submit-return-before-final：**720/840 = 85.714%**；
- early NCCL GPU-kernel-start-before-final：**447/720 = 62.083%**；
- 预注册 primary target：≥75%；**FAIL**；
- actual router-kernel/NCCL-kernel coexistence：**324/840 = 38.571%**；
- 三个 seed 均出现过正 overlap，但未达到冻结的 per-run stability 条件；
- **R2-O0 = TECHNICAL FAIL / PENDING SUPERVISOR**。

因此当前证据证明“真实 device overlap 偶发存在”，但不能证明稳定的
router future computation || compiled AICCL || NCCL execution pipeline。

## 2. Protocol 与 trace 可审计性

Canonical seeds 在运行前冻结为 4042/4043/4044；每个 seed、每个 rank 对
A/B/C 各运行 20 trials：

- A：router-only；
- B：router + EventBridge + compiled scheduler/guard；
- C：B + descriptor binding + real NCCL `all_reduce(async_op=True)`。

为控制 condition order，使用 Latin square：4042=A/B/C，4043=B/C/A，
4044=C/A/B。三组 run 均保持 8×4096×2048 reference-router workload、
partial_shards@75%、checkpoint8 和 partial_current_only。

统一 device timeline 来自 PyTorch Kineto/CUPTI Chrome trace，不使用 NCCL API
return 推断 GPU start/end。六份 raw traces 中：

- 2,880/2,880 router chunks 由 stream 14 上每 chunk 的 32,768-byte D2H
  delimiter 无歧义重建；
- 840/840 `record_param_comms` CPU-op External id 精确匹配 NCCL GPU kernel；
- association failures：**0**；
- router stream：14；descriptor H2D stream：18；NCCL internal stream：22；
- 每个 NCCL kernel 的实际 GPU start/end 均直接取自 CUPTI `kernel` event。

`actual_overlap_i` 是 NCCL kernel interval 与 trigger 后所有 future-router
kernel intervals 的交集总长度，不是两个 envelope 的粗略相交。

## 3. Host 与 GPU margins

| metric | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| host margin | 840 | 1909.057 | 4094.964 | 4217.562 | 6439.702 |
| GPU margin, all eligible | 840 | 419.854 | 4075.152 | 4236.126 | 6451.544 |
| GPU margin, early only | 720 | 1148.393 | 4098.417 | 4238.183 | 6451.544 |

正 host margin 不能保证 NCCL kernel 已启动：host fraction 85.714%，而 early GPU
fraction 仅 62.083%。

## 4. Actual overlap duration

| population | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| all eligible，含 zero | 840 | **0.000** | 130.178 | 775.720 | 790.621 |
| early eligible，含 zero | 720 | **0.000** | 150.887 | 785.981 | 790.621 |
| positive-overlap only | 324 | 5.343 | 719.387 | 787.375 | 790.621 |

all-eligible device-overlap fraction 为 **324/840 = 38.571%**；early-only 为
**324/720 = 45.000%**。

### Per-trigger results

| trigger chunk | eligible | GPU start-before-final | actual positive overlap |
|---:|---:|---:|---:|
| 0 | 120 | 98 / 120 = 81.67% | 71 / 120 = 59.17% |
| 1 | 120 | 75 / 120 = 62.50% | 64 / 120 = 53.33% |
| 2 | 120 | 72 / 120 = 60.00% | 56 / 120 = 46.67% |
| 3 | 120 | 70 / 120 = 58.33% | 45 / 120 = 37.50% |
| 4 | 120 | 67 / 120 = 55.83% | 44 / 120 = 36.67% |
| 5 | 120 | 65 / 120 = 54.17% | 44 / 120 = 36.67% |
| 7 / checkpoint8 | 120 | 0 / 120 | 0 / 120 |

## 5. Three-run stability

| seed | early GPU start-before-final | actual positive overlap, all eligible | stability notes |
|---:|---:|---:|---|
| 4042 | 71.67% | 47.50% | rank0/rank1 分别 20/20 trials 有正 overlap |
| 4043 | 64.17% | 33.57% | rank0 20/20；rank1 17/20 trials |
| 4044 | 50.42% | 34.64% | rank0 仅 **1/20**；rank1 20/20 trials |

三个 seed 的 early-start fraction 均低于 75%；seed4044 rank0 也不满足“至少三个
独立 trials 有正 overlap”的冻结稳定性定义。因此 three-run condition = **FAIL**。

## 6. Router interference

### Router final GPU latency

| condition | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| A router-only | 120 | 2549.040 | 2635.107 | 3391.971 | 3547.945 |
| B router+runtime | 120 | 4067.560 | 4371.370 | 5044.070 | 5112.512 |
| C router+runtime+NCCL | 120 | 4886.770 | 5134.065 | 6495.730 | 7270.580 |

Paired 10,000-bootstrap median deltas：

- scheduler/runtime interference B−A：**+1522.263µs**，95% CI
  [1493.208, 1540.598]µs，约 A median 的 **+59.72%**；
- NCCL-induced interference C−B：**+808.299µs**，95% CI
  [783.418, 839.482]µs，约 B median 的 **+19.87%**；
- NCCL 对 router 的显著正 slowdown：**YES**。

### Per-chunk router GPU latency p50

| condition | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 209.82 | 193.05 | 190.65 | 190.08 | 190.34 | 190.38 | 191.26 | 189.37 |
| B | 208.49 | 199.52 | 348.46 | 359.92 | 376.27 | 381.10 | 413.47 | 194.02 |
| C | 210.21 | 251.26 | 479.21 | 494.94 | 523.20 | 512.88 | 547.93 | 198.75 |

NCCL GPU-duration distribution 包含 rank rendezvous/device-side waiting：p50
26.719µs、p95 4553.398µs、p99 39299.017µs、max 415558.112µs。当前 trace
不将其拆成纯数据移动与等待，因此不对内部原因作超出证据的归因。

## 7. Semantic/safety preservation

三组 runs 的 B/C shadow checks 全部通过：

- runtime BFS = 0；full rebuild = 0；unrevealed execution = 0；
- candidate/action/checker/holder divergence = 0；
- legality = **1680/1680 scheduler events**；
- token integrity = **240/240 B/C rank-trials**；
- 75%/checkpoint8、partial_current_only、deterministic/fail-closed 语义不变。

未运行 formal E2E、real AlltoAllv、packing/GEMM/combine、DeepEP；未优化
scheduler，未修改 workload/chunk，未恢复 predictor/robust/adaptive。

## 8. Gate 与建议

R2-O0 的 primary ≥75% early GPU-start-before-final 条件失败，three-run stability
也失败。虽然真实 device coexistence 已被 CUPTI 直接观察到，但覆盖率和 rank/run
稳定性不足，且 NCCL 显著拖慢 router。

结论：**R2-O0 = TECHNICAL FAIL / PENDING SUPERVISOR**。

不建议当前进入 R3 real variable-size AlltoAllv。AlltoAllv 只会扩大 communication
mechanism 与负载复杂度，不能修复本 Gate 已暴露的 NCCL GPU-start 延迟、rank
不对称和 router interference。后续方向须由 Supervisor 另行裁决；本阶段立即停止。
