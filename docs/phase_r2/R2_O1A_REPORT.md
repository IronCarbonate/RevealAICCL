# Phase R2-O1A：Device Scheduling Diagnosis

更新日期：2026-08-10  
状态：**O1A COMPLETE / PENDING SUPERVISOR**  
边界：仅完成 O1A；O1B、R3、formal E2E 均未运行或获准。

## 1. 结论

严格 delayed-communication control 已完成。提前提交 C 相对 final-router 后提交 D，
在计入 router 干扰后的 distributed combined makespan 上观察到正收益：

- 60 个 distributed seed/trial paired samples；
- `paired_gain = T_D - T_C` p50 = **3,449.724µs**，p95 =
  **167,634.769µs**；
- paired-median 10,000-bootstrap 95% CI =
  **[2,718.223, 5,165.632]µs**；
- 三个独立 seed 的 median gain 均为正。

同时，device-plane 诊断不是单一瓶颈：

- C 相对 B 的 router slowdown median = **+775.659µs**，95% CI
  **[743.516, 815.579]µs**，约 B p50 的 **+19.14%**；
- C 的 submit-call-start→GPU-start p50/p95 =
  **105.770/22,242.998µs**；
- rank-start absolute skew p50/p95 = **715.747/14,519.292µs**；
- 因此按运行前冻结的分类规则，建议 Supervisor 将主因判为
  **C：launch/rendezvous 与 resource contention 同时存在**。

这不是 O1B scientific gate 的裁决。特别是 seed 4044 的 delayed-D 出现巨大
rank-rendezvous 长尾，必须原样保留，不能把 167.6ms p95 当作一般化收益大小。

## 2. A/B/C/D control

三组 canonical seeds 为 4042/4043/4044；每个 seed、rank、mode 各 20 trials：

- A：reference router only；
- B：router + EventBridge + compiled runtime，无通信；
- C：B + 每个 revealed action 后立即 real NCCL async all-reduce；
- D：与 C 生成完全相同的 action/descriptor，但所有通信仅在 final-router
  completion 后提交。

模式顺序在运行前冻结为：4042=`A,C,D,B`，4043=`B,D,C,A`，
4044=`D,A,B,C`。C/D 都执行 7 次 64-byte descriptor all-reduce，
每 rank/trial 总计 448 bytes；router、tokens、seeds、actions、operation 均不变。

C/D 严格控制结果：**120/120 rank-trial pairs** 的 ordered actions、descriptor
payload、communication count 和 bytes 全部相等。

## 3. Timeline 审计

六份 Kineto/CUPTI traces 中：

- 3,840/3,840 router chunks 由 32,768-byte D2H delimiter 重建；
- 1,680/1,680 NCCL GPU kernels 通过 CPU communication external-id 精确关联；
- association failure = **0**；
- router stream = 14，NCCL internal stream = 22；
- D 的 NCCL GPU-start-before-final = **0/840**。

主 makespan 样本单位是 distributed seed/trial：从两个 rank 中最早的 router
GPU start，到两个 rank 中最后的 router/NCCL GPU completion。不能将 collective
拆成两个独立 rank 样本冒充 distributed makespan。作为诊断保留的 120 个
rank-local pairs，其 gain p50 = 727.532µs，bootstrap 95% CI
[558.252, 1,781.300]µs。

跨 rank timestamp 可比性用相同 host 的 `time.monotonic_ns` NCCL call marker
独立核对；三个 seed 的 rank-median offset absolute difference 分别为
0.367/0.689/1.122µs，最大 **1.122µs**。

## 4. Combined makespan

| metric | count | p50 (µs) | p95 (µs) | p99 (µs) | max (µs) |
|---|---:|---:|---:|---:|---:|
| T_C, early communication | 60 | 16,190.684 | 33,593.086 | 36,203.950 | 36,701.378 |
| T_D, delayed communication | 60 | 23,097.784 | 177,851.879 | 180,159.234 | 181,245.510 |
| paired T_D−T_C | 60 | **3,449.724** | **167,634.769** | 172,670.610 | 175,796.742 |

| seed | pairs | gain p50 (µs) | gain p95 (µs) | median bootstrap 95% CI (µs) | positive pairs |
|---:|---:|---:|---:|---:|---:|
| 4042 | 20 | 3,168.925 | 5,279.663 | [2,107.875, 3,727.081] | 90% |
| 4043 | 20 | 2,113.193 | 3,503.095 | [1,691.674, 2,664.549] | 95% |
| 4044 | 20 | 139,984.940 | 170,763.140 | [131,989.541, 156,954.077] | 100% |

4044 的数值由 delayed-D 的极端跨 rank rendezvous/queued execution 主导；
它证明该 run 为正，但也证明 transport tail 不稳定。O1A 不据此选择 transport，
也不把该尾部外推到其他 workload。

## 5. Router interference

| condition | count | final-router p50 (µs) | p95 (µs) | p99 (µs) |
|---|---:|---:|---:|---:|
| A router only | 120 | 2,566.496 | 2,714.435 | 3,279.300 |
| B router + runtime | 120 | 4,053.159 | 4,573.097 | 5,965.437 |
| C router + runtime + early NCCL | 120 | 4,880.290 | 5,124.014 | 7,926.411 |
| D router + runtime, delayed NCCL | 120 | 4,070.504 | 4,313.131 | 5,622.279 |

Paired median deltas：

- B−A = +1,490.806µs，95% CI [1,472.967, 1,507.830]µs；
- C−B = **+775.659µs**，95% CI **[743.516, 815.579]µs**；
- D−B = +15.104µs，95% CI [-0.159, 27.264]µs。

因此 early NCCL 显著拖慢 router，而 delayed communication 在 router timed
interval 内没有显著 slowdown。C 的 per-chunk router duration：rank0
p50/p95 = 485.229/571.952µs，rank1 = 491.741/569.406µs。

## 6. Launch/rendezvous decomposition

以下均来自统一 CPU/GPU trace 中的 submit call 与 NCCL kernel，而非 API return
推断：

| C metric | rank | count | p50 (µs) | p95 (µs) | p99 (µs) |
|---|---:|---:|---:|---:|---:|
| submit-call-start→GPU-start | 0 | 420 | 8,026.560 | 26,238.651 | 30,540.232 |
| submit-call-start→GPU-start | 1 | 420 | 80.152 | 111.144 | 1,145.550 |
| NCCL kernel duration | 0 | 420 | 722.908 | 14,527.010 | 26,674.839 |
| NCCL kernel duration | 1 | 420 | 7.887 | 12.906 | 726.368 |

合并 rank 后 submit-call-start→GPU-start p50/p95/p99 =
105.770/22,242.998/29,736.940µs；NCCL kernel duration =
21.199/4,991.694/21,980.838µs。kernel duration 包含 collective rendezvous/device
waiting，不能解释为纯链路传输时间。

C rank-start absolute skew p50/p95/p99 =
715.747/14,519.292/26,667.273µs；D 为
201.903/27,117.957/44,282.264µs。该 rank asymmetry 与长尾直接支持
launch/rendezvous 问题存在。

## 7. Queued 与 concurrent execution

C 的 840 个 collectives 中：

- GPU-start-before-final：468/840 = 55.714%；early-only 为
  **468/720 = 65.000%**；
- 与 future-router kernel 真正 coexist：347/840 = 41.310%；early-only 为
  **347/720 = 48.194%**；
- start-before-final 但没有 kernel coexistence：121；
- post-final/queued：372。

| trigger | events | start-before-final | positive coexistence | overlap p50/p95 (µs) |
|---:|---:|---:|---:|---:|
| 0 | 120 | 120 | 90 | 12.432 / 789.335 |
| 1 | 120 | 74 | 57 | 0 / 123.451 |
| 2 | 120 | 72 | 58 | 0 / 124.583 |
| 3 | 120 | 70 | 44 | 0 / 39.503 |
| 4 | 120 | 67 | 49 | 0 / 7.751 |
| 5 | 120 | 65 | 49 | 0 / 6.909 |
| 7 | 120 | 0 | 0 | 0 / 0 |

每 rank-trial 的 actual-overlap sum p50/p95/p99 =
72.352/790.104/792.047µs。按 collective、含 zero 的 overlap p50/p95/p99 =
0/669.927/787.954µs；只看 347 个正 overlap，p50/p95/p99 =
5.792/756.325/790.377µs。

## 8. Semantic/safety

全部 360 个 B/C/D runtime rank-trials：

- runtime BFS = 0；full rebuild = 0；unrevealed execution = 0；
- candidate/action/checker/holder divergence = 0；
- legality = **2,520/2,520 actions**；
- token integrity = **360/360 trials**；
- partial_current_only、partial_shards@75%、checkpoint8、deterministic/fail-closed
  语义保持不变。

## 9. O1A 停止点

O1A 已完成，建议 Supervisor 归类为 **C：both**。结果同时显示：

1. early communication 具有正 combined-makespan value；
2. 当前 T0 transport 的 launch/rendezvous 极不稳定；
3. early NCCL 对 router 的 device contention 显著。

本轮到此停止。未运行 T0/T1/T2/T3 的 O1B intervention comparison，未选择
transport，未申请或进入 R3 real variable-size AlltoAllv。
