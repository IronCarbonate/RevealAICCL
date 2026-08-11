# Phase R2-O1B：Bounded Transport Interventions

更新日期：2026-08-10  
状态：**O1B PASS / NO VETO（Supervisor accepted）**  
边界：Transport selection 冻结为 **RETAIN T0 WITH LIMITATIONS**；后续 R3 不改变 O1B 历史结论。

## 1. 结论

四个预注册 transport 均已在 2× Tesla V100-SXM2-32GB、2 NCCL ranks、
3 个独立 seeds、20 distributed paired trials/seed 上完成严格 early/delayed 对照。

- **T0 PASS**：paired median `T_D−T_C` = **845.335µs**，bootstrap 95% CI
  **[661.897, 1,687.218]µs**，3/3 seed medians positive。
- **T1 FAIL**：总体 median 与 CI 为正，但 seed4044 median = **−3,675.363µs**，
  因此不满足 3/3 seeds。
- **T2 FAIL**：median = 780.394µs，但 CI = **[−629.628, 1,523.190]µs**，
  且 seed4044 median = **−91,269.019µs**。
- **T3 PASS**：paired median = **2,095.778µs**，bootstrap 95% CI
  **[1,251.219, 4,390.917]µs**，3/3 seed medians positive。

但 T3 相对 T0 没有达到运行前冻结的任何一项“至少 20% 改善”：launch p95、
rank-skew p95、router interference median、seed4044 delayed tail p99 均更差。
T1/T2 各改善一项，但 primary gate 失败。故没有 intervention 满足完整选择规则；
Supervisor 已正式选择：**RETAIN T0 WITH LIMITATIONS**。T1/T2 FAIL，T3 不进入主线；
不得继续增加 transport variant。R3 后续只可使用由 Supervisor 单独授权的 A2Av-T0 路径。

## 2. 预注册实现与公平控制

固定 transport：

- T0：每 action 一次 64-byte NCCL all-reduce。
- T1：router 使用 priority `-1` stream；communication 使用 priority `0` stream；
  operation 与 T0 相同。服务器 PyTorch 2.8 未暴露 priority-range API，但两个 stream
  均真实创建并在 trace 中分别为 stream 15 与 14；没有尝试其他 priority。
- T2：运行前唯一冻结为 **4-way slicing**；每个 64-byte descriptor 固定拆成
  4×16-byte NCCL all-reduce。
- T3：每 action 使用固定 64-byte `batch_isend_irecv`，没有 variable payload、
  packing 或 AlltoAllv。

每个 Ck/Dk 均保持同 tokens、router、AICCL actions、descriptor 语义、seed、
logical bytes 与 directional bytes。每 rank/trial 均为 7 actions、448 logical bytes、
896 directional wire bytes；T2 只将 7 operations 确定性细分为 28 primitive operations。

运行前冻结的 mode order：

- seed4042：`B,C0,D0,C1,D1,C2,D2,C3,D3`
- seed4043：`D1,C1,D2,C2,D3,C3,D0,C0,B`
- seed4044：`C2,D2,C3,D3,C0,D0,B,C1,D1`

选择改善阈值在看到结果前冻结为：相对 T0，在 launch p95、rank-skew p95、
`C−B` router-interference median、seed4044 delayed `T_D` p99 中至少一项降低 20%。

## 3. Primary combined-makespan gate

样本单位为 distributed seed/trial：从两个 rank 中最早的 router GPU start，到两个
rank 中最晚的 router/NCCL GPU completion。`Delta_k = T_Dk − T_Ck`。

| transport | Delta p50 (µs) | bootstrap 95% CI (µs) | 3/3 seeds positive | Primary |
|---|---:|---:|:---:|:---:|
| T0 | **845.335** | **[661.897, 1,687.218]** | 是 | **PASS** |
| T1 | 38,696.019 | [9,348.645, 43,425.659] | 否 | **FAIL** |
| T2 | 780.394 | [−629.628, 1,523.190] | 否 | **FAIL** |
| T3 | **2,095.778** | **[1,251.219, 4,390.917]** | 是 | **PASS** |

| transport | T_C p50/p95/p99/max (µs) | T_D p50/p95/p99/max (µs) | Delta p95/p99/min/max (µs) |
|---|---|---|---|
| T0 | 9,072.802 / 24,972.860 / 26,596.342 / 27,052.235 | 11,994.406 / 26,130.841 / 28,660.864 / 29,953.316 | 5,776.943 / 8,212.020 / −2,760.502 / 10,560.735 |
| T1 | 11,724.475 / 28,209.611 / 31,698.105 / 32,708.934 | 60,342.876 / 429,373.127 / 435,735.392 / 437,505.910 | 420,508.417 / 429,351.359 / −5,199.747 / 431,721.370 |
| T2 | 20,691.888 / 110,610.403 / 111,797.390 / 112,673.700 | 15,206.130 / 28,347.477 / 31,331.909 / 31,975.128 | 5,114.390 / 5,320.085 / −104,250.625 / 5,327.070 |
| T3 | 11,678.054 / 28,262.153 / 36,673.565 / 37,719.284 | 13,266.032 / 34,189.572 / 41,464.465 / 42,264.311 | 5,894.566 / 6,215.681 / −2,886.416 / 6,596.823 |

## 4. 三个独立 seed

| transport | seed | C p50/p95 (µs) | D p50/p95 (µs) | Delta p50/p95 (µs) | positive pairs |
|---|---:|---|---|---|---:|
| T0 | 4042 | 12,371.555 / 18,845.569 | 15,224.481 / 25,484.952 | 2,157.491 / 6,778.905 | 100% |
| T0 | 4043 | 17,784.338 / 26,318.169 | 18,517.686 / 26,858.536 | 779.108 / 4,670.765 | 70% |
| T0 | 4044 | 6,513.171 / 8,234.392 | 7,674.137 / 8,823.651 | 683.386 / 2,330.001 | 70% |
| T1 | 4042 | 20,561.735 / 31,081.328 | 60,342.876 / 68,896.355 | 38,696.019 / 43,656.476 | 100% |
| T1 | 4043 | 15,990.574 / 23,932.917 | 423,577.709 / 434,655.076 | 407,587.135 / 427,905.251 | 100% |
| T1 | 4044 | 11,027.106 / 11,783.246 | 6,810.490 / 7,591.422 | **−3,675.363 / −2,768.876** | 0% |
| T2 | 4042 | 19,295.335 / 29,457.170 | 20,378.129 / 30,939.437 | 1,042.541 / 1,984.020 | 80% |
| T2 | 4043 | 14,496.054 / 20,524.316 | 18,742.759 / 24,590.617 | 3,994.663 / 5,315.823 | 90% |
| T2 | 4044 | 104,882.626 / 111,262.693 | 12,718.003 / 17,022.574 | **−91,269.019 / −85,352.385** | 0% |
| T3 | 4042 | 20,661.198 / 36,035.499 | 23,798.315 / 40,976.424 | 2,874.750 / 5,972.624 | 90% |
| T3 | 4043 | 9,708.836 / 16,433.382 | 14,113.625 / 21,830.395 | 4,699.679 / 5,623.224 | 100% |
| T3 | 4044 | 11,051.488 / 15,699.859 | 12,247.718 / 13,452.978 | 1,264.502 / 2,969.543 | 70% |

T1 的 4043 delayed arm 出现约 0.42s 级长尾，而 4044 方向反转；T2 的 4044
early arm 出现约 0.105s 级排队。二者均是 transport instability 的负结果，原样保留，
不得将异常大正 Delta 当作一般收益，也不得删除异常负 Delta。

## 5. Launch、rank skew 与通信完成

| transport | submit→GPU-start p50/p95/p99/max (µs) | rank-start skew p50/p95/p99/max (µs) |
|---|---|---|
| T0 | 108.533 / 14,862.414 / 20,110.074 / 21,059.108 | 708.606 / 7,352.261 / 18,264.273 / 21,709.932 |
| T1 | 108.199 / 17,838.139 / 24,414.791 / 26,171.273 | 759.796 / 11,034.591 / 19,912.638 / 26,427.053 |
| T2 | 87.343 / 97,767.649 / 102,866.241 / 103,283.011 | 644.289 / 21,604.472 / 39,048.074 / 103,787.069 |
| T3 | 263.252 / 17,160.114 / 29,665.293 / 31,393.185 | 768.106 / 9,509.180 / 20,976.311 / 32,085.669 |

按 rank 分解的 early submit→GPU-start：

| transport | rank0 p50/p95/p99 (µs) | rank1 p50/p95/p99 (µs) |
|---|---|---|
| T0 | 82.135 / 111.637 / 125.080 | 1,638.760 / 18,133.360 / 20,823.913 |
| T1 | 81.646 / 111.996 / 121.376 | 5,299.769 / 21,341.262 / 25,809.309 |
| T2 | 78.684 / 106.228 / 114.911 | 11,546.164 / 102,517.718 / 103,052.810 |
| T3 | 201.179 / 283.148 / 292.533 | 4,819.763 / 21,696.644 / 31,283.762 |

| transport | C action-call→GPU-end p50/p95/p99/max (µs) | NCCL/P2P GPU envelope p50/p95/p99/max (µs) |
|---|---|---|
| T0 | 142.950 / 16,141.274 / 20,858.551 / 21,799.043 | 17.296 / 1,737.812 / 14,381.545 / 21,718.074 |
| T1 | 139.686 / 19,281.155 / 25,436.924 / 27,217.057 | 24.688 / 5,233.835 / 17,306.517 / 26,434.715 |
| T2 | 732.721 / 99,217.232 / 104,039.950 / 104,551.855 | 649.885 / 7,701.752 / 39,423.979 / 104,341.431 |
| T3 | 292.678 / 19,259.845 / 30,521.523 / 32,231.593 | 14.625 / 3,907.912 / 15,894.840 / 32,094.519 |

所有 transport 都保留明显 rank asymmetry。T2 把一次 collective 拆成四次后显著放大了
launch/rendezvous tail；T3 没有消除该问题。

## 6. Router interference 与 device overlap

| transport | C−B median (µs) | bootstrap 95% CI (µs) | early GPU-start-before-final | early actual coexistence |
|---|---:|---:|---:|---:|
| T0 | **+764.091** | [721.660, 795.674] | 542/720 = 75.278% | 420/720 = 58.333% |
| T1 | +845.355 | [780.923, 876.763] | 453/720 = 62.917% | 317/720 = 44.028% |
| T2 | −13.216 | [−218.143, 81.151] | 238/720 = 33.056% | 234/720 = 32.500% |
| T3 | +940.650 | [929.371, 963.434] | 470/720 = 65.278% | 379/720 = 52.639% |

| transport | overall GPU-start-before-final | overall actual coexistence | positive overlap p50/p95/p99/max (µs) |
|---|---:|---:|---|
| T0 | 542/840 = 64.524% | 420/840 = 50.000% | 6.656 / 672.389 / 761.195 / 790.167 |
| T1 | 453/840 = 53.929% | 317/840 = 37.738% | 2.784 / 759.944 / 794.558 / 801.498 |
| T2 | 238/840 = 28.333% | 234/840 = 27.857% | 16.593 / 675.506 / 703.229 / 756.287 |
| T3 | 470/840 = 55.952% | 379/840 = 45.119% | 9.152 / 672.561 / 675.089 / 705.306 |

全部 delayed arms 的 GPU-start-before-final 为 **0**，control 有效。Overlap 只作为
diagnostic，未替代 combined-makespan primary gate。

## 7. 预注册改善选择

| transport | launch p95 vs T0 | skew p95 vs T0 | router interference vs T0 | seed4044 D p99 vs T0 | clear improvements | eligible |
|---|---:|---:|---:|---:|---:|:---:|
| T1 | +20.0% | +50.1% | +10.6% | **−32.1%** | 1 | 否，primary FAIL |
| T2 | +557.8% | +193.8% | **router slowdown消失** | +56.1% | 1 | 否，primary FAIL |
| T3 | +15.5% | +29.3% | +23.1% | +36.8% | 0 | 否 |

seed4044 delayed `T_D` 的 p99/max 分别为：T0 11,204.084/11,799.192µs，
T1 7,604.264/7,607.474µs，T2 17,485.478/17,601.204µs，
T3 15,332.474/15,802.348µs。T1 对这一预注册 tail 指标有明确改善，但由于
seed4044 的 early-vs-delayed Delta 全部为负，仍不能通过 primary gate。

T0 在本次 O1B 中稳定复现 primary positive value；T3 虽也通过 primary，但没有改善
launch stability、rank skew、router interference 或 tail stability。因此按冻结规则保留 T0，
并明确保留其 launch/rendezvous、rank asymmetry 与 router contention 限制。

## 8. Semantic/safety 与禁止项

共 1,080 个 rank-trials、7,560 个动作：

- runtime BFS = 0；fast-path full rebuild = 0；unrevealed execution = 0；
- candidate/action/checker/holder divergence = 0；
- legality = **7,560/7,560**；token integrity = **1,080/1,080**；
- Ck/Dk strict comparisons = **480/480 PASS**；
- cross-transport action/descriptor comparisons = **480/480 PASS**；
- partial_current_only、partial_shards@75%、checkpoint8、deterministic/fail-closed
  checker 语义保持不变。

未实现或运行：real variable-size AlltoAllv、token packing、expert GEMM/combine、
DeepEP、formal R3 E2E、第五种 transport；未修改 workload/chunk、scheduler semantics、
75%/checkpoint8；未恢复 predictor/robust/adaptive；未创建 Subagent。

## 9. Trace 与独立回读

六份 unified Kineto/CUPTI raw traces 共重建：

- router chunks：**8,640/8,640**；
- communication action groups：**6,720/6,720**；
- all-reduce primitive kernels：**10,080/10,080**；
- P2P kernels：**1,680/1,680**；
- association failures：**0**；
- router streams：default 14、T1 15；NCCL internal stream：22。

10,080 个 all-reduce primitive kernels 中，10,077 个由 CUPTI external-id 直接匹配。
另外 3 个 kernel 的 raw trace 缺失 external metadata；analyzer 对每份 trace 分别先验证
u64 all-reduce CPU record 数与 NCCL kernel 数严格相等、所有存在的 external-id 精确一致，
再用全局 submission order 唯一补回 3 个缺元数据项。T3 的 SendRecv kernel 本身没有
CUPTI external-id，因此使用 exact annotation/kernel count + NCCL-stream submission order；
任何计数或顺序不一致均 fail closed。

六份 raw traces、三份 host artifacts、runner、analyzer 与服务器 SHA-256 全部本地匹配。
本地从 raw trace 独立重跑 analyzer 后，除路径字段外，controls、sample rows、全部统计与
selection recommendation 均与服务器结果完全一致。跨 rank 同 host monotonic marker 与
Kineto CPU annotation 的 median-offset difference 最大为 3.604µs。

## 10. 停止点

R2-O1B 已完成并由 Supervisor 裁决为 **PASS / NO VETO**：

1. T0 与 T3 primary PASS、T1/T2 primary FAIL；
2. 没有 intervention 满足完整选择规则；
3. 主线选择 T0，并显式携带 instability/contention limitations；
4. T3 不进入主线，不得继续添加 transport variant。

O1B 历史结论冻结；后续 R3 仅按新的 Supervisor 授权推进。
