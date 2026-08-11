# V0 Baseline Sequence Performance

## Scope

This benchmark evaluates the current-demand-only V0 policy. Moment context is
computed and timed but is **not** passed to `SlotLevelPolicy`; therefore these
results are not baseline-vs-moment evidence.

- Topology: Rear4GPU
- Traffic: 6 moment-bounded families
- Scale: 10 sequences/family, 64 collectives/sequence
- Samples: 3,840 paired collectives per device (7,680 schedules total)
- Model: `Rear4GPU_final.pth`, one training epoch, one training seed
- Devices: remote RTX 4090 and single-thread CPU
- PyTorch: 2.8.0+cu128
- Time limit: 20 slots
- Moment bounds: mean 0.20, variance 0.30, window 16

Although the traffic sample count meets the task's minimum sequence scale, the
model has only one preliminary training seed. Quality conclusions remain
preliminary.

## Overall results

| Metric | CPU (1 thread) | RTX 4090 |
|---|---:|---:|
| Collectives | 3,840 | 3,840 |
| Completion mean | 8.3583 | 8.3552 |
| Completion mean 95% CI (sequence-level normal CI) | [8.0612, 8.6554] | [8.0587, 8.6517] |
| Median | 9 | 9 |
| p95 / p99 | 11 / 12 | 11 / 12 |
| CVaR90 / CVaR95 | 10.4886 / 11.5970 | 10.4395 / 11.7193 |
| Timeout rate | 0% | 0% |
| Schedule legality | 100% | 100% |
| Synthesis mean | 24.342 ms | 34.641 ms |
| Synthesis p95 / p99 | 33.893 / 35.842 ms | 46.768 / 50.569 ms |
| Throughput | 41.081/s | 28.867/s |
| Estimator mean / p95 | 0.178 / 0.189 ms | 0.180 / 0.190 ms |

For this small, autoregressive workload, CUDA is not faster: its mean synthesis
latency is 1.423x the single-thread CPU latency. Kernel-launch and host/device
coordination dominate the small graph operations. A 64-thread CPU precheck was
also rejected as misleading because thread scheduling inflated mean latency to
about 705 ms.

## Per-family CPU quality

| Family | Mean | p95 | CVaR95 | Mean synthesis ms |
|---|---:|---:|---:|---:|
| smooth_ar | 8.6313 | 10 | 10.2069 | 26.057 |
| alternating_burst | 6.0000 | 7 | 7.0000 | 18.325 |
| moving_hotspot | 9.7500 | 12 | 12.0000 | 28.342 |
| sparse_switching | 8.8125 | 10 | 10.0000 | 23.683 |
| bimodal | 8.2063 | 10 | 10.1000 | 23.469 |
| heavy_tail_clipped | 8.7500 | 11 | 11.0000 | 26.179 |

`moving_hotspot` is the hardest family for the current baseline; its mean and
tail completion cost are materially worse than `alternating_burst`. This is a
useful target for V1, but does not prove that moments will improve it.

## CPU/CUDA numerical divergence

- Completion match rate: 97.6042% (3,748/3,840 pairs)
- Legality match rate: 100%
- Mismatches: 92, all exactly one completion slot
- CUDA minus CPU slot difference: 52 cases at -1, 40 cases at +1
- Families with mismatches: heavy_tail_clipped 40, smooth_ar 32, bimodal 20

The policy uses greedy `argmax`; near-equal logits can be ordered differently by
CPU and CUDA floating-point kernels. Device-level quality numbers should not be
treated as independent algorithm variants. Reproducibility-sensitive evaluation
should fix one inference device or add an explicit deterministic tie-break rule.

## Generator and estimator overhead

- 60/60 generated sequences passed final integer moment validation.
- Worst mean error: 0.15625 (bound 0.20).
- Worst variance error: 0.09766 (bound 0.30).
- Mean generation time: 19.31 ms/sequence.
- Sliding estimator mean cost: about 0.18 ms/collective, below 1% of CPU policy
  synthesis time.

An attempted 4-sample smoke window correctly failed for
`heavy_tail_clipped`: integer clip/round could not meet the strict mean bound.
The benchmark kept the requested 16-sample window rather than weakening bounds.

## Artifacts

- `scripts/benchmark_v0_sequences.py`: reproducible benchmark harness.
- `outputs/performance/remote_formal/v0_sequence_benchmark_detail.csv`: all
  7,680 device-specific rows.
- `outputs/performance/remote_formal/v0_sequence_benchmark_summary.json`: full
  environment, configuration, hashes, and aggregates.

## Go/No-Go relevance

This benchmark validates the V0 baseline and identifies meaningful family-level
tail variation. It cannot satisfy the V1 Go criterion because no policy consumed
moment context. The next valid experiment must compare baseline, mean-only,
mean+variance, and shuffled context on the exact same matrices.
