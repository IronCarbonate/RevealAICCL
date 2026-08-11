# Phase R4-F0 Formal Validation

Status: **R4-F0 PASS pending Supervisor review**

## Protocol and corpus

- Freshness-audited formal seeds: `9042 / 9142 / 9242`.
- 20 jobs per family per seed, five equally represented frozen families: 300 paired jobs.
- 1,200 rank-arm executions on 2x Tesla V100-SXM2-32GB with two real NCCL ranks.
- The zero-sized-pair family was retained without adjustment.
- C used progressive forward descriptors; D used identical descriptors delayed until final Router
  completion. Expert MLP was non-progressive in both arms.
- Primary timing was earliest first Router launch across ranks to latest actual combined output
  ready. Correctness-only oracle/checker work followed the primary timestamp.
- No profiler, optimization, workload tuning, transport/scheduler/config change, production
  backend, DeepEP, or PCCL was used.

## Formal primary result

`Delta = T_D - T_C`:

- n = 300 paired jobs
- paired median = **+2,800.709 us**
- 10,000-resample bootstrap median 95% CI = **[+967.251, +3,714.117] us**
- positive individual pairs = 183/300 (61.0%); diagnostic only

All preregistered formal conditions pass:

1. paired median > 0;
2. bootstrap 95% CI lower > 0;
3. all three fresh formal-seed medians > 0;
4. correctness and exact C/D equivalence pass.

## Per-seed Delta

| Seed | Positive | Median (us) | p95 (us) | p99 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|
| 9042 | 62/100 | +2,860.413 | +25,778.814 | +80,286.652 | +80,648.463 |
| 9142 | 65/100 | +3,597.016 | +61,881.145 | +90,329.703 | +91,316.149 |
| 9242 | 56/100 | +1,052.512 | +43,859.020 | +90,497.858 | +96,093.862 |

## Per-family Delta

| Family | Positive | Median (us) | p95 (us) | p99 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|
| balanced | 46/60 | +6,736.312 | +44,538.744 | +71,268.105 | +87,831.202 |
| skewed | 24/60 | **-2,131.204** | +16,625.507 | +53,953.701 | +90,441.333 |
| all-to-one-like | 28/60 | **-927.016** | +80,301.270 | +88,531.236 | +91,316.149 |
| zero-sized-pair | 45/60 | +6,856.332 | +60,692.146 | +83,670.876 | +90,319.739 |
| multiple-progressive-shards | 40/60 | +3,912.336 | +13,536.351 | +48,372.245 | +96,093.862 |

The two negative family medians are retained as formal heterogeneity. The preregistered Gate did
not require every family to be positive and was not changed after observing results.

## Full-MoE timing diagnostics

These are marginal arm distributions and must not replace paired Delta. In particular, marginal
medians need not have the same difference as the median of within-pair differences.

| Metric | C p50 / p95 / p99 / max (us) | D p50 / p95 / p99 / max (us) |
|---|---:|---:|
| Primary full-MoE makespan | 480,265.076 / 544,268.980 / 575,816.238 / 609,176.873 | 476,812.691 / 558,170.029 / 606,580.920 / 644,099.210 |
| Forward stage | 365,074.747 / 435,453.457 / 458,090.521 / 509,769.276 | 363,913.566 / 443,553.112 / 484,904.606 / 540,440.119 |
| Expert compute + D2H | 22,695.609 / 44,532.756 / 47,446.442 / 50,304.697 | 22,665.057 / 44,141.966 / 46,505.566 / 47,319.315 |
| Return stage | 53,405.708 / 155,500.940 / 161,778.222 / 236,441.577 | 53,446.583 / 160,970.466 / 165,634.107 / 241,799.187 |
| Actual combine | 173.209 / 188.336 / 252.630 / 828.934 | 175.024 / 191.391 / 384.545 / 1,324.753 |
| Full reference including checker | 684,794.698 / 751,668.477 / 804,346.103 / 812,837.599 | 682,660.418 / 780,446.646 / 835,188.831 / 856,610.889 |

## Packing and count-exchange tails

| Metric | C p50 / p95 / p99 / max (us) | D p50 / p95 / p99 / max (us) |
|---|---:|---:|
| Forward packing | 10,954.962 / 22,896.996 / 30,140.531 / 33,020.013 | 10,949.045 / 21,950.235 / 29,932.099 / 32,645.560 |
| Forward count exchange | 249.519 / 26,026.451 / 29,750.406 / 117,064.010 | 285.862 / 26,932.562 / 29,223.757 / 132,810.846 |
| Return packing | 2,994.748 / 8,520.320 / 12,532.249 / 93,527.317 | 3,000.359 / 8,556.735 / 12,528.367 / 96,197.095 |
| Return count exchange | 228.096 / 13,199.203 / 103,943.866 / 115,207.890 | 224.083 / 13,323.188 / 109,834.388 / 121,970.723 |

Packing remains substantial and count exchange remains strongly heavy-tailed. These limitations
were not optimized or excluded. The formal paired Gate remains positive after including both.

Router marginal p50 was 5,756.131 us for C and 5,753.614 us for D. C retained a 155,684.748 us
maximum outlier; marginal values do not establish a paired interference effect.

## Correctness and equivalence

- 300/300 paired comparisons and all 1,200 rank-arm executions passed.
- Router top-k and assignments, ordered forward descriptors, sendcounts, payload bytes, expert
  batches/weights/outputs, GEMM shapes, return descriptors, scheduler actions, and final output
  digests were identical within every C/D pair.
- Legality and token integrity: 100%.
- Lost, duplicate, wrong source/expert/destination/return/position, corruption, and expert-output
  mismatch: all zero.
- Runtime BFS, full rebuild, unrevealed execution, future access, and scheduler/checker divergence:
  all zero.

## Artifacts

- Raw formal evidence: `outputs/phase_r4/f0_full_moe_formal/r4_f0_primary_host.json`
  (`40f77a7e02d01f4b7abdcd9c412f18d73f3929962c5f31d133d683d05c01b675`)
- Canonical formal analysis: `outputs/phase_r4/f0_full_moe_formal/r4_f0_results.json`
  (`ac29d4dac11714aa28ed30056ea7c1a9c867ba631a1f662bd4deb84d0145cefa`)
- Compact table: `outputs/phase_r4/f0_full_moe_formal/r4_f0_summary.csv`
  (`6ffe68adafbd3df2f53271ebefb97f3a1ac8dc11edb54c44f88e2bc5e2d1e312`)

## Conclusion

R4-F0 independently reproduces a statistically positive corpus-wide paired full-MoE critical-path
benefit on fresh formal data. Status remains PASS pending Supervisor review. No subsequent phase is
authorized or started.
