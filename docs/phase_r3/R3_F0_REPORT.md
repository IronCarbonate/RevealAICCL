# Phase R3-F0 Formal Real Variable-Size A2Av Validation

Status: **PASS, pending Supervisor review**.

## Method

The formal seeds 5042/5142/5242 had no prior corpus/result artifacts under structured seed
audit. The primary run used a single long-lived two-rank process, one NCCL initialization,
one shared communicator, fixed warmup/barriers/GPU-idle boundaries, counterbalanced C/D arms,
and **no Kineto/CUPTI profiler**. It covered three seeds × five unchanged families × twenty
paired jobs/family = 300 pairs. The all-to-one-like family was retained unchanged.

A separately preregistered diagnostic subset reran corpus job indices 0 and 10 for every
seed/family (30 pairs) under CUPTI. It was excluded from all primary statistics.

## Formal primary Gate

For `Delta = T_delayed - T_progressive_early`:

- paired median: **+829.2965 us**;
- 10,000-resample paired bootstrap 95% CI: **[+242.1435, +1439.2545] us**;
- seed 5042 median: **+401.3085 us**;
- seed 5142 median: **+157.1760 us**;
- seed 5242 median: **+1643.0400 us**.

All three preregistered conditions pass. The R3-P0 positive result is independently reproduced.

Primary combined makespan:

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| Progressive C | 66.384 | 143.405 | 155.987 | 176.177 |
| Delayed D | 67.074 | 136.644 | 171.748 | 262.591 |

The median and p99/max favor C, while p95 favors D. This is formal evidence of positive median
critical-path value, not a claim of universal tail improvement.

## Per-family results

| Family | paired median Delta (us) | positive-pair fraction |
|---|---:|---:|
| balanced | +49.649 | 50.00% |
| skewed | +1213.212 | 58.33% |
| all-to-one-like | +1498.601 | 63.33% |
| zero-sized-pair | +829.296 | 63.33% |
| multiple-progressive-shards | **-46.955** | 50.00% |

Per-family heterogeneity is retained. Formal all-to-one-like is positive despite the negative
pilot result; multiple-progressive-shards is slightly negative in formal. Neither observation
replaces the corpus-wide Gate.

## Full primary latency breakdown

Each descriptor-stage distribution contains 4,200 samples/arm.

| Stage | C p50/p95/p99/max (us) | D p50/p95/p99/max (us) |
|---|---:|---:|
| count construction | 476.657 / 950.241 / 1266.697 / 4178.581 | 476.000 / 950.524 / 1268.979 / 2157.955 |
| offset construction | 0.791 / 0.898 / 1.174 / 12.640 | 0.789 / 0.892 / 1.122 / 11.130 |
| reference packing | 1747.126 / 3402.758 / 4668.754 / 8038.982 | 1737.643 / 3414.810 / 4677.805 / 7959.779 |
| payload H2D | 156.349 / 230.017 / 269.887 / 1052.652 | 101.178 / 151.449 / 216.400 / 653.713 |
| delta-count exchange | 310.715 / 739.338 / 3436.120 / 109692.735 | 203.260 / 808.471 / 3493.668 / 193403.111 |
| compiled AICCL | 277.500 / 355.112 / 469.537 / 1608.076 | 280.510 / 374.482 / 447.600 / 1160.998 |
| A2Av API submit | 163.660 / 244.670 / 291.101 / 3659.872 | 138.113 / 177.643 / 239.384 / 1065.940 |

The CUPTI subset reports A2Av GPU kernel time:

- C p50/p95/p99/max: 13.696 / 211.255 / 305.163 / 756.509 us;
- D p50/p95/p99/max: 13.440 / 143.858 / 180.758 / 287.070 us.

Full-reference makespan including receive D2H and verification is C p50/p95/p99/max
148.579/232.426/257.490/299.272 ms versus D 149.235/232.269/245.323/334.280 ms.

## Limitations and device diagnostics

Packing remains the largest stable per-descriptor host stage. In the profiler-off primary,
69.55% of C descriptors and 57.72% of C payload bytes were packed before final Router readiness.
It materially limits the available window but does not erase the formal positive result.

Count exchange remains the principal extreme-tail limitation: C p99/max 3.436/109.693 ms and
D p99/max 3.494/193.403 ms. It likewise does not erase the formal Gate, but it prevents a broad
tail-stability claim.

The excluded CUPTI subset reports:

- payload GPU-start-before-final: 42.381%;
- actual future-Router/A2Av coexistence: 20.952%;
- positive overlap p50/p95/p99/max: 7.184/19.334/23.169/23.392 us;
- submit-call-start to GPU-start p50/p95/p99/max: 138.154/245.286/282.299/615.921 us;
- rank-start skew p50/p95/p99/max: 116.627/250.341/511.156/744.006 us.

The diagnostic GPU Router C-D interval median is -13.905 ms with bootstrap CI
[-15.762, -10.194] ms, but the positive tail is very large (p95 +114.199 ms,
p99 +155.510 ms). Per preregistration, the negative median is **not** interpreted as causal
Router acceleration.

## Correctness and conclusion

Primary transmitted/received 4,915,200 token records (314,572,800 bytes across both arms/ranks).
Primary and diagnostic subset both satisfy:

- legality and token integrity 100%;
- lost, duplicate, wrong-destination, corruption zero;
- unrevealed/future/stale/duplicate dispatch zero;
- runtime BFS and full rebuild zero;
- scheduler/checker and C/D semantic divergence zero.

Six diagnostic traces reconstructed 960 Router chunks, 840 count collectives, and 840 payload
collectives with zero association failures and without mixing host-monotonic/GPU clock domains.

Therefore **R3-F0 = PASS / pending Supervisor review**. This constitutes formal evidence that
progressive early real uncertain variable-size A2Av improves the frozen reference system's
median combined makespan. It does not yet establish full-MoE E2E benefit or production-backend
readiness. The next scientifically justified step is a separately authorized full MoE
expert/return/combine validation; production integration should wait for that evidence.
