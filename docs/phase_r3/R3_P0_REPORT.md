# Phase R3-P0: Progressive Early A2Av Pilot

Status: **PASS, pending Supervisor review**. This is a pilot result, not formal R3 validation.

## Frozen execution

- Environment: two Tesla V100-SXM2-32GB GPUs, two NCCL ranks.
- Pilot seeds: 6042, 6142, 6242. Formal seeds 5042/5142/5242 were not touched.
- Five frozen traffic families, ten paired jobs/family/seed: 150 C/D pairs.
- C: progressive early A2Av-T0. D: identical ordered delta descriptors delayed until final router completion.
- Per pair, router parameters/top-k, tokens, destinations, descriptors/order, sendcounts,
  payload bytes, A2Av call count, and final receive multiset were identical. D did not merge calls.
- Primary interval: first router launch to final required variable-size A2Av payload completion.
  Receive D2H/unpack/checksum was outside primary and included in full-reference makespan.

## Primary result

For `Delta_A2Av = T_delayed - T_early`:

- paired median: **+958.144 us**;
- paired bootstrap 95% CI: **[+49.4115, +1889.6875] us**;
- seed 6042 median: **+765.712 us**;
- seed 6142 median: **+1752.732 us**;
- seed 6242 median: **+718.986 us**.

The preregistered Gate passes: median is positive, CI lower is positive, and 3/3 independent
seed medians are positive.

Primary combined makespan diagnostics:

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| Early C | 68.107 | 181.881 | 204.618 | 320.718 |
| Delayed D | 69.142 | 163.124 | 196.853 | 349.114 |

The pilot establishes a positive median critical-path value, but it does not establish tail
improvement: C has worse p95/p99 while D has the larger single maximum.

## Per-family result

| Family | paired median Delta (us) | positive-pair fraction |
|---|---:|---:|
| balanced | +2307.809 | 63.33% |
| skewed | +991.779 | 63.33% |
| all-to-one-like | **-401.402** | 46.67% |
| zero-sized-pair | +519.791 | 60.00% |
| multiple-progressive-shards | +357.421 | 56.67% |

P0 did not require every family to be positive. The negative all-to-one-like result is retained
as a limitation and must not be hidden by the corpus-wide PASS.

## Stage diagnostics

All distributions below contain 2,100 delta descriptors per arm.

| Stage | Early C p50/p95/p99/max (us) | Delayed D p50/p95/p99/max (us) |
|---|---:|---:|
| count construction | 487.022 / 963.702 / 1295.774 / 6296.991 | 485.328 / 955.501 / 1284.935 / 3169.679 |
| offset construction | 0.779 / 0.896 / 1.187 / 11.106 | 0.780 / 0.884 / 1.119 / 12.638 |
| reference packing | 1766.223 / 3480.957 / 4741.593 / 8003.462 | 1761.992 / 3456.575 / 4686.724 / 7252.226 |
| payload H2D | 178.641 / 286.484 / 400.036 / 1344.720 | 143.498 / 197.889 / 315.268 / 2196.149 |
| delta-count exchange | 390.073 / 924.846 / 6205.984 / 140005.429 | 293.492 / 1087.818 / 3887.420 / 161162.878 |
| compiled AICCL | 291.059 / 373.109 / 465.439 / 1416.868 | 296.558 / 378.135 / 493.542 / 1580.314 |
| A2Av API submit | 222.053 / 341.100 / 434.763 / 2477.393 | 202.939 / 252.580 / 354.261 / 2213.022 |
| A2Av GPU kernel time | 13.760 / 205.410 / 279.940 / 1919.000 | 13.568 / 135.151 / 166.439 / 2122.808 |

Full-reference makespan (including receive D2H and verification) is C p50 146.724 ms versus
D p50 143.955 ms. This secondary oracle-inclusive measure is not the preregistered primary.

Packing is material: only 55.81% of early descriptors and 45.08% of early payload bytes were
packed before final router readiness. It reduces the available hiding opportunity but did not
erase the positive primary median. Count exchange also did not erase the primary result, but its
tail remains severe: C p99/max 6.206/140.005 ms and D p99/max 3.887/161.163 ms.

## Device timeline and interference

Six Kineto/CUPTI traces were audited with zero association failures: 4,800 router chunks,
4,200 count collectives, and 4,200 payload A2Av collectives. Host monotonic and GPU timestamps
were not subtracted from one another; submit/start diagnostics use the unified trace timeline.

- early payload GPU-start-before-final fraction: **43.238%**;
- actual future-router/A2Av coexistence fraction: **18.333%**;
- positive overlap duration p50/p95/p99/max: **8.000 / 20.256 / 23.274 / 23.712 us**;
- payload submit-call-start to GPU-start p50/p95/p99/max:
  **135.119 / 225.463 / 285.019 / 2147.357 us**;
- rank-start skew p50/p95/p99/max:
  **112.431 / 235.751 / 377.243 / 2042.897 us**.

The paired GPU router interval C-D has median **-14.425 ms**, bootstrap 95% CI
**[-14.804, -13.680] ms**: no median router slowdown is observed in this pilot. This is not
claimed as causal router acceleration. The distribution is unstable (p95 +85.373 ms,
p99 +173.184 ms), so substantial positive interference tails remain.

## Correctness and conclusion

Across 2,457,600 sent/received token records (157,286,400 bytes across both arms/ranks):

- legality and token integrity: 100%;
- lost, duplicate, wrong-destination, and corruption: zero;
- unrevealed execution, future access, duplicate/stale dispatch: zero;
- runtime BFS and full rebuild: zero;
- scheduler/checker and C/D semantic divergence: zero.

Therefore **R3-P0 = PASS / pending Supervisor review**. The evidence qualifies the project to
apply for R3 formal validation; it does not authorize running formal R3 automatically.

Canonical result: `outputs/phase_r3/p0_pilot/analysis_v3/r3_p0_results.json`.
Earlier `analysis` and `analysis_v2` are retained as superseded diagnostics; v3 is canonical.
