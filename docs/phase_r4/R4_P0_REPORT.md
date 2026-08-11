# Phase R4-P0 — Progressive Forward Full-MoE Pilot

Status: **R4-P0 PASS pending Supervisor review**

## Frozen protocol

- Environment: 2x Tesla V100-SXM2-32GB, two NCCL ranks.
- Fresh pilot seeds: `8042 / 8142 / 8242`.
- Families: balanced, skewed, all-to-one-like, zero-sized-pair, and multiple-progressive-shards.
- Ten paired jobs per family and seed: 150 C/D pairs total.
- C uses progressive forward descriptors; D executes identical descriptors after final Router completion.
- Expert execution is non-progressive. Router/top-k, expert batches and weights, GEMM shapes,
  return descriptors, payloads, and final output are identical within every pair.
- Primary clock: earliest first Router launch across ranks to latest actual combined-output-ready.
- Actual combine is timed as the direct original-position scatter. Identity, checksum, independent
  expert oracle, allclose, and scheduler shadow checks execute only after the primary clock stops.
- `partial_shards=75%`, checkpoint8, compiled AICCL semantics, and A2Av-T0 remain unchanged.
- Primary profiler is disabled.

## Primary result

`Delta = T_D - T_C`, in microseconds:

| Scope | n | Median | p95 | p99 | Max |
|---|---:|---:|---:|---:|---:|
| Corpus | 150 | +5,370.287 | — | — | — |
| seed 8042 | 50 | +6,390.166 | +66,402.925 | +110,333.164 | +115,995.810 |
| seed 8142 | 50 | +2,507.236 | +66,902.649 | +77,447.498 | +77,750.422 |
| seed 8242 | 50 | +5,316.712 | +39,213.546 | +76,743.326 | +85,017.357 |

The 10,000-replicate paired bootstrap median 95% CI is
**[+2,232.069, +6,958.139] us**. All three seed medians are positive. The
preregistered primary Gate therefore passes.

98/150 individual pairs are positive. This fraction is diagnostic only and is not a substitute
for the preregistered median/CI/three-seed Gate.

## Per-family delta

| Family | Positive pairs | Median (us) | p95 (us) | p99 (us) | Max (us) |
|---|---:|---:|---:|---:|---:|
| balanced | 20/30 | +10,509.196 | +39,404.497 | +62,107.518 | +68,131.580 |
| skewed | 22/30 | +6,174.214 | +69,022.993 | +104,725.366 | +115,995.810 |
| all-to-one-like | 23/30 | +6,571.533 | +70,253.392 | +98,807.000 | +104,439.389 |
| zero-sized-pair | 10/30 | **-4,252.146** | +73,690.348 | +76,886.651 | +77,750.422 |
| multiple-progressive-shards | 23/30 | +5,304.081 | +31,066.449 | +50,315.648 | +55,259.625 |

The zero-sized-pair negative median is retained. P0 did not require every family to be positive,
so it does not change the corpus-wide Gate, but it limits generalization.

## Full-MoE timing diagnostics

All values below are marginal distributions; they are diagnostics rather than additional Gates.

| Metric | C p50 / p95 / p99 / max (us) | D p50 / p95 / p99 / max (us) |
|---|---:|---:|
| Primary full-MoE makespan | 478,165.896 / 531,726.537 / 551,210.611 / 590,659.183 | 486,389.998 / 556,687.113 / 609,445.391 / 613,664.558 |
| Forward stage | 366,222.481 / 434,960.212 / 456,797.299 / 494,198.164 | 364,799.946 / 444,078.504 / 510,174.197 / 520,670.883 |
| Expert compute + D2H | 19,812.584 / 38,256.320 / 39,976.942 / 48,659.085 | 19,669.836 / 38,113.318 / 38,609.726 / 41,991.119 |
| Return stage | 53,291.969 / 152,888.772 / 153,716.507 / 155,908.592 | 52,959.329 / 154,067.818 / 162,892.763 / 218,846.752 |
| Actual combine | 177.808 / 198.782 / 257.355 / 287.483 | 176.416 / 190.904 / 258.064 / 428.676 |
| Full-reference including oracle/checker | 682,683.782 / 737,915.668 / 763,052.352 / 795,757.114 | 690,459.283 / 769,859.937 / 815,522.128 / 820,618.263 |

## Packing and count-exchange tails

| Metric | C p50 / p95 / p99 / max (us) | D p50 / p95 / p99 / max (us) |
|---|---:|---:|
| Forward packing | 10,946.684 / 22,430.889 / 30,623.076 / 33,145.182 | 10,945.724 / 21,937.047 / 30,310.908 / 32,872.452 |
| Forward count exchange | 251.500 / 26,204.453 / 31,891.087 / 98,669.539 | 292.002 / 27,215.472 / 29,420.872 / 107,400.054 |
| Return packing | 3,033.055 / 8,462.010 / 12,546.094 / 13,591.795 | 3,018.124 / 8,802.270 / 12,720.193 / 77,215.533 |
| Return count exchange | 222.982 / 13,239.549 / 103,040.904 / 108,399.907 | 217.490 / 13,486.264 / 103,299.701 / 111,660.713 |

Packing is substantial but did not erase the corpus-wide positive paired median. Count exchange
remains highly heavy-tailed, especially at p99, and is a major limitation. No packing or count
exchange optimization was performed.

Router marginal p50 is 5,765.914 us for C and 5,754.534 us for D. This does not establish a
paired interference effect; C also contains one 144,147.248 us maximum outlier, which is retained.

## Correctness and equivalence

- 150/150 paired comparisons passed; 600 rank-arm executions passed.
- C/D Router top-k, assignments, forward descriptors, expert batches/weights/outputs,
  return descriptors, scheduler actions, and final output digests are identical.
- Legality and token integrity: 100%.
- Lost, duplicate, wrong-source, wrong-expert, wrong-destination, wrong-return,
  wrong-position, corruption, expert-output mismatch: all zero.
- Runtime BFS, full rebuild, unrevealed execution, future access, and semantic divergence: all zero.

## Artifacts

- Raw primary host evidence: `outputs/phase_r4/p0_full_moe_pilot/r4_p0_primary_host.json`
  (`15f5ccb661f1699228c9ec2296be24ba48a8ebd956fa7654dd10f313886e870e`)
- Canonical analysis: `outputs/phase_r4/p0_full_moe_pilot/r4_p0_results.json`
  (`0c7fe610eb566ce239512614fccab9636d06a2982aaea0789224f95e8b37d4ba`)
- Compact table: `outputs/phase_r4/p0_full_moe_pilot/r4_p0_summary.csv`
  (`070bfcd4e4a6d9cc85bf55fc43d816bcb746d8bb03682991a3c8b989f8e5c818`)

## Recommendation

R4-P0 qualifies for Supervisor review and, if accepted, an application for R4 formal validation.
It does not itself authorize or constitute formal R4 validation.
