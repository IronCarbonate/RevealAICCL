# Phase R5-P4: Optimized Progressive E1 vs Delayed D1 Diagnosis

Status: **diagnosis complete; pending Supervisor review**  
Scope: diagnosis only. No packing optimization, scheduler/reveal change, progressive expert/return, new transport, formal run, or MSCCL integration was performed.

## 1. Frozen input and conclusion

R5-P3 remains frozen:

- fast data preparation: PASS and retained as the new baseline;
- `E0 - E1 = +78.872 ms`;
- `D1 - E1 = -3.093 ms`;
- progressive timing gate: FAIL (`0/3` seeds and `0/5` families positive).

R5-P4 used fresh diagnostic seeds `13042/13142/13242`, five unchanged traffic families, three jobs per family, two NCCL ranks, and the identical fast-data-prep backend in E1 and D1. The diagnosis contains 45 paired jobs and 315 forward descriptors per arm.

**Diagnosis: collective/rank-rendezvous dominated.** The descriptive positive-cost attribution is 91.571% collective/rank-rendezvous group versus 8.429% resource group. The largest observations are cross-rank ready/issue skew, long count-collective lifetime, payload rank-start skew, and NCCL kernels that remain resident while waiting for the peer. Single-rank payload call-to-GPU-start is not the bottleneck, Router active compute is essentially unchanged, and no Router/future-chunk–payload device overlap was observed.

This supports applying for an **MSCCL backend-integration phase** next. It does not authorize or implement MSCCL, and it does not establish that MSCCL will improve the result.

## 2. Diagnostic sign reproduction

The profiled corpus independently reproduced the R5-P3 direction:

| Metric | Result |
|---|---:|
| E1 extra makespan, median (`E1-D1`) | +2.272 ms |
| p95 / p99 / max | +48.028 / +186.872 / +189.568 ms |
| E1 slower pairs | 30/45 |
| Seeds with positive median E1 extra cost | 3/3 |
| Families with positive median E1 extra cost | 5/5 |

This is a profiler diagnostic, not a replacement performance gate. The canonical R5-P3 value remains `D1-E1 = -3.093 ms`.

Per seed:

| Seed | Median E1 extra cost | E1 slower jobs |
|---:|---:|---:|
| 13042 | +4.415 ms | 10/15 |
| 13142 | +0.497 ms | 10/15 |
| 13242 | +0.866 ms | 10/15 |

Per family:

| Family | Median E1 extra cost | E1 slower jobs |
|---|---:|---:|
| balanced | +4.850 ms | 5/9 |
| skewed | +0.497 ms | 5/9 |
| all-to-one-like | +6.382 ms | 8/9 |
| zero-sized-pair | +0.412 ms | 5/9 |
| multiple-progressive-shards | +2.272 ms | 7/9 |

## 3. Per-descriptor decomposition

All values below are over 315 descriptors per arm and are reported as p50 / p95. Host monotonic timestamps are compared only with host monotonic timestamps; GPU timeline values are derived only from Kineto/CUPTI traces.

| Component | E1 progressive | D1 delayed | Interpretation |
|---|---:|---:|---|
| Cross-rank ready skew | 16.113 / 41.324 ms | 0.617 / 4.676 ms | E1 peers reveal corresponding descriptors at very different times. |
| Cross-rank count issue skew | 16.126 / 41.224 ms | 15.955 / 28.719 ms | E1 has a materially worse tail. |
| Count issue → both ranks complete | 29.219 / 61.943 ms | 16.405 / 29.158 ms | Main count-collective lifetime penalty. |
| Residual wait at consumption | 0.141 / 28.377 ms | 15.820 / 28.613 ms | E1 often issues earlier, so part of the wait is outstanding before consumption; this is not the full rendezvous lifetime. |
| Count GPU-event duration | 0.200 / 0.375 ms | 0.177 / 0.376 ms | Actual event duration is similar; event query is not the source. |
| Cross-rank payload call skew | 12.217 / 24.594 ms | 0.210 / 0.296 ms | Progressive ranks enter matching A2Av calls far apart. |
| Single-rank payload call → GPU start | 0.177 / 0.512 ms | 0.190 / 0.532 ms | Not worse in E1; local launch latency is not primary. |
| Cross-rank payload GPU-start skew | 12.228 / 24.926 ms | 0.267 / 0.612 ms | Peer/rank rendezvous is the dominant launch instability. |
| Payload NCCL kernel envelope | 0.733 / 24.965 ms | 0.378 / 0.809 ms | E1 NCCL kernels can remain resident while waiting; this envelope is not pure byte-transfer time. |

The key distinction is that E1's low residual wait does not mean rendezvous disappeared. E1 starts the count operation early, so some waiting moves into the outstanding collective lifetime. `issue→both-complete`, rank issue skew, payload start skew, and NCCL kernel envelope reveal the actual penalty.

## 4. Router/A2Av interference and overlap

| Metric | E1 | D1 |
|---|---:|---:|
| Router GPU active p50 / p95 | 303.487 / 308.574 us | 303.266 / 306.641 us |
| Router GPU envelope p50 / p95 | 6.033 / 7.987 ms | 5.870 / 8.968 ms |
| Router host visibility p50 / p95 | 6.711 / 8.851 ms | 6.564 / 9.720 ms |
| Actual future-Router/payload overlap | 0/315 descriptors | 0/315 descriptors |

Paired category diagnostics give only +0.989 us median Router-active difference and +250.591 us median Router-envelope difference. Thus there is no evidence that Router arithmetic contention explains the roughly 3 ms P3 regression. In this P4 corpus, progressive submission incurred rank-rendezvous cost but hid no payload GPU work under future Router execution.

## 5. Main extra-cost attribution

The preregistered classifier takes the positive part of each corpus-wide paired category median and normalizes the resulting values. The result is:

| Diagnostic group/category | Descriptive share |
|---|---:|
| **Collective/rank-rendezvous group** | **91.571%** |
| ├─ count rendezvous lifetime | 36.206% |
| ├─ cross-rank ready skew | 29.499% |
| └─ payload launch/rank-start skew | 25.866% |
| **Resource group** | **8.429%** |
| ├─ payload GPU kernel envelope | 8.345% |
| ├─ Router GPU envelope | 0.084% |
| └─ Router GPU active work | 0.0003% |

These shares are **descriptive, non-causal, and non-additive**. Ready skew, rendezvous lifetime, launch skew, and a waiting NCCL kernel are successive manifestations of the same dependency chain, so their absolute microsecond totals must not be added to predict makespan. The percentages answer which observed mechanism dominates, not how many milliseconds each mechanism independently causes.

## 6. Correctness and trace audit

- 45/45 paired jobs passed E1/D1 semantic equivalence.
- Final output maximum absolute difference: exactly 0.
- Six rank traces passed fail-closed cardinality and association checks.
- Per rank/seed expected and observed: Router 240, count 210, payload 210.
- Total audited associations: Router 1,440; count 1,260; payload 1,260.
- Count and payload kernels use `record_function` CPU External ID → NCCL kernel association.
- This PyTorch build did not propagate Router producer-thread annotations. Router intervals therefore use the preregistered pinned-D2H delimiter fallback, with exact cardinality, byte-size, single-stream, and non-empty-kernel checks.
- Runtime BFS and scheduling/reveal semantics were not changed.

## 7. Decision

**R5-P4 diagnosis: COMPLETE.**  
**Primary diagnosis: collective/rank-rendezvous dominated.**

Recommended next request: a bounded **MSCCL backend integration** phase focused on rank-synchronous, precompiled collective execution while freezing Router/top-k, descriptors, bytes, compiled AICCL semantics, expert/return/combine, and the optimized data-prep baseline. A paired NCCL-versus-MSCCL protocol must be preregistered before any performance run. No such phase was run here.

## 8. Canonical evidence

- [Preregistration](R5_P4_PREREGISTRATION.md) — SHA-256 `065be10207c8bf8ec821e27958d09f14ce676cafe1d7e7d41749803794d71c3a`
- [Canonical analysis](../../outputs/phase_r5/p4_diagnosis/r5_p4_results.json) — SHA-256 `fc1ea9e543d6ebc7bdad75abe0800e51b57b81e41797f25e0044e125d8a884d5`
- [Seed 13042 host artifact](../../outputs/phase_r5/p4_diagnosis/seed13042/r5_p4_seed13042_host.json) — SHA-256 `64c365567cee4edc7cd564d3a4e1694879e8e1789f3535c32ffe92f1854f1f05`
- [Seed 13142 host artifact](../../outputs/phase_r5/p4_diagnosis/seed13142/r5_p4_seed13142_host.json) — SHA-256 `230db99cbeb949fe566c7161c004a088bce6f1f5637d994dd0a5a5ba87fec236`
- [Seed 13242 host artifact](../../outputs/phase_r5/p4_diagnosis/seed13242/r5_p4_seed13242_host.json) — SHA-256 `702939c0fa56feb8200ab0f4d41b5fcbc0b793a909556fc2058283eb0d687089`
- [Instrumented full-MoE path](../../scripts/run_r4_a0_c0_full_moe.py) — SHA-256 `7622aab859691193b15fa4d02cce9880b897bb99daa98ec73b0c8bd4e67bc65a`
- [Profile runner](../../scripts/run_r5_p4_profiled_diagnosis.py) — SHA-256 `541f16be24541ca0b728b9572f0909e847a419f3b890cbff9b3822cdee86dd7a`
- [Fail-closed analyzer](../../scripts/analyze_r5_p4_diagnosis.py) — SHA-256 `befdf209dde72ef826a661d34632509f23ab2c2bbfde429b93f53771570dbfb6`

The six canonical trace files remain on the authorized server under `outputs/phase_r5/p4_diagnosis/seed*/`. Their SHA-256 values and byte sizes are frozen inside the canonical analysis artifact.
