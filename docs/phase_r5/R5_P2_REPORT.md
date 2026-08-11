# Phase R5-P2 — Progressive Return Pilot

Status: **R5-P2 FAIL / pending Supervisor review**  
Correctness: **PASS**  
Mechanism: **FAIL**  
Performance: **FAIL**  
Date: 2026-08-11

## 1. Executive conclusion

R5-P2 preserved the retained E0 progressive-forward baseline, full-size per-expert FP32 MLP
batches, return payload descriptors and non-progressive combine. P2 added only a dependency-aware
return worker: an unchanged return descriptor becomes legal when CUDA completion events show that
all full expert batches referenced by that descriptor have completed.

The implementation is correct, but the frozen R4 descriptor graph has no usable progressive-return
opportunity. Every existing forward-delta return descriptor contains tokens from both active local
experts. Therefore each descriptor depends on the final local expert batch. Across all **2,100 P2
descriptors**, none started NCCL GPU work before final expert completion and **0 us** of return GPU
work was hidden.

The paired full-MoE result is negative:

- `Delta_return = T_E0 - T_P2`: median **−2.935 ms**;
- 10,000-resample bootstrap 95% CI: **[−5.130, −2.061] ms**;
- positive pairs: **53/150**;
- seed medians: **−4.994 / −2.554 / −2.540 ms**; **0/3** positive;
- post-hoc paired relative makespan reduction `(E0-P2)/E0`: median **−0.564%**, 95% CI
  **[−1.163%, −0.460%]**.

Thus P2 neither establishes expert/return GPU overlap nor improves full-MoE makespan. The path must
stop at pilot. Splitting or regrouping return descriptors could create a different dependency graph,
but that would violate the authorized experiment and was not attempted.

## 2. Frozen paired experiment

### 2.1 Arms

- **E0:** retained progressive-forward full-MoE baseline; all full expert batches complete before
  executing return descriptors.
- **P2:** identical E0 path, except a worker non-blockingly observes per-expert completion events and
  launches each existing return descriptor as soon as all of its expert dependencies are complete.

The pairwise frozen items are Router/top-k, forward descriptors, expert input order, expert batch
membership, expert batch count and shapes, two GEMMs per active expert, expert weights, return
descriptor order/content/count/bytes, final payload multiset, and combine. Only legal return launch
time may differ.

### 2.2 Corpus and environment

- Fresh structured-search seeds: **11042 / 11142 / 11242**.
- Five unchanged traffic families × ten jobs × three seeds = **150 paired jobs**.
- Pair order deterministically counterbalanced.
- **2× Tesla V100-SXM2-32GB**, two NCCL ranks.
- Backend: PyTorch distributed NCCL uneven-split `all_to_all_single`; not MSCCL/MSCCL++, DeepEP,
  PCCL, or a production MoE backend.
- Frozen `partial_current_only`, `partial_shards@75%`, checkpoint8, compiled AICCL,
  reference deterministic packing, and A2Av-T0.
- No profiler was enabled; explicit CUDA events instrument expert and return streams.

Exact structured freshness was checked on both local and server histories before canonical
execution. A broad substring search was not used as freshness evidence because old timestamps,
durations and hashes naturally contain digit substrings such as `11042`; the structured seed-field
search had zero matches.

## 3. Correctness and equivalence

Correctness status: **PASS**.

| Check | Result |
|---|---:|
| Paired jobs completed | 150/150 |
| Rank-arm equivalence checks | 300/300 |
| Full-size P2 active expert batches | 480 |
| P2 reference GEMM launches | 960 |
| P2 return descriptors | 2,100 |
| P2 returned tokens | 1,228,800 |
| P2 return metadata+payload bytes | 157,286,400 |
| E0/P2 expert batches, shapes, count, weights and outputs | Exact match |
| E0/P2 return descriptors, order, tokens and bytes | Exact match |
| Maximum absolute final-output difference | 0.0 |
| Lost / duplicate / wrong expert / wrong destination | 0 |
| Wrong return / wrong position / corruption | 0 |
| Future/unrevealed return access | 0 |
| Runtime BFS / full rebuild | 0 / 0 |
| Legality / token integrity | 100% / 100% |

Both arms use the same explicit per-expert CUDA events, so instrumentation does not create an
asymmetric expert implementation. Historical R4 behavior remains the default when the opt-in P2
flags are absent. The existing R4 reference-MoE tests pass **2/2** after the extension.

## 4. Progressive-return mechanism

Mechanism status: **FAIL**.

| Diagnostic | Result |
|---|---:|
| Existing descriptors examined | 2,100 |
| Expert dependencies per descriptor p50/p95/p99/max | 2 / 2 / 2 / 2 |
| Return GPU starts before final expert | 0/2,100 |
| Return GPU completions before final expert | 0/2,100 |
| Descriptors with positive hidden return work | 0/2,100 |
| Hidden return GPU time p50/p95/p99/max | 0 / 0 / 0 / 0 us |
| Seeds with any early return start | 0/3 |

The key granularity mismatch is structural:

1. expert computation consists of one full batch per active local expert;
2. historical return descriptors are grouped by forward delta, not by expert;
3. each descriptor contains tokens produced by both active local experts;
4. consequently, every descriptor becomes complete only after the second/final expert batch.

P2 therefore legally degenerates to delayed return. It did not split a descriptor by expert, merge
descriptors, reorder collectives, or read a partially produced output buffer.

## 5. Primary full-MoE makespan

Positive delta favors P2. All primary timings run from the earliest first Router launch across ranks
to the latest actual combined output ready. Correctness-only reconstruction/checker work is excluded;
actual combine is included.

### 5.1 Corpus and seed results

| Scope | Median `E0-P2` (ms) | Positive pairs |
|---|---:|---:|
| Corpus | **−2.935** | **53/150** |
| Seed 11042 | −4.994 | 14/50 |
| Seed 11142 | −2.554 | 20/50 |
| Seed 11242 | −2.540 | 19/50 |

Corpus bootstrap 95% CI: **[−5.130, −2.061] ms**. The median, CI-lower and 3/3-seed preregistered
conditions all fail.

The paired relative normalization `(E0-P2)/E0` is **−0.564%**, 95% CI
**[−1.163%, −0.460%]**. This percentage is a post-hoc descriptive normalization of the same frozen
pairs, not a separate Gate.

### 5.2 Per-family results

| Family | Median `E0-P2` (ms) | Positive pairs |
|---|---:|---:|
| balanced | −2.034 | 13/30 |
| skewed | −5.956 | 6/30 |
| all-to-one-like | −1.501 | 14/30 |
| zero-sized-pair | −2.540 | 10/30 |
| multiple-progressive-shards | −6.161 | 10/30 |

All five family medians are negative. Individual pairs have large positive and negative tails, but
the paired corpus median and its confidence interval are unambiguously below zero.

### 5.3 Arm marginal makespans

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| E0 | 496.456 | 581.896 | 611.175 | 622.429 |
| P2 | 501.325 | 574.124 | 603.348 | 659.103 |

Marginal arm quantiles are diagnostic only; the Gate uses paired deltas.

## 6. Was the expert interval preserved?

The full expert batch identities, sizes, order, GEMM shapes and GEMM count are exactly equal. Typical
expert GPU interval is close, but P2 has a worse upper tail:

| Arm | p50 (us) | p95 (us) | p99 (us) | max (us) |
|---|---:|---:|---:|---:|
| E0 | 963.577 | 1,186.098 | 1,423.056 | 1,991.699 |
| P2 | 974.854 | 1,237.204 | 2,476.911 | 18,412.537 |

Paired P2-minus-E0 expert interval:

- absolute p50/p95/p99/max: **+3.586 / +242.780 / +1,390.690 / +17,149.963 us**;
- relative p50/p95/p99/max: **+2.345% / +30.453% / +128.041% / +1,797.328%**.

Because no return NCCL GPU work begins before final expert completion, these expert tails are not
evidence of expert/NCCL GPU contention. They are observed runtime/event-polling/system tails around
an otherwise identical expert sequence. The experiment establishes **zero actual expert/return GPU
coexistence**; it cannot claim a communication-contention mechanism.

## 7. How much return was hidden?

None. P2 return GPU work hidden before final expert is exactly **0 us** for every rank-arm. The entire
return tail remains after expert completion:

| Arm | return tail p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| E0 | 54.352 | 159.876 | 165.124 | 244.935 |
| P2 | 55.443 | 161.497 | 173.448 | 254.119 |

Per-descriptor return GPU duration:

| Arm | p50 (us) | p95 (us) | p99 (us) | max (us) |
|---|---:|---:|---:|---:|
| E0 | 763.916 | 14,674.484 | 108,588.149 | 114,831.360 |
| P2 | 815.598 | 14,298.026 | 107,318.510 | 123,704.285 |

P2 dependency polling p50/p95/p99/max is **5.773 / 13.633 / 17.437 / 29.951 us** per descriptor,
versus the no-wait E0 bookkeeping path **0.226 / 0.307 / 0.370 / 0.670 us**. With no legal early
descriptor, this is overhead without hidden communication.

## 8. Stage diagnostics

### 8.1 Host stages, p50 / p95 / p99

| Stage | E0 | P2 |
|---|---:|---:|
| Router (ms) | 5.966 / 7.910 / 11.065 | 5.931 / 7.175 / 11.269 |
| Forward stage (ms) | 379.853 / 469.018 / 508.142 | 379.069 / 463.757 / 490.560 |
| Expert compute + D2H (ms) | 21.099 / 43.843 / 45.823 | 24.370 / 50.139 / 52.669 |
| Return stage (ms) | 54.376 / 159.835 / 165.075 | 55.912 / 161.051 / 169.492 |
| Actual combine (us) | 178.987 / 243.862 / 284.039 | 188.233 / 262.909 / 304.879 |

The `expert compute + D2H` host stage includes host-visible synchronization/copy overhead and should
not be confused with the CUDA expert interval in Section 6.

### 8.2 Frozen packing and count-exchange tails

| Metric | E0 p50/p95/p99/max (us) | P2 p50/p95/p99/max (us) |
|---|---:|---:|
| Forward packing | 11,127.970 / 23,575.870 / 33,040.257 / 37,190.758 | 11,129.187 / 23,777.976 / 33,438.456 / 48,078.751 |
| Forward count exchange | 290.272 / 29,982.701 / 34,980.550 / 217,237.738 | 301.117 / 29,956.001 / 34,140.642 / 132,203.187 |
| Return packing | 3,020.331 / 8,699.015 / 12,652.201 / 98,487.633 | 3,047.039 / 8,800.397 / 12,635.230 / 102,216.432 |
| Return count exchange | 211.540 / 14,225.925 / 108,109.518 / 114,339.042 | 232.610 / 13,842.857 / 106,643.342 / 123,084.776 |

These tails were measured but not optimized, exactly as prohibited by the phase authorization.

## 9. Smoke and non-adaptation record

One already-consumed seed 8042/balanced pair was used only for correctness and timeline smoke. It
passed all equivalence checks and already showed **0** return GPU starts before final expert. No
descriptor, workload, stream, batching, packing, count-exchange or transport setting was changed
afterward. The smoke is excluded from the canonical 150-pair statistics.

## 10. Interpretation and decision

R5-P2 is a structural negative result, not merely a small speedup miss:

- **Correctness is solved.** Full expert batches and return payload semantics remain exact.
- **There is no legal overlap at the preserved granularity.** Every return descriptor needs both
  active expert batches and becomes ready only at the final expert boundary.
- **No GPU communication contention was introduced.** There is zero measured expert/return GPU
  coexistence.
- **The added observation path has net negative value.** Median full-MoE makespan is 2.935 ms
  slower, or a descriptive 0.564% regression.

Therefore **R5-P2 FAIL pending Supervisor review**. No formal experiment is justified. Any future
attempt would require separately authorizing a changed descriptor granularity or a different return
layout; that would be a new architecture, not an optimization or continuation of this P2 result.

## 11. Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| [Preregistration](R5_P2_PREREGISTRATION.md) | `48032adfe74780f4689632a7e655d4913f1a2cb7682ed8b0ca5ccc7de7d3f7ce` |
| [Canonical raw artifact](../../outputs/phase_r5/p2_progressive_return_pilot/r5_p2_primary_host.json) | `dc6e890de546828f91f6d34dc8e7eb4c61790fc70c14eaead6aadd74c4169b98` |
| [Canonical analysis](../../outputs/phase_r5/p2_progressive_return_pilot/r5_p2_results.json) | `d533f2e58980cdc8da90d47a4e6290b4b2b8f8af4af07cbc05cfdf647037f9aa` |
| [P2 runner](../../scripts/run_r5_p2_progressive_return.py) | `646ebf64ca397dbc700f4ca951731dbca43b6aeee03342eb5af14dc148595e66` |
| [P2 analyzer](../../scripts/analyze_r5_p2_progressive_return.py) | `b287648a66eb161ef7fccb94aac2eaa7b62c181c6f6806b4cc97dd22a802623a` |
| [Opt-in R4/P2 substrate](../../scripts/run_r4_a0_c0_full_moe.py) | `f7aeaa751e5f0741bcbb3b88c9e08ac1bee5823a8d913de491d55d95d25e3ac7` |

Remote canonical directory:
`/root/autodl-tmp/RLCCL-main/outputs/phase_r5/p2_progressive_return_pilot/`.
Canonical raw and analysis hashes match after download. The pilot did not use or generate a formal
corpus.
