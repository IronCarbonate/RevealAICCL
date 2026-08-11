# Phase R5-P3 — Fast Progressive Data Preparation Pilot

Status: **PARTIAL PASS / overall preregistered Gate FAIL pending Supervisor review**  
Correctness: **PASS**  
Packing mechanism: **PASS**  
E0 → E1 fast full-MoE: **PASS**  
Optimized D1 − E1 progressive comparison: **FAIL**  
Date: 2026-08-11

## 1. Executive conclusion

R5-P3 replaced the progressive forward reference data-preparation path with an opt-in preallocated,
incremental and vectorized implementation. The fast path preserves Router/top-k, AICCL decisions,
seven descriptor boundaries/order, sendcounts, offsets, metadata bytes, feature bytes, expert/return
execution and final outputs exactly.

The data-preparation optimization itself succeeds strongly:

- forward packing p50 falls from **10.844 ms** to **0.828 ms** per descriptor;
- paired packing speedup is **13.214× p50**;
- count construction p50 falls from **503.584 us** to **23.193 us**;
- paired full-MoE `E0-E1` median is **+78.872 ms**, 95% CI
  **[+76.626, +80.810] ms**;
- paired relative makespan reduction is **+16.423%**, 95% CI
  **[+15.722%, +18.253%]**;
- **147/150** pairs, **3/3** seeds and all five family medians are positive.

Static route-independent precompute is explicitly excluded from the inherited first-Router-launch
primary clock but retained as a secondary cost. Adding the entire E1 precompute back to every pair
still gives a paired median improvement of **+3.498 ms**, 95% CI **[+1.400, +5.549] ms**.

However, after both arms use the optimized preparation path, progressive E1 is slower than optimized
delayed D1:

- `D1-E1` median **−3.093 ms**;
- 95% CI **[−4.189, −1.813] ms**;
- paired relative reduction **−0.775%**, 95% CI **[−1.075%, −0.455%]**;
- **51/150** pairs positive; **0/3** seed medians and **0/5** family medians positive.

Therefore P3 is not a full progressive-pipeline PASS. It establishes a valid fast data-preparation
backend and a large E0→E1 improvement, but the earlier progressive-forward timing advantage does not
survive when compared against the equally optimized delayed arm.

## 2. Frozen experiment

### 2.1 Arms and contrasts

- **E0:** retained progressive-forward reference data-prep baseline.
- **E1:** identical progressive timing with fast preallocated/incremental/vectorized data prep.
- **D1:** the same fast data prep as E1, but forward communication waits until final Router
  completion.

Two independent preregistered contrasts are reported:

1. `Delta_fast = T_E0 - T_E1`, measuring the data-prep optimization;
2. `Delta_progressive = T_D1 - T_E1`, measuring progressive timing after both arms are optimized.

The contrasts are not added to each other or to historical R4 results.

### 2.2 Corpus and environment

- Fresh structured-search seeds: **12042 / 12142 / 12242**.
- Five unchanged traffic families × ten jobs × three seeds = **150 paired three-arm jobs**.
- Six arm orders deterministically counterbalanced.
- **2× Tesla V100-SXM2-32GB**, two PyTorch distributed NCCL ranks.
- Real uneven-split `all_to_all_single`; not MSCCL/MSCCL++, DeepEP, PCCL or production packing.
- Frozen `partial_current_only`, `partial_shards@75%`, checkpoint8, compiled AICCL and A2Av-T0.
- Expert execution, return A2Av and combine remain non-progressive and unchanged.

Exact structured freshness was zero locally and on the server before canonical execution. A single
already-consumed seed 8042 smoke was excluded from canonical analysis and did not change the frozen
implementation.

## 3. Implemented fast path

### 3.1 Preallocated buffers

- Pinned per-chunk/per-destination metadata and feature buffers.
- One pinned output buffer for each of the unchanged seven descriptor slots.
- Fixed descriptor partition: `(0),(1),(2),(3),(4),(5),(6,7)`.
- No per-descriptor token lists or dynamically selected buffer sizes.

### 3.2 Incremental Router-derived state

Only route-independent feature digests and static token identity fields are prepared before Router
launch. On actual chunk completion, real top-k expert IDs determine destination. The runtime performs
a vectorized destination scatter and updates a fixed `[chunk,destination]` count matrix. Descriptor
counts use a delta sum; offsets use a prefix sum.

No future top-k, expert ID or destination is precomputed. Completed/revealed/dispatched bitmaps and
fail-closed future/unrevealed/duplicate/stale checks are retained.

### 3.3 Vectorized packing

Already grouped buffers are copied through contiguous NumPy/Torch-backed slices into the fixed
descriptor buffer. Feature dictionary lookups, per-token list append, per-descriptor sort and
per-token feature hashing are removed from the timed reveal path. Dynamic checksum completion remains
Router-derived and byte-matches E0.

### 3.4 Count exchange overlap

For E1, the real NCCL delta-count exchange starts after revealed counts are known but before packing
finishes. It uses an independent count stream while CPU packing and pinned payload H2D progress.
D1 preserves strict delayed semantics: it does not launch count or payload communication before the
final Router boundary, though it still overlaps count exchange with payload H2D after release.

## 4. Correctness and equivalence

Correctness status: **PASS**.

| Check | Result |
|---|---:|
| Three-arm paired jobs | 150/150 |
| Rank-level equivalence | 300/300 |
| Forward descriptors per arm | 2,100 |
| Forward tokens per arm | 1,228,800 |
| Forward metadata+feature bytes per arm | 10,154,803,200 |
| E1/D1 descriptors byte-equal to E0 | 100% |
| Same sendcounts/offsets/order/bytes | 100% |
| Same scheduler actions/checker | 100% |
| Same expert batches/weights/outputs | 100% |
| Same return descriptors | 100% |
| Maximum final-output absolute difference | 0.0 |
| Lost/duplicate/wrong/corruption/future | 0 |
| Runtime BFS / full rebuild | 0 / 0 |
| Legality / token integrity | 100% / 100% |

The existing R4 reference-MoE regression tests remain **2/2 PASS**.

## 5. Packing and construction speedup

### 5.1 Per-descriptor latency

| Metric | E0 p50 | E1 p50 | D1 p50 |
|---|---:|---:|---:|
| Count construction | 503.584 us | 23.193 us | 23.398 us |
| Offset construction | 0.800 us | 18.354 us | 17.167 us |
| Packing | 10,843.896 us | 827.805 us | 830.108 us |

Fast count construction is approximately **21.71×** faster at p50. Offset construction is slower
because the vectorized prefix-sum setup costs more than a two-element Python loop; the roughly
17.5 us regression is small relative to the packing reduction and is retained as a negative
micro-result.

### 5.2 Packing distribution

| Arm | p50 (us) | p95 (us) | p99 (us) | max (us) |
|---|---:|---:|---:|---:|
| E0 | 10,843.896 | 22,946.763 | 32,260.843 | 34,395.750 |
| E1 | 827.805 | 1,652.165 | 2,249.149 | 4,165.277 |
| D1 | 830.108 | 1,647.733 | 2,244.550 | 3,515.607 |

Across 2,100 paired E0/E1 descriptors:

- median absolute reduction: **10,021.718 us per descriptor**;
- paired speedup p50/p95/p99/max: **13.214× / 14.928× / 16.950× / 20.071×**.

Packing mechanism Gate: **PASS**.

### 5.3 Work moved to incremental/static stages

- E1 vectorized `mark_completed` total per rank-arm: p50 **13.801 ms**, p95 **14.931 ms**.
- D1: p50 **13.519 ms**.
- Static route-independent precompute: E1 p50 **74.875 ms**, p95 **77.492 ms**;
  D1 p50 **74.905 ms**, p95 **77.373 ms**.

These costs are not hidden. `mark_completed` lies inside primary timing. Static precompute lies before
the inherited Router-launch boundary and is added back in Section 7.3.

## 6. Count-exchange behavior

### 6.1 E1 overlap mechanism

All **2,100/2,100** E1 forward descriptors used pinned metadata/features and prestarted count
exchange before packing completed. The host interval during which a count ticket remained outstanding
overlapped the entire packing interval at p50/p95/p99:

- overlap duration p50/p95/p99/max: **827.805 / 1,652.165 / 2,249.149 / 4,165.277 us**;
- fraction of packing interval covered: **100% p50/p95/p99**.

This does not mean the NCCL GPU kernel itself lasts the whole packing interval. E1 count GPU duration
is only **131.072 us p50**, **209.971 us p95**, **253.952 us p99**. The 100% figure describes the
host-visible in-flight ticket interval, while GPU-event duration describes actual count-stream work.

### 6.2 Count latency and residual wait

| Arm | host-visible p50/p95/p99/max (us) | residual wait p50/p95/p99/max (us) |
|---|---:|---:|
| E0 | 255.434 / 27,815.159 / 32,209.005 / 104,875.863 | 57.275 / 27,666.939 / 32,047.348 / 104,735.973 |
| E1 | 12,703.046 / 39,834.892 / 42,574.331 / 98,762.390 | 70.695 / 27,353.014 / 29,469.718 / 74,071.535 |
| D1 | 451.505 / 27,306.861 / 29,388.932 / 74,323.490 | 80.230 / 27,112.544 / 29,194.515 / 73,868.511 |

E1 host-visible count latency is measured from the earlier prestart, so it deliberately includes time
spent packing/hashing before the host consumes the result; it is not blocking latency. Residual wait
is the relevant critical-path diagnostic. Prestart removes most median blocking but the roughly
27 ms p95 rendezvous tail remains. R5-P3 did not change the NCCL transport or tune this tail.

## 7. Full-MoE results

### 7.1 Fast data prep: E0 − E1

| Scope | Median gain (ms) | Positive pairs |
|---|---:|---:|
| Corpus | **+78.872** | **147/150** |
| Seed 12042 | +82.059 | 49/50 |
| Seed 12142 | +76.262 | 49/50 |
| Seed 12242 | +79.438 | 49/50 |
| balanced | +80.148 | 30/30 |
| skewed | +75.799 | 29/30 |
| all-to-one-like | +78.003 | 30/30 |
| zero-sized-pair | +77.215 | 30/30 |
| multiple-progressive-shards | +81.531 | 28/30 |

- Bootstrap 95% CI: **[+76.626, +80.810] ms**.
- Paired relative makespan reduction: **+16.423%**.
- Relative 95% CI: **[+15.722%, +18.253%]**.

Fast full-MoE Gate: **PASS**.

### 7.2 Optimized progressive timing: D1 − E1

| Scope | Median gain (ms) | Positive pairs |
|---|---:|---:|
| Corpus | **−3.093** | **51/150** |
| Seed 12042 | −0.818 | 23/50 |
| Seed 12142 | −4.387 | 13/50 |
| Seed 12242 | −3.262 | 15/50 |
| balanced | −1.971 | 13/30 |
| skewed | −3.631 | 7/30 |
| all-to-one-like | −4.048 | 7/30 |
| zero-sized-pair | −2.277 | 11/30 |
| multiple-progressive-shards | −1.205 | 13/30 |

- Bootstrap 95% CI: **[−4.189, −1.813] ms**.
- Paired relative makespan reduction: **−0.775%**.
- Relative 95% CI: **[−1.075%, −0.455%]**.

Optimized progressive Gate: **FAIL**. All seed and family medians are negative.

### 7.3 Static-precompute-inclusive diagnostic

For each pair, the slower rank's complete E1 static setup is added to E1 primary makespan before
comparison with E0:

- `E0 - (E1 primary + E1 precompute)` median: **+3.498 ms**;
- 95% CI: **[+1.400, +5.549] ms**;
- positive pairs: **94/150**.

This secondary result shows that the E0→E1 gain is not solely a timing-boundary artifact, although
its conservative magnitude is much smaller than the inherited primary result.

### 7.4 Marginal arm makespans

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| E0 | 474.888 | 555.587 | 595.745 | 597.318 |
| E1 | 400.424 | 471.448 | 523.549 | 546.191 |
| D1 | 393.543 | 469.538 | 507.591 | 527.589 |

The Gate uses paired deltas, not differences between marginal quantiles.

## 8. Downstream stages and bottleneck shift

| Stage p50 | E0 | E1 | D1 |
|---|---:|---:|---:|
| Forward stage | 369.734 ms | 272.548 ms | 284.266 ms |
| Expert + D2H | 22.972 ms | 22.871 ms | 22.860 ms |
| Return stage | 52.925 ms | 56.271 ms | 54.069 ms |
| Actual combine | 177.832 us | 177.875 us | 177.832 us |

Expert and combine medians remain essentially unchanged, consistent with the frozen downstream path.
Return/NCCL tails remain important. The negative D1−E1 result indicates that, once Python packing is
removed, early forward communication no longer has positive net critical-path value in this pilot.
Early launch/rendezvous/device effects are a plausible explanation, but R5-P3 did not run a new GPU
profiler comparison and therefore does not claim a definitive causal decomposition.

## 9. Smoke and non-adaptation record

The one-pair seed-8042 smoke passed byte-exact equivalence and packing mechanism checks. Its
optimized progressive delta was negative. No buffer size, overlap rule, stream, descriptor, workload,
count protocol or corpus was changed afterward. Smoke data is not pooled into canonical statistics.

## 10. Decision

R5-P3 yields a split decision:

- **Retain the fast data-preparation implementation as a successful candidate:** it is byte-exact,
  reduces packing by 13.214× p50 and improves E1 versus E0 by 16.423% paired median.
- **Do not claim optimized progressive benefit:** D1 is faster than E1 by 3.093 ms paired median.
- **Do not run formal validation under the present authorization:** one preregistered primary Gate
  fails, and formal was explicitly prohibited.

Overall status remains **FAIL pending Supervisor review**, with correctness, packing mechanism and
E0→E1 optimization recorded as PASS subresults. Any next phase must explicitly decide whether the
project objective shifts to the optimized E1 backend itself or reopens device/transport scheduling;
P3 does neither automatically.

## 11. Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| [Preregistration](R5_P3_PREREGISTRATION.md) | `7bc83b3389db650fa02b34b2a0ebd599730c9cc9f55dcfeb9fd6514bb1a2e430` |
| [Canonical raw artifact](../../outputs/phase_r5/p3_fast_data_prep_pilot/r5_p3_primary_host.json) | `715c08c43e8dd062b4a2c80b0cf4ff02f3885c1e3d43fd77358469d61b3d1620` |
| [Canonical analysis](../../outputs/phase_r5/p3_fast_data_prep_pilot/r5_p3_results.json) | `b415a7ddd6d5f61126481b311909604f79f4a00fc5652244725e688822f139e1` |
| [Fast data-prep implementation](../../rlccl/transport/fast_progressive_data_prep.py) | `871d63ba4acc421a0560434d4e52d69b26d59c1bff737a28b0230d55b73fca2d` |
| [P3 runner](../../scripts/run_r5_p3_fast_data_prep.py) | `d256f050efd1739bb9a3339eb57c10e9412bab2e5f3726eef04503215e5430a2` |
| [P3 analyzer](../../scripts/analyze_r5_p3_fast_data_prep.py) | `5937a30c40f9dbae951abba9f80e12221afb2b5d5f65de52fcbae3a8e7debf86` |
| [Opt-in full-MoE substrate](../../scripts/run_r4_a0_c0_full_moe.py) | `149103efa74b53a437289e3e39e654cc0c328417ec1eea24a55fb91cd24dfe08` |

Remote canonical directory:
`/root/autodl-tmp/RLCCL-main/outputs/phase_r5/p3_fast_data_prep_pilot/`.
Canonical raw/analysis files were downloaded unchanged and their hashes match. No formal corpus was
run or generated.
