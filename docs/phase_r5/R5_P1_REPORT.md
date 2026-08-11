# Phase R5-P1 — Progressive Expert Execution Pilot

Status: **R5-P1 FAIL / pending Supervisor review**  
Correctness and mechanism: **PASS**  
Incremental performance gate: **FAIL**  
Date: 2026-08-11

## 1. Executive conclusion

R5-P1 implemented the authorized progressive-expert mechanism on top of the frozen R4 reference full-MoE path. Tokens received by an expert are appended to a per-expert FIFO ready buffer. A fixed, preregistered threshold of 256 tokens launches the expert MLP on an independent CUDA stream; after the final forward dispatch, each non-empty remainder is flushed. Return AlltoAllv and combine remain non-progressive.

The mechanism is correct and hides most observed expert GPU work behind the forward stage. However, under the frozen 2×V100 pilot workload, the extra fine-grained/contended expert execution greatly increases the observed expert GPU interval and forward-stage critical path. As a result, progressive expert execution is slower than the R4-style delayed-expert baseline:

- Incremental expert effect, `Delta_expert = T_E0 - T_P`: median **−53.951 ms**, bootstrap 95% CI **[−57.851, −47.824] ms**; **7/150** pairs positive; **0/3** seeds positive.
- Paired relative makespan reduction, `(T_E0 - T_P) / T_E0`: median **−11.016%**, 95% CI **[−11.927%, −10.397%]**. Under this convention, P is about 11.0% slower than E0 at the median.
- Total progressive pipeline effect, `Delta_pipeline = T_D - T_P`: median **−44.920 ms**, 95% CI **[−50.513, −39.587] ms**; **6/150** pairs positive; **0/3** seeds positive.
- Total paired relative makespan reduction, `(T_D - T_P) / T_D`: median **−9.636%**, 95% CI **[−10.029%, −9.141%]**.

The R4 forward-only signal reproduced on the same fresh corpus: `T_D - T_E0` median **+4.204 ms**, 95% CI **[+2.620, +6.680] ms**, with all three seeds positive. The R5 failure is therefore attributable to the newly enabled progressive-expert mechanism in this reference implementation, not a disappearance of the established progressive-forward effect.

No threshold was changed after observing the smoke or canonical results, and no packing, count-exchange, communication, GEMM, scheduler, or workload optimization was performed.

## 2. Frozen experiment

### 2.1 Arms

- **P — progressive expert:** progressive forward A2Av plus threshold-triggered expert execution; return/combine wait for all experts.
- **E0 — delayed expert control:** the same progressive forward A2Av, but expert execution starts only after all forward dispatch completes.
- **D — delayed forward control:** identical forward descriptors execute only after final Router completion; expert/return/combine are delayed as in E0.

For each three-arm pair, tokens, Router/top-k, expert mapping, expert weights, forward descriptors, expert inputs, final per-expert batches, return descriptors, payload multiset, and final outputs are identical. Only forward launch timing and, for P versus E0, expert launch timing differ.

### 2.2 Preregistered implementation

- Fixed expert threshold: **256 tokens per expert**.
- Buffer order: deterministic per-expert FIFO.
- Remainder: one final flush after all forward dispatch completes.
- Expert stream: independent CUDA stream.
- Expert model: frozen R4 non-production FP32 reference MLP.
- Return variable A2Av and combine: non-progressive and unchanged.
- Pilot seeds: **10042, 10142, 10242**.
- Corpus: five frozen traffic families × ten jobs × three seeds = **150 three-arm pairs**.
- Hardware/backend: **2× Tesla V100-SXM2-32GB**, two ranks, PyTorch distributed **NCCL** uneven-split `all_to_all_single`; not MSCCL/MSCCL++, DeepEP, or a production MoE backend.

The preregistration was written before the canonical seeds were executed. Exact freshness searches found no prior structured matches locally or on the server.

### 2.3 Gate

Correctness/mechanism required exact-once expert execution, no future/unrevealed access, final-output equivalence, and observable pre-forward expert execution. The incremental performance gate was frozen as:

1. median `T_E0 - T_P > 0`;
2. paired bootstrap 95% CI lower bound > 0;
3. all three independent seed medians > 0.

The total `T_D - T_P` effect was also reported, but it does not replace the incremental expert gate.

## 3. Correctness and mechanism

Correctness/mechanism status: **PASS**.

| Check | Result |
|---|---:|
| Canonical three-arm pairs | 150/150 completed |
| P rank-arms | 300 |
| Expert token executions | 1,228,800 |
| Token loss | 0 |
| Duplicate expert execution | 0 |
| Future/unrevealed expert execution | 0 |
| Final-output pair equivalence | 100% |
| Maximum absolute final-output difference | 1.66893e-6 |
| Runtime BFS | 0 |
| Fast-path full rebuild | 0 |
| Seeds with pre-forward expert batches | 3/3 |
| Positive observed hidden expert GPU time | Yes |

No P batch executed before its tokens arrived through a completed forward descriptor. Every token was processed exactly once with the same expert weights and mapped back to the same original position. Return and combine began only after all expert work completed.

## 4. Primary performance results

All following deltas use paired three-arm samples. Positive means progressive execution is faster; negative means it is slower.

### 4.1 Incremental progressive-expert effect: E0 − P

| Scope | Median delta (ms) | Positive pairs |
|---|---:|---:|
| Corpus | **−53.951** | **7/150** |
| Seed 10042 | −57.017 | 5/50 |
| Seed 10142 | −47.074 | 2/50 |
| Seed 10242 | −55.555 | 0/50 |
| balanced | −38.459 | 2/30 |
| skewed | −57.465 | 1/30 |
| all-to-one-like | −66.605 | 0/30 |
| zero-sized-pair | −66.110 | 1/30 |
| multiple-progressive-shards | −30.878 | 3/30 |

Corpus bootstrap 95% CI: **[−57.851, −47.824] ms**. The gate fails all three conditions.

The post-hoc descriptive paired normalization `(E0-P)/E0` is **−11.016%** at the median, 95% CI **[−11.927%, −10.397%]**. This percentage is not a preregistered Gate and does not change the absolute-time decision.

### 4.2 Total pipeline effect: D − P

| Scope | Median delta (ms) | Positive pairs |
|---|---:|---:|
| Corpus | **−44.920** | **6/150** |
| Seed 10042 | −42.499 | — |
| Seed 10142 | −44.602 | — |
| Seed 10242 | −49.962 | — |
| balanced | −33.981 | — |
| skewed | −49.452 | — |
| all-to-one-like | −64.674 | — |
| zero-sized-pair | −56.211 | — |
| multiple-progressive-shards | −27.718 | — |

Corpus bootstrap 95% CI: **[−50.513, −39.587] ms**. The paired normalization `(D-P)/D` is **−9.636%**, 95% CI **[−10.029%, −9.141%]**.

### 4.3 Forward-only control: D − E0

- Corpus median: **+4.204 ms**.
- Bootstrap 95% CI: **[+2.620, +6.680] ms**.
- Positive pairs: **100/150**.
- Seed medians: **+4.793 / +3.545 / +5.758 ms**; 3/3 positive.
- All five family medians are positive.

This is a fresh-corpus control confirming that progressive forward dispatch still has positive critical-path value. It is not added to the R4 formal result and is not used to rescue the R5-P1 gate.

## 5. How much expert compute was hidden?

The progressive mechanism launched **4,955 expert batches** across 300 P rank-arms, corresponding to **9,910 GEMM launches** (two GEMMs per reference MLP batch).

| Batch diagnostic | Result |
|---|---:|
| Full 256-token threshold batches | 4,625 (93.340%) |
| Remainder flush batches | 330 (6.660%) |
| Batches completed before final forward completion | 4,625 |
| Tokens completed before final forward completion | 1,184,000 / 1,228,800 (96.354%) |
| Batches/tokens completed before final Router completion | 0 |

Observed positive expert GPU interval hidden before final forward completion:

- duration p50/p95/p99/max: **2.505 / 5.155 / 5.978 / 21.077 ms**;
- hidden fraction p50/p95/p99: **87.233% / 100% / 100%**.

The absence of pre-Router completion batches is expected for this workload: the first usable expert inputs arrive after Router chunks have completed and forward communication has delivered them. The relevant R5 overlap boundary is expert work versus the remainder of forward dispatch, not expert work versus unfinished Router kernels.

## 6. GEMM batch distribution and efficiency degradation

### 6.1 Batch sizes

- All batches: min **1**, mean **247.992**, median/p95/p99/max **256/256/256/256**, with 160 distinct sizes.
- Threshold batches: 4,625 batches, all size **256**.
- Remainders: 330 batches totaling 44,800 tokens; min **1**, mean **135.758**, median **164**, p95 **248.55**, p99 **254**, max **255**.
- Per-expert batch-count histogram for 0/1/2/3 batches: **1,945 / 1,329 / 835 / 846** expert instances.

### 6.2 Observed expert GPU interval

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| P | 2.866 | 5.155 | 5.978 | 26.170 |
| E0 | 0.926 | 1.207 | 1.684 | 2.179 |
| D | 0.893 | 1.164 | 1.435 | 3.269 |

Paired P-versus-E0 observed expert GPU-interval degradation is **+207.204% p50**, **+390.099% p95**, **+426.746% p99**, with max **+1,397.197%**.

This is an in-pipeline CUDA-event interval under communication and Router contention, not an isolated kernel-efficiency benchmark. It includes scheduling gaps and device contention between the first and last expert events; it must not be interpreted as a pure GEMM throughput regression. Nevertheless, it accurately captures the cost imposed on the measured end-to-end reference pipeline.

Progression reduced the post-forward expert tail—p50 **6.642 ms** for P versus **22.663 ms** for E0—but the much larger forward/device contention cost outweighed that hidden tail.

## 7. Latency diagnostics

### 7.1 Primary full-MoE makespan

| Arm | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| P | 532.664 | 630.640 | 670.573 | 701.191 |
| E0 | 476.481 | 560.192 | 583.373 | 607.692 |
| D | 481.687 | 567.021 | 618.325 | 640.162 |

The primary boundary is first Router launch to actual combined output ready. Correctness-only oracle/checker work is excluded; actual combine is included.

### 7.2 Marginal stage latency, p50 / p95 / p99

| Stage | P | E0 | D |
|---|---:|---:|---:|
| Forward stage (ms) | 429.391 / 525.576 / 554.337 | 370.451 / 443.535 / 473.861 | 369.798 / 444.234 / 480.948 |
| Forward packing (ms) | 11.212 / 23.268 / 30.849 | 11.168 / 23.127 / 31.247 | 11.156 / 23.051 / 30.943 |
| Forward count exchange (ms) | 0.289 / 37.600 / 42.165 | 0.288 / 27.540 / 31.190 | 0.321 / 26.575 / 28.624 |
| Expert boundary-to-D2H (ms) | 17.324 / 30.517 / 32.743 | 22.874 / 44.977 / 46.680 | 22.747 / 44.805 / 46.976 |
| Return stage (ms) | 53.239 / 168.252 / 174.386 | 52.980 / 160.887 / 169.907 | 53.385 / 164.315 / 176.299 |
| Return count exchange (ms) | 0.232 / 14.758 / 114.190 | 0.241 / 13.960 / 106.996 | 0.233 / 14.494 / 109.140 |
| Actual combine (us) | 171.988 / 209.214 / 259.362 | 179.182 / 234.617 / 271.008 | 179.025 / 221.840 / 291.388 |

Forward packing is essentially unchanged, as required. P's forward stage is substantially longer, while return and combine remain broadly comparable. Count-exchange tails remain visible but were not altered in R5-P1.

## 8. Smoke and non-adaptation record

A one-pair smoke run used seed 8042/balanced only to verify executability and instrumentation. It already showed a negative incremental result (**−232.574 ms**) with 34 expert batches, 30 threshold batches and four remainder batches. The threshold, stream policy, workload, MLP shape, and measurement protocol were not changed afterward. The canonical corpus then ran exactly as preregistered.

The smoke result is not pooled into the 150-pair pilot statistics.

## 9. Interpretation and boundary

R5-P1 answers a narrower question than R4: whether the straightforward reference design “launch each expert at 256 ready tokens on a separate CUDA stream” increases the already demonstrated progressive-forward benefit. It does not. The mechanism exposes abundant theoretical overlap—96.354% of expert tokens complete before final forward—but that overlap is not free. More fragmented MLP launches and concurrent use of the V100 device increase the forward/device critical path enough to erase the saved expert tail.

This is a useful negative result:

- **Safety is not the blocker.** Progressive expert execution preserves mappings, exact-once execution, and final outputs.
- **Available overlap is not the same as net benefit.** Hidden work and overlap fraction do not replace combined makespan.
- **The current bottleneck is device execution efficiency/co-scheduling.** It is not the compiled AICCL control path or forward packing.

The result does not prove that every possible progressive-expert architecture must fail. It does show that this fixed-threshold, reference-PyTorch, independent-stream implementation on two V100 GPUs is not a viable performance improvement without a separately authorized redesign. No alternate threshold, fused/grouped expert kernel, CUDA graph, persistent kernel, MSCCL, DeepEP, packing optimization, or count-exchange optimization was tried.

R5-P1 should therefore stop at **FAIL pending Supervisor review**. No formal validation or next optimization phase is recommended under the present authorization.

## 10. Artifacts and provenance

| Artifact | SHA-256 |
|---|---|
| [Preregistration](R5_P1_PREREGISTRATION.md) | `f38c3379bf4301f557a1e3f77f68cb350705987d9604e9fcef2987a69939b0bd` |
| [Canonical raw artifact](../../outputs/phase_r5/p1_progressive_expert_pilot/r5_p1_primary_host.json) | `478e4b1d48073165111fd6435529bb2c6b60a349773a49f309a132ca7ba8d240` |
| [Canonical analysis](../../outputs/phase_r5/p1_progressive_expert_pilot/r5_p1_results.json) | `b43fb88a4116e341839cee70ebf7cbc2f9c845e09dbd591cab6eb3f7aaa21146` |
| [R5 runner](../../scripts/run_r5_p1_progressive_expert.py) | `5c43a44e9254f9d4b40ac0a38b35a8c0ff6e7bd768311d935605a9e01fa8ca33` |
| [R5 analyzer](../../scripts/analyze_r5_p1_progressive_expert.py) | `424bfa1529c31ec4e658957996c32f1aabb9d3e4fb7419e637b44be613547f88` |
| [R4 substrate with opt-in R5 path](../../scripts/run_r4_a0_c0_full_moe.py) | `d92b9aa732789767520e1e5de7a978ee13526742786d25ecdb79b23370188e6b` |

Canonical remote location: `/root/autodl-tmp/RLCCL-main/outputs/phase_r5/p1_progressive_expert_pilot/`. The canonical raw and analysis files were downloaded unchanged and their hashes read back locally. Existing R4 unit tests passed **2/2** after the opt-in substrate extension; historical R4 behavior remains the default.
