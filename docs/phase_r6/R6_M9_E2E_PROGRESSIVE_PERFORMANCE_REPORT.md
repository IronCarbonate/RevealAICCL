# R6-M9 End-to-End Progressive Performance Validation Report

## Result

**Correctness: PASS**

**Fairness: PASS**

**Progressive Mechanism: PASS**

**Performance: PASS**

Across `1200` measured P/D pairs, the paired median `D-P` is
`0.034303903579711914 ms`. The `10000`-sample paired-bootstrap 95% confidence
interval is `[0.015360116958618164, 0.04198408126831055] ms`, entirely above
zero. The pre-registered primary endpoint therefore passes. Median P and D
makespans are `3.422208070755005 ms` and `3.4406399726867676 ms`; the paired
median reduction divided by median D is `0.9970210150446017%`.

## 1. Frozen M7/M8 boundary

Router/top-k computation, reveal policy, `RevealRecord`, GPU Scheduler,
`DescriptorCommit`, `CommitPeerPlan`, queue layout, token-centric packing,
direct LSA stores, expert-contiguous epilogue, `ProgressiveEPHandle`, expert
GEMM, direct LSA return, and deterministic top-k reduction retain their M7/M8
semantics. M9 does not introduce a new packing path, transport backend, expert
kernel, combine kernel, buffer layout, or numerical rule.

The four original persistent roles retain their 256-thread bodies: Router,
Scheduler, dispatch, and remote acquire wait. M9 adds one control role whose
only active thread gates commit publication to the frozen dispatch consumer.
Both experimental arms execute the same five-role kernel, the same epilogue,
the same GEMM, the same combine, and the same stream ordering.

## 2. Sole P/D variable

Scheduler writes every legal result to a producer commit queue. The M9 control
role forwards those same `DescriptorCommit` values to the consumer queue:

- P forwards each available commit immediately;
- D first waits until `final_router_ns != 0`, then forwards every commit.

The sole arm branch is `arm == kDelayedArm` inside the control role. Router and
Scheduler continue to reveal and commit normally in D; only data-plane
consumption is delayed. The consumer remains the frozen M7 fused dispatch
role, so D does not substitute a kernel, packer, transport, or combine path.

## 3. Repeatable NCCL completion namespace

M7 and M8 were one-shot correctness gates. M9 reuses one communicator for
hundreds of trials, so reusing the same LSA barrier index would permit an old
session epoch to satisfy a later acquire. M9 reserves a disjoint completion-
index range for every run and rebases only the barrier index in an M9-private
copy of the frozen dispatch role. Validation, cursor reservation, metadata,
payload stores, release/acquire ordering, traces, and launch dimensions are
unchanged.

Each arm also executes a job rendezvous after both symmetric windows have been
cleared and before the timed interval begins. This prevents one rank from
writing into a peer window that the peer has not yet finished clearing. The
rendezvous and host pair barriers are identical for P and D and are excluded
from the CUDA-event makespan.

## 4. End-to-end measurement

The primary clock is a CUDA event pair on the pipeline stream:

`T0 = Router start -> T1 = final original-token output ready`

Python wall-clock time is not used for the performance claim. CUDA
`%globaltimer` timestamps record Router start, each chunk reveal publication,
Scheduler commit, gate forwarding, dispatch start/end, remote completion,
expert start/end, combine start/end, and final output readiness. The
`chunk_router_complete` and `reveal_publish` columns share the timestamp taken
immediately before the reveal queue's release publication.

Distributed makespan for one arm is the maximum CUDA-event duration across the
two ranks. Each paired difference is computed only after matching P and D from
the same configuration and pair index.

## 5. Experimental matrix and ordering

The fixed matrix contains 12 configurations:

- balanced traffic with `topk=1`;
- skewed traffic with `topk=2`;
- all-to-one-like traffic with `topk=3`;
- chunk counts `2`, `4`, `8`, and `16` for every traffic case.

Every rank uses 64 source tokens, 16 FP32 features, four experts, two experts
per rank, the same deterministic inputs, routing, top-k weights, expert
matrices, descriptors, and synchronization boundaries in both arms. No chunk
count was selected or tuned after observing performance.

Each configuration runs 5 paired warmups followed by 100 measured pairs. Pair
order alternates between `P,D` and `D,P`. The complete experiment therefore
contains `1200` measured pairs and `2400` rank-level samples per arm.

## 6. Fairness audit

The automatic fairness audit reports no divergence. P and D have identical
input, routing, top-k, expert matrices, chunk boundaries, descriptor counts,
LSA transfer counts, outputs, GPU kernels, stream assignment, warmup procedure,
and synchronization boundary. Output hashes and dispatch/combine counters are
compared for every rank-level pair.

Any mismatch would mark the performance result `INVALID`; none occurred.

## 7. Correctness gate

Every P and D output is compared both with its paired output and with the
independent frozen M8 FP32 reference using `rtol=2e-5` and `atol=2e-5`.
Maximum absolute error is `0.0` in the formal matrix.

Lost, duplicate, corruption, future-access, unrevealed-access, stale-action,
unauthorized-destination, cursor-overflow, wrong-rank, wrong-token, wrong-slot,
wrong-expert, collision, missing-return, and device-error counts are all zero.

## 8. Progressive mechanism and overlap

All `2400` Progressive rank-level trials satisfy:

`first_remote_dispatch_start < final_router_complete`

All `2400` Delayed rank-level trials satisfy:

`first_remote_dispatch_start >= final_router_complete`

There are no mechanism failures. P Router/dispatch interval overlap has median
`3314688 ns` and mean `4233980.586666667 ns`; D Router/dispatch overlap is
zero. Router/expert, dispatch/expert, and expert/combine overlap are zero in
both arms because M8 remains full-handle and the expert/combine launches remain
ordered after dispatch on the pipeline stream.

## 9. Paired performance result

The per-configuration sensitivity result is:

| Configuration | Median D-P (ms) | 95% CI (ms) | Relative reduction | Result |
|---|---:|---:|---:|---|
| balanced, top-k 1, 2 chunks | 0.000000 | [0.000000, 0.000000] | 0.0000% | INCONCLUSIVE |
| balanced, top-k 1, 4 chunks | 0.014336 | [0.014336, 0.015360] | 0.7011% | PASS |
| balanced, top-k 1, 8 chunks | 0.046080 | [0.046080, 0.047104] | 0.9823% | PASS |
| balanced, top-k 1, 16 chunks | 0.108544 | [0.107520, 0.109056] | 1.0867% | PASS |
| skewed, top-k 2, 2 chunks | 0.000000 | [0.000000, 0.000000] | 0.0000% | INCONCLUSIVE |
| skewed, top-k 2, 4 chunks | 0.010240 | [0.010240, 0.011264] | 0.5008% | PASS |
| skewed, top-k 2, 8 chunks | 0.044032 | [0.044032, 0.045056] | 0.9385% | PASS |
| skewed, top-k 2, 16 chunks | 0.107520 | [0.107520, 0.108543] | 1.0763% | PASS |
| all-to-one-like, top-k 3, 2 chunks | -0.001024 | [-0.001024, 0.000000] | -0.1376% | INCONCLUSIVE |
| all-to-one-like, top-k 3, 4 chunks | 0.007168 | [0.006144, 0.007168] | 0.3500% | PASS |
| all-to-one-like, top-k 3, 8 chunks | 0.044032 | [0.043008, 0.044032] | 0.9380% | PASS |
| all-to-one-like, top-k 3, 16 chunks | 0.106496 | [0.105472, 0.107520] | 1.0659% | PASS |

The 2-chunk sensitivity cases do not independently establish a gain, but they
do not invalidate the pre-registered pooled paired endpoint. Nine of twelve
configurations pass individually, and the pooled 95% CI is strictly positive.
No claim is based on a single run or an unpaired mean.

## 10. Contention and serialization diagnosis

The useful concurrency is limited to Router versus dispatch. The Router,
Scheduler, gate, and remote-wait control roles primarily occupy one active
thread in their 256-thread blocks, while the 256-thread dispatch role performs
metadata and payload work. Expert GEMM and combine never overlap with Router or
dispatch because the frozen full-handle boundary serializes them afterward.

The recorded dispatch interval includes time spent waiting for later commits;
its 3.31 ms median overlap is therefore not 3.31 ms of continuously active
copy or LSA work. This explains why the statistically stable E2E gain is much
smaller than the raw interval overlap. The stronger 8/16-chunk results are
consistent with more dispatch work becoming hideable behind later Router
chunks; the 2-chunk cases expose too little hideable work for a separate
positive conclusion.

No Nsight occupancy counters were collected, so the report does not invent an
achieved-occupancy number. The static launch structure and timestamps show no
Router-to-dispatch launch serialization in P, but do show dispatch-to-expert
and expert-to-combine serialization. No kernel, stream, chunk size, transport,
or occupancy tuning was performed.

## 11. CPU audit and environment

CPU per-descriptor Scheduler involvement, packing, transport submission,
return construction, and polling are all `0` in both arms. Host work is limited
to job/trial orchestration, input ownership, and post-run artifact collection;
it does not enter the measured per-descriptor data plane.

Formal validation runs on two `Tesla V100-SXM2-32GB` GPUs with CUDA `12.8`,
PyTorch `2.8.0+cu128`, NCCL `2.29.7`, and an `sm_70` binary. The benchmark uses
real single-node LSA. Runtime status remains **GIN_RUNTIME_NOT_AVAILABLE**; no
GIN benchmark or fallback claim is made.

## 12. Artifacts

The complete machine-readable result, all paired samples, 36000 per-chunk
timeline rows, fairness/correctness gates, overlap metrics, and contention
diagnosis are stored under `outputs/phase_r6/m9_e2e_perf/`.

## Stop rule

The fair P versus D end-to-end benchmark and requested diagnosis are complete.
Work stops here. Scheduler, dispatch/combine semantics, expert-ready
progressive return, chunk tuning, transport tuning, TMA/cp.async optimization,
GIN benchmarking, adaptive scheduling, and predictor/lookahead work were not
implemented.
