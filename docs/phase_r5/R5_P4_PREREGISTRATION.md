# Phase R5-P4 Optimized Progressive Diagnosis — Preregistration

Frozen before generating or running any R5-P4 diagnostic workload.

## Scope

R5-P4 diagnoses why optimized progressive `E1` was slower than optimized delayed `D1` by a paired
median 3.093 ms in R5-P3. It does not optimize either arm, add a transport, or make a new deployment
performance claim.

Both arms retain the accepted R5-P3 fast data-preparation implementation. The only arm difference
remains the timing of the same seven forward count/payload descriptors:

- `E1`: progressive descriptor execution after reveal;
- `D1`: identical descriptors delayed until final Router completion.

Router/top-k, AICCL scheduler/checker, descriptor/token/byte semantics, fast packing, full-size expert,
return A2Av and combine are identical.

## Diagnostic corpus

- Fresh diagnostic seeds: `13042 / 13142 / 13242`.
- Five unchanged traffic families, three jobs per family per seed.
- 45 paired E1/D1 jobs; two ranks; deterministic counterbalanced arm order.
- Same 2×V100 and PyTorch NCCL A2Av-T0 backend.
- One already-consumed R4 seed may be used for a one-job trace smoke only.
- This corpus is diagnostic, not pilot/formal validation and is not pooled with R5-P3.

## Unified timeline and fail-closed association

Each seed/rank exports one Kineto/CUPTI CPU+CUDA trace. Explicit `record_function` ranges identify:

- each Router chunk;
- each forward delta-count exchange;
- each paired metadata/feature A2Av payload call.

NCCL kernels are associated through enclosed CPU-op External IDs and filtered by kernel name. Router
kernels are associated directly to Router ranges, with the previously audited pinned-D2H delimiter
fallback only if thread-local ranges are absent. Analysis fails closed on missing/duplicate ranges,
wrong descriptor cardinality, ambiguous stream, missing NCCL kernels, or semantic mismatch.

Host monotonic timestamps are used only with host timestamps; CUPTI timestamps are used only within
the trace. No uncalibrated CPU/CUDA timestamp subtraction is allowed.

## Per-descriptor measurements

For E1 and D1, descriptor `i` reports:

1. cross-rank descriptor-ready skew: absolute difference between rank ready host timestamps;
2. cross-rank count issue skew;
3. count issue → both-ranks host-visible complete envelope;
4. per-rank residual count wait and actual count CUDA-event duration;
5. cross-rank payload-call skew;
6. payload range start/end → actual first/last NCCL GPU kernel;
7. call-start → GPU-start, API/range-end → GPU-start, NCCL GPU active duration and envelope;
8. cross-rank NCCL GPU-start skew;
9. Router future-kernel/NCCL actual coexistence and per-chunk coexistence;
10. descriptor GPU completion relative to final Router GPU completion.

## Router/A2Av interference

Compare paired E1/D1:

- final Router GPU envelope and summed Router kernel duration;
- per-chunk Router kernel duration;
- Router host visibility latency;
- payload/count NCCL GPU duration and envelope;
- actual Router/NCCL overlap duration.

Profiler makespan is reported only to check that the sign of `D1-E1` is directionally reproduced; it
does not replace the profiler-off R5-P3 frozen result.

## Diagnostic attribution

For each paired job, sum descriptor-level phase costs into the following non-overlapping labels where
the trace permits, and retain signed E1-minus-D1 medians:

- `ready_skew`;
- `count_rendezvous/residual_wait`;
- `payload_launch/rank_start_skew`;
- `payload_gpu_execution`;
- `router_interference`;
- `other/non-additive remainder` needed to reconcile with observed makespan.

Because progressive phases overlap Router work, phase deltas are not assumed additive. The report
must show both raw signed deltas and a clearly labeled descriptive share among positive diagnosed
costs. It may not call that share a causal decomposition.

Classification is frozen as:

- **collective/rank-rendezvous dominated** if ready-skew + count-rendezvous + payload-launch account
  for at least 50% of positive diagnosed cost and are positive in all three seed medians;
- **resource-contention dominated** if Router interference + payload GPU execution account for at
  least 50% under the same seed condition;
- **both/mixed** otherwise.

If the first classification holds, the report may recommend applying for an MSCCL backend integration
phase. It must not implement or benchmark MSCCL in R5-P4.

## Correctness

- E1/D1 byte-exact forward descriptors, scheduler actions, expert/return descriptors and outputs.
- Lost/duplicate/wrong/corruption/future = 0; legality/token integrity = 100%.
- Runtime BFS/full rebuild = 0.
- No profiler result is interpreted unless correctness and trace association both pass.

## Prohibitions

No packing/count optimization, scheduler/reveal/config change, progressive expert/return/combine,
return descriptor reconstruction, transport variant, MSCCL/MSCCL++ integration, workload tuning,
production integration, pilot Gate rewrite, or formal experiment.
