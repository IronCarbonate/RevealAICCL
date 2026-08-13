# R6-M2 Preregistration: MSCCL++ Progressive Pipeline Pilot

## Question and comparison

R6-M2 tests whether progressive revealed-only forward dispatch has more value
after replacing rank-synchronous NCCL collectives with real MSCCL++ one-sided
MemoryChannel primitives. The primary comparisons are within backend:

`Gain_NCCL = T(NCCL-D) - T(NCCL-P)`

`Gain_MSCCLPP = T(MSCCLPP-D) - T(MSCCLPP-P)`

Positive values mean progressive is faster. Absolute MSCCL++ versus NCCL speed
is not a primary conclusion.

## Frozen pipeline

The R5-P4 full-MoE primary endpoint is retained: earliest Router launch through
forward communication, unchanged full expert batches, unchanged NCCL return,
and final combine completion. Router/top-k, eight chunk profile, six early
descriptors plus the frozen `(6, 7)` checkpoint descriptor, partial-current
scheduler, fast packing, expert/return/combine, token traffic, and all legality
semantics are unchanged.

The only variables are forward backend and forward issue boundary:

- `NCCL-D`: original optimized delayed NCCL forward.
- `NCCL-P`: original optimized progressive NCCL forward.
- `MSCCLPP-D`: all unchanged descriptors submitted after final Router.
- `MSCCLPP-P`: each unchanged descriptor submitted after reveal and guard pass.

MSCCL++ uses the R6-M1 `MemoryChannelDeviceHandle::put<8>`, `signal`, and
`wait` path over CUDA IPC. It registers one preallocated buffer once. Metadata
and FP32 features are byte-packed without changing record identity, size,
destination, descriptor boundaries, or scheduler actions. Multiple descriptors
may be outstanding: puts/signals are submitted at their arm-defined boundary;
matching waits are queued only at the frozen forward completion boundary.

## Pilot population and ordering

- Seeds: `13042`, `13142`, `13242` (the R5-P4 diagnostic seeds).
- Families: balanced, skewed, all-to-one-like, zero-sized-pair, and
  multiple-progressive-shards.
- Jobs: 3 per family per seed.
- Total: 45 paired cases and 180 full-MoE arms.
- Arm order is deterministically rotated by seed/family/job to reduce order
  bias. This is a pilot, not a formal benchmark.

## Correctness gate

All four arms must have identical router digests, scheduler actions, forward
descriptor identities/counts/offsets/bytes/destinations, expert results, return
descriptors, and final combined output within the frozen numerical tolerance.
All lost, duplicate, wrong-destination, corruption, future, unrevealed, stale,
and scheduler/checker divergence counters must be zero. Every MSCCL++ case must
record real puts and positive transferred bytes. Any failure is Correctness
FAIL and VETO.

## Timing and diagnosis

Primary endpoint: cross-rank full-MoE makespan from earliest Router launch to
latest combined-output readiness. Final forward completion is secondary.

Per descriptor/rank record Router ready, guard pass, action commit, host issue,
GPU start/end, and completion. MSCCL++ additionally records put launch/kernel,
signal, and remote wait completion. Aggregate cross-rank ready, issue, and GPU
start skew; communication duration; Router interval; communication/Router GPU
overlap; kernel envelope; and ready-to-GPU-start delay.

Report paired medians, deterministic 95% bootstrap CIs (10,000 resamples),
3-seed direction, and five family medians.

## Frozen verdict

- Correctness PASS: the complete correctness gate passes.
- Mechanism PASS: median `(Gain_MSCCLPP - Gain_NCCL) > 0`, MSCCLPP-P has real
  pre-final-Router puts, and its median ready-to-GPU-start delay is below NCCL-P.
- Performance PASS: median `Gain_MSCCLPP > 0` and all 3 seed-specific
  `Gain_MSCCLPP` medians are positive.
- `NO VETO` requires Correctness PASS. Mechanism or Performance may fail without
  invalidating the implementation, but must be reported without tuning.

After four-arm correctness, paired performance, and rendezvous diagnosis, stop.
