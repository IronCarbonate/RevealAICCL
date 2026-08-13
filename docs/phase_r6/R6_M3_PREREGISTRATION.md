# R6-M3 Preregistration: MSCCL++ Post-Issue GPU-Start Diagnosis

## Frozen question

R6-M3 asks only why an already host-issued MSCCL++ progressive put starts on
the GPU after the remaining Router window. It is diagnosis, not optimization,
parameter tuning, or a performance comparison.

All R6-M2 Router/top-k, eight chunks and sizes, reveal profile,
`partial_current_only`, scheduler/checker, descriptor boundaries, CPU packing
layout, expert MLP, NCCL return/combine, MSCCL++ MemoryChannel `put<8>`,
registered-buffer layout, traffic families, and seeds remain frozen. Stream
priorities, kernel configuration, message size, and descriptor granularity are
not changed.

## Diagnostic corpus

- Primary path: `MSCCLPP-P` only.
- Seeds: 13042, 13142, 13242.
- Families: the five frozen R6-M2 families.
- Jobs: job 0 only, giving 15 primary diagnostic cases.
- Controls: balanced/job 0 only for each seed (three Router-absent and three
  dependency-resolved cases). Controls are causal probes and never enter a
  performance verdict.
- One separately profiled representative reproduction: seed 13042,
  balanced/job 0, normal MSCCLPP-P. Its timing is trace evidence only and is
  excluded from primary latency distributions.

The corpus is sufficient only if it reproduces host issue before final Router
GPU end while put GPU start is after final Router GPU end. Otherwise diagnosis
is INCOMPLETE and no root cause is claimed.

## Frozen timeline

Every descriptor records T0--T21 on host `steady_clock` where a real timestamp
exists. A missing independent operation is written as `N/A`, never estimated:

- T0 Router chunk host launch; T1/T2 Router CUDA start/end; T3 EventBridge
  publish; T4 incremental scheduler state updated; T5 FastBinder returns; T6
  DynamicGuard passes; T7 transport CommittedAction creation returns.
- T8 frozen CPU byte packing begins. T9/T10 registered-buffer H2D staging CUDA
  start/end; T11 the descriptor-specific staging event is recorded; T12 the
  communication stream wait on that exact event is enqueued.
- T13 native wrapper entry; T14 CUDA kernel launch call; T15 launch call
  returns; T16/T17 put CUDA start/end. Signal is internal to the same
  put-and-signal kernel, so T18 is `N/A` unless a separate event is observed.
- T19/T20 remote wait CUDA start/end; T21 final Router CUDA end.

The required decomposition uses real ordering:

`reveal_to_commit = T7 - T2`

`packing_delay = T10 - T7`

`post_pack_enqueue_delay = T15 - T10`

`GPU_queue_delay = T16 - T15`

`ready_to_gpu_start = T16 - T2`

Because frozen CPU descriptor packing begins before the transport creates its
CommittedActions, its separate T8-to-T7 contribution is also reported; no
timestamp is reordered to make the requested labels look sequential.

For each latency report p50, p95, p99, and max over remote descriptors.

## Dependency and stream audit

Instrumentation replaces the opaque `wait_stream(current_stream)` bookkeeping
with its semantic equivalent: one newly created event recorded after the same
descriptor's registered-buffer staging and one `comm_stream.wait_event(E_i)`.
The event is not reused. Audit counters must be:

`wrong_event_dependency = 0`

`future_pack_dependency = 0`

`event_reuse_hazard = 0`

Each put records stream IDs/priorities, sequence number, preceding comm-stream
operation and dependency. No new stream or priority change is allowed.

## Causal controls

- Router-absent: the producer launches the next Router chunk only after the
  current progressive descriptor has been submitted. It uses identical Router
  kernels, descriptors, packing, and real puts; it removes only concurrent
  future Router work from the put queue window.
- Dependency-resolved: normal future Router submission is retained, but the
  exact descriptor staging event is completed before the real put is enqueued.
  This distinguishes unresolved staging dependency from subsequent GPU
  scheduling. It is diagnostic-only and its explicit event synchronization is
  recorded.

## Attribution rule and verdict

Known Router, staging, previous-put, wait, expert, and profiler kernel intervals
are intersected with each T15--T16 queue interval. Coverage is assigned to the
requested categories A--G without double counting; uncovered time remains
Other/idle rather than being guessed. Kineto supplies the representative raw
kernel timeline and kernel names.

`Diagnosis COMPLETE` requires a reproduced zero-overlap pattern, full latency
decomposition, dependency/stream audit, and controls that distinguish the
dominant category. `Root Cause IDENTIFIED` requires quantitative normal/control
and timeline evidence. Correctness and legality retain the R6-M2 gates. Any
correctness/legality failure or trace insufficient for the claimed cause is a
VETO. Stop immediately after diagnosis; do not implement a candidate fix.
