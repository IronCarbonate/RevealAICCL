# R6-M4 GPU Scheduler Preregistration

## Scope and stop rule

R6-M4 implements only this path:

`GPU Router output -> DeviceRevealQueue -> DeviceIncrementalState -> GPU FastBinder -> GPU DynamicGuard -> DeviceActionQueue`

The implementation stops at `DeviceActionQueue`. It does not change feature
packing, registered-buffer staging, MSCCL++, NCCL Device API/LSA/GIN, NVSHMEM,
DeepEP, multi-node transport, Router top-k policy, chunk policy, or stream
priority. A successful result may be called **GPU Scheduler PASS**, never
**GPU-driven communication PASS**.

## Frozen semantic oracle

The existing CPU scheduler in `rlccl/scheduling/compiled_event_driven.py` and
the R6 committed-action adapter remain untouched. R6-M4 adds a shadow adapter
over their frozen descriptor/action boundary. CUDA output must match the shadow
action-by-action in deterministic destination order.

Compared fields are action count/order, `action_id`, `descriptor_id`,
`chunk_id`, `reveal_epoch`, `src_rank`, `dst_rank`, logical source/destination
offsets, `token_count`, payload `bytes`, `route_id`, and flags.

## Architecture boundary

The portable common layer contains only fixed-width integers and logical
offsets:

- `RevealRecord`
- `CompiledRouteTemplate`
- `CompiledPlanBlob`
- `CommittedAction`
- `SchedulerConfig`
- scheduler counters/error codes

It contains no CUDA pointer, `cudaStream_t`, CUDA event, NCCL communicator,
MSCCL++ handle, Python object, Torch tensor object, or STL container embedded in
the uploaded blob. CUDA-specific queues, state, atomics, and kernels are under
`rlccl/scheduler/cuda/`. A future Ascend backend may reuse the common schema.

## Frozen execution model

- one persistent scheduler block per GPU;
- one warp (`block_size=32`), with one strict consumer lane in R6-M4;
- one device producer on the Router-compatible current CUDA stream;
- release publication of queue tail after the complete record/action write;
- acquire observation of tail before the consumer reads a record/action;
- no cooperative launch, TMA, cluster launch, or SM90-only primitive;
- fat-binary compilation target list: sm_70, sm_80, sm_90;
- no CPU scheduler, action construction, polling, callback, or scheduler-kernel
  launch per descriptor.

CPU participation is permitted only for initialization/static compilation,
the single job-level launch, job completion, CPU shadow validation, and debug
artifact collection.

## Compiled plan gate

The CPU compiler serializes one contiguous pointer-free blob containing a
header, route templates, flattened rank-pair lookup, capacity table, and
legality flags. The header contains all byte offsets and capacities. FNV-1a-64
is computed with the checksum field treated as zero. The persistent kernel
recomputes the uploaded checksum before consuming a reveal.

Required result: CPU plan checksum equals the GPU-uploaded checksum. Runtime
BFS, topology search, and full plan rebuild counts must all be zero.

## Correctness matrix

Normal cases:

- balanced;
- skewed;
- all-to-one-like;
- zero-sized pair;
- multiple progressive shards;
- top-k greater than one;
- zero demand;
- maximum peer-slot capacity;
- Router-resident top-k output without D2H before scheduling.

Fault injection cases (all must reject without publishing any candidate from
the failing reveal):

- future demand;
- unrevealed demand;
- stale action epoch;
- duplicate action/descriptor;
- action queue overflow;
- zero-token candidate;
- candidate byte-count overflow;
- logical offset overflow;
- invalid source rank;
- invalid destination rank;
- invalid route;
- stale reveal;
- repeated descriptor.

## Timing instrumentation

The device records:

- T0 Router chunk GPU complete;
- T1 RevealRecord published;
- T2 reveal consumed;
- T3 FastBinder complete;
- T4 DynamicGuard complete;
- T5 final CommittedAction for the reveal published.

Derived mechanism timings are `router_to_reveal`, `reveal_queue_wait`,
`binder_latency`, `guard_latency`, and `reveal_to_commit`. This is not a
performance benchmark. The progressive gate requires at least one T5 before
the final T0 in the multi-shard run.

## PASS gate

Declare **GPU Scheduler PASS** only when all hold:

1. CPU/GPU action divergence is zero for every normal case.
2. All injected illegal cases fail closed.
3. Runtime BFS and full plan rebuild counts are zero.
4. IncrementalState, FastBinder, DynamicGuard, and action generation execute in
   the persistent CUDA kernel.
5. CPU per-descriptor scheduler involvement is zero.
6. CPU and GPU plan checksums match.
7. A committed action is published before the final Router completion.
8. The same sources compile into sm_70, sm_80, and sm_90 images and execute on
   the available V100 (sm_70).

## Artifacts

The formal runner writes:

- `outputs/phase_r6/m4_gpu_scheduler/results.json`
- `outputs/phase_r6/m4_gpu_scheduler/action_comparison.csv`
- `outputs/phase_r6/m4_gpu_scheduler/reveal_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/scheduler_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/legality_tests.json`

The report must cite these files and must preserve the stop rule.
