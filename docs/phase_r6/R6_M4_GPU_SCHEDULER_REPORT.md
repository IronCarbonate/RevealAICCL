# R6-M4 GPU Scheduler Report

## Result

**GPU Scheduler PASS**

This claim stops at `DeviceActionQueue`. It is not a claim of GPU-driven
communication; packing and transport are explicitly outside R6-M4.

## 1. Modified and added files

The existing CPU scheduler and transport implementation were not modified.
R6-M4 adds:

- `rlccl/scheduler/common/scheduler_ir.h`: portable POD IR and error codes;
- `rlccl/scheduler/common/scheduler_schema.py`: host schema/configuration;
- `rlccl/scheduler/common/compiled_plan.py`: contiguous plan serialization,
  validation, and checksum;
- `rlccl/scheduler/cpu/reference.py`: deterministic CPU shadow adapter;
- `rlccl/scheduler/cuda/gpu_scheduler.cuh`: device queue/state definitions;
- `rlccl/scheduler/cuda/gpu_scheduler.cu`: producer, persistent scheduler,
  FastBinder, DynamicGuard, action publication, instrumentation;
- `rlccl/scheduler/cuda/gpu_scheduler_bindings.cpp`: job-level binding;
- `rlccl/scheduler/cuda/gpu_scheduler_backend.py`: initialization/debug facade;
- `scripts/run_r6_m4_gpu_scheduler.py`: formal gates and artifacts;
- `tests/test_r6_gpu_scheduler_common.py`: portable CPU contracts.

## 2. New structures

The common boundary is `RevealRecord -> CompiledPlanBlob -> CommittedAction`.
All common records use fixed-width integers and logical byte offsets. CUDA
queues/state are separate and contain the only device pointers.

## 3. Logic moved to GPU

Runtime reveal consumption, flattened pair demand updates, deterministic route
lookup, token-count/offset binding, every dynamic legality check, action ID
allocation, committed-count updates, and CommittedAction queue publication run
inside `gpu_scheduler_progress_kernel`.

## 4. Logic retained on CPU

Topology/static-plan compilation, blob serialization, job initialization, the
single persistent-kernel launch, job completion, CPU shadow validation, and
debug artifact collection remain on CPU. Runtime graph search remains absent.

## 5. StaticPlan upload and checksum

The host serializes header, route templates, rank-pair lookup, capacities, and
legality flags into one blob and uploads it once. CPU checksum:
`0xd19e2f785e8b7be3`. GPU uploaded/recomputed checksum:
`0xd19e2f785e8b7be3`. Match: `True`.

## 6. RevealQueue

The fixed-capacity device ring uses monotonic head/tail counters. The producer
writes the entire record before a device-scope release store to tail. The
single consumer performs an acquire load of tail before reading the record.
The producer consumes Router-resident destination arrays on the current Router
stream; EventBridge and Python do not participate in scheduler correctness.

## 7. IncrementalState

`revealed_count`, `committed_count`, `next_send_offset`, and
`next_recv_offset` are flat `[src * world_size + dst]` arrays. Descriptor
epochs, last reveal epoch, and next action ID are GPU resident. The frozen
single-consumer model owns all state writes, avoiding unnecessary atomics.

## 8. FastBinder

FastBinder counts revealed assignments by destination, performs a single
`rank_pair_to_route[src * world_size + dst]` lookup, assigns deterministic
destination order, computes logical registered-region offsets, and produces
candidate actions. Runtime BFS/DFS/topology planning is zero.

## 9. DynamicGuard

The GPU guard is fail closed for revealed/future demand, duplicate descriptor,
stale epoch, rank/count validity, logical offset and byte bounds, route/template
identity, descriptor range, assignment range, and queue capacity. A reveal is
preflighted as a batch; no action from the failing reveal is published.

- `bytes_overflow`: PASS (CPU=12, GPU=12, expected=12)
- `duplicate_action`: PASS (CPU=5, GPU=5, expected=5)
- `future_access`: PASS (CPU=9, GPU=9, expected=9)
- `invalid_destination`: PASS (CPU=7, GPU=7, expected=7)
- `invalid_rank`: PASS (CPU=6, GPU=6, expected=6)
- `invalid_route`: PASS (CPU=13, GPU=13, expected=13)
- `offset_overflow`: PASS (CPU=11, GPU=11, expected=11)
- `overflow`: PASS (CPU=3, GPU=3, expected=3)
- `repeated_descriptor`: PASS (CPU=5, GPU=5, expected=5)
- `stale_action`: PASS (CPU=4, GPU=4, expected=4)
- `stale_reveal`: PASS (CPU=4, GPU=4, expected=4)
- `unrevealed_access`: PASS (CPU=10, GPU=10, expected=10)
- `zero_token_action`: PASS (CPU=8, GPU=8, expected=8)

## 10. ActionQueue

After every candidate for a reveal passes, actions enter a fixed-capacity
device ring in deterministic destination order. Each complete action write is
followed by a device-scope release publication of tail. Transport may consume
only this common `CommittedAction` in a later phase.

## 11. Persistent scheduler

Launch geometry is grid `1`, block
`32`. It uses one strict consumer lane in one warp.
Measured attributes: `55` registers/thread,
`0` static shared-memory bytes, and
`5120` local-memory bytes. It does not require
cooperative launch or generation-specific scheduling primitives.

## 12. CPU/GPU action equivalence

Aggregate divergence: `0`.

- `all_to_one_like`: PASS, CPU/GPU actions 1/1
- `balanced`: PASS, CPU/GPU actions 8/8
- `maximum_slot_capacity`: PASS, CPU/GPU actions 1/1
- `multiple_progressive_shards`: PASS, CPU/GPU actions 7/7
- `router_resident_top_k`: PASS, CPU/GPU actions 16/16
- `skewed`: PASS, CPU/GPU actions 2/2
- `top_k_gt_1`: PASS, CPU/GPU actions 3/3
- `zero_demand`: PASS, CPU/GPU actions 0/0
- `zero_sized_pair`: PASS, CPU/GPU actions 1/1

Every comparison covers count/order and all 12 common action fields.

## 13. Legality result

All injected cases fail closed: `True`. Detailed
codes are stored in `legality_tests.json`.

## 14. CPU per-descriptor involvement

Python callback: 0; CPU poll: 0; CPU scheduler: 0; CPU action construction: 0;
per-descriptor scheduler kernel launch: 0. CPU participation occurs only at the
job boundaries and in optional post-job shadow/debug collection.

## 15. Router-to-commit timing

Observed `reveal_to_commit`: min=8.192 us, max=2132.992 us, mean=264.448 us. Per-record T0-T5 values and derived
mechanism timings are in the two timeline CSVs. These measurements are not a
performance benchmark.

## 16. Commit before final Router completion

Gate result: `True`. The multi-shard producer
continues publishing later Router completions while the persistent scheduler
commits earlier revealed shards.

## 17. Volta/Ampere/Hopper compatibility

The extension compiled target list is `7.0;8.0;9.0` and
the formal execution target is V100/sm_70. The kernel uses one block, one warp,
CUDA device-scope atomics, and no Hopper-only feature.

## 18. Current limitations

- R6-M4 validation stops at DeviceActionQueue; packing and transport are not included.
- The formal producer is a small device publish kernel consuming Router-resident arrays; fusion into the frozen Router kernel is deferred.
- One persistent scheduler consumer and one source rank per backend instance are intentionally frozen for correctness.
- Timing is mechanism instrumentation, not a performance benchmark.

## 19. Next-phase transport boundary

R6-M4 ends at `DeviceActionQueue`. The next phase may add GPU packing and a
device transport consumer for NVIDIA, while a sibling Ascend branch may consume
the same host schema with an Ascend-specific queue/state/kernel. Scheduler IR
must remain unaware of NCCL, MSCCL++, NVSHMEM, DeepEP, or device pointers.

## Artifact index

- `outputs/phase_r6/m4_gpu_scheduler/results.json`
- `outputs/phase_r6/m4_gpu_scheduler/action_comparison.csv`
- `outputs/phase_r6/m4_gpu_scheduler/reveal_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/scheduler_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/legality_tests.json`
