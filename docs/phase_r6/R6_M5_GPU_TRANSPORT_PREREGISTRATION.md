# R6-M5 GPU-Driven Packing and Transport Preregistration

## Scope and frozen boundary

R6-M5 consumes the already-passed R6-M4 `DeviceActionQueue` and implements:

`DeviceActionQueue -> GPU packing -> persistent GPU transport consumer -> MSCCL++ MemoryChannel -> remote registered buffer`

The M4 scheduler schema and semantics are frozen. R6-M5 may not change action
order, destination, token count, route, reveal policy, or descriptor boundaries.
The work stops after remote registered-buffer decode. NCCL Device API, LSA,
GIN, NVSHMEM, DeepEP, multi-node work, Router changes, Scheduler changes,
performance tuning, and formal benchmarking are out of scope.

## Logical and physical offsets

M4 `CommittedAction.src_offset/dst_offset` remain logical offsets and retain the
`LOGICAL_OFFSETS` flag. R6-M5 introduces `PhysicalTransportAction` and validates
an explicit mapping:

- logical source: `(descriptor_id, destination_rank)` send slot;
- logical destination: `(descriptor_id, source_rank)` remote receive slot;
- physical source/destination: byte offsets in the one registered job buffer.

The current frozen layout makes this a validated identity mapping. Transport
must reject rank, descriptor, alignment, byte-count, capacity, offset, or slot
replay violations before packing or put.

## Registered buffer layout

The formal layout is two ranks, four descriptors, 32 tokens per peer slot,
nine int64 metadata fields, and 16 FP32 feature values. One record is therefore
`9*8 + 16*4 = 136` bytes. Each peer slot is:

`int64 token_count | packed records[32]`

The first half of the allocation is the send region and the second half the
receive region. Buffer allocation, MSCCL++ registration, remote-memory exchange,
semaphore, `MemoryChannel`, and device handle creation happen once during job
initialization. No runtime allocation or registration is permitted.

## GPU packing

The persistent transport block reads exactly one frozen `CommittedAction`,
locates its corresponding revealed assignment range, and directly writes the
count header plus byte-exact metadata/FP32 feature records into the registered
send slot. It preserves assignment order within each destination. There is no
D2H, Python byte packing, CPU memcpy/staging, or per-descriptor CUDA launch.

## GPU transport

One persistent transport block consumes the device action ring. For remote
actions its threads call:

`MemoryChannelDeviceHandle::put<8>(physical_dst_offset, physical_src_offset, physical_bytes, tid, blockDim.x)`

After a block barrier, lane zero calls `signal()`. A separate job-lifetime
device block calls `wait(-1)` for the frozen remote action sequence. Local
self-actions use a direct GPU copy into the deterministic local receive slot.

No NCCL code or fallback is present in the M5 native runtime.

## CPU participation

Allowed CPU work: static plan compilation, buffer/channel setup, one job-level
four-block kernel launch (Router publish, frozen scheduler, transport, remote
wait), job completion synchronization, and post-job reference/debug collection.

The required per-descriptor counts are all zero:

- Python callback;
- CPU poll;
- CPU scheduler/action construction;
- CPU packing/staging;
- CPU transport submission;
- CPU CUDA launch.

## Correctness protocol

Two V100 ranks each route four chunks of 32 tokens. The unchanged Router top-k
uses a deterministic four-expert input so every chunk has both local and remote
destinations. The GPU pipeline packs nine-field identity metadata and 16 FP32
values. After the unified completion boundary, a CPU debug oracle compares:

- all 12 M4 action fields and order;
- token identity and destination;
- all metadata bytes;
- all FP32 feature bytes;
- exact record order in every descriptor/source/destination slot.

Required totals: action divergence 0, payload divergence 0, lost 0, duplicate
0, wrong destination 0, corruption 0, future/unrevealed/stale 0, and transport
errors 0.

## Timing

Device global-timer stamps:

- T0 Router reveal;
- T1 Scheduler commit;
- T2 ActionQueue consume;
- T3 pack start;
- T4 pack end;
- T5 MSCCL++ put start;
- T6 put/signal end;
- T7 remote wait completion;
- T8 final Router completion.

The mechanism gate requires at least one formal remote put to start before T8
and have a non-empty interval overlapping the period before final Router
completion. Timing is diagnostic only.

## PASS gate

Declare **GPU-Driven Packing and Transport PASS** only if:

1. M4 CPU/GPU action divergence remains zero.
2. CPU per-descriptor packing and transport involvement are zero.
3. Real MSCCL++ put calls and bytes are positive, with no NCCL fallback.
4. Token/metadata/feature/destination correctness is exact.
5. Future, unrevealed, stale, replay, bounds, and transport-error counters are zero.
6. At least one real put starts before final Router completion.
7. The native sources compile for sm_70, sm_80, and sm_90 and execute on V100.

## Artifacts

- `outputs/phase_r6/m5_gpu_transport/results.json`
- `outputs/phase_r6/m5_gpu_transport/action_trace.csv`
- `outputs/phase_r6/m5_gpu_transport/packing_trace.csv`
- `outputs/phase_r6/m5_gpu_transport/transport_trace.csv`
- `outputs/phase_r6/m5_gpu_transport/correctness.json`
- `docs/phase_r6/R6_M5_GPU_TRANSPORT_REPORT.md`
