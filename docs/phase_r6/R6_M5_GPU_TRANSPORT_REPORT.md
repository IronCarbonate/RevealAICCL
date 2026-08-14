# R6-M5 GPU-Driven Packing and Transport Report

## Result

**GPU-Driven Packing and Transport PASS**

The claim ends at remote registered-buffer decode. It does not include expert
compute, return traffic, NCCL Device API, multi-node work, or a performance
benchmark.

## 1. Modified files

R6-M4 Scheduler files were not modified. R6-M5 adds the CUDA packing/layout
layer under `rlccl/transport/cuda/`, the native runtime in
`extensions/r6_m5_gpu_transport/gpu_transport_runtime.cu`, the job-level ctypes
owner, formal runner, tests, protocol, report generator, and result artifacts.

## 2. Packing buffer layout

The registered allocation is `69760` bytes. Each half is a
`34880`-byte send/receive region; descriptor stride is
`8720` and peer stride is `4360`.
Each `136`-byte record preserves nine int64 metadata fields
and 16 FP32 feature values, preceded by one 8-byte count header per slot.

## 3. Logical to physical offsets

M4 logical offsets remain unchanged. M5 creates `PhysicalTransportAction` only
after checking the unique descriptor/peer slot, exact payload byte count,
8-byte phase, send/receive region bounds, and replay bitmap. Mapping used:
`validated identity mapping into registered buffer`.

## 4. MSCCL++ memory registration

Each rank allocates its complete buffer before runtime creation. The native
runtime registers it once using CUDA IPC, exchanges remote memory once, and
keeps the semaphore, MemoryChannel, and device handle alive for the entire job.
There is no per-descriptor allocation or registration.

## 5. Persistent transport consumer

One 256-thread block directly pops `DeviceActionQueue`, validates the action,
packs metadata/features into the registered send slot, and performs either a
remote MemoryChannel put or a local GPU copy. The consumer exits only after the
frozen scheduler publishes its job-done flag and the queue is empty.

## 6. put/signal/wait

Remote actions call device-side `put<8>(dst, src, physical_bytes, tid, 256)`.
After a block barrier lane zero calls `signal()`. A separate job-lifetime block
calls `wait(-1)` and records remote completion. Put calls:
`8`; transferred bytes:
`17472`; signals:
`8`; waits: `8`.

## 7. CPU participation audit

Per descriptor: Python callback 0, CPU poll 0, CPU packing 0, CPU action
construction 0, CPU transport submission 0, and CPU CUDA launch 0. CPU is used
only for initialization, job-level launches/completion, and post-job debug
collection.

## 8. Correctness

Action divergence: `0`. Payload divergence:
`0`. Lost `0`, duplicate
`0`, wrong destination
`0`, corruption `0`.
Metadata and Router-derived destination validity are
`True` and `True`.

## 9. Legality

Future/unrevealed/stale counters are zero:
`True`. Transport errors are zero:
`True`. Slot replay count:
`0`.

## 10. Real MSCCL++ counters

Scheduler actions `16`, consumed transport actions
`16`, GPU pack calls `16`,
real puts `8`, bytes
`17472`. No NCCL symbol or fallback exists in
the M5 runtime.

## 11. Reveal-to-put timing

Commit-to-pack: min=1.024, max=20.480, mean=9.728 us.
Pack latency: min=11.264, max=12.288, mean=12.032 us.
Pack-to-put: min=0.000, max=0.000, mean=0.000 us.
Reveal-to-put: min=25.600, max=43.008, mean=33.408 us.
These are mechanism timings, not a performance benchmark.

## 12. Put before final Router

Gate: `True`. At least one formal remote action
has T5 < T8, as required by the progressive communication gate.

## 13. Router/communication overlap

Gate: `True`. Real MemoryChannel put
intervals occur while later Router reveal work remains outstanding.

## 14. Limitations

- R6-M5 is frozen to two local GPUs and CUDA IPC MemoryChannel.
- The formal path covers forward packing and transport only; expert compute and return are out of scope.
- One remote action per rank/descriptor is required by the frozen test workload for deterministic wait accounting.
- Router reveal pacing is mechanism instrumentation, not a performance benchmark.

## Stop rule

The device action queue, GPU packing, real MSCCL++ device put/signal/wait, and
remote byte-exact correctness gate are complete. Work stops here; no LSA, GIN,
NVSHMEM, DeepEP, multi-node implementation, tuning, or benchmark is started.
