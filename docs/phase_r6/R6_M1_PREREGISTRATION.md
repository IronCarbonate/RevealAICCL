# R6-M1 Preregistration: Progressive AICCL × MSCCL++

## Scope and frozen semantics

R6-M1 is restricted to two Tesla V100 GPUs, world size 2, and progressive
forward communication. Router top-k, partial/revealed-only semantics,
`StaticPlanCompiler`, `IncrementalState`, `FastBinder`, `DynamicGuard`, and the
fail-closed legality boundary remain unchanged. No expert compute, return path,
multi-node work, transport sweep, tuning, or performance pass gate is included.

The MSCCL++ adapter may consume only a descriptor whose corresponding scheduler
proposal has passed `DynamicGuard`. Future, incomplete, unrevealed, duplicate,
or stale descriptors must fail closed before packing or GPU issue. No classic
MSCCL XML, `msccl.init`, `ExecutionPlan`, executor, or NCCL fallback may be
present in the MSCCL++ execution path.

## Frozen implementation

- Official MSCCL++ version: 0.9.0 source archive from the `main` branch on
  2026-08-13; archive SHA-256
  `3e64fc12389bd5efd1be58062b3825069f76072ee528c6efa461e0e379867a1a`.
- GPU target: `sm_70`; CUDA 12.8; C++20.
- Channel: `mscclpp::MemoryChannel` over `mscclpp::Transport::CudaIpc` because
  the two V100 GPUs have a direct NVLink (`NV1`) and CUDA peer access.
- Setup APIs: `TcpBootstrap`, `Communicator`, `connect`, `buildSemaphore`,
  `registerMemory`, `sendMemory`, `recvMemory`, `MemoryChannel::deviceHandle`.
- Runtime device APIs: `MemoryChannelDeviceHandle::put<8>`, `signal`, `wait`.
- Buffers: one preallocated CUDA int64 buffer per rank, registered once. Its
  send and receive regions are disjoint. Each descriptor/peer slot contains an
  int64 count header followed by fixed 8×int64 token records.
- A committed remote action maps directly to
  `put<8>(dst_offset, src_offset, physical_bytes)` followed by `signal`; the
  receiver issues `wait` and decodes the count header at the deterministic
  `(descriptor_id, source_rank)` receive slot.
- The `8`-byte alignment is intentional: count headers make slots 8-byte
  aligned and the official primitive requires source/destination addresses to
  have identical phase modulo its selected alignment.

## Frozen correctness protocol

Four progressive router chunks of 32 tokens per rank are executed. Each chunk
is completed, revealed, incrementally scheduled, checked, packed, and issued
before the next chunk. The same packed records are then sent through the
reference `torch.distributed.all_to_all_single` NCCL backend.

The pass gate requires all of the following:

- real MSCCL++ put calls > 0 and transferred bytes > 0;
- MSCCL++ and NCCL received payload multisets are identical for every
  descriptor and rank;
- lost, duplicate, wrong-destination, corruption, and digest divergence are 0;
- future, unrevealed, and stale adapter counters are 0;
- at least one MSCCL++ issue per rank occurs before final Router completion;
- both ranks exit cleanly.

Timing is diagnostic only. Formal performance benchmarking is explicitly out of
scope, and completion of this gate triggers the stop rule.
