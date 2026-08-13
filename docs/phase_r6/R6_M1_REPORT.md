# R6-M1 Report: Real Progressive MSCCL++ Forward Integration

## Outcome

**R6-M1 PASS.** The local progressive/revealed-only AICCL path now maps guarded
committed descriptors to real MSCCL++ GPU communication on two V100 GPUs. The
MSCCL++ path executed 8 remote puts and transferred 8,896 physical bytes, with
no NCCL call in the adapter or native runtime. The separate NCCL path was used
only as a post-run correctness reference.

Artifacts:

- `outputs/phase_r6/m1/r6_m1_results.json`: complete descriptors, actions,
  timestamps, counters, NCCL comparison, and per-rank diagnostics.
- `outputs/phase_r6/m1/mscclpp_smoke.json`: independent two-GPU primitive smoke
  test (one 64-byte bidirectional put per rank).
- `rlccl/transport/mscclpp_backend.py`: fail-closed committed-action adapter and
  registered-buffer layout.
- `extensions/r6_mscclpp/mscclpp_runtime.cu`: thin real MSCCL++ CUDA runtime.
- `scripts/run_r6_m1_mscclpp.py`: progressive two-rank correctness runner.

## Required questions

### 1. MSCCL++ version

Official MSCCL++ 0.9.0 was built from the 2026-08-13 `main` source archive
(SHA-256 `3e64fc12389bd5efd1be58062b3825069f76072ee528c6efa461e0e379867a1a`).
The codeload archive does not retain `.git`, and a subsequent `ls-remote` query
timed out, so no unverifiable commit hash is claimed; version, retrieval date,
and exact archive digest are the reproducible source identity.
The library was compiled for V100 `sm_70` with CUDA 12.8 and C++20. Python
bindings, NCCL extension, collectives extension, IB, and GDRCopy were disabled.

### 2. APIs actually used

Host setup uses `TcpBootstrap`, `Communicator`, `connect` with
`Transport::CudaIpc`, `buildSemaphore`, `registerMemory`, `sendMemory`,
`recvMemory`, the `MemoryChannel` constructor, and `deviceHandle`. Runtime CUDA
kernels call `MemoryChannelDeviceHandle::put<8>`, `signal`, and `wait`.

No XML, classic MSCCL executor, `ExecutionPlan`, or MSCCL++ static executor is
used.

### 3. Channel and physical path

`MemoryChannel` was selected because `nvidia-smi topo -m` reports `NV1` between
the two V100-SXM2 GPUs. The connection transport is CUDA IPC. GPU threads copy
directly between peer-mapped CUDA allocations over the intra-node P2P/NVLink
path; no host proxy or rank-synchronous collective is used for payload issue.

### 4. CommittedAction mapping

Each completed and revealed router chunk produces a destination layout and
packed records. `FastBinder` creates the scheduler proposal and `DynamicGuard`
must accept it before `MscclppCommittedAdapter.commit_descriptor` runs. For each
destination slot the adapter records:

`action_id, descriptor_id, src_rank, dst_rank, src_offset, dst_offset,
token_count, bytes, physical_bytes, reveal_ids`.

Remote actions are issued as `put<8>(dst_offset, src_offset, physical_bytes)` and
one `signal`. Local self-destination actions use a device-to-device tensor copy
into the same deterministic receive layout and are not counted as MSCCL++ puts.
The backend does not schedule or alter destinations.

### 5. Buffer registration and lifetime

Each rank allocates one CUDA int64 buffer before creating the native runtime.
The complete allocation is registered once with `Communicator::registerMemory`
and remains alive until all actions, waits, correctness reads, and counter
collection finish. Communicator, registered memories, semaphore, channel, and
device handle are also long-lived. There is no per-descriptor GPU allocation or
memory registration.

The first half is the send region and the second half the receive region. Each
region has fixed descriptor/peer slots. A slot contains an 8-byte token-count
header and capacity for 32 records, where one record is 64 bytes. Capacity,
rank, descriptor, count, offset, zero-payload, and replay checks fail closed.

### 6. Runtime offsets and byte counts

`src_offset` selects `(descriptor_id, destination_rank)` in the local send
region. `dst_offset` selects `(descriptor_id, source_rank)` in the remote
receive region. Payload `bytes = token_count × 64`; physical bytes additionally
include the 8-byte count header. Empty remote payloads transfer only that
non-zero header so signal/wait pairing and receiver placement remain explicit.

The native kernel uses `put<8>` because official MSCCL++ requires matching
source/destination alignment phase. This fixed an experimentally observed
one-int64 corruption from the invalid default `put<16>` on 8-byte-phase slots.

### 7. Producer-consumer synchronization

The communication stream waits for PyTorch's current packing stream before
issuing puts. A put kernel signals after its device threads finish the peer copy.
The other rank issues one wait kernel for that descriptor and synchronizes the
communication stream before decoding. This establishes packing → put → signal
→ remote wait → decode order.

### 8–9. Proof of real execution and counters

The independent smoke test transferred 64 bytes in each direction with exact
values. The formal progressive run recorded:

| Counter | Value |
|---|---:|
| `mscclpp_put_calls` | 8 |
| `mscclpp_bytes_transferred` | 8,896 |
| `mscclpp_signals` | 8 |
| `mscclpp_waits` | 8 |
| guarded adapter actions (including local) | 16 |
| payload bytes represented by actions | 16,384 |

These counters originate in the native wrapper immediately around real device
kernel launches. NCCL executed later in a separate reference loop.

### 10. Progressive evidence

Both ranks issued their first MSCCL++ action before final Router completion:

| Rank | First issue before final Router |
|---|---:|
| 0 | 5,225,310 ns |
| 1 | 7,559,217 ns |

All four descriptors per rank name exactly one progressively revealed chunk.
Runtime BFS calls and full state rebuilds were both 0.

### 11. NCCL correctness

Eight descriptor×rank comparisons passed. MSCCL++ and NCCL payload multiset
digests matched in all cases. Totals were `lost=0`, `duplicate=0`,
`wrong_destination=0`, `corruption=0`, and `digest_divergence=0`. Scheduler
actions were 32 per chunk and each corresponding `DynamicGuard` decision passed.

### 12. Future/unrevealed legality

Both ranks recorded `future_access=0`, `unrevealed_access=0`, and
`stale_action=0`. Dedicated unit tests also force all three violations and
verify fail-closed rejection before runtime issue.

### 13. Rendezvous diagnosis

Payload issue no longer requires the peer to enter a matching collective; each
rank can launch a peer put when its local guarded action is ready. There is still
initialization rendezvous in `TcpBootstrap`/connection setup and producer-
consumer synchronization through signal/wait. The validation runner waits once
per descriptor so it does not yet exploit arbitrary rank skew or multiple
outstanding descriptors; this is a limitation, not a collective payload path.

### 14. Blockers and limitations

No R6-M1 blocker remains. Current limitations are the frozen two-rank,
single-node forward-only scope; fixed maximum descriptor/token capacity; one
outstanding remote descriptor per rank; an 8-byte metadata header per
source/descriptor; no progressive expert, return, combine, multi-node, or formal
performance result. The build required isolated CMake ≥3.25, `libnuma-dev`, and
system `nlohmann-json3-dev`; these are build prerequisites, not runtime fallbacks.

## Stop rule

Installation/build, real primitive smoke test, adapter integration, two-rank
forward correctness, legality, and progressive mechanism verification are
complete. Work stops here; no performance optimization or formal benchmark was
started.
