# R6-M6 NCCL Device API Transport Preregistration

## Frozen boundary

R6-M6 does not modify Router, RevealQueue, IncrementalState, FastBinder,
DynamicGuard, CommittedAction, DeviceActionQueue, GPU packing, descriptor or
record layout, or reveal/chunk policy. It only inserts a backend-neutral
`DeviceTransport` below the existing physical action mapping.

Supported explicit configuration values are `mscclpp`, `nccl_lsa`, and
`nccl_gin`. Scheduler and packer inputs contain no backend field.

## Source of truth and minimum NCCL

Compilation and capability decisions use the NCCL headers and library selected
on the validation server. R6-M6 requires NCCL 2.29 or newer because the task
mandates `ncclCommQueryProperties`, which was introduced in NCCL 2.29. The
runtime must call it with `NCCL_COMM_PROPERTIES_INITIALIZER` and require
`deviceApiSupport` before creating a device communicator.

No API name is inferred from an older NCCL installation. A version or symbol
mismatch is reported as `NCCL_DEVICE_API_NOT_AVAILABLE` and never falls back to
a collective or MSCCL++ while claiming NCCL LSA/GIN.

## Symmetric allocation and frozen layout

Each rank allocates one equal-sized job buffer using `ncclMemAlloc` and
collectively registers it with:

`ncclCommWindowRegister(comm, buffer, bytes, &window, NCCL_WIN_COLL_SYMMETRIC)`

The first two regions retain the exact R6-M5 send/receive slot layout and byte
payload. Transport-only completion storage may follow those regions; it does
not alter a descriptor slot, metadata field, FP32 feature, or action offset.
Allocation, registration, communicator creation, and destruction are job-level.

## DeviceTransport contract

Every request is the direct projection of one validated
`PhysicalTransportAction`:

`put(peer, dst_offset, src_offset, bytes, completion_id)`

plus `test_completion(completion_id)` and
`wait_completion(completion_id)`. Transport may validate but may not change,
merge, split, reorder, or reschedule an action.

## NCCL LSA backend

Host setup creates a normal two-rank communicator, queries properties, creates
an `ncclDevComm` from initialized requirements, and registers the symmetric
window. The device backend obtains the exact peer address with
`ncclGetPeerPointer(window, dst_offset, peer)` and copies the already-packed
bytes from the local symmetric source slot.

Completion uses the installed NCCL LSA synchronization API and must establish:

`payload store -> release publication -> remote acquire wait/test -> decode`

The chosen API and memory orders are recorded from the server header in the
formal report. No NCCL collective or rank-synchronous all-to-all is a transport
substitute.

## GIN rule

GIN work starts only after LSA correctness and progressive gates pass. The
runtime requests GIN resources only when `ncclCommQueryProperties` reports a
non-`NONE` GIN type and the environment provides a real multi-node NIC/RDMA
path. Otherwise it compiles the integration surface, records capability, emits
`GIN_RUNTIME_NOT_AVAILABLE`, and stops without a fake PASS.

## CPU audit

Allowed CPU work is communicator/window/devComm setup, one job-level persistent
pipeline launch, completion synchronization, and post-job debug collection.
Python callbacks, CPU polling, packing, action construction, transport
submission, and CUDA launches per descriptor must all be zero.

## Formal gates

LSA PASS requires frozen action equivalence, positive real LSA transfers and
bytes, byte-exact remote payload, zero loss/duplication/wrong destination/
corruption, zero future/unrevealed/stale/replay errors, zero per-descriptor CPU
transport work, and at least one LSA issue before final Router completion.

Required outputs are `results.json`, `lsa_trace.csv`, `gin_trace.csv`,
`correctness.json`, `capability.json`, and the final M6 report.
