# R6-M6 NCCL Device API Transport Report

## Result

**NCCL LSA Transport PASS**

The validated scope ends at remote registered receive-buffer decode. The LSA
path is real NCCL Device API traffic; GIN is not claimed as a runtime PASS.

## 1. NCCL version and Device API capability

The server system/PyTorch NCCL is `2.27.3`, which is
too old for the required `ncclCommQueryProperties`. Validation therefore uses
an independently selected NCCL `2.29.7` build
(version code `22907`). Its queried
properties report `deviceApiSupport=true`,
`nLsaTeams=1`, LSA team size
`2`, `multimemSupport=false`,
and `ginType=NCCL_GIN_TYPE_NONE`.

## 2. Frozen boundary and modified surface

Router, RevealQueue, IncrementalState, FastBinder, DynamicGuard,
CommittedAction, DeviceActionQueue, GPU packing, descriptor/record layout, and
reveal/chunk policy remain frozen. M6 adds only the backend-neutral transport
contract, NCCL capability wrapper, NCCL native runtime, GIN compile surface,
formal runner/tests, reports, and results. M4/M5 sources are unchanged.

## 3. Real NCCL APIs used

Job-level host setup calls `ncclGetUniqueId`, `ncclCommInitRank`,
`ncclCommQueryProperties`, `ncclMemAlloc`, `ncclCommWindowRegister` with
`NCCL_WIN_COLL_SYMMETRIC`, and `ncclDevCommCreate`. Device LSA calls
`ncclGetPeerPointer`. Teardown calls `ncclDevCommDestroy`,
`ncclCommWindowDeregister`, `ncclMemFree`, and `ncclCommDestroy`. No NCCL
collective, MSCCL++, or host-staged fallback exists in the M6 runtime.

## 4. Symmetric window layout

Each rank owns one `69760`-byte NCCL allocation and one
collectively registered symmetric window. The frozen region size is
`34880`, descriptor stride `8720`,
peer stride `4360`, and record size `136`.
Every record preserves `9` int64 metadata fields and
`16` FP32 features. Logical/physical mapping is:
`validated identity mapping into registered buffer`.

## 5. LSA implementation

The four-block job kernel keeps the frozen router, scheduler, transport, and
remote-wait roles. For every remote physical action, the transport block packs
the existing slot, evaluates
`ncclGetPeerPointer(window, physical_dst_offset, dst_rank)`, and GPU threads
copy the exact packed bytes by load/store. It executed
`8` real peer transfers totaling
`17472` bytes, alongside
`16` frozen GPU pack calls.

## 6. LSA completion

Completion ID is the frozen descriptor ID. The sender publishes
`ncclLsaBarrierSession<ncclCoopCta>::arrive(cuda::memory_order_release)` only
after the peer stores finish. The receiver uses
`wait(cuda::memory_order_acquire)` before post-job decode. Formal counters show
`8` arrives and `8` waits. Thus
the protocol is payload visibility -> release publication -> acquire wait ->
decode, rather than assuming that issuing a store is completion.

## 7. Action to DeviceTransport/API mapping

`PhysicalTransportAction` maps one-to-one to
`put(peer, dst_offset, src_offset, bytes, completion_id)`: `peer=dst_rank`,
`dst_offset=physical_dst_offset`, `src_offset=physical_src_offset`,
`bytes=physical_bytes`, and `completion_id=descriptor_id`. Transport does not
reschedule, reread Router top-k, change rank/bytes/order, or merge/split an
action. Explicit backend values are `mscclpp`, `nccl_lsa`, and `nccl_gin`.

## 8. MSCCL++ reference equivalence

The same deterministic input and frozen M5 artifacts were used as reference.
Scheduler action divergence is `0`;
MSCCL++ reference action divergence is
`0` and reference payload
divergence is `0`. M5 reference
PASS is `True`.

## 9. CPU participation audit

Per descriptor: Python callback `0`, CPU
poll `0`, action construction
`0`, packing
`0`, transport submission
`0`, and CUDA launch
`0`. CPU work is limited to job-level
communicator/window/devComm setup, one pipeline launch per rank, completion,
and post-job evidence collection.

## 10. Correctness

Across `16`
descriptor/source/destination cases: payload divergence
`0`, lost `0`, duplicate
`0`, wrong destination
`0`, and corruption
`0`. Metadata validity is
`True` and Router destination validity is
`True`.

## 11. Legality

Future access `0`, unrevealed access
`0`, stale action `0`,
slot replay `0`, and transport errors
`0`. The two-rank clean-exit and legality gates
are both true.

## 12. Progressive evidence

`6` of `8` real remote LSA actions have
`communication_start < final_router_completion`; the remaining final action
per rank begins at/after that rank's final reveal. Therefore both
`lsa_before_final_router` and `router_communication_overlap` gates are
`True` and
`True`. Trace times are mechanism evidence,
not a performance benchmark.

## 13. GIN integration and runtime status

Only after LSA PASS, the NCCL 2.29.7 header-backed GIN surface was compiled for
sm70/sm80/sm90. `ncclDevCommRequirements_t` requests one context, per-completion
signals/counters, and `NCCL_GIN_CONNECTION_FULL`. `ncclGin::put` directly uses
the action's peer/destination/source/bytes. Remote payload visibility maps to
`ncclGin_SignalInc` plus `waitSignal`; sender source reuse independently maps
to `ncclGin_CounterInc` plus `waitCounter`.

Runtime status is **GIN_RUNTIME_NOT_AVAILABLE**: queried GIN type is
`NCCL_GIN_TYPE_NONE` and `single-node container; /dev/infiniband is absent`. Hence real
GIN puts and network bytes are both zero, as recorded in `gin_trace.csv`; no
collective or MSCCL++ path is relabeled as GIN.

## 14. Blockers and limitations

- The validation host is one node with two V100 GPUs over NV2 and no exposed
  `/dev/infiniband`; true multi-node RDMA GIN execution cannot be performed.
- LSA validation is limited to two local ranks and one LSA team.
- The scope is forward packing/transport only; expert compute and return traffic
  remain out of scope.
- One remote action per rank/descriptor gives deterministic completion mapping.
- No tuning, coalescing, chunk changes, or formal performance benchmark was done.

## Stop rule

DeviceTransport abstraction, real NCCL LSA, byte-exact correctness, legality,
progressive overlap, GIN compilation, and capability detection are complete.
Because the environment lacks real GIN capability, work stops at
`GIN_RUNTIME_NOT_AVAILABLE` as required.
