# R6-M7 DeepEP-style GPU Data Plane Report

## Result

**DeepEP-style GPU Dispatch PASS**

The validated scope ends at expert-contiguous `recv_x`, paired
`recv_src_metadata`, and `ProgressiveEPHandle`. Combine, expert-progressive
scheduling, ElasticBuffer integration, and performance/chunk tuning are not
part of M7.

## 1. Frozen semantics and new boundary

Router reveal/chunk policy, `RevealRecord`, incremental revealed/committed
state, revealed-only legality, FastBinder/static route decisions, stale/future/
unrevealed checks, `StaticPlanCompiler`, and the M4/M5/M6 reference path remain
unchanged. The M7 fast path replaces action-centric execution with:

`RevealRecord -> DescriptorCommit -> CommitPeerPlan -> DeviceCommitQueue -> fused dispatch`

The old `CommittedAction` remains a shadow output only. It is never consumed by
the M7 data plane.

## 2. DescriptorCommit and scheduler publication

The GPU scheduler consumes one revealed descriptor, scans its assignment range
once to count destinations, performs the frozen rank-pair route lookup and
legality checks, writes `CommitPeerPlan[descriptor][dst]`, and publishes exactly
one `DescriptorCommit`. The fixed-capacity `DeviceCommitQueue` protocol is
payload write followed by release-store of `tail`; the persistent consumer
acquire-loads `tail` before reading the commit.

Formal execution produced `8` commits for `8` rank/descriptors and `16` shadow
actions. DescriptorCommit divergence, CommitPeerPlan divergence, and frozen M6
shadow-action divergence are all `0`. Shadow actions retain M6 logical offsets;
peer plans carry the new EP staging/direct-receive offsets.

## 3. GPU-only job execution

Each rank performs communicator/window allocation at job level, bridges the
Router input stream once, and launches one four-block persistent CUDA pipeline.
The four blocks continuously execute Router reveal publication, descriptor
scheduling, fused dispatch consumption, and LSA acquire waits. There is no
per-descriptor Python callback, CPU poll, CPU packing, CPU transport submission,
or CUDA launch; every corresponding audit counter is `0`.

This is what distinguishes M7 from the CPU scheduler: no host loop observes a
reveal or constructs/submits a descriptor action. Queue head/tail, destination
counts/cursors, metadata construction, copies, and completion publication are
all device memory operations performed inside CUDA kernels.

## 4. Token-centric fused dispatch

`progressive_dispatch_progress_kernel` consumes a descriptor commit and scans
the revealed `token x top-k` range once. For each assignment it reads
`topk_idx[token][k]`, computes `dst = expert / experts_per_rank`, checks the
commit's `authorized_dst_mask`, atomically reserves one destination slot, builds
`DispatchTokenMeta`, and copies `x[token]` into that slot. It does not execute a
destination loop that rescans all assignments.

Formal counters report `256` assignments scanned for exactly `256` input
assignments across both ranks, establishing the intended `O(assignments)` data-
plane scan. Metadata is generated in-kernel and contains source rank/token,
expert, top-k slot, descriptor, reveal epoch, and FP32 top-k weight.

## 5. Direct LSA layout and transport

Each rank registers one `49408`-byte NCCL symmetric window. The first `24704`
bytes are descriptor/source receive slots; the second equally sized region is
reserved for staged GIN. With 16 FP32 features, each M7 record is `96` bytes:
32 bytes of aligned metadata storage followed by 64 feature bytes. Peer stride
is `3088` and descriptor stride is `6176`.

For an LSA-capable peer, dispatch evaluates
`ncclGetPeerPointer(window, recv_offset, peer)` and writes metadata and features
directly to the remote receive slot. There is no local send-buffer packing or
send-buffer-to-remote bounce. Formal counters show `128` direct remote records
and `12288` direct payload bytes; the other `128` records are local.

## 6. Completion and visibility

After every descriptor's stores complete, all dispatch threads participate in
`ncclLsaBarrierSession<ncclCoopCta>::arrive` with release ordering. The peer's
wait block executes `wait` with acquire ordering before the receive epilogue is
launched. Formal counters show `8` release arrives and `8` acquire waits. A
store being issued is therefore not treated as remote completion.

## 7. Architecture copy policy

`DispatchCopyTraits<700>` selects 16-byte coalesced/vectorized load/store and
does not claim `cp.async` or TMA. The formal binary was compiled for `sm_70` and
executed on two V100 GPUs. `DispatchCopyTraits<800>` marks the same semantics as
cp.async-capable, and `DispatchCopyTraits<900>` marks it TMA/mbarrier-capable;
neither changes the common record or completion contract. M7 does not contain a
Hopper-only dispatch kernel.

## 8. Staged GIN integration

The same NCCL 2.29.7 compilation includes the staged GIN adapter. Because GIN
is not direct-address transport, the adapter maps a completed local staging
slot to `ncclGin::put(peer, recv_offset, staging_offset, bytes, completion)`.
Remote visibility remains `SignalInc/waitSignal`; local staging reuse remains
the independent `CounterInc/waitCounter` path.

Runtime status is **GIN_RUNTIME_NOT_AVAILABLE**. The current single-node host
reports `NCCL_GIN_TYPE_NONE` and exposes no real RDMA device, so no GIN runtime
PASS or benchmark is claimed.

## 9. Dispatch epilogue and handle

After acquire completion, `dispatch_count_experts_kernel` counts records for
each local expert, `dispatch_exclusive_scan_kernel` builds expert offsets, and
`dispatch_scatter_experts_kernel` writes expert-contiguous `recv_x` and
`recv_src_metadata`. Both ranks received `128` records. Each local expert owns
one contiguous 64-record range (`[0,64)` or `[64,128)`).

Each returned `ProgressiveEPHandle` reports `num_recv_tokens=128`,
`num_local_experts=2`, `num_topk=1`, and generation `1`. The payload and metadata
are scattered with the same output index, so `recv_x[i]` and
`recv_src_metadata[i]` remain paired.

## 10. Correctness and legality

Against the frozen M6 scheduler shadow and an independent CPU payload oracle:

- destination/count/route and legacy action offset divergence: `0`;
- byte-exact payload mismatch: `0`;
- lost, duplicate, corruption, and wrong destination: all `0`;
- future, unrevealed, stale, unauthorized destination, and cursor overflow:
  all `0`;
- device errors: `0`;
- expert-contiguous layout and handle pairing: both PASS.

Both ranks exited cleanly. The formal environment was two
`Tesla V100-SXM2-32GB` GPUs, CUDA `12.8`, PyTorch `2.8.0+cu128`, and independently
selected NCCL `2.29.7` with a symmetric LSA window.

## 11. Progressive evidence

`6` of `8` remote descriptor dispatches began before the corresponding rank's
final Router completion. Thus the progressive gate is true: already revealed
descriptors enter remote GPU memory while later chunks are still being revealed.
These timestamps establish mechanism overlap, not a performance claim.

## Stop rule

DescriptorCommit publication, token-centric fused dispatch, direct NCCL LSA
stores, staged GIN compilation, expert-contiguous epilogue, handle construction,
M6 equivalence, byte correctness, legality, and progressive execution are
complete. Work stops here; combine and all requested out-of-scope optimizations
were not implemented.
