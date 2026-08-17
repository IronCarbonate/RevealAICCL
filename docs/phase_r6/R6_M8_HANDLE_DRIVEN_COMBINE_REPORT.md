# R6-M8 Handle-Driven GPU Combine Report

## Result

**Handle-Driven GPU Combine PASS**

The validated scope ends at the original source-token order after deterministic
top-k reduction. M8 proves that the metadata retained by M7 is sufficient to
drive reverse routing without a return descriptor or a second scheduler pass.

## 1. Frozen M7 boundary

Router/reveal policy, `RevealRecord`, GPU Scheduler, `DescriptorCommit`,
`CommitPeerPlan`, `DeviceCommitQueue`, fused dispatch, dispatch record/window
semantics, expert-contiguous `recv_x`, `StaticPlanCompiler`, and all M4-M7
reference paths remain unchanged. M8 is implemented in a separate combine
module and separate formal extension.

`ProgressiveEPHandle` keeps its M7 ABI prefix and appends
`num_source_tokens`, `source_topk_idx`, and `source_topk_weights`. Existing M7
aggregate initialization therefore retains its original field meanings.

## 2. Handle-driven reverse routing

The new chain is:

`expert_output[i] -> handle.recv_src_metadata[i] -> src_rank/token/topk_slot -> return slot`

`progressive_combine_kernel` reads `DispatchTokenMeta` for each
expert-contiguous row and obtains `src_rank`, `src_token_idx`, `topk_slot`, and
`expert_id` directly. It does not rerun Router, rebuild a traffic matrix,
construct `CommittedAction`/return descriptors, call FastBinder, or enter the
GPU Scheduler.

Across all formal cases, `recv_x`/metadata pairing divergence, return-rank
divergence, return-token divergence, and return-topk-slot divergence are all
`0`.

## 3. CombineRange and handle legality

After M7 builds expert offsets, a GPU kernel creates one `CombineRange` per
local expert. Each range contains the expert-contiguous row interval, global
expert ID, handle generation, and transport flags. The combine kernel checks
handle/range generation, range bounds, source rank, source token, top-k slot,
expert/range consistency, return capacity, peer direct accessibility, and
unique-slot ownership.

The source reduction additionally compares every returned expert ID with
`source_topk_idx[token][slot]`. Formal stale-handle, range-bounds, wrong-rank,
wrong-token, wrong-slot, wrong-expert, missing-return, collision, and corruption
counters are all `0`.

## 4. Return buffer layout

Each source rank owns exactly one slot per `(source token, top-k slot)`:

`slot_id = src_token_idx * num_topk + topk_slot`

Every record is `80` bytes: a 16-byte `ReturnSlotMeta` followed by 16 FP32
expert-output values. `ReturnSlotMeta` records generation, expert ID, source
token, and top-k slot. An atomic generation claim detects duplicate writers;
the first implementation never performs remote floating-point reduction.

The tested return-region sizes are `3840`, `7680`, and `11520` bytes for top-k
1, 2, and 3 respectively. A same-sized second region is preallocated for GIN
staging.

## 5. Expert GEMM and metadata pairing

The formal extension executes the frozen M7 path through expert-contiguous
`recv_x`, then launches a GPU expert matrix multiplication. Its output row
index is unchanged, preserving:

`expert_output[i] <-> recv_src_metadata[i]`

The formal expert matrices are deterministic per-expert FP32 diagonal matrices;
the kernel nevertheless performs the normal row-by-matrix accumulation loop.
All `576` dispatched rows retained byte-exact input/metadata pairing before
expert execution.

## 6. Direct LSA return

For a local source rank, the expert output is copied directly into the local
return slot. For an LSA-accessible peer, M8 calls the backend-neutral
`device_transport_is_direct` and `device_transport_get_remote_ptr`, then copies
the raw expert output directly into the peer's symmetric return slot using the
M7 vectorized copy primitive. No local send-buffer bounce is used.

Formal counters report `576` mapped contributions: `288` local and `288` real
remote LSA returns. Remote return traffic is `23040` bytes. No NCCL collective,
NCCL send/recv, or MSCCL++ fallback exists in the combine path.

## 7. Full-handle completion

M8 intentionally implements full-handle combine, not expert-progressive return.
After all expert outputs have been copied, each rank performs one
`ncclLsaBarrierSession<ncclCoopCta>::arrive` with release ordering. The source
rank performs acquire `wait` before reduction. Three scenarios on two ranks
produced `6` release arrives and `6` acquire waits.

No CPU thread polls a row, output, descriptor, or completion. Python callback,
CPU poll, return construction, packing, transport submission, and CUDA launch
per output are all `0`.

## 8. Deterministic source-side reduction

`combine_reduce_epilogue` assigns one CUDA block to each original source token,
initializes its output to zero, and visits top-k slots strictly in ascending
order. For every valid slot it checks `ReturnSlotMeta`, reads the original
source-side top-k weight, and computes:

`output[token] += source_topk_weights[token][k] * return[token][k]`

Weights are not applied on the expert side. All `576` contributions were
reduced exactly once.

## 9. Required top-k and traffic coverage

The two-V100 formal gate covers:

- balanced traffic with `topk=1`;
- skewed traffic with `topk=2`;
- all-to-one-like traffic with `topk=3`.

The cases include local experts, remote experts, and multiple expert
contributions belonging to the same source token. Every scenario reports lost
`0`, duplicate `0`, corruption `0`, and PASS.

## 10. Final MoE equivalence

The end-to-end path is:

`Router -> M7 dispatch -> expert GEMM -> M8 direct return -> source reduction`

It is compared with an independent FP32 reference that executes the same
expert matrices and fixed ascending top-k reduction. With `rtol=2e-5` and
`atol=2e-5`, all six rank/scenario comparisons pass. Maximum absolute error is
`9.5367431640625e-07`.

## 11. Architecture and GIN status

The formal extension compiles for `sm_70` and runs on two
`Tesla V100-SXM2-32GB` GPUs with CUDA `12.8`, PyTorch `2.8.0+cu128`, and NCCL
`2.29.7`. It reuses the M7 `DispatchCopyTraits`: SM70 uses vectorized stores;
SM80 and SM90 preserve the same layout/semantics and remain cp.async/TMA-capable
extension points rather than mandatory Hopper-only code.

The same compile includes staged GIN return integration:

`expert_output -> local registered staging slot -> ncclGin::put -> source return slot`

Signal completion protects receiver visibility and counter completion protects
staging reuse. Runtime status remains **GIN_RUNTIME_NOT_AVAILABLE** because the
single-node host reports `NCCL_GIN_TYPE_NONE` and exposes no RDMA device. No GIN
runtime PASS or benchmark is claimed.

## 12. Regression result

Local full regression completes with `573 passed, 4 skipped`. The M8 native
extension and both LSA/GIN compile surfaces pass CUDA 12.8/NCCL 2.29.7 SM70
compilation. The dual-rank process exits cleanly with forward legality and all
combine device-error gates at zero.

## Stop rule

Handle-driven reverse routing, unique return slots, real LSA return, staged GIN
compilation, full-handle completion, deterministic top-k reduction, and complete
MoE equivalence are finished. Work stops here. Expert-ready progressive return,
local/hierarchical reduction optimization, GIN benchmarking, dispatch or
Scheduler changes, chunk tuning, and performance tuning were not implemented.
