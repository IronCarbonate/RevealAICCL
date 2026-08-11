# Phase R5-P3 Fast Progressive Data Preparation — Pilot Preregistration

Frozen before generating or running any R5-P3 pilot workload.

## Question and arms

Can a preallocated, incremental and vectorized forward data-preparation path reduce packing/count
overhead while preserving byte-exact Router-derived descriptors, and does that improve the retained
progressive full-MoE path?

- `E0`: retained progressive-forward baseline with reference destination-list rebuild,
  `pack_forward_payload`, and serial payload-H2D then count exchange.
- `E1`: the same progressive-forward timing with fast data preparation.
- `D1`: identical E1 data preparation and descriptors, but forward communication is delayed until
  final Router completion exactly as the historical delayed control.

Primary optimization contrast: `Delta_fast = T_E0 - T_E1`.

Primary optimized progressive contrast: `Delta_progressive = T_D1 - T_E1`.

Positive values favor E1. Both contrasts are paired within the same three-arm job and are assessed
independently; they are not added to R4 results.

## Frozen fast data-preparation design

### Static, route-independent setup before the Router clock

- Allocate pinned per-chunk/per-destination metadata and feature buffers.
- Allocate one pinned output buffer for each of the unchanged seven descriptor slots.
- Precompute only route-independent token fields and feature identity digests.
- Record setup/precompute latency separately. It is outside the inherited primary boundary just as
  E0's token-to-feature dictionaries are created before first Router launch; a conservative
  precompute-inclusive secondary makespan is also reported.

No future top-k, expert or destination is known or read during static setup.

### Reveal-time incremental path

- On each completed Router chunk, derive destination exclusively from actual top-k expert IDs.
- Vectorized scatter into preallocated destination buffers.
- Maintain a fixed `[chunk,destination]` counter matrix.
- Construct descriptor sendcounts by a small delta sum and offsets by prefix sum.
- Flatten already grouped buffers with contiguous slice copies; no Python token-list rebuild,
  per-token dictionary lookup, or per-descriptor sort.
- Vectorized metadata checksum completion must reproduce E0 metadata byte-for-byte.

### Count/payload overlap

- In progressive E1, launch the unchanged two-rank delta-count exchange after revealed sendcounts
  are known and before descriptor packing finishes.
- Pack into pinned buffers while the count collective progresses on an independent CUDA stream.
- Payload H2D uses the communication stream and may overlap the same count exchange.
- D1 does not start any communication before final Router completion; after release it uses the same
  count/H2D overlap but cannot prestart count exchange during earlier packing.
- Report count GPU duration, host-visible count latency, residual wait, and whether host buffers are
  pinned. No count value, payload call, descriptor order or bytes may change.

## Frozen semantics

E0/E1/D1 use identical:

- token inputs, Router weights/top-k and token-to-expert/destination assignments;
- compiled AICCL actions/checker semantics, `partial_current_only`, `partial_shards@75%`, checkpoint8;
- seven forward descriptor chunk groups/order, per-descriptor token order, sendcounts, offsets,
  metadata bytes, feature bytes and total bytes;
- full-size non-progressive expert MLP batches/weights/shapes/count;
- reference return A2Av descriptors, non-progressive return and actual combine;
- final combined outputs.

Runtime BFS/full rebuild and all unrevealed/future/duplicate/stale counters remain zero.

## Pilot corpus

- Candidate fresh seeds: `12042 / 12142 / 12242`.
- Exact local and server structured freshness must be confirmed before canonical execution.
- Five unchanged families × ten jobs × three seeds = 150 paired three-arm jobs.
- Deterministically counterbalanced six arm orders.
- Same 2×V100, two NCCL ranks, token dimensions, chunk layouts and A2Av-T0 backend.
- An already-consumed R4 seed may be used for one correctness smoke only and is excluded from the
  canonical analysis.

## Correctness and equivalence gates

- Every E1/D1 fast descriptor must byte-match E0 for chunk IDs, sendcounts, offsets, metadata and
  features before communication.
- Scheduler actions, expert batches/outputs, return descriptors and final outputs must match.
- Lost/duplicate/wrong/corruption/future access = 0; legality/token integrity = 100%.
- Fast buffers must be preallocated before first Router launch; no runtime token-list rebuild.

Any mismatch is fail-closed and blocks performance interpretation.

## Measurements

Report p50/p95/p99/max for each arm:

- count construction, offset construction, forward packing and reveal-time vectorized scatter;
- route-independent setup/precompute latency separately;
- forward count host latency, count GPU duration and residual wait;
- forward/return stages, expert, actual combine and full-MoE primary makespan.

Report:

- paired reference/fast packing speedup and reduction;
- `Delta_fast` and `Delta_progressive`: paired median, 10,000-resample bootstrap 95% CI,
  positive-pair count, per seed and per family;
- paired relative makespan reductions `(E0-E1)/E0` and `(D1-E1)/D1`;
- conservative `E1 primary + E1 static precompute` diagnostic;
- count-exchange p95/p99/max tails and amount of packing hidden by a prestarted count exchange.

## Pilot performance gates

Fast-path full-MoE PASS requires:

1. paired median `Delta_fast > 0`;
2. bootstrap 95% CI lower bound > 0;
3. 3/3 seed medians > 0.

Optimized progressive PASS independently requires the same three conditions for
`Delta_progressive`.

Packing mechanism PASS requires median fast packing latency below median E0 packing latency with
byte-exact descriptors. Count-exchange tails are diagnostic and cannot replace full-MoE gates.

Bootstrap seeds are frozen as `20260818` for `Delta_fast` and `20260819` for
`Delta_progressive`; paired-relative analyses reuse the corresponding resample indices.

## Prohibitions

No progressive expert, return descriptor reconstruction, progressive return/combine, scheduler or
reveal change, chunk/descriptor granularity change, packing/count parameter tuning after smoke,
MSCCL/MSCCL++, DeepEP, PCCL, transport replacement, workload selection, production integration, or
formal experiment is permitted.
