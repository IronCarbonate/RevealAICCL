# R3-P0 Preregistration (Frozen Before Pilot Results)

Status: **FROZEN BEFORE RUN**. This is a pilot protocol, not formal R3 validation.

## Scientific comparison and gate

The paired arms are:

- C: progressive early A2Av-T0;
- D: the identical ordered delta descriptors, submitted only after final router completion.

For every pair, tokens, router parameters, top-k, destination mapping, descriptor order,
sendcounts, packed payload bytes, number of variable-size A2Av calls, and final received
payload multiset are identical. Only descriptor execution time differs. D descriptors are
never merged.

The primary makespan is the distributed interval from the earliest first-router launch to
the latest completion of every required variable-size payload. It includes router work,
count/offset construction, reference packing, payload H2D, delta-count exchange, compiled
AICCL, A2Av submission, and A2Av completion. D2H/unpack/checksum verification is excluded
from the primary measure and included in the secondary full-reference makespan.

For `Delta_A2Av = T_D - T_C`, PASS was frozen as all of:

1. paired median greater than zero;
2. paired 10,000-resample bootstrap 95% CI lower bound greater than zero;
3. each of the three independent seed medians greater than zero.

No family-specific result may replace this corpus-wide primary gate.

## Frozen pilot corpus

- Pilot seeds: `6042`, `6142`, `6242`.
- Formal seeds `5042`, `5142`, `5242` are forbidden and are neither loaded nor generated.
- Families, in fixed canonical order: `balanced`, `skewed`, `all_to_one_like`,
  `zero_sized_pair`, `multiple_progressive_shards`.
- Jobs: 10 per family per seed (50 pairs/seed, 150 pairs total).
- Within each seed, family/job traversal is canonical. Pair arm order alternates by
  `(seed_index + family_index + job_index) % 2`: even runs C then D; odd runs D then C.
- Router: frozen reference router weights generated with parameter seed `20260805`;
  token dimension 2048, four experts, deterministic top-1.
- Each rank: 4,096 tokens and eight router chunks. Default chunks contain 512 tokens;
  `multiple_progressive_shards` uses `(128,256,384,512,640,768,512,896)`.
- Record payload: eight int64 fields (64 bytes/token).
- Scheduler profile: `partial_current_only`, `partial_shards@75%`, `checkpoint8`,
  compiled EventBridge/IncrementalState/FastBinder/DynamicGuard, runtime BFS zero and
  fast-path full rebuild zero.
- Backend: `A2Av-T0`, implemented by real unequal-split
  `torch.distributed.all_to_all_single`; no T1/T2/T3 intervention.

Token RNG is derived only from `(pilot_seed, family_index, job_index, rank)`. The workload,
descriptor granularity, bytes, and ordering are never changed in response to results.

## Frozen measurement and audit rules

- Host control/makespan uses monotonic timestamps; CUDA work uses a single Kineto/CUPTI
  timeline. Host and device timestamps are not directly mixed without the trace mapping.
- Count exchange is retained per descriptor and reported by arm, seed, chunk, and rank.
- Router kernels and payload/count NCCL kernels must be associated fail-closed through
  profiler annotations and external IDs. Missing or ambiguous associations invalidate the
  device-overlap diagnostic rather than being silently discarded.
- Correctness remains fail-closed: legality/token integrity 100%; lost, duplicate, wrong
  destination, corruption, unrevealed/future/stale/duplicate dispatch, runtime BFS, full
  rebuild, and C/D semantic divergence all zero.
- Packing and count exchange are measured without optimizing either implementation.
