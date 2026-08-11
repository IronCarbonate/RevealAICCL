# R3-F0 Formal Validation Preregistration

Status: **FROZEN BEFORE FORMAL CORPUS GENERATION OR EXECUTION**.

## Corpus contamination audit

Before this file was frozen, local and server artifact trees were audited for filenames and
structured seed fields containing 5042, 5142, or 5242. No matching corpus, run, or result
artifact existed. Broad byte/string matches inside hashes and binary traces were explicitly
rejected as non-semantic false positives. Therefore the formal seeds are frozen as:

- 5042
- 5142
- 5242

## Frozen system and corpus

- Reference Router; parameter seed 20260805; dimension 2048; four experts; deterministic top-1.
- Router-derived traffic only.
- EventBridge + compiled IncrementalState/FastBinder/DynamicGuard.
- `partial_current_only`, `partial_shards@75%`, `checkpoint8`.
- Runtime BFS zero; fast-path full rebuild zero.
- Reference deterministic packing and A2Av-T0 using real unequal-split
  `torch.distributed.all_to_all_single`.
- Five families in canonical order: balanced, skewed, all-to-one-like, zero-sized-pair,
  multiple-progressive-shards. The all-to-one-like family is retained unchanged.
- Twenty paired jobs/family/seed: 300 paired jobs total.
- 4,096 tokens/rank, eight chunks, 64-byte records. Descriptor granularity is unchanged.

Tokens are generated only from `(formal_seed, family_index, job_index, rank)` using the same
frozen generator as the pilot. No result-dependent sampling or family reweighting is allowed.

## Execution methodology

The primary run is one long-lived two-rank process with one NCCL initialization, one shared
communicator, one compiled plan, one EventBridge instance, fixed per-job router warmup, fixed
barriers, and GPU-idle completion before the next arm. C/D arm order is counterbalanced by
`(seed_index + family_index + job_index) % 2`.

The primary run has Kineto/CUPTI profiler **OFF**. Record-function labels may remain inert in
code but no profiler context or trace collection is active.

The CUPTI diagnostic subset is frozen before primary results as corpus job indices `{0, 10}`
for every seed and every family (30 paired diagnostic reruns). It runs separately after primary,
uses the same frozen system/data, and is excluded from every primary statistic. Its only purpose
is rank skew, submit-to-GPU-start, GPU-start-before-final, actual coexistence, and GPU Router
interference diagnostics.

## Paired arms and timing

- C: progressive early A2Av-T0.
- D: identical A2Av-T0 delta descriptors delayed until final Router completion.

Within every pair, tokens, Router/weights/top-k, destination map, descriptors/order, sendcounts,
payload bytes, total bytes, A2Av call count, and final payload multiset must match exactly. D may
not merge descriptors. Only descriptor execution timing differs.

Primary `T` is earliest first Router launch across ranks to latest completion of all required
variable-size A2Av payloads. It includes Router, count/offset construction, reference packing,
payload H2D, delta-count exchange, compiled AICCL, submission, and payload completion. Receive
D2H/unpack/checksum is excluded from primary and included in full-reference makespan.

For `Delta = T_D - T_C`, formal PASS requires all of:

1. paired median Delta > 0;
2. 10,000-resample paired bootstrap 95% CI lower > 0;
3. all three formal seed medians > 0.

All five families are reported, but no family alone replaces the corpus-wide Gate.

## Prohibitions

No packing/count-exchange optimization, transport change or additional transport, chunk/payload
change, scheduler/75%/checkpoint8 change, expert GEMM, return A2Av, combine, DeepEP/PCCL,
artificial Router delay, family removal, benefit-based selection, or exploratory formal variant
is permitted.
