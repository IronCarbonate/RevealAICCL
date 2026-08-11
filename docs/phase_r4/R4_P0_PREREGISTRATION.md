# R4-P0 Full Reference MoE Pilot Preregistration

Status: **FROZEN BEFORE PILOT CORPUS GENERATION OR EXECUTION**.

## Fresh corpus audit and frozen corpus

Local and server artifact trees were audited for filenames and structured seed fields. No
artifact or result for 8042, 8142, or 8242 existed. These are frozen as the fresh R4-P0 seeds.

- Seeds: 8042, 8142, 8242.
- Families, unchanged and equally weighted: balanced, skewed, all-to-one-like,
  zero-sized-pair, multiple-progressive-shards.
- Ten paired jobs/family/seed: 150 pairs total.
- Router/job generator, token count/dimension, eight chunks, 75% partial shards, checkpoint8,
  compiled AICCL, EventBridge, reference packing, A2Av-T0, expert MLP 2048->32->16, and return
  transport are unchanged from R4-A0/C0.
- Empty/single-token cases remain correctness coverage and are not introduced into this fixed
  performance corpus, matching the established five-family performance corpus.

No family may be removed or reweighted after results. No extra exploratory pilot variant is
allowed.

## Arms and only experimental variable

- C: progressive forward dispatch.
- D: identical forward descriptors delayed until final Router readiness.

Within every pair, tokens, Router/weights/top-k, forward descriptors/order/counts/payloads,
expert input batches, expert weights and GEMM shapes, return descriptors/order/counts/payloads,
and final combined outputs must match. Expert execution remains non-progressive and begins only
after all forward dispatch has completed. The only experimental variable is when forward
descriptor execution begins.

C/D order is counterbalanced by `(seed_index + family_index + job_index) % 2` in one long-lived
two-rank process with one NCCL initialization, a shared communicator, fixed per-job warmup,
barriers, and GPU-idle arm boundaries.

## Primary timing and Gate

Primary `T` is earliest first Router launch across ranks to latest final actual combined output
ready across ranks. It includes Router, forward construction/packing/H2D/count exchange/A2Av,
non-progressive expert compute, return packing/H2D/count exchange/A2Av, required return D2H for
the reference CPU combine, and actual original-position combine.

Actual combine is a direct position scatter only. Independent expert-oracle reconstruction,
token/expert/return/checksum checks, allclose comparison, and other correctness-only verification
occur after the primary timestamp and are reported only in full-reference diagnostics.

The primary run has profiler OFF.

For `Delta = T_D - T_C`, PASS requires all of:

1. paired median Delta > 0;
2. paired 10,000-resample bootstrap 95% CI lower > 0;
3. all three fresh pilot seed medians > 0.

All five family results and p50/p95/p99/max stage/tail diagnostics are reported; no secondary
metric may replace the primary Gate.

## Prohibitions

No progressive expert execution, GEMM/packing/count-exchange optimization, transport/scheduler/
configuration change, DeepEP/PCCL, formal R4 corpus, or additional variant is permitted.
