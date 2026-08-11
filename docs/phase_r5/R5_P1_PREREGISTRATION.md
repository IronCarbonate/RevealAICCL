# Phase R5-P1 Progressive Expert Execution — Preregistration

Frozen before generating or running any R5-P1 pilot workload.

## Question

Does executing already-received expert tokens in thresholded batches hide expert MLP work behind
the remaining progressive forward path and reduce full-reference-MoE makespan?

## Arms and causal contrasts

All arms use the same Router/top-k, token inputs, weights, forward descriptors, final
token-to-expert mapping, return descriptors, return A2Av, and combine.

- `P`: progressive forward A2Av plus progressive expert execution.
- `E0`: the same progressive forward A2Av, but the R4 non-progressive expert boundary.
- `D`: identical delayed forward A2Av plus the R4 non-progressive expert boundary.

Primary incremental contrast: `Delta_expert = T_E0 - T_P`.

Secondary total-pipeline contrast: `Delta_pipeline = T_D - T_P`.

Diagnostic forward-only contribution on the same corpus: `Delta_forward = T_D - T_E0`.

The three contrasts must not be added across different corpora or endpoint definitions.

## Frozen progressive-expert policy

- Each destination expert has an independent FIFO ready-token buffer.
- Only tokens received from a completed forward descriptor may enter a buffer.
- Fixed threshold: **256 tokens per expert**.
- When a buffer reaches 256 tokens, exactly the oldest 256 tokens launch one FP32 reference MLP
  batch on an independent CUDA expert stream.
- After final forward dispatch completes, each expert flushes its single remaining sub-threshold
  batch, if any.
- Expert output waits until every expert batch completes; return packing/count exchange/A2Av and
  combine remain non-progressive.
- Each batch uses the frozen R4 expert weights and `2048 -> 32 -> 16` FP32 MLP.
- Communication completion uses stream-scoped synchronization so it does not drain the independent
  expert stream. The same synchronization rule is used in P/E0/D; packing, count exchange,
  transport, scheduler, descriptors, and payload sizes are not optimized.

Threshold rationale is static, not result-selected: 256 is half of a normal 512-token Router
chunk, enabling repeated batches without turning the reference MLP into tiny per-token GEMMs. No
other threshold will be tested in P1.

## Pilot corpus

- Candidate fresh seeds: `10042 / 10142 / 10242`.
- Local structured search before preregistration found no use of these seeds.
- Exact server freshness must be confirmed before canonical execution; if the server cannot be
  reached, canonical execution is held rather than changing seeds.
- Five unchanged R4 traffic families, equally represented:
  balanced, skewed, all-to-one-like, zero-sized-pair, multiple-progressive-shards.
- Ten jobs per family per seed: 150 paired three-arm jobs.
- Same token dimensions, chunk layouts, `partial_shards@75%`, checkpoint8,
  `partial_current_only`, compiled AICCL, reference packing, and NCCL A2Av-T0.
- Arm order is deterministically counterbalanced before execution.
- Primary profiler is disabled; explicit CUDA events provide expert/forward overlap diagnostics.

## Correctness requirements

- P/E0/D Router assignments and token-to-expert mapping identical.
- Expert weight digest identical.
- Every token executes exactly once; expert execution loss/duplicate = 0.
- No unrevealed/future token enters expert buffers.
- Final output arrays pairwise equivalent at the frozen reference tolerance.
- Return descriptors semantically identical.
- Legality and token integrity 100%; all prior R4 loss/duplicate/wrong/corruption counters zero.
- Runtime BFS and full rebuild remain zero.

## Timings and diagnostics

Primary makespan remains earliest first Router launch across ranks to latest actual combined output
ready across ranks. Correctness-only oracle/checker work is excluded.

Report:

- `Delta_expert`, `Delta_pipeline`, and `Delta_forward`: paired median, 10,000-resample bootstrap
  95% CI, per seed, and per family.
- Expert batch-size distribution, batches/GEMM launches per expert, threshold versus flush batches.
- Expert GPU-active time P versus E0 and paired efficiency degradation.
- Expert GPU time and tokens completed before final Router and before final forward completion.
- Expert tail remaining after final forward completion.
- Router, forward, expert, return, combine, packing, and count-exchange p50/p95/p99/max.

## Pilot Gate

Correctness/mechanism PASS requires all correctness requirements plus at least one pre-final-forward
expert batch in each seed.

Performance PASS requires all of:

1. paired median `Delta_expert > 0`;
2. bootstrap 95% CI lower bound > 0;
3. 3/3 seed medians > 0.

The same conditions are reported separately for `Delta_pipeline`, but a positive total-pipeline
result cannot hide a failed incremental expert contrast. Bootstrap seeds are frozen as 20260815
for `Delta_expert` and 20260816 for `Delta_pipeline`.

## Prohibitions

No packing/count-exchange/GEMM optimization, alternative threshold, MSCCL/MSCCL++, DeepEP, PCCL,
transport/scheduler/config change, progressive return/combine, workload tuning, production
backend, or formal validation is permitted in R5-P1.
