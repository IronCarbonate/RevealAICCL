# Phase R4-F0 Formal Validation Preregistration

This protocol is frozen before generating or executing any R4-F0 formal workload.

## Freshness audit and corpus

- Formal seeds: `9042`, `9142`, `9242`.
- Local and server searches performed before preregistration found no structured seed records,
  filenames, corpus artifacts, or prior executions for these seeds.
- The earlier substring search produced only incidental matches inside hashes; a subsequent exact
  structured search for `"seed": N`, `seed=N`, and seed-bearing filenames returned no matches.
- Formal jobs per family per seed: 20.
- Families and proportions remain exactly equal:
  - balanced
  - skewed
  - all-to-one-like
  - zero-sized-pair
  - multiple-progressive-shards
- Total: 3 seeds x 5 families x 20 jobs = 300 paired C/D jobs.
- The zero-sized-pair family is retained without adjustment.
- RNG derivation remains `seed*100000 + family_index*1000 + job*10 + rank`.

No formal inputs are generated until this file exists. R4-P0 seeds `8042/8142/8242` and all
earlier R3 seeds are forbidden from the canonical R4-F0 corpus.

## Frozen system

The measured path remains:

`reference Router -> forward variable A2Av -> non-progressive reference FP32 expert MLP -> return variable A2Av -> actual combine`

- C: forward delta descriptors execute progressively when legally ready.
- D: the identical ordered forward delta descriptors execute only after final Router completion.
- The only experimental variable is forward descriptor execution timing.
- Router parameters, top-k, tokens, masks, chunk layouts, descriptor granularity and order,
  sendcounts, payload bytes, expert batches, expert weights, GEMM shapes, return descriptors,
  return payloads, and final outputs are paired-identical.
- `partial_current_only`, `partial_shards=75%`, checkpoint8, deterministic/fail-closed compiled
  AICCL, A2Av-T0, runtime BFS=0, and fast-path full rebuild=0 remain frozen.
- Expert execution is non-progressive and starts only after all forward descriptors finish.
- No profiler is enabled for the formal primary corpus.
- Pair arm order is counterbalanced by `(seed_index + family_index + job) % 2`.
- One long-lived two-rank NCCL process group is used for the corpus.

## Timing

Primary full-MoE makespan is:

`earliest first Router launch across ranks -> latest actual combined output ready across ranks`.

It includes Router, forward count/offset construction, reference packing, H2D, delta-count
exchange, compiled AICCL, forward variable A2Av, non-progressive expert compute, return packing,
return count exchange, return variable A2Av, required D2H, and actual original-position combine.

The primary clock stops immediately after actual combine. Token identity checks, checksums,
independent expert oracle, allclose, deterministic scheduler shadow/oracle, and artifact writing are
excluded from primary timing and retained as mandatory post-primary correctness work.

`Delta = T_D - T_C`.

## Formal Gate

R4-F0 passes if and only if all conditions hold:

1. paired corpus median Delta > 0;
2. paired bootstrap 95% CI lower bound > 0;
3. all three formal-seed medians > 0;
4. all semantic, legality, integrity, and C/D equivalence checks pass.

Bootstrap is frozen at 10,000 resamples using analysis RNG seed `20260814`.

Per-family results, including any negative family, are reported without changing the Gate.

## Required diagnostics

Report corpus, per-seed, and per-family Delta; C/D primary makespan; forward packing and count
exchange; expert compute; return packing/count exchange/stage; actual combine; p50/p95/p99/max
tails; positive-pair counts; and full-reference makespan including post-primary checks.

## Prohibitions

No progressive expert, packing/count-exchange/GEMM optimization, transport/scheduler/config
change, DeepEP, PCCL, workload tuning, family removal, production backend, or post-result Gate
change is permitted.
