# Phase R5-P2 Progressive Return — Pilot Preregistration

Frozen before generating or running any R5-P2 pilot workload.

## Question

Can an unchanged full-size expert computation expose completed return descriptors early enough to
overlap real variable-size return A2Av with later expert computation and reduce full-reference-MoE
makespan?

## Paired arms and only experimental variable

- `E0`: the retained R4 progressive-forward baseline; all expert batches finish before return begins.
- `P2`: the identical progressive-forward path and identical full-size expert batches, but each
  existing return descriptor is launched as soon as every expert batch on which it depends has
  completed.

The only allowed difference is the launch time of each return descriptor. Router/top-k, forward
descriptors, expert input order, per-expert batch membership, GEMM order/shapes/count, expert
weights, return descriptor order/content/count/bytes, final payload multiset, and combine are frozen
pairwise identical.

Primary paired contrast: `Delta_return = T_E0 - T_P2`, where positive favors P2.

## Descriptor dependency rule

Historical R4 return descriptors are forward-delta descriptors, while full-size expert batches are
grouped by expert. P2 does not change either granularity. For each existing return descriptor, the
runtime precomputes the set of expert IDs appearing in that descriptor. It may pack and launch that
descriptor only after CUDA completion events for all members of that dependency set are complete.

Descriptors execute in the same deterministic order as E0 so both NCCL ranks issue matching
collectives. Busy `event.query()` polling is allowed; no future output may be read. A descriptor with
no legal early-ready point must remain delayed. The experiment must report dependency-set sizes and
the number of descriptors actually submitted/completed before final expert completion. It is
forbidden to split, merge, regroup, reorder, or otherwise manufacture early-ready descriptors.

## Frozen expert and return execution

- The R4 FP32 reference expert MLP remains `2048 -> 32 -> 16`.
- Exactly one full-size batch per non-empty local expert, in ascending expert-ID order.
- Both arms use the same explicit per-expert CUDA events so instrumentation is symmetric.
- No progressive-expert threshold or remainder batches are used.
- Expert output is written into the same original received-token index layout.
- P2 launches the unchanged descriptor on the existing NCCL communication stream after its expert
  dependencies complete; E0 waits for the final expert event first.
- Return uses real PyTorch distributed NCCL uneven-split `all_to_all_single`.
- Actual combine stays non-progressive and begins after all returns complete.

## Pilot corpus

- Candidate fresh seeds: `11042 / 11142 / 11242`.
- Exact local and server freshness must be confirmed before canonical execution.
- Five unchanged families, equally represented: balanced, skewed, all-to-one-like,
  zero-sized-pair, and multiple-progressive-shards.
- Ten jobs per family per seed: 150 paired jobs.
- Same token dimensions, chunks, payload, `partial_shards@75%`, checkpoint8,
  `partial_current_only`, compiled AICCL, reference packing, and NCCL A2Av-T0.
- Pair order is deterministically counterbalanced.
- An already-consumed R4 seed may be used for one correctness smoke only; smoke is excluded from
  canonical analysis and cannot change this protocol.

## Correctness and equivalence

- E0/P2 Router assignments, top-k, forward descriptors, expert inputs, expert weights, per-expert
  index batches, batch shapes/count, expert outputs, return descriptors and final outputs identical.
- Return total descriptor count and total bytes identical.
- Lost, duplicate, wrong-expert, wrong-destination, wrong-return, wrong-position and corruption = 0.
- No future/unrevealed output access.
- Legality and token integrity = 100%; runtime BFS and full rebuild = 0.

## Measurements

Primary makespan is earliest first Router launch across ranks to latest actual combined output ready.
Correctness-only reconstruction/checker is excluded; actual combine is included.

Report:

- `Delta_return`: paired median, 10,000-resample bootstrap 95% CI, positive-pair count, per seed and
  per family;
- post-hoc paired relative makespan reduction `(E0-P2)/E0` with the same resamples;
- P2/E0 expert GPU interval and paired relative change;
- dependency-set sizes and descriptor eligible/submit/complete-before-final-expert counts;
- return GPU/host interval hidden before final expert completion and remaining return tail;
- Router, forward, expert, return, combine, packing and count-exchange p50/p95/p99/max;
- evidence of GPU contention from expert-interval and return-interval changes.

## Pilot gates

Correctness PASS requires every equivalence and safety condition.

Mechanism PASS requires at least one unchanged return descriptor to be submitted before final expert
GPU completion in each of the three seeds. If the preserved descriptor dependency graph provides no
such opportunity, mechanism FAIL is retained; descriptor granularity may not be changed in P2.

Performance PASS requires all of:

1. paired median `Delta_return > 0`;
2. 10,000-resample bootstrap 95% CI lower bound > 0;
3. 3/3 independent seed medians > 0.

Bootstrap seed is frozen as `20260817` for both absolute and paired-relative descriptive analyses.

## Prohibitions

No progressive-expert threshold, expert batch splitting, GEMM optimization, return descriptor
split/merge/reorder, packing/count-exchange optimization, progressive combine, MSCCL/MSCCL++,
DeepEP, PCCL, scheduler/transport/config change, workload tuning, production integration, or formal
experiment is permitted.
