# V1 Moment-Conditioned Policy: Formal Stage-Gate Report

## Decision

**NO-GO: stop at V1.** The moment-conditioned policy consumes the supplied
context (correct context consistently beats shuffled context), but it does not
improve held-out tail completion relative to the current-demand-only baseline.
It also degrades mean completion by 2.11%, just beyond the pre-registered 2%
limit. Per the project specification, V1.5 through V4 were not implemented.

## Experimental design

- Topology: `Rear4GPU`
- Training families: `smooth_ar`, `alternating_burst`, `moving_hotspot`,
  `sparse_switching`
- Strictly distribution-held-out families: `bimodal`, `heavy_tail_clipped`
- Training seeds: 42, 142, 242
- Training scale: 10 sequences/family, 64 collectives/sequence, 5 epochs
- Validation: separate sequences from the four training families only
- Evaluation scale: 10 sequences/held-out family, 64 collectives/sequence
- Methods: current-demand-only baseline, mean-only context, full moment context,
  and context shuffled across sequences
- Device: NVIDIA GeForce RTX 4090, PyTorch 2.8.0+cu128
- Paired controls: every method sees the same topology and current traffic
  matrices; only model/context changes
- Hard constraints: unchanged deterministic feasibility masks

An initial diagnostic run selected checkpoints on separate sequences from the
held-out family names. That design was rejected because those families were no
longer strictly held out. The results below are exclusively from the corrected
rerun, where model selection sees training families only.

## Overall held-out results

Values are means across the three independently trained seeds. Brackets show a
normal-approximation 95% confidence interval across training seeds.

| Method | Mean completion | p95 | p99 | CVaR90 | CVaR95 | Mean synthesis |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 8.5750 [8.5075, 8.6425] | 10.6667 | 10.6667 | 10.1482 | 10.6695 | 37.510 ms [36.781, 38.239] |
| Mean-only | 8.7589 [8.6004, 8.9173] | 11.0000 | 12.0000 | 10.9353 | 11.2153 | 42.069 ms [41.431, 42.707] |
| Moment-full | 8.7563 [8.5998, 8.9127] | 10.6667 | 12.3333 | 10.5808 | 10.9638 | 42.594 ms [42.257, 42.932] |
| Moment-shuffled | 9.0930 [8.8574, 9.3285] | 12.0000 | 13.3333 | 12.1606 | 12.3782 | 43.873 ms [42.694, 45.051] |

All 15,360 evaluated schedules were legal and none timed out. Compared with the
baseline, moment-full increases mean completion by 2.11% and synthesis latency
by 13.55%. It ties aggregate p95 but worsens p99 and both CVaR metrics.

## Per-family results

| Family | Method | Mean | p95 | CVaR95 |
|---|---|---:|---:|---:|
| bimodal | baseline | 8.4417 | 10.0000 | 10.2198 |
| bimodal | moment-full | 8.6245 | 10.6667 | 11.0673 |
| bimodal | shuffled | 8.7401 | 11.0000 | 11.3906 |
| heavy_tail_clipped | baseline | 8.7083 | 10.6667 | 10.6667 |
| heavy_tail_clipped | moment-full | 8.8880 | 10.6667 | 10.9443 |
| heavy_tail_clipped | shuffled | 9.4458 | 12.6667 | 12.8678 |

For `bimodal`, moment-full fails to improve baseline p95/CVaR in every seed.
For `heavy_tail_clipped`, only one of three seeds improves p95/CVaR. Thus neither
family has the required stable tail improvement.

## Context-ablation interpretation

Correct context stably outperforms shuffled context:

- On `heavy_tail_clipped`, moment-full beats shuffled p95 by 2 slots in all
  three seeds; CVaR95 improves by 1.72, 2.06, and 1.99 slots.
- On `bimodal`, full beats shuffled on mean or CVaR in two of three seeds.

This makes “the model ignored moment inputs” unlikely. Instead, history moments
are informative enough to alter decisions, but those alterations do not beat a
policy that already sees the complete current demand. Mean-only and full results
are also close, providing no evidence that the added variance/z-score features
improve scheduling quality in V1.

## Gate evidence

The formal gate required all of the following:

1. Stable p95 or CVaR improvement on every held-out family (positive in at
   least two of three seeds).
2. Correct context stably better than shuffled context.
3. Mean completion degradation no greater than 2%.
4. Schedule legality of 100%.

Criteria 2 and 4 pass. Criteria 1 and 3 fail. The result is therefore `NO_GO`,
not a preliminary or ambiguous pass.

## Tests and artifacts

- Remote CUDA/PyTorch suite: `51 passed`.
- Local NumPy suite after final edits: `41 passed, 3 skipped` because local
  PyTorch is not installed.
- CUDA baseline and moment one-epoch smoke training: passed.
- Paired four-method smoke evaluation: passed.
- Formal detail: `outputs/moment_v1/formal/v1_formal_detail.csv`
- Formal aggregate: `outputs/moment_v1/formal/v1_formal_summary.json`
- Per-seed training/evaluation logs and ablations:
  `outputs/moment_v1/formal/seed_{42,142,242}/`

Reproduce the complete formal experiment with:

```bash
python scripts/run_v1_ablation.py \
  --seeds 42 142 242 \
  --topology Rear4GPU \
  --num-train-sequences 10 \
  --num-validation-sequences 2 \
  --num-eval-sequences 10 \
  --sequence-length 64 \
  --window-size 16 \
  --min-history 8 \
  --epochs 5 \
  --batch-target 500 \
  --ppo-epochs 5 \
  --device cuda
```

## Recommended research direction

Do not stack CVaR, risk pricing, trust, or fallback modules on this failed V1
gate and call the result an improvement. If work resumes under a new hypothesis,
moments are better candidates for OOD detection or a guarded risk/fallback signal
than for direct schedule-quality improvement when the full current demand is
already known.
