# Phase R4-A0/C0 Reference Full-MoE Correctness

Status: **PASS, pending Supervisor review**. No performance conclusion is made.

## Implemented path

The canonical two-rank V100 run implements:

`reference Router -> progressive forward variable A2Av -> non-progressive expert MLP -> real variable return A2Av -> original-position combine`.

Forward and return each use two identically split real `torch.distributed.all_to_all_single`
payloads: exact int64 identity metadata and FP32 tensor values. This preserves token identity
without encoding large IDs in floats while transmitting real token features and expert outputs.
Counts are derived from Router assignments and original sources, not fabricated final counts.

The expert is a deterministic per-expert FP32 reference MLP with shapes 2048->32->16. Expert
execution starts only after all seven forward descriptors have completed. There is no progressive
expert execution. Return descriptors are built from the completed expert batches and route each
output to its original source rank and token position.

## Coverage and equivalence

Seven frozen correctness cases were executed in both Early and Delayed arms on both ranks:
balanced, skewed, all-to-one-like, zero-sized-pair, empty-shard, single-token-shard, and
multiple-progressive-shards.

- arm-rank runs: 28;
- Early/Delayed rank-case comparisons: 14, mismatches: 0;
- forward descriptors/tokens: 196 / 114,688;
- return descriptors/tokens: 196 / 114,688;
- distinct forward and return pair counts: 95 each;
- zero-sized forward and return pairs: 68 each.

Every Early/Delayed pair has identical Router/top-k, Router assignments, forward descriptors,
metadata/features, scheduler actions, expert input batches, expert weights/shapes/outputs,
return descriptors, and final combined output digest.

## Correctness

All canonical checks pass:

- token-to-expert mapping correct;
- forward source/destination and FP32 expert input integrity correct;
- expert output matches an independently reconstructed original-token oracle;
- return sender/source correct;
- original token position correct;
- final combine matches the oracle;
- legality and token integrity 100%;
- lost, duplicate, wrong-source, wrong-expert, wrong-destination, wrong-return,
  wrong-position, corruption, and expert-output mismatch all zero;
- runtime BFS, full rebuild, unrevealed/future/stale/duplicate dispatch, scheduler/checker
  divergence all zero.

## Diagnostic latency only

These measurements combine Early and Delayed correctness arms and are not a performance Gate.

| Stage | p50 / p95 / p99 / max |
|---|---:|
| Router host interval | 6.087 / 98.665 / 147.306 / 147.894 ms |
| Forward count construction | 484.132 / 1340.477 / 1576.101 / 3052.368 us |
| Forward offset construction | 0.798 / 0.914 / 1.365 / 1.858 us |
| Forward reference packing | 10.857 / 31.447 / 33.502 / 34.099 ms |
| Forward H2D | 753.359 / 1720.725 / 2068.421 / 2736.807 us |
| Forward count exchange | 360.416 / 27525.623 / 31304.656 / 34318.832 us |
| AICCL control | 204.079 / 299.944 / 373.857 / 579.904 us |
| Forward A2Av completion | 360.926 / 670.754 / 1005.379 / 1159.277 us |
| Expert H2D + FP32 MLP | 20.044 / 39.457 / 41.456 / 42.190 ms |
| Expert output D2H | 232.042 / 418.983 / 824.643 / 968.630 us |
| Return reference packing | 2.900 / 8.719 / 11.699 / 11.732 ms |
| Return H2D | 130.332 / 183.558 / 211.006 / 235.618 us |
| Return count exchange | 186.527 / 9810.260 / 103177.959 / 109705.434 us |
| Return A2Av completion | 186.751 / 258.930 / 322.648 / 330.408 us |
| Independent expert oracle | 5.747 / 6.682 / 6.934 / 7.024 ms |
| Python combine/check oracle | 158.767 / 165.754 / 169.165 / 170.209 ms |

The large reference packing, count-exchange, and Python verification numbers are retained as
diagnostics. They do not constitute an optimization target or performance conclusion in A0/C0.

## Conclusion

**R4-A0 = PASS and R4-C0 = PASS, pending Supervisor review.** The reference full-MoE
correctness substrate is sufficient to request R4-P0, provided P0 separately preregisters its
timing boundary and keeps correctness-only oracle work outside any performance primary while
retaining complete secondary reporting.
