# Paper Draft（Phase 4.11）

更新日期：2026-08-10（Phase R0 evidence correction）
状态：**DRAFT，仅用于内部审阅；所有 claim 必须逐条对照 `CLAIMS_TO_EVIDENCE_MATRIX.md`，未通过的不进入正文。**
语言：正文草案为英文（学术论文惯例）；如需中文版可另行生成。

> **R0 evidence note.** The L1 aggregate result survives, but its original raw-job
> artifact is lost and the claim therefore has degraded provenance. P10-1D measured
> per-chunk CUDA router timing and then replayed quantized readiness; it did not run
> a concurrent router/scheduler pipeline. The 419.8-µs value is a replay/quantized
> candidate window. The 1,043.1/1,139.5/2,047.2-µs values are implementation
> fast-path estimates, not strict or theoretical lower bounds.

---

## Working Title

**When Information Timing Beats Scheduling: Reveal-Timing Value and the Infeasibility of a Reference-Router Production Path on a Frozen Scheduler**

（备选：*Early Reveal Helps Completion but Cannot Survive Production-Path Scheduling Latency*）

## 1. Outline

1. Introduction
2. Background and Problem Setting（traffic AICCL、reveal 语义、调度语义、frozen profile）
3. Methods（corpus 隔离、预注册、配对统计、真实计时、evidence tiers M/E/D/S/O）
4. Results
   4.1 Negative results on prediction and robust planning（H1、H2、H2a/H2b、W1–W3）
   4.2 Reveal timing is the dominant lever（Route A、H5、H6、H7）
   4.3 L1 and L2-S deployment validation（pilot、formal L1、formal L2、真实 NCCL）
   4.4 L2-R reference-router correctness（substrate、17/17 equivalence）
   4.5 Historical L2-R replay-path inadmissibility（candidate window、scheduler breakdown、implementation estimates）
   4.6 Negative results and applicability boundaries（hotspot_random_walk、pilot E2E、SF0-B）
5. Discussion
6. Limitations
7. Related Work（留位）
8. Conclusion
9. Artifact Availability

## 2. Abstract（草案）

Adaptive communication control for heterogeneous GPU collectives is often framed as a scheduling problem: given partial knowledge of future traffic, decide which transfers to start and when. We study this claim across a six-phase research program with pre-registered corpora (five mutually disjoint seed families) and paired bootstrap statistics. Three findings stand out.

First, scheduling-side interventions have no residual value: a scenario-robust prefix planner (H2), candidate reordering and lookahead (W2), risk gating (W3), and adaptive reveal control (H7) all fail to beat a simple current-observation scheduler, while historical traffic prediction (H1) is dominated by previous-value baselines.

Second, the timing and granularity of information reveal is the dominant completion lever: revealing routed tokens earlier reduces mean completion by up to 9.2 slots (Route A), and a rank-local streaming reveal improves end-to-end (E2E) time by 6.1–9.2 ms per job once measurable costs are included (H5), with token-shard reveal (partial_shards @ 75%) as the best fixed budget profile (H6).

Third, the resulting fixed profile has positive aggregate E2E results at L1 (+10.95 ms, 95% CI [+3.60, +23.15]) and on two V100s with real NCCL at L2-S (+6.46 ms, 95% CI [+3.41, +9.38]). The L1 raw-job artifact is lost, so only the surviving aggregates support that historical result. On the L2-R reference path, a replay/quantized candidate window (419.8 µs) is smaller than the frozen scheduler step p95 (12.3 ms); implementation fast-path estimates are 1,043.1 µs step-only, 1,139.5 µs including bind/checker, and 2,047.2 µs including digests. These measurements close the historical replay-based formal path, but they neither measure nor exclude a new concurrent/event-driven architecture.

## 3. Introduction（草案）

Modern collective communication runs on heterogeneous GPU clusters where link capacities, topology, and bandwidth sharing groups are static but traffic is dynamic. A promising research line, which we call *traffic-informed adaptive collective control* (AICCL), augments the communication layer with traffic observations so that the scheduler can start useful transfers earlier. Previous proxy-scale studies reported completion gains, but rarely separated (i) information value, (ii) scheduling algorithm value, and (iii) deployment feasibility on real hardware with real collectives.

This paper consolidates a pre-registered program spanning ambiguity-set construction (Phase 3B), traffic predictability (H1), early planning (H2), reveal-mechanism sensitivity (Route A), realizable reveal cost (H5–H7), and two deployment scales (L1 single-GPU high-fidelity; L2-S single-node multi-GPU with real NCCL), followed by a reference-router production-path audit (L2-R). We make three contributions:

1. a falsified set of scheduling-side hypotheses (H1, H2, H2b, W2, W3, H7) with full negative results preserved;
2. a validated reveal-timing profile (partial_shards @ 75%, checkpoint 8, current-observation scheduler) with E2E benefit at L1 and L2-S;
3. a bounded negative result for the historical L2-R replay path: the frozen scheduler's single-step latency (p95 ≈ 12.3 ms) exceeds its replay/quantized candidate window (419.8 µs), while implementation fast-path estimates are 1,043.1/1,139.5/2,047.2 µs (step-only/bind-checker/digest-inclusive). This does not measure or rule out a real concurrent/event-driven pipeline.

We emphasize the claims-to-evidence discipline: every claim in this paper maps to a frozen artifact and a statistical result (Claims-to-Evidence Matrix), and every overclaim—production MoE status, L3/DeepEP/RDMA validation, or L2-R E2E benefit—is explicitly excluded.

## 4. Evaluation（草案）

### 4.1 Setup and statistics

- Corpora: H1/H2 base seeds 642/742/842; Route A 1042/1142/1242; H5–H7 2042/2142/2242; L1/L2 formal 3042/3142/3242; P10-1 pilot/timing 4042 (dev/val). Formal corpus 5042/5142/5242 was never generated. All corpora are digest-disjoint.
- Statistics: sequence-level paired bootstrap (10,000, fixed seed), family-stratified; completion and E2E reported separately; profiling OFF for primary metrics; real CUDA timing for router/shard events.
- Hardware: historical L1 = 1× RTX 2080 Ti (aggregate evidence; raw jobs lost); L2-S/L2-R = 2× Tesla V100-SXM2-32GB, torch 2.8.0+cu128, NCCL 2.27.3.

### 4.2 Scheduling-side interventions have no value（负结果）

| Gate | Result |
|---|---|
| H1 prediction | paired Δ −0.0790 RMSE（CI [−0.1133,−0.0478]）；1/5 family 正；LOFO 0/5 正 |
| H2 robust prefix | E2E Δ −938.58 ms vs passive baselines；scheduling-only +0.11 |
| H2b algorithm value | +0.11 slots（CI [0.087,0.140]）；98.5% action overlap；no working region |
| W2 ordering / W3 gate | lookahead CI crosses 0；gate degenerates to always-act |
| H7 adaptive reveal | controller ≡ fixed B75（Δ=0.000）；oracle upper bound 0.0014 ms |

### 4.3 Reveal timing is the dominant lever

- Route A: completion monotonically decreases as full reveal moves earlier (36.18 → 20.95 → 14.92 → 13.40 → 11.80); S3 (slot 1) is within 1.0 slot of full-information (10.80).
- H5 (costs included): A2 +6.06 ms, A3 +5.98 ms, A4 +9.22 ms（CI [+8.26,+10.13]）; A5 (global aggregation) −0.13 ms.
- H6: partial_shards beats random by +0.60/+0.81/+0.57 ms across budgets, 5/5 family, 3/3 seed.

### 4.4 Deployment validation（L1 / L2-S）

| Scale | ΔE2E (D1 vs D0) | 95% CI | completion Δ | throughput | legality |
|---|---:|---:|---:|---:|---:|
| L1 (2080 Ti) | +10,953 µs | [+3,598, +23,148] | +6.43 slots | +24.7% | 100% |
| L2-S (2×V100, real NCCL) | +6,458 µs | [+3,409, +9,385] | +6.43 slots | +13.7% | 100% |

Real NCCL microbenchmarks: allreduce 62–87 µs, allgather 122–136 µs (2-rank), replacing assumed values.

### 4.5 L2-R reference-router correctness

The reference router historically passed 17/17 equivalence checks. Phase R0 adds a 19/19 CUDA strengthening run with an actual prefix-only 75% view, an independently reconstructed token-to-traffic oracle, real hidden-suffix counterfactual perturbation with no-leak assertions, token conservation/loss/duplication checks, and deterministic tie tests.

### 4.6 L2-R production-path infeasibility

- Replay/quantized candidate actionable window: 419.8 µs (2 slots × median chunk time 209.9 µs); this is not a directly measured concurrent-pipeline window.
- Scheduler single-step p95: 12,290 µs (P10-1E); reproduced at 11,290–12,933 µs in the fast-path audit.
- First-commit preparation p95: 8,674 µs (incl. deterministic checker).
- Implementation fast-path estimates: step-only 1,043.1 µs; bind/checker-inclusive 1,139.5 µs; digest-inclusive 2,047.2 µs. These are not strict/theoretical lower bounds.
- Verdict: the historical replay-based formal admissibility P4 and SF0-B fail, and that formal path remains closed. A new concurrent/event-driven architecture is not covered by that closure.

### 4.7 Applicability boundaries

hotspot_random_walk is negative or near-zero at every stage (L1 −1.1 ms; L2 −1.7 ms; pilot −32.8 ms; 1D reveal −0.59 ms / deployment −3.6 ms) and is preserved as a pre-registered boundary. Pilot E2E at L2-R scale is negative (−19.7 ms) and attributed to fixed setup dominance; completion gains remain positive but E2E benefit is not established at that scale.

## 5. Discussion（草案）

1. **Scheduling is not the bottleneck; information timing is.** Four independent gate chains (H2/H2b, W2/W3, H7, and the L2-R scheduler breakdown) converge: completion regret decomposes into information delay (~10 slots) plus a full-information scheduling-efficiency gap (~6.5 slots), and no decision-side intervention recovers either within the frozen semantics.
2. **Cheap, local information beats expensive, global information.** Rank-local streaming reveal is the only profile with robust E2E value; global aggregation (A5) and group-level reveal are not worth their synchronization cost.
3. **Deployment feasibility is a separate axis from algorithmic value.** The historical replay path is not admissible because the CPU scheduler step is ≈12 ms. Its static-cacheable BFS-dominated cost is measurable, but the current fast-path numbers are implementation estimates rather than an architecture-independent floor; real concurrency remains unmeasured.
4. **Honest reporting of negative results is part of the contribution.** The preserved failures (H1, H2, H2b, W2, W3, A5, H7, hotspot, pilot E2E, P10-F0-v1, SF0-B) prevent future re-exploration of the same frozen semantics.

## 6. Limitations（草案）

1. L3 (multi-node RDMA/NVSHMEM/DeepEP) is **not validated**; V100 (sm_70) does not support DeepEP.
2. The L2-R "router" is a reference substrate, **not a production MoE router**; expert GEMM/packing/combine are not implemented.
3. L1/L2-S conclusions use synthetic shims and are limited to single-node scales; they do not imply production SLA or multi-node behavior.
4. Scheduler latency and fast-path estimates were measured on the frozen Python implementation; vectorized/memoized variants were not implemented, and no theoretical lower bound was established.
5. The 419.8-µs replay/quantized candidate window is specific to the frozen workload (48-token world, 8 chunks, Rear4GPU) and is not a concurrent-pipeline measurement.
6. Cost parameters were updated by real microbenchmarks (L2-S); control-message RTT on a real fabric (E/S) was not measured.
7. Completion and E2E are reported separately; they do not always move together (steady-state E2E ≈ 0 at P10-1D scale despite +5.2 completion slots).
8. All statistics use pre-registered paired bootstrap; absolute wall times vary with machine load and are not directly comparable across runs.

## 7. Conclusion（草案）

We consolidated a six-phase evidence chain and found that (i) scheduling-side algorithm improvements have no residual value under the frozen semantics, (ii) earlier and finer token-level reveal is the dominant completion lever, (iii) the fixed profile has positive historical aggregate E2E results at L1 (with raw provenance now known to be incomplete) and supported L2-S results with real NCCL, and (iv) the historical L2-R replay path is not admissible. A real concurrent/event-driven pipeline remains a separate, unmeasured architecture question.

## 8. Claims 合规检查

- 正文每条数字均可在 `CLAIMS_TO_EVIDENCE_MATRIX.md` C1–C20 找到 artifact + 统计；
- 已排除：生产 MoE 表述、L3/DeepEP/RDMA 已验证、L2-R E2E 收益、scheduler <336µs 可认证、formal 通过；
- 图表：论文图表将复用既有表格数值，无新实验、无选择性删减。
