# Phase R2-C0：Compiled Scheduler Semantic Equivalence

更新日期：2026-08-10  
状态：**TECHNICAL PASS / PENDING SUPERVISOR**

## 1. 结论

R2-C0 已真正实现 StaticPlanCompiler、IncrementalState、FastBinder、StaticProof、
DynamicGuard 和 single-process `CompiledEventDrivenRuntime`。旧 scheduler/checker 未修改，
并作为 E1/E2/E3 的逐项 oracle。

严格等价结果：

| Gate | tests | mismatches |
|---|---:|---:|
| E1 Static | **360** | **0** |
| E2 Single-step | **212** | **0** |
| E3 Trajectory | **36** | **0** |

E3 的 36 tests 由 24 条完整 trajectory 和 12 个 hidden-suffix paired tests 组成，
内部执行 524 个逐 step old/new 比较。另有旧路径与新增路径回归测试
`132/132 PASS`。

## 2. StaticPlanCompiler：runtime BFS 已替代

compile time 固化：

- usable edge 与 floored atomic capacity；
- canonical destination-first BFS 路径与全 OD distance table；
- destination×edge route templates；
- edge→所有 bandwidth-group 的重叠映射和 group credits；
- deterministic `(token ordinal, edge index)` template order；
- topology/path、endpoint、group mapping、template legality StaticProof digest。

E1 覆盖 6 个 topology（含 Rear4GPU、complete、shared-group、diamond ties、line、
zero/fractional capacity）：24 个 proof assertions、96 个 OD canonical paths、240 个
templates，共 360 tests。compile 执行 24 个 source BFS；所有 FastBinder/E3 runtime
`runtime_bfs_calls = 0`，FastBinder 源码不调用 reference `canonical_shortest_path`。

结论：**StaticPlanCompiler 已真正替代 runtime BFS**。

## 3. IncrementalState：fast-path full rebuild 已替代

预分配状态包括：

- ready/pending-ready chunk bitmap；
- revealed/ready token arrays；
- token×node holder bitmap；
- token×edge committed bitmap；
- residual-token 和 residual-demand matrix；
- remaining-hop table lookup；
- link/group credits；
- state version 与 fixed scratch arrays。

router chunk 先进入 pending-ready；只有 EventBridge/native worker 发布完成后，
`consume_event_ready` 才以 delta 激活 token。commit 只更新 proposal 涉及的 holder、
committed、residual 和 credits。24 条 trajectory 的 `full_rebuild_count` 全部为 0。

结论：**IncrementalState 已真正替代 trajectory fast path 的 full-state rebuild**。
`from_observation` 仅用于离线/random-state equivalence setup，不属于 runtime fast path。

## 4. FastBinder / StaticProof / DynamicGuard

FastBinder 只查 compiled distance/template arrays，按旧顺序生成 candidates，并按相同
单-token、edge capacity、所有 shared-group credits 贪心 packing；绑定对象仅来自 ready
范围内的 opaque `TruthTokenId`。

DynamicGuard fail closed，检查：

- expected state version；
- revealed/ready only；
- duplicate in-slot / duplicate commit；
- source possession / target absence；
- scheduler-selected residual progress；
- edge capacity；
- overlapping bandwidth-group conflicts。

旧 `commit_proposal` 始终保留为 oracle，没有删除或弱化 deterministic checker。

## 5. E2 Single-step

200 个 deterministic random states 加 12 个 adversarial cases，共 212 tests。随机集实测
包含 57 个 empty reveal、6 个 single reveal、50 个 full reveal case；其余覆盖 partial
prefix 和经过 0–2 个先验 commit 的 holder state。

12 个 adversarial checker cases：zero capacity、below-atomic capacity、exact capacity、
empty wait、duplicate in-slot、unrevealed counterfactual、invalid edge、source-not-holder、
edge-capacity conflict、shared-group contention、duplicate commit、stale state version。

结果：ordered candidates、selected action、opaque-token binding、accept/reject、applied
count、state version 和 holder state 全部 exact；**212/212，0 mismatch**。

## 6. E3 Trajectory

在 Rear4GPU 上运行 random/skew/hotspot workload，保持 8 chunks、前 6 chunks = 75%、
最后 2 chunks 到 checkpoint8 才可见。24 条完整 old/new trajectory 共比较 524 steps：

- 每步 ordered candidates exact；
- 每步 selected/bound action sequence exact；
- checker accept/reject、applied count、state version exact；
- 每步 holder state exact；
- completion 与 final residual state exact；
- link/group credit transition exact；
- runtime BFS=0，full rebuild=0。

此外：

- 192 次 pending-hidden 检查均未改变当前 action；
- 12 对真实 hidden-suffix perturbation 的 prefix action sequence 全部不变；
- 724 次 ordered candidate/action comparisons 中 tie/order divergence = **0**；
- 736 次 old/new checker accept/reject comparisons 中 mismatch = **0**。

因此 future chunk/top-k 仍不能影响当前 action。

## 7. Latency diagnostics（非 Gate 条件）

服务器 CPU、200 single-step cases；compile 和 state setup 排除。单位 µs：

| path | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| old build-view→enumerate→pack→bind | 873.680 | 2,204.015 | 2,992.499 | 3,478.816 |
| compiled template lookup→bind | 26.232 | 80.517 | 106.988 | 156.156 |
| compiled DynamicGuard apply | 42.349 | 133.017 | 201.510 | 237.551 |
| **compiled lookup→bind→guard apply direct** | **69.878** | **203.001** | **310.051** | **394.943** |

这些是 C0 semantic corpus diagnostics，不是 EventBridge→commit、concurrent pipeline 或
R2-F0 结果，不能与 W_host 直接宣称形成 overlap。

## 8. Gate、建议与停止点

R2-C0 technical requirements 全部通过，判定：

**R2-C0 = TECHNICAL PASS / PENDING SUPERVISOR**。

建议在 Supervisor 接受 C0 后进入 R2-F0：combined compiled diagnostic p95 约
203.001µs，值得在真实 EventBridge-ready timestamp 上测量 ready→commit feasibility。
这只是进入 F0 的建议，不是 F0 PASS，也不证明 commit-before-final-router-completion。

本轮未运行 R2-F0、R2-O0、formal E2E；未实现 AlltoAllv、expert packing/GEMM/combine、
DeepEP；未改变 partial_current_only、75%、checkpoint8；未使用 ProcessPool/Queue/
pickle/JSON critical path；未进入下一阶段。

## 9. Artifacts

- `rlccl/scheduling/compiled_event_driven.py`
- `scripts/run_r2_c0_equivalence.py`
- `tests/test_r2_compiled_scheduler.py`
- `outputs/phase_r2/c0_compiled_equivalence/r2_c0_results.json`
- `outputs/phase_r2/c0_compiled_equivalence/r2_c0_readback.json`
- `outputs/phase_r2/c0_compiled_equivalence/r2_c0_independent_readback.json`
