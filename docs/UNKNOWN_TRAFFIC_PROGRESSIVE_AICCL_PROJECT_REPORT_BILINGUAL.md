# 未知流量下的渐进式 AICCL：项目总结与证据报告

# Progressive AICCL under Unknown Traffic: Project Summary and Evidence Report

更新日期 / Updated: 2026-08-10  
范围 / Scope: Phase 0/1 uncertainty semantics, Phase 3B–Phase 4.10 historical route, evidence repair R0, and the real-concurrency R1–R4 route  
当前状态 / Current status: **Supervisor R4-F0 = PASS / NO VETO**; stopped after formal validation; no production-backend claim is made  
通信后端 / Communication backend: **PyTorch distributed with NCCL**, not MSCCL/MSCCL++

---

# 第一部分：完整中文报告

## 1. 研究问题与最终回答

项目最初的问题是：

> 原始 AICCL 更适合完整 traffic 已知时的 collective scheduling；真实 MoE 的
> AlltoAllv 流量却由 Router 动态产生，事先未知。能否在不预测未来流量的前提下，
> 只使用已经揭示的 demand 安全地提前调度，并最终得到端到端性能收益？

现在可以在冻结的 **双 V100、reference Router、compiled AICCL、真实 NCCL、reference
full-MoE** 系统范围内给出肯定回答：

1. **安全性成立**：未揭示 token/top-k 永远不能进入 ready state、descriptor、packing
   或通信；所有动作经过 deterministic、fail-closed guard/checker。
2. **机制成立**：Router 在独立 CUDA stream 上逐 chunk 运行；native EventBridge 只发布
   已完成 CUDA event；compiled scheduler 增量消费 readiness；真实 NCCL 通信可以在未来
   Router chunk 尚在计算时被提交；在直接做过 device profiling 的 R2/R3 diagnostic
   samples 中，部分 NCCL GPU work 与 future Router kernel 确实同时执行。
3. **真实 variable-size traffic 成立**：sendcounts 直接由 Router top-k assignment 推导，
   通过真实不等 split 的 `torch.distributed.all_to_all_single` 传输，不是固定 descriptor
   all-reduce，也不是人工生成最终 sendcounts。
4. **forward critical path 的正式收益成立**：R3-F0 中 progressive early A2Av 相对
   identical delayed A2Av 的 paired median 为 **+0.829 ms**，95% CI
   **[+0.242, +1.439] ms**，3/3 formal seeds 为正。
5. **完整 reference MoE 的正式收益成立**：R4-F0 把非渐进 expert MLP、真实 return
   variable A2Av 与 actual combine 全部纳入 primary，paired median 为 **+2.801 ms**，
   95% CI **[+0.967, +3.714] ms**；按每个 paired job 的 delayed makespan 归一化后，
   逐对相对完工时间降幅 median 为 **+0.608%**，bootstrap 95% CI
   **[+0.225%, +0.960%]**。
   3/3 fresh formal seeds 为正，correctness 100%。

这个回答有严格边界：它是 **reference full-MoE + NCCL backend** 的研究结论，不是
production MoE runtime、DeepEP/PCCL/MSCCL、expert-parallel training 吞吐或多节点部署结论。

## 2. 最终系统是什么

冻结路径为：

```text
reference PyTorch Router（deterministic top-k）
  -> per-chunk CUDA completion event
  -> native busy-poll EventBridge
  -> IncrementalState（只加入 revealed chunks）
  -> StaticPlanCompiler + FastBinder + DynamicGuard
  -> Router-derived delta sendcounts / offsets
  -> deterministic reference packing
  -> real NCCL variable-size forward all_to_all_single
  -> 等全部 forward dispatch 完成
  -> non-progressive per-expert FP32 reference MLP (2048->32->16)
  -> real NCCL variable-size return all_to_all_single
  -> original-token-position actual combine
```

固定语义是 `partial_current_only`、`partial_shards@75%`、`checkpoint8`、runtime BFS=0、
fast-path full rebuild=0、deterministic/fail-closed checker。AICCL 是控制面调度器名称；
通信 process group 是 `nccl`，forward/return primitive 是 PyTorch
`all_to_all_single(..., async_op=True)`。仓库未接入 MSCCL/MSCCL++。

### Gate 总账

| 阶段 | 最终/当前判定 | 不能省略的结论 |
|---|---|---|
| Phase 3B | PASS | prediction-free ambiguity support 可审计；不等于调度收益 |
| Phase 0 | ALLOW / NO VETO | 旧 partial-demand 只是静态 mask；完成设计与语义审计，无性能结论 |
| Phase 1 initial / return | HOLD / ALLOW, NO VETO | greedy oracle/digest blocker 被保留并返工闭环；revealed-only 环境成立 |
| H1 | FAIL | MLP 不优于 previous-value；不预测 future |
| H2 / H2a / H2b | FAIL / CONDITIONAL PASS / FAIL | 实现可加速，但 robust 算法增量仅 0.11 slot |
| W1 / W2 / W3 | semantic PASS / no value / no value | 静态查询可加速；排序与风险门控无收益 |
| Route A | PASS | reveal timing/granularity 是主杠杆 |
| H5 / H6 / H7 | PASS / PASS / FAIL | 成本计入仍正；partial_shards 胜出；无需 adaptive |
| L1 pilot / formal | pilot PASS / derived-summary positive | original formal raw provenance LOST |
| L2-S | PASS | 2×V100 real NCCL，但 Router/GEMM 仍 synthetic |
| P10-I1 | 17/17 PASS；R0 19/19 PASS | reference Router correctness/no-leak |
| P10-1C / 1D | conditional / timing complete | completion 正，但 E2E 未确立；1D 是 replay |
| P10-F0-v1 / SF0-B | FAIL / FAIL | 0.42 ms replay window 小于 scheduler；历史 formal CLOSED |
| R0 | PASS / NO VETO | provenance 与措辞修复 |
| R1-C0 / T0 | TECHNICAL FAIL / COMPLETE | Router||scheduler 成立，三路 overlap 未成立 |
| R2-E0 / C0 / F0 | PASS / PASS / PASS | EventBridge、compiled equivalence、integrated hard Gate 通过；<300 us FAIL |
| R2-O0 | FAIL | CUPTI device-start 62.083% < 75% |
| R2-O1A / O1B | PASS / PASS | delayed control 证实正 critical-path value；保留 T0 with limitations |
| R3-A0/C0 / P0 / F0 | PASS / PASS / PASS | real Router-derived variable A2Av；forward formal +0.829 ms |
| R4-A0/C0 / P0 | PASS / PASS | full reference MoE correctness 与 pilot 通过 |
| R4-F0 | **Supervisor PASS / NO VETO** | +2.801 ms；formal validation 后停止 |

## 3. 研究路线为何这样收敛

项目没有直接从“未知 traffic”跳到一个性能数字，而是逐层排除了错误解释：

- 首先证明预测未来、K=8 稳健规划、换候选排序、风险门控和自适应都没有足够价值；
- 然后证明真正有价值的是 **更早、更细地揭示已到达的真实信息**；
- reference Router 正确后，历史 replay 路径仍因窗口远小于 Python scheduler 而关闭；
- 新建真正并发架构后，又发现 control plane 太慢；compiled event-driven 路径解决它；
- host submit-before-final 出现后，CUPTI 证明这不等于 device overlap，O0 因而诚实 FAIL；
- delayed control 随后证明，即便有 Router 干扰，early communication 仍能缩短 combined
  makespan；
- 最后把固定小消息替换为 Router-derived variable-size A2Av，并从 forward-only 扩展到
  expert + return + combine 的完整 reference MoE。

这条路线的重要价值不只是“最终为正”，还在于每个中间负结果都改变了下一步工程设计。

## 4. 历史探索：确定真正有价值的变量

### 4.0 Phase 0/Phase 1：先冻结“未知不可执行”的语义

**Phase 0 code/semantics audit — ALLOW / NO VETO。** 审计确认旧
`PartialDemandObservation` 只是在完整 demand 上施加静态 mask，不是 rolling/progressive
reveal environment；完整 `ProblemInstance`、旧 decoder pruning、current-matrix feature 与
generator latent metadata 都可能形成 truth side channel。Phase 0 因而只冻结 private
truth、public observation、scenario、execution 与 evaluator 的能力边界，并规划新接口，
没有实现或性能实验。独立回归为 **73 passed, 4 skipped**；四项 Torch 测试明确保持
SKIPPED/NOT RUN。Supervisor 对 Phase 0 交付 ALLOW / NO VETO，当时对实际启动 Phase 1
保持 HOLD，等待用户授权。

**Phase 1 uncertainty environment — initial HOLD，返工后 ALLOW / NO VETO。** 新环境实现
private truth/immutable observation、monotonic reveal、`TruthTokenId` 与 `ScenarioTokenId`
类型/命名空间隔离、revealed-only commit，以及每个 paired method 的独立 world/reveal
process。首次 Supervisor 审查虽然测试全绿，仍用确定性反例发现 greedy
`FullInformationOracle` 不是上界，并发现 manifest digest provenance 可由调用者伪造，
因此正式裁决 **HOLD / NO VETO**，禁止用截断负 regret 掩盖问题。返工把 oracle 替换为
可证明、不可执行的 full-information completion lower-bound reference，并由 factory
构造且由 runner 重算 canonical truth/topology/config digests。最终独立测试为 focused
**61 passed**、full **134 passed, 4 skipped**、targeted **3 passed**，Return Gate
**ALLOW / NO VETO**。这个阶段只证明未知 demand 不能被普通 policy 看见或执行，不作
任何 AICCL 性能结论；原 HOLD 作为审计历史保留。

### 4.1 Phase 3B：prediction-free ambiguity support — PASS

`boundary_scenarios`、K=8、校准半径 0.343279。unknown-case joint coverage 为
0.93979，五个 family 为 0.925–0.950；selected-vs-random paired 95% CI 为
[0.02919, 0.03883]；LOFO 五个 held-out family 均无系统退化；tail coverage 分别为
140/140、3250/3280、4642/4800。十个 artifacts、八张 raw 表共 250,140 行通过 exact
read-back。结论仅是“无需预测也可构造可审计 ambiguity support”，不是调度收益。

### 4.2 H1：历史预测 — FAIL

MLP 的 total-traffic RMSE 1.6468，差于 previous-value 的 1.5678；paired delta
−0.0790，95% CI [−0.1133, −0.0478]。五个 family 只有 stochastic-volatility 略正，
LOFO 0/5 正且 aggregate delta −0.4280。source/destination 主目标也没有稳定通过。
因此后续主线明确不预测未来 demand。

### 4.3 H2/H2a/H2b：稳健多场景规划 — FAIL / 条件性 PASS / FAIL

H2 的 robust completion 20.49 slots，优于 wait 25.88、仅略优于 partial 20.61；但
robust E2E 1042.46 ms，partial 103.88 ms、wait 115.80 ms。相对 partial 的 E2E delta
为 −938.58 ms，CI [−992.59, −896.65]，H2 conditions 1/3/6 FAIL。

H2a 分解出 ambiguity construction 479.7 ms、prefix synthesis 450.9 ms，两者占 E2E
89.3%；92.3% 在线时间可解释。估计需要约 7.1× 在线加速才能进入 baseline 1.5×，故
“计算上可能优化”条件性 PASS。

H2b 则证明没有值得优化的算法价值：robust 相对 partial 仅 +0.113 slot，CI
[0.087, 0.140]；98.5% 执行动作集合重合，首动作 71.0% 相同，discarded action=0。
因此没有足够的独立算法价值来支持昂贵的 robust planning，robust/predictor 路线冻结关闭。

### 4.4 W1/W2/W3：静态预计算、排序与风险门控

- W1 预计算 12/12 OD canonical path exact，查询 57.7 µs 降到 0.19 µs，约 **302×**；
  这是等价实现加速，completion 不变。
- W2 的 distance/headroom 与 baseline 完全相同；lookahead 仅 +0.030 slot，CI
  [−0.0067, +0.0767] 跨 0，同时 wall 74.5 vs 45.2 ms（+65%），无价值。
- W3 中 partial 相对 wait mean +5.27 slots，99% coordinates 为正，wasted action=0；
  留出 seed 上所有 gate 100/100 选择 act，退化为 `partial_current_only`。

### 4.5 Route A：揭示时机与粒度 — PASS

在 5,400 episodes 中，partial completion 随 full-reveal slot 提前从 36.18（slot32）
降至 20.95（slot16）、14.92（slot8）、13.40（slot4）、11.80（slot1）；fullinfo 为
10.80。相对 slot16，slot8/4/1 分别改善 +6.03/+7.56/+9.16 slots，CI 全 >0；推迟至
slot32 为 −15.22。相同 slot8 下细粒度比粗粒度好 1.38 slots。由此确定“何时揭示、
揭示多细”才是主杠杆。

### 4.6 H5/H6/H7：从可实现揭示到冻结 profile

- H5 把 compute/control/sync 成本计入：A2 +6.06 ms，CI [5.50,6.59]；A3
  +5.98 ms；rank-local streaming A4 **+9.22 ms**，CI [8.26,10.13]；global
  group aggregate A5 **−0.13 ms**。H5 PASS，但全局聚合是明确负结果。
- H6 在 25/50/75% 预算下，`partial_shards` 相对 random 分别 +0.604/+0.807/
  +0.573 ms，CI 全 >0，5/5 families、3/3 seeds。entry-level selector 无差异；
  token 分片选择胜出。
- H7 controller 在 300/300 cases 都选择 75%，与固定 B75 的 delta=0；每 episode
  oracle 也只好 0.0014 ms。H7 FAIL，最终冻结 `partial_shards@75% + checkpoint8 +
  partial_current_only`。

## 5. 早期部署证据及其 provenance 边界

### 5.1 L1 pilot 与 formal

L1 pilot（validation，300 jobs/arm）中 D1 completion 14.45 vs D0 20.59，E2E
45.49 vs 55.30 ms，即约 +9.8 ms/job、吞吐 +21.6%，P0 PASS。

L1 formal 派生汇总（300 jobs/arm）记录：D1 completion 13.91 vs D0 20.34；E2E
44,271 vs 55,195 µs；paired improvement **+10,953 µs**，CI
[+3,598,+23,148]；3/3 seeds、4/5 families，吞吐 +24.7%，legality 100%。但 R0
在本地、服务器和备份中均未找到 original L1 raw jobs，正式状态为
`L1_RAW_ARTIFACT_LOST`。不得重造 raw 冒充历史证据；因此这里只保留 historical
derived summary，不称完整 raw-level provenance。

### 5.2 L2-S：双 V100 + real NCCL — PASS

环境经 R0 修复为 2× Tesla V100-SXM2-32GB、world size 2、real NCCL；旧 RTX2080Ti
manifest 保留为 `SUPERSEDED`。D1 completion 13.91 vs 20.34；E2E 47,272 vs
53,729 µs，paired **+6,458 µs**，CI [+3,409,+9,385]；3/3 seeds、4/5 families，
吞吐 +13.7%。hotspot_random_walk 为 −1.741 ms，完整保留。它是 synthetic
shim/GEMM + real NCCL 的 L2-S 证据，不是生产 MoE。

## 6. 历史 P10 replay 路径：正确但不可准入

P10-R0 审计确认仓库没有 production MoE Router、expert GEMM、DeepEP；NCCL 可用；
V100 sm_70 不支持 DeepEP，MSCCL 依赖未安装。P10-S0 选择 minimal PyTorch reference
Router。P10-I1 历史 17/17 PASS；R0 强化为 19/19 PASS，覆盖真正 75% partial view、
独立 CPU token→traffic oracle、未揭示 suffix 反事实、no-leak、loss/dup 与 tie tests。

P10-1C pilot：20 jobs/arm，completion +1.95 slots，但 E2E −19.7 ms；固定 setup
约 80–100 ms 主导，hotspot −32.8 ms。P10-1D 三臂 B0/C0/C1 中，C1 completion
22.9 vs 28.1（+5.2 slots），但 steady E2E 151.4 vs 150.0 ms，收益未成立。

关键口径修正：P10-1D 是 **真实逐 chunk CUDA Router timing 完成后，把 readiness
量化/replay 给 scheduler**，不是 Router↔scheduler concurrent pipeline。419.84 µs 只是
replay/quantized candidate actionable window。P10-F0-v1 的 P1/P2/P3 PASS、P4 FAIL：
419.84 µs < scheduler step p95 12,290.03 µs。scheduler 复现范围 11.29–12.93 ms；
implementation fast-path estimates 为 step-only 1,043.1 µs、bind/checker-inclusive
1,139.5 µs、digest-inclusive 2,047.2 µs，均不是 strict/theoretical lower bound。
历史 replay-based P10-1 formal 因此 CLOSED，但并不禁止新的 concurrent/event-driven
architecture。

## 7. R0：Evidence Repair — PASS

R0 没改算法，只修证据：L1 raw 标 LOST；L2 canonical environment 重建为双 V100/
real NCCL，错误 manifest 标 SUPERSEDED；P10-I1 strengthened 19/19；所有文档将
replay/concurrency、candidate window 与 implementation estimate 区分。R0 Supervisor
= PASS / NO VETO。

## 8. R1：第一次真实 Router–Scheduler 并发 — TECHNICAL FAIL

R1 使用独立 CUDA Router stream、每 chunk forward/event、非阻塞 `event.query()`、
无 per-chunk synchronize，建立了真实 Router || scheduler；checker 后也执行真实 async
NCCL all-reduce。但 unchanged Python/ProcessPool control plane 太慢，未出现
commit-before-final 或 NCCL-submit-before-final，三路并发未成立。

artifact 配置为每 rank 20 个 primary trials，即 **40 rank-trials total**。W_host
p50/p95/p99 = 655.551/847.062/895.198 µs；
ready→scheduler 10.632/33.708/46.155 ms；ready→action 15.579/42.964/53.009 ms；
ready→commit p95 **43.051 ms**；ready→NCCL-submit p95 **56.831 ms**；NCCL
API→submit-return p95 114.117 µs。280/280 actions legal，no-future/token-integrity PASS，
但 legal action、commit、NCCL submit before final 全 FAIL。R1-C0 technical FAIL，R1-T0
complete；这次失败把瓶颈锁定为 control plane，而非 NCCL API。

## 9. R2：Compiled Event-Driven AICCL

### 9.1 R2-E0 EventBridge — PASS

native C++ pinned busy poller + preallocated ring/bitmap，在 2×V100 上取得 canonical
**8,000 total / 4,000 per rank** valid samples，给出 event completion→host-ready
conservative bound：p50 2.949 µs、p95
4.743 µs、p99 5.322 µs、max 60.863 µs、worst-rank p95 5.019 µs。<100 µs 与
stretch <50 µs 均 PASS。R1 延迟主要来自 serial worker backlog（p95 32.561 ms）和
Python enumeration（p95 约 6.5–6.8 ms），不是 CUDA event query。

样本数 provenance 注：一份 Supervisor 摘要记录“8,000/rank”，与 canonical artifact
计数不一致。runner 的 500 trials/rank × 8 events/trial 与 `r2_e0_results.json` 实际
对应 **4,000/rank、8,000 total**；本报告按 runner + artifact 采用后者，并保留该差异。

### 9.2 R2-C0 compiled semantic equivalence — PASS

真正实现 StaticPlanCompiler、IncrementalState、FastBinder、StaticProof、DynamicGuard。
compile-time BFS sources=24，runtime BFS calls=0；trajectory full rebuild=0。E1 Static
360/360、E2 single-step 212/212、E3 trajectory 36/36，全部 0 mismatch；524 per-step、
736 checker、724 ordered candidate/action comparisons 均无 divergence；hidden pending
192/192、hidden suffix 12/12 不影响 prefix。compiled lookup/bind p95 80.517 µs，guard
p95 133.017 µs，direct combined p95 203.001 µs；这些只是 semantic diagnostics，
不是 integrated ready→commit。

### 9.3 R2-F0 integrated path — PASS，stretch FAIL

首次串联 EventBridge→IncrementalState→FastBinder→DynamicGuard→descriptor→real async
NCCL。700 eligible events：ready→state/action/guard/NCCL-call/submit-return p95 分别
70.235/237.218/386.068/470.499/**578.891 µs**；submit p50/p99/max 为
426.230/959.454/2267.696 µs。hard gate <655.551 µs PASS，stretch <300 µs FAIL。
semantic/safety 全 PASS，600/700 host submit-return-before-final。该 85.714% 只表示
host API 已返回，不能证明 NCCL GPU work 已启动。

### 9.4 R2-O0 device overlap — FAIL

CUPTI 统一时间线直接识别 Router 与 NCCL kernels。host submit-before-final 720/840
=85.714%，但 early NCCL GPU-start-before-final 447/720=62.083%，低于预注册 75%；
三 seed 分别 71.67%/64.17%/50.42%。actual coexistence 324/840=38.571%
overall、45.0% early-only；positive-only overlap p50/p95/p99=5.343/719.387/787.375 µs。
Router final p50：A 2549.040、B 4067.560、C 4886.770 µs；C−B
**+808.299 µs**，CI [783.418,839.482]，约 +19.87%。O0 FAIL 保留，证明 host
submit 不能替代 device overlap，瓶颈转为 launch/rendezvous、rank asymmetry 与 contention。

### 9.5 R2-O1A delayed control — PASS

严格 A/B/C/D 控制保持同 Router/actions/7×64B communication。60 distributed pairs 的
`T_D−T_C` median **+3.450 ms**，bootstrap CI [+2.718,+5.166] ms，3/3 seeds
positive；但 p95 167.635 ms 由 seed4044 delayed-D 极端 rendezvous tail 主导，不得
一般化。C−B Router slowdown +775.659 µs，CI [743.516,815.579]（约 +19.14%）；
submit→GPU-start p50/p95 105.770 µs/22.243 ms，rank skew 715.747 µs/14.519 ms；
early start 65.0%，actual coexistence 48.194%。结论是 launch/rendezvous 与 resource
contention 同时存在，但 early communication 在计入干扰后仍有 combined-path value。

### 9.6 R2-O1B bounded transports — PASS；保留 T0 with limitations

四个预注册 transport：T0 baseline all-reduce，T1 lower-priority comm，T2 fixed 4-way
slicing，T3 fixed-size NCCL P2P。Primary paired results：

| Transport | Paired median | Bootstrap 95% CI | 3/3 seed | Gate |
|---|---:|---:|:---:|:---:|
| T0 | +845.335 µs | [+661.897,+1687.218] µs | 是 | PASS |
| T1 | +38,696.019 µs | [+9,348.645,+43,425.659] µs | 否；seed4044 −3,675.363 µs | FAIL |
| T2 | +780.394 µs | [−629.628,+1,523.190] µs | 否；seed4044 −91,269.019 µs | FAIL |
| T3 | +2,095.778 µs | [+1,251.219,+4,390.917] µs | 是 | PASS |

T3 虽 primary PASS，但 launch tail、rank skew、Router interference、delayed tail 相对 T0
无一改善。最终选择 **RETAIN T0 WITH LIMITATIONS**：
Router slowdown +764.091 µs，launch/rendezvous tail 与 rank asymmetry 继续冻结。

## 10. R3：真实 Router-derived variable-size AlltoAllv

### 10.1 R3-A0/C0 substrate correctness — PASS

路径为 Router top-k→destination lists→delta sendcounts/offsets→contiguous reference
packing→真实 NCCL uneven-split `all_to_all_single`→verification。7 coverage cases、28
arm-rank runs、196 descriptors；114,688 tokens 全部接收，lost/duplicate/wrong-dst/
corruption=0；unrevealed/future/duplicate/stale dispatch=0；C/D exact 14/14。196
src→dst pair size p50/p95/p99/max=257/764/1024/1024 tokens，95 distinct sizes、34 zero
pairs、min nonzero=1，证明真正 variable size。count-exchange max 166.851 ms 与 Python
verification tail 被保留；本 Gate 只证明 substrate/correctness。

### 10.2 R3-P0 forward A2Av pilot — PASS

seeds 6042/6142/6242，5 families×10 jobs，共 150 pairs。paired median
**+958.144 µs**，CI [+49.412,+1889.688]，三个 seed +765.712/+1752.732/
+718.986 µs。family median：balanced +2307.809、skewed +991.779、all-to-one-like
**−401.402**、zero-size +519.791、multiple-progressive +357.421 µs。C p95/p99 比 D
更差，不能声称 tail improvement。packing 只有 55.81% descriptors/45.08% bytes 在
final Router 前完成；count-exchange max C/D 140.005/161.163 ms。device actual
coexistence 18.333%，正 overlap p50 8.000 µs。correctness 全 PASS。

### 10.3 R3-F0 forward A2Av formal — PASS

fresh formal seeds 5042/5142/5242，5 families×20 jobs，共 300 pairs；primary run 无
profiler，另有隔离 CUPTI subset。paired median **+829.297 µs**，10,000-bootstrap CI
[+242.144,+1439.255]；三个 seed +401.309/+157.176/+1643.040 µs。family median：
balanced +49.649、skewed +1213.212、all-to-one-like +1498.601、zero-size +829.296、
multiple-progressive **−46.955 µs**。C/D primary p50 66.384/67.074 ms；C p95
143.405 ms 高于 D 136.644 ms，因此只证明 paired median critical-path value，不证明
普遍 tail improvement。forward packing p50 ~1.747 ms；count-exchange max C/D
109.693/193.403 ms。正式传输/接收 4,915,200 token records，correctness 100%。

R3-F0 的本地 aggregate artifact 不含逐 job denominator，因此不把两个边际统计拼成
“严格 paired percentage”。作为量级参考，`0.829297 / 67.074446 × 100% ≈ 1.236%`
（绝对 paired median / delayed marginal p50）；该 **1.236% 仅是 reference-scale ratio**，
正式 Gate 仍是 +829.297 µs paired delta 及其 CI。

与 primary 严格隔离、排除于 primary 统计的 CUPTI diagnostic subset 报告：payload
GPU-start-before-final 42.381%，actual future-Router/A2Av coexistence 20.952%，positive
overlap p50/p95/p99 = 7.184/19.334/23.169 µs。这些只说明 subset 中的 device 行为，
不替代 300-pair profiler-off primary。

## 11. R4：完整 reference MoE

### 11.1 R4-A0/C0 correctness — PASS

在 forward A2Av 后等待所有 dispatch，再执行完全相同的 per-expert FP32 MLP
2048→32→16；随后用真实 variable `all_to_all_single` 返回原 source，并按 original
token position combine。7 cases、28 arm-rank runs、14 C/D comparisons；forward 与
return 各 196 descriptors、114,688 tokens、95 distinct pair sizes、68 zero pairs。
token→expert、expert input/output、return source、position、combine 全正确；所有错误计数
为 0。expert 非 progressive；此 Gate 不作性能结论。

### 11.2 R4-P0 full-MoE pilot — PASS

fresh seeds 8042/8142/8242，150 pairs、600 rank-arm executions。Primary 从两个 rank
最早 Router launch 到最晚 actual combined-output-ready，oracle/checker 在 primary 后；
唯一变量是 forward descriptor 时机。paired median **+5.370 ms**，CI
[+2.232,+6.958]，三 seed +6.390/+2.507/+5.317 ms。family median：balanced
+10.509、skewed +6.174、all-to-one +6.572、zero-size **−4.252**、multiple-progressive
+5.304 ms。按与 F0 相同的逐对定义 `(T_D−T_C)/T_D`，相对完工时间降幅 median 为
**+1.173%**，10,000-resample bootstrap（seed 20260813）95% CI
**[+0.480%,+1.632%]**；三 seed 为 +1.414%/+0.600%/+1.173%。这是基于同一冻结 raw
pairs 的 post-hoc descriptive normalization，不是预注册 Primary/Gate，也不改变 P0 裁决。
C/D primary p50
478.166/486.390 ms；actual combine p50 177.808/176.416 µs。
forward count p99 31.891/29.421 ms，return count p99 103.041/103.300 ms；packing 与
count tail 很重但未消除 corpus-wide paired value。correctness/equivalence 150/150。

### 11.3 R4-F0 full-MoE formal — Supervisor PASS / NO VETO

freshness-audited seeds 9042/9142/9242，五 family 等比例、20 jobs/family/seed，共
**300 paired jobs、1,200 rank-arm executions**；zero-sized family 原样保留。paired median
**+2.800709 ms**，bootstrap 95% CI **[+0.967251,+3.714117] ms**；三个 seed median
+2.860/+3.597/+1.053 ms，全部为正；183/300 pairs positive（只作诊断）。

百分比使用严格配对定义
`relative_makespan_reduction_i = 100% × (T_D,i − T_C,i) / T_D,i`，不是拿两个 arm 的边际
median 相除。其 corpus median 为 **+0.607879%**，10,000-resample bootstrap 95% CI
**[+0.225202%,+0.960018%]**；三个 seed median 为
+0.577217%/+0.932785%/+0.242032%，均为正。作为仅供直觉的量级检查，绝对 paired
median 2.800709 ms 除以 delayed arm 的 marginal p50 476.812691 ms 为 0.587%，但这不是
正式 paired-percentage statistic。该百分比 CI 沿用 10,000 resamples 和 seed 20260814；
它是基于同一冻结 raw pairs 的 post-hoc descriptive normalization，不是预注册
Primary/Gate，也不改变 R4-F0 裁决。

family heterogeneity 必须保留：balanced +6.736 ms；skewed **−2.131 ms**；
all-to-one-like **−0.927 ms**；zero-sized +6.856 ms；multiple-progressive +3.912 ms。
对应的逐对相对完工时间降幅 median 为 +1.744%、−0.456%、−0.177%、+1.284%、
+1.023%。
预注册 Gate 不要求每 family 为正，观察结果后没有修改 Gate。

Primary C/D marginal p50 为 480.265/476.813 ms；边际中位数的差不等于 paired-difference
median，不能用它否定配对结果。expert+D2H p50/p95 为 C 22.696/44.533 ms、D
22.665/44.142 ms；return p50/p95 为 53.406/155.501 与 53.447/160.970 ms；actual
combine p50/p99 为 173.209/252.630 与 175.024/384.545 µs。forward packing p99
30.141/29.932 ms；forward count p99/max 29.750/117.064 与 29.224/132.811 ms；return
packing p99 12.532/12.528 ms；return count p99/max 103.944/115.208 与
109.834/121.971 ms。所有这些成本均包含在 primary 中，未优化、未剔除。

300/300 C/D comparisons 与 1,200/1,200 rank-arm executions 正确；Router/top-k、ordered
forward descriptors、sendcounts、payload bytes、expert batches/weights/outputs、GEMM
shapes、return descriptors、scheduler actions、final digests exact；legality/token integrity
100%，所有 loss/dup/wrong/corruption/future/divergence 计数为 0。
Supervisor 已正式裁决 **R4-F0 = PASS / NO VETO**；本项目在 formal validation 后停止，
未自动进入 production integration。

## 12. 必须保留的负结果和限制

1. H1、H2、H2b、W2、W3、H7 均失败或无价值；不是所有尝试都成功。
2. L1 original raw artifact 丢失；只能保留历史 derived summary。
3. 历史 P10-1D 是 replay，不是真实 concurrent pipeline；419.84 µs 不是实测并发窗口。
4. 1.043/1.140/2.047 ms 是 implementation estimates，不是理论下界。
5. R1 三路并发失败；R2-F0 <300 µs stretch 失败；R2-O0 的冻结 75% device-start Gate
   失败。
6. Host NCCL submit return 不等于 NCCL GPU work start；只有 CUPTI timeline 能支持
   device-overlap 结论。
7. T1/T2 interventions 失败；T3 没改善预注册稳定性指标；T0 带 Router contention、
   rank asymmetry 与 rendezvous tail。
8. R3/R4 的部分 family 为负，且 pilot/formal 负 family 会变化；收益是 corpus-wide
   paired median，不是每 workload 都更快。
9. packing 是主要稳定 host 开销，count exchange 有约 100–193 ms 极端 tail；没有普遍
   p95/p99 改善结论。
10. reference Router、reference packing、reference FP32 MLP 与 CPU combine 不等于生产
    MoE；没有 expert progressive execution、return overlap、production packing、DeepEP、
    PCCL、MSCCL、RDMA、多节点或训练吞吐验证。

## 13. 方法—问题—结果速查

| 部分 | 方法 | 解决的问题 | 关键结果 |
|---|---|---|---|
| 未知 future 安全性 | revealed-only + hidden-suffix counterfactual + fail-closed checker | 防止 unknown demand 被执行 | R0 19/19；后续 unrevealed/future execution 均 0 |
| Reveal profile | Route A + H5/H6/H7 | 找到可实现的固定信息预算 | partial_shards@75% + checkpoint8；H5 +9.22 ms；H7 证明无需自适应 |
| 真实 Router readiness | per-chunk CUDA events | 避免 replay 冒充并发 | R1 建立 Router||scheduler；W_host p50 655.551 µs |
| 低延迟事件桥 | native pinned busy polling + ring/bitmap | 去除 event/IPC 延迟疑点 | event→host-ready p95 4.743 µs |
| 编译调度器 | StaticPlanCompiler + IncrementalState + FastBinder + DynamicGuard | 消除 runtime BFS/full rebuild/Python enumeration | E1 360/360、E2 212/212、E3 36/36；0 divergence；direct p95 203.001 µs |
| Integrated control path | single-process event-driven runtime | 进入真实 Router window | ready→NCCL submit p95 578.891 µs；hard Gate PASS，<300 µs FAIL |
| Device 审计 | Kineto/CUPTI | 区分 host submit 与 GPU execution | O0 62.083% early start、38.571% overlap，75% Gate FAIL |
| Co-scheduling | identical delayed control | 判断 overlap 是否抵消 contention | O1A median gain +3.450 ms；C−B Router +775.659 µs |
| Transport 选择 | T0–T3 preregistered paired tests | 控制 rendezvous/资源竞争 | T0 +0.845 ms CI>0，保留但带限制 |
| Variable A2Av | Router-derived delta counts + real NCCL all_to_all_single | 从小 descriptor 走到真实不等 traffic | 114,688-token A0/C0 correctness；R3 formal +0.829 ms（约 1.236% reference-scale ratio，非严格 paired percentage） |
| Full reference MoE | non-progressive expert + real return A2Av + combine | 验证收益穿过完整 reference critical path | R4 formal +2.801 ms；逐对相对完工时间降幅 +0.608%，CI [+0.225%,+0.960%]；3/3 seeds |
| Future prediction（负） | H1 MLP vs previous-value | 判断是否值得预测 unknown traffic | Δ −0.0790 RMSE，CI 全负；FAIL |
| Robust planning（负） | K=8 scenario robust prefix | 判断未来场景规划是否有独立价值 | H2 E2E −938.58 ms；H2b 仅 +0.113 slot；FAIL |
| Historical replay（关闭） | quantized readiness replay | 检验旧 Python scheduler 能否进入候选窗口 | 419.84 µs vs 12.29 ms；formal CLOSED |

R3 的 +0.829 ms 与 R4 的 +2.801 ms 使用不同 primary 边界：前者止于 final forward
A2Av，后者止于 final combined output。**禁止把两者相加为累计 +3.630 ms。**

## 14. 中文结论

原始问题已经在冻结 reference 系统内完整走通：**不预测未来，只执行 revealed demand，
也能让 AICCL 在 Router 继续生成 future traffic 时安全地调度；compiled event-driven
实现把控制延迟压进真实窗口；Router-derived variable-size NCCL A2Av 在 forward-only
formal 中获得 +0.829 ms paired median；加入非渐进 expert、真实 return A2Av 和 combine
后，full-MoE formal 仍获得 +2.801 ms、逐对相对完工时间降幅 median +0.608%，两者
95% CI 全正且 3/3 seeds 正。**

最准确的项目定位是：一套经严格等价性、反事实 no-leak、真实 NCCL 与 pilot/formal
配对实验验证的 **reference progressive MoE communication research substrate**。下一步
若继续，应另立 production-oriented Gate，而不能把当前结果直接改称 production backend。

---

# Part II: Complete English Report

## 1. Research question and final answer

The original question was whether AICCL, which naturally assumes a complete traffic matrix, can
remain safe and useful when MoE Alltoallv traffic is generated online by the Router and is unknown
in advance. More precisely: can the scheduler avoid predicting future demand, execute only demand
that has actually been revealed, overlap useful communication with future Router work, and still
produce an end-to-end benefit?

Within the frozen two-V100 **reference Router + compiled AICCL + real NCCL + reference full-MoE**
system, the answer is yes.

1. Safety holds: unrevealed tokens and top-k assignments cannot enter ready state, descriptors,
   packing, or communication. Every action is checked by deterministic fail-closed guards/oracles.
2. The mechanism is real: Router chunks execute on a separate CUDA stream; a native EventBridge
   publishes only completed CUDA events; the compiled scheduler consumes readiness incrementally;
   and real NCCL work can be submitted while future Router chunks are still running. In directly
   profiled R2/R3 diagnostic samples, some NCCL GPU work actually coexisted with future Router
   kernels; this is not inferred from R4 formal host timing.
3. Traffic is genuinely variable-sized and Router-derived: destination lists and sendcounts come
   directly from top-k assignments, and uneven splits are passed to real NCCL-backed
   `torch.distributed.all_to_all_single` calls.
4. Forward-only formal validation passes: R3-F0 reports a paired median benefit of **0.829 ms**,
   95% CI **[0.242, 1.439] ms**, with all three formal seed medians positive.
5. Full reference-MoE formal validation also passes: after adding the non-progressive expert MLP,
    a real variable-size return A2Av, and actual combine, R4-F0 reports **2.801 ms**, 95% CI
    **[0.967, 3.714] ms**. Normalizing each paired job by its own delayed makespan gives a median
    paired relative makespan reduction of **0.608%**, bootstrap 95% CI
    **[0.225%, 0.960%]**, with all three fresh
    formal seeds positive and 100% correctness.

This is not a production-runtime result. It does not validate DeepEP, PCCL, MSCCL/MSCCL++,
multi-node RDMA, production packing, training throughput, or production expert kernels.

## 2. The final frozen system

The implemented path is:

```text
deterministic reference PyTorch Router
  -> per-chunk CUDA completion event
  -> native pinned busy-poll EventBridge
  -> revealed-only IncrementalState
  -> StaticPlanCompiler + FastBinder + DynamicGuard
  -> Router-derived delta sendcounts and offsets
  -> deterministic reference packing
  -> real NCCL variable-size forward all_to_all_single
  -> wait for all forward dispatches
  -> non-progressive per-expert FP32 reference MLP (2048->32->16)
  -> real NCCL variable-size return all_to_all_single
  -> original-position combine
```

The frozen semantics are `partial_current_only`, `partial_shards@75%`, checkpoint 8, zero runtime
BFS, zero fast-path full rebuild, and deterministic fail-closed checking. “AICCL” names the
control-plane scheduler; the communication backend is **NCCL**, initialized through PyTorch
distributed. The repository does not integrate MSCCL or MSCCL++.

### Gate ledger

| Stage | Final/current verdict | Essential qualification |
|---|---|---|
| Phase 3B | PASS | auditable prediction-free support, not scheduling benefit |
| Phase 0 | ALLOW / NO VETO | legacy partial demand was a static mask; design audit only |
| Phase 1 initial / return | HOLD / ALLOW, NO VETO | greedy-oracle/digest blockers preserved and repaired; revealed-only environment established |
| H1 | FAIL | MLP did not beat previous-value; future prediction abandoned |
| H2 / H2a / H2b | FAIL / conditional PASS / FAIL | implementation was optimizable, algorithmic gain was only 0.11 slot |
| W1 / W2 / W3 | semantic PASS / no value / no value | static lookup accelerated; ordering and gating did not help |
| Route A | PASS | reveal timing and granularity were the primary lever |
| H5 / H6 / H7 | PASS / PASS / FAIL | costed reveal positive; shards won; adaptation unnecessary |
| L1 pilot / formal | pilot PASS / derived-summary positive | original formal raw provenance lost |
| L2-S | PASS | two V100s and real NCCL, but synthetic Router/GEMM |
| P10-I1 | 17/17; R0 19/19 PASS | reference Router correctness and no-leak |
| P10-1C / 1D | conditional / timing complete | positive completion but no E2E proof; 1D was replay |
| P10-F0-v1 / SF0-B | FAIL / FAIL | replay window too short; historical formal closed |
| R0 | PASS / NO VETO | provenance and claim repair |
| R1-C0 / T0 | technical FAIL / complete | Router–scheduler concurrency, not three-way overlap |
| R2-E0 / C0 / F0 | PASS / PASS / PASS | event bridge, equivalence, integrated hard gate; <300 us failed |
| R2-O0 | FAIL | CUPTI device-start 62.083% below 75% |
| R2-O1A / O1B | PASS / PASS | delayed control positive; T0 retained with limitations |
| R3-A0/C0 / P0 / F0 | PASS / PASS / PASS | real Router-derived A2Av; forward formal +0.829 ms |
| R4-A0/C0 / P0 | PASS / PASS | full reference correctness and pilot passed |
| R4-F0 | **Supervisor PASS / NO VETO** | +2.801 ms; stopped after formal validation |

## 3. Why the route converged this way

The work did not jump from an unknown-traffic hypothesis to a favorable benchmark. It first ruled
out prediction, robust multi-scenario planning, candidate reordering, risk gating, and adaptive
reveal control. It then established that the useful variable was the timing and granularity of
truthful reveal. The historical Router-timing replay path failed because a roughly 0.42 ms
candidate window was much smaller than the roughly 12 ms Python scheduler step. A new concurrent
architecture exposed a control-plane bottleneck, compiled event-driven scheduling fixed it, and
CUPTI then exposed the separate device-plane problem that host submission is not GPU overlap.
Strict delayed controls showed that early communication still reduced combined makespan despite
Router interference. Finally, fixed descriptors were replaced with Router-derived variable A2Av,
then extended through expert compute, return A2Av, and combine.

The negative stages are therefore part of the main result: each one eliminated an invalid claim or
identified the next bottleneck.

## 4. Historical exploration and selection of the useful variable

### 4.0 Phase 0/Phase 1: freezing “unknown means non-executable” first

**Phase 0 code/semantics audit — ALLOW / NO VETO.** The audit established that the legacy
`PartialDemandObservation` was only a static mask over complete demand, not a rolling/progressive
reveal environment. Full `ProblemInstance` objects, legacy decoder pruning, current-matrix
features, and generator latent metadata were identified as possible truth side channels. Phase 0
therefore froze the capability boundaries among private truth, public observation, scenarios,
execution, and the evaluator, and only designed the new interfaces. It ran no performance
experiment. Independent regression was **73 passed, 4 skipped**; the four Torch cases remained
explicitly SKIPPED/NOT RUN. Supervisor accepted the Phase 0 deliverable with ALLOW / NO VETO while
holding actual Phase 1 execution until user authorization.

**Phase 1 uncertainty environment — initial HOLD, then ALLOW / NO VETO after repair.** The new
environment implemented private truth and immutable observations, monotonic reveal, typed and
namespace-separated `TruthTokenId` versus `ScenarioTokenId`, revealed-only commit, and an isolated
world/reveal process for every paired method. Despite a green initial test suite, Supervisor found
a deterministic counterexample showing that the greedy `FullInformationOracle` was not an upper
bound, plus a manifest-digest provenance gap. The formal initial verdict was therefore
**HOLD / NO VETO**; clipping negative regret was explicitly forbidden. The repair replaced the
greedy comparator with a provable, non-executable full-information completion lower-bound
reference and made factories/runners construct and recompute canonical truth/topology/config
digests. Final independent tests were focused **61 passed**, full **134 passed, 4 skipped**, and
targeted **3 passed**; the Return Gate was **ALLOW / NO VETO**. This phase established that unknown
demand could neither be observed nor executed by ordinary policies; it made no AICCL performance
claim, and the original HOLD remains part of the audit history.

### 4.1 Phase 3B — PASS

The prediction-free `boundary_scenarios` support used K=8 and calibration radius 0.343279.
Unknown-case joint coverage was 0.93979, all five families were between 0.925 and 0.950, the
selected-vs-random paired CI was [0.02919, 0.03883], and all LOFO comparisons avoided systematic
degradation. Tail coverage was 140/140, 3250/3280, and 4642/4800. Ten artifacts containing
250,140 raw rows passed exact read-back. This established an auditable prediction-free support,
not a scheduling benefit.

### 4.2 H1 prediction — FAIL

The MLP total-traffic RMSE was 1.6468 versus 1.5678 for previous-value. The paired delta was
−0.0790 with 95% CI [−0.1133, −0.0478]. Only one of five families was slightly positive, and LOFO
was positive in 0/5 folds with aggregate delta −0.4280. The main route therefore stopped predicting
future demand.

### 4.3 H2/H2a/H2b robust planning — FAIL / conditional PASS / FAIL

Robust completion was 20.49 slots versus 25.88 for wait and 20.61 for partial, but robust E2E was
1042.46 ms versus 115.80 and 103.88 ms. Its E2E delta against partial was −938.58 ms, CI
[−992.59, −896.65], so H2 conditions 1/3/6 failed.

H2a attributed 479.7 ms to ambiguity construction and 450.9 ms to prefix synthesis; 92.3% of
online time was explained, and an estimated 7.1× online speedup could potentially reach the
1.5×-baseline envelope. Computational feasibility therefore passed conditionally. H2b then showed
that optimization was not scientifically worthwhile: robust beat partial by only 0.113 slot, CI
[0.087, 0.140], while 98.5% of executed actions overlapped and first actions matched 71.0% of the
time. The +0.113 figure is measured in slots, not milliseconds; its small magnitude and the
98.5% action overlap/71.0% first-action match provide no sufficient independent algorithmic value
for expensive robust planning.

### 4.4 W1/W2/W3

W1 precomputed all 12 OD paths with exact equivalence and reduced a lookup from 57.7 to 0.19 us,
about 302×, but did not change completion. Distance and headroom orderings in W2 were identical to
baseline; lookahead improved only 0.030 slot with CI [−0.0067, 0.0767] while increasing wall time
from 45.2 to 74.5 ms. W3 found partial ahead of wait by 5.27 slots on average, positive in 99% of
coordinates with zero wasted actions. Every held-out rule selected “act” in 100/100 cases, so the
gate collapsed to `partial_current_only`.

### 4.5 Route A — PASS

Across 5,400 episodes, partial completion dropped from 36.18 at full-reveal slot 32 to 20.95 at
slot 16, 14.92 at slot 8, 13.40 at slot 4, and 11.80 at slot 1, versus full-information 10.80.
Relative to slot 16, the slot-8/4/1 improvements were 6.03/7.56/9.16 slots with CIs entirely above
zero; delaying to slot 32 cost 15.22 slots. At the same slot-8 endpoint, fine-grained reveal beat
coarse reveal by 1.38 slots. Reveal timing and granularity, rather than more complex planning, were
identified as the main lever.

### 4.6 H5/H6/H7

H5 included compute, control, and synchronization costs. A2 improved by 6.06 ms, CI
[5.50, 6.59]; A3 by 5.98 ms; rank-local streaming A4 by **9.22 ms**, CI
[8.26, 10.13]. Global group aggregation A5 was **−0.13 ms**, an explicit negative result.

H6 showed that `partial_shards` beat random reveal by 0.604/0.807/0.573 ms at 25/50/75% budgets,
with every CI above zero, 5/5 families, and 3/3 seeds. H7 selected 75% in 300/300 cases and was
exactly equal to fixed B75; even the per-episode oracle improved by only 0.0014 ms. The final profile
was frozen as `partial_shards@75% + checkpoint8 + partial_current_only`.

## 5. Early deployment evidence and provenance limits

The L1 validation pilot reported completion 14.45 versus 20.59 and E2E 45.49 versus 55.30 ms,
about 9.8 ms/job with 21.6% higher throughput. The L1 formal derived summary reported completion
13.91 versus 20.34, E2E 44,271 versus 55,195 us, a paired benefit of **10,953 us**, CI
[3,598, 23,148], 3/3 seeds, 4/5 families, and 24.7% higher throughput. R0 later failed to recover
the original L1 raw jobs locally, remotely, or from backups. Its status is
`L1_RAW_ARTIFACT_LOST`; the historical derived summary remains, but raw-level provenance must not
be claimed or recreated.

R0 corrected the L2 environment to two Tesla V100-SXM2-32GB GPUs, world size two, and real NCCL,
retaining the erroneous RTX2080Ti manifest as superseded. L2-S reported completion 13.91 versus
20.34 and E2E 47,272 versus 53,729 us: **6,458 us**, CI [3,409, 9,385], 3/3 seeds, 4/5
families, and 13.7% higher throughput. `hotspot_random_walk` was −1.741 ms. This was a synthetic
shim/GEMM path with real NCCL, not production MoE.

## 6. The historical P10 replay path

P10-R0 found no production MoE Router, expert GEMM, or DeepEP path. NCCL was usable; V100 sm_70
could not support DeepEP, and MSCCL dependencies were absent. P10-I1 first passed 17/17 tests; R0
strengthened it to 19/19 with an actual 75% partial view, an independently reconstructed CPU
token-to-traffic oracle, real hidden-suffix perturbation, no-leak, loss/duplication, and tie tests.

P10-1C improved completion by 1.95 slots but had −19.7 ms E2E because 80–100 ms fixed setup
dominated; hotspot was −32.8 ms. In P10-1D, C1 completion was 22.9 versus 28.1, but steady E2E
was about 151.4 versus 150.0 ms.

Critically, P10-1D measured real per-chunk CUDA Router timing and then quantized/replayed
readiness. It was not a concurrent Router–scheduler pipeline. The 419.84 us value was only a
replay-derived candidate window. P10-F0-v1 passed P1/P2/P3 but failed P4 because 419.84 us was
smaller than scheduler-step p95 12,290.03 us. Reproduced scheduler p95 was 11.29–12.93 ms.
The 1,043.1/1,139.5/2,047.2 us values are implementation fast-path estimates (step-only,
bind/checker-inclusive, and digest-inclusive), not strict theoretical lower bounds. Historical
replay-based P10-1 formal was closed, while a new concurrent architecture remained permitted.

## 7. R0 and R1

R0 repaired evidence without changing algorithms: L1 raw was marked lost, L2 provenance was
rebuilt, P10-I1 passed 19/19, and replay/concurrency and estimate/lower-bound language was fixed.
Supervisor verdict was PASS / NO VETO.

R1 then built the first real Router–scheduler concurrency with a separate Router stream,
per-chunk forwards/events, nonblocking queries, and no per-chunk synchronization. It also issued
real async NCCL after checker acceptance. However, no legal action, commit, or NCCL submit occurred
before final Router completion. The artifact contains **20 primary trials/rank, 40 rank-trials
total**. W_host p50/p95/p99 was 655.551/847.062/895.198 us;
ready-to-scheduler was 10.632/33.708/46.155 ms; ready-to-action was
15.579/42.964/53.009 ms; ready-to-commit p95 was **43.051 ms**; and ready-to-NCCL-submit p95
was **56.831 ms**. All 280 actions were legal, but the three-way gate failed. R1-C0 was a
technical FAIL and R1-T0 was complete, localizing the bottleneck to the control plane.

## 8. R2 compiled event-driven AICCL

R2-E0 used a native pinned busy poller and preallocated ring/bitmap. Across the canonical
**8,000 total / 4,000 per rank** valid samples, conservative event-completion-to-host-ready
p50/p95/p99 was 2.949/4.743/5.322 us, maximum
60.863 us, and worst-rank p95 5.019 us. Both <100 us and <50 us targets passed. Serial-worker
backlog and Python enumeration, not event query, dominated R1.

Sample-count provenance note: one Supervisor summary records “8,000/rank,” which differs from the
canonical artifact count. The runner uses 500 trials/rank × 8 events/trial, and
`r2_e0_results.json` therefore contains **4,000/rank, 8,000 total**. This report follows the
runner plus artifact and preserves the discrepancy.

R2-C0 implemented StaticPlanCompiler, IncrementalState, FastBinder, StaticProof, and DynamicGuard.
Compile-time BFS sources were 24; runtime BFS and trajectory full rebuild counts were zero. E1 was
360/360, E2 212/212, and E3 36/36 with zero mismatches. There were zero divergences across 524
per-step, 736 checker, and 724 ordered candidate/action comparisons; hidden-pending was 192/192
and hidden-suffix 12/12. Lookup/bind p95 was 80.517 us, guard p95 133.017 us, and direct combined
p95 203.001 us; these were semantic diagnostics, not integrated latency.

R2-F0 integrated the complete path through real async NCCL. For 700 eligible events, p95
ready-to-state/action/guard/NCCL-call/submit-return was
70.235/237.218/386.068/470.499/**578.891 us**. Submit-return p50/p99/max was
426.230/959.454/2267.696 us. The hard <655.551 us gate passed; the <300 us stretch failed.
All semantic gates passed, and 600/700 host submissions returned before final Router completion.
That host-side fact did not establish device overlap.

R2-O0 used Kineto/CUPTI rather than API return. Host submit-before-final was 720/840=85.714%, but
early NCCL GPU-start-before-final was only 447/720=62.083%, below the frozen 75% gate; the three
seeds were 71.67%, 64.17%, and 50.42%. Actual coexistence was 324/840=38.571% overall and 45.0%
early-only. Positive-overlap-only p50/p95/p99 was 5.343/719.387/787.375 us. Router final p50 was
2549.040 us for A, 4067.560 for B, and 4886.770 for C; C−B was **808.299 us**, CI
[783.418, 839.482], about 19.87%. O0 correctly failed.

R2-O1A added an identical delayed-communication control. Across 60 distributed pairs, median
`T_D−T_C` was **3.450 ms**, CI [2.718, 5.166], with 3/3 positive seeds. The 167.635 ms p95 was
dominated by the pathological seed4044 delayed rendezvous tail and is not a general gain estimate.
Router slowdown C−B was 775.659 us, CI [743.516, 815.579]. Submit-to-GPU-start p50/p95 was
105.770 us/22.243 ms, rank skew 715.747 us/14.519 ms, early start 65.0%, and actual coexistence
48.194%. Both launch/rendezvous instability and resource contention were present.

R2-O1B compared only four preregistered transports:

| Transport | Paired median | Bootstrap 95% CI | 3/3 seeds | Gate |
|---|---:|---:|:---:|:---:|
| T0 | +845.335 us | [+661.897,+1687.218] us | yes | PASS |
| T1 | +38,696.019 us | [+9,348.645,+43,425.659] us | no; seed4044 −3,675.363 us | FAIL |
| T2 | +780.394 us | [−629.628,+1,523.190] us | no; seed4044 −91,269.019 us | FAIL |
| T3 | +2,095.778 us | [+1,251.219,+4,390.917] us | yes | PASS |

T3 improved none of the preregistered launch-tail, rank-skew, Router-interference, or delayed-tail
metrics relative to T0. T0 was retained with its 764.091 us Router
slowdown, rendezvous tail, and rank asymmetry.

## 9. R3 real variable-size A2Av

R3-A0/C0 constructed Router-derived destination lists, delta sendcounts/offsets, contiguous
reference packing, real uneven-split NCCL `all_to_all_single`, and receive verification. Seven
cases and 28 arm-rank runs covered 196 descriptors and 114,688 tokens with zero loss, duplication,
wrong destination, corruption, unrevealed/future access, or stale dispatch. C/D equivalence was
14/14. Pair-size p50/p95/p99/max was 257/764/1024/1024 tokens, with 95 distinct sizes, 34 zeros,
and minimum nonzero size one. The 166.851 ms count-exchange maximum was retained.

R3-P0 used seeds 6042/6142/6242 and 150 pairs. Paired median was **958.144 us**, CI
[49.412, 1889.688], with seed medians 765.712/1752.732/718.986 us. Family medians were
2307.809, 991.779, **−401.402**, 519.791, and 357.421 us for balanced, skewed, all-to-one,
zero-size, and multiple-progressive. C had worse p95/p99 than D. Only 55.81% of descriptors and
45.08% of bytes were packed before final Router; count maxima were 140.005/161.163 ms; actual
device coexistence was 18.333%, with positive-overlap p50 8.000 us.

R3-F0 used fresh seeds 5042/5142/5242 and 300 pairs. Paired median was **829.297 us**, CI
[242.144, 1439.255], and seed medians were 401.309/157.176/1643.040 us. Family medians were
49.649, 1213.212, 1498.601, 829.296, and **−46.955 us**. C/D primary p50 was
66.384/67.074 ms, but C p95 143.405 ms exceeded D 136.644 ms; this is not universal tail
improvement. Packing p50 was about 1.747 ms and count-exchange maxima were 109.693/193.403 ms.
All 4,915,200 formal token records passed correctness.

The local R3-F0 aggregate artifact does not retain every job's denominator, so two marginal
statistics are not relabeled as an exact paired percentage. For scale only,
`0.829297 / 67.074446 × 100% ≈ 1.236%` (absolute paired median divided by delayed marginal p50).
This **1.236% is a reference-scale ratio**; the formal Gate remains the paired 829.297 us delta
and its confidence interval.

The CUPTI diagnostic subset was strictly excluded from the profiler-off primary. It reported
payload GPU-start-before-final 42.381%, actual future-Router/A2Av coexistence 20.952%, and positive
overlap p50/p95/p99 of 7.184/19.334/23.169 us. These are subset device diagnostics, not substitutes
for the 300-pair primary result.

## 10. R4 full reference MoE

R4-A0/C0 waited for all forward dispatches, ran identical non-progressive per-expert FP32 MLPs,
performed a real variable-size return A2Av, and combined by original token position. Seven cases,
28 arm-rank runs, and 14 C/D comparisons covered 196 forward and 196 return descriptors and
114,688 tokens in each direction, with 95 distinct sizes and 68 zero pairs. Every token/expert,
input/output, return-source, position, and combine check passed; this stage made no performance
claim.

R4-P0 used fresh seeds 8042/8142/8242, 150 pairs, and 600 rank-arm executions. Paired median was
**5.370 ms**, CI [2.232, 6.958], with seed medians 6.390/2.507/5.317 ms. Family medians were
10.509, 6.174, 6.572, **−4.252**, and 5.304 ms. Primary C/D p50 was 478.166/486.390 ms and
actual-combine p50 177.808/176.416 us. Using the same per-pair definition as F0,
`(T_D−T_C)/T_D`, the paired relative makespan-reduction median was **1.173%**, with a
10,000-resample bootstrap (seed 20260813) 95% CI **[0.480%, 1.632%]** and seed medians
1.414%/0.600%/1.173%. This is a post-hoc descriptive normalization of the same frozen raw pairs,
not a preregistered Primary/Gate, and does not change the P0 verdict. Forward count p99
was 31.891/29.421 ms and return count p99 103.041/103.300 ms. All 150 equivalence pairs passed.

R4-F0 used freshness-audited seeds 9042/9142/9242, five equally represented families, and 20 jobs
per family/seed: **300 paired jobs and 1,200 rank-arm executions**. Paired median was
**2.800709 ms**, bootstrap CI **[0.967251, 3.714117] ms**; seed medians were
2.860/3.597/1.053 ms, all positive. Positive individual pairs were 183/300 (diagnostic only).
For each pair, relative makespan reduction was defined as
`100% × (T_D − T_C) / T_D`. Its corpus median was **0.607879%**, with a 10,000-resample
bootstrap 95% CI of **[0.225202%, 0.960018%]**; seed medians were
0.577217%/0.932785%/0.242032%, all positive. The simple scale ratio of 2.800709 ms to the delayed
marginal p50 of 476.812691 ms is 0.587%, but it is not the formal paired-percentage statistic.
The percentage CI reuses 10,000 resamples and seed 20260814. It is a post-hoc descriptive
normalization of the same frozen raw pairs, not a preregistered Primary/Gate, and does not alter
the R4-F0 verdict.
Family medians were balanced 6.736 ms, skewed
**−2.131 ms**, all-to-one **−0.927 ms**, zero-size 6.856 ms, and multiple-progressive 3.912 ms.
The corresponding paired relative makespan-reduction medians were 1.744%, −0.456%, −0.177%,
1.284%, and
1.023%.

Marginal primary C/D p50 was 480.265/476.813 ms; marginal medians must not replace the median of
paired differences. Expert+D2H p50/p95 was 22.696/44.533 versus 22.665/44.142 ms; return was
53.406/155.501 versus 53.447/160.970 ms; combine p50/p99 was 173.209/252.630 versus
175.024/384.545 us. Forward packing p99 was 30.141/29.932 ms; forward count p99/max
29.750/117.064 versus 29.224/132.811 ms; return packing p99 12.532/12.528 ms; and return count
p99/max 103.944/115.208 versus 109.834/121.971 ms. These costs were included and not optimized.
All 300 C/D comparisons and 1,200 executions passed exact equivalence and correctness. Legality
and token integrity were 100%; loss, duplication, wrong source/expert/destination/return/position,
corruption, expert-output mismatch, runtime BFS, full rebuild, unrevealed/future access, and
scheduler/checker divergence were all zero. Supervisor formally accepted
**R4-F0 = PASS / NO VETO** and the project stopped after formal validation.

## 11. Negative results and limitations that must remain visible

1. H1, H2, H2b, W2, W3, and H7 failed or had no useful value.
2. Original L1 raw evidence is lost; only the historical derived summary remains.
3. P10-1D was replay, not concurrency; 419.84 us was not a measured concurrent window.
4. The 1.043/1.140/2.047 ms figures were implementation estimates, not theoretical lower bounds.
5. R1 failed three-way concurrency; R2-F0 failed <300 us; R2-O0 failed its frozen 75% gate.
6. Host NCCL API return is not GPU execution; device overlap requires direct profiling.
7. T1/T2 failed, T3 did not improve registered stability, and T0 retains contention/asymmetry.
8. Several R3/R4 families are negative; pilot and formal do not identify the same negative family.
9. Packing is a major host cost, count exchange has roughly 100–193 ms extreme tails, and no
   universal p95/p99 improvement is claimed.
10. Router, packing, FP32 expert MLP, and combine are reference implementations. There is no
    progressive expert execution, return-path overlap proof, production packing, DeepEP, PCCL,
    MSCCL/MSCCL++, RDMA, multi-node, production runtime, or training-throughput validation.

## 12. Compact method/result map

| Area | Method | Problem solved | Result |
|---|---|---|---|
| Unknown-future safety | revealed-only state, counterfactual suffix, fail-closed checker | Prevent unknown demand execution | R0 19/19; all later future/unrevealed counters zero |
| Reveal profile | Route A and H5/H6/H7 | Select truthful, implementable reveal | 75% shards + checkpoint8; H5 +9.22 ms; adaptation unnecessary |
| Real readiness | per-chunk CUDA events | Replace replay with real concurrency | R1 Router/scheduler concurrency; W_host p50 655.551 us |
| Event delivery | native pinned busy poll + ring | Eliminate event/IPC uncertainty | event-to-host-ready p95 4.743 us |
| Compiled scheduler | static plans, incremental state, binder, guard | Remove BFS/rebuild/enumeration | E1 360, E2 212, E3 36, zero divergence; direct p95 203.001 us |
| Integrated control | single-process event-driven path | Fit control into the Router window | ready-to-submit p95 578.891 us; hard pass, <300 us fail |
| Device audit | Kineto/CUPTI | Separate host submit from GPU overlap | O0 62.083% early start, 38.571% overlap; 75% gate fail |
| Co-scheduling | identical delayed control | Test value after contention | O1A +3.450 ms; Router slowdown +775.659 us |
| Transport choice | fixed T0–T3 paired study | Bound launch/rendezvous alternatives | retain T0: +0.845 ms, CI>0, with limitations |
| Variable A2Av | Router-derived delta counts + real NCCL A2Av | Carry real unequal traffic | A0/C0 correctness; R3 formal +0.829 ms (~1.236% reference-scale ratio, not an exact paired percentage) |
| Full reference MoE | non-progressive expert + return A2Av + combine | Preserve benefit through full path | R4 formal +2.801 ms; paired relative makespan reduction +0.608%, CI [0.225%,0.960%]; 3/3 seeds |
| Future prediction (negative) | H1 MLP vs previous-value | Test whether unknown traffic should be predicted | Delta −0.0790 RMSE, wholly negative CI; FAIL |
| Robust planning (negative) | K=8 scenario robust prefix | Test independent value of future scenarios | H2 E2E −938.58 ms; H2b only +0.113 slot; FAIL |
| Historical replay (closed) | quantized readiness replay | Test old Python scheduler against candidate window | 419.84 us vs 12.29 ms; formal CLOSED |

R3 +0.829 ms and R4 +2.801 ms use different primary endpoints: final forward A2Av versus final
combined output. They **must not be added** into a cumulative +3.630 ms claim.

## 13. English conclusion

The original research line is complete within the frozen reference system. Without predicting
future demand, AICCL can safely schedule only revealed Router traffic while future traffic is still
being computed. The compiled event-driven implementation brought the control path inside the real
Router window. Router-derived variable-size NCCL A2Av produced a formal forward-only paired median
benefit of 0.829 ms. After adding a non-progressive expert, real return A2Av, and actual combine,
the formal full-reference-MoE benefit remained positive at 2.801 ms and a paired relative
makespan-reduction
median of 0.608%, both with wholly positive 95% CIs and 3/3 positive formal seeds.

The correct project description is an auditable **reference progressive MoE communication research
substrate**, not a production backend. A production-oriented continuation would require a new Gate
rather than relabeling the current evidence.

---

# 第三部分 / Part III：证据索引与当前 SHA-256

以下 SHA-256 由当前工作区文件重新计算；它们用于定位 canonical summary/results。大量 raw
rows/traces 仍由各阶段 manifest/read-back 管理，不能用本表代替 raw artifact universe。

| Stage | Canonical artifact | Current SHA-256 |
|---|---|---|
| Phase 0 review | [SUPERVISOR_REVIEW_PHASE_0.md](agent_coordination/SUPERVISOR_REVIEW_PHASE_0.md) | `d9b724e99cd0784ba767c3cc80584be6c5dc0d8c96410652556d3b2d574e8d42` |
| Phase 1 review | [SUPERVISOR_REVIEW_PHASE_1.md](agent_coordination/SUPERVISOR_REVIEW_PHASE_1.md) | `13d3985527fb618f82fd0ed363ec9104808a1a4da561d01864abc7e672af8799` |
| Phase 3B | [summary.json](../outputs/phase3b_ambiguity/summary.json) | `0628310c6a061b9c1609b24e43713d2eb66ae4ebf043625cd7cc81126f4bdbca` |
| H1 | [summary.json](../outputs/h1_predictability/summary.json) | `c48e35230030215148e3def46340a991d226b69fd97797eb6d2086be6a26dfce` |
| H2 formal | [summary.json](../../phase4_formal_artifacts/summary.json) | `308b7730c4fbbd6fa823dc08f293bd3c71ee4fe0a2f3e8fcde35d5393e125961` |
| H2a | [a1_all_methods_e2e.json](../outputs/phase4_5/h2a_profile/a1_all_methods_e2e.json) | `a36f87becaaa524432b573da9257d168e0601757cd3937a95596f9044ec153ef` |
| H2b | [h2b_analysis.json](../outputs/phase4_5/h2b_analysis/h2b_analysis.json) | `6c743aaeea267aec83d0875a441c054c9add9d8f43fdf8c80284e97612a16b09` |
| W2 | [w2_diagnostic_full.json](../outputs/phase4_6/w2_scheduler/w2_diagnostic_full.json) | `fde36ec20bca7bab5e0bb27b10e181eca5e79cdfb11049f012135733543d376f` |
| W3 | [w3_risk_gate.json](../outputs/phase4_6/w3_risk_gate/w3_risk_gate.json) | `cbebf7ba457be905e0d40d4f3c0cbacc0e99dae398d1b7f05ca4eafa1f7de244` |
| Route A | [route_a_results.json](../outputs/phase4_6/route_a_reveal/route_a_results.json) | `567cb65776679602ec3bd2bf0fe49c9ebf75e5b49df6bf49e2263d2214655d16` |
| H5 | [h5_test.json](../outputs/phase4_7/h5_realizable_reveal/h5_test.json) | `543f6fdb2eca6eba753b3d52f65a8a9537ca592086740407de0a99185476bbfc` |
| H6 | [h6_test.json](../outputs/phase4_7/h6_selective_reveal/h6_test.json) | `56ac97db2378599fbb786a4e1b4133ea0852cc2092a6c25d8d51d53a5c6cafc2` |
| H7 | [h7_test.json](../outputs/phase4_7/h7_adaptive_reveal/h7_test.json) | `93c429ef86710e45eb512319aa0ce66df74fb87ced8ba9f751ab64f874772980` |
| L1 derived | [final_summary.json](../outputs/phase4_8/deployment_validation/final_summary.json) | `b5bd8fbc36f94bc1ab33e79efcc2bc02cfad1886466818bc7e195666f9358e02` |
| R0 L1 provenance | [l1_provenance_status.json](../outputs/phase_r0/evidence_repair/l1_provenance_status.json) | `baf46a74815304d4e2de206014d7e45f9281224b5333d8a893317c1a299f0aa3` |
| L2-S | [l2_final_summary.json](../outputs/phase4_8/deployment_validation/l2_final_summary.json) | `bba0fd87c712ba64590a77a002b1a050883b5c6f66b81acd58e61448af219fdd` |
| L2 canonical environment | [l2_environment_manifest.json](../outputs/phase4_8/deployment_validation/l2_environment_manifest.json) | `c75a2a645723881980f12104ff1eb038a92247ab93af37857436a737209de7a1` |
| L2 superseded environment | [l2_environment_manifest.pre_r0_SUPERSEDED.json](../outputs/phase4_8/deployment_validation/l2_environment_manifest.pre_r0_SUPERSEDED.json) | `9b5d67c6ded926e909f06d5848a6bc3434e2b875fce044183a628a918322c50c` |
| P10-I1 | [p10_i1_results.json](../outputs/phase4_10/p10_1a_substrate/p10_i1_results.json) | `65689ae979539e392bb682fd57ac12b3a1c1966b245982b51a90b9b8c255ad7a` |
| P10-1C | [p10_1c_pilot_results.json](../outputs/phase4_10/p10_1c_pilot/p10_1c_pilot_results.json) | `26c53c5861f628beba2e0c075b3ae60af971ed4047108e26389d557e1ae196a7` |
| P10-1D | [p10_1d_timing_results.json](../outputs/phase4_10/p10_1d_timing/p10_1d_timing_results.json) | `9412573224f8ffe66b13530fec7c877996cadec74a37c97847ee898870c3ac2f` |
| P10-1E | [p10_1e_readiness_test.json](../outputs/phase4_10/p10_1e_admissibility/p10_1e_readiness_test.json) | `6a61750a9ff99de68621f101d8060401e76846a8af54b6b61b9f1e7f50635b2a` |
| P10-1F | [p10_1f_scheduler_breakdown.json](../outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.json) | `5df48d65141af8d574e3bc64386593e88ed5ee1ad139fcd1ad0743fb8420f164` |
| R0 I1 strengthened | [p10_i1_strengthened_results.json](../outputs/phase_r0/evidence_repair/p10_i1_strengthened_results.json) | `54bf6e25b10b5a921bee846dbd9de44295c3bc7c8ff1640f2c3a7ef54b890f24` |
| R1 | [r1_concurrent_pipeline_results.json](../outputs/phase_r1/concurrent_pipeline/r1_concurrent_pipeline_results.json) | `ff3f1f39cceada8ab77a17ae5f578ac5da0fe9a90d72a4715f8f450721ffb417` |
| R2-E0 | [r2_e0_results.json](../outputs/phase_r2/e0_event_bridge/r2_e0_results.json) | `23e8eb9360d71efcf98b88ef267e8107fe28007c1ee724e9e98c5d048c7b6329` |
| R2-C0 | [r2_c0_results.json](../outputs/phase_r2/c0_compiled_equivalence/r2_c0_results.json) | `2acd0f4b933c3409c506566a1bbdb27011de87c06887ad2baf2c356ea54f1d66` |
| R2-F0 | [r2_f0_results.json](../outputs/phase_r2/f0_integrated_ready_commit/r2_f0_results.json) | `d3666b27eee8eba345eb31c5692154b6b12d22493b8e298f9d0e7c59dc7a4cbc` |
| R2-O0 | [r2_o0_results.json](../outputs/phase_r2/o0_device_overlap/r2_o0_results.json) | `d06b2756afc1895cb54bd6dc523b847399fdce0480f55f51d360f77ab7fbba92` |
| R2-O1A | [r2_o1a_results.json](../outputs/phase_r2/o1a_device_scheduling/r2_o1a_results.json) | `219fbe797704d1313e48eed455a4c5d57a1c4511c3ebe107cf75cb3c21e05673` |
| R2-O1B | [r2_o1b_results.json](../outputs/phase_r2/o1b_transport_interventions/r2_o1b_results.json) | `68a80d29bac00b8dc8fdc492a08bae94a69002cd86dc3037e4db8fecc20ec793` |
| R3-A0/C0 | [r3_a0_c0_results.json](../outputs/phase_r3/a0_c0_variable_alltoallv_v2/r3_a0_c0_results.json) | `64b63b39946d33c77f0fc015b783d289f747c68745ea1546eff28bfe15ed6034` |
| R3-P0 | [r3_p0_results.json](../outputs/phase_r3/p0_pilot/analysis_v3/r3_p0_results.json) | `0cba6e4c7b77e2f534955202f96335bb70b81681e17c21172c216fc2c693bf95` |
| R3-F0 | [r3_f0_results.json](../outputs/phase_r3/f0_formal/analysis/r3_f0_results.json) | `dcc98501366826b0b34f58979576d3e8991f9c5d7530506051e954478ead218d` |
| R4-A0/C0 | [r4_a0_c0_results.json](../outputs/phase_r4/a0_c0_full_moe/r4_a0_c0_results.json) | `4f38f08bf7cc38e5d7da2012b1065c94f4d4a089286ce2235779fe5322364421` |
| R4-P0 | [r4_p0_results.json](../outputs/phase_r4/p0_full_moe_pilot/r4_p0_results.json) | `0c7fe610eb566ce239512614fccab9636d06a2982aaea0789224f95e8b37d4ba` |
| R4-F0 | [r4_f0_results.json](../outputs/phase_r4/f0_full_moe_formal/r4_f0_results.json) | `ac29d4dac11714aa28ed30056ea7c1a9c867ba631a1f662bd4deb84d0145cefa` |

R4 raw primary evidence hashes:

- [R4-P0 raw host JSON](../outputs/phase_r4/p0_full_moe_pilot/r4_p0_primary_host.json):
  `15f5ccb661f1699228c9ec2296be24ba48a8ebd956fa7654dd10f313886e870e`
- [R4-F0 raw host JSON](../outputs/phase_r4/f0_full_moe_formal/r4_f0_primary_host.json):
  `40f77a7e02d01f4b7abdcd9c412f18d73f3929962c5f31d133d683d05c01b675`

主要实现入口 / Main implementation entry points:

- [compiled_event_driven.py](../rlccl/scheduling/compiled_event_driven.py)
- [event_bridge.cpp](../extensions/r2_event_bridge/event_bridge.cpp)
- [reference_a2av.py](../rlccl/transport/reference_a2av.py)
- [reference_full_moe.py](../rlccl/transport/reference_full_moe.py)
- [run_r3_f0_formal.py](../scripts/run_r3_f0_formal.py)
- [run_r4_f0_full_moe.py](../scripts/run_r4_f0_full_moe.py)
