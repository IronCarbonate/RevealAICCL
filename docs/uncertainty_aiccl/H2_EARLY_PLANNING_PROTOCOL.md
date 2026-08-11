# Phase 4 / Gate 2：提前规划与在线补救协议

日期：2026-08-01  
状态：Main protocol return；等待 `P4-SUP-PROTOCOL-RETURN-001`。在 Supervisor 明确 `ALLOW / NO VETO` 前，RED、生产实现与正式实验均为 `HOLD`。

## 1. 研究对象、目标与停止边界

本阶段只检验 H2：在完整 demand 尚未揭示时，使用 Phase 3B 已验证的 prediction-free ordinary ambiguity recipe，为**已经精确揭示的真实 token**选择短公共 schedule prefix，并在 reveal 后从当前真实 state 做 residual repair，是否优于等待完整 demand 与只看当前 reveal 的方法。

本阶段的 authoritative execution path 是 Phase 1 的 NumPy reveal-aware token proxy 与 `commit_proposal()`，不是 legacy Torch AICCL decoder/solver。所有结果标题、manifest、summary 与报告必须写作 **AICCL uncertainty scheduling proxy / Gate 2 evidence**。即使 H2 PASS，也不得外推到 legacy model、Torch solver、真实部署或完整生产 AICCL 性能。legacy decoder 仍含 truth-dependent 大候选 pruning，Phase 4 模块禁止导入它、Torch、Moment action policy 或旧 partial-demand evaluator。

本阶段不进入 Phase 5 rolling Pareto、Phase 6 三时间尺度或 synthesis optimization；不修改 feasibility checker；不恢复 direct moment-conditioned action policy；不把 Phase 3B traffic-space support PASS 写成 scheduling-gain。

Phase 3B 只冻结并传入：

- selector `boundary_scenarios`、requested `K=8`；
- calibration radius `0.34327919716983946`；
- 由 Phase 3B frozen fit specs重建、并与正式 manifest `normalizer_digest`逐字节核对的 fit-only descriptor normalizer；
- ordinary narrow construction view、observation reconciliation 与 provenance规则。

Phase 3B oracle support、truth-nearest metric、formal test truth/raw rows、summary派生指标与 evaluator callback 永久不得进入 ordinary planner。十个正式 artifact没有可直接复用的scenario matrices；每次需要新plan时，必须从该Phase4 sequence的 `X_{t-32:t-1}` 与当前 demand observation重新构造ordinary support。

## 2. Fresh whole-sequence corpus 与所有 seed

### 2.1 固定索引

family 顺序精确为：

```text
0 regime_switching_long
1 stochastic_volatility
2 rare_shock_recovery
3 hotspot_random_walk
4 same_moments_different_dynamics
```

base seeds顺序为 `(642,742,842)`，`seed_index={642:0,742:1,842:2}`。split顺序与索引为 `{fit:0, validation:1, test:2}`。same-moments variant顺序为 `(smooth, random_switching, long_regime, shock_recovery)`。

对每个 `family_index, seed_index, split_index`：

```text
record_index = family_index*9 + seed_index*3 + split_index
actual_seed = base_seed + family_index*1_000_000 + split_index*10_000
variant_index = (seed_index + split_index) mod 4
sequence_id = p4-{family}-base{base_seed}-{split}-seed{actual_seed}
```

共 `5*3*3=45` 条完整sequence。每条length 256、4 nodes、mean 2.0、std 1.5、max entry 8、calibration candidates 1、Rear4GPU；除length/seed/ID外与Phase3B canonical generator config相同。checkpoints顺序与索引为 `(32,96,160,224)`；每个checkpoint `t`严格只读 `X[t-32:t]`，不读 `X_t`。

reveal mode顺序与索引为：

```text
0 random_entries
1 source_totals_first
2 source_destination_totals_first
3 partial_shards
4 time_based_arrival
```

对每个 split record/checkpoint/mode：

```text
reveal_seed = 202608010 + record_index*1000 + checkpoint_index*10 + mode_index
construction_seed = reveal_seed + 1_000_000
```

所有ordinary planner/tie-break均确定性，无planner RNG。H1 MLP/model seed固定 `20260731`；bootstrap seed固定 `20260801`。

### 2.2 Digest exclusion

formal manifest必须保存并核对：

- H1 manifest SHA-256 `C702D8CEA33BCEC805FA0AB4B1EEA58C7E0BCBF6AAEF697E01523BB86D65B48C`；
- Phase3B manifest SHA-256 `DF8218052A635A683CE0CA848BB31171C740A4FC9C8E31DDB764BB60F2DEE527`；
- 两旧manifest中的全部sequence digest集合；
- 新45条完整record与digest。

新45个sequence ID/digest必须彼此唯一，且与两旧集合交集为空。任何collision、缺record、跨split重复或片段跨split均为hard error。fit只拟合H1 point comparator；validation只选robust config和Wait/Partial primary comparator；test只运行一次。

## 3. 两级 observation 能力与 cadence API

现有 `PartialObservationState` 含family/sequence identity，不能直接传ordinary planner。Phase4新增 immutable `SchedulingObservationView`，只包含：stage、ratio、exact observed entries/mask、aggregate constraints、公开topology、当前state version，以及已揭示token的局部ordinal/source/destination/current holders。局部ordinal按当前 `revealed_tokens` tuple位置确定；ordinary planner不接收原始 `TruthTokenId.opaque_value`、family、sequence_id、world、manifest、future reveal、generator metadata或oracle。

trusted executor私下保留 `ordinal -> TruthTokenId` 映射，只能把planner选中的当前revealed ordinal绑定为 `TransferAction`；真实API需要字符串时使用 `str(token_id)` 或 `token_id.opaque_value`，不存在 `token_id.value`。对仅改变family/sequence ID/opaque token名称的counterfactual，ordinary结构动作必须在ordinal alpha-renaming后相同；opaque ID不得进入Q、tie或语义分支。

新增 evaluator-owned public API：

```text
DemandRevealProcess.observation_for_stage(stage: int) -> fresh PartialObservationState
```

它只选择已有冻结ratio/mask语义，并每次从当前world重建holders/state version；不修改checker。Phase4 execution slot `s`使用：

```text
stage(s) = min(floor(s/4), 4)
```

slots 0..3/4..7/8..11/12..15/16+分别为stage 0/1/2/3/4；full reveal在slot16前到达。ratios固定 `(0,.25,.5,.75,1)`，五mode相同。每slot fresh full observation只给trusted executor/checker；ordinary planner只收由它sanitize出的view。

## 4. 可执行边界与容量

scenario只评分公共候选；未揭示truth token、`ScenarioTokenId`和ratio0 fictional demand不可执行。ratio0 executed transfer必须0；第一版不做或计入topology preparation。

所有actual commit调用未修改的 `commit_proposal(world, fresh_full_observation, proposal)`。任何非法proposal令episode `legality=false`，不得drop-and-continue。checker用raw float capacity/limit作最终权威比较。

planner packing把每slot atomic action budget冻结为：

```text
edge_units(e) = floor(raw_edge_capacity(e))
group_units(g) = floor(raw_group_limit(g))
```

raw值必须finite/nonnegative；`<1`为0，`1.5`可承载1个atomic action。每batch整数edge load `<=edge_units`，重叠shared group中所有相关edge load之和 `<=group_units`。checker随后仍按raw float复核。

## 5. 路径、候选与一批动作

usable edge须 `edge_units>=1`，且其所属每个shared group均 `group_units>=1`。在usable directed graph上，以hop数为距离；canonical path精确复用Phase3B `_shortest_path_edges` 的语义：每个node的outgoing邻接按 `(edge_dst, edge_index)`升序，BFS第一次发现destination时保存的parent edge序列即唯一path。不得再用edge-index-first重排。parallel edge以edge index区分；unreachable距离为`+inf`，该OD不产生candidate，计入`unreachable_od_count`。toy必须包含 `(edge=9,dst=1)` 对 `(edge=2,dst=3)` 的冲突tie以锁定destination-first。

对revealed token当前holder集合 `Holders` 与真实destination `d`：

```text
d_pre = min_h dist(h,d)
d_post(a) = min(dist(edge_dst(a),d), d_pre)
```

候选action必须：source在Holders、target不在Holders、edge usable、`d_pre` finite且大于0、`d_post=d_pre-1`。这保证加入target holder后全局最近holder距离严格减少1。一个batch每token至多1个action；一个token已到destination则不候选。tie使用 `(local_token_ordinal, edge_index)`，不使用opaque ID。packing与最终checker共同覆盖duplicate、source possession、destination possession、capacity/group。

`wasted_executed_actions`是在最终truth下未使上述全局最近holder距离减少1的已commit动作；按冻结candidate应为0，但仍从before/after state实测并由Gate检查，不硬编码。

## 6. Scenario residual load 与 Q 数值合同

### 6.1 当前 residual scenario

每个reconciled scenario matrix先按canonical paths投影。若一条path在同一shared group命中多个edge，group load按edge次数累加，与Phase3B group-coefficient语义一致。任何scenario在positive OD上没有usable path时立即fail closed：不忽略load、不产生普通episode row、不发布formal目录，data status为HOLD；toy/tamper测试必须命中此分支。

对每个已揭示truth token，ordinary planner只用公开source/destination/holders：从scenario的原始source→destination canonical path减去1份load；若token尚未完成，再从使距离最小、node ID最小的当前holder沿canonical path加回1份load。reconciliation保证scenario含足够revealed count；出现负load或不一致即fail closed。未揭示需求仍从scenario source出发。所得edge/group projected load是当前plan的**static residual heuristic**；在同一次H-slot simulation内不随假想action扣减，直到下一次真实replan才重建。

### 6.2 Score

令 `cap_units=max(edge_units,1)`、`limit_units=max(group_units,1)`。对candidate `a`位于edge `e`：

```text
criticality_k(e)
 = residual_edge_load_k(e)/cap_units(e)
   + sum_g[I(e in g)*residual_group_load_k(g)/limit_units(g)]

Q_k(a)
 = 1.0 + 0.25*criticality_k(e) - 0.05*d_post(a)
```

正criticality项明确表示“优先推进未来静态负载高的瓶颈工作”，不是避开拥塞；这是预注册heuristic，结果后不得改符号。若结果失败可归因E，不可事后重写Q。

support weights先以float64归一化；`mu=sum(w_k Q_k)`，population variance `v=sum(w_k*(Q_k-mu)^2)`，`std=sqrt(max(v,0))`，`robust_score=mu-lambda*std`。ratio1 requested K8但actual K1合法，std=0。所有数值必须finite；score差绝对值 `<=1e-12`视为tie，再按 `(local_token_ordinal, edge_index)`。每simulated slot重新生成candidate/holder进度，但使用同一static scenario load。

## 7. Prefix plan 与 recourse 状态机

`H in {2,4,8,16}`，`P in {1,2,4,8}`，强制 `2P<=H`；exact allowlist优先，`P=H`非法。合法pair精确为：

```text
(2,1),
(4,1),(4,2),
(8,1),(8,2),(8,4),
(16,1),(16,2),(16,4),(16,8)
```

共10 pair；与 `lambda in {0,.5,1}`形成30 configs。一次plan在planning-only holder copy上模拟至多H slots，每slot按 `(-score, ordinal, edge)` packing；stored prefix保留前 `min(P, planned_batches)`批。P8在L4下每stage最多执行4批，额外suffix预期在stage change被替换；它不是独立latency发现，validation tie固定优先更小P。

`PrefixPlan` immutable，含origin stage/state version、revision、support/config digest、ordered batches与每批precondition。executor状态含append-only executed ledger、remaining suffix、revision与wait latch。

每slot顺序精确为：

1. cooperative deadline check；
2. 得到fresh full observation与sanitized scheduling view；
3. 若new stage：强制作废旧revision，记录discarded suffix，重建support与plan，`reveal_replan_events +=1`（slot0 initial plan不计replan）；
4. 否则若prefix刚耗尽且world未完成：重建support与plan，`exhaustion_replan_events +=1`；
5. 否则若下一batch的stage/revision/公开precondition不满足：作废suffix并重建，`invalidation_replan_events +=1`；
6. 若plan为空，首次进入same-stage wait latch时同时 `no_common_action_events +=1` 与 `fallback_events +=1`；fallback exclusive timing只覆盖“确认plan为空后的suffix清理、latch/counter设置和wait Proposal封装”，不包含先前construction/selection/synthesis；在stage/state不变的latch期间重复wait不再加counter或fallback timing，stage改变或外部state改变才清latch；
7. 只取第一batch，用本slottrusted `ordinal->TruthTokenId`映射重新封装fresh `Proposal`，再由checker提交；绝不复用旧observation对象；
8. commit成功后append executed ledger、移除batch；已执行的同一action identity `(TruthTokenId, edge_index, plan_revision, batch_index)`不可撤销或replay，但同一truth token可在后续slot/revision沿不同edge继续合法多跳传播；
9. 若checker拒绝，episode illegal并立即停止，不做fallback掩盖。

fallback只指plan construction的可预期无candidate/空suffix路径，并严格按step6首次进入latch计一次；它丢弃未执行suffix并wait，不放宽checker。没有“deadline接近”模糊阈值：只有monotonic clock达到deadline才触发wall timeout并停止。`true_replan_events`为reveal+exhaustion+invalidation三类之和；三类互斥按上述优先级计数。

ambiguity/support只在initial plan、stage change、prefix exhaustion或invalidation时重建；每slotfresh execution observation不等于每slot重建support。同stage只允许在同一method episode内、相同demand-observation digest与state residual digest下复用immutable support；不得跨stage/coordinate/method共享mutable cache。state version不进入demand observation digest，但公开holders产生独立residual-state digest。

## 8. 九种方法与准确参数

所有方法共享同一candidate/path/packing/checker；direct scheduler不读Q，按 `(ordinal,edge)`贪心packing并每slot重算。

1. `full_information_lower_bound`：现有provable、nonexecutable lower bound；role=`lower_bound`、uses_oracle=true、executable=false；algorithm time在E2E reference中固定0，只作regret基准。
2. `full_information_executable_reference`：slot0开始每slot获得fresh full observation，使用direct scheduler；无prefix/suffix，H=P=K=0、lambda=0；role=`executable_reference`，不声称数学最优。
3. `wait_until_known`：stage<4只wait；stage4起每slotdirect scheduler；H=P=K=0。
4. `partial_current_only`：每slot只用当前revealed token与direct scheduler；H=P=K=0。
5. `long_term_mean_point_plan`：32 history逐entry float mean→`np.rint`(banker's rounding)→clip `[0,8]`→diagonal置0→用当前observation `reconcile_candidate`；K1、lambda0，使用validation胜出的同一H/P与完整prefix状态机。
6. `previous_value_point_plan`：`X_{t-1}`复制后reconcile；K1、lambda0，使用同一winner H/P。
7. `h1_best_point_plan`：§9定义的MLP point matrix；K1、lambda0，使用同一winner H/P。
8. `scenario_robust_prefix`：Phase3B boundary/K8 support与validation winner H/P/lambda；唯一ordinary H2候选。
9. `oracle_scenario_robust_reference`：truth+至多7个ordinary recent candidates的truth-assisted support，使用ordinary winner H/P/lambda；role=`oracle_ceiling`、uses_oracle=true、`reference_kind=truth_assisted_support_ceiling_not_proven_performance_bound`。它是信息reference，不保证performance upper bound，不参与selection、ordinary cache、comparator或Gate。

validation对Wait与Partial的primary tie：先比15-sequence equal mean E2E；差绝对值 `<=1e-12`固定选择Partial。test paired improvement统一为 `comparator_value - robust_value`，正值表示robust更好。Gate仍同时检查Wait和Partial，primary只用于seed/family分解。

## 9. H1 best point comparator 的唯一重建

复用 `HistoryPredictorSuite`/`RecentHistoryMLP` 现有H1 recipe：recent_steps=8、model seed20260731；MLP固定 `hidden_layer_sizes=(32,), activation=tanh, solver=adam, alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=80, shuffle=true, early_stopping=false`，其他sklearn defaults由source SHA绑定；只使用`recent_history_mlp`输出。fit universe是15条fresh fit sequence的每个 `t=8..255`，共 `15*248=3720` examples；输入/target summary与standardizers只由fit examples产生。model state、H1 source SHA、fixed config、fit sequence digests、fit example count、target mean/scale、training wall/cpu time与canonical model-state digest写入artifact。任何nonfinite prediction/state hard fail。

summary顺序精确复用 `summary_vector`：total；4 source loads；4 destination loads；hotspot strength；sparsity；Rear4GPU bandwidth-group loads。group coefficients复用Phase3B canonical path语义并核对digest。hotspot destination categorical不在continuous vector中。

在每次point plan时，先用当前32 history与fresh-fit standardizers得到MLP predicted summary。unknown stages的候选是当前 `build_empirical_ambiguity_set` 的32个reconciled ordinary pool；ratio1必须保持Phase3B actual-K1 truth singleton，只在该actual pool内选择，不得为MLP绕过singleton。对每个actual-pool矩阵计算相同summary vector，距离为：

```text
sqrt(mean(((candidate_summary-prediction)/fit_target_scale)^2))
```

scale小于1e-8的维度在fit时置1。距离差 `<=1e-12`时取更大的history offset（更recent），仍tie取更小pool index。prediction不clip；candidate已由physical/reveal reconciliation保证合法。无法完整重建即HOLD，不得把previous-value重命名best predictor。

## 10. Validation config 与 exact run universe

每split coordinate数：`15 sequences * 4 checkpoints * 5 modes = 300`；五ratios/stages属于一个online episode，不另乘5。

validation exact method-config rows：

- `scenario_robust_prefix`：300 coordinates × 30 configs = 9,000；
- `wait_until_known`：300 × 1 = 300；
- `partial_current_only`：300 × 1 = 300；
- 合计 exact 9,600 rows。

其余reference/point methods不参与validation，避免重复运行。30个robust config先按每个sequence的20 coordinates等权聚合；选择顺序：15-sequence equal mean E2E latency、mean sequence CVaR95、mean total online time、较小H、较小P、较小lambda。test不得改变winner。

test exact rows：300 coordinates × 9 methods = 2,700。每个method episode使用fresh world/planner state，不跨method共享scenario cache。method顺序由coordinate digest对固定九方法列表做cyclic rotation；toy另做reverse-order，alpha-equivalent scientific fields必须相同。fit模型可作为immutable read-only state复用，但每个H1 episode实际inference计时。

## 11. 时间、completion 与 timeout

使用 `time.perf_counter_ns()` monotonic clock。`time_limit=80` slots；初始complete记0，slot `s` commit后首次complete记`s+1`，未完成记81。cooperative online deadline为toy 5s、formal 10s：从method-specific inference/construction前开始，到complete/illegal/timeout后结束；common manifest/world reconstruction和进程启动不计online algorithm time，但另记runner wall。

第一版是**cooperative deadline，不是可抢占hard watchdog**：在每个model/construction/selection/plan/slot/repair/fallback/checker边界前后检查；若单个NumPy/sklearn call越过10s，只能在其返回后标记并停止，不声称中断该call。触发时 `wall_timeout=true`、`discrete_timeout=true`、completion=81；保留已执行prefix与已完成exclusive timing，未完成component为0；total_online取实际deadline区间到检测停止的elapsed，可大于10s。formal outer window只在toy实测后由Supervisor admission冻结，任何不可返回阻塞使formal fail closed/HOLD。

exclusive timing components为：`h1_inference`、`ambiguity_construction`、`support_selection`、`prefix_synthesis`、`recourse_repair`、`fallback`、`checker_commit`。同一ns不能进入两component；repair中的construction/selection分别进对应component，`recourse_repair`只计exclusive orchestration。`total_online_ns`由整体区间实测，不由component相加；`unattributed_ns=max(total-sum(components),0)`。offline MLP fit wall/cpu time单列model artifact，不进入online主终点，依据是部署前只训练一次；但报告不得隐藏。

`slot_duration_ms=1.0`；`end_to_end_latency_ms=completion_slots*1.0+total_online_ns/1e6`。`end_to_end_regret_ms=method_e2e-lower_bound_slots*1.0`，不加入lower-bound自身synthesis time。另报scheduling-only与每个comparator break-even slot duration。

first action不存在记80。full reveal slot固定16。字段 `reveal_lead_lag_slots=completion-16`保留signed值：负数表示full reveal前完成，未完成为65。wall timeout后的prefix/repair指标按已发生事实保存。

## 12. Phase4 专用指标

不得复用Phase1同名surrogate。每episode至少记录：completion、lower-bound regret、E2E latency/regret、first action、signed reveal lead/lag；planned/executed prefix batches/actions；discarded unexecuted batches/actions；wasted executed/unexecuted actions；reveal/exhaustion/invalidation/true replan events；residual repair actions（slot>=16的commit）；no-common/fallback；unreachable OD；legality/illegal reason；discrete/wall timeout；七类exclusive timing、unattributed/total online/runner wall。

`wasted_unexecuted_actions`等于被stage/invalidation/exhaustion替换的suffix action数，不是scenario L1。prefix exhaustion时suffix为空，因此discard 0。所有counter由事件ledger重算，不能直接相信summary。

## 13. 聚合、tail 与统计

每条test sequence有20 episodes（4 checkpoint×5 mode），排序为checkpoint index主、mode index次。对每method/sequence：mean/median/p95/p99/CVaR95从20个episode E2E值等权计算。排序值 `z[0..n-1]`时，higher quantile `q_higher(p)=z[ceil(p*(n-1))]`；CVaR95为从`ceil(.95*(n-1))`到末尾的均值。completion slots同样报告。Gate tail为15个sequence tail统计的等权mean；可另报pooled300描述值，但不得当独立样本。

paired sequence delta为同一sequence的comparator sequence-mean E2E减robust sequence-mean E2E。`family_delta(f)=mean_{3 base seeds} sequence_delta(f,b)`；`base_seed_delta(b)=mean_{5 families} sequence_delta(f,b)`，不得从已聚合family delta反推seed。若family delta<0，相对退化：

```text
(robust_family_mean - comparator_family_mean) /
max(abs(comparator_family_mean), 1e-12)
```

family-stratified bootstrap共10,000次：每次在五family各自3条sequence内有放回抽3条，合并15个delta取等权mean。排序bootstrap值后95% CI用higher index `ceil(.025*(B-1))`与`ceil(.975*(B-1))`。不把episode/config/scenario当独立样本。

依赖描述：对每method/sequence/mode，按四checkpoint计算completion。非constant series的 `rho_l=sum_{i=0}^{n-l-1}(x_i-xbar)(x_{i+l}-xbar)/sum_{i=0}^{n-1}(x_i-xbar)^2`；constant series令所有rho=0。positive-sequence ESS从lag1到3逐lag扫描，遇第一个`rho_l<=0`立即停止，否则累加；`ESS=clip(n/(1+2*sum accepted rho),1,n)`。报告lag1 ACF及75个series ESS的mean/sum，不称Geyer pair截断，不用于虚增n。

正式15/15 test sequences、每条9方法×20episode必须完整存在。illegal与timeout是有效失败结果，仍以completion81/E2E入统计，不可丢弃。缺row、duplicate、schema/artifact/environment中断使data status HOLD/fail-closed；不得挑剩余10条PASS。

## 14. H2 Gate、FAIL/HOLD优先级

在Supervisor前，summary只能写conditions1--8与 `gate_status=PENDING_SUPERVISOR`；Core/Main不得写Condition9。

H2 PASS需全部满足：

1. robust相对Wait与Partial两者的15-sequence paired mean E2E improvement均>0，且各自bootstrap95% CI lower>0；
2. robust相对两者的mean sequence CVaR95均不高，p95/p99完整报告；
3. 相对validation冻结primary comparator，3/3 base seed正、至少4/5 family正；任一负family相对退化<=10%；
4. ordinary与executable methods legality精确100%，所有actual action走原checker；
5. robust discrete/wall timeout率分别不高于Wait与Partial；
6. 1ms/slot主E2E已含online overhead，收益不能只来自scheduling-only；
7. exact15/15 sequence与完整2,700 test rows有效，收益不来自事后tiny bucket；
8. fresh exclusion、ordinary/oracle能力、exact artifact/schema/digest、raw→sequence→Gate、focused/full tests全部通过；
9. Supervisor独立复核给出Condition9 PASS/NO VETO。

conditions1--8任一明确FAIL，则data Gate FAIL优先于tail/environment HOLD。只有无明确FAIL、但外部环境使完整证据无法产生时才HOLD。legality<100%、timeout增加或overhead抹去收益本身是Gate FAIL；若仍宣称PASS或绕过边界才升级为Supervisor VETO。

FAIL后Main按A--G归因：A reveal latency短；B无公共安全动作；C replan overhead高；D ambiguity迁移不足；E objective错误；F只能做preparation；G提前信息无价值。只有质量收益存在但C抵消时，才可另向用户申请synthesis优化。

## 15. Exact formal artifacts、schema 与重算

正式目录精确八文件：

```text
manifest.json
h1_best_point_model.json
raw_validation_metrics.csv
raw_test_episode_metrics.csv
raw_test_sequence_metrics.csv
raw_test_execution_events.csv
raw_timing_metrics.csv
summary.json
```

exact row counts：validation 9,600；test episode 2,700；test sequence 135；timing `2,700*8=21,600`，八component为七exclusive加`unattributed`。execution events为结果依赖的variable universe：每个2,700 episode至少`episode_start/episode_end`两行，`event_index`必须从0连续无缺口；final manifest保存实际exact count、primary-key universe与digest，read-back按状态机重建episode counters。JSON无row count。

所有CSV共同前缀列：

```text
schema_version,split,coordinate_id,sequence_id,family,base_seed,
sequence_digest,checkpoint,checkpoint_index,reveal_mode,mode_index,
reveal_seed,topology_digest,config_digest,method,role,uses_oracle,
executable
```

`raw_validation_metrics.csv`在共同列后精确为：

```text
horizon,prefix,requested_k,actual_k_min,actual_k_max,risk_lambda,
completion_slots,total_online_ns,end_to_end_latency_ms,
legality,discrete_timeout,wall_timeout,
row_digest
```

role只允许ordinary；methods精确只允许 `scenario_robust_prefix,wait_until_known,partial_current_only`。H/P/K不适用使用整数0，不使用NA。

`raw_test_episode_metrics.csv`在共同列后精确为：

```text
reference_kind,horizon,prefix,requested_k,actual_k_min,actual_k_max,risk_lambda,
completion_slots,lower_bound_slots,oracle_regret_slots,
total_online_ns,runner_wall_ns,end_to_end_latency_ms,end_to_end_regret_ms,
first_action_slot,reveal_lead_lag_slots,
prefix_planned_batches,prefix_planned_actions,
prefix_executed_batches,prefix_executed_actions,
discarded_unexecuted_batches,discarded_unexecuted_actions,
wasted_executed_actions,wasted_unexecuted_actions,
reveal_replan_events,exhaustion_replan_events,invalidation_replan_events,
true_replan_events,residual_repair_actions,no_common_action_events,
fallback_events,unreachable_od_count,legality,illegal_reason,
discrete_timeout,wall_timeout,
row_digest
```

nonexecutable lower-bound row：executable=false、legality=true（vacuous）、completion/lower_bound相同、online/timing/prefix全0、timeout按lower bound81、illegal_reason空；其他不适用numeric为0。禁止空numeric/NaN/Inf。

`raw_test_sequence_metrics.csv`共同列的sentinel映射精确为：`split=test`、`coordinate_id=ALL`、`checkpoint=-1`、`checkpoint_index=-1`、`reveal_mode=ALL`、`mode_index=-1`、`reveal_seed=-1`；sequence/family/base_seed/digest/topology_digest为真实值，config_digest为该method冻结test config。其后精确为：

```text
episode_count,completion_mean,completion_median,completion_p95,completion_p99,
completion_cvar95,end_to_end_mean_ms,end_to_end_median_ms,end_to_end_p95_ms,
end_to_end_p99_ms,end_to_end_cvar95_ms,oracle_regret_mean_slots,
total_online_mean_ns,total_online_p95_ns,total_online_p99_ns,
legality_rate,discrete_timeout_rate,wall_timeout_rate,
prefix_executed_actions_sum,discarded_actions_sum,true_replan_sum,
residual_repair_actions_sum,row_digest
```

该表主键为 `(sequence_id,method)`，每row episode_count=20。

`raw_test_execution_events.csv`精确列为：

```text
schema_version,split,coordinate_id,sequence_id,family,base_seed,
sequence_digest,checkpoint,checkpoint_index,reveal_mode,mode_index,
reveal_seed,topology_digest,config_digest,method,role,uses_oracle,executable,
event_index,slot,stage,state_version_before,state_version_after,plan_revision,
event_kind,reason,observation_digest,residual_state_digest,support_digest,
requested_k,actual_k,batch_index,batch_count,action_count,local_token_ordinal,
truth_binding_digest,edge_index,before_distance,after_distance,
commit_legal,elapsed_ns,event_payload_digest,row_digest
```

event kind枚举为 `episode_start,plan_built,suffix_discarded,wait_latch_entered,proposal_bound,batch_committed,action_committed,checker_rejected,episode_end`；`plan_built.reason`只允许`initial/reveal/exhaustion/invalidation`。action之外的ordinal/edge/distance为-1；无digest使用64个`0`；无reason使用`NONE`；bool仍为lowercase。`action_committed`每actual action一行，含trusted truth binding的SHA-256而非opaque明文、before/after全局holder distance；`batch_committed`绑定batch action count；`suffix_discarded`绑定discard reason/batch/action count；`plan_built`绑定support/revision/planned counts，并且只有它的`requested_k,actual_k`可非0。direct/lower-bound事件K均为0；三种point method每个`plan_built`为1/1；ordinary robust为requested8，stage0--3 actual8、stage4 actual1；oracle robust为requested8、actual为truth-assisted support去重后的1..8且stage4必须1。其他event的两个K字段均为0。episode的`actual_k_min,actual_k_max`只能从本episode全部`plan_built.actual_k`重算；没有`plan_built`则二者均0。validation/test read-back逐stage核对event K、support digest与episode min/max；config digest仍只含requested K。episode metrics中的planned/executed/discarded/wasted/replan/residual/fallback/legality必须由该ledger重算，不能信聚合counter。lower-bound也有start/end两行。

`raw_timing_metrics.csv`共同列后精确为：

```text
component,elapsed_ns,row_digest
```

component枚举为 `h1_inference,ambiguity_construction,support_selection,prefix_synthesis,recourse_repair,fallback,checker_commit,unattributed`；每test episode八行。

method registry精确为：

| method | role | uses_oracle | executable | reference_kind |
|---|---|---:|---:|---|
| `full_information_lower_bound` | `lower_bound` | true | false | `provable_full_information_lower_bound` |
| `full_information_executable_reference` | `executable_reference` | true | true | `full_information_feasible_scheduler_not_optimal` |
| `wait_until_known` | `ordinary` | false | true | `ordinary_comparator` |
| `partial_current_only` | `ordinary` | false | true | `ordinary_comparator` |
| `long_term_mean_point_plan` | `ordinary` | false | true | `ordinary_point_comparator` |
| `previous_value_point_plan` | `ordinary` | false | true | `ordinary_point_comparator` |
| `h1_best_point_plan` | `ordinary` | false | true | `ordinary_point_comparator` |
| `scenario_robust_prefix` | `ordinary` | false | true | `ordinary_h2_candidate` |
| `oracle_scenario_robust_reference` | `oracle_ceiling` | true | true | `truth_assisted_support_ceiling_not_proven_performance_bound` |

CSV bool只能lowercase `true/false`；int十进制无小数；float使用Python finite round-trip `repr`；UTF-8、LF、header固定、primary key排序固定。typed canonical grammar为：null=`n;`、bool=`b:0;|b:1;`、int=`i:<decimal>;`、float=`f:<float.hex>;`、UTF8 string=`s:<byte_length>:<bytes>;`、list=`l:<count>:[ordered items]`、mapping=`m:<count>:{UTF8-key-sorted encoded key/value}`；NumPy array编码为dtype.str、shape list、flattened typed data mapping。`row_digest`是除自身外ordered header fields编码的SHA-256。manifest保存协议/source/test/runner/environment hashes、45 records、两旧manifest/exclusion、P3B recipe/normalizer/model digests、method/config universes、schema/row counts、timing/seed/stat constants与各artifact logical/scientific hashes。

`manifest.json` top-level exact keys为：`schema_version,study_name,protocol_sha256,authorized_source_sha256,authorized_test_sha256,runner_sha256,environment,old_manifests,excluded_sequence_digests,sequence_records,families,base_seeds,splits,checkpoints,reveal_modes,reveal_ratios,seeds,topology,phase3b_recipe,h1_model,method_registry,validation_config_universe,selected_config,selected_primary_comparator,timing_contract,statistics_contract,artifact_names,artifact_row_counts,artifact_logical_sha256,artifact_scientific_sha256,integrity_complete,evidence_complete,data_status,gate_status,summary_sha256`。

`summary.json` top-level exact keys为：`schema_version,study_name,integrity_complete,evidence_complete,data_status,gate_status,selected_config,selected_primary_comparator,test_sequence_count,test_episode_count,method_metrics,comparator_evidence,seed_evidence,family_evidence,timeout_evidence,legality_evidence,timing_evidence,conditions_1_to_8,failed_conditions,insufficient_conditions,combined_scientific_evidence_sha256`。

`h1_best_point_model.json` top-level exact keys为 `schema_version,model_name,config,fit_sequence_records,fit_example_count,input_mean,input_scale,target_mean,target_scale,parameter_arrays,model_state_sha256,source_sha256,group_coefficients_digest,fit_wall_ns,fit_cpu_ns`；parameter arrays精确保存dtype/shape/float.hex data，不得pickle。

execution ledger→episode→sequence→selection→conditions1--8→summary是唯一重算链；manifest/summary/episode聚合不得覆盖ledger/raw。RED必须覆盖missing/duplicate/type/domain/nonfinite/role/oracle/seed/digest/selection/bootstrap/timing/ledger联动篡改。staging先写provisional `integrity=false/data=HOLD/gate=PENDING_SUPERVISOR`，完整read-back后写final hashes，再第二次read-back，最后单次directory rename。import/collection/toy不得创建formal目录。

## 16. Test-first 与监督状态机

计划新增：

```text
rlccl/scheduling/robust_prefix.py
rlccl/scheduling/recourse.py
rlccl/scheduling/scenario_adapter.py
rlccl/scheduling/phase4_experiment.py
scripts/run_phase4_early_planning.py
tests/test_phase4_robust_prefix.py
tests/test_phase4_experiment.py
docs/uncertainty_aiccl/H2_EARLY_PLANNING_RESULTS.md
docs/agent_coordination/SUPERVISOR_REVIEW_GATE_H2.md
outputs/phase4_early_planning/
```

Supervisor protocol ALLOW → Core只写两份RED → Main独立验收RED/hash → Supervisor开放production → Core最小实现 → Main focused/full/toy/tamper/runtime → Supervisor formal admission并据toy冻结outer window/hash/destination/staging/process → 唯一formal → Main独立重算/results → Supervisor Condition9/final Gate → Main停止在Phase5前等待用户。任何green不自动开放下一门。

## 17. Supervisor VETO 条件

以下任一触发VETO：scenario/未揭示token执行；ratio0 fictional transfer计收益；修改/绕过checker或静默丢非法动作；ordinary读取truth/private map/future reveal/family-ID/oracle；oracle或旧formal test污染ordinary/confirmatory；paired fresh world/manifest不一致；test调参/挑comparator；executed prefix可撤销或repair从initial重启；少于3seed或episode当独立sequence；漏overhead/伪造hard deadline/单位不可比；把非法、timeout增加、overhead抵消仍判PASS；tiny bucket选择；把lower bound或truth-assisted reference冒充executable/performance upper bound；把proxy结果外推legacy/production AICCL。
