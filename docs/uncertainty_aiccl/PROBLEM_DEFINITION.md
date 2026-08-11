# 不确定流量下的 AICCL：冻结问题定义

## 1. 研究目标

唯一主目标是：**提升 AICCL 在不确定流量条件下的调度效果。**

“不确定”必须发生在决策时刻，而不是只在数据生成或事后评估中存在。环境可以持有真实流量，但普通调度方法在做决定时不能取得尚未揭示的真实信息。

以上为 Phase 0 的历史范围说明。用户已于 2026-07-30 批准进入 Phase 1；本轮只实现本文件第 11 节冻结的 Phase 1 环境、揭示、执行、基线与评估接口，不进入 H1/H2/H3，也不训练新模型。

## 2. 已冻结的路线

以下分支冻结，代码与结果保留用于回归和负面证据，但不继续扩展：

- historical moments 直接进入当前 shard-edge action policy；
- 扩大 `MomentEncoder` 或添加更多 moment feature；
- 直接进入旧 V2/V3/V4；
- 用单 seed、小桶或选择性成功结果重启相同路线；
- 把 decoder/synthesis/cache/pruning/GPU/CPU 优化改成主研究问题。

冻结依据包括：相同当前 `X` 时只改变历史会导致有害 schedule 干扰；moment-only 预测弱于 previous value/recent history；旧 partial-demand 结果没有跨 seed 稳定收益；正式 V1 为 NO-GO。

这只否定 `historical moments -> current fine-grained action`，不否定 uncertainty-aware AICCL。

## 3. 决策时序与信息边界

对完整 traffic sequence 的第 `t` 个问题：

- 世界真值：`X_t`，由环境私有持有；
- 历史：调度前已完成且允许使用的 `X_0, ..., X_{t-1}`；
- reveal stage：`r = 0, ..., R`；
- entry reveal mask：`M_{t,r}`；
- 精确 entry 观测：`O_{t,r} = M_{t,r} ⊙ X_t`；
- 必须满足 `M_{t,r-1} subset of M_{t,r}` 且 `M_{t,R}` 全可见。

`source_totals_first` 和 `source_destination_totals_first` 还会提供 aggregate observation。这些 aggregate 不是精确 entry reveal，不能被伪装成真实目的地 demand；它们只能约束 scenario/ambiguity set，或者支持等待/准备性决策。

一个 stage 的顺序冻结为：

1. 环境推进 reveal，产生新的 policy observation；
2. 调度器只能读取历史、当前 observation、当前已执行后的公开 schedule state 和合法的场景集合；
3. 调度器可等待、规划，或提交只涉及已精确揭示真实 demand 的动作/prefix；
4. deterministic checker 在当前真实 world state 上验证动作；
5. 合法动作不可撤销，更新 world state；
6. 未执行的计划可在下一 reveal stage 被替换；
7. 最终 reveal 后处理全部 residual demand；
8. evaluator 只在决策外使用完整真值计算 completion、regret、legality 和 timeout。

未揭示 demand 可以作为不确定变量参与 scenario planning，但不能：

- 出现在普通 policy 的对象属性、feature、candidate/pruning、cache key 或 metadata；
- 提前获得可执行的真实 destination/chunk binding；
- 绕过 source-has-chunk、edge capacity 或 shared-group checker；
- 被 evaluator 或 oracle 反向泄漏给普通方法。

## 4. 世界状态、观测状态与计划状态

### 4.1 世界状态（仅环境/evaluator）

至少包含：完整 `X_t`、真实 chunk/demand 映射、真实 initial/current possession、已执行动作、剩余真实 demand、topology、capacity/shared-group 约束和时间限制。

现有 `ProblemInstance` 适合继续作为世界真值/确定性评估数据结构，但不适合直接作为普通 policy 的输入，因为它公开 `demands`、`traffic_matrix`、`initial_state`、`C` 和 metadata。

### 4.2 Policy observation

只包含：stage、reveal ratio/mask、已揭示精确 traffic、明确标注的 aggregate、已公开的真实 demand token、当前公开 schedule state、历史 `X_{<t}` 的允许摘要、topology 的静态公开信息。

当 exact demand 尚未揭示时，aggregate 或 proxy 不能生成可执行真实动作。若实现需要固定大小 tensor，padding/mask 也不能暴露真实未揭示 chunk 数；应使用 observation-derived 上限、opaque padding 或 reveal-time ID mapping，并通过反事实测试验证。

### 4.3 Scenario plan

`ScenarioSet` 中的 demand 是假设。计划可以引用 scenario-local token，但只有 reveal 后与真实 demand 显式绑定的动作才可执行。oracle scenario set 必须单独标记，且只能作为上界。

### 4.4 Execution state

已执行动作不可撤销；未执行 prefix 可替换。replan 必须从当前真实 possession/usage 状态继续，而不是从原始 initial state 重启。动作必须先被 reveal-aware validator 证明为有限、非负、二值且只引用已揭示真实 token，再由现有 deterministic feasibility checker 复核 source possession、edge capacity 和 shared-group 约束。

## 5. Phase 1 必需抽象及唯一职责

名称可按代码风格微调，但语义必须存在。

### `UncertainProblemInstance`

- 封装 topology、time limit、sequence identity 和私有 world truth；
- 负责创建 reveal process、公开 observation、接收待验证动作；
- 不向普通 policy 暴露内部完整 `ProblemInstance` 或真值数组引用；
- 提供隔离的 oracle view，仅供上界 baseline。

### `DemandRevealProcess`

- 维护 stage、mask、aggregate 与 reveal event；
- 保证 mask 单调、seed 可复现和最终全部揭示；
- 至少支持 `random_entries`、`source_totals_first`、`source_destination_totals_first`、`partial_shards`、`time_based_arrival`；
- 至少支持 reveal ratios `0.00, 0.25, 0.50, 0.75, 1.00`。

### `PartialObservationState`

- 是普通 policy 的唯一当前 demand 入口；
- 区分精确已揭示 entries、aggregate constraints、unknown mask 和可执行 demand token；
- 不保存 truth 引用、未来 reveal 计划、生成器 latent state 或 oracle 标记值。

### `ScenarioSet`

- 保存 `(scenario, probability/weight, provenance)`；
- 验证 shape、非负性、概率和 observation 一致性；
- scenario 是规划假设，不能直接变成真实执行许可；
- Phase 1 只做容器和 baseline 所需的简单场景，不训练 predictor。

### `RecourseMetrics`

- 记录 completion、oracle regret、reveal wait、recourse count、replanned actions、wasted plan、synthesis/replan time、legality、timeout；
- 保留 sequence、family、seed、topology、reveal stage/mode、method 和原始 paired row；
- 不把同一 sequence 的时刻行当成独立统计样本。

## 6. 现有代码的可复用边界

可以复用但必须隔离真值：

- `traffic_matrix_to_scenario()` / `scenario_to_traffic_matrix()`：truth-side matrix/chunk 转换；
- `TrafficSequence`、long-horizon generators：生成完整 truth sequence；generator latent metadata 仅限审计/evaluator；
- `SlidingMomentEstimator` / history example 工具：只在每条 sequence 内使用已完成的 `X_{<t}`；不得把完整当前 `X_t` 传给 Phase 1 policy feature；
- `TopologyInfo`、`compute_received_chunks()`、capacity/shared constraints：静态拓扑和状态转移；
- `evaluate_schedule()`：决策完成后的 deterministic legality/completion checker；
- 现有 metrics 的 completion/CVaR 计算：可作为基础，但需扩展 recourse/reveal 指标。

不能直接复用为 Phase 1 环境：

- `ProblemInstance` 的完整对象作为 policy input；
- `TrafficSequenceRunner` 在调度前公开完整 `traffic_matrix/demands` 的接口；
- C4 的静态 `PartialDemandObservation` 作为 rolling reveal；
- `SlotDecoder.decode_slot()` 的当前 partial 路径，直到所有 candidate/pruning/feature 分支均证明只读 observation；
- `MomentContext` 当前 z/global 特征使用完整当前 matrix 的调用方式。

## 7. 已确认的关键泄漏风险

1. `SlotDecoder.decode_slot()` 正常 feature 路径使用 `policy_demands`，但大候选剪枝的 `is_demand_score` 读取真实 `demands`；规模增大时会泄漏隐藏目的地。
2. 完整 `ProblemInstance` 的 `C`、`initial_state` 和 chunk 编码可能泄漏总 demand 和 source ownership，即使 destination feature 被置零。
3. 旧 source-total proxy 会生成猜测目的地；若不区分 hypothesis 与 executable demand，会把 imputation 当真值执行。
4. long-horizon `TrafficSequence.metadata` 含完整未来 regime/shock/hotspot 轨迹，只能用于审计。
5. evaluator 合法地读取真值做事后计分，但必须与 policy 调用栈分离。
6. reveal ratio 0、空 chunk 集和等待路径可能迫使实现为了保持 tensor shape 偷偷使用 truth padding。

## 8. Phase 1 baseline 口径

所有方法必须使用同一 `X_t`、topology、reveal process、seed、timeout、evaluator 和 legality checker。

- Full-information oracle：通过隔离 oracle view 读取完整真值，仅作为上界；不能参与普通方法 feature/cache。
- Wait-until-known：在最终全揭示前不执行未知 demand；记录 reveal wait。
- Partial-current-only：只使用当前 `PartialObservationState`，不读历史预测。
- Long-term mean：只使用训练/历史完整 sequence 中的过去数据，不能用当前或未来 test sequence 统计。
- Previous-value：只能使用同一 sequence 的 `X_{t-1}`；sequence 起点有显式 cold-start。

Phase 1 只验证环境和 baseline 语义，不据此判断 H1，也不训练 recent-history/GRU/TCN/quantile predictor。

## 9. Phase 1 强制测试与 Gate

专项测试至少证明：

1. ground truth 与 observation 是不同对象、不同数组，普通 policy 无真值能力；
2. 改变未揭示 truth、保持 observation 相同时，普通方法的输入/candidate/动作在 reveal 前不变；
3. reveal mask 单调，最终全揭示；
4. 五种 mode 和五个 ratio 均有确定性覆盖；
5. 未揭示 demand 不可执行，scenario/proxy 不能解锁执行；
6. ratio 0 和空可执行 demand 正确等待，不崩溃、不 truth padding；
7. 每条 sequence 的 estimator 独立且只读 `X_{<t}`；
8. 所有 baseline 使用相同 truth/reveal/topology/seed/evaluator/timeout；
9. full oracle 只作为上界；
10. 所有实际 schedule legality 为 100%；
11. 强制构造超过 decoder pruning 阈值的 case，证明隐藏 truth 不影响候选集合；
12. generator latent metadata、future reveal schedule、cache key 均不可从 policy API 到达。
13. 负数、非二值、NaN/Inf 和引用未揭示 token 的动作全部被 deterministic reveal-aware validator 拒绝。

只有专项测试通过、现有测试无新失败、原始跳过/环境缺口已记录，并经 Supervisor 独立确认无泄漏后，Phase 1 Gate 才可通过。若 reveal semantics 无法严格定义，必须停止，不得进入 H1 建模。

## 10. Phase 0 边界声明（历史状态）

Phase 0 完成时本文档只是设计冻结，没有新增 uncertainty 业务模块、训练模型或实现 predictor、robust prefix、recourse、三时间尺度或 synthesis 优化。当时进入 Phase 1 仍需用户批准；该批准随后已于 2026-07-30 获得，当前执行状态以第 11 节和任务账本 D-012 为准。

## 11. Phase 1 实现语义冻结（2026-07-30）

用户已批准进入 Phase 1。本节在写测试和实现前冻结 Supervisor 预审指出的歧义；后续代码和测试以本节为准。

### 11.1 两级 reveal 表示

环境把真实 matrix `X_t` 私有展开为 atomic demand token 集合 `A_t`。存在两种 mask：

- entry-level `M^entry_{t,r}`，shape 为 `(V,V)`；
- token-level `M^token_{t,r}`，只存在于环境内部，长度等于私有 atomic token 数。

对 entry-level mode：

```text
O^entry_{t,r} = M^entry_{t,r} ⊙ X_t
```

一个 entry 被 reveal 时，其值（包括真实 0）全部精确可知，且该 entry 的全部非零 token 一起成为 executable token。

对 `partial_shards`：token-level mask 是权威 mask。公开 observation 只包含已经 reveal 的 token 及每个 entry 的已揭示计数下界；不能包含 full-length token mask、总 token 数或未揭示 token 的占位 shape。只有环境确认某 entry 的全部 token 均已 reveal 时，才把该 entry 标为 entry-level exact。所有 mask/已揭示 token 集都必须单调，最终 stage 全部可见。

### 11.2 五种 mode 的 ratio、排序与 stage-0 aggregate

默认 stage ratios 严格冻结为：

```text
(0.00, 0.25, 0.50, 0.75, 1.00)
```

允许调用者提供其他严格递增、首项 0、末项 1 的 ratios，但上述五点必须被测试覆盖。

| Mode | reveal 原子/分母 | 排序 | `r=0` aggregate | ratio 含义 |
|---|---|---|---|---|
| `random_entries` | 全部 `V*(V-1)` 个 off-diagonal entry，包括真实 0 | 由环境私有 RNG 对 entry 做确定性 permutation | 无 | reveal permutation 的前 `floor(r*N)` 个 entry；`r=1` 强制全部 |
| `source_totals_first` | 同上 | 私有 RNG permutation，seed 固定 | 精确 source row totals | aggregate 从 stage 0 起持续可见；exact entry 仍按 `floor(r*N)` reveal |
| `source_destination_totals_first` | 同上 | 私有 RNG permutation，seed 固定 | 精确 source row totals 与 destination column totals | aggregate 从 stage 0 起持续可见；不得用 transportation proxy 冒充 exact entry/executable demand |
| `partial_shards` | 私有 atomic demand token，分母为真实 token 数但该数不进入 public payload | 私有 RNG 对 token 做 permutation | 无 | reveal 前 `floor(r*C_private)` 个 token；公开层只看到 revealed token 集/计数下界，`r=0` 不泄漏 `C_private` |
| `time_based_arrival` | off-diagonal entry | 每个 entry 由私有 RNG 采样 arrival time `u∈(0,1]`，按 `(u, stable-entry-id)` 排序 | 无 | `r` 是当前公开时间阈值；只 reveal `u<=r` 的 entry，实际可见比例不要求恰等于 r；`r=1` 强制全部 |

所有 mode 的 diagonal 是公开的结构性零，但 ratio 分母不包含 diagonal。普通 policy 只能看到当前 ratio/time、当前与过去 observation；不得看到 future ratios 之外的 mask、entry order、arrival times、RNG state 或下一 event。

### 11.3 Public token ID 与 action schema

已揭示真实 demand 使用 `TruthTokenId` 和 `RevealedDemandToken`：

```text
TruthTokenId(opaque_value)
RevealedDemandToken(token_id, source, destination)
TransferAction(token_id: TruthTokenId, edge_index: int)
```

规则：

- `opaque_value` 不编码 global chunk index、总 `C`、未来 token 数、reveal 顺序或完整 matrix shape；
- opaque ID 只在 token 首次 reveal 时由环境签发，环境私下维护 `public token -> truth token` 映射；不得直接复用 truth chunk index；
- token ID 在同一 world 的多个 stage 中稳定，但 future token ID 不可从当前 token 推导；
- observation 只携带已揭示 token 及其当前公开 holders；ratio 0 时 token/action 集为空，不用 truth-derived padding；
- `TransferAction` 只能引用 `TruthTokenId`，edge 必须为公开 topology 中的整数 edge id；
- commit 前验证 action 类型、edge range、同 slot 重复、source possession、destination 未持有、edge capacity、shared group 和当前 stage 的 executable token 集；
- legacy schedule matrix adapter 如存在，必须先拒绝非二维正确 shape、NaN/Inf、负数和非二值值，再进入物理 checker。

### 11.4 Scenario token 与 truth token 分域

scenario planning 使用不同的 `ScenarioTokenId` / `ScenarioDemandToken` 类型，命名空间固定为 `scenario:<scenario-id>:<local-id>`。它与 `TruthTokenId` 不可隐式转换、不可比较为相等，也不能构造可提交的 `TransferAction`。

只有 reveal 后，planner 丢弃/重建假设计划，并用 observation 给出的 `TruthTokenId` 生成真实 action。禁止把 scenario token “绑定”到尚未 reveal 的 truth；oracle scenario 也不能获得例外执行许可。

### 11.5 不可变 paired manifest

统一评测由环境侧 `EvaluationManifest` 控制，至少包含：

- `sequence_id`、family、history provenance；
- truth digest（不含 truth matrix 本身）；
- topology/config digest；
- reveal mode、ratios、reveal seed；
- timeout/time limit、checker version；
- method-independent manifest id。

manifest 为冻结 dataclass/不可变序列，只在 runner/evaluator 持有，不传给普通 planner。每个 baseline 都从同一 manifest 和 evaluator 私有 truth **重建一个独立** `DemandRevealProcess` 与 world state，避免一个方法消费 reveal/RNG/状态后污染另一个方法。逐方法 raw row 必须记录同一 manifest id/digests；oracle 另有 `uses_oracle=True` 标记，但仍使用同一 truth/topology/timeout/checker。

### 11.6 普通 planner 的唯一签名

普通 baseline 只允许以下逻辑签名：

```text
propose(
    observation: PartialObservationState,
    *,
    history: SanitizedHistoryView | None = None,
    scenarios: ScenarioSet | None = None,
) -> Proposal
```

其中：

- `PartialObservationState`、`SanitizedHistoryView` 和 `ScenarioSet` 都必须深拷贝 NumPy 数据并设为只读；
- `SanitizedHistoryView` 只含同一 sequence 已完成的 `X_0..X_{t-1}` 及公开 sequence step，不含当前/未来 matrix、generator metadata 或 truth/process 引用；
- `ScenarioSet` 只能由 history 或当前授权 aggregate/observation 构造；不能持有 world、oracle、reveal process、manifest 或回调/闭包反向引用；
- `Proposal` 只能包含 wait、基于 `TruthTokenId` 的 executable transfer，或单独标记的 scenario-only plan；scenario-only plan 不可传给 commit；
- `EvaluationManifest`、world、oracle view、`DemandRevealProcess`、RNG 和 evaluator 对象都不属于 planner 参数；
- oracle 通过 evaluator 内部的独立 factory/runner 运行，不实现或调用上述普通 planner 签名中的隐藏 oracle 开关。

### 11.7 Phase 1 模块/API 冻结

首版模块职责冻结为：

```text
rlccl/uncertainty/
├── __init__.py
├── observation.py   # TruthTokenId, RevealedDemandToken, PartialObservationState, SanitizedHistoryView
├── reveal.py        # DemandRevealProcess and five reveal modes
├── problem.py       # UncertainProblemInstance; evaluator-private truth/world
├── scenarios.py     # ScenarioTokenId, ScenarioDemandToken, ScenarioSet
├── execution.py     # TransferAction, reveal-aware validator, proposal/commit state
├── baselines.py     # five Phase-1 baselines; planners only receive observation
├── metrics.py       # RecourseMetrics and raw row schema
└── evaluation.py    # immutable EvaluationManifest and paired runner
```

`UncertainProblemInstance` / `DemandRevealProcess` / private truth 不得作为普通 planner 的参数。Phase 1 不导入或调用旧 Torch `SlotDecoder` partial-demand 路径；旧 line-425 truth leak 保留为历史风险并由新路径隔离，不在本阶段扩展旧 decoder。

### 11.8 Full-information oracle 的可证明上界语义

`FullInformationOracle` 在 Phase 1 中不是固定顺序 greedy schedule，也不是普通 planner。它是 evaluator-private 的 **full-information completion lower-bound reference**：completion 越小越好，因此该 lower bound 对应性能上界，只用于计算非负 `oracle_regret`，不生成可执行 action，也不参与普通 policy 调用栈。

令每个 unit demand token 从 source 到 destination 至少沿公开有向 topology 传输。以 unit action 口径把每条 edge 的每 slot 可用容量取为 `floor(capacity_e)`；shared constraints 只会进一步收紧可行域，故在 lower-bound relaxation 中忽略它们仍保持乐观。对可达实例冻结：

```text
LB_path     = max_{X[s,d] > 0} shortest_hops(s,d)
LB_work     = ceil(sum_{s,d} X[s,d] * shortest_hops(s,d)
                   / sum_e floor(capacity_e))
LB_source   = max_s ceil(sum_d X[s,d]
                         / sum_{e out of s} floor(capacity_e))
LB_dest     = max_d ceil(sum_s X[s,d]
                         / sum_{e into d} floor(capacity_e))
LB           = max(LB_path, LB_work, LB_source, LB_dest)
```

空 demand 的 `LB=0`。若任一正 demand 在 unit-capable edge 子图上不可达，或相应 source/destination 没有 unit capacity，则在本阶段的 `T+1` timeout convention 下 oracle completion 记为 `T+1`；若 `LB>T` 也记为 `T+1`。否则 oracle completion 为 `LB`。

正确性依据：任何合法 schedule 都必须满足每个 token 的最短 precedence/path 长度、全网 unit transmission work、每个 source 的首次发出容量和每个 destination 的最终进入容量；忽略 shared constraints 只会放宽而不会收紧可行域。因此任意普通方法 completion 均不得小于此 reference，`oracle_regret = completion - oracle_completion >= 0`。禁止用 `max(0, delta)`、绝对值、删行或更改普通方法结果来伪造非负性。

oracle raw row 必须显式记录 `reference_kind="provable_full_information_lower_bound"`、`executable=False` 和 `upper_bound_only=True`，避免把 lower bound 冒充一条实际 schedule。

### 11.9 Canonical manifest digest 与 timeout 字段

Phase 1 evaluator 冻结三种 canonical digest：

- truth digest：canonical `int64` contiguous truth matrix bytes；
- topology digest：版本、`V/E`、有序 edge list、每条 capacity 的精确 float hex、shared-group edge indices/limit 的 canonical JSON；
- config digest：版本、reveal mode、ratios、reveal seed、`timeout`、`time_limit`、checker version 的 canonical JSON。

`EvaluationManifest.create(...)` 由 evaluator 根据实际 truth/topology/config 构造摘要；`PairedEvaluationRunner` 必须在入口重新计算并拒绝 truth/topology/config 任一不匹配。普通 planner 不接收 manifest 或摘要。

`time_limit` 是 D-022 定义的离散 execution slot 上限；per-method raw `timeout: bool` 表示是否在该 slot 上限内未完成。manifest 的 `timeout` 在 Phase 1 仅作为所有方法共同的外层调用预算 provenance，raw row 名为 `timeout_limit`；本纯 NumPy runner 不得声称已经执行 wall-clock 中断。后续若接入 subprocess wall-clock enforcement，必须保持该配置在 paired 方法间相同。
