# 不确定流量 AICCL：Phase 0 代码审计

日期：2026-07-30（Asia/Shanghai）  
任务：`P0-AUDIT-001`  
角色：Core Research Engineer  
范围：只读审计与现有测试；未实现 Phase 1，未训练模型，未修改业务代码、测试、配置或 checkpoint。

## 1. 审计结论

当前仓库具备可复用的确定性 `ProblemInstance`、All-to-All-V traffic/chunk 转换、history-only 滑窗估计器、slot decoder、确定性 schedule evaluator、长短两套 traffic generator 和 sequence-level 评测工具，但**尚不存在真正的不确定流量环境**。

现有 partial-demand 实验只是“完整 ground truth problem + 被遮蔽/插补的策略特征视图”：完整 chunk action space、真实初始 state、真实 residual demand、真实 state transition 和最终 evaluator 从始至终存在。它没有 reveal stage、单调 reveal 状态、未揭示 demand 的执行禁令、已执行前缀、recourse/replan 状态或相应指标，因此不能作为 Phase 1 环境直接宣称完成。

此外存在一个可复现到代码行的 ground-truth 旁路：`SlotDecoder.decode_slot()` 的正常特征使用 `policy_demands`，但候选数超过 pruning 阈值时，heuristic 在 `rlccl/envs/decoder.py:425` 读取原始 `demands`。partial-demand 调用传入的是完整 `true_demands`，所以该分支可能泄漏真实目的地信息。Phase 1 必须以接口隔离和测试封堵，而不是继续沿旧 partial-demand 入口扩展。

研究边界冻结如下：

- 主目标固定为“提升 AICCL 在不确定流量条件下的调度效果”。
- `historical moments -> current fine-grained shard-edge action` 分支冻结；不得扩大 `MomentEncoder`、增加 moment feature、追加同路线训练或直接进入旧 V2/V3/V4。
- 保留 Moment 代码、旧 checkpoint 兼容与回归测试，不删除负面实验资产。
- 冻结不否定 uncertainty-aware AICCL；Phase 1 应先建立 ground truth/observation/reveal/execution 的硬隔离。
- predictor、scenario generator、robust prefix、rolling recourse、三时间尺度和 synthesis 优化均不属于本轮实现范围。

历史文档存在一个需要显式按新指令解释的方向冲突：`docs/NEXT_DIRECTION_DECISION.md` 在 2026-07-27 选择“终止 moments 主线”后，把后续工程投入转向 baseline decoder/synthesis；本轮上位指令重新固定主目标为“不确定流量下的 AICCL”，并规定 synthesis 只能在 Gate 2 证明算法收益但 overhead 抵消收益后作为辅助任务。Phase 0 继承旧文档中“冻结 direct moment action”的实证结论，但**不继承**“以 synthesis 为后续唯一主路线”的方向选择。

本次完整阅读并交叉核验了 `CODEX_MULTI_AGENT_UNCERTAINTY_AICCL_INSTRUCTIONS.txt`、`CODE_AUDIT.md`、`docs/AMR_AICCL.md`、`docs/PERFORMANCE_V0.md`、`docs/PERFORMANCE_V1.md`、`docs/V1_FAILURE_DIAGNOSIS.md`、`docs/NEXT_DIRECTION_DECISION.md`；另为核对具体实验语义读取了 `docs/PARTIAL_DEMAND_EXPERIMENT.md` 和 `docs/TRAFFIC_PREDICTABILITY.md`。历史报告中的 GPU/正式实验数字仅作为既有证据引用，没有冒充本轮运行结果。

## 2. 真实调用链概览

```text
TrafficSequence.matrices[t] = X_t
    -> traffic_matrix_to_scenario(X_t)
       -> initial_state[C,V] + demands[C,V]
    -> ProblemInstance（同时持有完整 X_t、完整 demands 和 moment_context）
    -> SlotDecoder.decode_slot(state, demands, ...)
       -> Y_t[C,E]
    -> compute_received_chunks(Y_t, edge_dst, V)
       -> state <- max(state, received)
       -> demands <- demands * (1 - received)
    -> evaluate_schedule(schedule, original ProblemInstance)
```

旧 partial-demand 仅在 decoder 特征处插入：

```text
完整 ProblemInstance
    + PartialDemandObservation(observed_matrix, observation_demands, mask)
    -> decoder 的部分特征使用 observation
    -> action space/state transition/completion/evaluator 仍使用完整 truth
```

## 3. 十一项强制审计

### 3.1 当前 `ProblemInstance`

权威定义只有 `rlccl/envs/problem.py:33` 的 `ProblemInstance`；evaluator 导入该类型，不再维护第二份定义。核心字段为：

- 静态/规模：`V/C/E/T`、`capacities`、`topology`、`shared_constraints`、`topology_info`；
- ground truth：`demands[C,V]` 与 `initial_state[C,V]`（`rlccl/envs/problem.py:81-82`）；
- traffic/sequence 上下文：完整 `traffic_matrix`、`scenario_type`、`sequence_id/step`、`moment_context`、任意 `metadata`（`rlccl/envs/problem.py:84-89`）。

`TopologyInfo` 位于 `rlccl/envs/problem.py:161`，缓存 `edge_src/edge_dst`（`:186-187`）、最短路和 edge-to-shared-group 映射（`:196-201`）。dense destination-incidence `D` 已移除；`compute_received_chunks()` 位于 `rlccl/envs/problem.py:266`。

不确定性缺口：现类型把完整 `X_t`、完整 `demands`、策略可见信息与 evaluator truth 装在同一对象中，没有能力边界。只要把该对象传给 policy/decoder，就能读取 ground truth。Phase 1 不应依赖“调用者自觉不读”，而应引入分离对象或只读受限 view。

### 3.2 traffic matrix 表达

All-to-All-V traffic 采用整数矩阵 `X[src,dst]=k`：方阵、数值有限、非负、整数、对角为零，校验在 `rlccl/traffic/matrix_utils.py:8`。`k` 表示从 `src` 到 `dst` 的独立 chunk 数；总 chunk 数 `C=sum(X)`（`:34`）。

时序容器 `TrafficSequence` 在 `rlccl/traffic/types.py:92`，保存有序 `matrices`、配置参考 `mean_ref/var_ref`、bounds 与 metadata。注意 `mean_ref/var_ref` 是生成配置参考，不等同于 reveal observation。

现有生成器：

- legacy moment-bounded generator：`TrafficProcessConfig` 与 6 family 位于 `rlccl/traffic/process_generator.py:13-40`；候选由短 period 构造，并在 `:215` 用 `np.tile` 重复。这与既有审计中的 2/16 步精确周期结论一致。
- long-horizon generator：5 family 与 `LongHorizonTrafficConfig` 位于 `rlccl/traffic/long_horizon_generator.py:20-87`，可复用作 Phase 1 的 truth sequence 来源。
- long-horizon metadata 含 `latent_regime`、`shock_flags` 等完整生成器隐变量（例如 `rlccl/traffic/long_horizon_generator.py:300-301`）。虽然 metadata 声明只用于 audit/evaluation（`:662`），若整个 `TrafficSequence.metadata` 暴露给 scheduler，仍构成未来信息泄漏通道。

### 3.3 scenario/chunk 转换

`traffic_matrix_to_scenario()` 位于 `rlccl/traffic/matrix_utils.py:25`，以稳定的 `src -> dst -> count` 顺序分配 chunk id；每个 All-to-All-V chunk 恰有一个 source owner（`:44`）和一个 destination demand（`:45`）。`scenario_to_traffic_matrix()` 位于 `:62`；无内嵌 matrix 时只允许“每个 chunk 恰一 source、恰一 destination”的无歧义逆转换（`:87`），因此不适用于 AllGather 的多 destination chunk。

不确定性关键矛盾：完整 `X_t` 一旦先转换为 `C` 个 chunk，`C`、每个 chunk id 和 source ownership 就已经编码了未揭示 truth。旧 partial-demand 明确保留这些信息以维持固定 action space（`rlccl/evaluation/partial_demand.py:198-200`）。Phase 1 必须明确：

- reveal 前哪些 chunk identity 可以存在于 observation/action space；
- source total 已知时能否预先分配匿名 shard token；
- 完全未知 demand 时 action space 是空、准备性动作，还是 scenario-only 虚拟动作；
- scenario 中的虚拟 chunk 不得被当作真实可执行 chunk。

### 3.4 schedule state

当前跨 slot state 是：

- `state[C,V]`：节点是否持有 chunk；
- residual `demands[C,V]`：尚未满足的真实需求；
- `schedule[t]=Y_t[C,E]`：该 slot 每个 chunk 使用哪些 directed edge。

主训练/评估循环用 `compute_received_chunks()` 得到接收矩阵，然后执行 `state=max(state,received)` 和 `demands=demands*(1-received)`；证据见 `rlccl/training/ppo_trainer.py:92-118`、`:258-282` 与 `rlccl/evaluation/sequence_evaluator.py:116-144`。

slot 内部还有 `edge_usage[E]`、`group_usage[G]`、`received_mask[V,C]` 与 active candidates；选中 action 后写入 `Y_t` 并更新三者（`rlccl/envs/decoder.py:499-504`）。这些是单 slot 临时状态，不是跨 reveal stage 的 recourse state。

缺口：没有 reveal stage/ration、observed-vs-truth residual、committed/executed prefix、可替换未执行计划、不可撤销已执行 action、replan interval 或 wasted plan 状态。

### 3.5 feasibility mask

decoder 的初始硬候选由“source 当前持有 chunk”与“destination 尚未持有 chunk”组成（`rlccl/envs/decoder.py:353-356`）。每个 micro-step 再施加：

- edge capacity mask（`:399`）；
- 同 slot 相同 `(chunk,destination)` 不重复接收（`:400`）；
- shared bandwidth group mask（`:401-411`）。

`current_mask` 是这些条件与 active mask 的交集（`:411`）。真实 demand/reveal mask **不在硬候选条件中**；demand 只是 soft feature/heuristic。结果是未揭示 chunk 或无真实需求的 destination 仍可能被发送，只是通常得分较低。

明确泄漏：大候选集 pruning 在 `rlccl/envs/decoder.py:425` 使用函数参数 `demands`，而不是已经构造的 `policy_demands`；partial-demand 脚本在 `scripts/evaluate_partial_demand.py:189-200` 将完整 `true_demands` 作为 `demands`、partial view 作为 `observation_demands` 传入。因此候选超过 train=256/eval=512 时，ground truth 可影响保留候选集合。

Phase 1 必须保持现 deterministic topology/capacity/shared-group checker，并额外引入 reveal execution mask：未揭示真实 demand 不得进入 executable candidate set；scenario/预测 action 只能作为未提交计划，不能直接修改真实 state。

### 3.6 evaluator

唯一 schedule 质量入口为 `rlccl/envs/evaluator.py:541` 的 `evaluate_schedule()`。它从原始 `problem.initial_state` 与完整 `problem.demands` 重放（`:560-561`），检查 edge capacity（`:572-575`）、shared group（`:577-581`）和 source ownership（`:583-589`），用 `compute_received_chunks()` 更新真实 state（`:597-608`），以 completion step 为主、早满足 demand 为次级分数（`:600-617`）。

可复用价值：它是所有 baseline 的共同 ground-truth scorer，可继续作为最终 truth-only evaluator；policy 不应获得其内部 `problem`。

现有限制：

- 不知道 reveal wait、recourse、replanned actions、wasted plan、replan time；
- 不检查“未揭示 demand 不可执行”；
- 没有显式验证 `Y_t` 元素必须是二进制/非负；当前 decoder 自身生成二进制，但 Phase 1 新入口必须加确定性 action schema 检查；
- 无法区分 scenario-only proposal、已提交 prefix 与已执行 action。

### 3.7 moment context

`SlidingMomentEstimator` 位于 `rlccl/traffic/moment_estimator.py:13`，历史是每实例私有、固定窗口 deque（`:37`）。`TrafficSequenceRunner` 每条 sequence 新建 estimator（`rlccl/envs/sequence_env.py:44`），先 `get_context()`、后 `yield`，且仅在恢复迭代后 `update(matrix)`（`:53-93`），所以 mean/variance history 只含 `X_0..X_{t-1}`。

但 `MomentContext` 不是纯 history-only feature：`get_context(current_matrix,...)` 在 `rlccl/traffic/moment_estimator.py:88-96` 用当前 matrix 计算 `current_send_z/current_recv_z`。decoder 还可直接从 `current_matrix` 计算 sparsity、load CV 和 max entry（`rlccl/envs/decoder.py:63-95`）。`tests/test_no_future_leakage.py:12` 明确测试的是“改变当前 matrix 只改变 current z，不改变历史 moments”，并非“policy 看不到当前 truth”。

这在旧 V1 的“完整当前 demand 已知”语义下不是违规；在 Phase 1 中若传入完整 `X_t` 就会违规。旧 partial-demand 脚本为每种 observation 用 `observed_matrix` 重新计算 context（`scripts/evaluate_partial_demand.py:450-454`），这一做法可作为 view 构造参考，但不能解决完整 problem/action-space 与 pruning 旁路。

冻结边界：`scripts/train_moment_policy.py:107-112` 的 12-d node、9-d candidate、8-d global moment action policy，decoder 的 moment feature/replay 分支，以及相关 checkpoint 均只保留兼容和回归用途；不得作为 Phase 1 主策略继续扩展或训练。

### 3.8 当前 partial-demand 实验语义

`PartialDemandObservation` 位于 `rlccl/evaluation/partial_demand.py:23`，支持 `random_entries`、`source_totals`、`source_destination_totals`、`partial_shards`（`:14-19`）。它是一次性 observation builder，不是 reveal process：

- random entries/partial shards 对完整 truth 的 entry/chunk 做一次采样遮蔽（`:180-216`）；
- source totals 用确定性平衡矩阵插补（`:218-224`）；
- source+destination totals 用 transportation max-flow 插补（`:228-235`）；
- totals 模式 `revealed_chunk_mask` 全为 false，但仍为全部真实 chunk 构造代理 destination（`:220-230`）。

执行脚本 `_run()`（`scripts/evaluate_partial_demand.py:162`）同时持有 `true_demands` 与 `observed_demands`（`:173-180`），decoder 后 state transition、truth demand 清除、completion/timeout 均由 truth 控制（`:207-216`），最后 `evaluate_schedule(schedule, problem)` 再用完整 truth 评分（`:219`）。每个 traffic 时刻各 observation condition 相互独立运行，没有 `r=0..R` 的递增揭示，也没有 reveal 后基于当前执行 state 的 recourse。

旧实验的 15 条 sequence / 3 training seed 结果全部无稳定收益，且 overall timeout 38.97%；这只否定旧的 partial feature-conditioning 方式，不否定真正 uncertainty environment。

### 3.9 可复用代码

| 模块 | 可复用内容 | 复用边界 |
|---|---|---|
| `rlccl/envs/problem.py` | topology、truth problem、`compute_received_chunks` | truth 对象不得直接暴露给 policy |
| `rlccl/traffic/matrix_utils.py` | matrix 校验与 deterministic chunk conversion | reveal 前不可先泄漏完整 chunk map |
| `rlccl/traffic/types.py` | `TrafficSequence` 序列容器 | metadata 要白名单过滤 latent/future 字段 |
| `rlccl/traffic/long_horizon_generator.py` | 非周期 truth sequences | 仅 generator/evaluator 持有 latent metadata |
| `rlccl/envs/sequence_env.py` | sequence 顺序、每序列 estimator 隔离范式 | 需新 uncertainty runner，不能直接返回完整 problem 给 scheduler |
| `rlccl/evaluation/partial_demand.py` | row/column totals 插补、可重复采样思路 | 需重构为单调多阶段 reveal；旧类型不是执行环境 |
| `rlccl/envs/decoder.py` | topology/capacity/shared-group mask 与 state transition inputs | 必须清除 truth pruning 旁路并加入 executable reveal mask |
| `rlccl/envs/evaluator.py` | 统一 truth-only legality/completion scorer | policy 与 scorer 必须隔离；扩展 recourse metrics |
| `rlccl/evaluation/metrics.py` / `sequence_evaluator.py` | completion/tail/paired rows | CI 单位必须保持完整独立 sequence |
| `rlccl/models/traffic_predictor.py` | history-only example slicing与简单基线资产 | H1 才使用；Phase 1 不训练 predictor |

### 3.10 必须新增的接口（仅定义缺口，不实现）

Phase 1 至少需要下列语义明确、相互隔离的抽象：

1. `UncertainProblemInstance`
   - evaluator-only truth：`X_t`、truth chunk/demand map、initial truth state；
   - scheduler-visible 静态信息：topology、capacity、shared groups、time budget；
   - 禁止通过 `.problem`、metadata、cache key 或 debug payload 回取 truth。
2. `DemandRevealProcess`
   - 产生 stage `r` 与 `M_{t,r}`；保证 `M_{t,r-1} subseteq M_{t,r}`、末 stage 全可见；
   - 支持 `random_entries`、`source_totals_first`、`source_destination_totals_first`、`partial_shards`、`time_based_arrival` 及 ratios 0/0.25/0.5/0.75/1；
   - seed、时钟和 truth 由环境持有，scheduler 只取当前 reveal event。
3. `PartialObservationState`
   - 只含当前 stage 可见的 observation、mask、公开的匿名/真实 chunk identity、已执行 state 与合法 action view；
   - 明确 unknown 与数值 0 不同，不能继续用“填 0”同时表达二者。
4. `ScenarioSet`
   - 场景 matrix/概率/校准元数据只用于 planning；
   - scenario chunk/action 必须与 executable truth token 分域，除非 reveal 后显式绑定。
5. `RecourseMetrics`
   - completion、oracle regret、reveal wait、recourse count、replanned actions、wasted plan、synthesis/replan time、legality、timeout。
6. decoder/environment 执行协议
   - `propose(observation, planning_state) -> uncommitted plan`；
   - `commit(executable_action, reveal_mask, truth_state) -> checked transition`；
   - 未揭示 demand、scenario-only chunk、已失效未执行 prefix 均不得绕过 checker；已执行 action 不可撤销。
7. paired evaluator 协议
   - 所有 baseline 共享同一 truth sequence/topology/reveal process/seed/timeout/checker；full oracle 只取上界结果，不把其 observation 传给其他方法。

### 3.11 未来信息泄漏风险

| 风险 | 现有证据 | Phase 1 必须测试/防护 |
|---|---|---|
| 完整 problem 对象泄漏 | `ProblemInstance` 同时含 `traffic_matrix`、`demands`、metadata | scheduler API 只接收受限 view；反射/序列化 payload 不含 truth |
| pruning 读取 truth | `rlccl/envs/decoder.py:425` 使用 `demands` 而非 `policy_demands` | 构造 >512 candidates 的 counterfactual test：仅改变 hidden truth 不得改变候选/logits/action |
| 完整 chunk map 泄漏 | old partial 保留全部 chunk id/source ownership（`partial_demand.py:198-200`） | ratio=0 时检查 observation/action space 不暴露未授权 C/destination；明确 source-total 模式例外语义 |
| hidden=0 混淆 | observed matrix 用零同时表示真实 0 与 unknown | observation 必须携带 mask，所有 feature/cache 使用 `(value,mask)` |
| estimator/current feature 泄漏 | current z 使用 `current_matrix`（`moment_estimator.py:88-96`）；global feature读取 current matrix | history feature builder只取 `<t`；current feature只能读取 `O_{t,r}` |
| generator latent 泄漏 | long-horizon metadata 含 regime/shock/future arrays | policy-visible metadata 白名单；latent 仅标为 oracle/audit |
| evaluator 反向泄漏 | scorer 必须持有 full truth | scorer 与 decision 进程/对象接口隔离；评估输出只在 episode/decision 后返回 |
| cache 泄漏 | 未来若以完整 X 或 truth-derived shape/key 缓存 | cache key 只能用当前允许 observation + static config；paired hidden-truth test |
| reveal RNG/未来 mask 泄漏 | 一次生成完整 mask 序列容易把后续 reveal 暴露 | scheduler 只收到当前/过去 stage；不得收到 future reveal schedule 或 RNG state |
| train/test sequence 泄漏 | step-level 随机切分会跨同序列 | 完整 sequence split；测试 manifest 无 sequence overlap |
| 统计伪重复 | 多 step、多 method、多 model seed 不增加独立 traffic sequence | sequence-cluster bootstrap，报告 raw rows 与独立 sequence 数 |

## 4. V1 scripts/tests 审计

V1 的正式脚本链为：

- `scripts/train_moment_policy.py`：baseline/moment 两种 policy mode；train 与 validation 使用相同 family 集但不同 sequence seed；
- `scripts/evaluate_moment_policy.py`：同一 held-out matrices 的 baseline/mean-only/full/shuffled 配对；
- `scripts/run_v1_ablation.py`：默认训练 seeds 42/142/242，formal gate 在 `scripts/run_v1_ablation.py:129-177`；
- `scripts/evaluate_partial_demand.py`：旧的一次性 partial feature-conditioning 实验。

已有测试覆盖：matrix 转换、ProblemInstance 统一、history estimator、sequence isolation、V1 moment feature/replay、旧 partial builder、traffic generator、long-horizon dynamics、predictor history slicing、counterfactual helpers、metrics 与 checkpoint。关键边界：

- `tests/test_sequence_runner.py:8` 证明 estimator update 在当前 step 之后；
- `tests/test_traffic_predictor.py:37` 证明 predictor feature slice 不含目标 `X_t`；
- `tests/test_no_future_leakage.py:12` 允许当前 X 改变 current z，因此不是 Phase 1 truth-isolation test；
- `tests/test_partial_demand.py:16-68` 检查一次性 observation 不捏造 revealed demand，但未测试单调 reveal、末 stage 全揭示、hidden demand 不可执行或 pruning truth leak；
- `tests/test_partial_demand.py:71` 只检查 full observation 与默认 decoder 等价。

结论：当前测试对旧 V1 语义有效，但不能作为 Phase 1 Gate 的 uncertainty-environment 证明。

## 5. 现有完整测试发现与执行事实

### 5.0 工作树状态事实

审计开始时先执行了 `git status --short`，随后也显式尝试 `git -C F:\AMR-AICCL status --short` 与 `git -C F:\AMR-AICCL\RLCCL-main status --short`。两处都返回 `fatal: not a git repository (or any of the parent directories): .git`；`F:\AMR-AICCL\.git` 是空目录，`RLCCL-main` 下没有 `.git`。因此无法用 Git 证明 clean/dirty 或枚举用户既有改动。本任务采用的保护措施是：不改任何已存在源码、测试、配置、checkpoint 或其他文档，只新增本审计文件；测试以禁止 bytecode/pytest cache 写入的方式运行。

### 5.1 权威入口

- `README.md:18` 明确给出 `python -m pytest -q`；
- `pytest.ini:2-3` 指定 `testpaths = tests`、`addopts = -ra`；
- 仓库没有 CI 配置或 `pyproject.toml/setup.cfg` 中的另一套测试入口；
- `test.sh`/`test_single_problem.sh` 是依赖特定 checkpoint、`msccl` 环境和 XML 导出的模型应用脚本，不是单元/回归全套入口。

为遵守本任务“唯一可写输出”的边界，实际全套命令使用 `-B` 和 `-p no:cacheprovider`，只关闭 `.pyc`/`.pytest_cache` 写入，不改变测试选择或断言：

```powershell
$sw=[System.Diagnostics.Stopwatch]::StartNew(); & F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider; $code=$LASTEXITCODE; $sw.Stop(); Write-Output ("CODEX_TEST_EXIT_CODE={0}" -f $code); Write-Output ("CODEX_TEST_ELAPSED_SECONDS={0:N3}" -f $sw.Elapsed.TotalSeconds); exit $code
```

结果：exit code `0`；`73 passed, 4 skipped in 14.44s`；外层实测 elapsed `15.504s`。

原始 pytest 结尾：

```text
...............................................s........................ [ 97%]
..                                                                       [100%]
=========================== short test summary info ===========================
SKIPPED [1] tests\test_moment_policy_shapes.py:5: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_optimizer_checkpoint.py:6: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_sequence_evaluator.py:5: could not import 'torch': No module named 'torch'
SKIPPED [1] tests\test_partial_demand.py:72: could not import 'torch': No module named 'torch'
73 passed, 4 skipped in 14.44s
CODEX_TEST_EXIT_CODE=0
CODEX_TEST_ELAPSED_SECONDS=15.504
```

环境事实：

- 默认 `python`：`C:\Python313\python.exe`，Python 3.13.5，缺少 pytest；尝试 `python -B -m pytest --collect-only -q -p no:cacheprovider` 退出 1，原始错误为 `No module named pytest`。
- 可执行测试环境：`F:\AnaConda\python.exe`，Python 3.12.7、NumPy 1.26.4、pytest 7.4.4；没有 Torch。
- Conda 列出的其他环境中，`CS231n` 无 pytest/NumPy/Torch，`SIIHE` 无可用 `python.exe`；未发现本地可运行 Torch 全套的现成环境。
- 没有安装依赖、没有伪造或复用历史 GPU 结果。四项 Torch 用例本轮实际未执行，必须保持为 skipped，而不能报告通过。

原始输出没有另写日志文件（受唯一输出文件约束）；上述输出逐字嵌入本审计文档。历史远程 `84 passed` 只作为既有报告事实，不是本轮测试结果。

## 6. Phase 1 建议任务拆分（不实施）

以下拆分只供主 Agent 建账与用户审核；本轮未创建文件或代码：

1. `UENV-SPEC-001`：冻结 truth/observation/reveal/action/commit 语义与五个必需抽象；先写接口不变量和 threat model。
2. `UENV-TEST-001`：先写 uncertainty environment tests，覆盖 truth/view 分离、mask 单调、ratio 0/0.25/0.5/0.75/1、末 stage 全揭示、hidden demand 不可执行、每 sequence 状态隔离、>512 candidate pruning counterfactual、metadata/cache 泄漏。
3. `UENV-IMPL-001`：实现 `UncertainProblemInstance`、`DemandRevealProcess`、`PartialObservationState`，复用 generator/topology，但不接 predictor。
4. `UENV-EXEC-001`：实现 proposal/commit adapter 与 reveal-aware deterministic mask；保持原 topology/capacity/shared-group checker，不实现 robust prefix。
5. `UENV-METRIC-001`：实现 `RecourseMetrics` 与 truth-only paired evaluator；五个 baseline 共用同一 truth/reveal/seed/checker。
6. `UENV-BASE-001`：仅实现 Phase 1 指定 baseline：Full-information oracle、Wait-until-known、Partial-current-only、Long-term mean、Previous-value；oracle 只作上界。
7. `UENV-GATE-001`：运行全套 unit/regression + baseline smoke，要求 schedule legality 100%，记录 timeout/原始失败；由 Supervisor 独立检查后才申请进入 H1。

推荐文件所有权应互斥；Core Research Engineer 写实现/测试，Supervisor 只读审查与书面 Gate 判定，主 Agent 维护账本、问题定义与决策日志。未经用户批准不得增加 Subagent。

## 7. Phase 0 Gate 审计意见

从执行 Agent 视角：

- 目标与冻结边界：已明确；
- 真实项目结构和 11 项审计：已记录并可追溯；
- 不确定性发生时刻：应定义为 scheduler 在完整 `X_t` 到达前、仅持有 `O_{t,r}` 时做决定；旧 partial-demand 不满足；
- 现有测试：可运行的完整 pytest suite 通过，4 项因本地无 Torch 原样 skipped；
- Phase 1 接口缺口与任务拆分：已给出；
- Phase 1 代码：未开始。

是否允许进入 Phase 1 仍应由 Supervisor 的独立 `SUPERVISOR_REVIEW_PHASE_0.md` 和用户审核决定。Supervisor 建议特别检查：pruning truth leak、完整 chunk map 泄漏、generator latent metadata、truth-only evaluator 的能力隔离，以及四项 Torch skipped 是否需要在用户批准 Phase 1 后于有 Torch 环境补跑。
