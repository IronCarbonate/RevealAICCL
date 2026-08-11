# Phase 3B：Prediction-Free Empirical Ambiguity Set 协议

## 1. 目标、前提与停止边界

Gate H1 已由 Supervisor 最终裁定为 `FAIL`。本阶段只回答：不预测下一流量时，能否从
已完成的真实历史、当前已授权 observation 与公开 topology 构造有限、非空、可审计的
经验不确定集合，并从中选出固定预算的 traffic scenarios，作为后续 Phase 4 的候选输入。

本阶段允许：traffic-space ambiguity constraints、有限 support 构造、五类 support selector、
coverage/sharpness/distance/diversity/tail/overhead 比较。本阶段禁止：`TransferAction`、commit、
schedule prefix、horizon、robust action score、residual repair、recourse/replan、completion、
oracle regret、wasted prefix、legality 收益或任何 AICCL 调度收益声明。那些属于 Phase 4。
Phase 3B 完成并经 Supervisor 最终审查后必须停止；未经用户再次批准不得进入 Phase 4。

## 2. 信息边界与窄构造接口

普通 constructor 只能接收新的 `AmbiguityConstructionView`：

- 最近 32 个已完成矩阵 `X_{t-32},...,X_{t-1}`；
- 当前公开 `entry_mask`、`observed_matrix`、reveal mode/stage/ratio；
- 已公开的 source/destination totals；
- 深复制、只读的公开 topology；
- evaluator 显式提供的 construction seed 与 fit-only feature normalization。

该 view 必须删除 `family`、`sequence_id`、base/actual/reveal seed、generator metadata、latent
regime/shock/hotspot、truth/world、manifest、reveal process、future mask/order/arrival、callback
与 closure。constructor 不得按标识符或 family 建集合。evaluator 只在完成构造后用这些字段
做 provenance 和分组。

固定 ordinary counterfactual 必须逐字节不变：保持 view、history、topology 与 construction
seed 不变，只改变未揭示 `X_t`、future reveal、metadata、family 或 sequence ID。普通模块不得
导入 `uncertainty.execution`，不得产生 `Proposal` 或 `TruthTokenId`。

`oracle_support_upper_bound` 在独立 evaluator-private 调用栈中接收 `X_t`。它在 support 中
直接包含当前 truth，因此只是 tautological support-selection ceiling；固定标记
`uses_oracle=true`、`upper_bound_only=true`，不参与 validation 选择、通过判据或 Phase 4 输入。
它不得把 truth、半径、候选或阈值缓存/回传给普通方法。Phase 1 的 full-information completion
lower bound 与本 oracle support 不是同一个对象或性能语义。

## 3. 新的确认性完整序列 corpus

不得复用已经决定 H1 路由的 42/142/242 corpus 作为 Phase 3B 确认性证据。正式实验固定：

- topology：`Rear4GPU`；
- families（固定顺序）：`regime_switching_long`、`stochastic_volatility`、
  `rare_shock_recovery`、`hotspot_random_walk`、`same_moments_different_dynamics`；
- 新 base seeds：`342/442/542`；
- 每个 `(family, base_seed)` 五条 length-1024 完整序列；
- sequence index `0/1/2/3/4` 分别为 `fit/fit/validation/calibration/test`；
- actual seed：`base_seed + family_index*1_000_000 + sequence_index*10_000`；
- same-moments variant：`SAME_MOMENT_VARIANTS[(seed_index+sequence_index)%4]`；
- generator：`mean_level=2.0`、`std_level=1.5`、`max_entry=8`、
  `calibration_candidates=1`，其余当前 dataclass 默认值与完整 `asdict(config)` 写入 manifest。

共 75 条互异完整序列：30 fit、15 validation、15 calibration、15 test。sequence ID 与完整
sequence digest 在全部 split 之间必须互斥。H1 corpus digest 也写入禁交叉集合并验证无重合。
fit 只用于 feature normalization 和 tail threshold；validation 只选择一个非 oracle selector；
calibration 只冻结 envelope radius；test 只运行一次正式评价。

每条 validation/calibration/test sequence 固定在
`t=32,96,160,...,992` 共 16 个 checkpoint 评价。历史窗固定 32，任何 constructor 只见
`X_{t-32:t}`，不见 `X_t`。每个 checkpoint 运行五种 reveal mode 与固定 ratios
`0.00/0.25/0.50/0.75/1.00`。`time_based_arrival` 另记录实际 entry fraction；ratio 1.00
只作为完全揭示 singleton sanity control，不进入未知需求主统计。

canonical record index 按 protocol 的 family、base seed、sequence index 顺序从 0 开始。
reveal seed 固定为
`31_000_000 + record_index*100_000 + t*10 + mode_index`。construction case index 按
`split, record_index, t, mode_index, stage_index, K` 的字典序从 0 开始；随机 replicate seed 为
`41_000_000 + case_index*100 + replicate_index`。

## 4. Traffic descriptor、normalization 与物理范围

复用公开 topology 的 deterministic shortest-path offered-load proxy；它不是 learned schedule
的真实 group utilization。descriptor 固定顺序：

1. total traffic；
2. 4 个 source loads；
3. 4 个 destination loads；
4. hotspot strength，定义为 `max(destination_load)/max(mean(destination_load),1e-8)`；
5. off-diagonal zero fraction sparsity；
6. 全部 bandwidth-group offered loads。

fit-only center/scale 从 30 条新 corpus fit sequence 的全部矩阵计算；scale 为 population std，
小于 `1e-8` 时置 1。所有 distance/severity/calibration score 使用同一 center/scale；不得按
family、test 或 reveal case 重拟合。物理范围由 4 nodes、zero diagonal 与 frozen `max_entry=8`
机械计算：total `[0,96]`、每个 source/destination `[0,24]`、hotspot strength `[0,4]`、
sparsity `[0,1]`、group load 由 group coefficients 对 entry-wise 8 的矩阵计算上界。

## 5. Observation 一致性与 candidate reconciliation

每个 ambiguity support matrix 必须是有限、整数、非负、zero diagonal，并通过统一 validator：

- `entry_mask=true` 的 exact entry 必须等于 `observed_matrix`；
- `partial_shards` 的非 exact pair 必须不少于已 reveal shard count；
- source totals 存在时必须逐 source 精确相等；
- destination totals 存在时必须逐 destination 精确相等；
- ratio 1.00 必须退化为与完整 observation 相同的唯一 truth matrix；
- 矛盾 totals、无整数可行 completion、空 history 或任何空/非法集合立即报错，不能静默放宽。

每个 off-diagonal entry 的物理上界固定为 generator `max_entry=8`。先建立 lower matrix `B` 与
capacity matrix `U`：exact entry 令 `B=observed`、`U=0`；partial-shard 非 exact entry 令
`B=observed`、`U=8-B`；其他未知 entry 令 `B=0`、`U=8`。任何 `B>8`、负 residual margin、
row/column residual 不守恒或总 residual 超过 capacity 立即判 infeasible。真实 truth 只由 evaluator
用于证明 canonical observation 有可行 witness，不能传给 reconciliation；corrupt observation 必须报错。

32 个历史候选 `C` 按 oldest-to-newest 逐个 candidate-sensitive reconciliation：

1. 无 aggregate totals：exact 使用 `B`，其余为 `clip(max(C,B),B,8)`；随机 entry mode 的
   non-exact pair 保留 `C`，partial-shard pair 至少为 revealed count。
2. source-only totals：每 row 令 residual `R_s=source_total_s-sum_d B_sd`。循环到 `R_s=0`：
   active destinations 是剩余 capacity `c_d=8-current_sd>0` 的非 exact off-diagonal cells；权重
   `w_d=max(C_sd-current_sd,0)`；若全部为 0，改用 `w_d=c_d`。计算
   `q_d=R_s*w_d/sum(w)`，先同时加入 `min(c_d,floor(q_d))` 且不超过剩余 `R_s` 的单位；再按
   fractional remainder `q_d-floor(q_d)` 降序、`max(C_sd-current_sd,0)` 降序、destination index
   升序逐个加 1。若仍有 residual，重新计算 active/weights/quotas。每轮至少分配一个单位；active
   为空但 residual 非零则 infeasible。返回前精确复核 row totals 与 `[B,8]`。
3. source+destination totals：做 lower-bound transform 后的 deterministic integer min-cost flow。
   节点顺序固定为 source-node、row `0..V-1`、column `0..V-1`、sink；source→row capacity 为
   row residual，column→sink capacity 为 column residual。每个允许 `(s,d)` 建 `8-B_sd` 条
   capacity-1 parallel unit arcs；第 `m=1..8-B_sd` 条的 integer cost 为
   `1 + |B_sd+m-C_sd|-|B_sd+m-1-C_sd|`，故 cost 只为 0 或 2，优先保持该 history candidate。
   原始 arc 顺序固定 `(s,d,m)` 字典序，reverse arc 紧随原 arc。逐单位 successive shortest
   augmenting path；每次在完整 residual graph 上用 deterministic Bellman-Ford 求最小 cost，
   equal-distance 时选择 lexicographically smaller arc-index path。若找不到 source→sink path则
   infeasible；发送 `sum_s R_s` 单位后终止。所有 capacity/cost/flow 为 integer，输出
   `X=B+row-column unit flows`，再精确复核 row/column totals、exact/lower bound、zero diagonal
   与 entry `<=8`。该 cost 与 tie-break 保证多解时结果仍受 `C` 影响且字节可复现。

ratio 0 且无 aggregate 时 reconciled support 必须与原始 32 个 recent real samples 精确相同。
保留全部 32 个带时间身份的 indexed candidate，即使矩阵内容重复；不得因 digest 去重而改变
经验 multiplicity。ratio 1.00 不执行上述 32 次选择，直接建立 truth-consistent singleton control。

## 6. EmpiricalAmbiguitySet

每个集合同时保存：

- 32 个 reconciled recent support matrices 与各自 history offset；
- descriptor names/vectors；
- calibrated pointwise lower/upper descriptor bounds；
- empirical mean/variance 与 full-32 probability-weight ambiguity；
- observation constraint fingerprint、normalizer/group-coefficient digest、history cutoff；
- `uses_oracle=false`、`upper_bound_only=false`。

令 32 个 indexed candidate descriptor 为 `Y_i in R^D`。逐 component 定义
`mu_j=(1/32) sum_i Y_ij`、`v_j=(1/32) sum_i (Y_ij-mu_j)^2`（population/ddof=0）、
`delta_mu_j=0.25*fit_scale_j`、`v_low_j=0.5*v_j`、
`v_high_j=1.5*v_j+0.01*fit_scale_j^2`。full-support probability-weight ambiguity 精确定义为：

```
P(A) = {p in R^32:
          p_i >= 0,
          |sum_i p_i - 1| <= 1e-10,
          |sum_i p_i Y_ij - mu_j| <= delta_mu_j + 1e-10,  all j,
          v_low_j - 1e-10
            <= sum_i p_i (Y_ij-mu_j)^2
            <= v_high_j + 1e-10,                         all j}
```

variance 固定围绕 empirical `mu_j`，不随候选 `p` 重新定中心。uniform witness
`p_i=1/32` 的 weighted mean 恰为 `mu`、weighted variance 恰为 `v`，因此必在上述非空集合中；
validator 必须显式验证该 witness。NaN/Inf、负 budget、非归一化权重或 uniform witness 失败均硬错。

未校准 point envelope 是 32 个 reconciled descriptors 的 componentwise min/max `a_j/b_j`。
calibration score 对所有 descriptor joint exceedance：
`s=max_j(max((a_j-Y_truth,j)/fit_scale_j,(Y_truth,j-b_j)/fit_scale_j,0))`。
calibration universe 与 K/selector 无关：每条 calibration sequence 恰含 16 checkpoints × 5 modes ×
4 个 ratio `<1` stages = 320 cases，每 case 等权；先对各 sequence 的 320 scores 取 90% quantile，
再对 15 个 sequence quantiles 等权取 90% quantile，均用 NumPy `method="higher"`，得到唯一
nonnegative radius。time-based arrival 仍按 stage case 等权，实际 entry fraction 只另行报告。
LOFO 仅用 seen-family 12 条 calibration sequences，以相同两层 higher 规则重算 radius。

正式 point bounds 为 `max(physical_low,a-radius*fit_scale)` 与
`min(physical_high,b+radius*fit_scale)`。radius 只扩张 descriptor envelope；exact entries、totals、
partial-shard lower bounds 永远是优先的 hard observation constraints，不能被 radius 放宽。

必须把三个对象分开：calibrated point envelope；full-32 probability-weight ambiguity `P(A)`；固定
K reduced support。后者是下游有限场景近似，不声称 K 个 uniform scenarios 满足 full-32 moments。
这不是无限宽 pure-moments set：finite indexed recent support、finite point bounds、非空概率权重
集合与 observation consistency 同时存在。

## 7. 五类固定预算 support selector

requested `K in {1,4,8,16}`。除 ratio 1 singleton control 外，四个 ordinary 方法从完全相同的
32 个 reconciled indexed candidates 选择不重复 index；矩阵内容可以重复。实际 support size
必须为 K，uniform weights，总预算、输入、normalization 与 tie-break 公平一致。

1. `random_empirical`：每 case 做 8 个固定 seed 的无放回均匀选择；指标先在 paired case 内
   平均，replicate 不是独立统计样本。导出 Phase 4 候选时只能使用显式单 seed，不能择优。
2. `worst_recent_cases`：severity 为 descriptor 中 total/source/destination/hotspot/group 的最大
   upper-sided fit-standardized value以及 density `(1-sparsity)` 的 upper-sided值之最大值；降序取 K，
   tie 取更近的 history offset。
3. `boundary_scenarios`：target 顺序严格为 descriptor `j=0..D-1`，每个 descriptor 先 calibrated
   lower、再 calibrated upper。对当前 target，从所有**尚未选择的 index**中取
   `abs(Y_ij-target_j)/fit_scale_j` 最小者，tie 取更近 history offset；加入后立即进入下一 target，
   达到 K 即停止。若遍历全部 `2D` targets 后仍不足 K，则重复 minimax 的 farthest-from-selected
   步骤从未选 index 补齐，tie 取更新历史。重复 matrix 内容仍可由不同 index 选入。它是
   empirical boundary support，不声称生成未观测 truth。
4. `minimax_subset`：在 fit-standardized descriptor Euclidean distance 上做确定性 greedy k-center。
   第一个 center 是最小化全 pool 最大距离的 medoid，之后反复加入离已选集合最近距离最大的
   candidate；全部 tie 取更新历史。必须明确报告这是 deterministic greedy minimax approximation，
   不得宣传为精确组合最优。
5. `oracle_support_upper_bound`：evaluator 直接放入当前 truth，再以同一 minimax tie-break 从 recent
   candidates 填满 K；K=1 时只含 truth。它只作 tautological ceiling，永不参与普通 selector 选择。

## 8. Raw metrics 与配对统计

raw paired case identity 为
`sequence_id × t × reveal_mode × stage/ratio × K × construction seed`。四个 ordinary 方法与
oracle 使用相同 truth/observation/history/topology/config；random 的 8 repeats 在 case 内聚合。
每行至少记录：

- constraint joint/component coverage；
- physical-normalized mean bound width；
- truth 到 support 的 fit-standardized nearest RMS distance与 matrix L1 distance；
- support 对 32-candidate pool 的 covering radius；
- mean pairwise diversity、duplicate fraction、actual/requested K；
- total-tail、group-tail、hotspot-destination support hit/event；
- invalid/empty flag；set construction 与 selector time；
- family/base seed/complete sequence/checkpoint/reveal/config 与全部 digests；
- `uses_oracle`、`upper_bound_only`。

所有 primary metric 的精确定义如下。令 descriptor 数为 D、fit scale 为 `sigma_j`，support 为
`Z_1..Z_K`，truth 为 X，32-candidate pool 为 `C_1..C_32`：

- descriptor distance
  `d_phi(A,B)=sqrt((1/D)*sum_j((phi_j(A)-phi_j(B))/sigma_j)^2)`；nearest RMS 为
  `min_k d_phi(X,Z_k)`；
- matrix L1 为 `min_k sum_{s!=d}|X_sd-Z_k,sd|`；
- covering radius 为 `max_i min_k d_phi(C_i,Z_k)`；
- pairwise diversity：K=1 时为 0，否则
  `2/(K*(K-1))*sum_{k<l} d_phi(Z_k,Z_l)`；
- duplicate fraction 按 matrix 的 canonical shape/dtype/bytes digest，而非 history index：
  `1-number_of_unique_matrix_digests/K`；actual K 仍按所选 index 数，ratio1 control 例外为 1；
- component coverage 为 `(1/D)*sum_j I(lower_j-1e-10<=phi_j(X)<=upper_j+1e-10)`；joint
  coverage 要求所有 D 个 descriptor bounds 同时成立且 truth 自然满足 hard observation constraints；
- physical-normalized width 为对 `physical_high_j>physical_low_j` 的 components 等权平均
  `(upper_j-lower_j)/(physical_high_j-physical_low_j)`；zero physical range component 排除并报告数量。

fit total 与每个 group offered-load 的 90% quantile（NumPy `method="linear"`）定义 tail threshold。
total event 是一个 raw case 上 `truth_total>q90_total`，hit 是 `max_k scenario_total_k>q90_total`；
group 使用 **micro recall**：每个 `(raw case, group j)` 的 `truth_group_j>q90_group_j` 是一个 event，
hit 是同一 group 的 `max_k scenario_group_kj>q90_group_j`，多个 group events 分别计数；hotspot
denominator 是全部 ratio `<1` raw cases，hit 表示 truth hotspot destination 至少出现在一个 scenario
中。event/hit 只在普通 support 已冻结后由 evaluator 计算，不进入 constructor。

random 的 8 repeats 先对同一个 paired case 的 support metrics 算术平均，coverage envelope 不按
replicate/K重复加权。每个 `sequence×method×K` 再对 16 checkpoints、5 modes、4 个 ratio `<1`
stage 共 320 cases逐 case等权聚合；time-based actual fraction 不改变权重。然后 15 个完整 sequence
等权，不能把 step、stage、entry、scenario 或 replicate 当独立样本。tail/group/hotspot recall
以各 sequence event/hit counts 先求 pooled counts，同时另报 sequence-level rates；无事件的单条
sequence 不伪造 rate。

selected-vs-random 的 primary paired effect 为每条 sequence 的
`nearest_RMS(random_empirical)-nearest_RMS(selected)`，正值表示 selected 更好。95% CI 使用五
family 内各 3 条 test sequence 有放回抽样的 family-stratified sequence bootstrap，10,000 次、
seed `20260731`。报告15 independent sequences、raw cases、total-traffic checkpoint ACF 与
positive-sequence ESS；另报 family、base-seed 和五折 LOFO。ratio1 只单列 singleton sanity。

## 9. Validation 选择、LOFO 与进入 Phase 4 的预注册条件

validation 只在 `K=8`、ratio `<1.0` 上，从四个 ordinary selector 选择 sequence-equal mean
nearest RMS distance 最小者。差值绝对值不超过 `1e-12` 时 tie order 固定为
`minimax_subset`、`boundary_scenarios`、`worst_recent_cases`、`random_empirical`。test 不得改变选择。
若 validation 选择 `random_empirical`，则 selected-vs-random delta 机械为 0、条件 2 必然 FAIL；
这是预注册的保守 Gate，不得在看到 test 后更换 comparator、K 或 selector。

五折 LOFO 对 held-out family 完全排除 fit normalization、validation selection 与 calibration
radius，再只在该 family 三条 test sequence 评价。family identity 不是 constructor 输入。

Phase 3B 只有在下列条件全部成立时才可向用户建议进入 Phase 4；任何数据条件失败为 `FAIL`，
证据/实现不完整为 `HOLD`：

1. selected `K=8` 的 ambiguity joint coverage 在未知-demand test cases overall 至少 `0.85`，
   每 family 至少 `0.80`；
2. selected-vs-random nearest-distance paired delta 的 95% CI 下界严格大于 0；3/3 base seeds
   mean delta 为正，至少 4/5 family 为正；
3. LOFO aggregate delta 非负、至少 3/5 held-out family 为正，且至多一个 family 的 selected
   mean distance 相对 random 恶化超过 10%；
4. selected support 的 total-tail recall、group-tail recall、hotspot support recall 均至少 0.70；
   total/group event 各不足 10 时，只有在其他数据条件 1--3、5--6 全部通过时才为 `HOLD`；若
   任一其他数据条件已失败，总判定仍为 `FAIL`，tail insufficiency 不得遮蔽明确失败；
5. selected ambiguity physical-normalized mean bound width 不超过 0.75；所有 ordinary method 的
   invalid/empty rate 为 0，ratio-1 singleton coverage 为 1；construction/selector time finite；
6. manifest/raw/summary 的全部数据声明可从持久化 raw evidence 精确重算并逐字段比对；determinism、read-only、observation consistency、
   ordinary counterfactual、family/ID 剥离、oracle isolation、finite support、selection/tie、完整
   sequence split/禁 digest 交叉测试全部通过；
7. Supervisor 最终 `NO VETO`。

即使 1--7 全部成立，本阶段也只证明存在可审计的 traffic ambiguity support，不证明提前规划有
调度收益。Phase 4 仍须用户另行授权并独立验证 H2。若失败，必须报告失败来自 coverage、集合过宽、
support selection、跨 family、tail 或 observation reconciliation，不能用 oracle ceiling 改写。

## 10. 实现、artifact 与审查 Gate

开工前强制 red matrix 至少覆盖：窄 view 删除 family/ID/truth/capability；ordinary 输出对未揭示
truth、metadata、future reveal、family/ID 反事实逐字节不变；ratio0/no-aggregate 精确 history-only；
exact/partial-shard/source/source+destination reconciliation 的 zero row、saturated cap、multiple
optima、infeasible margins、candidate sensitivity 与同 seed 字节确定性；entry `<=8`；full-32
uniform probability witness；五 selector/tie/nested K；oracle exact truth/flags/nearest-zero；在调用或
不调用 oracle、用不同 oracle truth 后 ordinary artifacts 不变且无 module-global oracle cache；
calibration universe/LOFO；全部 metric 手算；raw corruption/recompute；fresh split/digest 禁交叉。

静态与 fresh-process tests 必须断言 Phase3B 模块不导入 `uncertainty.execution`、decoder、Torch、
scheduling，不公开/返回 `Proposal`、`TransferAction`、`TruthTokenId`、commit、prefix、horizon、
robust score、repair/recourse API；summary schema 禁止 completion/oracle_regret/legality/wasted-prefix
字段和 AICCL 调度收益措辞。scenario token 继续 non-executable。违反任一项为 blocker/VETO。

计划 Core-owned 文件：

- `rlccl/uncertainty/ambiguity.py`
- `rlccl/uncertainty/ambiguity_experiment.py`
- `tests/test_phase3b_ambiguity.py`
- `tests/test_phase3b_experiment.py`
- `scripts/run_phase3b_ambiguity.py`
- `outputs/phase3b_ambiguity/{manifest.json,raw_calibration_scores.csv,raw_validation_metrics.csv,raw_case_metrics.csv,raw_sequence_metrics.csv,raw_lofo_calibration_scores.csv,raw_lofo_validation_metrics.csv,raw_lofo_test_metrics.csv,raw_dependence_metrics.csv,summary.json}`

Main-owned：本协议、Phase 3B 结果报告、任务账本/决策/风险日志。Supervisor-owned：
`docs/agent_coordination/SUPERVISOR_REVIEW_PHASE_3B.md`。`summary.json` 必须保持
`gate_status="PENDING_SUPERVISOR"`，最终条件 7 只写入 Supervisor 报告。

执行顺序固定：Main 审计与协议 → Supervisor preflight → Core red tests → Main 验收 red →
Core implementation → Core/Main green 与 toy artifacts → Supervisor 正式运行准入 → frozen formal
run → Main raw 重算/full tests/结果报告 → Supervisor 最终审查 → Main 向用户报告并停止。
Supervisor 的任何 `HOLD` 必须闭环后才能继续；不得以已有绿色测试绕过。

## 11. Supervisor formal-run VETO return：artifact integrity schema v2

本节由正式运行准入审查的 `HOLD / VETO` 触发，覆盖本协议前文中较宽松的 artifact/recompute
表述。schema 固定升级为 `schema_version=2`。旧四文件 toy green、旧 77-case green 和旧源码 hash
均不是 formal 准入证据；必须重新 red/green、重新冻结 hash、重新申请 Supervisor 准入。

### 11.1 十个唯一正式 artifact 与精确行宇宙

正式输出只能包含下列十个文件；所有 CSV 使用稳定列顺序、UTF-8、LF 和无 NaN/Inf 表示：

1. `manifest.json`；
2. `raw_calibration_scores.csv`：15 calibration sequence × 320 unknown cases = `4,800` 行；
3. `raw_validation_metrics.csv`：15 validation sequence × 320 unknown cases × 4 ordinary methods
   = `19,200` 行；
4. `raw_case_metrics.csv`：15 test sequence × 16 checkpoints × 5 modes × 5 ratios × 4 K ×
   5 methods = `120,000` 行；
5. `raw_sequence_metrics.csv`：15 test sequence × 5 methods × 4 K = `300` 行，每行只聚合
   ratio `<1` 的 320 cases；
6. `raw_lofo_calibration_scores.csv`：5 held-out family × 12 seen-family calibration sequence ×
   320 cases = `19,200` 行；
7. `raw_lofo_validation_metrics.csv`：5 folds × 12 seen-family validation sequence × 320 cases ×
   4 ordinary methods = `76,800` 行；
8. `raw_lofo_test_metrics.csv`：5 folds × 3 held-family test sequence × 320 cases ×
   `{selected,random_comparator}` 两个 role = `9,600` 行；
9. `raw_dependence_metrics.csv`：15 test sequence × 16 frozen checkpoint totals = `240` 行；
10. `summary.json`。

不得只把 validation、calibration、LOFO 或 dependence 的派生结论塞进 manifest。上述 raw 表是其
唯一可审计输入。fit matrices 不另存；fit/LOFO normalizer 由冻结 sequence specs 可重新生成并以
digest 绑定，正式运行当场从内存 truth 重算 raw 值。任何缺行、增行、重复/未知 identity、跨 split
污染或等长 identity 替换立即失败。

八张 raw 表的 canonical identity tuple 固定如下，tuple 按列出的 typed value 做升序排列；
`fold_id` 固定为 `lofo-{held_family_index}-{held_out_family}`：

- calibration：`(record_index,checkpoint,mode_index,stage_index)`；
- validation：`(record_index,checkpoint,mode_index,stage_index,method)`；
- test case：`(case_index,method)`；
- test sequence：`(record_index,method,requested_k)`；
- LOFO calibration：`(fold_id,record_index,checkpoint,mode_index,stage_index)`；
- LOFO validation：`(fold_id,record_index,checkpoint,mode_index,stage_index,method)`；
- LOFO test：`(fold_id,record_index,checkpoint,mode_index,stage_index,role)`；
- dependence：`(record_index,checkpoint)`。

LOFO 三表必须同时保存 `fold_id` 与 `held_out_family`。calibration/validation row 的 `family` 必须
不等于 held-out family；test row 的 `family` 必须等于 held-out family。同一 seen sequence 出现在
四个不同 fold 是四个不同 identity，不能凭 sequence ID 或 normalizer digest 合并。

### 11.2 精确列 schema、数值域与 provenance 绑定

类型记号：`s` string/enum，`i` strict integer（bool 禁止），`f` finite float，`b` strict boolean，
`f?` finite float 或 null。每张 CSV 的 header 必须与下列 ordered tuple **完全相等**；缺列、未知列、
换序均失败。

```text
raw_calibration_scores = (
 case_id:s, sequence_id:s, split:s, family:s, base_seed:i, record_index:i,
 checkpoint:i, mode_index:i, reveal_mode:s, stage_index:i, reveal_ratio:f,
 actual_entry_fraction:f, construction_seed:i, score:f, sequence_digest:s,
 generator_config_digest:s, topology_digest:s, normalizer_digest:s,
 observation_digest:s, ambiguity_digest:s)

raw_validation_metrics = (
 case_id:s, sequence_id:s, split:s, family:s, base_seed:i, record_index:i,
 checkpoint:i, mode_index:i, reveal_mode:s, stage_index:i, reveal_ratio:f,
 actual_entry_fraction:f, requested_k:i, construction_seed:i, method:s,
 replicate_count:i, nearest_rms_distance:f, uses_oracle:b, upper_bound_only:b,
 sequence_digest:s, generator_config_digest:s, topology_digest:s,
 normalizer_digest:s, observation_digest:s, ambiguity_digest:s, support_digest:s)

raw_case_metrics = (
 case_id:s, case_index:i, sequence_id:s, split:s, family:s, base_seed:i,
 record_index:i, checkpoint:i, mode_index:i, reveal_mode:s, stage_index:i,
 reveal_ratio:f, actual_entry_fraction:f, requested_k:i, construction_seed:i,
 method:s, replicate_count:i, nearest_rms_distance:f,
 nearest_matrix_l1_distance:f, covering_radius:f, mean_pairwise_diversity:f,
 duplicate_fraction:f, actual_k:i, component_coverage:f, joint_coverage:f,
 physical_normalized_mean_width:f, zero_physical_range_components:i,
 total_tail_events:f, total_tail_hits:f, group_tail_events:f, group_tail_hits:f,
 hotspot_events:f, hotspot_hits:f, invalid_or_empty:f,
 construction_seconds:f, selector_seconds:f, uses_oracle:b, upper_bound_only:b,
 sequence_digest:s, generator_config_digest:s, topology_digest:s,
 normalizer_digest:s, observation_digest:s, ambiguity_digest:s, support_digest:s)

raw_sequence_metrics = (
 sequence_id:s, split:s, family:s, base_seed:i, record_index:i, method:s,
 requested_k:i, raw_case_count:i, nearest_rms_distance:f,
 total_tail_events:f, total_tail_hits:f, total_tail_recall:f?,
 group_tail_events:f, group_tail_hits:f, group_tail_recall:f?,
 hotspot_events:f, hotspot_hits:f, hotspot_recall:f?, sequence_digest:s,
 generator_config_digest:s, topology_digest:s, normalizer_digest:s)

raw_lofo_calibration_scores = (
 case_id:s, fold_id:s, held_out_family:s, sequence_id:s, split:s, family:s,
 base_seed:i, record_index:i, checkpoint:i, mode_index:i, reveal_mode:s,
 stage_index:i, reveal_ratio:f, actual_entry_fraction:f, construction_seed:i,
 score:f, sequence_digest:s, generator_config_digest:s, topology_digest:s,
 normalizer_digest:s, observation_digest:s, ambiguity_digest:s)

raw_lofo_validation_metrics = (
 case_id:s, fold_id:s, held_out_family:s, sequence_id:s, split:s, family:s,
 base_seed:i, record_index:i, checkpoint:i, mode_index:i, reveal_mode:s,
 stage_index:i, reveal_ratio:f, actual_entry_fraction:f, requested_k:i,
 construction_seed:i, method:s, replicate_count:i, nearest_rms_distance:f,
 uses_oracle:b, upper_bound_only:b, sequence_digest:s,
 generator_config_digest:s, topology_digest:s, normalizer_digest:s,
 observation_digest:s, ambiguity_digest:s, support_digest:s)

raw_lofo_test_metrics = (
 case_id:s, fold_id:s, held_out_family:s, role:s, sequence_id:s, split:s,
 family:s, base_seed:i, record_index:i, checkpoint:i, mode_index:i,
 reveal_mode:s, stage_index:i, reveal_ratio:f, actual_entry_fraction:f,
 requested_k:i, construction_seed:i, method:s, replicate_count:i,
 nearest_rms_distance:f, uses_oracle:b, upper_bound_only:b, sequence_digest:s,
 generator_config_digest:s, topology_digest:s, normalizer_digest:s,
 observation_digest:s, ambiguity_digest:s, support_digest:s)

raw_dependence_metrics = (
 case_id:s, sequence_id:s, split:s, family:s, base_seed:i, record_index:i,
 checkpoint:i, total_traffic:f, sequence_digest:s, generator_config_digest:s,
 topology_digest:s)
```

`role` 只允许 `selected/random_comparator`；LOFO `requested_k` 恒为 8。validation/LOFO validation
只允许四个 ordinary methods、ratio `<1`、K=8；calibration 表无 method/K/support。test random 在
unknown ratio 的 `replicate_count=8`，ratio1 singleton 和其余 method 为 1；LOFO random role 为 8，
selected role 若其 method 为 random 也为 8，否则为 1。random support digest 是按 replicate index
`0..7` 顺序连接八个 canonical support bytes 后的 SHA-256；其他 support digest 是单 support bytes。

最终 manifest 的 top-level key set 也必须精确为：

```text
(schema_version, protocol_sha256, artifact_names, artifact_logical_sha256,
 artifact_scientific_sha256, combined_scientific_evidence_sha256,
 authorized_source_sha256, families, base_seeds, splits, sequence_length,
 max_entry, history_window, checkpoints, reveal_modes, reveal_ratios,
 requested_k, random_replicates, sequence_specs, sequence_records,
 h1_exclusion_manifest_sha256, h1_excluded_sequence_digests, topology,
 normalizer_digest, group_coefficients_digest, lofo_fold_normalizer_digests,
 calibration_radius, selected_method, selected_k, validation_method_means,
 lofo_fold_evidence, test_total_traffic_dependence, gate_evidence, data_status,
 summary_sha256, environment)
```

`authorized_source_sha256` 的 key set 固定且仅为
`rlccl/uncertainty/ambiguity.py`、`rlccl/uncertainty/ambiguity_experiment.py`、
`scripts/run_phase3b_ambiguity.py`。`environment` 的 key set 固定为
`python/python_executable/numpy/platform`；这四项全部排除于 scientific-evidence digest，但仍进入
manifest 正常 JSON 与人工审计。manifest 不保存自己的 digest，避免自引用。

final `summary.json` 的 top-level key set 固定且仅为：

```text
(schema_version, protocol_sha256, selected_method, selected_k,
 calibration_radius, validation_method_means, gate_evidence,
 test_total_traffic_dependence, raw_row_counts, data_status, gate_status,
 conditions_evaluated, failed_conditions, insufficient_conditions,
 combined_scientific_evidence_sha256)
```

`raw_row_counts` 必须精确映射八个 raw filename 到第 11.1 节行数；`gate_status` 恒为
`PENDING_SUPERVISOR`。CSV strict bool 只接受小写 `true/false`，nullable float 的 null 只接受空 cell；
strict integer lexical form 是无 `+`、无前导零（零本身除外）的 base-10。`construction_seed` 就是
本协议冻结的 reveal/construction seed，两者不另存两个可分叉字段。

每个适用 raw row 必须记录上述完整 provenance 和 metric，且 case ID 必须由对应 canonical identity
机械生成；不得用任意 opaque ID 掩盖重复 identity。

validator 必须 fail closed：

- bool 不作为 integer 接受；integer、float、enum、ID 与精确 seed 公式逐字段验证；
- 所有 numeric 必须 finite；distance/diversity/radius/width/timing 非负；coverage、duplicate、fraction
  与 recall 在 `[0,1]`；event/hit 满足 `0<=hit<=event`，无 event 时 recall 为 null，否则必须精确等于
  `hit/event`；ratio1 actual K 为 1，其余 actual K 等于 requested K；
- digest 必须匹配 lowercase hexadecimal `[0-9a-f]{64}`，不能只查长度；
- 每行 sequence/config/topology/normalizer digest 必须与 manifest 对应记录一致；同一个 canonical
  construction coordinate 的 observation/ambiguity digest 必须跨 method/K 一致；ordinary/oracle
  flags、support digest 和 role 必须满足冻结隔离规则；
- manifest 的 protocol/source/topology/H1-exclusion/normalizer/group-coefficient digest 必须与实际
  冻结文件或由其机械重算的值相等，不能只与 raw 自洽；
- raw CSV 的 canonical logical digest 与 summary 的 canonical digest 写入 manifest 并在重算时核对。

generator config 使用完整 canonical `asdict(config)` 的 SHA-256；不得以 sequence ID 代替。正式
destination 已存在（即使为空）时
runner 必须失败，不覆盖；十文件先写 sibling staging directory、完成 read-back/recompute 后再一次性
发布，避免半写 artifact 被误认正式结果。

### 11.3 Canonical logical/scientific digest 算法

所有 digest 必须由独立 reference tests 按下述规范计算，不能由 production helper 自己生成 expected：

1. CSV 先按本节 exact schema 做 typed parse 和 domain validation，再按该表 canonical identity 排序；
   identity 必须唯一。
2. canonical scalar 编码固定为：null→`["n",null]`；strict bool→`["b","true"]` 或
   `["b","false"]`；strict int→`["i","<base10>"]`；finite float→`["f","<hex>"]`，其中
   `-0.0` 先归一成 `0.0`，`<hex>` 为 Python/IEEE-754 binary64 `float.hex()` 的 lowercase 输出；
   string→`["s","<原UTF-8字符串>"]`。container 编码为 list→`["l",[...]]`；mapping→
   `["m",[[key,encoded_value],...]]`，key 必须 string 并按 Unicode code point 升序。
3. 表的 canonical payload 是
   `["table-v1",table_name,[ordered_column_names],[rows_as_ordered_encoded_scalars]]`；以
   `json.dumps(...,ensure_ascii=False,separators=(",",":"),allow_nan=False)` 编码 UTF-8，**无**末尾换行，
   再做 SHA-256。logical digest 使用全部列，包括 timing。
4. scientific table digest 使用同一算法，但只从 `raw_case_metrics` 删除
   `construction_seconds/selector_seconds` 两列；其余七表不删除任何列。不得删除 case identity、
   provenance 或 replicate count。
5. `artifact_logical_sha256` 精确映射八个 CSV logical digest 与 `summary.json` 的 canonical-object
   digest；`artifact_scientific_sha256` 只映射八个 CSV scientific digest。combined scientific digest
   的 payload 为按第 11.1 节 raw 文件顺序排列的
   `["phase3b-scientific-v1",[[filename,digest],...]]`，使用同一 JSON/UTF-8/SHA-256规则。
6. summary canonical-object digest 对解析后的 JSON 使用第 2 项递归 scalar/container 编码；不包含
   文件空白或 key 原顺序。`summary_sha256` 必须等于它。manifest 的四个 environment 字段和 raw timing
   是全部且唯一的 scientific determinism exclusions；完整 logical/file artifact 不要求跨机器字节一致。

### 11.4 唯一重算链

`recompute_artifacts` 不得信任 manifest 的派生数字。它必须从九类持久化输入（八个 raw CSV 加
manifest 中仅作为 provenance 的冻结 specs/digests）机械执行并比较：

1. 从 4,800 calibration scores 重算两层 `higher` radius；
2. 从 19,200 validation rows 按 15 sequence 等权重算四 method mean、tie 与唯一 selected method，
   并验证 `K=8`/unknown-only/oracle-free；
3. 从 120,000 test rows 对每个 `(sequence,method,K)` 的 320 unknown cases调用同一聚合定义，逐字段
   比较全部 300 sequence rows：nearest distance、raw count、total/group/hotspot event、hit 和 recall；
4. 从 selected/random `K=8` sequence rows 重算 15 paired deltas、3 base-seed mean、5 family mean 与
   固定 10,000 次 bootstrap CI；
5. 每个 LOFO fold 从 19,200 fold calibration rows重算 radius，从 76,800 fold validation rows重算
   selector，从 9,600 held-family test rows重算每 sequence selected/random mean、family delta、aggregate
   delta 和 relative degradation；held family 出现在 fit/calibration/validation 任一 provenance 即失败；
6. 从 240 checkpoint totals 逐 sequence 重算 lag-1 ACF、positive-sequence ESS 与 aggregate；
7. 从 test raw 重算 joint/component coverage、width、tail/group/hotspot、invalid/empty、ratio1 singleton、
   timing finite/nonnegative和全部 Gate evidence；oracle rows不得进入 ordinary Gate；
8. 用重算 evidence 重新执行条件 1--6、`data_status` 与 `gate_status=PENDING_SUPERVISOR`，逐字段/逐字节
   比较 manifest 的 selected method/K、radius、validation/LOFO/dependence/gate evidence、data status，
   再比较 expected `summary.json`。任何 coordinated manifest+summary tampering 只要 raw 未同样提供对应
   auditable evidence，必须失败。

`integrity_checks_complete/passed` 的发布状态机固定为：

1. 在 destination 的同一 parent 下创建唯一 sibling staging directory；destination 此时必须不存在；
2. 生成八表，构建 `integrity=false/false` 的 provisional manifest/summary，写入十文件；
3. 从 staging **重新读取文件字节**，执行第 11.4 节 1--8（condition 6按 provisional false处理，但
   其余重算与所有 digests 必须通过）；内存原对象不能代替 read-back；
4. 仅在 provisional read-back 成功后，用完全相同 raw 将 integrity 改为 `true/true`，重算 Gate、final
   summary、summary digest、raw logical/scientific/combined digests，materialize final manifest/summary；
5. 再从 staging 重新读取最终十文件，执行完整第 11.4 节 1--8，此次 condition 6 和 final digests/
   expected summary 也必须通过；最终文件写入后不得再变更任何字节；
6. 关闭全部 handle，重新确认 destination 不存在，再做同 volume 单次 directory rename。任何失败
   不得创建/覆盖 destination；staging 不是正式 evidence并须以 `.phase3b-staging-` 前缀明确标识。

重算器必须返回独立新对象，不能把 manifest 派生 map 原样回传。final read-back 未通过时，禁止把
integrity 标为 true 的内存对象直接返回给调用者。

### 11.5 VETO-return red matrix 与重新准入

Core 必须先新增并运行真实失败测试，逐类改变：每个 Gate-affecting test raw 字段；每个 sequence
aggregate count/hit/recall；calibration score/radius；validation selected method/K/tie；LOFO fold/role/
delta/degradation；dependence total/ACF/ESS；data status；protocol/source/topology/normalizer/config/sequence/
observation/ambiguity/support digest；以及联动 manifest gate evidence + summary。至少一个测试构造精确
`120,000/300` formal universe并证明同样 fail closed。测试还必须永久比较五种 mode × 五 stage 的
formal observation 与 Phase 1 `DemandRevealProcess`。所有 corruption test 必须保留原 expected artifact
时被拒绝，不能以删字段触发替代语义篡改。

Main 独立验收 red 后才开放 schema v2 实现；Core/Main 重新跑 focused、full、toy 双跑、parity、完整
universe 与 corruption tests，冻结新 protocol/test/source hashes，再由 Supervisor 进行第二次正式运行
准入。Supervisor 再次 `ALLOW / NO VETO` 前，正式 corpus/output 仍禁止生成。
