# Gate H1：历史可预测性验证协议

## 1. 目标与停止边界

本 Gate 只回答：在预测阶段 `t` 的流量矩阵 `X_t` 时，完整历史
`X_0, ..., X_{t-1}` 是否提供相对于 previous-value 的稳定、可校准、跨流量族增量信息。
H1 不训练调度策略，不进入 robust prefix / rolling recourse，不使用当前 reveal，
不据此声称调度收益。Gate 完成后由 Supervisor 给出 `ALLOW/HOLD/VETO`，Main
向用户报告并停止；未经用户再次批准不进入 Phase 3A/3B。

## 2. 信息边界

- 所有普通 predictor 的输入只能由 `X_{<t}` 构造。
- 禁止输入 `X_t` 的任意 entry、当前 reveal/mask、未来 regime/shock/hotspot 标签、
  generator latent state、完整 sequence metadata 或未来统计量。
- `X_t` 只允许出现在训练标签、校准残差与事后评估中。
- oracle/current-truth 只可作为明确标注的 evaluator-only 上界，不参与模型选择或
  H1 通过判定。
- bandwidth-group load 定义为公开 topology 上确定性最短路的 offered-load proxy；
  不称为 learned schedule 的真实 group utilization。

## 3. 数据与完整 sequence 划分

正式实验固定使用 `Rear4GPU`、sequence length `1024`、数据 seeds
`42/142/242`，覆盖五个 long-horizon family：

1. `regime_switching_long`
2. `stochastic_volatility`
3. `rare_shock_recovery`
4. `hotspot_random_walk`
5. `same_moments_different_dynamics`

生成配置冻结为 `mean_level=2.0`、`std_level=1.5`、`max_entry=8`、
`calibration_candidates=1`，其他 `LongHorizonTrafficConfig` 字段使用当前 dataclass
默认值并将完整 `asdict(config)` 写入 manifest；不得使用旧精确周期 generator。
family index 与 seed index 均按上面和 `42/142/242` 的书写顺序从 0 开始，实际 seed 为
`base_seed + family_index * 1_000_000 + sequence_index * 10_000`。
`same_moments_different_dynamics` 的 variant 固定为
`SAME_MOMENT_VARIANTS[(seed_index + sequence_index) % 4]`。

每个 `(family, seed)` 生成 5 条互异完整 sequence，按 `sequence_index` 固定为：

- 0、1：fit；
- 2：validation，仅用于候选模型选择；
- 3：calibration，仅用于概率校准；
- 4：test，仅用于一次性 Gate 评估。

因此正式 pooled 评估包含 30 条 fit、15 条 validation、15 条 calibration、15 条
独立 test sequence。任何相邻时间步不得跨 sequence 拆分；四个 split 的 canonical
sequence ID/digest 必须互斥并写入 manifest。

所有方法共同评估 `t=8,...,1023`，即每条 sequence 1016 个预测点；最近历史长度为 8，
moment 最大历史窗为 16（`t<16` 时只用已有历史）。输入 standardizer、目标
standardizer、long-term mean、ridge、MLP 与 TCN 的所有拟合统计只能来自 fit split。

另做五折 leave-one-family-out（LOFO）：每折从 fit/validation/calibration 中完全排除
目标 family，再只在该 family 的 3 条 test sequence 上评估。LOFO 不替代 pooled
主检验。

## 4. 预测目标

至少输出以下目标；连续向量不得只报告整体拼接后的一个数：

- total traffic；
- source-load vector；
- destination-load vector；
- hotspot destination（分类）；
- hotspot strength；
- sparsity；
- bandwidth-group offered-load vector。

hotspot destination 的普通预测由模型预测的 destination-load vector 的 `argmax`
得到；previous-value 使用上一阶段 hotspot。并列时用最小 node index，确保可复现。

## 5. 必需方法

正式表中必须同时出现，且不得用不相符的名称包装简单线性模型：

1. `long_term_mean`：只用 fit labels 的均值；
2. `previous_value`：`X_{t-1}` summary；
3. `ewma`：固定 `alpha=0.30`，初值为 `summary(X_0)`，随后
   `E_k=0.30*summary(X_k)+0.70*E_(k-1)`；预测 `X_t` 时只输出 `E_(t-1)`；
4. `moment_only`：16-step 最大历史窗 entry-wise mean/variance 的 multi-output ridge，
   `alpha=10.0`；
5. `recent_history_mlp`：最近 8 个 summary 展平后的标准化 MLP，固定
   `hidden_layer_sizes=(32,)`、`activation=tanh`、`solver=adam`、`alpha=1e-4`、
   `batch_size=256`、`learning_rate_init=1e-3`、`max_iter=80`、无 early stopping；
6. `causal_tcn`：最近 8 个 summary 上的单层可学习 causal Conv1D，固定 kernel size 3、
   hidden channels 8、共享 kernel、`tanh`；输出表示拼接最后一个 hidden 与所有 causal
   positions 的 mean，再接可学习线性 head。NumPy Adam 固定 40 epochs、batch 256、
   learning rate `5e-3`、L2 `1e-4`、`beta1=0.9`、`beta2=0.999`、`epsilon=1e-8`；
7. `quantile_scenario`：先在 validation 上从 MLP/TCN 中选择 point backbone，随后
   仅用 calibration sequence 的 signed residual 构建 10/50/90% quantile
   与 joint residual scenarios。

当前可运行解释器没有 Torch，因此正式 TCN 必须是可审计的 NumPy 实现。若无法提供
真正可学习的 causal convolution、梯度更新和相应单测，H1 保持 `HOLD`；不得把随机
固定特征、ridge AR 或 MLP 改名为 TCN。模型随机种子和超参数写入 manifest。
MLP、TCN、mini-batch shuffle、scenario 与 bootstrap 的统一 seed 为 `20260731`；TCN
使用该 seed 的 Glorot-uniform 初始化。loss 为标准化连续 target 的 mean squared error
加所述 L2，训练轮数固定，validation/test 不参与 early stopping 或调参。

## 6. 指标与原始统计单位

点预测逐 method、target、family、seed 报告：MAE、RMSE、R²、Spearman；另报
hotspot accuracy。主 paired effect 定义为每条 test sequence 上
`delta_rmse = RMSE(previous_value) - RMSE(method)`，正值表示优于 previous-value。
scalar target 的每 sequence RMSE 对 1016 个时间点计算；vector target 的每 sequence
RMSE 对该 sequence 的全部时间点和该 target 的全部 components 等权计算。每条
sequence 在总体 effect 中等权，不按 step 数或 ESS 加权。

概率预测至少报告：

- 80% quantile interval empirical coverage：point backbone 加 calibration signed
  residual 的逐 component 10%/90% empirical quantile；point metric 使用 50% residual；
- interval width；
- calibration error `abs(coverage - 0.80)`；
- scenario coverage：对每个 test prediction，从全部 calibration rows 中以 seed
  `20260731 + stable_test_example_index` 有放回抽取 64 个完整 residual vectors，保持
  target components 的 joint residual；64 个 scenarios 的逐 component 10%/90%
  empirical quantile构成 central envelope；
- tail-event recall。tail event 预先定义为 total traffic 超过 fit 集 total 的 90%
  quantile；预测事件定义为 quantile predictor 的 total 90% 上界超过同一阈值。事件数
  在 pooled test 统计；若不足 10，记为 `insufficient_events`，不得伪造 recall。

empirical quantile 统一使用 NumPy `method="linear"`。calibrator 不接收 family/seed 或
latent metadata，不构造 family-specific residual pool；family/seed 只允许在全部预测
完成后由 evaluator 用于分组报告。

每条 test sequence 是独立统计单元。95% CI 使用完整 sequence 的 paired cluster
bootstrap（固定 10,000 次、seed `20260731`）：先计算 15 条 sequence delta，再在
每个 family 的 3 条 test sequence 内有放回抽取 3 条，合并为每次 15 条等权均值；
CI 为 10,000 个均值的 NumPy `method="linear"` 2.5%/97.5% percentile。不得把 step
当独立样本。每条 test
sequence 另报 total traffic lag-1 ACF 和 positive-sequence ESS，并汇总独立 sequence
数、raw step 数和 ESS。ESS 从 lag 1 起累加正 ACF，到第一个非正 ACF 为止，使用
`n / (1 + 2 * sum(rho_lag))`，最大 lag 64。

## 7. 预注册的 H1 通过判据

候选 `recent_history_mlp` 与 `causal_tcn` 只能依据 validation 集选出一个
`selected_recent`：先在每条 validation sequence 上计算 total RMSE，再比较 15 条
sequence 的等权 mean；差值绝对值不超过 `1e-12` 时固定选择
`recent_history_mlp`。test 结果不得改变选择。

每个 LOFO fold 必须从 fit/validation/calibration 完全排除 held-out family，重新拟合
standardizer、两候选模型、按 seen-family validation 重新选择、再只用 seen-family
calibration residual 校准，最后评估 held-out family 的 3 条 test sequence。family
identity 不是模型特征。H1 仅在以下条件全部满足时通过：

1. `selected_recent` 在 pooled test 的 total traffic paired `delta_rmse` 95% CI 下界
   严格大于 0；
2. total traffic 的 seed-level（这里 seed 明确指生成配置的 `base_seed`）mean delta在
   3/3 base seeds 均为正，family-level mean
   delta 至少 4/5 为正；
3. 预注册的三个 primary target block：total traffic、source-load vector、
   destination-load vector，三者在 pooled sequence-bootstrap paired delta CI 下界均
   严格大于 0；其他连续 targets 为次要报告，不参与事后择优；
4. LOFO aggregate mean delta 非负，至少 3/5 held-out families 为正，且至多一个
   family 相对 previous-value 的 RMSE 恶化超过 10%。每个 family 的相对变化固定为
   `mean_sequence_RMSE(selected) / mean_sequence_RMSE(previous) - 1`，其中三条
   base-seed test sequence 等权；
5. `quantile_scenario` 的 total 80% interval：overall calibration error 不超过 0.05，
   每个 family 不超过 0.10；overall scenario coverage calibration error 不超过 0.10；
6. test tail event 至少 10 个时，tail-event recall 不低于 0.70；不足 10 个时 H1
   保持 `HOLD`，不能按通过处理；
7. 历史-only、完整 sequence split、determinism、无 NaN/Inf、原始
   rows/manifest/summary 完整性测试全部通过；TCN 测试必须包含代表性参数的 finite-
   difference 梯度核对、kernel 训练后确实更新、合成 causal-lag 任务 loss 下降、因果
   反事实、`.npz` round-trip 与同 seed 确定性；且 Supervisor 无 veto。

任何一个条件失败，H1 判为 `FAIL`（若是证据/实现不完整则为 `HOLD`），下一阶段建议
转向 Phase 3B prediction-free robust route；该建议不自动授权实施。

## 8. 输出与可复现性

Core-owned 输出固定为：

- `rlccl/prediction/**`
- `scripts/run_h1_predictability.py`
- `tests/test_h1_predictability.py`
- `outputs/h1_predictability/manifest.json`
- `outputs/h1_predictability/raw_sequence_metrics.csv`
- `outputs/h1_predictability/raw_probability_metrics.csv`
- `outputs/h1_predictability/summary.json`

`raw_sequence_metrics.csv` 必须同时区分 pooled 与 LOFO scope，并保留逐完整 sequence、
method、target 的点指标、hotspot accuracy、ACF/ESS 和可重算的 previous-value 配对键；
`raw_probability_metrics.csv` 必须按完整 sequence/target 保留 coverage numerator、
denominator、width sum、scenario coverage numerator/denominator、tail event/true-positive
count。summary 的 delta、bootstrap、coverage、LOFO 和条件 1--6 必须由这两份 raw 文件
与 manifest 重算，不得只信任预聚合字段。

正式实验生成时 Supervisor 尚未完成最终审查，因此 `summary.json` 只能写
`gate_status="PENDING_SUPERVISOR"` 及条件 1--6 的数据判定；不得预填
`supervisor_veto=false` 并宣称最终 PASS。条件 7 和最终 H1 Gate 只在 Main 完整测试、
Supervisor 独立复核后确定。

Main-owned 协调/协议文件不得由 Core 脚本覆盖。Supervisor 独立报告固定为
`docs/agent_coordination/SUPERVISOR_REVIEW_GATE_H1.md`。
