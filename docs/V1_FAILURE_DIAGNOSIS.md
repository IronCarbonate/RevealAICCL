# V1 失败诊断

日期：2026-07-27（Asia/Shanghai）  
范围：Phase A–C 的代码、生成器审计和真实实验结果；不引入新的策略结构，不用最好 seed 代替整体结果。

## 1. 执行摘要

V1 的失败不能再主要归因于“旧流量生成器太简单”。旧生成器确实存在严重缺陷，但在补充长期非周期流量、反事实控制、严格 history-only 预测和 partial-demand 实验后，证据一致指向更直接的问题：**在完整当前 demand 已知时，把压缩的历史 moments 注入细粒度动作打分，会改变本来可行的调度决策，却没有提供足以补偿这种扰动的当前状态信息。**

最强证据是 C2：只改变历史、不改变当前 X、topology 或初始状态时，Moment-full 在 100% pair 上改变完整 schedule，60.83% 构成有害的 action-level context interference；平均 completion 显著变差 0.6925。C3 又说明 moment-only 对当前流量的预测明显弱于 previous-value 和有序 recent-history。C4 表明即使当前需求部分缺失，现有 moments 也没有跨 seed 稳定收益。

原正式 V1 的 `NO_GO`、等配置重建后的多 seed 结果、训练/held-out family 分桶、C2–C4 因而形成相互独立但方向一致的证据链。唯一形式上稳定的正向 family 桶是训练族 `sparse_switching`，只有 3 条独立 sequence 且仅 2/3 seed 为正，不能支撑扩大 MomentEncoder 或重新训练同一路线。

## 2. 现有生成器审计结论

旧 `rlccl/traffic/process_generator.py` 不是长期不确定流量生成器，而是 2/16 步整数模板的精确重复：

- 正式审计覆盖 6 个 family、长度 64/1024/4096、3 个 base seed、每组 20 条，共 1,080 条 sequence；1,080/1,080 第一次生成成功。
- `alternating_burst` 的最短精确矩阵周期恒为 2，其余 family 恒为 16。
- 长度 4096 时 exact duplicate ratio 为 0.996–0.9995；增长 `sequence_length` 没有增加潜在动力学长度。
- 16/128/512 窗口的 moment violation 都为 0，因为 128/512 只是短 period 的整数倍；这不是长期稳定性的经验证据。
- 没有随机长 regime、随机 volatility、rare shock/recovery 或随机 hotspot dwell；`heavy_tail_clipped` 的尾部也被固定模板、clip 和整数化显著削弱。
- 正式 V1 训练数据完全来自该合成生成器，不是生产流量 trace。训练长度 64 对大多数 family 只有 4 个独立 period。

因此，旧数据集是可信的放大因素：它鼓励策略适应短模板，并夸大重叠时间步数量。但它不是充分解释，因为后续反事实实验在当前问题完全相同时仍观察到历史引发的有害动作变化。

完整证据见 `docs/TRAFFIC_GENERATOR_AUDIT.md` 和 `outputs/traffic_audit/audit_summary.json`。

## 3. 新增长期生成器验证

Phase B 新增了独立的 `LongHorizonTrafficConfig` 和 5 个 long-horizon family：

- `regime_switching_long`：32–512 步随机 regime dwell；
- `stochastic_volatility`：calm/volatile 随机切换；
- `rare_shock_recovery`：低概率 shock、4–16 步持续及指数恢复；
- `hotspot_random_walk`：8–96 步随机 hotspot dwell 和随机迁移；
- `same_moments_different_dynamics`：长期 mean/variance 接近但 ACF、tail 和动力学不同的四种变体。

正式审计覆盖长度 1024/4096、3 个 base seed、每组 20 条，共 600 条 sequence：

- 600/600 生成成功，0 条检测到精确周期；
- medium/long 的 total 和 matrix soft constraints 全部在允许 violation fraction 内；
- rare shock 的 16 步 mean 最大相对误差 1.76、variance 最大相对误差 8.89，证明短时偏离没有被整条 rejection 过滤；
- family 在 lag-1/8/32 ACF、tail、sparsity、hotspot concentration 和 dwell 上明显可分；
- same-moments 四种变体的长期总量 mean/variance 几乎相同，但 1024 步 lag-8 ACF 约为 0.788、0.173、0.891、0.643。

新生成器解决了旧生成器的长期动态缺陷，并为 C2–C4 提供非周期流量。其 latent metadata 仅用于审计/评估，没有泄漏给 policy。

完整证据见 `docs/LONG_HORIZON_TRAFFIC_VALIDATION.md` 和 `outputs/traffic_audit/long_horizon/audit_summary.json`。

## 4. 每个 family 的 V1 结果

下表为等配置重建的 3 个 V1 seed 在独立训练-family/held-out-family sequence 上的 Moment-full paired 结果。正 delta 表示比相同样本的 baseline 少用 completion slot；CI 以完整 sequence 为 cluster。

| family | distribution | raw rows | independent sequences | mean delta | sequence-bootstrap 95% CI | positive seeds | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| `alternating_burst` | train | 576 | 3 | 0.0000 | [0.0000, 0.0000] | 0/3 | 无收益 |
| `moving_hotspot` | train | 576 | 3 | -0.2587 | [-0.2812, -0.2448] | 0/3 | 稳定变差 |
| `smooth_ar` | train | 576 | 3 | 0.0087 | [-0.1042, 0.1094] | 1/3 | 不稳定 |
| `sparse_switching` | train | 576 | 3 | 0.0399 | [0.0260, 0.0469] | 2/3 | 小样本形式稳定，但非全 seed 同向 |
| `bimodal` | held-out | 1,920 | 10 | -0.1828 | [-0.3156, -0.0505] | 0/3 | 稳定变差 |
| `heavy_tail_clipped` | held-out | 1,920 | 10 | -0.1797 | [-0.2052, -0.1552] | 1/3 | 稳定变差 |

原正式 3-seed 汇总也给出 `NO_GO`：mean degradation relative 为 2.11%，两个 held-out family 均未满足 tail improvement。等配置重建的 held-out overall completion delta 分别为 seed 42 `-0.2500`、seed 142 `-0.0117`、seed 242 `-0.2820`；只看 seed 142 的 provisional gate 会掩盖另外两个 seed 的负结果。

`mean-only`、`moment-full`、`moment-shuffled` 和 baseline 的逐条结果均保存在 `outputs/v1_diagnosis/bucket_enriched_detail.csv`，没有只保留最优 ablation。

## 5. 分桶结果

C1 对 24,576 条 method/model 结果、2,048 个独立 traffic 时刻和 32 条完整 sequence 分析了以下维度：family、训练/held-out 分布、current total、current-vs-history mean/variance deviation、sparsity、source/destination hotspot strength、hotspot migration、burst、regime duration、sequence ACF、baseline completion 和 baseline synthesis time。

- 共生成 184 个 condition/method summary。
- Moment-full 只有一个稳定正向桶：`family=sparse_switching`；其独立 sequence 数为最低允许值 3，且一个训练 seed 明显为负。
- 除该 family 标签桶外，没有任何其他条件桶满足 mean delta > 0、cluster-bootstrap CI 下界 > 0、至少 2/3 seed 为正和至少 3 条独立 sequence 的联合判据。
- ACF q1–q4 的 mean delta 分别为 `-0.1384/-0.1782/-0.0885/-0.1217`；最高 ACF 桶 CI `[-0.2884, 0.0104]`。高时间相关性没有带来更好的 moment conditioning 效果。
- 全部 schedule 合法，timeout 为 0；synthesis time 单独统计，Moment 路径在所有 family 上均慢于 baseline。

完整结果见 `docs/V1_BUCKET_ANALYSIS.md` 和 `outputs/v1_diagnosis/bucket_summary.csv`。

## 6. Counterfactual-history 结果

C2 构造 200 个 `History A -> current X` / `History B -> current X` pair，在 3 个训练 seed 上得到 600 条 paired 记录。每个 pair 的当前 X、topology、初始 schedule state 和 ground-truth demands 完全相同；历史 context 在读取当前 X 前生成，不包含未来信息。

| 指标 | 结果 |
|---|---:|
| baseline 两历史等价率 | 100% |
| Moment 完整 schedule 改变率 | 100% |
| action-level context interference | 60.83% |
| 至少一个历史有益 | 11.50% |
| baseline - 平均 Moment completion | -0.6925 |
| pair-bootstrap 95% CI | [-0.7759, -0.6092] |
| 首 slot logits L2 差异均值 | 2.1731 |
| action edit distance 均值 | 33.482 |
| edge-use L1 均值 | 19.095 |
| legality / timeout | 100% / 0% |

baseline completion mean 为 8.4150，两个 Moment history 分别为 9.0817 和 9.1333。baseline synthesis mean 为 38.94 ms，两个 Moment 路径约为 46.26/46.32 ms。该实验提供直接因果证据：历史是唯一变化源，而动作与 completion 均被有害改变。

完整结果见 `docs/COUNTERFACTUAL_HISTORY.md` 和 `outputs/v1_diagnosis/counterfactual_detail.csv`。

## 7. Predictability 结果

C3 使用 45 条完整训练 sequence 和 15 条完全不重叠的测试 sequence。所有非 oracle 方法预测 `X_t` 时只读取 `X_0...X_{t-1}`。

| 方法 | total RMSE | R² | Spearman | vs constant | hotspot accuracy |
|---|---:|---:|---:|---:|---:|
| constant | 4.8097 | -0.0000 | 0.0000 | 0.00% | 42.53% |
| previous value | 1.6747 | 0.8788 | 0.9027 | 65.18% | 78.80% |
| moment-only | 3.1729 | 0.5648 | 0.7060 | 34.03% | 63.59% |
| recent-history | 1.6463 | 0.8828 | 0.9017 | 65.77% | 70.53% |
| oracle current summary | 0 | 1 | 1 | 100% | 100% |

moment-only 优于 constant，却远弱于 previous-value 和有序 recent-history。source/destination load、hotspot strength、sparsity 和 bandwidth-group offered-load proxy 上结论相同。`hotspot_random_walk` 的 moment-only 甚至比 constant 差 1.87%。这与 Phase B 的 same-moments/different-dynamics 结果一致：均值/方差不是充分的时序状态表示。

完整结果见 `docs/TRAFFIC_PREDICTABILITY.md` 和 `outputs/v1_diagnosis/predictability/predictability_summary.json`。

## 8. Partial-demand 结果

C4 使用 15 条独立长序列、480 个真实 traffic 时刻、7 种观测条件、2 个方法和 3 个训练 seed，生成 20,160 条 paired 记录。partial/proxy matrix 只进入 policy observation；真实 demand 清除、状态转移、completion、timeout 和 legality 始终使用 ground truth。

| observation | Moment mean completion | paired delta | sequence-bootstrap 95% CI | positive seeds | stable |
|---|---:|---:|---:|---:|---|
| full | 11.8257 | -0.5326 | [-0.8222, -0.2681] | 1/3 | no |
| partial shards 25% hidden | 17.0528 | -0.1938 | [-0.4139, 0.0195] | 1/3 | no |
| partial shards 50% hidden | 17.6722 | -0.2042 | [-0.3847, -0.0347] | 1/3 | no |
| random entries 25% hidden | 16.6799 | -0.3986 | [-0.7578, -0.1104] | 1/3 | no |
| random entries 50% hidden | 17.4153 | -0.1840 | [-0.3875, -0.0125] | 1/3 | no |
| source + destination totals | 17.0562 | -0.0514 | [-0.2278, 0.1000] | 1/3 | no |
| source totals | 16.1243 | -0.1507 | [-0.2437, -0.0618] | 1/3 | no |

全部结果 legality 为 100%。整体 timeout 为 38.97%，其中 Full baseline/Full moment 为 1.39%/3.54%，partial 条件明显更高；timeout 与 synthesis time 已分开报告。没有一个 partial 条件满足跨 seed 稳定收益，因此路线 B 的触发条件不成立。

完整结果见 `docs/PARTIAL_DEMAND_EXPERIMENT.md` 和 `outputs/v1_diagnosis/partial_demand_summary.json`。

## 9. 数据量、独立统计单位和 effective sample size

| 阶段 | raw 规模 | 主要独立单位 | ACF/ESS 解释 |
|---|---:|---:|---|
| Phase A legacy audit | 1,866,240 traffic steps；1,080 sequences | 1,080 sequences | 形式 ESS 合计约 1,325,482，但序列是精确周期复制，正相关截断 ESS 会被负相关/周期抵消误导；精确 period 比该 ESS 更有解释力 |
| Phase B long audit | 1,536,000 traffic steps；600 sequences | 600 sequences | 平均 lag-1 ACF 0.8776；ESS 合计约 94,089，仅为 raw steps 的 6.13% |
| C1 bucket | 24,576 result rows；2,048 traffic steps | 32 sequences | method 和训练 seed 对同一 traffic 的重复不增加独立样本数；CI 按 sequence cluster |
| C2 counterfactual | 600 model rows | 200 counterfactual pairs | 3 个训练 seed 是对同一 pair 的重复模型评估；CI 按 pair cluster |
| C3 predictability test | 15,240 test steps | 15 sequences | 平均 lag-1 ACF 0.8790；ESS 合计 973.98，为 raw test steps 的 6.39% |
| C4 partial demand | 20,160 result rows；480 traffic steps | 15 sequences | observation/method/model seed 扩展不增加独立流量序列；CI 按 sequence cluster |

这解释了为什么不能用数万条重叠时刻得到极窄置信区间。C1 的 `sparse_switching` 正结果虽有 576 method/model 行，本质上只有 3 条独立 sequence；它应被视为脆弱线索，而不是大样本收益。

## 10. 是否存在训练过拟合证据

结论是：**存在训练分布适配/seed 敏感的迹象，但没有充分证据证明传统意义的 sequence 记忆型过拟合。**

支持“训练分布适配”的证据：

- 唯一正向 family 桶只出现在训练 family `sparse_switching`；两个 held-out family 都显著为负。
- 旧训练生成器只有短周期模板，训练长度 64 的有效动力学多样性很低。
- 三个重建 seed 方向明显不一致，seed 142 单独可得到 provisional gate，而 seed 42/242 为 `NO_GO`。

反对“简单记忆过拟合已被证明”的证据：

- 训练 family 补充评估使用独立 eval seed，不是训练 sequence 本身。
- 另一个训练 family `moving_hotspot` 显著变差，`alternating_burst` 无收益，`smooth_ar` 不稳定；不是所有训练 family 都获益。
- 代码审计未发现 sequence 片段跨 train/test 泄漏；estimator 按 sequence 独立且严格 history-only。

因此，数据模板化和训练样本独立性不足属于可能放大因素；V1 的核心失败仍由 C2 的动作级干扰和 C3 的表示不足更直接解释。

## 11. V1 失败原因的证据排序

### 根本原因

1. **历史 context 对细粒度 action 的直接有害干扰。** C2 在当前问题完全相同的控制实验中得到 60.83% 有害干扰率和显著负 completion delta，是最直接、最强的因果证据。
2. **moment 表示丢失有序时序信息。** C3 中 moment-only 明显落后于 previous/recent-history；Phase B 同时构造出了 moments 接近但 dynamics 不同的流量。均值/方差不能可靠决定当前 action。
3. **任务信息结构不匹配。** V1 在完整 `X_t` 已知时额外注入历史统计；这些统计与 baseline 已有的当前需求特征高度重叠，却能改变 logits。C4 进一步表明现有 context 即使在 partial observation 下也未产生稳定信息价值。

### 可能放大因素

1. **旧训练生成器过度周期化。** 2/16 步模板和严格短窗 bounds 限制了训练动力学多样性。
2. **独立 sequence 数少且时间相关性高。** 原训练每 family 仅 10 条 sequence；长序列测试 ESS 约为 raw steps 的 6%，step-level shuffle 不能创造新的独立信息。
3. **训练 seed 敏感。** 一个 seed 的 provisional 正结果无法复现在另外两个 seed。
4. **synthesis overhead。** C2 中 Moment synthesis 约 46.3 ms，baseline 约 38.9 ms；C4 full 条件约 56.0 ms 对 48.7 ms，同时 execution completion 还更差。

### 尚无法确定因素

1. moments/drift 是否能预测 baseline failure、OOD、schedule reuse risk 或所需 search budget；当前实验没有训练或验证 failure predictor，因此不能据此选择路线 C。
2. 结论能否泛化到 `Rear4GPU` 以外的 topology、不同规模节点或真实生产 trace。
3. 每一类 node/candidate/global moment feature 对干扰的单独因果贡献；现有 C2 识别了整体 context 干扰，但没有逐 feature intervention。
4. 更大训练预算或超参数搜索能否偶然改善结果。由于多个停止条件已经触发，不应先投入该实验来延续原主线。
5. simulator completion 与真实通信硬件时延之间的外部有效性；当前 synthesis time 是真实运行计时，execution completion 是项目调度模型指标。

## 12. 最终测试结果

- 本地 Phase D 最终全量测试：`73 passed, 4 skipped in 14.39s`。
- 4 项跳过均因本地 Python 环境没有 Torch：moment policy shape、optimizer checkpoint、sequence evaluator 和 partial-demand decoder 等价性测试。
- 同一代码的 GPU/Torch 环境最终全量测试：`84 passed in 14.16s`，无跳过、无失败；环境为 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 4090。
- Phase D 只新增/修改 Markdown 和 C1 报告文字生成逻辑，没有改变策略或实验语义。

## 13. 可复现性与限制

- 项目目录不是有效 Git worktree，`.git` 是空目录，因此所有阶段都无法保存真实 git commit；报告明确记录为 unavailable，不能伪造 commit 或声称工作区干净。
- C2/C4 正式运行环境为 RTX 4090、PyTorch 2.8.0+cu128、CUDA 12.8、Python 3.12.3；生成器审计和 C3 ridge predictor 为 NumPy CPU 工作。
- 原 checkpoint 不在工作区。C1 先分析旧正式 detail 后，才使用原正式配置等价重建 3 个 checkpoint；模型结构和正式超参数没有改变。
- 完整可复制命令见仓库根目录 `README.md`。Phase D 不重新训练、不启动 V2/V3。
