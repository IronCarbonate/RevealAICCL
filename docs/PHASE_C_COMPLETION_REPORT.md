# Phase C 完成报告

## 范围与复现口径

本轮严格停留在 `CODEX_AICCL_TRAFFIC_DIAGNOSIS_COMMANDS.txt` 的 Phase C（C0–C4）。没有修改 V1 策略网络结构，也没有进入 Phase D、V2 或 V3。

C1 首先直接分析既有正式 paired detail。后续实验发现工作区没有可用 V1 checkpoint，因此才按原正式配置等价重建 3 个训练种子（42、142、242），用于 C2、C4 和训练 family 的补充评估。重建没有改变模型、数据族、训练轮数或评价口径。

GPU 环境为 NVIDIA GeForce RTX 4090、PyTorch 2.8.0+cu128、CUDA 12.8、Python 3.12.3。本地 Python 3.12 环境不含 Torch，仅用于无 Torch 分析与测试。

## C1：逐 family / 条件分桶

- 合并数据：24,576 条 paired 记录、2,048 个 traffic 时刻、32 条独立 sequence、3 个训练 seed。
- 覆盖训练 family：`alternating_burst`、`moving_hotspot`、`smooth_ar`、`sparse_switching`；覆盖 held-out family：`bimodal`、`heavy_tail_clipped`。
- 唯一满足预设稳定判据的 family 桶是训练族 `sparse_switching`：Moment-full 相对 baseline 的 completion 改善为 `+0.0399`，sequence-cluster bootstrap 95% CI `[0.0260, 0.0469]`，但只有 `2/3` 个训练 seed 为正，并非三个 seed 同向。
- held-out `bimodal` 为 `-0.1828`，95% CI `[-0.3156, -0.0505]`；`heavy_tail_clipped` 为 `-0.1797`，95% CI `[-0.2052, -0.1552]`。
- 最高 lag-1 ACF 桶为 `-0.1217`，95% CI `[-0.2884, 0.0104]`，不支持“高时间相关场景更适合 moments”。
- 全部调度合法，timeout 为 0；completion 与 synthesis time 分开统计。

详见 `docs/V1_BUCKET_ANALYSIS.md`、`outputs/v1_diagnosis/bucket_summary.csv` 和 `outputs/v1_diagnosis/bucket_enriched_detail.csv`。

## C2：相同当前 X、不同历史

- 构造 200 个严格反事实 pair，并在 3 个训练 seed 上评估，共 600 条 paired 记录。
- 当前 traffic matrix、topology、初始 schedule state 和 ground-truth demand hash 均一致；History A/B 只含当前时刻以前的矩阵。
- baseline 在两段历史下的等价率为 `100%`。
- Moment-full 因历史变化而改变完整 schedule 的比例为 `100%`；首 slot logits L2 差异均值 `2.1731`，动作序列编辑距离均值 `33.482`，edge-use L1 差异均值 `19.095`。
- action-level context interference 为 `60.83%`，出现有益变化的比例仅 `11.50%`。
- baseline completion 减去两种 Moment completion 的平均值为 `-0.6925`，pair-cluster bootstrap 95% CI `[-0.7759, -0.6092]`，即 Moment 显著更差。
- legality `100%`、timeout `0%`；baseline/Moment synthesis time 均单独记录。

结论：历史 moments 在当前问题完全不变时会直接扰动动作序列，而且多数扰动有害，构成明确的 action-level context interference。

详见 `docs/COUNTERFACTUAL_HISTORY.md` 和 `outputs/v1_diagnosis/counterfactual_detail.csv`。

## C3：当前流量可预测性

- 使用 45 条完整训练 sequence 和 15 条完全不重叠的测试 sequence；45,720 个训练时刻、15,240 个测试时刻。
- 所有非 oracle 方法预测 `X_t` 时只使用 `X_0...X_{t-1}`。bootstrap 单位为完整 sequence；测试序列总流量的平均 lag-1 ACF 为 `0.8790`，ESS 合计 `973.98`。
- 当前总流量 RMSE：constant `4.8097`、previous `1.6747`、moment-only `3.1729`、recent-history `1.6463`、oracle `0`。
- moment-only 相对 constant 改善 `34.03%`，但明显差于 previous 和 recent-history；recent-history 相对 constant 改善 `65.77%`。
- source load、destination load、hotspot strength、sparsity 和 bandwidth-group offered-load proxy 上也呈现相同趋势。
- moment-only 在 `hotspot_random_walk` 上甚至比 constant 差 `1.87%`；其余 family 的收益也普遍弱于 previous/recent-history。

结论：历史确实具有预测信息，但均值/方差压缩丢失了关键的有序时序信息，moment-only 不满足“优于 previous-value”的判断标准。

详见 `docs/TRAFFIC_PREDICTABILITY.md` 和 `outputs/v1_diagnosis/predictability/predictability_summary.json`。

## C4：Partial-demand 信息价值

- 15 条独立长序列、480 个时刻、7 种观测条件、2 个方法、3 个模型 seed，共 20,160 条 paired 记录。
- observation 与 ground-truth execution 已分离：partial/proxy matrix 只进入策略特征；真实 demand 清除、状态转移、completion、timeout、legality 始终使用完整真值。
- Full moment 相对 Full baseline 的 completion delta 为 `-0.5326`，95% CI `[-0.8222, -0.2681]`。
- 隐藏 25%/50% entries、只给 source totals、给 source+destination totals、只揭示 25%/50% shards 的所有条件均未获得跨 seed 稳定正收益。
- 各 partial 条件的 Moment paired delta 介于 `-0.3986` 与 `-0.0514`；部分 CI 跨 0，但每个条件都只有 `1/3` seed 为正。
- 整体 legality `100%`。整体 timeout `38.97%`，其中 Full baseline 为 `1.39%`、Full moment 为 `3.54%`；partial 条件 timeout 显著升高，报告中与 synthesis time 分开列出。

结论：partial-demand 场景也没有稳定收益，不支持把现有 action-level moment conditioning 改解释为“只在信息缺失时有用”。

详见 `docs/PARTIAL_DEMAND_EXPERIMENT.md`、`outputs/v1_diagnosis/partial_demand_summary.json` 和 `outputs/v1_diagnosis/partial_demand_detail.csv`。

## C0 共同要求核对

- baseline/Moment 使用相同 traffic matrices：满足，明细为 paired 设计。
- 三个或更多 seed：满足；策略实验使用 3 个训练 seed，流量实验使用完整 sequence seed 划分。
- 按完整 sequence 划分和 bootstrap：满足。
- paired detail：C1、C2、C4 已输出；C3 保存完整 sequence split、指标和元数据。
- schedule legality：所有正式调度实验均为 `100%`。
- timeout 与 synthesis time：均单独报告。
- 不修改 V1 模型：满足；decoder 只新增可选 observation 输入路径，默认完整需求路径有等价性测试。

## 测试结果

- GPU/Torch 环境：`84 passed in 14.16s`，无跳过、无失败。
- 本地无 Torch 环境最终复测：`73 passed, 4 skipped in 21.11s`；4 项仅因本地没有安装 Torch，且已在远端通过。

Phase C 到此完成并停止。Phase D 的最终方向决策文档未生成，V2/V3 未实现。
