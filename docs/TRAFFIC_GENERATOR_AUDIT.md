# 现有流量生成器 Phase A 审计报告

审计时间：2026-07-27  
对象：`rlccl/traffic/process_generator.py` 的六个现有 family  
结论：**不应把当前实现称为“长期不确定流量生成器”**。它是一个严格短周期、moment-bounded 的模板生成器；增加 `sequence_length` 只会精确重复同一短周期。

## 1. 审计配置与产物

正式命令覆盖：

- families：`smooth_ar`、`alternating_burst`、`moving_hotspot`、`sparse_switching`、`bimodal`、`heavy_tail_clipped`；
- sequence lengths：64、1024、4096；
- base seeds：42、142、242；
- 每个 family × length × base seed 生成 20 条，实际 seed 为 `base_seed + sequence_index`；
- short/medium/long windows：16/128/512；
- `num_nodes=4`、mean=2、std=1、max-entry=8、epsilon mean/var=0.20/0.30；
- 共 1080 条 sequence；统计推断单位是 sequence。重叠 rolling windows 只作为一条 sequence 内的诊断，不能冒充独立样本或独立置信区间。

产物：

- `outputs/traffic_audit/audit_summary.json`；
- `outputs/traffic_audit/audit_detail.csv`；
- `outputs/traffic_audit/family_<name>.json`（六个）；
- 本报告。

命令使用 `--workers 8` 做独立 CPU 并行。审计和生成都是 NumPy CPU 工作，不需要 GPU。

## 2. 总体证据

- 1080/1080 生成成功，0 失败；所有成功样本的 `generation_attempt=1`，观测 rejection rate 为 0。
- `alternating_burst` 的最短精确矩阵周期在全部 180 条样本中都是 2；其余 family 在全部样本中都是 16。
- 长度 4096 的 exact duplicate ratio：`alternating_burst=0.999512`；`smooth_ar/moving_hotspot/sparse_switching=0.996094`；`heavy_tail_clipped=0.996826`；`bimodal` 约 0.99668–0.99702。
- 所有可用的 16/128/512 rolling windows 的 mean/variance bound violation fraction 都是 0。
- 以上“0 违规”不是多尺度随机过程稳定性的证据。生成器直接重复 16 步 period，16 的整数倍窗口必然得到同一 moments。
- 所有 family 的 `fraction above mean+3σ` 都是 0。除 `smooth_ar/bimodal` 的少量单步事件外，大多数 family 连 `mean+2σ` 都从不超过；最大高负载连续段仅 1 步。

## 3. family 级动力学与空间结果

下表聚合三个 base seed、每种长度共 60 条。数值在不同长度保持不变，正是周期复制的直接表现。

| family | total mean | total CV | mean max/mean | exact period | 4096 duplicate ratio | hotspot max dwell | spatial 证据 |
|---|---:|---:|---:|---:|---:|---:|---|
| alternating_burst | 24.00 | 0.1667 | 1.1667 | 2 | 0.999512 | 4096 | source/destination max share 均 0.25，空间完全平衡 |
| smooth_ar | 24.75 | 0.1265 | 1.2034 | 16 | 0.996094 | 9 | sparsity 0.0625，max src/dst share 约 0.324 |
| moving_hotspot | 23.25 | 0.1070 | 1.1613 | 16 | 0.996094 | 5 | max destination share 0.419，hotspot strength 1.677，固定循环 |
| sparse_switching | 21.75 | 0.0551 | 1.1034 | 16 | 0.996094 | 5 | 正式参数 sparsity=0；max source share 0.403，主要是固定 source 不均衡 |
| bimodal | 24.00 | 0.1279 | 1.2069 | 16 | 约 0.996875 | 9 | max src/dst share 约 0.320/0.322 |
| heavy_tail_clipped | 27.75 | 0.1077 | 1.1892 | 16 | 0.996826 | 2 | final entry max=6；max src/dst share 0.308 |

`bandwidth-group concentration` 无法从 `TrafficSequence` 唯一识别：它只有 source-destination demand，没有将 demand 映射到 topology group 的路由或 schedule。审计输出将其标为 unavailable，而不是假设直连路由。

生成器也没有输出 latent regime label，所以 regime dwell 同样标为 unavailable。

## 4. 按 family、length、seed 分组的完整摘要

每行是 20 条独立 sequence。`period 2:20` 表示 20/20 检测到最短精确周期 2；窗口栏顺序为 16/128/512，`NA` 表示窗口比序列长。

表中的“生成秒/条”记录第一次完整 1080 条正式运行；补入逐 source/destination 输出字段后的同规模最终重跑会有正常的 wall-time 波动，当前逐组实测时间以 `audit_summary.json` 为准。两次运行的成功数、attempt、周期、重复率、total CV 和所有窗口违规率一致。

| family | L | base seed | 成功/请求 | attempts均值 | 生成秒/条 | total CV | 重复率 | 精确周期 | 16/128/512违规率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| alternating_burst | 64 | 42 | 20/20 | 1.0 | 0.0579 | 0.1667 | 0.968750 | 2:20 | 0.000/NA/NA |
| alternating_burst | 64 | 142 | 20/20 | 1.0 | 0.0612 | 0.1667 | 0.968750 | 2:20 | 0.000/NA/NA |
| alternating_burst | 64 | 242 | 20/20 | 1.0 | 0.0565 | 0.1667 | 0.968750 | 2:20 | 0.000/NA/NA |
| alternating_burst | 1024 | 42 | 20/20 | 1.0 | 0.8862 | 0.1667 | 0.998047 | 2:20 | 0.000/0.000/0.000 |
| alternating_burst | 1024 | 142 | 20/20 | 1.0 | 0.8755 | 0.1667 | 0.998047 | 2:20 | 0.000/0.000/0.000 |
| alternating_burst | 1024 | 242 | 20/20 | 1.0 | 0.8615 | 0.1667 | 0.998047 | 2:20 | 0.000/0.000/0.000 |
| alternating_burst | 4096 | 42 | 20/20 | 1.0 | 3.4960 | 0.1667 | 0.999512 | 2:20 | 0.000/0.000/0.000 |
| alternating_burst | 4096 | 142 | 20/20 | 1.0 | 3.5862 | 0.1667 | 0.999512 | 2:20 | 0.000/0.000/0.000 |
| alternating_burst | 4096 | 242 | 20/20 | 1.0 | 3.6268 | 0.1667 | 0.999512 | 2:20 | 0.000/0.000/0.000 |
| bimodal | 64 | 42 | 20/20 | 1.0 | 0.0546 | 0.1248 | 0.803125 | 16:20 | 0.000/NA/NA |
| bimodal | 64 | 142 | 20/20 | 1.0 | 0.0560 | 0.1253 | 0.809375 | 16:20 | 0.000/NA/NA |
| bimodal | 64 | 242 | 20/20 | 1.0 | 0.0537 | 0.1334 | 0.787500 | 16:20 | 0.000/NA/NA |
| bimodal | 1024 | 42 | 20/20 | 1.0 | 0.8406 | 0.1248 | 0.987695 | 16:20 | 0.000/0.000/0.000 |
| bimodal | 1024 | 142 | 20/20 | 1.0 | 0.9088 | 0.1253 | 0.988086 | 16:20 | 0.000/0.000/0.000 |
| bimodal | 1024 | 242 | 20/20 | 1.0 | 0.8731 | 0.1334 | 0.986719 | 16:20 | 0.000/0.000/0.000 |
| bimodal | 4096 | 42 | 20/20 | 1.0 | 3.6779 | 0.1248 | 0.996924 | 16:20 | 0.000/0.000/0.000 |
| bimodal | 4096 | 142 | 20/20 | 1.0 | 3.6503 | 0.1253 | 0.997021 | 16:20 | 0.000/0.000/0.000 |
| bimodal | 4096 | 242 | 20/20 | 1.0 | 3.7095 | 0.1334 | 0.996680 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 64 | 42 | 20/20 | 1.0 | 0.0582 | 0.1077 | 0.796875 | 16:20 | 0.000/NA/NA |
| heavy_tail_clipped | 64 | 142 | 20/20 | 1.0 | 0.0594 | 0.1077 | 0.796875 | 16:20 | 0.000/NA/NA |
| heavy_tail_clipped | 64 | 242 | 20/20 | 1.0 | 0.0531 | 0.1077 | 0.796875 | 16:20 | 0.000/NA/NA |
| heavy_tail_clipped | 1024 | 42 | 20/20 | 1.0 | 0.8981 | 0.1077 | 0.987305 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 1024 | 142 | 20/20 | 1.0 | 0.8870 | 0.1077 | 0.987305 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 1024 | 242 | 20/20 | 1.0 | 0.9111 | 0.1077 | 0.987305 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 4096 | 42 | 20/20 | 1.0 | 3.3760 | 0.1077 | 0.996826 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 4096 | 142 | 20/20 | 1.0 | 3.3129 | 0.1077 | 0.996826 | 16:20 | 0.000/0.000/0.000 |
| heavy_tail_clipped | 4096 | 242 | 20/20 | 1.0 | 3.2385 | 0.1077 | 0.996826 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 64 | 42 | 20/20 | 1.0 | 0.0542 | 0.1070 | 0.750000 | 16:20 | 0.000/NA/NA |
| moving_hotspot | 64 | 142 | 20/20 | 1.0 | 0.0638 | 0.1070 | 0.750000 | 16:20 | 0.000/NA/NA |
| moving_hotspot | 64 | 242 | 20/20 | 1.0 | 0.0577 | 0.1070 | 0.750000 | 16:20 | 0.000/NA/NA |
| moving_hotspot | 1024 | 42 | 20/20 | 1.0 | 0.8949 | 0.1070 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 1024 | 142 | 20/20 | 1.0 | 0.8939 | 0.1070 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 1024 | 242 | 20/20 | 1.0 | 0.8745 | 0.1070 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 4096 | 42 | 20/20 | 1.0 | 3.4768 | 0.1070 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 4096 | 142 | 20/20 | 1.0 | 3.6050 | 0.1070 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| moving_hotspot | 4096 | 242 | 20/20 | 1.0 | 3.5804 | 0.1070 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 64 | 42 | 20/20 | 1.0 | 0.0526 | 0.1188 | 0.750000 | 16:20 | 0.000/NA/NA |
| smooth_ar | 64 | 142 | 20/20 | 1.0 | 0.0549 | 0.1263 | 0.750000 | 16:20 | 0.000/NA/NA |
| smooth_ar | 64 | 242 | 20/20 | 1.0 | 0.0601 | 0.1344 | 0.750000 | 16:20 | 0.000/NA/NA |
| smooth_ar | 1024 | 42 | 20/20 | 1.0 | 0.8785 | 0.1188 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 1024 | 142 | 20/20 | 1.0 | 0.8858 | 0.1263 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 1024 | 242 | 20/20 | 1.0 | 0.8572 | 0.1344 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 4096 | 42 | 20/20 | 1.0 | 3.5018 | 0.1188 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 4096 | 142 | 20/20 | 1.0 | 3.5253 | 0.1263 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| smooth_ar | 4096 | 242 | 20/20 | 1.0 | 3.6016 | 0.1344 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 64 | 42 | 20/20 | 1.0 | 0.0559 | 0.0551 | 0.750000 | 16:20 | 0.000/NA/NA |
| sparse_switching | 64 | 142 | 20/20 | 1.0 | 0.0620 | 0.0551 | 0.750000 | 16:20 | 0.000/NA/NA |
| sparse_switching | 64 | 242 | 20/20 | 1.0 | 0.0594 | 0.0551 | 0.750000 | 16:20 | 0.000/NA/NA |
| sparse_switching | 1024 | 42 | 20/20 | 1.0 | 0.8936 | 0.0551 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 1024 | 142 | 20/20 | 1.0 | 0.8866 | 0.0551 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 1024 | 242 | 20/20 | 1.0 | 0.8664 | 0.0551 | 0.984375 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 4096 | 42 | 20/20 | 1.0 | 3.5829 | 0.0551 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 4096 | 142 | 20/20 | 1.0 | 3.5488 | 0.0551 | 0.996094 | 16:20 | 0.000/0.000/0.000 |
| sparse_switching | 4096 | 242 | 20/20 | 1.0 | 3.4983 | 0.0551 | 0.996094 | 16:20 | 0.000/0.000/0.000 |

## 5. 对任务书九个问题的明确回答

1. **sequence length 增大后 rejection rate 是否上升？** 观测上没有：64/1024/4096 的 1080 条都在第一次通过，rejection rate 均为 0。原因不是长时生成更强，而是长度只复制已校准 period。失败配置仍有重复无效 rejection 的代码风险。

2. **1024/4096 是否保持 family 原本动态？** 只保持了短模板的精确复制，没有产生新的长期动态。total CV、max/mean、空间统计跨长度几乎完全不变。

3. **alternating_burst 是否接近固定周期？** 是，而且比配置 window 更短：180/180 条的最短精确矩阵周期都是 2。

4. **moving_hotspot 是否按固定轨迹移动？** 是。destination 相位由固定公式决定，180/180 条周期 16，最长单次 destination dwell 为 5；seed 只平移起点，不改变轨迹形状或驻留分布。

5. **heavy_tail_clipped 尾部是否被明显削弱？** 最终证据显示尾部很弱：entry max=6（上限 8），total max/mean=1.189，`>mean+2σ` 和 `>mean+3σ` 比例均为 0，spike 每 16 步固定出现。不能把削弱量精确分摊给 clip 或 rejection：生成器没有保留 pre-clip 数据，本次也没有发生 rejection。可确认的代码事实是先 clip、后 round，再从整数候选中挑 moment 最接近者。

6. **主要改变总量还是空间 pattern？** 不统一。`alternating_burst` 基本只改变总量；`moving_hotspot` 明显改变 destination pattern 且伴随小幅总量变化；`sparse_switching` 主要改变 source pattern；其余 family 同时改变总量与空间 pattern。当前实现没有显式分解 `S_t` 与 `P_t`，两者由 pair 相位偶然耦合。

7. **是否允许持续几十/几百步 burst/regime？** 不允许。没有 latent regime 或随机 dwell；按 `mean+2σ` 定义，实测最大连续高负载段只有 1 步，多数 family 为 0。`alternating_burst` 的 hotspot dwell=4096 是空间完全平衡导致 argmax 固定，并不是长负载 regime。

8. **是否允许短窗口明显偏离、长窗口恢复 moments？** 在当前设计中不允许。每个 16 步窗口都必须硬满足同一 bounds，allowed violation fraction 为 0；128/512 只是 period 的整数倍，也全部 0 违规。不存在“短期明显偏离、长期恢复”的训练样本。

9. **是否足以称为长期不确定流量生成器？** 否。它可作为可复现、合法、严格 moment-bounded 的短周期回归生成器，但缺少长期随机 regime、随机 volatility、rare shock/recovery、随机 hotspot walk、非固定 dwell 和多时间尺度约束。

## 6. 生成过程可观测性与限制

审计记录了 attempts、rejections、wall time 和最终整数统计。以下内容因当前生成 API 不保留而明确为 unavailable：

- pre-clip 与 post-clip/pre-round 的 mean/variance/p99/max；
- post-round/pre-diagonal 数据；
- 每个被拒绝 candidate；
- latent regime、shock flag、真实 hotspot label。

近重复率采用相邻矩阵的 normalized L1 阈值作描述；精确重复和精确周期则直接比较完整矩阵。总流量 ACF/ESS 对确定性周期序列仅是描述性指标；尤其正相关截断 ESS 不能替代精确周期检测。

## 7. 测试验收

- Phase A 专项：`6 passed`；
- 本机全套：`47 passed, 3 skipped`，三项跳过均因本机无 torch；
- 远程 PyTorch/CUDA 最终全套：`57 passed in 3.69s`；
- 最终远程复测前有两次 Windows ZIP 到 Linux 的打包前置失败：一次 pytest exit 5（未发现测试文件），一次路径存在断言 exit 1；两次均未执行测试用例。改用保留 Unix 路径的 tar 包并先断言测试文件存在后，完整测试通过；
- 默认系统 Python 的原始失败仍保留记录：无 pytest；
- 没有训练、没有 checkpoint 写入、没有 V1 模型修改。

本轮在此停止，不实施 Phase B、C 或 D。
