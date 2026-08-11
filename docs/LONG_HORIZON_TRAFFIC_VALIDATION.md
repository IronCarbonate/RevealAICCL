# Phase B：长期不确定流量生成器验证报告

验证时间：2026-07-27（Asia/Shanghai）  
阶段边界：本轮只完成 Phase B；没有进入 Phase C/D，没有修改或训练 V1 policy。

## 1. 执行门与结论

Phase A 已证明旧生成器将 2/16 步 period 精确复制到 1024/4096，缺少随机 regime、rare shock/recovery、随机热点驻留和多时间尺度结构，因此满足 Phase B 执行门。

Phase B 新生成器已通过实现和审计验收：

- 新入口与旧六个 family 完全独立，旧 `rlccl/traffic/process_generator.py` 未覆盖或删除；
- 正式审计 600/600 条生成成功，0 条检测到精确周期；
- medium/long 对总量和完整矩阵分别执行软约束，600 条都在允许 violation fraction 内；
- rare shock 没有因短窗偏离被 rejection 过滤；
- family 在 ACF、regime/hotspot dwell、tail 和 spatial pattern 上有明显区别；
- `same_moments_different_dynamics` 的四种变体具有几乎相同的长期总量 mean/variance，但时间相关性和尾部不同。

## 2. 真实实现与数据流

主要文件：

- `rlccl/traffic/long_horizon_generator.py`：`LongHorizonTrafficConfig`、`generate_long_horizon_sequence`、`generate_same_moment_group`；
- `scripts/audit_traffic_generator.py`：`--generator auto|legacy|long`，同时支持旧、新 family；
- `rlccl/evaluation/traffic_audit.py`：识别 long-horizon metadata、latent dwell、shock、pre/post clip 和两套多尺度约束；
- `tests/test_long_horizon_traffic.py`：17 项 Phase B 测试。

生成过程显式分离：

```text
S_t = base + short AR component + regime/volatility component + shock/recovery component
X_float_t = S_t * P_t
X_t = capacity-aware integer allocation(X_float_t)
```

其中：

- `P_t >= 0`，每一步所有非对角元素之和为 1，对角线始终为 0；
- 最终分配器按最大 deficit 逐单位分配，并保证每个 entry 不超过 `max_entry`；
- 正式 600 条样本的 `spatial_distribution_validation` 均满足非负、归一化误差小于浮点精度、对角线非零数为 0；
- pre-clip、post-clip 和 post-integer 统计均保留在 metadata；
- latent regime、shock flags、hotspot source/destination/strength、总量轨迹明确标为 `audit/evaluation only`，没有接入 `TrafficSequenceRunner` 的 policy context。

有限 calibration 最多检查配置数量的候选并选择 soft-constraint penalty 最小者；它不对 short window 应用 hard bound，也不会因单个 rare event 直接拒绝整条序列。

## 3. 可配置过程

`LongHorizonTrafficConfig` 可以控制：

- base/mean/std level、AR coefficient、short noise；
- low/normal/high regime levels 和 32–512 步随机 dwell；
- shock probability、magnitude、4–16 步 duration、recovery rate；
- minimum/maximum total traffic 和每 entry 最大值；
- short/medium/long window 和各级 tolerance/allowed violation fraction；
- hotspot dwell、strength、source group 和 spatial mode；
- same-moment dynamics variant 和有限 calibration candidate 数。

空间过程实际支持：balanced、single-source hotspot、single-destination hotspot、dual hotspot、sparse support、cross-group concentration 和 hotspot random walk。没有使用“每个 entry 独立加噪声”作为唯一生成逻辑。

## 4. 五个新 family

| family | 总量过程 | 空间过程 | 非周期证据 |
|---|---|---|---|
| `regime_switching_long` | low/normal/high，完整 dwell 随机采样 32–512 步，叠加 AR noise | regime 决定 balanced/source/dual hotspot，并有随机热点 | 120/120 条无精确周期；4096 步平均约 3040 个唯一矩阵 |
| `stochastic_volatility` | calm/volatile 随机切换，innovation scale 分层，长期均值回到 base | calm 为 balanced，volatile 为 sparse support | 120/120 条无精确周期；sparsity 约 0.30，区别于其他 family |
| `rare_shock_recovery` | 低概率大 shock，持续 4–16 步，随后指数恢复；保证审计长度内至少出现一次 | shock/recovery 期间 dual hotspot，平时 balanced | 1024 平均 28.1 个 shock step，4096 平均 103.5；max/mean 约 2.55/2.72 |
| `hotspot_random_walk` | 总量保持相对稳定并有短期 AR noise | source/destination 随机游走，随机 dwell 8–96，强度随机 | 120/120 条无精确周期；4096 平均约 2849 个唯一矩阵，max destination share 约 0.535 |
| `same_moments_different_dynamics` | 四种变体统一标准化到同一配置 mean/std | balanced 慢变空间分布，用于隔离时间动力学 | smooth、random switching、long regime、shock recovery 的 ACF 和 tail 显著不同 |

序列末尾被截断的最后一个 dwell 可能短于配置下限；所有完整 regime dwell 都在 32–512 内，完整 hotspot dwell 都在 8–96 内。正式审计观察到数百种不同 dwell 长度，不是固定轮换。

## 5. 多时间尺度约束

本实现分别验证总量 `S_t` 和完整矩阵逐元素 moments：

| 层级 | 总量约束 | 矩阵约束 | allowed violation |
|---|---|---|---:|
| short 16 | 只记录 rolling mean/variance error | 只记录逐元素 rolling error | 不施加 hard/soft bound |
| medium 128 | relative mean epsilon 0.60；variance epsilon 6.00 | relative-L2 mean epsilon 1.10；variance epsilon 3.50 | 20% |
| long 512 | relative mean epsilon 0.30；variance epsilon 3.00 | relative-L2 mean epsilon 0.70；variance epsilon 2.50 | 5% |

这些 variance tolerance 特意允许 rare shock，数值比旧生成器宽；报告不把它描述为旧的严格 moment bound。总量参考来自配置而非整条未来序列的经验统计：常规 family 的 reference mean/variance 为 24/27，rare-shock 的方差参考由 shock 参数确定为 51.84。矩阵参考为配置的 off-diagonal mean/std level。

600 条正式结果：

- total medium violation：平均 0.014%，最大 3.049%，允许 20%；
- total long violation：平均 0.010%，最大 2.455%，允许 5%；
- matrix medium violation：平均 0.037%，最大 8.250%，允许 20%；
- matrix long violation：平均 0.00019%，最大 0.112%，允许 5%；
- 没有任何 sequence 超过 medium 或 long 允许比例。

short 层确实允许明显不稳定：`rare_shock_recovery` 的 16 步 rolling mean 最大相对误差 1.76、variance 最大相对误差 8.89；same-moments `shock_recovery` 的 variance 最大相对误差 14.19。这些窗口被记录，但没有触发整条序列 rejection。

## 6. family 动力学审计

下表按 family、variant 和 length 聚合全部三个 base seed。`ACF` 为总量 lag 1/8/32。

| family | variant | L | n | total CV | max/mean | >mean+2σ | ACF 1/8/32 | unique matrices | duplicate ratio | shock steps | max dst share | total medium/long violation |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| hotspot_random_walk | - | 1024 | 60 | 0.0412 | 1.1250 | 0.0355 | 0.724/0.143/-0.005 | 757.1 | 0.2606 | 0.0 | 0.5294 | 0/0 |
| hotspot_random_walk | - | 4096 | 60 | 0.0419 | 1.1403 | 0.0244 | 0.735/0.160/0.001 | 2849.0 | 0.3044 | 0.0 | 0.5351 | 0/0 |
| rare_shock_recovery | - | 1024 | 60 | 0.3022 | 2.5529 | 0.0543 | 0.957/0.669/0.073 | 520.4 | 0.4918 | 28.1 | 0.3155 | 0/0 |
| rare_shock_recovery | - | 4096 | 60 | 0.3054 | 2.7249 | 0.0532 | 0.959/0.679/0.084 | 1542.6 | 0.6234 | 103.5 | 0.3136 | 0/0 |
| regime_switching_long | - | 1024 | 60 | 0.1924 | 1.4069 | 0.0139 | 0.967/0.851/0.668 | 775.4 | 0.2428 | 0.0 | 0.3590 | 0/0 |
| regime_switching_long | - | 4096 | 60 | 0.2062 | 1.4535 | 0.0060 | 0.974/0.881/0.736 | 3040.4 | 0.2577 | 0.0 | 0.3612 | 0/0.0005 |
| stochastic_volatility | - | 1024 | 60 | 0.1247 | 1.4973 | 0.0374 | 0.783/0.149/-0.004 | 532.0 | 0.4804 | 0.0 | 0.3684 | 0/0 |
| stochastic_volatility | - | 4096 | 60 | 0.1290 | 1.6153 | 0.0381 | 0.795/0.172/0.000 | 2064.3 | 0.4960 | 0.0 | 0.3736 | 0/0 |
| same_moments | smooth | 1024/4096 | 30 | 0.2169 | 1.55/1.68 | 0.017/0.022 | 0.967/0.788/0.371（1024） | 619/1973 | 0.396/0.518 | 0 | 0.281 | 0/0 |
| same_moments | random_switching | 1024/4096 | 30 | 0.2168 | 1.35/1.37 | 0 | 0.863/0.173/-0.076（1024） | 779/2556 | 0.239/0.376 | 0 | 0.288 | 0/0 |
| same_moments | long_regime | 1024/4096 | 30 | 0.2169 | 1.36/1.40 | <0.001 | 0.981/0.891/0.645（1024） | 550/1854 | 0.463/0.547 | 0 | 0.282 | 0/0 |
| same_moments | shock_recovery | 1024/4096 | 30 | 0.2168 | 2.30/2.70 | 0.044/0.043 | 0.948/0.643/0.096（1024） | 326/877 | 0.682/0.786 | 17/62 | 0.271 | ≤0.0057/0.0020 |

family 差异不是标签差异：regime 在 lag-32 仍保持约 0.67–0.74 的相关性；random switching lag-32 接近或低于 0；rare shock 和 shock-recovery 有显著更大的 max/mean；hotspot random walk 的 destination concentration 最高；stochastic volatility 的稀疏度和 entropy 与其他 family 显著不同。

## 7. same moments、不同 dynamics

独立 `generate_same_moment_group` 对同一 seed 成组生成 smooth、random switching、long regime、shock recovery。正式审计的长期总量 moments：

| L | smooth mean/var | random switching | long regime | shock recovery |
|---:|---:|---:|---:|---:|
| 1024 | 23.9987 / 27.1170 | 24.0036 / 27.0593 | 24.0016 / 27.0890 | 24.0018 / 27.0763 |
| 4096 | 24.0016 / 27.0943 | 23.9999 / 27.0819 | 24.0000 / 27.1043 | 23.9986 / 27.0766 |

对应的 lag-8 ACF 在 1024 步分别约为 0.788、0.173、0.891、0.643；shock-recovery 的 max/mean 又明显更高。因此“moments 接近、动力学不同”成立。

## 8. 按 length 和 base seed 分组

每个常规 family 每组 20 条；same-moments 的 20 条在四个 variant 间轮换，因此每 variant 每组 5 条。数字顺序均为 base seed 42/142/242。

| family | variant | L | n/seed | unique mean 42/142/242 | total long max 42/142/242 | matrix long max 42/142/242 |
|---|---|---:|---:|---|---|---|
| hotspot_random_walk | - | 1024 | 20 | 765.0/752.9/753.5 | 0/0/0 | 0/0/0 |
| hotspot_random_walk | - | 4096 | 20 | 2866.6/2836.6/2843.9 | 0/0/0 | 0/0/0 |
| rare_shock_recovery | - | 1024 | 20 | 523.9/494.1/543.3 | 0/0/0 | 0/0/0 |
| rare_shock_recovery | - | 4096 | 20 | 1525.7/1576.9/1525.2 | 0/0/0 | 0/0/0 |
| regime_switching_long | - | 1024 | 20 | 783.5/776.1/766.5 | 0/0/0 | 0/0/0 |
| regime_switching_long | - | 4096 | 20 | 3032.1/3018.7/3070.6 | 0.0245/0/0.0064 | 0/0.0011/0 |
| same_moments | long_regime | 1024 | 5 | 535.8/571.2/544.2 | 0/0/0 | 0/0/0 |
| same_moments | long_regime | 4096 | 5 | 1846.4/1885.2/1829.8 | 0/0/0 | 0/0/0 |
| same_moments | random_switching | 1024 | 5 | 773.0/788.0/776.2 | 0/0/0 | 0/0/0 |
| same_moments | random_switching | 4096 | 5 | 2546.2/2569.0/2552.0 | 0/0/0 | 0/0/0 |
| same_moments | shock_recovery | 1024 | 5 | 333.8/329.6/314.6 | 0/0/0 | 0/0/0 |
| same_moments | shock_recovery | 4096 | 5 | 888.2/857.0/885.4 | 0/0.0243/0.0050 | 0/0/0 |
| same_moments | smooth | 1024 | 5 | 618.6/614.6/622.8 | 0/0/0 | 0/0/0 |
| same_moments | smooth | 4096 | 5 | 1942.6/1988.2/1989.2 | 0/0/0 | 0/0/0 |
| stochastic_volatility | - | 1024 | 20 | 529.0/539.6/527.6 | 0/0/0 | 0/0/0 |
| stochastic_volatility | - | 4096 | 20 | 2045.2/2089.5/2058.3 | 0/0/0 | 0/0/0 |

完整逐 sequence 指标、三个 base seed 和实际 seed 均保存在 `outputs/traffic_audit/long_horizon/audit_detail.csv` 与各 family JSON 中。

## 9. 测试和环境

- Phase B 专项：`17 passed`；
- Phase B + audit 专项：`23 passed`；
- 本机最终复测：`64 passed, 3 skipped in 13.18s`，三项因本机缺少 torch；
- 此前远程 Python 3.12.3 / PyTorch 2.8.0+cu128 / RTX 4090 全套：`74 passed in 12.38s`；其后只增加了审计结果中的 `spatial_distribution_validation` 字段透传，并已由上述本机全套复测覆盖；
- 审计/生成本身只需要 CPU，远程 GPU 仅用于补齐 torch 测试；
- 本地传输包已删除；最后一次被中断的冗余远程复核所用临时目录尚未重新连接核验，按当前要求暂不访问远程服务器。

## 10. 阶段边界与限制

- 新生成器尚未接入 V1 训练或评估数据管线；这是 Phase C 及之后才能决定的工作。
- 配置的 soft epsilon 为保留 shock/hotspot 而明显宽于旧 short-horizon hard bound，应在后续实验中作为显式超参数报告，不能与旧 epsilon 直接等同。
- default `source_groups` 使用节点前后半组的抽象分组；真实 topology bandwidth group 映射需要后续显式传入，不能由 demand matrix 唯一推断。
- metadata 含未来 latent trajectory，只能用于审计和评测；当前没有任何 policy 路径读取它。

Phase B 到此完成并停止；未开始 Phase C。
