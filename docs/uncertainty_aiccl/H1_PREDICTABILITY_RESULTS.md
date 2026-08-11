# Gate H1 历史可预测性正式结果

## 1. 结论

Main 对正式 raw evidence 的预注册判定为 **H1 FAIL**。正式 summary 仍正确保留
`gate_status="PENDING_SUPERVISOR"`；最终 Gate 由 Supervisor 独立审查后确定。

validation 按 15 条完整 sequence 等权选择 `recent_history_mlp`：MLP total-RMSE
`1.4498`，TCN `1.5442`。在 15 条独立 test sequence 上，selected MLP 的 total-RMSE
为 `1.6468`，差于 previous-value 的 `1.5678`。paired delta 定义为
`RMSE(previous)-RMSE(selected)`，因此 selected 的 mean delta 为 `-0.0790`，95% CI
`[-0.1133, -0.0478]`，不仅没有稳定增益，反而稳定更差。

## 2. 七项预注册条件

| 条件 | 结果 | 正式证据 |
|---|---|---|
| 1. pooled total paired CI 下界 > 0 | FAIL | mean `-0.0790`；CI `[-0.1133, -0.0478]` |
| 2. 3/3 base seeds 正、至少 4/5 families 正 | FAIL | seeds：`-0.0568/-0.0543/-0.1259`；仅 stochastic volatility 为正（1/5） |
| 3. total/source/destination 三个 primary CI 下界 > 0 | FAIL | total `[-0.1133,-0.0478]`；source `[-0.0134,-0.0008]`；destination `[-0.0102,0.0021]` |
| 4. LOFO 不系统失败 | FAIL | aggregate delta `-0.4280`；0/5 family 为正；3/5 family RMSE 恶化超过10% |
| 5. quantile/scenario calibration | FAIL | overall interval error `0.0184`、scenario error `0.0393` 合格；但 hotspot-random-walk `0.1140`、stochastic-volatility `0.1228` 超过 family 0.10 阈值 |
| 6. tail recall | PASS | 1444 个 pooled tail events，recall `0.9314` |
| 7. integrity tests + Supervisor no veto | PENDING | Main 完整测试通过；等待 Supervisor 最终独立审查 |

条件 1--5 失败，故即使条件 6/7 合格也不能通过 H1。

## 3. 关键点预测对比

以下数值均为 15 条 test sequence 的等权 sequence metric mean。

| target | previous RMSE | selected MLP RMSE | quantile median RMSE | selected paired mean delta |
|---|---:|---:|---:|---:|
| total traffic | 1.5678 | 1.6468 | 1.6451 | -0.0790 |
| source-load vector | 0.8852 | 0.8921 | 0.8919 | -0.0070 |
| destination-load vector | 0.9550 | 0.9587 | 0.9583 | -0.0037 |
| hotspot strength | 0.1153 | 0.1118 | 0.1119 | +0.0035 |
| sparsity | 0.0317 | 0.0341 | 0.0340 | -0.0024 |
| bandwidth-group offered load | 0.9227 | 0.9288 | 0.9285 | -0.0061 |

hotspot destination accuracy：previous-value `79.15%`，selected MLP `73.44%`，
quantile median `70.33%`。单个次要 target 的小幅改善不能覆盖三个预注册 primary
targets 和 LOFO 的失败。

## 4. 跨 family 与 LOFO

pooled selected total delta（正值才优于 previous）：

| family | mean delta |
|---|---:|
| regime_switching_long | -0.1131 |
| stochastic_volatility | +0.0499 |
| rare_shock_recovery | -0.1650 |
| hotspot_random_walk | -0.1058 |
| same_moments_different_dynamics | -0.0610 |

LOFO 每折均从 fit/validation/calibration 完全排除 held-out family，并重新训练、选模和
校准；五折都选择 MLP。五个 held-out family 的 mean delta 全为负：`-0.1116`、
`-0.9273`、`-0.8350`、`-0.2084`、`-0.0577`。对应 RMSE 相对恶化约
`9.90%/44.22%/39.50%/28.37%/3.27%`。

## 5. 统计单位与执行事实

- fit/validation/calibration/test：30/15/15/15 条完整 sequence；
- test 独立 sequence：15；raw test steps：15,240；
- mean lag-1 ACF：`0.8714`；逐 sequence ESS 合计：`1062.09`；
- paired CI：family-stratified、完整 sequence cluster bootstrap 10,000 次；
- 正式命令 exit 0，耗时 `121.9s`；出现 3 个 sklearn `ConvergenceWarning`：固定
  80 iterations 未满足收敛阈值，未调参、未隐藏、未把 warning 当成收敛证据；
- Main raw validators 全通过，raw 重算的 summary 与落盘 JSON 精确相等。

## 6. 输出与 SHA-256

| artifact | rows/records | SHA-256 |
|---|---:|---|
| `outputs/h1_predictability/manifest.json` | 75 records | `C702D8CEA33BCEC805FA0AB4B1EEA58C7E0BCBF6AAEF697E01523BB86D65B48C` |
| `outputs/h1_predictability/raw_sequence_metrics.csv` | 1590 rows | `D03DAC115E2DE839FBEF32326AD90E9E662053E678C5CCEFB1605736A5402517` |
| `outputs/h1_predictability/raw_probability_metrics.csv` | 180 rows | `7C0F0C2CB8056BAF32466AB4D519D816E5DBF09FAC85A5C96908329901752829` |
| `outputs/h1_predictability/summary.json` | schema 2 | `C48E35230030215148E3DEF46340A991D226B69FD97797EB6D2086BE6A26DFCE` |

完整回归：`203 passed, 4 skipped, 18 warnings in 22.78s`。4 个 skipped 测试均因
当前环境缺 Torch，不能记为通过。

## 7. 路线建议与停止边界

H1 没有证明近期历史模型相对 previous-value 的稳定增量信息，且 LOFO 系统退化。
依据预注册规则，建议下一阶段选择 **Phase 3B：prediction-free robust route**，而不是
Phase 3A predictive scenario route。该建议不构成执行授权；H1 最终审查完成后停止，
等待用户决定。
