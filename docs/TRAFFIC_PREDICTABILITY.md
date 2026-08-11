# 历史 moments 对当前流量的可预测性

## 设计

训练 45 条完整 sequence，测试 15 条完全不重叠的完整 sequence；测试样本 15240 个。
预测 X_t 时所有非 oracle 方法只使用 X_0...X_{t-1}。moment-only 使用滑窗矩阵均值/方差；recent-history 是对有序最近 summary 序列的多输出 ridge autoregressor；oracle 只作为上界。
bandwidth-group load 是 topology 上确定性最短路的 offered-load proxy，不冒充 learned schedule 的真实 group utilization。

## Overall 当前总流量

| method | MAE | RMSE | R² | Spearman | vs constant RMSE | hotspot accuracy | sequence-bootstrap ΔRMSE CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 2.9989 | 4.8097 | -0.0000 | 0.0000 | 0.00% | 42.53% | oracle/constant |
| previous | 0.8785 | 1.6747 | 0.8788 | 0.9027 | 65.18% | 78.80% | [1.8280, 3.6689] |
| moment_only | 1.8085 | 3.1729 | 0.5648 | 0.7060 | 34.03% | 63.59% | [0.7638, 2.0883] |
| recent_history | 0.9088 | 1.6463 | 0.8828 | 0.9017 | 65.77% | 70.53% | [1.8617, 3.7009] |
| oracle_current_summary | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | oracle/constant |

结论：moment-only 虽优于 constant，但其总流量 RMSE `3.1729` 明显高于 previous-value 的 `1.6747`；recent-history 为 `1.6463`。因此 moments 不满足“优于简单 previous-value”判据，而有序近期序列明显优于 moment 压缩。

## 其他当前流量 summary

| target | previous RMSE | moment RMSE | recent RMSE | moment vs constant | recent vs constant |
|---|---:|---:|---:|---:|---:|
| current_source_load_vector | 0.9016 | 1.6401 | 0.8802 | 40.21% | 67.91% |
| current_destination_load_vector | 0.9699 | 1.8331 | 0.9498 | 38.54% | 68.16% |
| current_hotspot_strength | 0.1173 | 0.4868 | 0.1127 | 4.07% | 77.79% |
| current_sparsity | 0.0400 | 0.1734 | 0.0388 | 8.98% | 79.62% |
| current_bandwidth_group_load | 0.9364 | 1.7393 | 0.9157 | 39.30% | 68.04% |

## 按 family 的关键判断

| family | constant RMSE | previous improvement | moment improvement | recent improvement | moment hotspot acc | recent hotspot acc |
|---|---:|---:|---:|---:|---:|---:|
| hotspot_random_walk | 1.0147 | 28.31% | -1.87% | 30.30% | 81.36% | 97.51% |
| rare_shock_recovery | 7.0319 | 70.95% | 34.75% | 71.11% | 59.32% | 62.07% |
| regime_switching_long | 5.3112 | 77.08% | 57.83% | 77.24% | 57.19% | 63.42% |
| same_moments_different_dynamics | 5.2182 | 62.25% | 27.27% | 62.89% | 49.84% | 50.52% |
| stochastic_volatility | 3.1227 | 36.30% | 5.05% | 38.29% | 70.24% | 79.13% |

## 统计单位

独立测试 sequence 15 条；raw step 样本 15240；各 sequence 总流量 ESS 合计 973.98；平均 lag-1 ACF 0.8790。
bootstrap 以完整 sequence 为 cluster，不把时间步当成独立样本。

## 输出与复现

- summary：`outputs/v1_diagnosis/predictability/predictability_summary.json`

```bash
python scripts/train_traffic_predictor.py \
  --sequence-length 1024 \
  --families regime_switching_long stochastic_volatility rare_shock_recovery hotspot_random_walk same_moments_different_dynamics \
  --seeds 42 142 242 \
  --output-dir outputs/v1_diagnosis/predictor
python scripts/evaluate_traffic_predictability.py \
  --model-dir outputs/v1_diagnosis/predictor \
  --output-dir outputs/v1_diagnosis/predictability
```
