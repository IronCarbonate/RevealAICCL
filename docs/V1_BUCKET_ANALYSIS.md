# V1 逐桶诊断

## 执行范围

本报告分析按原正式配置重建 checkpoint 后生成的训练/held-out 合并 paired detail；C1 的第一步已先直接分析旧正式 detail，确认缺少训练 family 覆盖后才进行重建复评。分桶分析本身不更新模型参数，V1 模型结构未修改。
原始记录 24576 条，独立 traffic sequence 32 条，训练 seed 3 个。
bootstrap 以完整 `sequence_id` 为 cluster，不把重叠时间步当作独立样本。正的 paired delta 表示该方法比 baseline 少用 completion slot。

## 结论

- 稳定受益桶：1 个（判据：mean delta > 0、sequence-cluster bootstrap 95% CI 下界 > 0、至少三分之二训练 seed 为正、至少 3 条独立 sequence）。
- 全部输入 schedule 合法：是；timeout 记录：0。
- 当前合并 detail 覆盖训练 family `alternating_burst`, `moving_hotspot`, `smooth_ar`, `sparse_switching`；覆盖 held-out family `bimodal`, `heavy_tail_clipped`。
- family 级稳定受益：`sparse_switching`；仅出现在训练 family。稳定判据要求至少 2/3 seed 为正，不等同于三个 seed 方向全部一致。
- 最高时间相关（lag-1 ACF 的 q4）桶：mean delta=-0.1217，95% CI=[-0.2884, 0.0104]，稳定受益=否；没有证据表明高相关场景比低相关场景更适合 moments。
- 高时间相关性是否更适合 moments 由下表直接给出；不能仅用 family 标签推断。

## Moment-full family 结果

| family | raw n | sequences | mean delta | bootstrap 95% CI | positive seeds | stable | synthesis delta ms |
|---|---:|---:|---:|---:|---:|---|---:|
| alternating_burst | 576 | 3 | 0.0000 | [0.0000, 0.0000] | 0/3 | no | -3.220 |
| bimodal | 1920 | 10 | -0.1828 | [-0.3156, -0.0505] | 0/3 | no | -5.227 |
| heavy_tail_clipped | 1920 | 10 | -0.1797 | [-0.2052, -0.1552] | 1/3 | no | -5.030 |
| moving_hotspot | 576 | 3 | -0.2587 | [-0.2812, -0.2448] | 0/3 | no | -6.218 |
| smooth_ar | 576 | 3 | 0.0087 | [-0.1042, 0.1094] | 1/3 | no | -2.347 |
| sparse_switching | 576 | 3 | 0.0399 | [0.0260, 0.0469] | 2/3 | yes | -5.967 |

## 按 sequence lag-1 ACF 分桶

| ACF bucket | sequences | mean delta | bootstrap 95% CI | stable |
|---|---:|---:|---:|---|
| q1:[-1,-0.191485] | 7 | -0.1384 | [-0.2217, -0.0402] | no |
| q2:[-0.191485,0.0958391] | 9 | -0.1782 | [-0.2060, -0.1522] | no |
| q3:[0.0958391,0.571387] | 8 | -0.0885 | [-0.2259, 0.0293] | no |
| q4:[0.571387,0.862449] | 8 | -0.1217 | [-0.2884, 0.0104] | no |

## 输出与复现

- 汇总：`outputs/v1_diagnosis/bucket_summary.csv`
- 带重建流量特征的 paired 明细：`outputs/v1_diagnosis/bucket_enriched_detail.csv`
- 分桶边界与环境元数据：`outputs/v1_diagnosis/bucket_metadata.json`

```bash
python scripts/analyze_v1_by_bucket.py \
  --detail outputs/v1_diagnosis/training_family_eval/combined_bucket_detail.csv \
  --formal-summary outputs/moment_v1/formal/v1_formal_summary.json \
  --output-dir outputs/v1_diagnosis
```

限制：当前流量特征通过 formal summary 中记录的配置与 `sequence_id` seed 确定性重建；分析脚本会校验重建后的 sequence ID。`regime_duration` 是相对配置参考总量的 low/normal/high 在线驻留长度，不使用未来状态。
