# Partial-demand 信息价值实验

## 语义

策略 observation 与 ground-truth execution 明确分离：解码特征使用 partial observation，物理 state transition、真实 demand 清除、completion、timeout 和 legality 始终使用完整真实 X_t。partial observation 不能修改或新增真实 demand；imputed destination 只影响策略特征，任何实际传输仍必须通过原确定性 topology/capacity/shared-group 可行性约束。
历史 context 在调度 X_t 前只由 X_0...X_{t-1} 更新；partial moment 的当前 z/global 特征使用 partial/proxy current matrix，而不是完整 X_t。
random-entry 设置为保持既有 chunk action space 会暴露 chunk 数和 source ownership，但隐藏 destination entry；这是当前 V1 chunk 表示的明确局限。

## Paired 结果

正的 delta 表示 Moment 比相同 observation 的 baseline 少用 completion slot。

| observation | hidden | sequences | Moment mean | paired delta | sequence bootstrap 95% CI | positive seeds | stable | legality | timeout | synthesis ms |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| full | - | 15 | 11.8257 | -0.5326 | [-0.8222, -0.2681] | 1/3 | no | 100.00% | 3.54% | 55.972 |
| partial_shards | 25% | 15 | 17.0528 | -0.1938 | [-0.4139, 0.0195] | 1/3 | no | 100.00% | 45.28% | 77.351 |
| partial_shards | 50% | 15 | 17.6722 | -0.2042 | [-0.3847, -0.0347] | 1/3 | no | 100.00% | 54.24% | 79.188 |
| random_entries | 25% | 15 | 16.6799 | -0.3986 | [-0.7578, -0.1104] | 1/3 | no | 100.00% | 40.14% | 75.789 |
| random_entries | 50% | 15 | 17.4153 | -0.1840 | [-0.3875, -0.0125] | 1/3 | no | 100.00% | 49.79% | 78.373 |
| source_destination_totals | - | 15 | 17.0562 | -0.0514 | [-0.2278, 0.1000] | 1/3 | no | 100.00% | 44.03% | 78.218 |
| source_totals | - | 15 | 16.1243 | -0.1507 | [-0.2437, -0.0618] | 1/3 | no | 100.00% | 47.15% | 73.636 |

Full moment 稳定受益：否。
Partial moment 稳定受益条件：`[]`。
所有运行的整体 legality：100.00%；timeout：38.97%。

判断规则：只有 Full moment 不优于 Full baseline、且至少一个 partial 条件跨 seed 且 bootstrap CI 稳定为正时，才支持把 moments 转向 partial-observation action conditioning。

## 输出与复现

- summary：`outputs/v1_diagnosis/partial_demand_summary.json`
- paired detail：`outputs/v1_diagnosis/partial_demand_detail.csv`

```bash
python scripts/evaluate_partial_demand.py \
  --checkpoint-dir checkpoints/v1_diagnosis/rebuilt \
  --hide-ratios 0.25 0.50 \
  --observation-modes random_entries source_totals source_destination_totals partial_shards \
  --seeds 42 142 242 \
  --output-dir outputs/v1_diagnosis
```
