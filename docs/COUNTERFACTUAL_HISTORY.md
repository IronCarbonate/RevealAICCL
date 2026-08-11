# 相同当前 X、不同历史的反事实诊断

## 设计

构造 200 个相同当前 traffic matrix、相同 topology、相同初始 schedule state 的 pair，并在 3 个独立 V1 训练 seed 上评测。
History A/B 都只由当前 X 之前的独立矩阵组成；`SlidingMomentEstimator.get_context` 后才会在真实时序中更新当前矩阵，本实验没有把未来 X 放入 estimator。
baseline 对两个历史分别实际运行；Moment-full 使用显著不同的两个 history-only context。

## 结果

| method | mean completion | median | p95 | p99 | CVaR95 | mean synthesis ms |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8.4150 | 7.0000 | 14.0000 | 15.0000 | 14.1928 | 38.935 |
| moment_history_a | 9.0817 | 8.0000 | 15.0000 | 17.0000 | 15.8298 | 46.257 |
| moment_history_b | 9.1333 | 9.0000 | 15.0000 | 17.0100 | 16.1837 | 46.315 |

- baseline 两历史等价率：100.00%
- Moment-full 因历史改变完整 schedule 的比例：100.00%
- action-level context interference（变化且至少一个历史结果劣于 baseline）：60.83%
- 变化且至少一个历史优于 baseline：11.50%
- baseline completion - 两个 Moment completion 平均值：-0.6925，按 pair cluster bootstrap 95% CI [-0.7759, -0.6092]；正数才表示 Moment 更好。
- 首 slot logits L2 差异均值：2.1731；动作 edit distance 均值：33.482；edge-use L1 均值：19.095。
- schedule legality：100.00%；timeout：0.00%。

若 `action-level context interference` 非零，则它是直接证据：完整当前 demand 与初始状态相同，仅历史 moments 就能改变 action-level schedule，并且至少一个变化方向恶化 completion。

## 输出与复现

- paired 明细：`outputs/v1_diagnosis/counterfactual_detail.csv`
- 汇总与运行元数据：`outputs/v1_diagnosis/counterfactual_summary.json`

```bash
python scripts/evaluate_counterfactual_history.py \
  --checkpoint-dir checkpoints/v1_diagnosis/rebuilt \
  --num-pairs 200 \
  --seeds 42 142 242 \
  --output-dir outputs/v1_diagnosis
```
