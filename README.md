# RLCCL / AMR-AICCL 诊断工作区

本工作区已完成 `CODEX_AICCL_TRAFFIC_DIAGNOSIS_COMMANDS.txt` 的 Phase A–D。最终主路线是 **D：终止 action-level moments 主线，转向 baseline decoder 与 synthesis 性能优化**。

结论和证据入口：

- `docs/V1_FAILURE_DIAGNOSIS.md`
- `docs/NEXT_DIRECTION_DECISION.md`
- `docs/PHASE_C_COMPLETION_REPORT.md`

项目目录当前不是有效 Git worktree，无法提供真实 commit hash。以下命令应从仓库根目录运行。

## 环境

CPU 审计、分桶和 predictor 使用 Python 3.12 + NumPy + pytest。C2/C4 和 V1 checkpoint 评估需要 PyTorch；正式 GPU 结果使用 PyTorch 2.8.0+cu128、CUDA 12.8、RTX 4090。

```bash
python -m pytest -q
```

最近一次结果：本地无 Torch 环境 `73 passed, 4 skipped in 14.39s`；GPU/Torch 环境 `84 passed in 14.16s`。

## Phase A：旧生成器审计

这是本轮正式审计的参数；输出直接写入现有正式目录。

```bash
python scripts/audit_traffic_generator.py \
  --generator legacy \
  --families smooth_ar alternating_burst moving_hotspot sparse_switching bimodal heavy_tail_clipped \
  --sequence-lengths 64 1024 4096 \
  --num-sequences 20 \
  --short-window 16 \
  --medium-window 128 \
  --long-window 512 \
  --seeds 42 142 242 \
  --mean-level 2 \
  --std-level 1 \
  --max-entry 8 \
  --epsilon-mean 0.20 \
  --epsilon-var 0.30 \
  --max-period-lag 512 \
  --workers 8 \
  --output-dir outputs/traffic_audit
```

## Phase B：长期生成器审计

```bash
python scripts/audit_traffic_generator.py \
  --generator long \
  --families regime_switching_long stochastic_volatility rare_shock_recovery hotspot_random_walk same_moments_different_dynamics \
  --sequence-lengths 1024 4096 \
  --num-sequences 20 \
  --short-window 16 \
  --medium-window 128 \
  --long-window 512 \
  --seeds 42 142 242 \
  --mean-level 2 \
  --long-std-level 1.5 \
  --max-entry 8 \
  --max-period-lag 512 \
  --calibration-candidates 3 \
  --workers 8 \
  --output-dir outputs/traffic_audit/long_horizon
```

## Phase C1：V1 分桶

只分析原正式 held-out detail：

```bash
python scripts/analyze_v1_by_bucket.py \
  --detail outputs/moment_v1/formal/v1_formal_detail.csv \
  --formal-summary outputs/moment_v1/formal/v1_formal_summary.json \
  --output-dir outputs/v1_diagnosis \
  --report docs/V1_BUCKET_ANALYSIS.md
```

本轮最终报告使用训练/held-out 合并 detail：

```bash
python scripts/evaluate_v1_training_families.py \
  --checkpoint-dir checkpoints/v1_diagnosis/rebuilt \
  --training-seeds 42 142 242 \
  --families smooth_ar alternating_burst moving_hotspot sparse_switching \
  --num-sequences 3 \
  --eval-seed 2000042 \
  --heldout-eval-dir outputs/v1_diagnosis/rebuilt_v1 \
  --output-dir outputs/v1_diagnosis/training_family_eval

python scripts/analyze_v1_by_bucket.py \
  --detail outputs/v1_diagnosis/training_family_eval/combined_bucket_detail.csv \
  --formal-summary outputs/moment_v1/formal/v1_formal_summary.json \
  --output-dir outputs/v1_diagnosis \
  --report docs/V1_BUCKET_ANALYSIS.md
```

## Phase C2：相同当前 X、不同历史

`checkpoint-dir` 必须包含 `seed_<seed>/baseline/baseline_best.pth` 和 `seed_<seed>/moment/moment_best.pth`。

```bash
python scripts/evaluate_counterfactual_history.py \
  --checkpoint-dir checkpoints/v1_diagnosis/rebuilt \
  --num-pairs 200 \
  --seeds 42 142 242 \
  --training-seeds 42 142 242 \
  --topology Rear4GPU \
  --window-size 16 \
  --min-history 8 \
  --time-limit 20 \
  --max-entry 8 \
  --bootstrap-samples 2000 \
  --device cuda \
  --output-dir outputs/v1_diagnosis \
  --report docs/COUNTERFACTUAL_HISTORY.md
```

## Phase C3：流量可预测性

```bash
python scripts/train_traffic_predictor.py \
  --sequence-length 1024 \
  --families regime_switching_long stochastic_volatility rare_shock_recovery hotspot_random_walk same_moments_different_dynamics \
  --seeds 42 142 242 \
  --sequences-per-seed 4 \
  --train-sequences-per-seed 3 \
  --history-window 16 \
  --recent-steps 8 \
  --min-history 8 \
  --mean-level 2 \
  --std-level 1.5 \
  --max-entry 8 \
  --calibration-candidates 1 \
  --ridge-alpha 10 \
  --output-dir outputs/v1_diagnosis/predictor

python scripts/evaluate_traffic_predictability.py \
  --model-dir outputs/v1_diagnosis/predictor \
  --bootstrap-samples 2000 \
  --output-dir outputs/v1_diagnosis/predictability \
  --report docs/TRAFFIC_PREDICTABILITY.md
```

## Phase C4：Partial demand

```bash
python scripts/evaluate_partial_demand.py \
  --checkpoint-dir checkpoints/v1_diagnosis/rebuilt \
  --hide-ratios 0.25 0.50 \
  --observation-modes random_entries source_totals source_destination_totals partial_shards \
  --seeds 42 142 242 \
  --training-seeds 42 142 242 \
  --families regime_switching_long stochastic_volatility rare_shock_recovery hotspot_random_walk same_moments_different_dynamics \
  --num-sequences-per-family 1 \
  --sequence-length 32 \
  --window-size 16 \
  --min-history 8 \
  --time-limit 20 \
  --max-entry 8 \
  --bootstrap-samples 2000 \
  --device cuda \
  --output-dir outputs/v1_diagnosis \
  --report docs/PARTIAL_DEMAND_EXPERIMENT.md
```

正式运行得到 20,160 条结果，在 RTX 4090 上约 24 分钟。脚本不会训练模型，但会执行大量 policy forward 和 schedule decode。

## 结果目录

- `outputs/traffic_audit`：旧生成器和长期生成器逐 sequence 审计；
- `outputs/v1_diagnosis`：C1–C4 paired detail、summary 和 predictor；
- `checkpoints/v1_diagnosis/rebuilt`：按原正式配置重建的 3 个 baseline/Moment checkpoint；
- `docs`：Phase A–D 报告。

Phase D 只完成诊断和路线选择，没有实现路线 D 的性能优化。
