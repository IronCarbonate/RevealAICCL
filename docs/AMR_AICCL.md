# AMR-AICCL 实现与运行说明

本仓库已完成 V0 和 V1。V1 已按正式规模完成三训练种子、严格 held-out family 的配对评测，但阶段门结果为 **NO-GO**，所以根据任务书的停止条件，没有继续实现 V1.5、V2、V3、V4。

## 实现状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| V0 | completed | problem/evaluator 统一、移除 `D` 依赖、持久 optimizer/checkpoint、cache schema 和 A2AV 范围修复 |
| V1 | completed / NO-GO | history-only moments、6 类流量序列、MomentEncoder、node/global/candidate 特征、PPO 精确重放、训练和四路配对评测均已实现 |
| V1.5 | not started | V1 未通过阶段门，未实现 episode CVaR |
| V2 | not started | 未实现 soft risk budget |
| V3 | not started | 未实现 trust controller |
| V4 | not started | 未实现 OOD/fallback |

## V1 核心行为

- 当前完整 traffic matrix `X_t` 始终用于构造真实 demand。
- 第 `t` 个 context 只使用 `X_0..X_{t-1}`；`X_t` 不会提前写入 estimator。
- baseline 保持原始 `5/2/5/2` 特征宽度，可严格加载旧 state dict。
- moment mode 使用 node 12 维、edge 2 维、candidate 9 维、chunk 2 维和 global moment 8 维。
- moments 只影响软特征和网络打分，不修改 source-has-chunk、edge capacity、shared group 等硬约束。
- PPO buffer 保存构造后的紧凑特征和 candidate 重放数组，不在每个 slot 重复保存完整 mean/var/current matrix。
- rollout 和 `recompute_logp_slot()` 使用完全相同的 moment features。

## 运行

单模型训练：

```bash
python scripts/train_moment_policy.py \
  --policy-mode moment \
  --topology Rear4GPU \
  --num-sequences 10 \
  --sequence-length 64 \
  --window-size 16 \
  --min-history 8 \
  --epochs 5 \
  --device cuda
```

baseline 使用同一序列但不给 moments：

```bash
python scripts/train_moment_policy.py --policy-mode baseline --device cuda
```

单 seed 四路配对评测：

```bash
python scripts/evaluate_moment_policy.py \
  --baseline-checkpoint checkpoints/baseline_best.pth \
  --moment-checkpoint checkpoints/moment_best.pth \
  --families bimodal heavy_tail_clipped \
  --num-sequences 10 \
  --sequence-length 64 \
  --device cuda
```

三 seed 正式实验：

```bash
python scripts/run_v1_ablation.py --seeds 42 142 242 --device cuda
```

## 测试与性能结论

- 远程 RTX 4090 / PyTorch 2.8：`51 passed`。
- 本机最终 NumPy 测试：`41 passed, 3 skipped`；跳过项仅因为本机未安装 PyTorch。
- 正式 V1：15,360 个 schedule，合法率 100%，timeout 0%。
- moment-full 相比 baseline：mean completion `8.5750 -> 8.7563`（退化 2.11%），p95 持平为 `10.6667`，CVaR95 `10.6695 -> 10.9638`，平均合成耗时增加约 13.55%。
- correct context 稳定优于 shuffled，说明模型确实使用了 moments；但 moments 没有稳定优于已知当前 demand 的 baseline。

详细证据见：

- `docs/PERFORMANCE_V0.md`
- `docs/PERFORMANCE_V1.md`
- `outputs/moment_v1/formal/v1_formal_summary.json`
- `outputs/moment_v1/formal/v1_formal_detail.csv`

## 阶段结论

V1 在 `bimodal` 和 `heavy_tail_clipped` 上均未获得跨训练种子的稳定 p95/CVaR 改善，且 mean completion 退化超过预设 2% 上限。因此不能进入 V1.5。若后续提出新的研究假设，建议把 moments 用于 OOD 检测或受保护的风险/fallback 信号，而不是直接声称能提高已知当前 demand 下的 schedule quality。
