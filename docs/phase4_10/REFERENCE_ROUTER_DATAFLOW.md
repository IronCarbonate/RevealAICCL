# Reference Router 数据流（Dataflow）

更新日期：2026-08-05

## 1. 数据流（真实 tensor → router logits → top-k → histogram → shard readiness）

```text
token tensor (B×D, fp32, GPU)
  │
  ▼
router gating: Linear(D→E)  ──► logits (B×E)
  │
  ▼
top-k: torch.topk(k=1)  ──► (expert_idx (B,), score (B,))
  │                        tie-break: logits 相等取小 expert index（确定性）
  ▼
histogram: torch.bincount(expert_idx)  ──► per-expert token 计数 (E,)
  │
  ▼
shard readiness: 按 shard 批次聚合已路由 token 数  ──► shard-ready 事件（CPU 时间戳）
  │
  ▼
（P10-2）expert packing → expert Linear → combine（本轮不实现真实 GEMM）
```

## 2. 到调度器的映射

- router 输出决定 token 的 destination/expert（对应 proxy 的 token destination）；
- 按 frozen profile：D0 在 checkpoint 16、D1 在 checkpoint 8 之前向调度器揭示对应比例（75%）的已路由 token；
- 调度器 = partial_current_only（协调器视图），只接收已揭示 token 集。

## 3. D0/D1 相同 token / 相同权重 / 相同 top-k（证明设计）

1. **相同 token**：两臂使用相同 corpus（3042/3142/3242）与相同 seed 生成的相同 token 序列；
2. **相同权重**：router gating 权重为单一冻结 checkpoint，D0/D1 共用；
3. **相同 top-k**：`torch.topk` 确定性；相同输入必得相同 expert_idx/score；
4. 唯一差异 = reveal 时机/粒度（profile）。等价性测试（P10-I1）将逐 token 断言两臂 router 输出一致。
