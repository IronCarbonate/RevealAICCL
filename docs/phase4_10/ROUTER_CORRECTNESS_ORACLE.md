# Router / Reference Correctness Oracle

更新日期：2026-08-05

## 1. Oracle 定义

正确性 oracle = **确定性朴素 PyTorch reference**：

- 逐 token 计算 gating logits：`logits[b] = W x[b] + b`（无 batch 依赖）；
- top-k 选择：显式排序 + 确定性 tie-break（logits 相等取小 expert index）；
- histogram：逐 token 计数；
- 输出：expert_idx (B,)、score (B,)、per-expert 计数 (E,)。

优化路径（后续若引入 Triton/批量 kernel）必须与 oracle 输出**逐位一致**（数值容差内 + 完全一致的 index 与计数）。

## 2. Token 无丢失 / 无重复 / 最终 traffic 一致性检查

1. **无丢失**：每个输入 token 恰好出现在一个 expert 的路由结果中；`sum(histogram) == B`；
2. **无重复**：每个 token 的 expert_idx 唯一（k=1）；`len(unique(token_ids)) == B`；
3. **最终 traffic 一致性**：按 expert 聚合的 token 计数与 proxy truth matrix 的 per-destination 计数一致（映射：expert j ↔ destination j）；逐 sequence 断言。

## 3. 检查实现（P10-1A 协议冻结）

- 这些检查写入 substrate 的 self-test（新文件）；
- 若任一检查失败 → fail closed，不进入 P10-1B。
