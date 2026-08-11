# 下一方向决策

日期：2026-07-27（Asia/Shanghai）

## 决策

**唯一主路线：路线 D——终止 moments 主线。**

这里的“终止”指：不再扩大 MomentEncoder，不再重新训练相同的 action-level moment-conditioned policy，不进入原 V2/V3，也不把 partial-demand 或 meta-control 当作未经验证的续命理由。现有代码、checkpoint 和实验结果保留为可复核的负面资产。

后续工程投入转向 baseline decoder 与 synthesis path：decoder 并行化、synthesis latency 优化、CPU/GPU 交互优化、candidate pruning、current-traffic-based cache 和 beam/search distillation。

## 四条路线的证据判定

| 路线 | 触发条件与实证 | 判定 |
|---|---|---|
| 路线 A：重新训练 Moment-conditioned Policy | moments 必须有稳定预测力、重要 family 中稳定优于 baseline、扩大独立 sequence 后收益稳定 | **否决。** moment-only total RMSE 3.1729，明显差于 previous 1.6747/recent 1.6463；两个 held-out family 均显著为负；唯一小正向桶只有 3 条 sequence、2/3 seed 为正 |
| 路线 B：Partial-Observation AICCL | Full moment 不优于 baseline，同时 Partial moment 稳定优于 Partial baseline | **否决。** 前半条件成立，后半条件不成立；6 个 partial 条件全部只有 1/3 seed 为正，没有稳定正 CI |
| 路线 C：Statistical Meta-Control | moments/drift 不能改善 action，但能预测 baseline failure、OOD 或 reuse risk | **不选择。** 前半条件成立，但项目尚无 failure/OOD/reuse predictor 的正面实验；不能把未验证假设当作触发证据 |
| 路线 D：终止 moments 主线 | 当前预测弱、partial 无收益、分桶收益不稳，且继续 action conditioning 已触发停止条件 | **选择。** C1–C4、原正式 V1 和重建多 seed 结果方向一致 |

## 为什么唯一正向桶不改变决策

`sparse_switching` 的 Moment-full delta 为 `+0.0399`，sequence-bootstrap 95% CI `[0.0260, 0.0469]`，形式上满足预设桶判据。但它不能支撑路线 A：

- 只有 3 条独立 sequence，576 行主要来自方法、时刻和模型 seed 的重复；
- 只有 2/3 seed 为正，其中一个 seed 明显为负；
- 它是训练 family，两个 held-out family 分别为 `-0.1828` 和 `-0.1797`；
- 另一个训练 family `moving_hotspot` 为 `-0.2587`；
- 按 ACF、current/history deviation、hotspot、burst、regime、baseline completion/synthesis 等其他桶均无稳定收益。

这正符合停止条件“收益仅来自个别 seed 或极小样本桶”的实质，而不是扩大模型的依据。

## 触发路线 D 的关键证据

1. **停止条件 1 已满足。** 相同当前 X、topology 和初始状态下，仅改变历史就使 Moment 100% 改变 schedule；60.83% 为有害 action-level context interference，平均 completion delta `-0.6925`，95% CI `[-0.7759, -0.6092]`。
2. **停止条件 2 已满足。** moment-only 对当前 traffic summary 的预测不如 previous-value；有序 recent-history 也显著优于 moment 压缩。
3. **停止条件 3 已满足。** partial-demand 的所有条件都没有跨 seed 稳定收益。
4. **停止条件 5 的方向成立。** 在新增长期流量上的 C2/C4 中，Moment 仍整体劣于 baseline；旧生成器缺陷被修复后负结论没有消失。
5. **停止条件 6 已满足。** 唯一正向 family 桶是最小独立样本规模且 seed 方向不一致。
6. **停止条件 7 已满足。** Moment synthesis 比 baseline 慢约 7–10 ms，同时 completion 没有改善。

任一停止条件已足以禁止继续扩大 MomentEncoder；当前同时满足多项。

## 路线 D 的实施优先级

本文件只决定方向，不在 Phase D 中实现下列改动。

### P0：建立 baseline synthesis profile

对当前 baseline decoder 分阶段计时：candidate 构造、policy forward、合法性过滤、状态更新、schedule 序列化和 CPU/GPU 同步。按相同 traffic/topology/seed 做 paired profile，报告 mean/p50/p95/p99，而不是只看总 wall time。

### P1：candidate pruning

优先减少明显不可能或支配劣势的 candidate，并保证：

- schedule legality 100%；
- completion mean/p95/p99/CVaR95 不劣于未剪枝 baseline；
- timeout 不上升；
- synthesis p95 有配对置信区间的稳定下降。

### P2：decoder 并行化与 CPU/GPU 交互优化

批量化 candidate feature 和 logits 计算，减少逐 candidate Python 调用、重复 tensor 构造、host-device copy 和隐式同步。是否保留 GPU 路径应由小 batch 的真实 p95 决定，而不是默认 GPU 更快。

### P3：current-traffic-based cache

缓存键只使用当前可观测且与正确性相关的 topology、demand、initial state 和 decoder config hash。复用前执行合法性校验；任何历史 moments 都不进入 action cache key。

### P4：beam/search distillation

只有在 profile 证明 search 是主要瓶颈后，才用高预算 search 生成 teacher schedule，并以 baseline completion/legality 为硬验收门。蒸馏目标是降低 synthesis effort，不重新引入 history-moment action conditioning。

## 后续验收门

任何路线 D 优化都必须使用至少 3 个 seed、相同 traffic matrices、完整 sequence split 和 sequence-cluster bootstrap，并同时报告：

- completion mean、median、p95、p99、CVaR95；
- synthesis mean、p95、p99；
- legality 和 timeout；
- per-sequence paired delta 与 95% CI；
- raw result rows、独立 sequence 数、ACF 和 ESS；
- hostname、Python/PyTorch/CUDA、命令行、配置和 seed；git commit 在当前非 Git 工作区只能记录为 unavailable。

最低通过标准：legality 100%，completion/tail 不出现统计显著退化，timeout 不上升，并且 synthesis p95 跨 seed 稳定改善。

## 对路线 C 的边界说明

Statistical Meta-Control 不是当前主路线。现有 C3 预测的是当前 traffic summary，不是 baseline failure、OOD 或 reuse risk；因此没有证据声称 moments 能做好 gate/fallback/search-budget control。

如果未来有人单独提出并授权 failure-prediction 研究，它必须从只读离线评估开始，并与 current-X-only、previous-value、recent-history 和 oracle 标签基线比较。只有跨 topology、跨 seed 的 sequence-level 指标显著为正，才可形成新的路线决策；该可能性不构成本轮“同时选择 C”。

## 未选择路线的资源处理

- 保留 `checkpoints/v1_diagnosis/rebuilt`、paired CSV、JSON summary 和 Phase A–C 报告，用于复核与防止重复试验。
- 不删除 Moment 代码，以维持旧 checkpoint 和回归测试兼容；但将其视为冻结实验分支。
- 不追加 epoch、不扩展 hidden dimension、不叠加新的 moment feature，也不以单 seed provisional result 启动训练。

完整失败证据见 `docs/V1_FAILURE_DIAGNOSIS.md`；可复制命令见仓库根目录 `README.md`。
