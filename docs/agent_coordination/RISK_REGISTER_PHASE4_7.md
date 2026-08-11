# Risk Register — Phase 4.7

更新日期：2026-08-04

| # | 风险 | 等级 | 状态 | 缓解 |
|---|---|---|---|---|
| R-4.7-1 | 把 proxy 的"免费提前精确 entry"当作真实部署事实 | 高 | 已控制 | 语义审计明确：真实系统只有本地流式计数/直方图；全局量需同步；free-early-truth 仅 oracle |
| R-4.7-2 | 未测量同步/控制成本被设 0，导致虚假 E2E 收益 | 高 | 已控制 | cost model 强制参数化并校准；H5 PASS 必须计入成本后 E2E>0 |
| R-4.7-3 | 把 completion-only 收益冒充 E2E 收益 | 高 | 已控制 | H5 主指标为 J（含全部 T 分项），禁止只报 completion |
| R-4.7-4 | 新 corpus 与旧正式 test/Route A 重合 | 高 | 未开始（H5） | 预注册 zero-overlap digest 检查（H2 45 + Route A 45） |
| R-4.7-5 | 普通方法读取全局真值/未来信息 | 高 | 已控制 | fast layer 只用 `partial_current_only` 语义；full-info/cost-free 仅上界 |
| R-4.7-6 | 真实系统语义（router/top-k/sync）与审计不符 | 中 | 未验证 | H5 前用 server micro-benchmark 校准；必要时申请 Real-System Semantics Agent（需用户批准） |
| R-4.7-7 | R0 误判（无真实可实现中间信息却进入 H5） | 高 | 已控制 | R0 五条件逐条核对（见 Supervisor 复核） |
| R-4.7-8 | 同一 sequence 跨 split 或行级当独立样本 | 高 | 未开始（H5） | 按完整 sequence 划分；sequence-level bootstrap/ESS |
| R-4.7-9 | reveal 频次过高导致同步/控制开销吞噬收益 | 中 | 未开始（H5/H6） | 预算 B 与 marginal benefit/unit 指标 |
| R-4.7-10 | full-information scheduler 优化路线被同时启动 | 高 | 已控制 | 明确禁止；Route A 结论不构成该路线依据 |
