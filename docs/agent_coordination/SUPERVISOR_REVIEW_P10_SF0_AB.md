# Supervisor Review — Phase 4.10-1F：P10-SF0-A / P10-SF0-B

更新日期：2026-08-06
审核对象：Scheduler Fast-Path Feasibility Audit（有界可行性阶段，未实施优化）

> **SUPERSEDED IN PART BY PHASE R0（2026-08-10）**：本审查是历史记录。
> 其中“1.045ms 理论下界”改为 implementation estimates：step-only 1,043.1µs、
> 含 bind/checker 1,139.5µs、含 digest 2,047.2µs；419.8µs 是 replay
> candidate window，不能作为真实 concurrent pipeline window。

## 1. 独立复核（逐项对照本轮 12 项要求）

| # | 要求 | 证据 | 结果 |
|---:|---|---|---|
| 1 | 冻结 router/workload/75%/ckpt8/partial_current_only/checker | 冻结声明 + 本地/远程关键源文件 md5 一致 | PASS |
| 2 | 精确分解单步 12.29ms p95 | p95 复现 11.29–12.93ms；enumerate 83.5% / pack 8.9% / gate 0.1% | PASS |
| 3 | 至少解释 90% wall time | p95 口径组件合计 92.4%、均值口径 99.2%；BFS 占 enumerate 均值 90.1% | PASS |
| 4 | 静态/动态/可缓存/可增量/不可消除分类 | 分类表完成；~99% 可消除/可缓存，不可消除仅 bind+checker ~96µs | PASS |
| 5 | 测量 first-commit preparation latency | prep_full p95 = 8,673.8µs（view 962 + step 7,663 + bind/checker 95.5；slot 4、2 动作） | PASS |
| 6 | 实现 fast-path estimates | step-only 1,043.1µs；bind/checker-inclusive 1,139.5µs；digest-inclusive 2,047.2µs | PASS（测量锚定 estimate） |
| 7 | 旧 replay 配置能否满足预注册目标 | **否**：step-only estimate 1,043.1µs > 336µs；当前 11.3–12.9ms | PASS（仅限旧配置） |
| 8 | 实际目标预注册 <336µs | L_total < 336µs（L_sched ≤ 200µs、L_commit ≤ 135µs）已预注册 | PASS |
| 9 | 输出允许的纯实现层优化计划 | DRAFT 计划（E1–E7，未实施） | PASS |
| 10 | 不实施大规模优化，先判定 SF0-A/SF0-B | 仅测量探针；生产路径零改动 | PASS |
| 11 | 只有 SF0-B PASS 才向用户申请实施 | SF0-B = FAIL → 不申请实施 | PASS |
| 12 | 完成后停止等待用户审核 | 本轮停止点 | PASS |

## 2. 禁止项复核

未运行 P10-1 formal；未生成/查看 formal test；未按结果更换 workload；未修改 scheduler production path
（`rlccl/scheduling/*`、`rlccl/uncertainty/execution.py` 等零改动，md5 前后一致）；
未跳过/弱化 checker；未改变候选动作或排序语义；未实现真实 expert GEMM/combine；无人工 delay；
未改 75%/checkpoint 8；未恢复任何被冻结机制；未进入 DeepEP/L3；未创建额外 Subagent。全部 PASS。

## 3. 判定

### P10-SF0-A = PASS / NO VETO

审计完整：单步 12.29ms 已复现并分解（≥90% 解释）、首提交延迟实测、实现 fast-path estimates 给出、
冻结项未变、禁止项未触碰。

### P10-SF0-B = FAIL / NO VETO

允许的纯实现层优化计划（BFS 预计算、增量 pack、融合 gate、惰性 digest、零拷贝 view）的实测锚定下界
step-only estimate 为 1,043.1µs、含 bind/checker 为 1,139.5µs；达到 <336µs 依赖未实施/未测量的
记忆化或向量化技术，无法认证。

## 4. 结论

1. **P10-1 formal 继续 HOLD**；P10-F0 = FAIL/NO VETO 维持；P4 判定式更新为
   `L_sched_p95 + L_commit_p95 < 336µs`（子目标 200µs + 135µs）；
2. **不向用户申请开始实施优化**（SF0-B FAIL）；
3. 优化计划作为草稿保留；下一合法动作是用户审核并给出方向：
   - 批准向量化/记忆化实现阶段（新授权，实测认证 <336µs 后再过 P10-1A），或
   - 调整窗口/工作负载/语义定义（用户方向），或
   - 终止该路径。
4. 本轮停止，等待用户审核。
