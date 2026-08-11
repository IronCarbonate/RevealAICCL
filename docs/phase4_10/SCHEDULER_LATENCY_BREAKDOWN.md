# Phase 4.10-1F：Scheduler Fast-Path Feasibility Audit — 调度器单步延迟分解

更新日期：2026-08-06
测量环境：region-41（2× Tesla V100-SXM2-32GB）/ Python 3.12.3 / torch 2.8.0+cu128 /
仓库 `/root/autodl-tmp/RLCCL-main`（与本地 172 文件逐项一致，关键源文件 md5 已核对）

## 1. 冻结范围（本审计不得更改）

| 冻结项 | 冻结内容 |
|---|---|
| router | `outputs/phase4_10/p10_1a_substrate/reference_router.py`（L2-R reference；`seed_router_params(2048, 4, 20260805)`；K=1；lexicographic tie-break；逐 chunk 真实 CUDA 完成事件） |
| workload | P10-1E 冻结输入：8 chunks × 4096 tokens × D=2048（chunk 计时）；48 tokens × 16 features 派生 world（T 由 router top-k 得到，seed 4042）；ratios (0.0, 0.75, 1.0)；stage_len=4；slots=80；time_limit=80；Rear4GPU 拓扑（容量/组带宽按最小值归一化为 1 单位） |
| 75% budget | reveal ratio 0.75（stage 1） |
| checkpoint 8 | NCHUNKS=8、per_chunk=48/8=6 |
| partial_current_only | `METHODS[3]`：每槽结构路径 = `enumerate_candidates(view)` + gate + `pack_candidate_batch`（`phase4_experiment.py`） |
| deterministic checker | `rlccl/uncertainty/execution.py::commit_proposal`（phase1-atomic-v1 语义：原子校验 + state_version 递增 + possession 更新） |

## 2. 方法与复现

测量脚本：`outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.py`
结果：`outputs/phase4_10/p10_1f_audit/p10_1f_scheduler_breakdown.json`

- Pass A：完整复现 P10-1E 单步定义（enumerate + gate + pack，`perf_counter_ns`，30 reps × 80 slots = 2400 样本）；
- Pass B：同循环逐组件计时（view / enumerate / gate / pack，15 reps × 80 slots = 1200 样本）；
- Pass C：first-commit 准备路径（view→enumerate→gate→pack→bind→checker commit，60 份 fresh world）；
- Pass D：子成本探针（单次 BFS、距离查找、增量 pack、digest、fast-view、enumerate-min）。

单步 p95 复现：

| 来源 | scheduler step p95 (µs) |
|---:|---:|
| P10-1E 记录（2026-08-06） | 12,290.03 |
| P10-1F Pass A（第 1 次运行） | 12,932.76 |
| P10-1F Pass A（第 2 次运行，最终 JSON） | 11,290.10 |

中位数 ~10.3ms、均值 9,830.6µs（最终运行）。p95 在 11.3–12.9ms 之间波动（Python 重负载 + CPU 抖动），与 P10-1E 12.29ms 同量级，复现成立。

## 3. 单步组件分解（Pass B，1200 样本）

| 组件 | mean (µs) | median (µs) | p95 (µs) | p95/step p95 | mean/step mean |
|---|---:|---:|---:|---:|---:|
| enumerate_candidates | 8,808.0 | 9,327.1 | 9,425.0 | 83.5% | 89.6% |
| gate（arrival 过滤） | 10.6 | 11.0 | 11.5 | 0.1% | 0.1% |
| pack_candidate_batch | 929.9 | 992.2 | 1,005.7 | 8.9% | 9.5% |
| 组件合计（step） | 9,748.5 | 10,332.0 | 10,436.2 | 92.4% | 99.2% |
| 单步整体（Pass A 独立计时） | 9,830.6 | — | 11,290.1 | 100% | 100% |
| build_scheduling_view（步骤外，首提交准备用） | 1,057.8 | 1,089.7 | 1,104.2 | — | — |

**归因结论：p95 口径组件合计占 92.4%（≥90%），均值口径占 99.2%**；残余 ~82µs（0.8%）为组件计时点引入的 `perf_counter_ns` 开销与抖动。

## 4. 主导子成本：BFS 距离计算（enumerate 内部）

通过插桩精确统计（本地复算与远程同源码）：

| stage | revealed tokens | 距离调用总数 | 自环（立即返回） | 真实 BFS |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 36 | 144 | 36 | 108 |
| 2 | 48 | 192 | 48 | 144 |

每次调用即 `canonical_shortest_path`（BFS）。探针：单次非自环 BFS 中位 55.1µs、p95 56.4µs。

stage-2 每次 enumerate：144 × 55.1µs ≈ 7,935µs ≈ enumerate 均值（8,808µs）的 **90.1%**；
对 stage-2 p95（9,427.9µs）≈ 84.2%，其余为 48 次自环调用与 Python 循环开销。

**≥90% 单步 wall time 由两类静态可缓存开销构成：BFS 距离重算（~7.9ms）与 `pack` 的重复 `batch_loads` 重算（~0.93ms）。**

## 5. 每 stage 分解（step p95，µs）

| stage | enumerate | gate | pack | step_sum |
|---:|---:|---:|---:|---:|
| 0（0 tokens，slots 0–3） | 1.5 | 4.8 | 1.1 | 7.2 |
| 1（36 tokens，slots 4–7） | 7,037.2 | 8.8 | 736.5 | 7,764.6 |
| 2（48 tokens，slots 8–79） | 9,427.9 | 11.5 | 1,006.2 | 10,444.1 |

p95 主要落在 stage-2（占 80 slots 中的 72 slots）。候选/门控/批大小统计：candidates p95=48、gated p95=48、batch p95=4（容量归一化为 1 单位/边/组后 pack 上限 ~4）。

## 6. First-commit preparation latency（Pass C，60 份 fresh world）

| 段 | p95 (µs) | 说明 |
|---|---:|---|
| prep_view（build_scheduling_view） | 962.0 | 含 SHA-256 digest 计算 |
| prep_step（enumerate+gate+pack） | 7,662.8 | 冻结单步定义 |
| prep_bind（bind_action + Proposal） | 14.1 | 序数→truth token 绑定 |
| prep_checker（commit_proposal） | 82.4 | 确定性 checker（原子校验+应用） |
| **prep_full（view→commit）** | **8,673.8** | 首提交总准备延迟 |

首提交发生在 slot 4（与 P10-1E 一致），提交动作数 = 2（p95=2）。决策后提交段（bind+checker）p95 = 95.5µs。

## 7. 开销分类（静态 / 动态 / 可缓存 / 可增量 / 不可消除）

| 组件 | 分类 | 依据 |
|---|---|---|
| BFS 距离调用（~7.9ms） | 静态 + 全量可缓存 | 距离是拓扑的纯函数，与 token/槽位无关；全对预计算即可 O(1) 查找（实测单次查找 p95 0.305µs） |
| enumerate 外层循环（~0.9ms） | 动态 + 部分可缓存 | 随 revealed tokens 变化；候选判定可用预计算距离 + 按 (holders, destination) 记忆化 |
| gate（11.5µs） | 动态 + 可合并 | arrival_slots 静态，谓词极廉价；可并入 enumerate 循环 |
| pack（~1.0ms） | 动态 + 可增量 | `can_add_candidate` 每次都重建 `batch_loads`（O(batch)）；运行累计 loads 可降为 O(1)/候选（探针 p95 138.3µs） |
| build_scheduling_view / digest（~1.1ms，其中 digest ~0.9ms） | 动态 + 可延迟/可惰性 | 结构路径的决策不消费 digest，仅事件账本消费；可惰性计算（值不变）或移出决策关键路径 |
| bind_action（14.1µs） | 动态 + 不可消除 | 必须把序数绑定到不透明 truth token |
| commit_proposal checker（82.4µs） | 动态 + 不可消除（不得跳过/弱化） | 每槽原子校验 + 应用；冻结语义 |

## 8. 诚实结论

1. **P10-1E 的 12.29ms 单步 p95 已精确复现并分解**：enumerate 83.5%、pack 8.9%、gate 0.1%（p95 口径合计 92.4% ≥ 90%）；enumerate 内 90% 以上是静态可缓存的 BFS 距离重算。
2. **当前实现没有“不可消除”的 12ms 瓶颈**：~99% 是静态/可缓存/可增量开销；真正不可消除的只有 bind+checker（~96µs）与观察读取。
3. first-commit 准备延迟 p95 = 8.67ms（slot 4，2 动作），其中决策段 7.66ms、view 0.96ms、bind+checker 0.096ms。
4. 单步 p95 存在运行间波动（11.3–12.9ms），与 P10-1E 12.29ms 同量级；任何快速路径目标必须以 p95 且在冻结 workload 上实测认证。

## 9. 限制

- 本分解只覆盖冻结的 48-token world 与 Rear4GPU 拓扑；更大规模（N tokens、D features、chunks ∈ {4,8}）未测。
- 探针（enumerate-min、incremental pack、fast-view）为纯测量代码，未改动任何生产模块；enumerate-min 与生产 enumerate 输出已做 48/48 候选一致性校验。
- 未运行 P10-1 formal；未生成/查看 formal test；未修改 router/workload/75%/ckpt8/partial_current_only/checker。
