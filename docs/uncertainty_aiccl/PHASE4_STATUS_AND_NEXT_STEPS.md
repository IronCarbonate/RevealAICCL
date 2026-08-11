# Phase 4 完成情况与后续工作

更新日期：2026-08-03

## 一句话结论

Phase 4 正式执行前的协议冻结、核心代码实现、测试、流水线建设和远程环境准备已经完成；但是最后一次经批准的远程正式执行收到类似 `SIGKILL` 的强制终止，未生成任何正式结果。因此当前状态是 **FORMAL FAILURE / HOLD（正式执行失败、阶段暂停）**，不是“Phase 4 已完成”，也暂时不能判断 H2 是否成立。Phase 5 Gate 保持关闭。

这里的“失败”只指执行过程没有跑完，不代表所研究的 robust prefix 方法在科学效果上失败。

## 1. Phase 4 原本要回答什么

Phase 4 对应 Gate H2，核心问题是：在完整 demand 尚未揭示时，先执行一个对多种可能场景都较安全的短 schedule prefix，随后根据新揭示的信息进行修复或重规划，是否比完全等待、只看当前已揭示信息或使用单点方案更好。

正式比较需要同时考虑：

- 平均 completion、p95、p99 和 CVaR95；
- 相对 full-information oracle lower bound 的 regret；
- synthesis 与 replan 的时间开销；
- recourse 次数、被替换的未执行计划和 reveal 后恢复时间；
- legality 必须为 100%，timeout 不能增加；
- 至少三个 seed、完整 sequence 划分、配对比较和 sequence-level 置信区间。

只有正式结果满足预注册的 H2 条件，才能建议进入 Phase 5；工程代码通过测试本身不能证明 H2 成立。

## 2. 已经完成的工作

### 2.1 协议和研究边界

- 冻结了 Phase 4 的数据划分、方法集合、`H/P/risk lambda` 配置、validation 选择规则、test 一次评估规则和 H2 判据。
- 使用全新的 Phase 4 sequence 集合，并通过摘要排除了 Phase 1/H1 和 Phase 3B 已使用的 150 条 sequence，避免测试集复用。
- 保持 Phase 3B 的 prediction-free ambiguity 路线；scenario 只用于给候选动作评分，不能把未揭示 demand 变成可执行动作。
- 明确普通方法与 oracle upper bound 隔离，防止未来信息泄漏。
- 冻结 paired traffic、topology、reveal、seed、evaluator、timeout 和 legality checker。

### 2.2 调度与在线补救实现

已实现并审计：

- scenario adapter 和 observation reconciliation；
- 多场景共同候选动作评价与 robust prefix；
- `H = 2/4/8/16`、`P = 1/2/4/8` 的合法组合；
- reveal 后 residual repair 和 suffix replacement；
- 已执行动作不可撤销，未执行 suffix 可以替换；
- 每次 replan 使用当前真实 holder/state version，而不是回到初始状态；
- 所有 commit 继续经过原有 deterministic feasibility checker；
- ratio 0 时未知 token 不可执行，ratio 1 时场景退化为已知 truth singleton；
- 九种正式方法的 paired episode 执行，包括 Wait、Partial、point-plan、scenario-robust 和 oracle 上界类对照；
- 完整的 event ledger、episode 汇总、sequence 汇总、timing 和 Gate 条件重算。

### 2.3 正式产物与完整性机制

正式流水线设计为只在全部计算和两阶段 read-back 校验成功后，原子发布以下八个产物：

1. `manifest.json`
2. `h1_best_point_model.json`
3. `raw_validation_metrics.csv`
4. `raw_test_episode_metrics.csv`
5. `raw_test_sequence_metrics.csv`
6. `raw_test_execution_events.csv`
7. `raw_timing_metrics.csv`
8. `summary.json`

流水线会检查固定文件集合、精确 schema、行数、主键与排序、row digest、logical/scientific hash、raw → episode → sequence → Gate 的重算一致性，以及 provisional → final → atomic rename 状态机。异常会 fail closed，不发布看似完整的结果。

### 2.4 重要缺陷及修复

正式准备过程中发现 oracle 在 reveal stage 1–3 直接使用原始 recent-history scenario 时，部分 scenario 的载荷可能小于当前已经揭示的 token 数，导致 residual projection 拒绝继续。该问题经过真实 RED 用例复现后，只修改了 `phase4_experiment.py`：oracle stage 0–3 先按照可信 observation 做 Phase 3B reconciliation，再调用既有 `oracle_support_upper_bound(k=8)`；stage 4 truth singleton 和 planner 的 fail-close 行为保持不变。

修复后的关键冻结文件：

- `rlccl/scheduling/phase4_experiment.py`：SHA-256 `696E75BD502A511B5578F7390A81020BDB03783E71B01265F100ED849454E52A`
- `tests/test_phase4_experiment.py`：SHA-256 `E3C3D6A47E0CC6E636E088ACD9DE06C2C33DAFFD4CB9DF33E1A719CA88E82214`
- Phase 4 protocol：SHA-256 `4246D661D3E9E316B10F730E3AC17B61BDBD15C6677965EFDC1C24F7898F2068`

## 3. 测试和远程准备完成情况

### 3.1 修复后的本地独立验证

- 定向 oracle undercoverage 回归：`1 passed`
- Phase 4 focused：`125 passed`
- 全仓：`494 passed, 4 skipped, 18 warnings`

四个 skip 均为既有 Torch 环境缺口，没有伪报为通过；18 个 warning 为已知 sklearn warning，也没有隐藏。

### 3.2 服务器准备与独立验证

服务器端完成了：

- 172 个文件逐项 hash 校验，差异为 0；
- 固定 Python 3.12.3、NumPy 1.26.4、SciPy 1.13.1、scikit-learn 1.5.1 等环境；
- 定向测试：`1 passed`；
- focused 测试：`125 passed`；
- 全仓测试：`494 passed, 4 skipped, 18 warnings`；
- 原子目录 rename 演练通过；
- `nohup + setsid + flock + timeout` 的断开/重连演练通过；
- 正式启动前确认 destination 不存在、staging 为 0、正式进程为 0；
- 固定单线程 BLAS/OMP 环境，以保证方法顺序、浮点行为和产物 hash 的可复现性。

这些证据说明正式代码和运行通道通过了预运行检查，但不能替代正式效果实验。

## 4. 正式执行的实际情况

### 4.1 前期执行事故

Phase 4 在到达最终服务器运行前经历过三类已记录事故：

1. Windows PowerShell 首次调用在 Python import 前发生 argv 引号损坏，未生成任何输出。
2. 修正调用后运行约 4 小时，触发 oracle residual undercoverage；流水线正确 fail closed，随后按 RED → 最小修复 → 全量回归的流程完成修复。
3. 修复后的本地长运行因前台工具/会话中断而丢失执行单元；无 destination、无 staging、无可用结果。为此才改用服务器原生 detached 运行。

这些事故都没有被包装成有效实验结果，也没有绕过监督准入自动重跑。

### 4.2 最终远程正式执行

监督准入任务：`P4-SUP-REMOTE-FORMAL-ADMISSION-001`，结论为只允许一次 detached remote attempt，失败后不得自动重跑。

| 项目 | 记录 |
|---|---|
| 服务器 | `connect.westb.seetacloud.com:26969` |
| 启动时间（UTC） | `2026-08-02T16:22:38.462996903Z` |
| 结束时间（UTC） | `2026-08-03T05:21:24.340385047Z` |
| 北京时间 | 2026-08-03 00:22:38 至 13:21:24 |
| 实际运行时长 | 约 12 小时 58 分 46 秒 |
| 外层 timeout | 144000 秒，即 40 小时 |
| 退出码 | `137` |
| 结束后正式进程 | 0 |
| destination | 不存在 |
| staging | 0 |
| 正式八产物 | 0 个 |

`137 = 128 + 9`，说明进程受到类似 `SIGKILL` 的强制终止。现有只读诊断表明：

- 实际运行不足 40 小时，因此不是冻结的外层 timeout 直接触发；
- 主机没有在该时段重启；
- 在当前可访问的 kernel 日志中没有发现 OOM-kill 记录；
- 当前 cgroup 的 `oom`、`oom_kill` 和 `max` 计数均为 0；
- stderr 只有 sklearn 的收敛 warning，没有 Python traceback。

运行时长排除了冻结外层 40 小时 timeout 计时器的直接触发；当前可访问证据未支持 OOM，但当前 cgroup 计数和可访问日志未必构成完整历史记录，因此仍不能排除历史 OOM、平台/宿主机事件或其他 `SIGKILL` 来源，也不能唯一确定信号发送者。sklearn warning 没有证据表明与终止存在因果关系。

## 5. 当前到底完成了多少

| 工作项 | 状态 | 能否作为科学结论 |
|---|---|---|
| Phase 4 问题定义与预注册协议 | 已完成 | 只能说明实验设计已冻结 |
| robust prefix、recourse 和九方法实现 | 已完成 | 不能单独证明有效 |
| 信息隔离、legality、artifact 完整性测试 | 已完成并通过 | 只能说明已覆盖的工程合同通过 |
| oracle undercoverage 缺陷修复 | 已完成并回归通过 | 只能说明已知缺陷已关闭 |
| 本地与远程预运行测试 | 已完成并通过 | 不能代替正式效果实验 |
| 远程正式执行尝试 | 已终止，exit 137 | 没有结果可分析 |
| 八个正式 artifact | 未生成 | 无法做 raw/summary/Gate 复算 |
| H2 最终判定 | 未完成，HOLD | 既不能判 PASS，也不能判 FAIL |
| Phase 4 总体 | 未完成，FORMAL FAILURE / HOLD | 禁止声称 Phase 4 有效果或无效果 |
| Phase 5 Gate | CLOSED / HOLD | 不得进入 Phase 5 |

当前唯一允许的正式结论是：**经批准的远程正式执行以退出码 137 非正常终止，没有生成正式结果产物。**

不能据此声称：robust 方法优于或劣于 baseline、H2 成立或不成立、tail/overhead/legality 的正式统计表现如何，或 Phase 4 已经完成。

## 6. 接下来要做什么

后续应严格按以下顺序进行。

### 第一步：保留证据并完成只读诊断

- 保留 `formal.start`、`formal.end`、`formal.exit`、stdout/stderr、冻结 hash、launcher 和 preflight 记录；
- 向云平台查询 2026-08-03 13:21（北京时间）附近的实例、容器或宿主机事件；
- 检查平台是否存在最长作业时间、空闲清理、进程组清理、容器回收或隐藏的资源限制；
- 如平台无法提供日志，明确记录“终止发送者未知”，不能猜测性归因为 OOM 或平台驱逐。

### 第二步：由用户决定是否授权新的正式重跑

不得自动重跑。若要继续，需要用户重新明确授权一次 Phase 4 formal retry。GPU 不是当前实现的必要条件；更重要的是选择经平台文档或供应方确认可连续运行至少 40 小时、不会按策略清理 detached 进程，并能提供实例、容器或宿主机事件记录的稳定 CPU 环境。

### 第三步：重新冻结并接受监督准入

新的运行必须逐项说明哪些内容不变、哪些内容改变：

- 科学代码、数据 corpus、manifest、seed、方法和 Gate 判据；
- Python/NumPy/SciPy/sklearn 与 BLAS 环境；
- CPU 数、线程数、内存、timeout、launcher 和输出路径；
- 是否保留单线程。

如果改用多线程或并行 episode，它就属于新的执行配置，必须先实现确定性并行归并、补充测试、重新冻结 hash，并重新验证结果不受执行顺序影响；不能在已经失败的运行上直接切换后声称是同一次实验。

Supervisor 必须在新的 clean preflight、完整测试、单实例检查和输出目录检查之后，再给出新的 `ALLOW / NO VETO`。没有新准入不得启动。

### 第四步：完成一次有效正式运行并独立复核

有效运行至少需要满足：

- 进程 exit 0；
- 恰好生成八个正式产物；
- staging 清零且 destination 通过原子发布；
- 两阶段 read-back、schema、行数、digest 和 hash 全部通过；
- Main 独立执行 raw → episode → sequence → H2 conditions 1–9 重算；
- Supervisor 独立审查并给出 Gate H2 的 `PASS / FAIL / HOLD`。

### 第五步：根据 H2 裁决分支

- 若 H2 PASS：仍需先停止并等待用户授权，之后才可进入 Phase 5 的 rolling configuration/Pareto frontier。
- 若 H2 FAIL：按 reveal latency、公共安全动作、replan 开销、scenario coverage、目标定义、只能提前准备或提前信息无价值等原因分类，再决定后续研究路线。
- 若仍为 HOLD：继续处理证据或执行有效性问题，不能绕过 Gate。

## 7. 建议的最近一步

当前最合理的下一步不是直接改算法，也不是进入 Phase 5，而是：**先取得云平台对 2026-08-03 13:21 附近强制终止事件的解释；随后由用户明确授权，在稳定且可连续运行的环境中重新执行一次经过重新冻结和监督准入的 Phase 4 正式实验。**
