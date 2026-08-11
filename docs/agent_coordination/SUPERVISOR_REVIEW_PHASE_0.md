# Supervisor Review — Phase 0 Gate

日期：2026-07-30（Asia/Shanghai）  
任务：`P0-SUP-002`  
角色：Supervisor / Project Director  
审查性质：独立、只读 Gate 审查；除本报告外未修改任何文件

## 1. Gate 判定

| 判定对象 | 结论 | 含义 |
|---|---|---|
| Phase 0 审计与问题冻结交付 | **ALLOW** | Phase 0 的文档、审计、任务账本和可执行测试证据满足本阶段 Gate；无须返工后再审。 |
| 实际启动 Phase 1 | **HOLD** | 用户尚未批准进入 Phase 1。`ALLOW` 只表示 Phase 0 交付合格，不构成实施授权；所有 P1 任务必须保持 `planned`。 |
| Veto | **NO VETO** | 未发现需要否决 Phase 0 交付的实质问题。若在用户批准前实施 P1、增加 Subagent 或修改冻结路线，将立即触发 veto。 |

Supervisor 允许 Main 将 Phase 0 标记为完成并向用户提交审核，但必须在此处停止。只有用户明确批准后，Main 才可启动 Phase 1。

## 2. 本阶段目标

Phase 0 只负责：

1. 确认项目真实结构和现有数据流；
2. 冻结 direct moment-conditioned action policy 及其扩展路线；
3. 把研究目标固定为“提升 AICCL 在不确定流量条件下的调度效果”；
4. 明确不确定性发生的决策时刻，以及 ground truth、observation、scenario planning、execution 和 evaluator 的边界；
5. 建立 Main、Supervisor、Core 三方非重叠的任务账本与监督机制；
6. 运行并如实记录当前可执行的完整测试；
7. 只拆分 Phase 1，不实现 Phase 1；
8. 完成后等待用户审核。

## 3. 审查输入与证据

本次完整重读：

- `docs/agent_coordination/TASK_LEDGER.md`
- `docs/agent_coordination/DECISION_LOG.md`
- `docs/agent_coordination/RISK_REGISTER.md`
- `docs/uncertainty_aiccl/PROBLEM_DEFINITION.md`
- `docs/uncertainty_aiccl/CODE_AUDIT_UNCERTAINTY.md`

并按需复核总指令、历史报告、`ProblemInstance`、traffic/sequence runner、decoder、partial-demand、evaluator、predictor 和相关测试。Core 审计文件实测 SHA-256 为：

```text
766A5B8B5EEB5111927620381C09E97D5623E16922A51A9196711C2A4B35D193
```

该值与任务账本记录一致。

## 4. Phase 0 Gate 逐项核对

| Gate 条件 | 证据 | 判定 |
|---|---|---|
| 用户目标被正确写入文档 | `PROBLEM_DEFINITION.md` 第 1 节、`DECISION_LOG.md` D-001、任务账本均固定为不确定流量下的 AICCL 调度效果 | PASS |
| direct moment action 分支被冻结 | `PROBLEM_DEFINITION.md` 第 2 节、D-002 和 Core 审计明确禁止扩大 `MomentEncoder`、增加 moment feature、追加同路线训练或直进旧 V2/V3/V4 | PASS |
| synthesis 方向冲突已解决 | D-003 明确以新指令取代 `NEXT_DIRECTION_DECISION.md` 中的 synthesis 主路线；仅 Gate 2 证明收益受 overhead 抵消并经用户批准后可作为辅助任务 | PASS |
| 不确定性发生时刻被明确 | `PROBLEM_DEFINITION.md` 第 3 节定义私有 `X_t`、`O_{t,r}=M_{t,r}⊙X_t`、单调 reveal、最终全揭示和逐 stage 的 reveal/decision/check/execute 顺序 | PASS |
| ground truth / observation / scenario / execution / oracle 分离 | `PROBLEM_DEFINITION.md` 第 4–5、8 节和 D-004/D-006 给出能力隔离与 oracle 单独入口 | PASS（设计层） |
| Agent 职责无重叠 | 账本只有 Main、Supervisor、Core；文件 owner 唯一；Main 管协调文档，Core 管审计/未来实现，Supervisor 只读审查与 Gate 报告 | PASS |
| 现有测试通过或原始失败被记录 | Main/Core 两次一致得到 `73 passed, 4 skipped`；默认 Python 无 pytest 的失败和 4 项 skip 原因均记录；Supervisor 又独立复跑得到相同计数 | PASS（带明确覆盖缺口） |
| Phase 1 只拆分未实施 | 所有 P1 任务为 `planned` 且阻塞于用户批准；未发现 `rlccl/uncertainty` 业务模块 | PASS |
| Phase 1 拆分可验收且顺序正确 | `P1-SEM-001 -> P1-TEST-001 -> P1-UENV/REVEAL/EXEC -> BASE/METRIC -> INTEG -> SUP`，先冻结语义、再写失败测试、后实现 | PASS |

## 5. 已完成事项

1. Main 已建立字段完整的任务账本、决策日志和风险登记，并给所有可写文件分配唯一 owner。
2. Core 审计覆盖总指令要求的 11 个强制项：`ProblemInstance`、traffic matrix、scenario/chunk、schedule state、feasibility mask、evaluator、moment context、旧 partial-demand、可复用代码、新接口和泄漏风险。
3. direct moment action 分支已被文档冻结，历史负面结果被保留，没有删除 Moment 代码或 checkpoint。
4. 历史 synthesis 主路线冲突已显式解决，没有用工程优化替代新研究目标。
5. 旧 partial-demand 被正确降级为审计/对照资产，没有冒充 Phase 1 reveal environment。
6. 关键泄漏已被准确登记：decoder pruning 的真实 demand 旁路、完整 `ProblemInstance`、完整 chunk map、current-matrix moment feature、generator latent metadata、evaluator 能力边界和 future reveal/cache 通道。
7. checker 的动作域缺口已被承认，并在 Phase 1 计划中加入 reveal-aware deterministic validator。
8. Phase 1 的五个必需抽象、五类 reveal mode、五个 ratio、五个 baseline、recourse metrics 和专项测试均已进入可验收任务拆分。
9. Main、Core、Supervisor 共三次运行当前可执行的完整 pytest，结果一致；没有安装依赖或伪造 Torch/GPU 结果。
10. 本轮未创建额外 Subagent，未训练模型，未实现 predictor、robust prefix、recourse、三时间尺度或 synthesis 优化。

## 6. 未完成事项

以下项目未完成，但属于明确记录的环境缺口、Phase 1 工作或外部授权，不构成 Phase 0 文档审计失败：

1. 四项 Torch 测试本轮实际未执行：
   - `tests/test_moment_policy_shapes.py`
   - `tests/test_optimizer_checkpoint.py`
   - `tests/test_sequence_evaluator.py`
   - `tests/test_partial_demand.py` 中 decoder 等价测试
2. 当前目录不是有效 Git worktree，无法提供 clean/dirty 状态、commit 或可靠 diff 基线。
3. `UncertainProblemInstance`、`DemandRevealProcess`、`PartialObservationState`、`ScenarioSet` 和 `RecourseMetrics` 尚未实现；这是正确的 Phase 0 停止边界。
4. decoder line 425、policy/truth 能力隔离和 reveal-aware checker 尚未修复/实现；它们已被隔离为 Phase 1 的强制任务与测试，不得在旧 partial 路径上绕过。
5. 用户尚未批准实际启动 Phase 1。
6. H1/H2/H3 尚未开始，不能给出任何 Gate 1/2/3 收益结论。

## 7. 测试与实验有效性

Supervisor 独立运行：

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider
```

实际结果：

```text
73 passed, 4 skipped in 22.52s
SUPERVISOR_TEST_EXIT_CODE=0
SUPERVISOR_TEST_ELAPSED_SECONDS=23.551
```

四项 skip 均为 `could not import 'torch': No module named 'torch'`。处置如下：

- 这 4 项保持 **NOT RUN / SKIPPED**，不得改写为通过。
- 它们覆盖 Moment/optimizer/sequence evaluator/partial decoder 的 Torch 路径，所以当前证据不能声称完整 Torch 回归通过，也不能复用历史远程 `84 passed` 冒充本轮结果。
- direct moment policy 已冻结，Phase 0 的产物是审计与问题定义，且本轮未改业务代码，因此该环境缺口不否决 Phase 0。
- 如果 Phase 1 修改或依赖 decoder/model/Torch 路径，必须在 Phase 1 Gate 前于具备 Torch 的环境实跑相关测试；若届时仍不可运行，必须继续公开为 Gate 风险，由 Supervisor 单独判断，不能自动豁免。

本轮没有新的调度收益实验。历史 V1/C1–C4 数据只作为冻结方向和识别风险的既有证据，不能充当新 uncertainty environment 的性能结果。当前 `73 passed` 只说明可执行回归子集没有失败，不证明 Phase 1 语义已经实现。

## 8. 代码风险

| 风险 | 审查结论 | Gate 处置 |
|---|---|---|
| decoder 大候选剪枝在 `rlccl/envs/decoder.py:425` 读取真实 `demands` | 已被 Main/Core 独立识别并登记为 R-004 | Phase 1 必须以 >512 candidates 的反事实测试封堵；在此之前旧 partial decoder 不可复用 |
| 完整 `ProblemInstance` 同时公开 truth 与 metadata | 已登记为 R-005，问题定义禁止普通 policy 接收该对象 | Phase 1 必须实现受限 view/能力隔离；不能靠调用约定 |
| evaluator checker 未拒绝负数、非二值、NaN/Inf，也不识别 reveal token | 已登记为 R-015，D-010 冻结前置 reveal-aware validator | Phase 1 专项测试必须确定性拒绝非法动作，再调用旧物理 checker |
| ratio 0/空 action set 兼容性 | 已登记为 R-013 | 不得用 truth-derived padding 或完整 `C` 偷渡；专项测试必须覆盖 |
| 无 Git 基线 | 已登记为 R-001 | 继续使用唯一 owner、明确新增文件清单，不声称工作区干净 |

这些风险是 Phase 1 的已知输入，不是被掩盖的 Phase 0 缺陷。若 Phase 1 直接复用旧接口而不闭环，将触发 veto。

## 9. 数据泄漏风险

当前设计审计已覆盖以下泄漏面：

- 完整 `X_t`、真实 `demands`、`initial_state`、`C` 和 chunk/source binding；
- decoder feature、candidate pruning、mask 和 cache key；
- `SlidingMomentEstimator` 的 current z/global current-matrix 特征；
- long-horizon metadata 中的 `latent_regime`、`shock_flags`、future hotspot 等；
- proxy/scenario 被错误当作可执行真实 demand；
- evaluator/oracle 通过共享对象反向泄漏；
- future reveal schedule 或 RNG state；
- unknown 与真实数值 0 混淆。

`PROBLEM_DEFINITION.md` 不只列出风险，还给出能力隔离、stage 顺序、scenario/execution 分域和 13 项强制测试。设计层审计合格，但实现层结论仍为 **未验证**；只有 Phase 1 实现和反事实测试通过后，才能声称 ground truth 与 observation 严格分离。

## 10. 统计风险

1. 历史正式 V1 的部分 CI 是跨训练 seed 的正态近似，不满足未来 Gate 的 sequence-level 统一规范；该历史证据可以支持负面冻结，不能直接作为 H1/H2 的统计模板。
2. raw step、method、observation condition 和模型 seed 的重复不能增加独立 traffic sequence 数；R-011 已明确登记。
3. Phase 1 任务要求保存 sequence/family/seed/topology/reveal/method 和原始 paired rows；H1 以后才允许做 sequence-cluster bootstrap、paired delta、95% CI、ACF/ESS 和至少 3 seed 的收益判定。
4. 当前 Phase 0 没有作算法收益主张，因此不存在用单 seed 或最好桶通过 Gate 的情况。
5. 现有 C3 只是历史可预测性的初步证据，不是 H1 Gate；后续必须按完整 sequence 和 held-out family 独立验证。

## 11. Phase 1 拆分审查

拆分与主目标一致，职责基本互斥，且顺序正确：

1. `P1-SEM-001` 先冻结 API、threat model、oracle 和 owner；
2. `P1-TEST-001` 在实现前写失败测试；
3. `P1-UENV-001`/`P1-REVEAL-001` 建立 truth/view 与单调 reveal；
4. `P1-EXEC-001` 建立 reveal-aware validator 和 truth-side transition；
5. `P1-SCEN-001` 只建无模型场景容器，不训练 predictor；
6. `P1-BASE-001` 在统一 truth/reveal/checker 上实现五个 baseline；
7. `P1-METRIC-001` 保留逐 sequence 原始记录；
8. `P1-INTEG-001` 完整回归；
9. `P1-SUP-001` 独立审查并有 veto 权。

未发现任务重叠、把 Phase 2 工作偷渡到 Phase 1、或把 synthesis 优化重新设为主线。实际分配前，Main 仍须维持同一可写文件只有一个 owner。

## 12. 必须返工项与准入条件

### Phase 0 必须返工项

**无。** 当前 Phase 0 审计文档和证据达到本阶段验收标准，不要求返工后重新提交 Supervisor。

### 启动 Phase 1 前的硬条件

| 条件 | Owner | 验收 |
|---|---|---|
| 用户明确批准进入 Phase 1 | 用户 / Main | 当前会话取得明确授权；未授权时所有 P1 状态保持 `planned` |
| Main 在启动前确认文件 owner 与依赖顺序 | Main | 不让多个 Agent 同时写同一文件；不新增 Subagent |
| 先完成 P1 语义冻结和失败测试，再写实现 | Main / Core | `P1-SEM-001`、`P1-TEST-001` 先于 `P1-UENV-001` |
| 不复用旧 partial-demand 作为合格环境 | Core | 新接口满足问题定义；旧 C4 仅作对照资产 |
| Phase 1 Gate 前补足相关 Torch 覆盖或明确保持为未运行风险 | Main / Core / Supervisor | 不把 skip 标通过；任何修改过的 Torch 路径必须有真实测试证据 |
| Phase 1 完成后独立 Supervisor 报告 | Supervisor | 无泄漏、deterministic legality、oracle 隔离全部通过后才允许进入 H1 |

## 13. 最终监督意见

Phase 0 的核心问题不是“旧代码是否已经支持不确定流量”，而是是否准确承认它尚不支持，并冻结一个可验证、不会泄漏的下一阶段问题。当前交付满足这一要求：负面结果没有被改写，旧 partial-demand 的局限和真实旁路没有被隐藏，synthesis 方向冲突已被纠正，测试缺口也被如实保留。

因此，本 Supervisor 对 **Phase 0 交付给出 ALLOW**；对 **实际进入 Phase 1 给出 HOLD，等待用户明确批准**。Main 应提交本轮结果后停止。
