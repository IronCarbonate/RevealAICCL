# Supervisor Review — Phase 1 Gate

日期：2026-07-30（Asia/Shanghai）  
任务：`P1-SUP-001`  
角色：Supervisor / Project Director  
审查性质：独立 Gate 审查；除本报告外未修改任何文件；未创建额外 Subagent

## 1. 最终判定

| 判定对象 | 结论 | 含义 |
|---|---|---|
| Phase 1 Gate | **HOLD** | 当前测试集全部绿色，但一个确定性反例证明 `FullInformationOracle` 不是冻结语义要求的上界，故不能把 Phase 1 标记为 Gate 通过。 |
| Veto | **NO VETO** | 未发现普通 policy 获得私有真值能力、scenario token 被执行、旧 Torch decoder 泄漏路径被接入或需推翻整体 Phase 1 架构的证据；缺陷可局部返工。 |
| 进入 H1/H2/H3 | **NOT ALLOWED** | 在本报告列出的 oracle 与 provenance 返工完成、补充测试转绿并由 Supervisor 重新独立审查前，不得进入 predictor、robust prefix、训练或收益实验。 |

`HOLD` 的原因不是现有 pytest 失败，而是现有测试遗漏了一个会产生负 `oracle_regret` 的关键语义分支。不得通过裁剪负 regret、只改字段名或放宽断言使 Gate 变绿。

## 2. 审查范围

Supervisor 完整复核了以下输入：

- `docs/uncertainty_aiccl/PROBLEM_DEFINITION.md`，包括 Phase 1 语义冻结 §11；
- `docs/agent_coordination/TASK_LEDGER.md`、`DECISION_LOG.md`、`RISK_REGISTER.md`；
- `rlccl/uncertainty/__init__.py`；
- `rlccl/uncertainty/observation.py`；
- `rlccl/uncertainty/problem.py`；
- `rlccl/uncertainty/reveal.py`；
- `rlccl/uncertainty/scenarios.py`；
- `rlccl/uncertainty/execution.py`；
- `rlccl/uncertainty/baselines.py`；
- `rlccl/uncertainty/metrics.py`；
- `rlccl/uncertainty/evaluation.py`；
- `tests/test_uncertainty_environment.py` 全部 1334 行。

当前目录仍不是可验证的 Git worktree，无法以 `git diff` 或 commit 证明完整改动边界。因此本审查使用账本的唯一文件 owner、逐文件阅读、静态搜索、独立测试和额外性质反例作为证据；不声称工作区干净。

## 3. 独立测试结果

### 3.1 Phase 1 专项测试

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider tests/test_uncertainty_environment.py
```

实际结果：

```text
58 passed in 1.88s
SUP_FOCUSED_EXIT=0
SUP_FOCUSED_ELAPSED_SECONDS=3.046
```

### 3.2 完整可运行回归

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider
```

实际结果：

```text
131 passed, 4 skipped in 16.23s
SUP_FULL_EXIT=0
SUP_FULL_ELAPSED_SECONDS=17.344
```

四个 skip 均因当前环境缺少 Torch，位置为：

- `tests/test_moment_policy_shapes.py:5`；
- `tests/test_optimizer_checkpoint.py:6`；
- `tests/test_sequence_evaluator.py:5`；
- `tests/test_partial_demand.py:72`。

它们保持 **SKIPPED / NOT RUN**，不能改写为通过。Phase 1 新路径的 fresh-process 测试确认未导入 `rlccl.models`、`rlccl.training`、旧 decoder 或 `torch`，所以该环境缺口不是本次 `HOLD` 的直接原因，但 R-003 仍需保留。

## 4. Phase 1 强制 Gate 逐项核对

| Gate 条件 | Supervisor 证据 | 判定 |
|---|---|---|
| truth 与 observation 对象、数组、能力分离 | truth/mapping/possession 只在 `UncertainProblemInstance` 私有侧；policy payload 深复制、只读；反射测试覆盖禁止能力名 | PASS |
| 隐藏 truth 反事实不改变普通方法输入/候选/动作 | ratio 0 与超过 512 token 的反事实测试通过；新 NumPy 路径不调用旧 pruning decoder | PASS |
| mask/token 单调且最终全揭示 | 五种 mode 的参数化测试覆盖 entry-level 与 token-level 两套冻结语义 | PASS |
| 五种 mode、五个默认 ratio | `random_entries`、两种 totals-first、`partial_shards`、`time_based_arrival` 与 `(0,.25,.5,.75,1)` 均有确定性覆盖 | PASS |
| 未揭示 demand 与 scenario/proxy 不可执行 | `TruthTokenId`/`ScenarioTokenId` 类型与命名空间分离；commit 拒绝 scenario 与未揭示 token | PASS |
| ratio 0、空动作、未知/真实零分离 | 无 private `C`、无 truth padding；aggregate 不生成 proxy truth token | PASS |
| 历史只读同一 sequence 的 `X_<t` | history shape/step/sequence 校验通过；mean/previous 确实产生不同 history-derived scenario-only plan | PASS |
| paired truth/reveal/topology/seed/evaluator/time limit | 每方法重建独立 world/process；metadata 深复制；共同 manifest 字段写入 raw row | PASS，但 digest 完整性有返工项 |
| full oracle 仅作为上界 | 定向反例产生 `oracle_regret=-1` | **FAIL / HOLD** |
| 实际 schedule legality | 现有覆盖的正常行 legality 为真；非法提交路径被记录而非硬编码；capacity/shared-group/source/duplicate/stale 均检查 | PASS（仅限已覆盖范围） |
| 强制超过旧 decoder pruning 阈值 | >512 hidden-truth counterfactual 通过，且 fresh process 无旧 decoder/Torch import | PASS |
| latent metadata、future reveal、cache key 不可达 | observation/history/scenario 深复制与反射测试未发现回引用；generator metadata 仅 evaluator/world 私有 | PASS |
| 负数、非二值、NaN/Inf、未知 token 被拒绝 | legacy numeric validator 与 typed commit 路径覆盖动作域、edge、容量和 shared group | PASS |

## 5. Gate blocker：full-information “oracle” 不是上界

### 5.1 确定性最小反例

Supervisor 构造了以下不含 Torch 的合法 Phase 1 case：

- topology：6 节点有向链 `0 -> 1 -> 2 -> 3 -> 4 -> 5`；
- 每条 edge capacity 为 1，无 shared-group；
- truth：`X[0,1]=6`、`X[0,5]=1`，其余为 0；
- reveal mode：`time_based_arrival`；
- ratios：`(0.0, 0.25, 0.5, 0.75, 1.0)`；
- reveal seed：2；
- `time_limit=timeout=30`。

实际 raw result：

| method | completion | oracle_regret | reveal_wait |
|---|---:|---:|---:|
| `full_information_oracle` | 11 | 0 | 0 |
| `wait_until_known` | 15 | 4 | 4 |
| `partial_current_only` | 10 | **-1** | 1 |
| `long_term_mean` | 10 | **-1** | 1 |
| `previous_value` | 10 | **-1** | 1 |

命令退出为非零是诊断脚本在发现首个负 regret 后主动退出；被测 runner 本身成功返回上述行。

### 5.2 根因

`run_oracle()` 并未求解 full-information 最优调度，而是对 full observation 重复调用 `_direct_revealed_proposal()`。该函数按 reveal process 给出的 token 顺序做单 slot 贪心。对 `time_based_arrival`，部分 observation 是对私有 entry order 的过滤而非严格前缀；隐藏的短流在 full view 中可能先占用首边，而 partial method 可以先推进长流并在后续 slot 形成 pipeline。因此“信息更多”并不保证这条固定 greedy 规则的 completion 更小。

这不是统计波动：同一 manifest、truth 和 seed 可稳定复现。Supervisor 还先运行了 600 个随机 case × 5 modes（3000 个 paired case、12000 个普通 baseline row），未发现负 regret；随后定向构造立即找到反例。这说明随机 smoke 无反例不能替代上界证明。

`upper_bound_only=True` 是标签，不构成语义保证；当前 `oracle_regret` 也不能被解释为非负 regret。

## 6. Paired provenance 缺口

`PairedEvaluationRunner.__init__()` 校验了 `truth_digest`，但没有计算或校验 `topology_digest` 与 `config_digest`。Supervisor 使用合法 truth/topology 和字符串 `bogus-topology`、`bogus-config` 构造 manifest，runner 接受后将两者原样写入 raw rows：

```text
{'constructed': True,
 'row_topology_digest': 'bogus-topology',
 'row_config_digest': 'bogus-config'}
```

这不会把真值泄漏给 policy，也不会导致方法间 state 污染，但会让不可变 paired manifest 的 provenance 可以与实际 topology/config 不一致。它是 Gate 返工项：应由 evaluator 以冻结的 canonical serialization 构造摘要，或在 runner 入口验证摘要；不能只验证字段存在。

`manifest.timeout` 当前作为共同 `timeout_limit` provenance 记录，而离散执行边界由 D-022 冻结的 `time_limit` 控制。结合 D-023，本审查不把两字段未共用一个执行循环判为独立 blocker；但返工时应在 API 文档中继续明确二者含义，避免后续把未执行的 wall-clock timeout 冒充已经强制执行。

## 7. 泄漏、执行与范围审查结论

1. 未发现普通 baseline 参数、对象属性或 public payload 可到达 world、truth matrix、private token count、future mask/RNG、manifest、evaluator 或 generator latent metadata。
2. `partial_shards` 的 public observation 只有已揭示 token/计数下界；ratio 0 不暴露 private `C` 或 full token mask。
3. `source_totals_first` 与 `source_destination_totals_first` 的 aggregate 不创建可执行 proxy token。
4. truth token 与 scenario token 在类型、字符串命名空间和 commit 权限上分离；scenario-only proposal 不能执行。
5. commit 在原子应用前验证 stage/state version、token reveal、edge、source possession、destination possession、edge capacity 和 shared-group；同 slot 不前递。
6. ordinary planner 签名符合冻结接口；oracle 没有普通 `propose` API，也没有 hidden oracle flag。
7. 每个 method 从 manifest/private truth 新建 world 与 reveal process；嵌套 metadata 已深复制。
8. 静态搜索未发现 `rlccl/uncertainty/**` 导入旧 `SlotDecoder`、`rlccl.models`、`rlccl.training`、Torch、predictor 或 robust 实现。代码中的 `synthesis_time_ms` 是冻结 metrics 字段，不是 synthesis 优化路线。
9. 本轮没有 H1/H2/H3、predictor、GRU/TCN、robust prefix、模型训练或收益主张。

## 8. 测试修改与断言语义审查

账本 D-020 所述 line 625 fixture 修订仅把 `wrong_source` 的候选从“不是 token source”收紧为“既不是 token source 也不是 token destination”，从而能在无自环的 complete topology 中构造真正的 wrong-source edge。异常断言、topology、commit 调用和 source-possession 语义均未放宽；未发现为使实现转绿而弱化该测试。

现有 58 cases 对泄漏、reveal、validator、paired isolation、history、capacity/shared-group、multi-hop、真实 metrics 和 provenance 字段覆盖较强。缺口在于没有测试 full-info comparator 的支配性，也没有拒绝伪造 topology/config digest；绿色结果不足以覆盖这两个语义约束。

## 9. 必须返工项

| ID | Owner | 必须完成 | 验收证据 |
|---|---|---|---|
| P1-RETURN-ORACLE-001 | Main 冻结语义；Core 测试/实现 | 把 full-information comparator 实现为对受支持实例确实成立的上界参考。优先采用可验证的 full-info 最优/有证明的调度器；若要降级为普通 greedy reference 并把 regret 改为 signed delta，必须先修改冻结语义并取得相应批准，不能静默改名。 | chain6/seed2 反例不再出现负 regret；给出算法语义与正确性依据。 |
| P1-RETURN-TEST-001 | Core | 先加入上述反例的 red test，再实现修复；对 default baseline raw rows 明确断言普通方法的 `oracle_regret >= 0`。 | 保存 red-before-green 原始结果；不得通过 `max(0, delta)`、绝对值或删行修复。 |
| P1-RETURN-MANIFEST-001 | Main 定义 canonical 口径；Core 实现 | canonical 计算/校验 topology 与 config digest，至少让实际 topology/config 与 manifest 不一致时确定性拒绝。 | bogus digest probe 先红后绿；共同 raw provenance 仍保持一致。 |
| P1-RETURN-INTEG-001 | Main | 更新 TASK_LEDGER/DECISION_LOG/RISK_REGISTER，登记本次反例与返工；重新运行 focused/full。 | focused 全绿；full 无新失败；4 个 Torch skip 继续如实列出。 |
| P1-RETURN-SUP-001 | Supervisor | 对返工 diff、red/green 证据和完整回归做独立复核。 | 新的 Supervisor 判定为 ALLOW 前，Phase 1 保持 HOLD。 |

## 10. 非阻塞但必须保留的限制

- 当前仍无有效 Git baseline，不能声称 clean worktree 或精确 diff。
- 四项 legacy Torch 测试仍未运行；新 NumPy uncertainty 路径的隔离已验证，但旧路径风险不能因此改写为消失。
- 旧 decoder line-425 truth leak 仍存在于冻结历史路径；Phase 1 新路径已隔离，但不得重新接入。
- 当前 raw rows 和 smoke 只验证环境/基线语义，不构成 AICCL 性能收益、H1 或统计显著性证据。
- completion/regret 的独立统计单位后续必须是完整 sequence；本阶段不得把 method/stage row 数冒充独立样本数。

## 11. 十项交接

1. **当前状态**：`P1-SUP-001 = HOLD`；`NO VETO`；不得进入 H1/H2/H3。
2. **已完成审查**：问题定义、三份协调文档、9 个 uncertainty 文件和 1334 行专项测试均已完整阅读。
3. **已完成测试**：Supervisor 独立得到 focused `58 passed`，full `131 passed, 4 skipped`。
4. **新增失败证据**：chain6、`time_based_arrival`、seed 2 稳定产生 `oracle_regret=-1`。
5. **首要根因**：所谓 oracle 是 fixed-order greedy，不是 full-information 最优/有证明的上界。
6. **次要完整性缺口**：topology/config digest 可伪造并被原样写入 raw row。
7. **未发现事项**：未发现 policy truth capability、scenario execution、metadata/future reveal 反向引用或旧 Torch decoder 接入新路径。
8. **本 Agent 改动**：仅新增 `docs/agent_coordination/SUPERVISOR_REVIEW_PHASE_1.md`；未修改业务代码、测试或 Main/Core-owned 文档。
9. **下一责任链**：Main 先登记/冻结返工口径；Core 按 red-before-green 修复；Main 集成；Supervisor 复审。不得新增 Subagent。
10. **停止边界**：本报告提交后 Supervisor 停止；未经用户/Main 按既定链路重新派发，不继续修改、训练或启动后续阶段。

## 12. 最终监督意见

Phase 1 的能力隔离、两级 reveal、typed token、reveal-aware commit、独立 paired episode 和真实轨迹 metrics 已形成一个实质性且大体可信的基础；本审查没有发现需要否决该架构的数据泄漏证据。但是，冻结标准明确要求 full oracle 只作为上界，而当前实现存在普通方法确定性优于 oracle 的反例。该问题会直接破坏 `oracle_regret` 的解释，并可能在后续实验中把 greedy 顺序效应误报为不确定性代价。

因此最终判定为 **HOLD / NO VETO**。完成第 9 节返工并通过新的独立 Gate 前，Main 不得把 Phase 1 标记为通过，也不得开始 H1/H2/H3。

---

## 13. Return Re-review — P1-RETURN-SUP-001

复审日期：2026-07-30（Asia/Shanghai）  
复审输入：`P1-RETURN-SEM/TEST/ORACLE/MANIFEST/INTEG-001` 全部交付  
复审性质：保留第 1–12 节原始 `HOLD`、反例和证据；本节记录返工后的新状态与最终 Gate 判定

### 13.1 Return 最终判定

| 判定对象 | Return 结论 | 含义 |
|---|---|---|
| Phase 1 Return Gate | **ALLOW** | 原 `HOLD` 的两个确定性 blocker 已按新冻结语义闭环：greedy comparator 已替换为有证明的 non-executable full-information completion lower bound；三种 canonical digest 均由 factory 构造并由 runner 复核。 |
| Veto | **NO VETO** | 未发现数据泄漏、普通轨迹改写、regret clamp、伪造 legality、digest 绕过或需推翻 Phase 1 架构的证据。 |
| Phase 1 状态 | **MAY MARK COMPLETE** | Main 可以把 Phase 1 Return Gate 标记为完成并向用户提交审核。 |
| H1/H2/H3 | **NOT AUTHORIZED BY THIS ALLOW** | 本判定只关闭 Phase 1 Gate，不构成启动 predictor、训练、robust prefix 或后续收益实验的授权；本轮仍应停止并等待用户。 |

本节的 `ALLOW` 是对返工后当前代码的最新判定；第 1–12 节原 `HOLD` 继续作为审计历史保留，不得删除或改写。

### 13.2 新冻结语义与协调记录

Supervisor 重新阅读并确认：

- `PROBLEM_DEFINITION.md` §11.8 把 `FullInformationOracle` 明确定义为 evaluator-private、不可执行的 full-information completion lower-bound reference，而非一条 greedy schedule；
- §11.8 明确禁止 `max(0, delta)`、绝对值、删行或修改普通 completion 来制造非负 regret；
- §11.9 冻结 truth/topology/config 三种 canonical digest，要求 factory 从实际输入构造并由 runner 重新计算；
- §11.9 与 D-022/D-023 一致地区分离散 `time_limit`、per-method `timeout: bool` 和只作外层预算 provenance 的 `timeout_limit`，并明确当前纯 NumPy runner 没有声称执行 wall-clock interrupt；
- D-025、D-026、D-027 分别记录 lower-bound reference、canonical digest 和两轮 red-before-green/timing 返工；
- R-017、R-018 保留原 proven 风险与反例，并处于 `mitigated-pending-gate`，没有掩盖原失败。

历史 red 证据在账本中记录为：

```text
red-1: 2 failed, 59 passed in 2.27s
  - chain6 greedy oracle actual 11 != required lower bound 7
  - EvaluationManifest factory missing

red-2: 1 failed, 60 passed in 2.16s
  - oracle synthesis_time_ms remained hard-coded zero
```

由于当前目录没有有效 Git worktree，Supervisor 无法把代码回退到历史 red 版本独立重现这两个旧输出；本次复审核对了测试断言、最终实现、账本原始记录和当前 green 行为，四者因果链一致。

### 13.3 Lower-bound 数学审查

最终 `_oracle_completion_lower_bound()` 使用：

```text
LB = max(LB_path, LB_work, LB_source, LB_dest)
```

逐项正确性如下：

1. **Unit capacity floor**：每个 `TransferAction` 消耗一个原子 unit，commit 以 `edge_load += 1` 检查浮点 capacity。因此每 slot 可执行的 unit 数至多为 `floor(capacity_e)`；容量小于 1 的 edge 不能承载 unit action。
2. **Directed unit-capable shortest path**：Floyd–Warshall 只把 `floor(capacity_e)>0` 的有向 edge 设为一步。commit 禁止同一 token 在同 slot 重复动作，且状态只在全 slot 校验后原子更新，因此一个 token 至少需要 `shortest_hops(s,d)` 个 slot，`LB_path` 成立。
3. **Network work bound**：任意合法路径至少使用 shortest-hop 数量的 unit transmissions；每 slot 全网 edge unit capacity 不超过 `sum_e floor(capacity_e)`，所以 `LB_work` 成立。
4. **Source cut bound**：每个 `(s,d)` atomic token 至少从 source 发出一次，每 slot 首次发出的数量不超过 source 的 unit-capable outgoing capacity，故 `LB_source` 成立。
5. **Destination cut bound**：每个 token 至少进入最终 destination 一次，每 slot 最终进入数量不超过 destination 的 unit-capable incoming capacity，故 `LB_dest` 成立。
6. **Shared groups**：实现故意不把 shared-group 限制增加为可用容量；忽略这些额外约束只会放宽可行域、使 lower bound 更乐观，不会产生超过合法 completion 的假下界。
7. **边界映射**：empty demand 返回 0；正 demand 在 unit-capable 有向图不可达、相关 unit capacity 为 0，或 `LB>T` 时返回 D-022 的 `T+1`。这些情形下任何合法普通方法也不可能在 `T` 内完成。

因此对当前 atomic unit-action 执行模型，任意合法普通 completion 都满足：

```text
ordinary_completion >= oracle_completion_lower_bound
oracle_regret = ordinary_completion - oracle_completion_lower_bound >= 0
```

实现仍使用直接 subtraction；未发现 clamp、绝对值或普通结果改写。这个 reference 不是一条可执行最优 schedule，可能比真正 optimum 更乐观/更松，但这不破坏其作为性能上界参考的数学有效性。

### 13.4 Oracle 实现与 raw row 审查

Supervisor 完整复核最终 `evaluation.py` 与 `baselines.py`：

- `FullInformationOracle` 仅保留 evaluator marker，没有普通 `propose()`；
- `run_oracle()` 不再调用 `_direct_revealed_proposal()`、`Proposal` 或 `world.commit()`，只在 evaluator 私有 truth/topology 上计算 lower bound；
- oracle 不生成 action，不改变普通方法 world、reveal 或 RNG；
- raw row 明示：
  - `reference_kind="provable_full_information_lower_bound"`；
  - `executable=False`；
  - `upper_bound_only=True`；
  - `legality_basis="vacuous_no_executable_actions"`；
- `legality=True` 的含义被明确限定为“没有可执行动作，故无非法 commit”，没有冒充实际 schedule legality；
- `synthesis_time_ms` 在 `perf_counter()` 包围 lower-bound 计算后取得，`replan_time_ms=0`；不再硬编码 synthesis 为 0；
- `_run_ordinary()` 的 proposal、commit、completion、timeout、recourse 和 raw metrics 路径未因 oracle 返工而改写。

`_Episode.next_full_observation()` 仍作为 evaluator-private helper 存在，但返工后的 `run_oracle()` 不调用它；普通 planner 无法获得该对象或 manifest。

### 13.5 Canonical digest 审查

`EvaluationManifest.create(...)` 与 runner 入口满足 §11.9：

- truth：验证后转为 contiguous `int64` bytes，再做 SHA-256；
- topology canonical JSON 包含版本、`V/E`、有序 edge list、每条 capacity 的 `float.hex()`、有序 shared-group edge indices 与 limit `float.hex()`；
- config canonical JSON 包含版本、reveal mode、ratio 的 float hex、seed、`timeout`、`time_limit` 和 checker version；
- JSON 使用稳定 key 排序与紧凑 separators；
- runner 对 truth、topology、config 三个摘要逐一重算，不匹配均在构建 episode 或调用 planner 前确定性拒绝；
- manifest 和摘要仍只在 evaluator/runner 一侧，未进入普通 planner 签名或 policy payload。

当前 canonical representation 保留输入 edge/shared-group 顺序；这会让语义等价但表示顺序不同的 topology 得到不同 digest，属于严格 provenance 身份而非错误等价合并，不破坏 paired 一致性。

### 13.6 三个新增测试复核

新增测试没有放宽原断言，分别覆盖：

1. `test_oracle_is_provable_nonexecutable_lower_bound_on_chain6_counterexample`：冻结原反例的 lower bound 7、普通 completion 10、非负直接 subtraction、non-executable/vacuous legality 和实际 synthesis timing；
2. `test_oracle_lower_bound_empty_unreachable_zero_unit_capacity_and_over_t`：覆盖 empty 0、不可达 `T+1`、subunit edge 的 zero-unit capacity 和 `LB>T`；
3. `test_manifest_factory_and_runner_reject_topology_and_config_mismatch`：覆盖 factory 重复性、实际 topology mismatch 和 manifest config mutation 的确定性拒绝。

这些测试正对原 `HOLD` 两项 blocker；没有删除 chain6、把 regret 取绝对值、修改 ordinary completion 或只检查字段存在。

### 13.7 Supervisor 独立命令与原始输出

#### Focused

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider tests/test_uncertainty_environment.py
```

```text
61 passed in 2.02s
SUP_RETURN_FOCUSED_EXIT=0
SUP_RETURN_FOCUSED_ELAPSED_SECONDS=3.295
```

#### Full regression

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider
```

```text
134 passed, 4 skipped in 15.80s
SUP_RETURN_FULL_EXIT=0
SUP_RETURN_FULL_ELAPSED_SECONDS=16.943
```

四个 skip 仍是相同的缺 Torch 项：

```text
tests/test_moment_policy_shapes.py:5
tests/test_optimizer_checkpoint.py:6
tests/test_sequence_evaluator.py:5
tests/test_partial_demand.py:72
```

它们保持 **SKIPPED / NOT RUN**，没有计入通过。

#### Return targeted tests

```powershell
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider tests/test_uncertainty_environment.py -k "oracle_is_provable_nonexecutable_lower_bound_on_chain6_counterexample or oracle_lower_bound_empty_unreachable_zero_unit_capacity_and_over_t or manifest_factory_and_runner_reject_topology_and_config_mismatch"
```

```text
3 passed, 58 deselected in 0.58s
SUP_RETURN_TARGETED_PYTEST_EXIT=0
```

#### Chain6 / seed2 定向 raw probe

Supervisor 使用原 topology、truth、mode、ratios、seed 和 `T=30` 独立重建 manifest/runner：

| method | completion | oracle_regret | legality | timeout |
|---|---:|---:|---|---|
| `full_information_oracle` | 7 | 0 | true（vacuous） | false |
| `wait_until_known` | 15 | 8 | true | false |
| `partial_current_only` | 10 | 3 | true | false |
| `long_term_mean` | 10 | 3 | true | false |
| `previous_value` | 10 | 3 | true | false |

oracle 额外 raw：

```text
synthesis_time_ms=0.2158999996026978
replan_time_ms=0.0
reference_kind=provable_full_information_lower_bound
executable=False
legality_basis=vacuous_no_executable_actions
```

普通 completion 与原 `HOLD` 反例一致为 10；变化仅是 comparator 从错误 greedy completion 11 改为有证明 lower bound 7。

#### Digest mismatch 定向 probe

同一实际 truth/topology 下，Supervisor 分别注入伪 topology digest、伪 config digest 和“改变 seed 但保留旧 digest”的 config：

```text
REJECTED bogus_topology_digest ValueError Manifest topology digest does not match input topology
REJECTED bogus_config_digest ValueError Manifest config digest does not match evaluation config
REJECTED mutated_config_same_digest ValueError Manifest config digest does not match evaluation config
```

#### 独立随机性质诊断

Supervisor 另运行 120 seeds × 5 modes，使用 4-node bidirectional ring、随机额外有向 edge、capacity 1/2、随机 3-edge shared group 和随机 `0..2` traffic：

```text
{'seeds': 120,
 'modes': 5,
 'paired_cases': 600,
 'ordinary_rows': 2400,
 'negative_regret_rows': 0}
```

该 smoke 不是数学证明；非负性准入依据是 §13.3 的必要条件证明，随机诊断只用于发现实现偏差。

本次复核文件 SHA-256：

```text
evaluation.py                    F0A2EE238F4529812773E1592C09915635A9DBE9A249C3C910B38B139657C98A
baselines.py                     ABBD7FFB524BA28ACD2931383D5FB4FE5FBE0CF702D7383BDCDAF7C322C5E63E
test_uncertainty_environment.py  BFE6A0F7EF6F97D9D4D264ED23586D721E4B6145D409DF0823A555FBFF98E962
```

### 13.8 返工逐项判定

| Return task | 复审结果 | 证据 |
|---|---|---|
| `P1-RETURN-SEM-001` | **PASS** | §11.8/11.9 与 D-025/D-026 冻结 lower-bound、digest 和 timeout 语义；禁止 clamp。 |
| `P1-RETURN-TEST-001` | **PASS** | 三个新增 test 精确覆盖 chain6、边界 lower bound、timing 和 digest mismatch；历史两轮 red 已记录。 |
| `P1-RETURN-ORACLE-001` | **PASS** | 四项 bound 均有必要条件证明；oracle 不 proposal/commit；chain7/ordinary10；raw 明示 non-executable。 |
| `P1-RETURN-MANIFEST-001` | **PASS** | factory canonical 构造三摘要；runner 重算；三种独立 mismatch probe 均拒绝。 |
| `P1-RETURN-INTEG-001` | **PASS** | Main 记录的 green 与 Supervisor 独立 `61/134+4skip` 一致；无新回归。 |
| `P1-RETURN-SUP-001` | **PASS / ALLOW** | 原两个 blocker 均闭环，无新的确定性 blocker。 |

### 13.9 剩余限制与 Gate 后口径

以下事项不阻止 Phase 1 完成，但必须继续公开：

- 当前仍无有效 Git worktree；不能声称 clean baseline、commit 或精确 diff。
- 四项 legacy Torch 测试仍未运行；新 NumPy uncertainty 路径已隔离旧 decoder/Torch，但不能把旧路径写成已覆盖。
- lower-bound reference 是不可执行、可能较松的性能上界，不是最优 schedule 的可行 witness；`oracle_regret` 是到该 lower bound 的 gap，可能同时包含信息、策略和 relaxation 松弛，后续不得把它单独解释为纯 uncertainty cost。
- shared-group 被忽略是有意 relaxation；它保持 bound 正确，但可能增大 gap。
- `timeout_limit` 只记录共同外层调用预算 provenance；当前 runner 实际强制的是离散 `time_limit`，没有 wall-clock interrupt。
- 旧 decoder 的 truth-pruning 风险仍存在于冻结旧路径；不得在后续重新接入而绕过新 capability boundary。
- 本阶段没有训练或收益实验；`134 passed` 证明的是当前可执行语义/回归，不证明 AICCL 性能提升或 H1。

### 13.10 Return 十项交接

1. **最新 Gate**：Phase 1 Return 为 `ALLOW / NO VETO`；原 `HOLD` 作为历史保留。
2. **语义闭环**：oracle 已冻结并实现为 evaluator-private、non-executable、provable completion lower bound。
3. **数学结论**：path/work/source/destination 四项均为 unit-action 合法 schedule 必要条件；shared-group 忽略只放宽。
4. **反例复跑**：chain6/seed2 得到 oracle 7、ordinary 10/15，所有普通 regret 非负且 ordinary completion 未变。
5. **provenance 闭环**：truth/topology/config canonical digest 均由 factory 构造、runner 复算；bogus/mutated probe 全拒绝。
6. **独立测试**：focused `61 passed`；full `134 passed, 4 skipped`；targeted `3 passed`。
7. **性质诊断**：独立 600 paired/2400 ordinary rows 中负 regret 为 0；证明仍以数学审查而非随机 smoke 为准。
8. **剩余限制**：无 Git baseline、4 个 Torch skip、lower bound 非可执行 optimum、无 wall-clock timeout、无性能收益结论。
9. **本 Agent 改动**：仅在同一 `SUPERVISOR_REVIEW_PHASE_1.md` 追加本 Return Re-review；未修改业务代码、测试或 Main/Core 文档，未创建 Subagent。
10. **停止边界**：Main 可标记 Phase 1 完成并提交用户；所有 Agent 随后停止等待用户，不能把本 `ALLOW` 自动扩展为 H1/H2/H3 授权。

### 13.11 Return 最终监督意见

原 `HOLD` 正确识别了“绿色测试不能证明 greedy oracle 上界”以及 manifest provenance 可伪造。返工没有隐藏或删除这两项失败，而是先冻结更精确的 reference 语义，再以 red test 驱动实现：oracle 现在是对 atomic unit-action 模型成立的 completion lower bound，raw 明确不可执行；普通 completion 保持不变，regret 直接相减；canonical factory/runner 则关闭了 topology/config digest 的信任缺口。

Supervisor 的独立数学审查、focused/full regression、原反例和 digest probes 均通过，未发现新的确定性 blocker。因此 P1-RETURN-SUP-001 的最终判定为：

```text
ALLOW
NO VETO
PHASE 1 MAY BE MARKED COMPLETE
STOP AND WAIT FOR USER REVIEW
```
