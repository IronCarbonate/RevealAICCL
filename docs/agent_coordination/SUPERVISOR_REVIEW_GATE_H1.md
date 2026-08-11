# Supervisor Review — Phase 2 / Gate H1

日期：2026-07-31  
角色：Supervisor / Project Director  
审查性质：正式实验完成后的独立最终监督审查

## 1. 最终判定

- **H1 最终判定：FAIL**。
- **条件 7：PASS**。H1 focused tests、完整性测试和全仓回归均满足本 Gate 的完整性要求，Supervisor 不提出 veto。
- **监督处置：ALLOW Phase 2 / Gate H1 以 FAIL 结论关闭并由 Main 向用户报告；HOLD 任何 Phase 3 实现，等待用户明确授权；NO VETO。**
- 路由建议：选择 **Phase 3B — prediction-free robust route**，不选择 Phase 3A predictive scenario route。本建议不是实施授权。
- `outputs/h1_predictability/summary.json` 保持 `gate_status="PENDING_SUPERVISOR"`，本审查没有修改该文件；最终监督结论仅记录在本报告中。

H1 FAIL 是对预注册假设的有效否定结果，不是证据缺失造成的 HOLD。近期历史 MLP 在 pooled test 和 LOFO 中均未提供相对 previous-value 的稳定增量信息。

## 2. 审查阶段与范围

本次审查位于 **Phase 2 / Gate H1 最终独立监督审查**。审查严格限于：

- `docs/uncertainty_aiccl/H1_PREDICTABILITY_PROTOCOL.md`；
- `docs/uncertainty_aiccl/H1_PREDICTABILITY_RESULTS.md`；
- `outputs/h1_predictability/manifest.json`；
- `outputs/h1_predictability/raw_sequence_metrics.csv`；
- `outputs/h1_predictability/raw_probability_metrics.csv`；
- `outputs/h1_predictability/summary.json`；
- `rlccl/prediction/**`、`scripts/run_h1_predictability.py`；
- `tests/test_h1_predictability.py`、`tests/test_h1_experiment_pipeline.py`；
- focused H1 tests、全仓 tests、manifest/raw 独立复算和序列 digest 重生成核对。

未创建新 Subagent，未修改业务代码、测试、正式输出或 Main-owned 文件，未进入或实现 Phase 3。

## 3. 已完成事项

1. 完整阅读协议、正式结果报告、schema-2 manifest、两份 raw CSV、summary、prediction 实现、runner 和两份 H1 测试。
2. 验证四个正式 artifact 的 SHA-256 与 Main 报告完全一致：

   | artifact | SHA-256 |
   |---|---|
   | `manifest.json` | `C702D8CEA33BCEC805FA0AB4B1EEA58C7E0BCBF6AAEF697E01523BB86D65B48C` |
   | `raw_sequence_metrics.csv` | `D03DAC115E2DE839FBEF32326AD90E9E662053E678C5CCEFB1605736A5402517` |
   | `raw_probability_metrics.csv` | `7C0F0C2CB8056BAF32466AB4D519D816E5DBF09FAC85A5C96908329901752829` |
   | `summary.json` | `C48E35230030215148E3DEF46340A991D226B69FD97797EB6D2086BE6A26DFCE` |

3. 验证 manifest：75 条记录、75 个唯一 sequence ID、75 个唯一 digest；split 数为 fit/validation/calibration/test = 30/15/15/15；family、base seed、actual seed、sequence index、variant、length 和完整 generator config 经 JSON 规范化后与 canonical 75 specs 逐条一致。
4. 重新生成全部 75 条正式 traffic sequence 并独立计算 digest：**75/75 匹配，0 mismatch**。协议、source tree、Rear4GPU topology 和 deterministic group coefficients 的摘要也均与 manifest 匹配。
5. 独立读取并检查全部 raw rows：

   - point rows：1590，唯一 identity 1590；
   - probability rows：180，唯一 identity 180；
   - 无 NaN/Inf；
   - 所有正式 test point row 均为 1016 steps；
   - pooled 与五个 LOFO scope、方法、目标、family、seed、previous-value pairing identity 均完整；
   - repository manifest/point/probability validators 全部通过。

6. 不依赖 `summary.json` 的预聚合字段，从 raw CSV 独立重算 validation 选择、三个 primary target 的 10,000 次 family-stratified paired bootstrap、seed/family mean、LOFO、quantile/scenario coverage、tail recall 和条件 1–6。独立结果与落盘 summary 数值精确一致。
7. 验证 pooled validation 仅从 15 条 validation sequence 等权选模：MLP RMSE `1.449827`，TCN RMSE `1.544218`，选择 `recent_history_mlp`；五个 LOFO fold 也均仅由各自 seen-family validation rows 选择 MLP。
8. 独立运行测试：

   - focused：`F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider tests\test_h1_predictability.py tests\test_h1_experiment_pipeline.py` → **69 passed, 18 warnings in 11.84s**；
   - full repository：`F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider` → **203 passed, 4 skipped, 18 warnings in 22.91s**。

## 4. 未完成事项与停止边界

- 未运行 Phase 3A/3B 实验，未实现 robust prefix、rolling recourse 或 prediction-free scheduler。
- 未改写 `summary.json` 的 `PENDING_SUPERVISOR`，也未向正式 artifact 预填最终 PASS/FAIL。
- 未证明 prediction-free robust scheduling 有效；Phase 3B 只是由 H1 失败触发的预注册路线建议。
- 未消除固定 80-iteration MLP 的 convergence warning，因为改变训练轮数或事后调参会违反冻结协议。
- 未处理与 H1 无关且依赖 Torch 的 4 个 legacy skips；它们不能记为通过，但不影响本 Gate 使用的 NumPy TCN 和 H1 focused integrity tests。

## 5. 实验有效性审查

### 5.1 数据、划分与配对

- 正式设计严格使用 5 families × 3 base seeds × 5 complete sequences = 75 sequences；actual seed 公式、same-moments variant 和 split assignment 符合预注册协议。
- fit/validation/calibration/test 均按完整 sequence 划分，没有把相关时间步随机分散到不同 split。
- pooled test 有 15 条独立 sequence；每条 1016 个预测点，共 15,240 raw test steps。
- 所有方法在相同 sequence、topology 和 seed 上配对；主效应为每条 sequence 的 `RMSE(previous_value) - RMSE(method)`，没有选择性汇报最好 seed。
- LOFO 每折从 fit/validation/calibration 排除 held-out family，并只在该 family 的 3 条 test sequence 上评估。

### 5.2 信息边界与数据泄漏

- `build_history_examples` 对时刻 `t` 的 predictor features 只使用 `X_{<t}`；current/future truth 只进入 label/evaluation。
- metadata、latent regime、shock flag、family identity 和 base seed 不进入模型或 calibrator features；family/seed 仅用于事后分组。
- validation 只选 MLP/TCN backbone，calibration 只拟合 signed residual distribution，test 不参与选择或校准。
- history-only current/future/metadata counterfactual、LOFO 排除和 fit-only scaling 测试全部通过。
- **泄漏结论：未发现会改变 H1 判定的未来信息泄漏。**

### 5.3 统计有效性

- primary CI 使用 15 条完整 test sequence 的 paired delta，并在每个 family 的 3 条 sequence 内有放回抽样，固定 10,000 次、seed `20260731`；没有把 15,240 个相关 step 当作独立样本。
- pooled test mean lag-1 ACF 为 `0.871429`，逐 sequence ESS 合计 `1062.092`；高自相关被如实报告，而没有用于虚假扩大独立样本量。
- seed、family、LOFO 和 probability family calibration 均完整报告，负面结果未被 secondary target 或单一 family 的正结果覆盖。
- tail 有 1444 个事件，超过最少 10 个事件的预注册门槛，因此条件 6 可作正式判定而非 HOLD。

## 6. 代码、泄漏与统计风险

### 6.1 代码风险

- focused/full tests 均无失败；真实 NumPy causal TCN 具备 finite-difference gradient、kernel update、合成 causal-lag loss 下降、causal counterfactual、same-seed determinism 和 `.npz` round-trip 覆盖。
- 18 个 warnings 来自 toy pipeline 的小样本 batch clipping 和固定 80 epochs 未收敛；正式结果另报告 3 个固定 80-iteration convergence warnings。warnings 没有被隐藏，也没有触发 test-set 调参。
- 防御性 hardening 仍可加强：`_execute_experiment` 的通用私有入口只核对 sequence ID set，未再次把传入矩阵 digest 与 manifest 绑定；manifest validator 对完整 config 的字段集合和 identity 字段做强校验，但未逐值重建每个冻结默认值。对此正式 run 已通过 75/75 序列重生成、canonical config 和全部 provenance digest 的独立核对，故不影响本次证据有效性，也不构成 veto。
- formal raw validators 的通用规则可进一步强制 `raw_step_count == sequence_length - 8`；本次正式 rows 已全部核实为 1016。

### 6.2 泄漏风险

- generator sequence metadata 包含潜在 latent 信息是结构性风险，但 prediction pipeline 没有把 metadata 传入 feature builder/model/calibrator，且反事实测试覆盖 current/future/metadata 隔离。
- family/seed 只出现在 raw provenance 与 evaluator grouping，不作为 predictor 输入。未发现 test selection、calibration contamination 或 LOFO held-out family 泄漏。

### 6.3 统计风险

- 只有每 family 三条 test sequence，family-level 和 LOFO family-level 估计仍可能较噪；但多个核心结果方向一致且 primary total CI 完全位于 0 以下，不能据此把 FAIL 重解释为不确定 HOLD。
- MLP 未达到 optimizer convergence tolerance，意味着该冻结候选可能尚未达到其最佳性能；协议禁止事后延长训练或调参。当前 Gate 只能评价预注册实现，不能用该风险推翻正式 FAIL。
- probability overall calibration 合格，但两个 family 超阈值，显示 pooled residual calibration 的跨 family 稳定性不足；按预注册规则必须判条件 5 失败。

## 7. 七项预注册条件独立裁定

| 条件 | Supervisor 裁定 | 独立复算证据 |
|---|---|---|
| 1. pooled total paired CI lower > 0 | **FAIL** | mean delta `-0.079003`；95% CI `[-0.113293, -0.047796]`。CI 整体为负。 |
| 2. 3/3 seeds 正且至少 4/5 families 正 | **FAIL** | seeds 42/142/242：`-0.056787/-0.054280/-0.125942`，0/3 为正；仅 stochastic-volatility family 为正，1/5。 |
| 3. 三个 primary target CI lower > 0 | **FAIL** | total lower `-0.113293`；source lower `-0.013416`；destination lower `-0.010208`。三者均未通过。 |
| 4. LOFO 不系统失败 | **FAIL** | aggregate delta `-0.427998`；0/5 family 为正；3/5 family RMSE 恶化超过 10%。 |
| 5. quantile/scenario calibration | **FAIL** | overall interval error `0.018373`、scenario error `0.039304` 合格；但 hotspot-random-walk `0.114042`、stochastic-volatility `0.122835` 超过 family `0.10`。 |
| 6. tail recall | **PASS** | 1444 events，1345 true positives，recall `0.931440` ≥ `0.70`。 |
| 7. integrity tests + no Supervisor veto | **PASS** | focused `69 passed`；full `203 passed, 4 skipped`；H1 所需 history-only、split、determinism、finite、raw/manifest/summary、TCN tests 全通过；**NO VETO**。 |

条件 1–5 明确失败，条件 6–7 通过，因此机械 Gate 结论为 **H1 FAIL**，不是 HOLD。

## 8. 结果解释与路线建议

pooled test 的 previous-value total RMSE 为 `1.567810`，selected MLP 为 `1.646813`；selected MLP 不仅未取得稳定增益，而且 paired CI 显示稳定退化。LOFO aggregate delta 为 `-0.427998`，五个 held-out families 全部为负，说明该 predictive route 对未见动态尤其脆弱。hotspot strength 这一 secondary target 的小幅改善不能覆盖三个 primary targets、seed/family 稳定性和 LOFO 的预注册失败。

因此：

1. 接受 H1 的负面结论并关闭 Phase 2 / Gate H1；
2. 不进入 Phase 3A predictive scenario route；
3. 向用户建议 Phase 3B prediction-free ambiguity set / minimax-regret / robust scheduling route；
4. 在用户明确批准前，不创建新任务、不实现 Phase 3B、不开展额外调参或补充实验。

## 9. Supervisor 签署式结论

**ALLOW**：允许 Main 以本报告为条件 7 的独立监督证据，正式记录并向用户报告 **H1 FAIL**。  
**HOLD**：任何 Phase 3A/3B 代码实现或实验继续保持暂停，等待用户审核与明确授权。  
**NO VETO**：未发现证据造假、未来信息泄漏、split 污染、统计单位错误或足以使本 Gate 无效的实现缺陷。  
**推荐路由**：Phase 3B prediction-free robust route，仅为建议，不是执行许可。
