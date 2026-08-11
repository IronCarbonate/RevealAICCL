# H2 Early Planning 正式实验结果（Gate H2）

更新日期：2026-08-04

## 一句话结论

Phase 4 正式实验已**有效完成**（进程 exit 0、八产物原子发布、两阶段 read-back 与独立重算通过），H2 判定为 **FAIL**：conditions 1/3/6 明确不满足，`data_status=FAIL`，Phase 5 Gate 保持关闭。

## 1. 运行元数据

| 项目 | 记录 |
|---|---|
| 服务器 | `region-42.seetacloud.com:21569`（`autodl-container-36da11a152-db2cf032`） |
| GPU | NVIDIA RTX 2080 Ti 11GB（实验不使用 GPU） |
| 资源 | cgroup：40 GiB 内存、12 CPU 核（`cpu.cfs_quota_us=1200000`） |
| 执行配置 | `MULTI_THREADED_DEFAULT`（OMP/MKL/OPENBLAS/NUMEXPR 均未限制） |
| 启动（UTC） | `2026-08-03T14:13:15.750997501Z` |
| 结束（UTC） | `2026-08-03T18:42:08.764052690Z` |
| 实际时长 | 4 小时 28 分 53 秒 |
| 退出码 | `0` |
| 外层 timeout | 144000 s（未触发） |
| destination | `outputs/phase4_early_planning`（原子发布成功） |
| staging | 0 |
| 八产物 | 8/8 |
| 准入记录 | `P4-SUP-REMOTE-FORMAL-ADMISSION-003`（ALLOW） |

环境与代码 hash：Python `0c05a22b...`、pip-freeze `6f27b26b...`、`phase4_experiment.py` `696E75BD...`、协议 `4246D661...`，均与冻结值一致；172 文件清单 hash 校验通过；多线程预运行测试 `494 passed, 4 skipped, 18 warnings`。

## 2. 完整性验证

- 正式八产物：`manifest.json`、`h1_best_point_model.json`、`raw_validation_metrics.csv`、`raw_test_episode_metrics.csv`、`raw_test_sequence_metrics.csv`、`raw_test_execution_events.csv`、`raw_timing_metrics.csv`、`summary.json`。
- exact 行数（实际 = manifest）：validation 9,600；test episode 2,700；test sequence 135；execution events 147,690；timing 21,600。
- manifest：`integrity_complete=True`、`evidence_complete=True`、`data_status=FAIL`、`gate_status=PENDING_SUPERVISOR`。
- `summary_sha256`（manifest 内）= `308b7730c4fbbd6fa823dc08f293bd3c71ee4fe0a2f3e8fcde35d5393e125961`，与实际 summary 一致。
- 独立 `read_back_artifacts(require_final=True)` 通过：artifact 名称宇宙、manifest 精确 schema、formal sequence universe/排除集、logical/scientific hash、summary hash、H1 artifact schema/state digest、CSV 精确 schema、row digest，以及 raw→episode→sequence→conditions→summary 重算链全部一致（任一不一致会抛异常）。
- `combined_scientific_evidence_sha256 = c56168c73cdd77d61c240f69a540d78a45b8cf818ba6d3cec7173095edbd62df`。

## 3. H2 条件判定（conditions 1–8）

| 条件 | 判定 | 证据 |
|---|---|---|
| 1. robust 相对 Wait 与 Partial 的 15-sequence paired 平均 E2E 改善 > 0 且 bootstrap 95% CI lower > 0 | **FAIL** | E2E Δ（robust−comparator）：Partial −938.58，CI [−992.59, −896.65]；Wait −926.66，CI [−976.98, −887.96]；CI 全部 < 0 |
| 2. 相对两者 mean sequence CVaR95 均不高，p95/p99 完整报告 | PASS | CVaR95 Δ：Partial −1568.97、Wait −1554.77（robust 更优） |
| 3. 相对 validation 冻结 primary comparator：3/3 seed 正、≥4/5 family 正，负 family 相对退化 ≤10% | **FAIL** | seed Δ 全负（−996.19 / −887.98 / −931.56，0/3）；family Δ 5/5 全负，相对退化 8.48–9.48% |
| 4. ordinary 与 executable 方法 legality 精确 100%，action 全部走原 checker | PASS | 9 个方法 legality 均为 1.0 |
| 5. robust 的 discrete/wall timeout 率不高于 Wait 与 Partial | PASS | 全部 timeout 率 0.0 |
| 6. 1ms/slot 主 E2E 已含 online overhead，收益不能只来自 scheduling-only | **FAIL** | E2E Δ −938.58（overhead_included=True）；scheduling-only Δ 仅 +0.11，收益不存在 |
| 7. exact 15/15 sequence、完整 2,700 test rows 有效 | PASS | 15/15 sequence、2,700 episode、ESS=15 |
| 8. fresh exclusion、ordinary/oracle 能力、exact artifact/schema/digest、raw→sequence→Gate、focused/full tests 全部通过 | PASS | 均通过 |

`failed_conditions = [1, 3, 6]`，`insufficient_conditions = []`。任一明确 FAIL → data Gate FAIL（不判 HOLD）。

## 4. 核心证据

方法均值（15 条 test sequence）：

| 方法 | completion_mean | end_to_end_mean (ms) |
|---|---:|---:|
| scenario_robust_prefix | 20.49 | 1042.46 |
| wait_until_known | 25.88 | 115.80 |
| partial_current_only | 20.61 | 103.88 |

robust prefix 的 completion 略优于 Wait（20.49 vs 25.88）、与 Partial 相当（20.49 vs 20.61），CVaR95 也更好；但其端到端耗时约 1042 ms，是被动基线（约 104–116 ms）的约 9 倍——**在线 ambiguity 构造 / prefix synthesis / support selection 的 overhead 完全吞掉了 completion 收益**。`end_to_end_delta=−938.58`（overhead 已计入），`scheduling_only_delta=0.1133`。

## 5. FAIL 归因（按协议 A–G）

主要归因 **C（replan/在线 overhead 过高）**，伴随 **D/G（当前 reveal 与 ambiguity 设置下提前信息没有带来可执行收益）**：

- completion 层面存在质量收益（优于 Wait，尾部 CVaR95 优于两者），legality 100%、timeout 0——不是 legality/timeout 失败；
- 但 E2E 中 online overhead 约 940 ms，约为基线全程耗时的 9 倍，收益被完全抵消（条件 1、6）；
- seed/family 方向性全负（条件 3），说明收益不稳健且被 overhead 主导。

按协议，只有"质量收益存在但 C 抵消"时才可另行申请 synthesis 优化。当前 completion/尾部风险收益存在但被 overhead 抵消，因此**可以**作为后续研究路径向用户申请优化在线 synthesis/replan 开销；但 Gate H2 本身判定为 FAIL。

## 6. 结论

- Phase 4 正式实验有效完成：`FORMAL SUCCESS`（执行层面），`H2 FAIL`（科学判定层面）。
- 不能声称 robust prefix 优于 Wait/Partial；也不能声称其无效——只能说在当前冻结配置（H/P/risk lambda、reveal 机制、1ms/slot、NumPy proxy）下，其 E2E 表现因在线 overhead 而显著劣于被动基线。
- Phase 5 Gate 保持 **CLOSED**；不得进入 rolling configuration / Pareto frontier。
- 后续动作（按协议）：仅当用户授权时，可针对 synthesis/replan overhead 优化单独立项，重新冻结后再评估。
