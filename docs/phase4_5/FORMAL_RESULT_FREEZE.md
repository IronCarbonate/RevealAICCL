# Formal Result Freeze：Phase 4 / Gate H2

更新日期：2026-08-04
状态：**FROZEN**（只读，不得覆盖）

## 1. 冻结结论

**H2 = FAIL（正式，非 HOLD）。** Phase 5 Gate = CLOSED。

来源：`outputs/phase4_early_planning/`（正式八产物，exit 0，原子发布），判定依据为 H2 协议 conditions 1–8：`1=F, 2=T, 3=F, 4=T, 5=T, 6=F, 7=T, 8=T`，`failed_conditions=[1,3,6]`。

## 2. 冻结的关键数字（不得改写）

| 指标 | 值 |
|---|---|
| robust − Partial E2E Δ | −938.58，95% CI [−992.59, −896.65] |
| robust − Wait E2E Δ | −926.66，95% CI [−976.98, −887.96] |
| robust E2E | ≈ 1042.46 ms |
| Wait / Partial E2E | ≈ 115.80 / 103.88 ms |
| scheduling-only Δ | +0.1133 |
| completion（robust / Partial / Wait） | 20.49 / 20.61 / 25.88 |
| seed 正向数 | 0/3（Δ：−996.19 / −887.98 / −931.56） |
| family 正向数 | 0/5（Δ 全负，相对退化 8.48–9.48%） |
| legality | 100%（9 方法全 1.0） |
| timeout 率 | 0.0 |
| CVaR95 Δ（robust−comparator） | Partial −1568.97；Wait −1554.77（robust 更优） |
| sequence ESS | 15.0 |

正式归因：主因 **C**（在线 ambiguity 构造 / prefix synthesis / replan overhead 过高），伴随 **D/G**（当前 reveal 设置下提前信息未形成足够可执行收益）。

## 3. 正式 artifact 只读约束

- 正式八产物位于 `outputs/phase4_early_planning/`：`manifest.json`、`h1_best_point_model.json`、`raw_validation_metrics.csv`、`raw_test_episode_metrics.csv`、`raw_test_sequence_metrics.csv`、`raw_test_execution_events.csv`、`raw_timing_metrics.csv`、`summary.json`；
- exact 行数：validation 9,600；episode 2,700；sequence 135；events 147,690；timing 21,600；
- manifest：`integrity_complete=True`、`evidence_complete=True`、`data_status=FAIL`、`gate_status=PENDING_SUPERVISOR`；
- `summary_sha256 = 308b7730c4fbbd6fa823dc08f293bd3c71ee4fe0a2f3e8fcde35d5393e125961`；
- `combined_scientific_evidence_sha256 = c56168c73cdd77d61c240f69a540d78a45b8cf818ba6d3cec7173095edbd62df`；
- 上述文件与目录一律只读；任何分析不得原地覆盖、修改或重排。

## 4. 新分析约束

- 所有新分析、诊断、实验必须使用**新目录**（如 `outputs/phase4_5/`、`docs/phase4_5/`）；
- 新实验不得复用正式 destination 路径；
- 未来任何 H2 重新评估都必须**重新预注册协议**（新协议 hash、新输出目录、新准入），不得在旧协议/旧 artifacts 上直接改判；
- completion-only 或 CVaR-only 优势不得被描述为 H2 成功；
- 理想化估计（“免费 ambiguity / 免费 scoring”等）只能标记为估计上界/理想化下界，不得冒充真实结果。

## 5. 冻结的运行环境与 hash（2026-08-03 正式运行）

| 项目 | 记录 |
|---|---|
| 服务器 | `region-42.seetacloud.com:21569`，hostname `autodl-container-36da11a152-db2cf032` |
| GPU | NVIDIA RTX 2080 Ti 11GB（实验不使用 GPU） |
| 资源 | cgroup 40 GiB 内存、12 CPU 核（`cpu.cfs_quota_us=1200000/100000`） |
| Python | 3.12.3，`sha256=0c05a22b0b180580a76437114a95cf138f67c8f46245acad26017c803b42b8c1` |
| pip-freeze | `sha256=6f27b26b9edf508acebb0c26dd8d2eda3f7e2c85d24cda632101f3cc43bb4271` |
| 执行配置 | `MULTI_THREADED_DEFAULT`（OMP/MKL/OPENBLAS/NUMEXPR 未限制） |
| 启动 / 结束（UTC） | 2026-08-03T14:13:15.750997501Z / 2026-08-03T18:42:08.764052690Z |
| 时长 / 退出码 | 4h28m53s / 0 |
| 准入记录 | `P4-SUP-REMOTE-FORMAL-ADMISSION-003` |

科学代码 `phase4_experiment.py` SHA-256 `696E75BD...54E52A`、协议 `4246D661...89F2068`，与冻结一致。
