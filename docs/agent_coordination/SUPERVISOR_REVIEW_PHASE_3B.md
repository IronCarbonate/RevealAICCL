# Supervisor Review — Phase 3B Prediction-Free Empirical Ambiguity

日期：2026-07-31  
角色：Supervisor / Project Director  
审查性质：正式实验完成后的独立最终监督审查（`P3B-SUP-001`）

## 1. 最终裁决

- **Phase 3B 最终判定：PASS**。
- **条件 7：PASS**。Supervisor 对正式十文件、schema-v2 final read-back、raw→derived→Gate→summary 重算、外部 provenance、测试和边界完成独立复核，结论为 **NO VETO**。
- **监督处置：ALLOW Main 以 Phase 3B PASS 关闭本 Gate、向用户报告，并仅建议用户另行评审是否授权 Phase 4。**
- **Phase 4 执行继续 HOLD**。本报告不是 Phase 4 授权，不允许实现或运行 robust prefix、recourse、rolling planning 或 H2 实验。
- `outputs/phase3b_ambiguity/summary.json` 按冻结协议保持 `gate_status="PENDING_SUPERVISOR"`；Supervisor 条件 7 只记录在本报告中，没有修改正式 artifact。

本次 PASS 的窄结论是：在冻结 fresh corpus、观察语义和指标下，存在可审计的 prediction-free traffic ambiguity support，且预注册的 coverage、selected-vs-random、LOFO、tail、width 和完整性条件全部成立。它**不证明 AICCL 提前规划或在线补救具有调度收益**。

## 2. 审查目标与范围

本阶段目标是构造有限、prediction-free、与 partial observation 一致的经验 ambiguity set，并判断其 traffic-space support 是否满足进入 Phase 4 评审前的预注册条件。Supervisor 最终审查范围包括：

- `docs/uncertainty_aiccl/PHASE3B_AMBIGUITY_PROTOCOL.md`；
- `docs/uncertainty_aiccl/PHASE3B_AMBIGUITY_RESULTS.md`；
- `outputs/phase3b_ambiguity/` 下十个正式 artifact；
- `rlccl/uncertainty/ambiguity.py`、`ambiguity_experiment.py`；
- `scripts/run_phase3b_ambiguity.py`；
- 两份 Phase 3B tests；
- H1 exclusion、fresh sequence/config、Rear4GPU topology、normalizer/group coefficient provenance；
- 既往 formal timeout、converter failure、RED/修复与重新准入记录。

未创建额外 Subagent，未修改正式 artifact、生产源码、测试、协议、Main 结果报告或 Main-owned 账本，未再次运行 formal，未进入 Phase 4。本报告是本次审查唯一写入文件。

## 3. 冻结输入与正式 artifact

### 3.1 协议、测试、源码与结果报告

| 文件 | SHA-256 |
|---|---|
| `PHASE3B_AMBIGUITY_PROTOCOL.md` | `7E01108E362973461B5E676CF163A491D7E90E5D30D40AE0356CA83D6680D7A3` |
| `test_phase3b_ambiguity.py` | `3C5BE400A6D6EFAFEB6143CBDE6C3D514E10477A2AE8A4E4A7BAD9B11D756F35` |
| `test_phase3b_experiment.py` | `CB9A1F736A1E17B56D72375DDBE5A1686652D30F462A9EEB129854408B0DFE3C` |
| `ambiguity.py` | `56A08BA4E78DED671E0092E2F7350A17BB2E59ECC2447D0BF75DF5DA565F2EB4` |
| `ambiguity_experiment.py` | `F850E4902C62C2EACF23BE052D514DCB41362C1E4495B5E9736B1F21D6B273C5` |
| `run_phase3b_ambiguity.py` | `A765DDFD120751ABE70F81A14856F60C12DD232CC574FD5EF9D328B2E0279B72` |
| `PHASE3B_AMBIGUITY_RESULTS.md` | `7CFC13B9AE7311456857F377FCFE9F8B8A79044C71C86AB0BDAFDAA15D9FA51C` |

### 3.2 正式十文件

| artifact | raw 行数 | 文件 SHA-256 |
|---|---:|---|
| `manifest.json` | — | `DF8218052A635A683CE0CA848BB31171C740A4FC9C8E31DDB764BB60F2DEE527` |
| `raw_calibration_scores.csv` | 4,800 | `B926CBE96E715FB0DF575FF893B547CB47B59A1CD495D7ECC7A6BEE9CFA75898` |
| `raw_validation_metrics.csv` | 19,200 | `DEC3CFEEE477E3456908FB6EE2865500D099956D70C23BF922AF1F145B947EDA` |
| `raw_case_metrics.csv` | 120,000 | `CA4DCC8CB0F0266BA15E31B381F22919CCAE1601015EEDE7CA3DBD086F47F663` |
| `raw_sequence_metrics.csv` | 300 | `B900F57FE8467BD264B263B2A22784A4BA3204D2CFBC8BD8B084E9C3524D8432` |
| `raw_lofo_calibration_scores.csv` | 19,200 | `EB32E644AC7BCBC32919EB15740BF614D10CC95A4E33E3856C6FBC9ED57DC01A` |
| `raw_lofo_validation_metrics.csv` | 76,800 | `DC9B6C1210B08B6B8495541FB2F3F99897D3CB81AFEF953EDF55477F92F92E01` |
| `raw_lofo_test_metrics.csv` | 9,600 | `4BC9079DA95F0F4F2EF3945F418C2101F9AA5AF7F9A2AB63D1AFDC57A86F49DA` |
| `raw_dependence_metrics.csv` | 240 | `6A6CBFCDC3AF0F0E51229CB6CBBC86CD24678904B2567C50A67FC0707ECA1330` |
| `summary.json` | — | `0628310C6A061B9C1609B24E43713D2EB66AE4EBF043625CD7CC81126F4BDBCA` |

八张 raw 表合计 `250,140` 行。combined scientific evidence SHA-256 为 `5729B95FA929A7E66F1EF3363EE602252480EEBE4025DDC5487E3F85CB209941`。正式目录精确包含上述十文件，发布后 staging 为 0，目标 Python 进程为 0。

## 4. 已完成事项

1. 逐文件复核正式 artifact、协议、测试、源码、runner 和结果报告 SHA-256，全部与冻结清单一致。
2. 独立调用 production `read_back_artifacts(..., integrity_expected=True, allow_incomplete_universe=False)`；133.75 秒完成并通过：schema version 2、integrity `true/true`、manifest/summary `data_status=PASS`、无 failed/insufficient condition、`gate_status=PENDING_SUPERVISOR`。
3. final read-back 重新生成并核对 fresh sequence、完整 generator config、H1 exclusion、Rear4GPU topology、global/LOFO normalizer、group coefficient 和授权源码 provenance；精确校验八表 schema、typed lexical forms、identity universe、logical/scientific/combined digest。
4. 不依赖 manifest/summary 派生数字，用独立 stdlib CSV + NumPy reference script 从 raw 重算 calibration radius、validation sequence-equal selector、15 sequence paired delta、3 seed mean、5 family mean、10,000 次 family-stratified bootstrap、LOFO、tail、width、invalid/empty、ratio1、timing 与 dependence。结果逐项匹配。
5. 独立核对 formal converter 修复后的持久化 K 语义：96,000 个 unknown rows 中 `actual_k != requested_k` 为 0；K1/4/8/16 各 24,000 行；24,000 个 ratio1 rows 全部 `actual_k=1`，requested K 各 6,000 行。
6. 独立运行 Phase 3B focused 和全仓测试；无失败或新增 skip/warning。
7. 静态 AST 检查 Phase 3B 生产模块无 `uncertainty.execution`、decoder、Torch 或 scheduling 禁止导入；Phase 4 边界保持关闭。
8. 复核既往失败历史：第一次为 artifact 物化前的外部 30 分钟超时；第二次由 provisional read-back 捕获 converter 擦除 requested K，未发布正式目录。缺口经过真实 40-row RED、严格单行修复、Core/Main/Supervisor green 与重新准入闭环。

## 5. 实际命令与结果

```text
Get-FileHash -Algorithm SHA256 <protocol/tests/source/runner/results/artifacts>
```

结果：全部冻结 SHA 匹配；正式 artifact 数为 10；staging 数为 0；目标 Python 进程数为 0。

```text
F:\AnaConda\python.exe -B -
# read_back_artifacts('outputs/phase3b_ambiguity',
#   integrity_expected=True, allow_incomplete_universe=False)
```

结果：PASS，`133.746s`；schema2、true/true、data PASS、无 failed/insufficient 条件、combined scientific digest 匹配。

```text
F:\AnaConda\python.exe -B -
# 独立 stdlib CSV + NumPy raw reference recomputation
```

结果：calibration、validation、bootstrap、LOFO、tail、width、dependence 和 K 语义均与正式 summary/manifest 精确一致。

```text
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider \
  tests\test_phase3b_ambiguity.py tests\test_phase3b_experiment.py
```

结果：`166 passed in 100.39s`。

```text
F:\AnaConda\python.exe -B -m pytest -q -p no:cacheprovider
```

结果：`369 passed, 4 skipped, 18 warnings in 126.75s`。4 个 skip 均为环境缺少 Torch 的既有 legacy tests；18 个 warning 均为既有 H1 toy sklearn batch-size/convergence warning。

```text
F:\AnaConda\python.exe -B -
# AST parse + forbidden-import scan
```

结果：3 个生产/runner 文件 parse 通过，forbidden import hits 为 0。

## 6. 实验有效性审查

### 6.1 数据、划分与选择

- fresh corpus 与 H1 corpus 通过 sequence digest exclusion 隔离；fit/validation/calibration/test 均按完整 sequence 划分，每个 split 为 5 families × 3 base seeds = 15 条 sequence。
- fit 只用于 global/fold-specific normalizer 和 tail threshold；validation 只在普通方法、K=8、unknown ratios 上选 selector；calibration 只确定 radius；test 未参与选择或 calibration。
- validation sequence-equal mean 选择 `boundary_scenarios`、K=8：`0.0824600166`，优于 minimax `0.0920618447`、random `0.1299967945` 和 worst recent `0.1989968706`。
- calibration 两层 `higher` radius 为 `0.3432791972`。
- 五折 LOFO 在 fit/validation/calibration 中排除 held-out family，再只在该 family 的三条 test sequence 上评估；五折均选择 boundary scenarios。

### 6.2 Observation、oracle 与泄漏边界

- ordinary constructor 接收去除 family/sequence ID、truth、future reveal 和执行能力的窄 construction view；family/seed 仅用于 provenance 和事后分组。
- 五 reveal mode × 五 stage 与 Phase 1 `DemandRevealProcess` 的正式 observation parity 测试通过；同一 construction coordinate 的 observation/ambiguity digest 跨 method/K 一致。
- ordinary supports 对未揭示 truth、metadata、future reveal、family/ID 和 oracle 调用反事实保持不变；oracle support 通过 `uses_oracle/upper_bound_only` 独立标识，不参与 validation、普通 Gate 或 Phase 4 输入。
- 未发现 current/future truth 泄漏、held-out-family fit 污染、跨 split sequence/digest 交叉或 oracle 污染。

### 6.3 统计有效性

- primary effect 使用 15 条完整 test sequence 的 paired delta；没有把 4,800 cases、scenario 或 random replicate 当成独立样本。
- bootstrap 在每个 family 的三条 test sequence 内有放回抽样，固定 10,000 次、seed `20260731`。95% CI 为 `[0.0291910460, 0.0388309674]`，下界严格大于 0。
- 3/3 base seeds 和 5/5 families 的 paired mean 均为正；没有选择性汇报最好 seed/family。
- total/group tail events 分别为 140/3,280，均超过预注册最小 10 event；condition 4 不是样本不足造成的 HOLD。
- 15 条 test sequence 的 checkpoint-total mean lag-1 ACF 为 `0.0509651117`，positive-sequence ESS 合计 `184.4480235`，依赖性被报告而未用于虚增独立样本数。

## 7. 风险审查

### 7.1 代码与 artifact 风险

- schema-v2 final read-back、exact universes、全派生链、canonical digests、外部 provenance 和 final atomic publication全部通过；未发现可使正式证据无效的剩余缺陷。
- 第二次失败暴露的 converter coverage gap 是真实风险，但 validator 正确 fail closed，未创建正式 evidence。新增 regression 覆盖真实 converter 的 K1/4/8/16、unknown/ratio1 和五方法；最终正式 raw 中 K 语义也已逐行核对。
- 第一次 timeout 和第二次 provisional failure 都发生在正式发布前；生成器与科学配置确定、未因观察结果改变 selector/K/radius/threshold。不存在以多次科学结果择优发布的证据。
- 正式运行耗时 `3474.8s`，说明 pipeline 成本较高；这是后续工程可观测性/性能风险，不改变本次 traffic-space Gate，也不授权 synthesis 优化。
- raw timing 与 environment 按协议排除于 scientific determinism digest；完整 logical/file digests仍被保存和核对。

### 7.2 数据泄漏风险

- fresh/H1 exclusion、whole-sequence split、narrow view、ordinary counterfactual、LOFO exclusion 与 oracle isolation 均通过。
- family 与 base seed 存在于 raw provenance 是审计所需；测试确认其不进入 ordinary constructor。
- 未发现足以提出 veto 的 future information、test selection 或 held-out family leakage。

### 7.3 统计与外推风险

- 每 family 只有三条独立 test sequence，family/LOFO 估计仍可能有较大方差；本次条件使用冻结阈值、完整 family/seed 分解和 sequence-level bootstrap，没有超出证据作更强推断。
- support nearest-distance、coverage 与 tail recall 是 traffic ambiguity 指标，不是 completion、regret、legality 或 end-to-end AICCL 指标。
- Phase 3B PASS 不能外推为 robust prefix/recourse 有收益；Phase 4 必须在用户另行授权后独立验证 H2。

## 8. 条件 1–7 独立裁定

| 条件 | Supervisor 裁定 | 独立证据 |
|---:|---|---|
| 1 | **PASS** | selected K8 unknown joint coverage overall `0.9397916667` ≥ 0.85；family 最小值 `0.925` ≥ 0.80。 |
| 2 | **PASS** | paired 95% CI `[0.0291910460, 0.0388309674]`；3/3 seed 为正，5/5 family 为正。 |
| 3 | **PASS** | LOFO aggregate delta `0.0455307513` ≥ 0；5/5 held-out family delta 为正；5/5 relative degradation 均为负。 |
| 4 | **PASS** | total `140/140=1.0`；group `3250/3280=0.9908536585`；hotspot `4642/4800=0.9670833333`；event 数充足。 |
| 5 | **PASS** | width `0.1502813718` ≤ 0.75；四普通方法 invalid/empty 均 0；ratio1 coverage 1；timing finite/nonnegative。 |
| 6 | **PASS** | schema-v2 final read-back、250,140 raw rows、digests/provenance、raw→summary、10k bootstrap、LOFO/dependence、atomic/oracle/reveal/Phase4 boundaries及 focused/full 全部通过。 |
| 7 | **PASS** | Supervisor 独立复核无 blocker，正式裁定 **NO VETO**。 |

条件 1–7 全部通过，无 insufficient 条件，故最终机械结论为 **Phase 3B PASS**，不是 HOLD 或 FAIL。

## 9. 未完成事项、必须返工项与下一步边界

未完成但不属于 Phase 3B Gate 的事项：

- 未实现或评估 robust schedule prefix、reveal 后 recourse/replan、H2 completion/regret/legality/overhead；
- 未证明 traffic ambiguity support 能转化为 AICCL 调度收益；
- 未获用户授权进入 Phase 4；
- 未处理与本 Gate 无关的四个 Torch legacy skips 或 H1 toy sklearn warnings。

**Phase 3B 必须返工项：无。**

允许的下一步只有：Main 向用户报告 Phase 3B PASS、完整风险和窄结论，并询问用户是否另行授权 Phase 4。用户明确批准前，Phase 4、H2、robust prefix/recourse 实现与实验全部保持 HOLD。

## 10. Supervisor 签署式结论

**Phase 3B：PASS。**  
**Condition 7：PASS。**  
**Supervisor：ALLOW Phase 3B 关闭并向用户报告；NO VETO。**  
**Phase 4：HOLD，必须等待用户单独授权。**

本报告仅支持可审计 prediction-free traffic ambiguity support 的结论，不作 AICCL scheduling-gain 声明。
