# Phase 3B Prediction-Free Empirical Ambiguity Results

日期：2026-07-31  
协议：`PHASE3B_AMBIGUITY_PROTOCOL.md`，SHA-256 `7E01108E362973461B5E676CF163A491D7E90E5D30D40AE0356CA83D6680D7A3`

## 1. 结论边界

冻结正式运行已成功完成，schema-v2 十个 artifact 通过 final read-back。Main 独立重算判定预注册数据条件 1--6 全部通过；`summary.json` 仍按协议保持 `gate_status=PENDING_SUPERVISOR`。条件 7 与最终 Phase 3B PASS/FAIL 必须由 Supervisor 独立裁决。

本结果只支持“存在可审计、prediction-free 的 traffic ambiguity support”这一窄结论，不证明 robust prefix、recourse 或 AICCL 调度收益。Phase 4 未获授权且未执行。

## 2. 执行与完整性

成功命令：

```text
F:\AnaConda\python.exe -B scripts\run_phase3b_ambiguity.py --formal --output-dir outputs\phase3b_ambiguity
```

- 成功运行：exit 0，`3474.8s`；
- 正式目录：`outputs/phase3b_ambiguity`；精确 10 个文件；
- final manifest：`schema_version=2`，`integrity_checks_complete=true`，`integrity_checks_passed=true`；
- summary：`data_status=PASS`，`gate_status=PENDING_SUPERVISOR`；
- 发布后 `.phase3b-staging-*` 为 0，目标 Python 进程为 0；
- Main 独立 `read_back_artifacts(..., integrity_expected=True, allow_incomplete_universe=False)`：PASS，`137.2s`；
- combined scientific evidence SHA-256：`5729b95fa929a7e66f1ef3363ee602252480eebe4025ddc5487e3f85cb209941`。

成功前两次未发布尝试如实保留：第一次在 artifact 物化前被外部 30 分钟工具窗口终止；第二次 provisional read-back 检出 formal converter 擦除 `requested_k`，72,000 个 unknown rows 被 fail closed 拒绝。该缺口经真实 converter 40-row RED、严格单行修复、Core/Main/Supervisor 独立回归与新准入闭环；失败 staging 完整 forensic 后按授权精确删除。两次均未创建正式 destination，也未产生科学结论。

## 3. Artifact 清单

| 文件 | 行数（CSV） | 文件 SHA-256 |
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

八张 raw 表合计 250,140 行，行数和 identity universe 均由 final read-back 精确验证。

## 4. Validation、calibration 与 dependence

- calibration radius：`0.34327919716983946`；
- validation sequence-equal nearest RMS mean：
  - `boundary_scenarios`: `0.08246001662905347`；
  - `minimax_subset`: `0.09206184465236308`；
  - `random_empirical`: `0.12999679448257667`；
  - `worst_recent_cases`: `0.1989968706011134`；
- 冻结选择：`boundary_scenarios`，`K=8`；
- test total-traffic aggregate mean lag-1 ACF：`0.050965111693624075`；
- mean positive-sequence ESS：`12.29653490102115`；15 条 test sequence 的 positive-sequence ESS 合计 `184.44802351531726`。

## 5. 预注册条件 1--6 的 Main 重算

| 条件 | 证据 | Main 判定 |
|---:|---|---|
| 1 | unknown-case joint coverage overall `0.9397916666666667`；五 family 分别 `0.940625 / 0.925 / 0.9375 / 0.95 / 0.9458333333333333`，均高于 `0.80` | PASS |
| 2 | selected-vs-random paired delta 95% CI `[0.029191045984302454, 0.03883096742658576]`；base seed 342/442/542 均正；五 family 均正 | PASS |
| 3 | LOFO aggregate delta `0.04553075131309523`；五个 held-out family delta 均正；五个 relative degradation 均为负，即无 family 相对 random 恶化超过 10% | PASS |
| 4 | total-tail `140/140 = 1.0`；group-tail `3250/3280 = 0.9908536585`；hotspot `4642/4800 = 0.9670833333`；event 数充足 | PASS |
| 5 | physical-normalized mean width `0.15028137184651943`；四 ordinary invalid/empty rate 全为 `0`；ratio1 singleton coverage `1.0`；全部 timing finite/nonnegative | PASS |
| 6 | 十文件 exact schema/row universe、typed digests、raw→sequence→Gate→summary、10,000 bootstrap、LOFO/dependence、外部 provenance、双 read-back、单次 rename、oracle/observation/Phase4 隔离均通过；Main post-formal focused/full green | PASS |

family 顺序按协议为 `regime_switching_long / stochastic_volatility / rare_shock_recovery / hotspot_random_walk / same_moments_different_dynamics`。

### Paired delta 分解

- base seed：342=`0.030034647050192383`，442=`0.04325206697453289`，542=`0.028860629218377322`；
- family：
  - regime switching long=`0.03083428856338988`；
  - stochastic volatility=`0.05757234201966057`；
  - rare shock recovery=`0.01833594826279027`；
  - hotspot random walk=`0.034863497322627765`；
  - same moments different dynamics=`0.02863949590336916`。

### LOFO

| held-out family | selected method | calibration radius | family delta | relative degradation |
|---|---|---:|---:|---:|
| regime switching long | boundary scenarios | `0.3471738280539375` | `0.031564233872813` | `-0.26661133024503786` |
| stochastic volatility | boundary scenarios | `0.3527324308478382` | `0.11428533741984029` | `-0.42490540376610864` |
| rare shock recovery | boundary scenarios | `0.3332118725355287` | `0.01810686374490761` | `-0.21722031623091753` |
| hotspot random walk | boundary scenarios | `0.36956786834406047` | `0.037008377237594925` | `-0.29067688100942507` |
| same moments different dynamics | boundary scenarios | `0.31513761943421986` | `0.02668894429032033` | `-0.2942337315158733` |

## 6. Post-formal 测试

- Phase3B focused：`166 passed in 100.48s`；
- full repository：`369 passed, 4 skipped, 18 warnings in 210.49s`；
- 4 skips 均为环境缺少 Torch 的既有 legacy tests；
- 18 warnings 均为既有 H1 toy sklearn batch-size/convergence warnings；
- 未出现新增 skip、warning 或失败。

## 7. 待 Supervisor 裁决

Main 的数据与实现审查结论为：条件 1--6 PASS、无 insufficient 条件。最终条件 7、Phase 3B PASS/FAIL/HOLD，以及是否仅向用户建议另行评审 Phase 4，均由 Supervisor 独立审查决定。在其裁决前不得改写 `summary.json` 的 `PENDING_SUPERVISOR`，也不得进入 Phase 4。
