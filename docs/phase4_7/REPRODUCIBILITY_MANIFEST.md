# 可复现清单（REPRODUCIBILITY_MANIFEST）

更新日期：2026-08-04

## 1. 环境（冻结）

| 项 | 值 |
|---|---|
| Python | 3.12.3（SHA-256 `0c05a22b0b180580a76437114a95cf138f67c8f46245acad26017c803b42b8c1`） |
| pip-freeze | SHA-256 `6f27b26b9edf508acebb0c26dd8d2eda3f7e2c85d24cda632101f3cc43bb4271`（numpy 1.26.4/scipy 1.13.1/sklearn 1.5.1/joblib 1.4.2/threadpoolctl 3.5.0/pytest 7.4.4） |
| 主机 | autodl-container-36da11a152-db2cf032（RTX 2080 Ti；40GB/12 核 cgroup） |
| 执行配置 | MULTI_THREADED_DEFAULT（OMP/MKL/OPENBLAS/NUMEXPR 未限制），PYTHONHASHSEED=0 |

## 2. 冻结代码

- `rlccl/scheduling/phase4_experiment.py` SHA-256 `696E75BD...54E52A`
- 协议 `H2_EARLY_PLANNING_PROTOCOL.md` SHA-256 `4246D661...89F2068`
- 正式运行产物：`outputs/phase4_early_planning/`（8 文件，summary_sha `308b7730...`）

## 3. 各实验 corpus 与种子

| 实验 | base seeds | universe digest | 零重合 |
|---|---|---|---|
| H2 正式 | 642/742/842 | 正式 manifest | — |
| Route A | 1042/1142/1242 | 见 `outputs/phase4_6/route_a_reveal/route_a_results.json` | 与 H2 ✓ |
| H5/H6/H7 | 2042/2142/2242 | `3d69637aba5eadd7e575902bde0bad17c3a02ab26f500e20dcd189de91678451` | 与 H2、Route A ✓ |

## 4. 脚本与产物路径

| 环节 | 脚本 | 产物 |
|---|---|---|
| Route A | `outputs/phase4_6/route_a_reveal/route_a_runner.py` | `route_a_results.json`（SHA-256 `567cb657...`） |
| H5 | `outputs/phase4_7/h5_realizable_reveal/{corpus_h5,cost_calibration,h5_runner}.py` | `corpus_h5_manifest.json`、`cost_params.json`、`h5_test.json` |
| H6 | `outputs/phase4_7/h6_selective_reveal/h6_runner.py` | `h6_test.json` |
| H7 | `outputs/phase4_7/h7_adaptive_reveal/h7_runner.py` | `h7_test.json` |

## 5. 成本参数（calibrated + assumed）

| 参数 | 值 | 类型 |
|---|---:|---|
| histogram 更新 | 336 ns/token | measured |
| matrix 构造 | 613 ns/entry | measured |
| 控制消息 RTT | 8.8 µs | measured（localhost） |
| allreduce 延迟系数 | 10 µs × log2(P) | assumed |
| 带宽 | 10 GB/s | assumed |
| 阻塞/pipeline 系数 | 1.0 / 0.1 | assumed |
| P | 4 | 固定 |

## 6. 统计约定

- sequence-level paired bootstrap 10,000（seed 20260801），95% CI；
- 结论桶 ≥5 条独立 sequence；不把行级当独立样本；
- 跨 run 的 scheduler wall 中位数不跨 run 比较；结论基于 run 内配对。

## 7. 复现命令

```bash
cd /root/autodl-tmp/RLCCL-main
export PYTHONPATH=/root/autodl-tmp/RLCCL-main PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C
P=/root/autodl-tmp/phase4-env/bin/python
$P -B outputs/phase4_6/route_a_reveal/route_a_runner.py gate && $P -B outputs/phase4_6/route_a_reveal/route_a_runner.py main
$P -B outputs/phase4_7/h5_realizable_reveal/cost_calibration.py
$P -B outputs/phase4_7/h5_realizable_reveal/corpus_h5.py
$P -B outputs/phase4_7/h5_realizable_reveal/h5_runner.py test
$P -B outputs/phase4_7/h6_selective_reveal/h6_runner.py
$P -B outputs/phase4_7/h7_adaptive_reveal/h7_runner.py
```
