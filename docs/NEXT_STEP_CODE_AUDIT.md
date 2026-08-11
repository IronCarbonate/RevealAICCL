# AICCL 下一阶段真实代码审计（第一轮）

审计时间：2026-07-27（Asia/Shanghai）  
审计范围：环境、现有测试、真实代码、现有流量生成器（Phase A）。本轮未实现 Phase B/C/D，未修改 V1 模型，未训练或重新训练任何模型。

## 1. 执行环境与基线

项目实际目录为 `F:\AMR-AICCL\RLCCL-main`。`F:\AMR-AICCL\.git` 是空目录，项目目录及其父目录都不是有效 Git worktree，因此：

- `git status`：失败，`fatal: not a git repository`；
- `git rev-parse --show-toplevel`：失败，`fatal: not a git repository`；
- 当前 commit：无法取得；
- 工作区是否干净：无法由 Git 判定。

不能把“无法判定”写成“工作区干净”。

环境和测试的真实结果：

| 环境 | Python / PyTorch / CUDA | 结果 |
|---|---|---|
| Windows 默认 `C:\Python313\python.exe` | Python 3.13.5；无 pytest；无 torch | 原始 `python -m pytest -q` 失败：`No module named pytest` |
| Windows `F:\AnaConda\python.exe` | Python 3.12.7；NumPy 1.26.4；pytest 7.4.4；无 torch | 修改前 41 passed, 3 skipped；本轮完成后 47 passed, 3 skipped |
| 用户提供的远程服务器 | Python 3.12.3；NumPy 2.3.2；PyTorch 2.8.0+cu128；CUDA 12.8；RTX 4090 | 安装 pytest 7.4.4 后完整测试 57 passed |

远程服务器只用于测试，没有启动训练。测试源码上传到 `/root/autodl-tmp/aiccl_phase_a_test_20260727`，测试结束后已删除；远程安装的 pytest 7.4.4 保留在 Miniconda 环境。

## 2. 已核对的真实文件、类和函数

| 职责 | 真实路径 | 真实符号 |
|---|---|---|
| 现有 moment-bounded 生成器 | `rlccl/traffic/process_generator.py` | `TrafficProcessConfig`、`_family_profile`、`_calibrate_integer_period`、`_phase_for_pair`、`_generate_candidate`、`generate_traffic_sequence` |
| 最终矩阵/序列类型 | `rlccl/traffic/types.py` | `MomentBounds`、`MomentContext`、`TrafficSequence` |
| 矩阵合法性和 scenario 转换 | `rlccl/traffic/matrix_utils.py` | `validate_traffic_matrix`、`traffic_matrix_to_scenario`、`scenario_to_traffic_matrix` |
| moment bound 校验 | `rlccl/traffic/moment_validation.py` | `compute_window_moments`、`relative_l2_error`、`validate_sequence_moment_bounds` |
| history-only 估计器 | `rlccl/traffic/moment_estimator.py` | `SlidingMomentEstimator.get_context`、`update`、`state_dict`、`load_state_dict` |
| 时序运行器 | `rlccl/envs/sequence_env.py` | `TrafficSequenceRunner` |
| sequence 数据构造 | `rlccl/training/sequence_sampler.py` | `SequenceDatasetConfig`、`build_sequence_problems` |
| V1 moment 特征进入 decoder | `rlccl/envs/decoder.py` | `get_moment_node_features`、`get_global_moment_features`、`get_candidate_moment_features`、`SlotDecoder` |
| V1 policy | `rlccl/models/moment_encoder.py`、`rlccl/models/slot_policy.py` | `MomentEncoder`、`SlotLevelPolicy` |
| V1 训练和评估 | `rlccl/training/ppo_trainer.py`、`rlccl/evaluation/sequence_evaluator.py` | `train_epoch`、`evaluate_model`、`evaluate_sequence_policy`、`build_shuffled_context_map` |
| V1 命令脚本 | `scripts/train_moment_policy.py`、`scripts/evaluate_moment_policy.py`、`scripts/run_v1_ablation.py` | `dataset_config`、`make_model`、`provisional_gate`、`formal_gate` 等 |
| 旧的非时序场景生成/缓存 | `rlccl/envs/evaluator.py`、`scripts/train.py`、`scripts/test.py` | `generate_scenarios`、`load_or_generate_scenarios`、`LEGACY_GENERATOR_NAME` |

同时核对了 `CODE_AUDIT.md`、`docs/AMR_AICCL.md`、`docs/PERFORMANCE_V0.md`、`docs/PERFORMANCE_V1.md` 和 `outputs/moment_v1/formal/v1_formal_summary.json`。

## 3. 现有时序生成器的实际语义

### 3.1 配置、窗口和约束

- `sequence_length` 可配置，要求 `sequence_length >= window_size > 1`（`TrafficProcessConfig.__post_init__`）。
- V1 正式实验实际使用 `sequence_length=64`、`window_size=16`、`min_history=8`、`mean_level=2`、`std_level=1`、`max_entry=8`、`epsilon_mean=0.20`、`epsilon_var=0.30`。
- 默认 `validation_stride=1`，因此每个完整重叠窗口都检查；默认 `allowed_violation_fraction=0.0`，任一窗口违规就使整条候选失败。
- 每个窗口对每个矩阵元素计算总体均值和总体方差（`ddof=0`）。误差是
  `||actual-reference||_2 / (||reference||_2 + 1e-8)`；mean 和 variance 分别与 epsilon 比较。
- 验证对象是最终的非负整数、方阵、零对角 traffic matrix，而不是连续原始样本。

### 3.2 clip、round、diagonal 和 rejection 的真实顺序

真实顺序是：

1. 为一个 `window_size` 周期构造并标准化 family profile；
2. 对每种 `(target_mean, target_var)` 扫描 13 个 mean offset × 21 个 scale factor；
3. 每个网格点先 `np.clip(..., 0, max_entry)`，再 `np.rint(...)`，然后以最终整数 period 的 mean/variance 选最优；
4. period 数组先初始化为零，并跳过 `src == dst`，所以对角线是“结构性保持为零”，没有独立的 post-hoc diagonal-zeroing 步骤；
5. 对 off-diagonal pair 做固定或随机相位滚动；
6. 用 `np.tile(period, ...)` 重复到 `sequence_length`；
7. 对最终整数序列执行所有滑窗 moment 验证；
8. 不通过时才进入下一次 rejection attempt。

当前公开 API 只返回最终接受序列，没有保留 pre-clip、post-clip/pre-round、rejected candidate 或 diagonal-zeroing 前数组。因此本轮审计将这些字段明确记为 `null/unavailable`，没有用私有逻辑重建并冒充真实观测。

### 3.3 六个 family 的真实规则

| family | 实际 profile / 相位规则 | 长时行为 | 主要改变 |
|---|---|---|---|
| `smooth_ar` | 正弦基波加 0.3 倍二次谐波；每个 pair 的相位由 RNG 抽取 | 名称含 AR，但代码中没有 AR 状态或创新噪声；16 步周期精确重复 | 总量和空间 pattern 都变 |
| `alternating_burst` | 奇偶 `-1/+1`，叠加四步方波；pair 相位为 `(src+dst+seed) mod W` | 最终矩阵的实测最短精确周期为 2；空间负载完全平衡 | 几乎只改变总量，且只是两点交替 |
| `moving_hotspot` | 周期内固定平滑脉冲；相位只由 destination 的等距偏移和 seed 决定 | hotspot 沿固定轨迹循环，不是 random walk；精确周期 16 | 总量和 destination 空间 pattern 都变 |
| `sparse_switching` | 前 `W/4` 高活动位置加小正弦背景；相位为 `(src*V+dst+seed) mod W` | support/强度按固定表循环；正式参数下最终 off-diagonal sparsity 实测为 0 | 以 source 空间不均衡为主，总量变化很小 |
| `bimodal` | 前半周期 -1，后半 +1；每 pair 随机相位 | 随机性只决定 16 步 period 的排列，此后精确重复 | 总量和空间 pattern 都变 |
| `heavy_tail_clipped` | 一个大 spike、半周期处一个次 spike，其余为零，随后标准化和整数校准 | 没有随机 rare-event 到达过程；固定 16 步重复 | 总量和空间 pattern 都变，但尾部是固定模板 |

## 4. 长序列、复杂度和接受率风险

`_generate_candidate` 先生成形状 `[window_size,V,V]` 的一个 period，然后在 `process_generator.py` 第 214–216 行直接 `np.tile`。所以：

- `sequence_length=1024/4096` 只增加重复次数，不增加潜在动力学长度；
- 任意完整 16 步窗口都是同一 period 的循环排列，moment 误差完全相同；
- 128/512 窗口又是 16 的整数倍，必然复制同一 moments；
- 在当前正式参数下，接受率不会随长度下降：本轮 1080 条全部第一次接受；
- 生成和存储约为 `O(L V²)`；现有验证逐窗口重新 `stack`，约为 `O((L-W+1) W V²)`，固定 W 时随 L 近线性，但 W 同时增大时有二次风险；
- 失败配置存在“重复拒绝但无法修复”的风险。确定性相位 family 每次 attempt 生成同一候选；即使 `smooth_ar/bimodal` 的 RNG 相位改变，完整 period 的逐元素 moments 不因相位改变，通常也不能靠下一次 attempt 修复不可满足的 moment 设置。

## 5. history-only、数据划分、缓存与随机种子

### 5.1 history-only 正确性

`TrafficSequenceRunner` 在 step t 先调用 `estimator.get_context(X_t, ...)`，`yield` 后才 `estimator.update(X_t)`。`SlidingMomentEstimator` 的 mean/variance 只从 deque 中已完成的历史矩阵计算，因此没有把 `X_t` 加入历史 moments。

但是 `MomentContext.current_send_z/current_recv_z` 明确由真实 `X_t` 计算；decoder 的 7 个 node moment features 和 4 个 candidate moment features都包含这些当前负载 z-score。8 个 global features 中还有 4 个直接由真实 `X_t` 得到的 sparsity、source CV、destination CV 和 max-entry 特征。这不构成未来泄漏，因为完整当前 demand 按任务定义可见，但说明 V1 `moment-full` 并非纯历史 moment 方案，且与 baseline 已知的 `X_t` 信息高度重叠。

### 5.2 sequence split 和 seed

- `build_sequence_problems` 先按完整 sequence、时间顺序物化 context；seed 为 `base + family_index*10000 + sequence_index`。
- 训练阶段之后对物化后的 problem/step 做随机排列，但 context 已在完整 sequence 内按历史计算，没有跨 sequence 共享 estimator。
- `train_moment_policy.py` 的 validation base seed 是 training seed `+1_000_000`；最终 held-out evaluation 使用独立 eval seed。正式 V1 的 train families 是四个前置 family，held-out families 是 `bimodal/heavy_tail_clipped`。
- 旧 `scripts/train.py/test.py` 使用另一套 `legacy_mixed_v2` 独立场景生成器：50% random All-to-All-V、30% AllGather、20% AllToAll；train/test seed 相差 1000。缓存带 schema version、generator name、完整 config 和 SHA-256 config hash，不匹配会重建。

没有发现同一 sequence 的不同片段被显式分配到 train/test 两侧。需要注意，PPO 训练以 step/problem 为 shuffle 单位而不是以 sequence 为 batch 单位；这不会造成已物化 context 的未来泄漏，但会削弱训练 batch 的时序结构。

## 6. 与 V1 结果的关系（只诊断，不改模型）

正式 V1 结果仍是 `NO_GO`，overall mean degradation relative 为约 2.11%，legality 为 100%。真实训练数据来自上述合成生成器，不是生产流量 trace；正式长度 64 只包含 4 个 16 步重复周期（`alternating_burst` 实际为 32 个两步周期）。

因此，数据集很可能是效果不佳的重要原因：它缺少随机长 regime、rare shock/recovery、随机 hotspot dwell 和多尺度波动；同时完整 `X_t` 已知，历史 moment features 又混入多项当前 `X_t` 派生量，增量信息有限。仅凭本轮 Phase A 不能把 V1 失败全部归因于数据，二者都是有代码证据的候选原因。

## 7. 本轮新增的独立审计资产

- `rlccl/evaluation/traffic_audit.py`：每条 sequence 的总量、时序、空间、多窗口和生成过程指标；
- `scripts/audit_traffic_generator.py`：所需 CLI、分 seed 执行、JSON/CSV 输出和可选 CPU 多进程；
- `tests/test_traffic_audit.py`：6 个独立测试；
- `rlccl/evaluation/__init__.py`：仅将 torch-backed sequence evaluator 改为惰性导入，使 CPU-only Phase A 审计不因缺少 torch 而无法 import；没有修改模型或评估语义。

下一步是否进入 Phase B/C/D 必须由后续轮次决定；本文件不包含任何相应实现。
