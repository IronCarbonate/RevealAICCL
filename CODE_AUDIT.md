# AMR-AICCL 首次执行代码审计

审计范围是当前 `RLCCL-main` baseline 及任务文件指定的训练、评估、拓扑、decoder 和 cache 路径。未读取或集成 DeepEP。本目录没有 `.git` 元数据，因此本次无法创建任务说明中的 `v0-baseline-hardening` commit；代码改动保留在工作区。

## 指定核查项

| 核查项 | 修改前的实际情况 | 本次处理 |
|---|---|---|
| `ProblemInstance` / `TopologyInfo` 是否重复 | 是。`rlccl/envs/problem.py` 与 `rlccl/envs/evaluator.py` 各有一套定义，且 evaluator 版本仍构造 dense `D`。 | evaluator 改为导入 `problem.py` 的权威类型；重复定义已移除。 |
| `evaluate_schedule()` 是否依赖 `problem.D` | 是，通过 `np.dot(Y_t, problem.D)` 计算接收状态。 | 改为 `compute_received_chunks()`；优先使用 `topology_info.edge_dst`，缺失时从 `problem.topology[:, 1]` 回退。 |
| optimizer 生命周期 | `train_epoch()` 每个 epoch 内部创建新的 Adam，动量状态每轮丢失。 | optimizer 由 `scripts/train.py` 创建并传入 `train_epoch()`，整个训练过程复用。 |
| All-to-All-V `[0,4]` | 注释称 `[0,4]`，实现是 `rng.integers(1, 5)`，非对角 pair 永远不为 0。 | 改为 `rng.integers(0, 5)`；全零样本确定性修复一个 demand；转换逻辑抽到 `traffic/matrix_utils.py`。 |
| edge capacity 归一化 | `load_topology_info()` 将所有 edge capacity 除以最小正 capacity。 | 原逻辑正确，保留并加入真实 Rear4GPU 数据测试。 |
| shared group limit 归一化 | 与 edge capacity 使用同一个 `min_capacity` 相除。 | 原逻辑正确，保留并加入逐 group 对照测试。 |
| decoder rollout / replay 接口 | `decode_slot()` 每个 slot 保存动态 `node_feats`、`edge_feats`、`chunk_feats`、`demands`、`dist_to_demand`；静态 topology 单独保存。每个 micro-action 保存候选 `(cand_c,cand_e)`、所选 edge、相对 action index 和 step。`recompute_logp_slot()` 按 edge/group usage 重放 5 维 candidate feature。 | 接口已核实；首次执行不修改 policy 或 decoder。V1 必须把 moment 的 node/global/candidate 重放量加入同一 `state_info`，避免 rollout/recompute 不一致。 |
| checkpoint optimizer / scheduler | checkpoint 只保存 model、epoch、best score、config；resume 只恢复 model。代码中没有 scheduler。 | 所有新 checkpoint 保存 optimizer state，resume 在存在该字段时恢复；旧 model-only checkpoint 仍可加载。没有引入未被训练使用的 scheduler。 |
| train/test cache 隔离 | 文件名仅含 split/count/seed；没有 schema、generator 名或 config hash，旧数据会静默加载。 | legacy cache 文件名加入 `legacy_mixed_v2`；payload 加入 schema version 2、generator name、完整 config 和 SHA-256 config hash；不匹配会重建。moment sequence 使用独立数据结构和 schema。 |

## 其他实际发现

- `rlccl/envs/decoder.py` 末尾还有一份未被主脚本调用的 legacy `train()` 实现；它与 `rlccl/training/ppo_trainer.py` 重复，并引用未在该模块导入的 `SlotLevelPolicy`、`optim`、`SlotBuffer`。主训练入口不走这条路径，本次未把它扩展成第二套训练实现。
- `scripts/test.py` 原本在未传 `--export_xml` 时也会强制导入可选 `msccl`，使普通 baseline 测试无法启动；本次已改为只在 XML 分支懒加载。XML helper 仍读取当前 `TopologyInfo` 未提供的 `chunk_nodes/is_switch`，该可选分支未纳入本次验收，后续应单独统一 metadata 接口。
- Windows GBK 控制台无法编码训练/测试脚本中的 `✓/✅` 状态符号，导致数据加载后立即异常；主 `train.py`/`test.py` 的状态输出已改成 ASCII。
- `scripts/train.py` 和 `scripts/test.py` 曾各自重复 cache 读写逻辑；本次统一复用 evaluator 的 schema-aware cache 函数。
- `SlotLevelPolicy` 当前输入仍严格保持 baseline 的 `5/2/5/2`（node/edge/candidate/chunk）。本次未增加 moment encoder 或修改 checkpoint shape。
- 原始 `ProblemInstance` 已经没有 dense `D`，但 evaluator 的重复旧类型和评估函数仍要求它，导致两个模块事实不一致；本次已消除该不一致。

## V0 验收映射

- 权威 problem 类型：已完成。
- evaluator 去除 `D`：已完成并有 dense-D 等价测试。
- optimizer 生命周期和 checkpoint resume：已实现；checkpoint round-trip 测试和真实 CPU 训练已通过。
- cache schema / generator 隔离：已完成并有 stale cache 重建测试。
- All-to-All-V 范围：已完成并测试零 pair 与非空 collective。
- baseline 模型 shape：未修改；PyTorch shape 测试已通过。
- CPU 单元测试：`44 passed`。

## 实际验收结果

```text
$env:PYTHONPATH=(Resolve-Path '.codex_deps').Path; F:\AnaConda\python.exe -m pytest -q
44 passed in 9.86s

$env:PYTHONPATH=(Resolve-Path '.codex_deps').Path; F:\AnaConda\python.exe scripts/train.py --topology Rear4GPU --num_train 8 --num_test 4 --epochs 1 --batch_target 16 --device cpu
exit 0; policy=-0.0306, value=3.8576, entropy=2.1980, reward=-0.4529

$env:PYTHONPATH=(Resolve-Path '.codex_deps').Path; F:\AnaConda\python.exe scripts/test.py --model_path checkpoints/Rear4GPU_final.pth --topology Rear4GPU --num_test 4 --device cpu
exit 0; 4 scenarios; average score=-5.4922; average completion steps=5.50

remote: /root/miniconda3/bin/python scripts/train.py --topology Rear4GPU --num_train 8 --num_test 4 --epochs 1 --batch_target 16 --device cuda --output_dir checkpoints_gpu
exit 0; NVIDIA GeForce RTX 4090; torch 2.8.0+cu128; policy=-0.0129, value=2.6379, entropy=2.2668, reward=-0.4045

remote: /root/miniconda3/bin/python scripts/test.py --model_path checkpoints_gpu/Rear4GPU_final.pth --topology Rear4GPU --num_test 4 --device cuda --output_dir outputs_gpu
exit 0; 4 scenarios; average score=-6.2423; average completion steps=6.25
```

本地验收时 `.codex_deps` 是临时项目本地 PyTorch 2.13.0+cpu；测试后已删除 519 MB 临时依赖，源码与结果保留。常规环境只需安装 PyTorch 后直接运行同一 Python 命令。远端 GPU 验证使用隔离的 `/tmp/amr-aiccl-codex-*` 目录，结束后已清理，没有覆盖服务器已有项目。生成的本地 `Rear4GPU_final.pth` 实际包含 `model_state_dict`、`optimizer_state_dict`、`epoch`、`best_score`、`config` 和 `topologies`；optimizer state 中有 34 个参数状态项。上述 CPU/GPU 数字都是 baseline smoke 验证，不是 moment policy 性能实验，也不能直接互相比性能。

## 首次执行范围外

没有实现或声称完成 V1 policy integration、CVaR、soft risk budget、trust controller、OOD/fallback 或最终性能提升实验。当前 moments 只用于数据生成、验证、历史 context 和时序 problem 组织，不会修改真实 demands 或 hard feasibility mask。
