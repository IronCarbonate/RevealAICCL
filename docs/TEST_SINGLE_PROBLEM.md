# RLCCL 单问题测试工具使用说明

## 概述

`test_single_problem.py` 是一个灵活的测试脚本，用于测试训练好的RLCCL模型在单个问题实例上的性能。

## 功能

1. 支持多种集合通信类型：
   - **AllGather**: 每个节点的chunks分发到所有其他节点
   - **AllToAll**: 每个节点向每个其他节点发送不同的chunks
   - **AllReduce**: 每个节点的chunks需要reduce到所有节点
   - **Broadcast**: 从root节点广播chunks到所有其他节点

2. 生成人类可读的策略文件（.txt）
3. 可选导出MSCCL XML文件

## 使用方法

### 基本用法

```bash
# 测试AllGather, chunk_factor=2
conda run -n msccl python scripts/test_single_problem.py \
    --model_path checkpoints/multi_topology_final.pth \
    --topology Rear8GPU_NoSwitch_Test \
    --collective allgather \
    --chunk_factor 2 \
    --output_dir test_results

# 测试AllToAll, chunk_factor=4
conda run -n msccl python scripts/test_single_problem.py \
    --model_path checkpoints/multi_topology_final.pth \
    --topology Rear8GPU_NoSwitch_Test \
    --collective alltoall \
    --chunk_factor 4 \
    --output_dir test_results

# 测试Broadcast, chunk_factor=8, root=0
conda run -n msccl python scripts/test_single_problem.py \
    --model_path checkpoints/multi_topology_final.pth \
    --topology Rear8GPU_NoSwitch_Test \
    --collective broadcast \
    --chunk_factor 8 \
    --root_node 0 \
    --output_dir test_results
```

### 导出XML

```bash
# 测试并导出XML文件
conda run -n msccl python scripts/test_single_problem.py \
    --model_path checkpoints/multi_topology_final.pth \
    --topology Rear8GPU_NoSwitch_Test \
    --collective allgather \
    --chunk_factor 2 \
    --export_xml \
    --xml_instances 1 \
    --output_dir test_results
```

## 参数说明

### 必需参数

- `--model_path`: 训练好的模型路径（.pth文件）
- `--topology`: 拓扑名称（如 Rear8GPU_NoSwitch_Test）
- `--collective`: 集合通信类型（allgather, alltoall, allreduce, broadcast）
- `--chunk_factor`: Chunk factor，决定chunk数量

### 可选参数

- `--time_limit`: 时间上限，默认30
- `--root_node`: Broadcast的root节点，默认0（仅broadcast需要）
- `--output_dir`: 输出目录，默认 ./test_results
- `--export_xml`: 添加此标志以导出MSCCL XML文件
- `--xml_instances`: XML instances数量，默认1
- `--device`: 设备（cuda:0, cpu等），默认cuda:0
- `--hidden_dim`: 模型hidden dimension，默认128

## 输出文件

脚本会在输出目录生成以下文件：

1. **策略文件** (`strategy_*.txt`): 人类可读的调度策略
   - 包含每个时间步的详细传输信息
   - 边利用率统计
   - 完成步数等

2. **摘要文件** (`test_summary.txt`): 测试结果摘要
   - 完成步数
   - 总分数
   - 总传输次数

3. **XML文件** (可选，`*.xml`): MSCCL格式的调度策略
   - 可用于NCCL测试

## 示例

### 测试多个配置

```bash
# AllGather, chunk_factor=1,2,4,8
for cf in 1 2 4 8; do
    conda run -n msccl python scripts/test_single_problem.py \
        --model_path checkpoints/multi_topology_final.pth \
        --topology Rear8GPU_NoSwitch_Test \
        --collective allgather \
        --chunk_factor $cf \
        --output_dir test_results/allgather_cf$cf \
        --export_xml
done

# AllToAll, chunk_factor=2,4
for cf in 2 4; do
    conda run -n msccl python scripts/test_single_problem.py \
        --model_path checkpoints/multi_topology_final.pth \
        --topology Rear8GPU_NoSwitch_Test \
        --collective alltoall \
        --chunk_factor $cf \
        --output_dir test_results/alltoall_cf$cf \
        --export_xml
done
```

## Checkpoint修复工具

如果你的checkpoint文件中的`best_score`是0（错误的），使用以下脚本修复：

```bash
# 修复所有checkpoint文件
conda run -n msccl python scripts/fix_checkpoint_scores.py

# 修复特定目录
conda run -n msccl python scripts/fix_checkpoint_scores.py \
    --checkpoint_dir /path/to/checkpoints

# 修复特定文件
conda run -n msccl python scripts/fix_checkpoint_scores.py \
    --checkpoint_dir /path/to/checkpoints \
    --pattern "multi_topology_epoch*.pth"
```

脚本会将所有值为0的`best_score`修改为`-1000.0`（更合理的默认差值）。

## 注意事项

1. 确保模型的`hidden_dim`参数与训练时一致
2. 拓扑名称必须在evaluator中已定义
3. XML导出需要aiccl/src目录中的`strategy_to_msccl_xml`工具
4. score是负数，越接近0越好（-5比-10好）
