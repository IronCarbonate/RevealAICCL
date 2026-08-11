# RLCCL 简洁使用指南

## 快速开始

### 训练模型

```bash
python scripts/train.py \
    --topology Rear4GPU Rear8GPU_NoSwitch_Test \
    --num_train 80 \
    --num_test 30 \
    --epochs 30 \
    --batch_target 500
```

### 测试模型

```bash
python scripts/test.py \
    --model_path checkpoints/checkpoint_epoch_30.pth \
    --topology Rear4GPU Rear8GPU_NoSwitch_Test \
    --num_test 30 \
    --export_xml
```

## 参数说明

### 训练参数 (train.py)

- `--topology`: 拓扑名称,**空格分隔**多个拓扑
  - 例如: `--topology Rear4GPU Rear8GPU_NoSwitch_Test Heterogeneous_12GPU`
- `--num_train`: 每个拓扑的训练场景数 (默认: 80)
- `--num_test`: 每个拓扑的测试场景数 (默认: 30)
- `--epochs`: 训练轮数 (默认: 30)
- `--batch_target`: 目标batch大小 (默认: 500)
- `--hidden_dim`: 隐藏层维度 (默认: 128)
- `--device`: 训练设备 (默认: cuda, 可选: cpu)
- `--resume`: 从检查点恢复训练
- `--save_dir`: 检查点保存目录 (默认: checkpoints)

### 测试参数 (test.py)

- `--model_path`: 模型检查点路径 (**必需**)
- `--topology`: 拓扑名称,**空格分隔**多个拓扑
- `--num_test`: 每个拓扑的测试场景数 (默认: 30)
- `--export_xml`: 导出MSCCL XML文件
- `--output_dir`: 输出目录 (默认: outputs)
- `--device`: 测试设备 (默认: cuda, 可选: cpu)

## 数据缓存

训练和测试脚本会**自动管理场景数据**:

- 如果 `Data/{topology}/scenarios_{train|test}_{num}_seed{seed}.json` 存在,直接加载
- 如果不存在,自动生成并保存

**无需手动管理数据文件!**

## 可用拓扑

- `Rear4GPU`: 4 GPU后置拓扑
- `Rear8GPU_NoSwitch_Test`: 8 GPU后置拓扑(无交换机)
- `Heterogeneous_12GPU`: 12 GPU异构拓扑
- `Heterogeneous_16GPU_3Server`: 16 GPU三服务器异构拓扑
- `Heterogeneous_6GPU_Ring`: 6 GPU环形拓扑

## 快捷脚本

### 训练

```bash
./train.sh  # 使用预配置的多拓扑训练
```

### 测试

```bash
./test.sh  # 测试所有拓扑并导出XML
```

## 注意事项

1. **拓扑名称用空格分隔**,不是逗号:
   - ✅ `--topology A B C`
   - ❌ `--topology A,B,C`

2. **自动缓存**: 首次运行会生成并保存场景,后续运行直接加载

3. **恢复训练**: 使用 `--resume` 从检查点继续训练

4. **GPU/CPU**: 使用 `--device cpu` 在CPU上训练/测试
