#!/bin/bash
conda activate msccl

# Simple multi-topology testing
# Just list topologies space-separated after --topology
python scripts/test.py \
    --model_path checkpoints/super_trained/checkpoint_epoch_30.pth \
    --topology Rear8GPU_NoSwitch_Test Rear4GPU Heterogeneous_16GPU_3Server Heterogeneous_6GPU_Ring Heterogeneous_12GPU \
    --num_test 30 \
    --export_xml \
    --device cpu
