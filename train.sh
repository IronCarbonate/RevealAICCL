conda activate msccl

# Simple multi-topology training
# Just list topologies space-separated after --topology
python scripts/train.py \
    --topology Heterogeneous_64GPU_8Server Rear8GPU_NoSwitch_Test Rear4GPU Heterogeneous_16GPU_3Server Heterogeneous_6GPU_Ring Heterogeneous_12GPU \
    --num_train 200 \
    --num_test 100 \
    --epochs 90 \
    --ppo_epochs 10 \
    --batch_target 500 \
    --hidden_dim 128 \
    --resume checkpoints/multi_topology_epoch60.pth

# python scripts/train.py \
#     --topology Heterogeneous_64GPU_8Server \
#     --num_train 80 \
#     --num_test 100 \
#     --epochs 70 \
#     --ppo_epochs 10 \
#     --batch_target 500 \
#     --hidden_dim 128 \
#     --resume checkpoints/multi_topology_epoch60.pth