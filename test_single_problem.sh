python scripts/test_single_problem.py \
    --model_path checkpoints/multi_topology_final.pth \
    --topology Heterogeneous_64GPU_8Server \
    --collective allgather \
    --chunk_factor 2 \
    --export_xml \
    --time_limit 50 \
    --xml_instance 1