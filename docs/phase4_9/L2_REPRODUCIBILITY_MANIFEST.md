# L2 可复现清单（Reproducibility Manifest）

更新日期：2026-08-05

## 1. 环境

| 项 | 值 |
|---|---|
| 主机 | autodl-container-8eeb11bf52-0720a64c（region-41） |
| GPU | 2× Tesla V100-SXM2-32GB |
| 驱动 / CUDA | 580.105.08 / 13.0 |
| torch / NCCL | 2.8.0+cu128 / 2.27.3 |
| numpy | 2.3.2 |
| 环境变量 | PYTHONHASHSEED=0、PYTHONDONTWRITEBYTECODE=1、LC_ALL=C、PH4_8_PROFILE=1、PH4_8_L2=1 |

## 2. Corpus

- base seeds (3042, 3142, 3242)；universe digest `d1daf2fa44f876b56197290929050670a6ff4057b6c48117a2f785f75ed753c2`；与 H2/Route A/H5-H7 零重合。

## 3. 脚本与命令

```bash
cd /root/autodl-tmp/RLCCL-main
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C PYTHONPATH=$PWD PH4_8_L2=1 PH4_8_PROFILE=1
/root/miniconda3/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29501 \
  outputs/phase4_8/deployment_validation/formal_test.py
/root/miniconda3/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \
  outputs/phase4_8/deployment_validation/l2_collective_bench.py
```

## 4. Artifacts（与 hashes_l2.json 一一对应）

l2_final_summary.json、l2_condition_summary.json、l2_timing_breakdown.json、l2_throughput_results.json、l2_job_sequence_results.json、l2_environment_manifest.json、l2_collective_results.json、raw_jobs.json、protocol_manifest.json、run_command.txt。

## 5. 统计

sequence-level paired bootstrap 10,000（seed 20260801）；独立 job/sequence 为单位；read-back（read_back_report_l2.json）通过。
