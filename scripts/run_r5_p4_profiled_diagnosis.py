"""R5-P4 profiler-only diagnosis of optimized E1 versus D1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import StaticPlanCompiler  # noqa: E402
from rlccl.transport.reference_full_moe import seed_reference_experts  # noqa: E402
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from scripts.run_r2_f0_integrated import CHUNKS, D, EXPERTS, _load_bridge_extension  # noqa: E402
from scripts.run_r3_a0_c0 import TOTAL_TOKENS, _warm_variable_alltoallv, sha256_file  # noqa: E402
from scripts.run_r3_p0_profiled import FAMILIES, FAMILY_SPECS  # noqa: E402
from scripts.run_r4_a0_c0_full_moe import (  # noqa: E402
    EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERT_SEED, ROUTER_SEED,
    _descriptor_equivalence, _run_arm,
)


DIAGNOSTIC_SEEDS = (13042, 13142, 13242)
JOBS_PER_FAMILY = 3
SMOKE_SEED = 8042
ARMS = ("E1", "D1")


def _job_inputs(seed: int, family: str, job: int, rank: int, *, smoke: bool) -> dict[str, Any]:
    allowed = (SMOKE_SEED,) if smoke else DIAGNOSTIC_SEEDS
    if seed not in allowed:
        raise ValueError("seed not allowed by R5-P4 preregistration")
    family_index = FAMILIES.index(family)
    spec = FAMILY_SPECS[family]
    chunk_sizes = tuple(int(value) for value in spec["chunk_sizes"])
    rng = np.random.default_rng(seed * 100_000 + family_index * 1_000 + job * 10 + rank)
    tokens = rng.standard_normal((TOTAL_TOKENS, D)).astype(np.float32)
    topology_sources = np.arange(TOTAL_TOKENS, dtype=np.int64) % EXPERTS
    token_base = seed * 10_000_000_000 + family_index * 100_000_000 + job * 10_000_000
    token_ids = token_base + rank * 1_000_000 + np.arange(TOTAL_TOKENS, dtype=np.int64)
    first = tokens[:, 0].view(np.uint32).astype(np.uint64)
    second = tokens[:, 1].view(np.uint32).astype(np.uint64)
    payload_words = ((first << np.uint64(30)) ^ second ^ token_ids.astype(np.uint64))
    payload_words &= np.uint64((1 << 62) - 1)
    offsets = np.cumsum((0,) + chunk_sizes)
    return {
        "case": family, "family_index": family_index, "job": job,
        "chunk_sizes": chunk_sizes, "chunk_offsets": tuple(int(value) for value in offsets),
        "tokens": tokens, "topology_sources": topology_sources, "token_ids": token_ids,
        "payload_words": payload_words.astype(np.int64),
        "bias_delta": np.asarray(spec["bias"], dtype=np.float32),
    }


def _order(seed: int, family_index: int, job: int) -> tuple[str, str]:
    return ARMS if (DIAGNOSTIC_SEEDS.index(seed) + family_index + job) % 2 == 0 else tuple(reversed(ARMS))


def _equivalent(e1: dict[str, Any], d1: dict[str, Any], outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    forward_keys = (
        "descriptor_index", "trigger", "chunk_ids", "sendcounts_tokens", "offsets_tokens",
        "tokens", "metadata_digest", "feature_digest",
    )
    return_keys = (
        "descriptor_index", "sendcounts_tokens", "offsets_tokens", "tokens",
        "metadata_digest", "output_digest",
    )
    checks = {
        "same_router_topk": e1["topk_digests"] == d1["topk_digests"],
        "same_router_assignments": e1["router_assignment_digest"] == d1["router_assignment_digest"],
        "same_forward_descriptors": _descriptor_equivalence(e1["forward_descriptors"], forward_keys)
        == _descriptor_equivalence(d1["forward_descriptors"], forward_keys),
        "same_scheduler_actions": e1["scheduler_actions"] == d1["scheduler_actions"],
        "same_expert": e1["expert"] == d1["expert"],
        "same_return_descriptors": _descriptor_equivalence(e1["return_descriptors"], return_keys)
        == _descriptor_equivalence(d1["return_descriptors"], return_keys),
        "correct": all(
            arm["correctness"]["final_combine_correct"] and arm["correctness"]["token_integrity"]
            for arm in (e1, d1)
        ),
        "final_output_equivalent": bool(
            np.allclose(outputs["E1"], outputs["D1"], atol=2e-3, rtol=2e-3)
        ),
    }
    return {
        **checks,
        "max_abs_output_difference": float(np.max(np.abs(outputs["E1"] - outputs["D1"]))),
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.allow_smoke:
        if args.seed != SMOKE_SEED:
            raise ValueError("smoke must use consumed seed 8042")
        families, jobs_per_family = (FAMILIES[0],), 1
    else:
        if args.seed not in DIAGNOSTIC_SEEDS:
            raise ValueError("non-preregistered diagnostic seed")
        families, jobs_per_family = FAMILIES, JOBS_PER_FAMILY

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R5-P4 requires two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    plan = StaticPlanCompiler().compile(topology)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    bridge = extension.IntegratedEventBridge(CHUNKS, allowed_cpus[-(rank + 1)], rank)
    router_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)
    count_stream = torch.cuda.Stream(device=device)
    expert_stream = torch.cuda.Stream(device=device)
    router_weight_cpu, router_bias_cpu = seed_router_params(D, EXPERTS, ROUTER_SEED)
    router_weight = router_weight_cpu.to(device)
    expert_weights = tuple(value.to(device) for value in seed_reference_experts(
        D, EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERTS, EXPERT_SEED,
    ))
    local_pairs: list[dict[str, Any]] = []
    trace_path = args.output_dir / f"r5_p4_seed{args.seed}_rank{rank}.trace.json"
    try:
        _warm_variable_alltoallv(device, rank)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            for family in families:
                family_index = FAMILIES.index(family)
                for job in range(jobs_per_family):
                    case_data = _job_inputs(args.seed, family, job, rank, smoke=args.allow_smoke)
                    tokens = torch.from_numpy(case_data["tokens"]).to(device)
                    bias = (router_bias_cpu + torch.from_numpy(case_data["bias_delta"])).to(device)
                    mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
                    mask[
                        torch.arange(TOTAL_TOKENS, device=device),
                        torch.from_numpy(case_data["topology_sources"]).to(device),
                    ] = True
                    chunks = tuple(
                        tokens.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i])
                        for i in range(CHUNKS)
                    )
                    masks = tuple(
                        mask.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i])
                        for i in range(CHUNKS)
                    )
                    arms, outputs = {}, {}
                    arm_order = _order(args.seed, family_index, job) if not args.allow_smoke else ARMS
                    for arm in arm_order:
                        dist.barrier()
                        arms[arm] = _run_arm(
                            arm=arm, case_name=f"r5-p4-{args.seed}-{family}-{job}", rank=rank,
                            topology=topology, plan=plan, bridge=bridge, case_data=case_data,
                            tokens_device=tokens, token_chunks=chunks, mask_chunks=masks,
                            router_weight=router_weight, router_bias=bias,
                            expert_weights=expert_weights, router_stream=router_stream,
                            comm_stream=comm_stream, count_stream=count_stream,
                            expert_stream=expert_stream, instrument_full_expert_batches=True,
                            stream_scoped_sync=True, split_primary_timing=True,
                            fast_data_prep=True, overlap_count_with_h2d=True,
                            trace_context={"seed": args.seed, "family_index": family_index, "job": job},
                            retain_final_output=True,
                        )
                        outputs[arm] = arms[arm].pop("_final_output_array")
                    equivalence = _equivalent(arms["E1"], arms["D1"], outputs)
                    if not equivalence["pass"]:
                        raise RuntimeError(f"R5-P4 equivalence failed: {equivalence}")
                    local_pairs.append({
                        "seed": args.seed, "family": family, "family_index": family_index,
                        "job": job, "arm_order": list(arm_order), "arms": arms,
                        "equivalence": equivalence,
                    })
                    dist.barrier()
        profiler.export_chrome_trace(str(trace_path))
    finally:
        bridge.stop()

    local = {"rank": rank, "pairs": local_pairs, "trace": str(trace_path)}
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        paired_rows = []
        for pair_index in range(len(local_pairs)):
            rank_pairs = [rank_row["pairs"][pair_index] for rank_row in gathered]
            exemplar = rank_pairs[0]
            row = {key: exemplar[key] for key in ("seed", "family", "family_index", "job", "arm_order")}
            row["rank_equivalence"] = {
                str(rank_id): rank_pairs[rank_id]["equivalence"] for rank_id in range(world_size)
            }
            for arm in ARMS:
                rank_arms = [pair["arms"][arm] for pair in rank_pairs]
                first = min(value["timing"]["first_router_launch_host_ns"] for value in rank_arms)
                done = max(value["timing"]["primary_done_host_ns"] for value in rank_arms)
                row[f"{arm}_primary_us"] = (done - first) / 1e3
            row["delta_D1_minus_E1_us"] = row["D1_primary_us"] - row["E1_primary_us"]
            row["pass"] = all(value["pass"] for value in row["rank_equivalence"].values())
            paired_rows.append(row)
        if not all(value["pass"] for value in paired_rows):
            raise RuntimeError("R5-P4 aggregate equivalence failed")
        payload = {
            "schema_version": 1, "study": "R5-P4 optimized progressive diagnosis",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size,
                "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "python": platform.python_version(),
            },
            "frozen_protocol": {
                "seed": args.seed, "families": list(families),
                "jobs_per_family": jobs_per_family, "pairs": len(paired_rows),
                "smoke": args.allow_smoke, "arms": list(ARMS),
                "fast_data_prep_both_arms": True, "profiler": True,
                "only_variable": "forward descriptor communication timing",
                "partial_shards_ratio": 0.75, "checkpoint8": True,
            },
            "rank_results": gathered, "paired_rows": paired_rows, "pass": True,
        }
        host_path = args.output_dir / f"r5_p4_seed{args.seed}_host.json"
        host_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "host": str(host_path), "host_sha256": sha256_file(host_path),
            "pairs": len(paired_rows), "pass": True,
        }, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
