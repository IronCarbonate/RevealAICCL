"""R5-P1 preregistered progressive-expert three-arm latency pilot."""

from __future__ import annotations

import argparse
import itertools
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


PILOT_SEEDS = (10042, 10142, 10242)
JOBS_PER_FAMILY = 10
EXPERT_BATCH_THRESHOLD = 256
SMOKE_SEED = 8042
ARMS = ("P", "E0", "D")
ARM_ORDERS = tuple(itertools.permutations(ARMS))


def _job_inputs(seed: int, family: str, job: int, rank: int, *, smoke: bool) -> dict[str, Any]:
    allowed = (SMOKE_SEED,) if smoke else PILOT_SEEDS
    if seed not in allowed:
        raise ValueError("seed not allowed by R5-P1 preregistration")
    family_index = FAMILIES.index(family)
    spec = FAMILY_SPECS[family]
    chunk_sizes = tuple(int(value) for value in spec["chunk_sizes"])
    if len(chunk_sizes) != CHUNKS or sum(chunk_sizes) != TOTAL_TOKENS:
        raise RuntimeError("invalid frozen chunk layout")
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


def _arm_order(seed_index: int, family_index: int, job: int) -> tuple[str, ...]:
    return ARM_ORDERS[(seed_index * len(FAMILIES) * JOBS_PER_FAMILY + family_index * JOBS_PER_FAMILY + job) % len(ARM_ORDERS)]


def _semantic_descriptors(result: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    if direction == "forward":
        return _descriptor_equivalence(
            result["forward_descriptors"],
            ("chunk_ids", "sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest", "feature_digest"),
        )
    return _descriptor_equivalence(
        result["return_descriptors"],
        ("sendcounts_tokens", "offsets_tokens", "tokens", "metadata_digest"),
    )


def _compare_arms(arms: dict[str, dict[str, Any]], outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    base = arms["E0"]
    checks = {
        "same_router_topk": all(arms[name]["topk_digests"] == base["topk_digests"] for name in ARMS),
        "same_router_assignments": all(arms[name]["router_assignment_digest"] == base["router_assignment_digest"] for name in ARMS),
        "same_forward_descriptors": all(_semantic_descriptors(arms[name], "forward") == _semantic_descriptors(base, "forward") for name in ARMS),
        "same_expert_inputs": all(arms[name]["expert"]["input_digest"] == base["expert"]["input_digest"] for name in ARMS),
        "same_expert_mapping": all(arms[name]["expert"]["batch_shapes"] == base["expert"]["batch_shapes"] for name in ARMS),
        "same_expert_weights": all(arms[name]["expert"]["weight_digest"] == base["expert"]["weight_digest"] for name in ARMS),
        "same_return_descriptors": all(_semantic_descriptors(arms[name], "return") == _semantic_descriptors(base, "return") for name in ARMS),
        "same_scheduler_actions": all(arms[name]["scheduler_actions"] == base["scheduler_actions"] for name in ARMS),
        "all_arm_correctness": all(arms[name]["correctness"]["final_combine_correct"] and arms[name]["correctness"]["token_integrity"] for name in ARMS),
    }
    max_abs = {
        "P_vs_E0": float(np.max(np.abs(outputs["P"] - outputs["E0"]))),
        "P_vs_D": float(np.max(np.abs(outputs["P"] - outputs["D"]))),
        "E0_vs_D": float(np.max(np.abs(outputs["E0"] - outputs["D"]))),
    }
    checks["final_outputs_equivalent"] = all(
        np.allclose(outputs[left], outputs[right], atol=2e-3, rtol=2e-3)
        for left, right in (("P", "E0"), ("P", "D"), ("E0", "D"))
    )
    return {**checks, "max_abs_output_difference": max_abs, "pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    seeds = (SMOKE_SEED,) if args.allow_smoke else PILOT_SEEDS
    families = (FAMILIES[0],) if args.allow_smoke else FAMILIES
    jobs_per_family = 1 if args.allow_smoke else JOBS_PER_FAMILY

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R5-P1 requires exactly two NCCL ranks")
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
    expert_stream = torch.cuda.Stream(device=device)
    router_weight_cpu, router_bias_cpu = seed_router_params(D, EXPERTS, ROUTER_SEED)
    router_weight = router_weight_cpu.to(device)
    expert_weights = tuple(value.to(device) for value in seed_reference_experts(
        D, EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERTS, EXPERT_SEED,
    ))
    local_pairs: list[dict[str, Any]] = []
    try:
        _warm_variable_alltoallv(device, rank)
        for seed_index, seed in enumerate(seeds):
            for family in families:
                family_index = FAMILIES.index(family)
                for job in range(jobs_per_family):
                    case_data = _job_inputs(seed, family, job, rank, smoke=args.allow_smoke)
                    tokens = torch.from_numpy(case_data["tokens"]).to(device)
                    bias = (router_bias_cpu + torch.from_numpy(case_data["bias_delta"])).to(device)
                    mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
                    mask[torch.arange(TOTAL_TOKENS, device=device), torch.from_numpy(case_data["topology_sources"]).to(device)] = True
                    chunks = tuple(tokens.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
                    masks = tuple(mask.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i]) for i in range(CHUNKS))
                    arms: dict[str, dict[str, Any]] = {}
                    outputs: dict[str, np.ndarray] = {}
                    for arm in _arm_order(seed_index, family_index, job):
                        dist.barrier()
                        arms[arm] = _run_arm(
                            arm=arm, case_name=f"r5-p1-{seed}-{family}-{job}", rank=rank,
                            topology=topology, plan=plan, bridge=bridge, case_data=case_data,
                            tokens_device=tokens, token_chunks=chunks, mask_chunks=masks,
                            router_weight=router_weight, router_bias=bias,
                            expert_weights=expert_weights, router_stream=router_stream,
                            comm_stream=comm_stream, expert_stream=expert_stream,
                            progressive_expert=arm == "P",
                            expert_batch_threshold=EXPERT_BATCH_THRESHOLD if arm == "P" else 0,
                            stream_scoped_sync=True, split_primary_timing=True,
                            retain_final_output=True,
                        )
                        outputs[arm] = arms[arm].pop("_final_output_array")
                    equivalence = _compare_arms(arms, outputs)
                    if not equivalence["pass"]:
                        raise RuntimeError(f"R5-P1 arm equivalence failed: {equivalence}")
                    local_pairs.append({
                        "seed": seed, "family": family, "job": job,
                        "arm_order": list(_arm_order(seed_index, family_index, job)),
                        "arms": arms, "equivalence": equivalence,
                    })
                    dist.barrier()
    finally:
        bridge.stop()

    local = {"rank": rank, "pairs": local_pairs}
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        rows = []
        for pair_index in range(len(local_pairs)):
            rank_pairs = [rank_row["pairs"][pair_index] for rank_row in gathered]
            exemplar = rank_pairs[0]
            row: dict[str, Any] = {key: exemplar[key] for key in ("seed", "family", "job", "arm_order")}
            row["rank_equivalence"] = {str(rank_id): rank_pairs[rank_id]["equivalence"] for rank_id in range(world_size)}
            row["ranks"] = {
                str(rank_row["rank"]): rank_pairs[rank_row["rank"]]["arms"]
                for rank_row in gathered
            }
            for arm in ARMS:
                rank_arms = [pair["arms"][arm] for pair in rank_pairs]
                first = min(value["timing"]["first_router_launch_host_ns"] for value in rank_arms)
                done = max(value["timing"]["primary_done_host_ns"] for value in rank_arms)
                row[arm] = {"primary_makespan_us": (done - first) / 1e3}
            row["delta_expert_us"] = row["E0"]["primary_makespan_us"] - row["P"]["primary_makespan_us"]
            row["delta_pipeline_us"] = row["D"]["primary_makespan_us"] - row["P"]["primary_makespan_us"]
            row["delta_forward_us"] = row["D"]["primary_makespan_us"] - row["E0"]["primary_makespan_us"]
            row["pass"] = all(value["pass"] for value in row["rank_equivalence"].values())
            rows.append(row)
        if not all(row["pass"] for row in rows):
            raise RuntimeError("R5-P1 aggregate equivalence failed")
        result = {
            "schema_version": 1, "study": "R5-P1 progressive expert execution pilot",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {"world_size": world_size, "devices": [torch.cuda.get_device_name(i) for i in range(world_size)], "torch": torch.__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(), "python": platform.python_version()},
            "frozen_protocol": {"seeds": list(seeds), "families": list(families), "jobs_per_family": jobs_per_family, "pairs": len(rows), "smoke": args.allow_smoke, "arms": list(ARMS), "expert_batch_threshold": EXPERT_BATCH_THRESHOLD, "expert_progressive_only_in_P": True, "return_progressive": False, "profiler": False, "partial_shards_ratio": 0.75, "checkpoint8": True},
            "pairs": rows, "pass": True,
        }
        output = args.output_dir / ("r5_p1_smoke_host.json" if args.allow_smoke else "r5_p1_primary_host.json")
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "sha256": sha256_file(output), "pairs": len(rows), "pass": True}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
