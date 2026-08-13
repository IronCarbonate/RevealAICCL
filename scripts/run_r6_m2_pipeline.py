"""R6-M2 four-arm full-MoE progressive pipeline pilot (one seed per run)."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rlccl.scheduling.compiled_event_driven import StaticPlanCompiler  # noqa: E402
from rlccl.transport.mscclpp_full_moe import MscclppFullMoeForwardTransport  # noqa: E402
from rlccl.transport.reference_full_moe import seed_reference_experts  # noqa: E402
from rlccl.transport.reference_router import seed_router_params  # noqa: E402
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from scripts.run_r2_f0_integrated import CHUNKS, D, EXPERTS, _load_bridge_extension  # noqa: E402
from scripts.run_r3_a0_c0 import TOTAL_TOKENS  # noqa: E402
from scripts.run_r3_p0_profiled import FAMILIES, FAMILY_SPECS  # noqa: E402
from scripts.run_r4_a0_c0_full_moe import (  # noqa: E402
    EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERT_SEED, ROUTER_SEED,
    _descriptor_equivalence, _run_arm,
)


SEEDS = (13042, 13142, 13242)
JOBS_PER_FAMILY = 3
ARMS = ("NCCL-D", "NCCL-P", "MSCCLPP-D", "MSCCLPP-P")


def _job_inputs(seed: int, family: str, job: int, rank: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("R6-M2 seed is outside preregistration")
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
        "chunk_sizes": chunk_sizes,
        "chunk_offsets": tuple(int(value) for value in offsets),
        "tokens": tokens, "topology_sources": topology_sources,
        "token_ids": token_ids, "payload_words": payload_words.astype(np.int64),
        "bias_delta": np.asarray(spec["bias"], dtype=np.float32),
    }


def _arm_order(seed: int, family_index: int, job: int) -> tuple[str, ...]:
    shift = (SEEDS.index(seed) + family_index + job) % len(ARMS)
    return ARMS[shift:] + ARMS[:shift]


def _equivalence(arms: dict[str, dict[str, Any]], outputs: dict[str, np.ndarray]) -> dict[str, Any]:
    forward_keys = (
        "descriptor_index", "trigger", "chunk_ids", "sendcounts_tokens",
        "offsets_tokens", "tokens", "metadata_digest", "feature_digest",
    )
    return_keys = (
        "descriptor_index", "sendcounts_tokens", "offsets_tokens", "tokens",
        "metadata_digest", "output_digest",
    )
    reference = arms[ARMS[0]]
    checks: dict[str, bool] = {}
    for arm in ARMS[1:]:
        value = arms[arm]
        prefix = arm.lower().replace("-", "_")
        checks[f"{prefix}_router"] = value["topk_digests"] == reference["topk_digests"]
        checks[f"{prefix}_assignments"] = (
            value["router_assignment_digest"] == reference["router_assignment_digest"]
        )
        checks[f"{prefix}_forward"] = (
            _descriptor_equivalence(value["forward_descriptors"], forward_keys)
            == _descriptor_equivalence(reference["forward_descriptors"], forward_keys)
        )
        checks[f"{prefix}_scheduler"] = value["scheduler_actions"] == reference["scheduler_actions"]
        checks[f"{prefix}_expert"] = value["expert"] == reference["expert"]
        checks[f"{prefix}_return"] = (
            _descriptor_equivalence(value["return_descriptors"], return_keys)
            == _descriptor_equivalence(reference["return_descriptors"], return_keys)
        )
        checks[f"{prefix}_final"] = bool(
            np.allclose(outputs[arm], outputs[ARMS[0]], atol=2e-3, rtol=2e-3)
        )
    for arm, value in arms.items():
        correct = value["correctness"]
        semantic = value["semantic"]
        checks[f"{arm}_correct"] = bool(
            correct["final_combine_correct"] and correct["token_integrity"]
            and all(v == 0 for k, v in correct.items()
                    if k not in ("final_combine_correct", "token_integrity"))
            and semantic["legal"] == semantic["total"]
            and all(semantic[k] == 0 for k in (
                "runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution",
                "future_access", "duplicate_dispatch", "stale_dispatch",
                "candidate_divergences", "action_divergences",
                "checker_divergences", "holder_divergences",
            ))
        )
    return {**checks, "pass": all(checks.values())}


def _strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_internal(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, list):
        return [_strip_internal(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--mscclpp-library", required=True)
    parser.add_argument("--mscclpp-port-base", type=int, default=52000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    families = (FAMILIES[0],) if args.smoke else FAMILIES
    jobs = 1 if args.smoke else JOBS_PER_FAMILY

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R6-M2 requires exactly two ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    plan = StaticPlanCompiler().compile(topology)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    bridge = extension.IntegratedEventBridge(CHUNKS, cpus[-(rank + 1)], rank)
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
    try:
        for family in families:
            family_index = FAMILIES.index(family)
            for job in range(jobs):
                data = _job_inputs(args.seed, family, job, rank)
                tokens = torch.from_numpy(data["tokens"]).to(device)
                bias = (router_bias_cpu + torch.from_numpy(data["bias_delta"])).to(device)
                mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
                mask[
                    torch.arange(TOTAL_TOKENS, device=device),
                    torch.from_numpy(data["topology_sources"]).to(device),
                ] = True
                chunks = tuple(
                    tokens.narrow(0, data["chunk_offsets"][i], data["chunk_sizes"][i])
                    for i in range(CHUNKS)
                )
                masks = tuple(
                    mask.narrow(0, data["chunk_offsets"][i], data["chunk_sizes"][i])
                    for i in range(CHUNKS)
                )
                arms: dict[str, dict[str, Any]] = {}
                outputs: dict[str, np.ndarray] = {}
                order = _arm_order(args.seed, family_index, job)
                for arm in order:
                    dist.barrier()
                    transport = None
                    if arm.startswith("MSCCLPP"):
                        case_index = family_index * JOBS_PER_FAMILY + job
                        arm_offset = 0 if arm.endswith("-D") else 1
                        endpoint_port = (
                            args.mscclpp_port_base
                            + SEEDS.index(args.seed) * 100
                            + case_index * 2 + arm_offset
                        )
                        max_tokens = max(
                            max(data["chunk_sizes"][:6]),
                            data["chunk_sizes"][6] + data["chunk_sizes"][7],
                        )
                        transport = MscclppFullMoeForwardTransport(
                            library=args.mscclpp_library, rank=rank, device=rank,
                            endpoint=f"lo:127.0.0.1:{endpoint_port}",
                            comm_stream=comm_stream, max_descriptors=7,
                            max_tokens_per_peer_descriptor=max_tokens,
                            feature_width=D,
                        )
                    try:
                        result = _run_arm(
                            arm=arm, case_name=f"r6-m2-{args.seed}-{family}-{job}",
                            rank=rank, topology=topology, plan=plan, bridge=bridge,
                            case_data=data, tokens_device=tokens, token_chunks=chunks,
                            mask_chunks=masks, router_weight=router_weight,
                            router_bias=bias, expert_weights=expert_weights,
                            router_stream=router_stream, comm_stream=comm_stream,
                            count_stream=count_stream, expert_stream=expert_stream,
                            instrument_full_expert_batches=True,
                            stream_scoped_sync=True, split_primary_timing=True,
                            fast_data_prep=True,
                            overlap_count_with_h2d=arm.startswith("NCCL"),
                            trace_context={
                                "seed": args.seed, "family_index": family_index,
                                "job": job,
                            },
                            retain_final_output=True,
                            forward_transport=transport,
                        )
                        outputs[arm] = result.pop("_final_output_array")
                        arms[arm] = _strip_internal(result)
                    finally:
                        if transport is not None:
                            transport.close()
                equivalent = _equivalence(arms, outputs)
                if not equivalent["pass"]:
                    raise RuntimeError(f"R6-M2 four-arm equivalence failed: {equivalent}")
                local_pairs.append({
                    "seed": args.seed, "family": family,
                    "family_index": family_index, "job": job,
                    "arm_order": list(order), "arms": arms,
                    "equivalence": equivalent,
                })
                dist.barrier()
    finally:
        bridge.stop()

    local = {"rank": rank, "pairs": local_pairs}
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        output = args.output_dir / f"r6_m2_seed{args.seed}_host.json"
        payload = {
            "schema_version": 1,
            "study": "R6-M2 four-arm full-MoE progressive pipeline pilot",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size,
                "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "python": platform.python_version(),
                "mscclpp": "0.9.0", "channel": "MemoryChannel",
            },
            "protocol": {
                "seed": args.seed, "families": list(families),
                "jobs_per_family": jobs, "arms": list(ARMS),
                "full_moe_endpoint": True, "smoke": args.smoke,
            },
            "rank_results": gathered,
            "pass": True,
        }
        output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(json.dumps({"output": str(output), "pairs": len(local_pairs), "pass": True}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
