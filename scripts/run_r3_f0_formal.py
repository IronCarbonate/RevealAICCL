"""R3-F0 formal real variable-size A2Av validation.

Primary mode is deliberately profiler-free. Diagnostic mode is a separately
preregistered CUPTI subset and never contributes to primary statistics.
"""

from __future__ import annotations

import argparse
import hashlib
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

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import StaticPlanCompiler  # noqa: E402
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from scripts.run_r2_f0_integrated import (  # noqa: E402
    CHUNKS, D, EXPERTS, TOP_K, _load_bridge_extension,
)
from scripts.run_r3_a0_c0 import (  # noqa: E402
    TOTAL_TOKENS, _warm_variable_alltoallv, distribution, sha256_file,
)
from scripts.run_r3_p0_profiled import (  # noqa: E402
    A2AV_NAME, DEFAULT_CHUNKS, FAMILIES, FAMILY_SPECS, PARAMETER_SEED,
    _descriptor_signature, _run_arm,
)


FORMAL_SEEDS = (5042, 5142, 5242)
FORMAL_JOBS_PER_FAMILY = 20
DIAGNOSTIC_CORPUS_JOBS = (0, 10)
SMOKE_SEED = 7042


def _job_inputs(seed: int, family: str, corpus_job: int, rank: int) -> dict[str, Any]:
    family_index = FAMILIES.index(family)
    spec = FAMILY_SPECS[family]
    chunk_sizes = tuple(int(value) for value in spec["chunk_sizes"])
    if len(chunk_sizes) != CHUNKS or sum(chunk_sizes) != TOTAL_TOKENS:
        raise RuntimeError("invalid frozen chunk layout")
    rng_seed = seed * 100_000 + family_index * 1_000 + corpus_job * 10 + rank
    rng = np.random.default_rng(rng_seed)
    tokens = rng.standard_normal((TOTAL_TOKENS, D)).astype(np.float32)
    topology_sources = np.arange(TOTAL_TOKENS, dtype=np.int64) % EXPERTS
    token_base = seed * 10_000_000_000 + family_index * 100_000_000 + corpus_job * 10_000_000
    token_ids = token_base + rank * 1_000_000 + np.arange(TOTAL_TOKENS, dtype=np.int64)
    first = tokens[:, 0].view(np.uint32).astype(np.uint64)
    second = tokens[:, 1].view(np.uint32).astype(np.uint64)
    payload_words = ((first << np.uint64(30)) ^ second ^ token_ids.astype(np.uint64))
    payload_words &= np.uint64((1 << 62) - 1)
    offsets = np.cumsum((0,) + chunk_sizes)
    return {
        "case": family, "family_index": family_index, "job": corpus_job,
        "chunk_sizes": chunk_sizes,
        "chunk_offsets": tuple(int(value) for value in offsets),
        "tokens": tokens, "topology_sources": topology_sources,
        "token_ids": token_ids, "payload_words": payload_words.astype(np.int64),
        "bias_delta": np.asarray(spec["bias"], dtype=np.float32),
    }


def _pair_order(seed_index: int, family_index: int, corpus_job: int) -> tuple[str, str]:
    return ("C", "D") if (seed_index + family_index + corpus_job) % 2 == 0 else ("D", "C")


def _run_seed(
    *, seed: int, seed_index: int, corpus_jobs: tuple[int, ...], rank: int,
    topology: Any, plan: Any, bridge: Any, device: torch.device,
    weight: torch.Tensor, base_bias_cpu: torch.Tensor,
    router_stream: torch.cuda.Stream, comm_stream: torch.cuda.Stream,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for ordinal, corpus_job in enumerate(corpus_jobs):
            case_data = _job_inputs(seed, family, corpus_job, rank)
            tokens_device = torch.from_numpy(case_data["tokens"]).to(device)
            bias = (base_bias_cpu + torch.from_numpy(case_data["bias_delta"])).to(device)
            mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
            mask[
                torch.arange(TOTAL_TOKENS, device=device),
                torch.from_numpy(case_data["topology_sources"]).to(device),
            ] = True
            chunks = tuple(
                tokens_device.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i])
                for i in range(CHUNKS)
            )
            masks = tuple(
                mask.narrow(0, case_data["chunk_offsets"][i], case_data["chunk_sizes"][i])
                for i in range(CHUNKS)
            )
            # Frozen shape-matched warmup and GPU-idle boundary outside primary timing.
            with torch.inference_mode(), torch.cuda.stream(router_stream):
                router_topk(chunks[0], weight, bias, TOP_K, mask=masks[0])
            torch.cuda.synchronize(device)
            visible_job = corpus_job if len(corpus_jobs) == FORMAL_JOBS_PER_FAMILY else ordinal
            order = _pair_order(seed_index, family_index, corpus_job)
            arms: dict[str, Any] = {}
            for arm in order:
                dist.barrier()
                arms[arm] = _run_arm(
                    arm=arm, seed=seed, family=family, job=visible_job, rank=rank,
                    topology=topology, plan=plan, bridge=bridge, case_data=case_data,
                    tokens_device=tokens_device, token_chunks=chunks, mask_chunks=masks,
                    weight=weight, bias=bias, router_stream=router_stream,
                    comm_stream=comm_stream,
                )
            pairs.append({
                "seed": seed, "family": family, "job": visible_job,
                "corpus_job": corpus_job, "order": list(order),
                "C": arms["C"], "D": arms["D"],
            })
            dist.barrier()
    return pairs


def _aggregate(rank_results: list[dict[str, Any]], seeds: tuple[int, ...], mode: str) -> dict[str, Any]:
    rank_lookup = {
        (int(row["rank"]), int(pair["seed"]), pair["family"], int(pair["job"])): pair
        for row in rank_results for pair in row["pairs"]
    }
    jobs = sorted({int(pair["job"]) for row in rank_results for pair in row["pairs"]})
    pair_rows, equivalence = [], []
    for seed in seeds:
        for family in FAMILIES:
            for job in jobs:
                rank_pairs = [rank_lookup[(rank, seed, family, job)] for rank in (0, 1)]
                c_rows, d_rows = [value["C"] for value in rank_pairs], [value["D"] for value in rank_pairs]
                c_primary = (
                    max(row["primary_done_host_ns"] for row in c_rows)
                    - min(row["first_router_launch_host_ns"] for row in c_rows)
                ) / 1e3
                d_primary = (
                    max(row["primary_done_host_ns"] for row in d_rows)
                    - min(row["first_router_launch_host_ns"] for row in d_rows)
                ) / 1e3
                c_full = (
                    max(row["full_reference_done_host_ns"] for row in c_rows)
                    - min(row["first_router_launch_host_ns"] for row in c_rows)
                ) / 1e3
                d_full = (
                    max(row["full_reference_done_host_ns"] for row in d_rows)
                    - min(row["first_router_launch_host_ns"] for row in d_rows)
                ) / 1e3
                pair_rows.append({
                    "seed": seed, "family": family, "job": job,
                    "corpus_job": rank_pairs[0]["corpus_job"],
                    "C_primary_us": c_primary, "D_primary_us": d_primary,
                    "delta_us": d_primary - c_primary,
                    "C_full_reference_us": c_full, "D_full_reference_us": d_full,
                })
                for rank, pair in enumerate(rank_pairs):
                    c, d = pair["C"], pair["D"]
                    checks = {
                        "same_router": c["router_assignment_digest"] == d["router_assignment_digest"],
                        "same_topk": c["topk_by_chunk_digests"] == d["topk_by_chunk_digests"],
                        "same_descriptors": _descriptor_signature(c) == _descriptor_signature(d),
                        "same_bytes": c["total_sent_bytes"] == d["total_sent_bytes"],
                        "same_calls": len(c["descriptors"]) == len(d["descriptors"]),
                        "same_actions": c["scheduler_action_signatures"] == d["scheduler_action_signatures"],
                        "same_sent_multiset": c["final_sent_payload_multiset_digest"] == d["final_sent_payload_multiset_digest"],
                        "same_received_multiset": c["final_received_payload_multiset_digest"] == d["final_received_payload_multiset_digest"],
                    }
                    equivalence.append({
                        "seed": seed, "family": family, "job": job, "rank": rank,
                        **checks, "pass": all(checks.values()),
                    })
    arms = [pair[arm] for row in rank_results for pair in row["pairs"] for arm in ("C", "D")]
    semantics, verifications = [row["semantic"] for row in arms], [row["verification"] for row in arms]
    correctness = {
        "runtime_bfs_zero": all(row["runtime_bfs_calls"] == 0 for row in semantics),
        "full_rebuild_zero": all(row["full_rebuild_count"] == 0 for row in semantics),
        "unrevealed_execution_zero": all(row["unrevealed_execution"] == 0 for row in semantics),
        "future_access_zero": all(row["future_access"] == 0 for row in semantics),
        "duplicate_dispatch_zero": all(row["duplicate_dispatch"] == 0 for row in semantics),
        "stale_dispatch_zero": all(row["stale_dispatch"] == 0 for row in semantics),
        "semantic_divergence_zero": all(
            row[key] == 0 for row in semantics
            for key in ("candidate_divergences", "action_divergences", "checker_divergences", "holder_divergences")
        ),
        "legality_100pct": all(row["legal"] == row["total"] for row in semantics),
        "token_integrity_100pct": all(row["token_integrity"] for row in semantics),
        "lost_zero": all(row["lost"] == 0 for row in verifications),
        "duplicate_zero": all(row["duplicate"] == 0 for row in verifications),
        "wrong_destination_zero": all(row["wrong_destination"] == 0 for row in verifications),
        "corruption_zero": all(row["corruption"] == 0 for row in verifications),
        "cd_equivalence_zero": all(row["pass"] for row in equivalence),
    }
    return {
        "schema_version": 1, "study": f"R3-F0 {mode}", "mode": mode,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_protocol": {
            "seeds": list(seeds), "families": list(FAMILIES),
            "jobs_per_family": len(jobs), "corpus_jobs": sorted({int(row["corpus_job"]) for row in pair_rows}),
            "parameter_seed": PARAMETER_SEED, "tokens_per_rank": TOTAL_TOKENS,
            "dimension": D, "experts": EXPERTS, "top_k": TOP_K, "chunks": CHUNKS,
            "partial_shards_ratio": 0.75, "checkpoint8": True, "transport": A2AV_NAME,
            "profiler_enabled": mode == "diagnostic",
            "single_nccl_initialization": True, "shared_communicator": True,
        },
        "correctness": correctness, "pass": all(correctness.values()),
        "paired_rows": pair_rows, "equivalence": equivalence,
        "host_diagnostics": {
            "C_primary_us": distribution([row["C_primary_us"] for row in pair_rows]),
            "D_primary_us": distribution([row["D_primary_us"] for row in pair_rows]),
            "delta_us": distribution([row["delta_us"] for row in pair_rows]),
        },
        "rank_results": rank_results,
        "environment": {
            "world_size": 2, "devices": [torch.cuda.get_device_name(i) for i in range(2)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(), "python": platform.python_version(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("primary", "diagnostic", "smoke"), required=True)
    args = parser.parse_args()
    if args.mode == "primary":
        seeds, corpus_jobs = FORMAL_SEEDS, tuple(range(FORMAL_JOBS_PER_FAMILY))
    elif args.mode == "diagnostic":
        seeds, corpus_jobs = FORMAL_SEEDS, DIAGNOSTIC_CORPUS_JOBS
    else:
        seeds, corpus_jobs = (SMOKE_SEED,), (0,)

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R3-F0 requires exactly two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler()
    plan = compiler.compile(topology)
    if compiler.compile_bfs_sources <= 0 or not plan.proof.valid:
        raise RuntimeError("compiled static proof unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed[-(rank + 1)] if allowed != [-1] else -1
    bridge = extension.IntegratedEventBridge(CHUNKS, cpu_core, rank)
    router_stream = torch.cuda.Stream(device=device, priority=0)
    comm_stream = torch.cuda.Stream(device=device, priority=0)
    _warm_variable_alltoallv(device, rank)
    weight_cpu, base_bias_cpu = seed_router_params(D, EXPERTS, PARAMETER_SEED)
    weight = weight_cpu.to(device)
    all_pairs: list[dict[str, Any]] = []
    trace_paths: dict[int, Path] = {}
    try:
        for seed_index, seed in enumerate(seeds):
            if args.mode == "diagnostic":
                seed_dir = args.output_dir / f"seed{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                trace_path = seed_dir / f"r3_p0_seed{seed}_rank{rank}.trace.json"
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False,
                ) as profiler:
                    seed_pairs = _run_seed(
                        seed=seed, seed_index=seed_index, corpus_jobs=corpus_jobs, rank=rank,
                        topology=topology, plan=plan, bridge=bridge, device=device,
                        weight=weight, base_bias_cpu=base_bias_cpu,
                        router_stream=router_stream, comm_stream=comm_stream,
                    )
                    torch.cuda.synchronize(device)
                profiler.export_chrome_trace(str(trace_path))
                trace_paths[seed] = trace_path
            else:
                # No profiler object or context exists in primary/smoke mode.
                seed_pairs = _run_seed(
                    seed=seed, seed_index=seed_index, corpus_jobs=corpus_jobs, rank=rank,
                    topology=topology, plan=plan, bridge=bridge, device=device,
                    weight=weight, base_bias_cpu=base_bias_cpu,
                    router_stream=router_stream, comm_stream=comm_stream,
                )
            all_pairs.extend(seed_pairs)
            dist.barrier()
    finally:
        bridge.stop()

    local = {
        "rank": rank, "poller_cpu_core": cpu_core, "poller_pinned": bool(bridge.pinned),
        "pairs": all_pairs,
    }
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        if args.mode == "diagnostic":
            for seed in seeds:
                sliced = []
                for row in gathered:
                    rank_trace = trace_paths[seed].parent / f"r3_p0_seed{seed}_rank{row['rank']}.trace.json"
                    sliced.append({
                        **{key: value for key, value in row.items() if key != "pairs"},
                        "seed": seed, "trace_path": str(rank_trace),
                        "trace_sha256": sha256_file(rank_trace),
                        "trace_size_bytes": rank_trace.stat().st_size,
                        "pairs": [pair for pair in row["pairs"] if int(pair["seed"]) == seed],
                    })
                payload = _aggregate(sliced, (seed,), "diagnostic")
                output = args.output_dir / f"seed{seed}" / f"r3_p0_seed{seed}_host.json"
                output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            payload = _aggregate(gathered, seeds, args.mode)
            output = args.output_dir / f"r3_f0_{args.mode}_host.json"
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not payload["pass"]:
            raise RuntimeError(f"R3-F0 {args.mode} correctness gate failed")
        print(json.dumps({
            "mode": args.mode, "output": str(output), "sha256": sha256_file(output),
            "pairs": sum(len(row["pairs"]) for row in gathered) // world_size,
            "pass": payload["pass"], "profiler_enabled": args.mode == "diagnostic",
        }, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
