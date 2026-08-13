"""R6-M3 diagnostic-only MSCCL++ post-issue GPU-start corpus."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import sys
import threading
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
from scripts.run_r3_p0_profiled import FAMILIES  # noqa: E402
from scripts.run_r4_a0_c0_full_moe import (  # noqa: E402
    EXPERT_HIDDEN, EXPERT_OUTPUT, EXPERT_SEED, ROUTER_SEED,
    _descriptor_equivalence, _run_arm,
)
from scripts.run_r6_m2_pipeline import SEEDS, _job_inputs, _strip_internal  # noqa: E402


MODES = ("normal", "router_absent", "dependency_resolved")


class RouterAbsentGate:
    """Diagnostic serialization: no future Router while current put is submitted."""

    def __init__(self) -> None:
        self._chunk_gates = [threading.Event() for _ in range(CHUNKS)]
        self._chunk_gates[0].set()

    def before_router_chunk(self, chunk: int) -> None:
        if not self._chunk_gates[chunk].wait(timeout=120):
            raise RuntimeError("R6-M3 Router-absent diagnostic gate timed out")

    def after_router_chunk_enqueued(self, chunk: int) -> None:
        if chunk == 6:
            self._chunk_gates[7].set()

    def after_descriptor_submitted(self, descriptor: int) -> None:
        if descriptor < 6:
            self._chunk_gates[descriptor + 1].set()


def _equivalent(reference: dict[str, Any], candidate: dict[str, Any], a: np.ndarray, b: np.ndarray) -> dict[str, bool]:
    forward_keys = (
        "descriptor_index", "trigger", "chunk_ids", "sendcounts_tokens",
        "offsets_tokens", "tokens", "metadata_digest", "feature_digest",
    )
    checks = {
        "router": candidate["topk_digests"] == reference["topk_digests"],
        "assignments": candidate["router_assignment_digest"] == reference["router_assignment_digest"],
        "forward_descriptors": (
            _descriptor_equivalence(candidate["forward_descriptors"], forward_keys)
            == _descriptor_equivalence(reference["forward_descriptors"], forward_keys)
        ),
        "scheduler": candidate["scheduler_actions"] == reference["scheduler_actions"],
        "expert_indices": (
            candidate["expert"]["full_batch_index_digests"]
            == reference["expert"]["full_batch_index_digests"]
        ),
        "final_output": bool(np.allclose(a, b, atol=2e-3, rtol=2e-3)),
    }
    return {**checks, "pass": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--mscclpp-library", required=True)
    parser.add_argument("--mscclpp-port-base", type=int, default=54000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    families = ("balanced",) if args.profile else FAMILIES
    modes = ("normal",) if args.profile else MODES
    dist.init_process_group("nccl", init_method="env://", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R6-M3 requires exactly two ranks")
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
    local_cases: list[dict[str, Any]] = []
    try:
        for family in families:
            data = _job_inputs(args.seed, family, 0, rank)
            tokens = torch.from_numpy(data["tokens"]).to(device)
            bias = (router_bias_cpu + torch.from_numpy(data["bias_delta"])).to(device)
            mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
            mask[
                torch.arange(TOTAL_TOKENS, device=device),
                torch.from_numpy(data["topology_sources"]).to(device),
            ] = True
            chunks = tuple(tokens.narrow(0, data["chunk_offsets"][i], data["chunk_sizes"][i]) for i in range(CHUNKS))
            masks = tuple(mask.narrow(0, data["chunk_offsets"][i], data["chunk_sizes"][i]) for i in range(CHUNKS))
            reference: dict[str, Any] | None = None
            reference_output: np.ndarray | None = None
            family_modes = modes if family == "balanced" else ("normal",)
            for mode_index, mode in enumerate(family_modes):
                dist.barrier()
                family_index = FAMILIES.index(family)
                port = args.mscclpp_port_base + SEEDS.index(args.seed) * 100 + family_index * 10 + mode_index
                max_tokens = max(max(data["chunk_sizes"][:6]), data["chunk_sizes"][6] + data["chunk_sizes"][7])
                transport = MscclppFullMoeForwardTransport(
                    library=args.mscclpp_library, rank=rank, device=rank,
                    endpoint=f"lo:127.0.0.1:{port}", comm_stream=comm_stream,
                    max_descriptors=7, max_tokens_per_peer_descriptor=max_tokens,
                    feature_width=D,
                    diagnostic_mode="dependency_resolved" if mode == "dependency_resolved" else "normal",
                )
                gate = RouterAbsentGate() if mode == "router_absent" else None
                trace_context: dict[str, Any] = {
                    "seed": args.seed, "family_index": family_index, "job": 0,
                }
                profiler_context: Any = nullcontext()
                profiler = None
                if args.profile:
                    profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                    )
                    profiler_context = profiler
                try:
                    with profiler_context:
                        result = _run_arm(
                            arm="MSCCLPP-P", case_name=f"r6-m3-{args.seed}-{family}-job0",
                            rank=rank, topology=topology, plan=plan, bridge=bridge,
                            case_data=data, tokens_device=tokens, token_chunks=chunks,
                            mask_chunks=masks, router_weight=router_weight, router_bias=bias,
                            expert_weights=expert_weights, router_stream=router_stream,
                            comm_stream=comm_stream, count_stream=count_stream,
                            expert_stream=expert_stream, instrument_full_expert_batches=True,
                            stream_scoped_sync=True, split_primary_timing=True,
                            fast_data_prep=True, overlap_count_with_h2d=False,
                            trace_context=trace_context, retain_final_output=True,
                            forward_transport=transport, router_launch_gate=gate,
                        )
                    if profiler is not None:
                        profiler.export_chrome_trace(str(args.output_dir / f"r6_m3_kineto_rank{rank}.json"))
                    output = result.pop("_final_output_array")
                    stripped = _strip_internal(result)
                    equivalence: dict[str, bool] = {"pass": True}
                    if mode == "normal":
                        reference, reference_output = stripped, output
                    else:
                        assert reference is not None and reference_output is not None
                        equivalence = _equivalent(reference, stripped, reference_output, output)
                    if not equivalence["pass"]:
                        raise RuntimeError(f"R6-M3 {mode} control changed frozen result: {equivalence}")
                    local_cases.append({
                        "seed": args.seed, "family": family, "job": 0,
                        "mode": mode, "result": stripped, "equivalence": equivalence,
                    })
                finally:
                    transport.close()
                dist.barrier()
    finally:
        bridge.stop()

    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object({"rank": rank, "cases": local_cases}, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        output = args.output_dir / f"r6_m3_seed{args.seed}{'_profile' if args.profile else ''}_host.json"
        output.write_text(json.dumps({
            "schema_version": "r6-m3-v1", "study": "R6-M3 post-issue diagnosis",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size, "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "python": platform.python_version(),
                "mscclpp": "0.9.0", "channel": "MemoryChannel",
            },
            "protocol": {"seed": args.seed, "families": list(families), "modes": list(modes), "profile": args.profile},
            "rank_results": gathered, "pass": True,
        }, indent=1) + "\n")
        print(json.dumps({"output": str(output), "cases_per_rank": len(local_cases), "pass": True}, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
