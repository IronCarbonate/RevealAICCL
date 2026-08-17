#!/usr/bin/env python3
"""R6-M7 two-V100 DescriptorCommit/fused LSA dispatch correctness gate."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlccl.ep.gpu_progressive_ep import GPUProgressiveEPRuntime  # noqa: E402
from rlccl.ep.layout import ProgressiveDispatchLayout  # noqa: E402
from rlccl.ep.reference import build_commit_reference, expected_rank_records  # noqa: E402
from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan  # noqa: E402
from rlccl.scheduler.common.scheduler_schema import RevealRecord, SchedulerConfig  # noqa: E402
from rlccl.transport.reference_router import router_topk  # noqa: E402


WORLD_SIZE = 2
CHUNKS = 4
TOKENS_PER_CHUNK = 32
TOTAL_TOKENS = CHUNKS * TOKENS_PER_CHUNK
FEATURE_WIDTH = 16
EXPERTS = 4
EXPERTS_PER_RANK = EXPERTS // WORLD_SIZE
NUM_TOPK = 1
M6_RECORD_BYTES = 136
OUTPUT = ROOT / "outputs" / "phase_r6" / "m7_deepep_style"
TRACE_FIELDS = (
    "commit_id", "descriptor_id", "peer", "token_count", "is_remote",
    "bytes", "t_commit_consumed", "t_dispatch_start", "t_dispatch_end",
    "t_remote_completion", "error_code",
)


def _load_torch_after_selected_nccl(library: str):
    ctypes.CDLL(str(Path(library).resolve()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    import torch  # pylint: disable=import-outside-toplevel
    return torch


def _layout() -> ProgressiveDispatchLayout:
    return ProgressiveDispatchLayout(
        WORLD_SIZE, CHUNKS, TOKENS_PER_CHUNK * NUM_TOPK, FEATURE_WIDTH,
    )


def _plan(rank: int):
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=WORLD_SIZE, source_rank=rank, record_bytes=M6_RECORD_BYTES,
        max_descriptors=CHUNKS, max_chunks=CHUNKS,
        max_tokens_per_peer=TOKENS_PER_CHUNK * NUM_TOPK,
        reveal_queue_capacity=8, action_queue_capacity=16, block_size=32,
    ))


def _records() -> tuple[RevealRecord, ...]:
    return tuple(RevealRecord(
        chunk, chunk + 1, chunk * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK,
        chunk * TOKENS_PER_CHUNK * NUM_TOPK,
        TOKENS_PER_CHUNK * NUM_TOPK, chunk,
    ) for chunk in range(CHUNKS))


def _features(rank: int) -> np.ndarray:
    values = np.zeros((TOTAL_TOKENS, FEATURE_WIDTH), dtype=np.float32)
    for token in range(TOTAL_TOKENS):
        expert = (token + rank) % EXPERTS
        values[token, expert] = np.float32(10 + rank)
        values[token, 8] = np.float32(rank)
        values[token, 9] = np.float32(token / TOTAL_TOKENS)
        values[token, 10:] = np.arange(6, dtype=np.float32) + token
    return values


def _meta_rows(values: np.ndarray) -> list[list[int | float]]:
    return [[
        int(row["src_rank"]), int(row["src_token_idx"]),
        int(row["expert_id"]), int(row["topk_slot"]),
        int(row["descriptor_id"]), int(row["reveal_epoch"]),
        float(row["topk_weight"]),
    ] for row in values]


def _worker(rank: int, library: str, unique_id: bytes, delay_cycles: int, queue) -> None:
    try:
        torch = _load_torch_after_selected_nccl(library)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        features_cpu = _features(rank)
        x = torch.from_numpy(features_cpu).to(device)
        weight = torch.zeros((FEATURE_WIDTH, EXPERTS), dtype=torch.float32, device=device)
        weight[:EXPERTS, :EXPERTS] = torch.eye(EXPERTS, device=device)
        bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
        topk_idx, topk_weights = router_topk(x, weight, bias, NUM_TOPK)
        topk_idx = topk_idx.reshape(TOTAL_TOKENS, NUM_TOPK).to(torch.int64).contiguous()
        topk_weights = topk_weights.reshape(TOTAL_TOKENS, NUM_TOPK).to(torch.float32).contiguous()
        records = _records()
        reveal_records = torch.tensor(
            [record.as_tuple() for record in records], dtype=torch.int64, device=device,
        )
        plan = _plan(rank)
        reference = build_commit_reference(
            plan, _layout(), records, topk_idx.cpu().numpy(), EXPERTS_PER_RANK,
        )
        with GPUProgressiveEPRuntime(
            library, rank=rank, device=rank, unique_id=unique_id,
            plan=plan, layout=_layout(),
        ) as runtime:
            output = runtime.run(
                reveal_records=reveal_records, x=x, topk_idx=topk_idx,
                topk_weights=topk_weights, experts_per_rank=EXPERTS_PER_RANK,
                producer_delay_cycles=delay_cycles,
                router_stream=torch.cuda.current_stream(rank).cuda_stream,
            )
            cpu_audit = {
                "cpu_per_descriptor_packing": runtime.cpu_per_descriptor_packing,
                "cpu_per_descriptor_transport_submission": runtime.cpu_per_descriptor_transport_submission,
                "cpu_per_descriptor_poll": runtime.cpu_per_descriptor_poll,
                "cpu_per_descriptor_cuda_launch": runtime.cpu_per_descriptor_cuda_launch,
                "python_callback_per_descriptor": 0,
            }
        queue.put({
            "rank": rank,
            "features": features_cpu.tolist(),
            "topk_idx": topk_idx.cpu().numpy().tolist(),
            "topk_weights": topk_weights.cpu().numpy().tolist(),
            "commits": output["commits"].tolist(),
            "expected_commits": [[getattr(c, field) for field in c.__dataclass_fields__]
                                 for c in reference.commits],
            "peer_plans": output["peer_plans"].tolist(),
            "expected_peer_plans": [[getattr(p, field) for field in p.__dataclass_fields__]
                                     for p in reference.peer_plans],
            "shadow_actions": output["shadow_actions"].tolist(),
            "expected_shadow_actions": [list(a.comparison_tuple())
                                        for a in reference.shadow_actions],
            "traces": output["traces"].tolist(),
            "timings": output["timings"].tolist(),
            "expert_counts": output["expert_counts"].tolist(),
            "expert_offsets": output["expert_offsets"].tolist(),
            "recv_x": output["recv_x"].tolist(),
            "recv_metadata": _meta_rows(output["recv_metadata"]),
            "handle": output["handle"], "counters": output["counters"],
            "capability": output["capability"], "cpu_audit": cpu_audit,
        })
    except BaseException as error:
        queue.put({"rank": rank, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def _record_key(meta, feature) -> tuple[Any, ...]:
    weight_bits = np.float32(meta[6]).tobytes().hex()
    feature_bytes = np.asarray(feature, dtype="<f4").tobytes().hex()
    return (*map(int, meta[:6]), weight_bits, feature_bytes)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--producer-delay-cycles", type=int, default=4_000_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = GPUProgressiveEPRuntime.get_unique_id(args.library)
    torch = _load_torch_after_selected_nccl(args.library)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(
        target=_worker,
        args=(rank, args.library, unique_id, args.producer_delay_cycles, queue),
    ) for rank in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    ranks = [queue.get(timeout=240) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    errors = [item for item in ranks if "error" in item]
    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(json.dumps(errors, indent=2))
    ranks.sort(key=lambda item: item["rank"])

    commit_rows = []
    commit_divergence = peer_plan_divergence = shadow_divergence = 0
    for result in ranks:
        commit_divergence += int(result["commits"] != result["expected_commits"])
        peer_plan_divergence += int(result["peer_plans"] != result["expected_peer_plans"])
        shadow_divergence += int(result["shadow_actions"] != result["expected_shadow_actions"])
        for index, row in enumerate(result["commits"]):
            commit_rows.append({
                "rank": result["rank"], "commit_index": index,
                "descriptor_id": row[1], "authorized_dst_mask": row[8],
                "equal": row == result["expected_commits"][index],
            })

    lost = duplicate = corruption = wrong_destination = 0
    payload_rows = []
    expert_rows = []
    for receiver in range(WORLD_SIZE):
        expected = []
        for source in range(WORLD_SIZE):
            source_result = ranks[source]
            expected.extend(expected_rank_records(
                source_rank=source, destination_rank=receiver,
                records=_records(),
                x=np.asarray(source_result["features"], dtype=np.float32),
                topk_idx=np.asarray(source_result["topk_idx"], dtype=np.int64),
                topk_weights=np.asarray(source_result["topk_weights"], dtype=np.float32),
                experts_per_rank=EXPERTS_PER_RANK,
            ))
        actual_meta = ranks[receiver]["recv_metadata"]
        actual_x = ranks[receiver]["recv_x"]
        expected_keys = Counter(_record_key(meta, np.frombuffer(payload, dtype="<f4"))
                                for meta, payload in expected)
        actual_keys = Counter(_record_key(meta, feature)
                              for meta, feature in zip(actual_meta, actual_x, strict=True))
        one_lost = sum((expected_keys - actual_keys).values())
        one_duplicate = sum((actual_keys - expected_keys).values())
        one_wrong = sum(not (
            receiver * EXPERTS_PER_RANK <= int(meta[2]) <
            (receiver + 1) * EXPERTS_PER_RANK
        ) for meta in actual_meta)
        lost += one_lost
        duplicate += one_duplicate
        wrong_destination += one_wrong
        corruption += one_lost + one_duplicate
        payload_rows.append({
            "receiver_rank": receiver, "expected": len(expected),
            "received": len(actual_meta), "lost": one_lost,
            "duplicate": one_duplicate, "wrong_destination": one_wrong,
            "byte_exact": expected_keys == actual_keys,
        })
        counts = ranks[receiver]["expert_counts"]
        offsets = ranks[receiver]["expert_offsets"]
        contiguous = offsets[0] == 0 and offsets[-1] == len(actual_meta)
        for local_expert, count in enumerate(counts):
            begin, end = offsets[local_expert], offsets[local_expert + 1]
            expert = receiver * EXPERTS_PER_RANK + local_expert
            segment_valid = all(int(meta[2]) == expert for meta in actual_meta[begin:end])
            contiguous &= end - begin == count and segment_valid
            expert_rows.append({
                "rank": receiver, "local_expert": local_expert,
                "global_expert": expert, "count": count,
                "offset_begin": begin, "offset_end": end,
                "contiguous": segment_valid,
            })
        ranks[receiver]["expert_contiguous"] = contiguous

    trace_rows = []
    progressive = []
    for result in ranks:
        final_router = max(int(row[7]) for row in result["timings"])
        for raw in result["traces"]:
            trace = dict(zip(TRACE_FIELDS, map(int, raw), strict=True))
            before = bool(trace["is_remote"] and
                          0 < trace["t_dispatch_start"] < final_router)
            progressive.append(before)
            trace_rows.append({"rank": result["rank"], **trace,
                               "final_router_completion": final_router,
                               "before_final_router": before})

    counters = {name: sum(int(rank["counters"][name]) for rank in ranks)
                for name in ranks[0]["counters"]}
    cpu_audit = {name: sum(int(rank["cpu_audit"][name]) for rank in ranks)
                 for name in ranks[0]["cpu_audit"]}
    gates = {
        "descriptor_commit_divergence": commit_divergence,
        "commit_peer_plan_divergence": peer_plan_divergence,
        "m6_shadow_action_divergence": shadow_divergence,
        "lost": lost, "duplicate": duplicate, "corruption": corruption,
        "wrong_destination": wrong_destination,
        "future_access": counters["future_access"],
        "unrevealed_access": counters["unrevealed_access"],
        "stale_action": counters["stale_action"],
        "device_errors": counters["errors"],
        "cpu_per_descriptor_involvement": sum(cpu_audit.values()),
        "one_commit_per_reveal": counters["descriptor_commits"] == CHUNKS * WORLD_SIZE,
        "single_scan_per_assignment": counters["assignments_scanned"] == TOTAL_TOKENS * NUM_TOPK * WORLD_SIZE,
        "direct_lsa_records_present": counters["direct_remote_records"] > 0,
        "expert_contiguous": all(rank["expert_contiguous"] for rank in ranks),
        "handle_pairing": all(rank["handle"]["num_recv_tokens"] == len(rank["recv_metadata"])
                              == len(rank["recv_x"]) for rank in ranks),
        "dispatch_before_final_router": any(progressive),
        "device_api": all(rank["capability"]["device_api_support"] for rank in ranks),
        "symmetric_window": all(rank["capability"]["symmetric_window"] for rank in ranks),
        "clean_exit": all(process.exitcode == 0 for process in processes),
    }
    zero_gates = (
        "descriptor_commit_divergence", "commit_peer_plan_divergence",
        "m6_shadow_action_divergence", "lost", "duplicate", "corruption",
        "wrong_destination", "future_access", "unrevealed_access",
        "stale_action", "device_errors", "cpu_per_descriptor_involvement",
    )
    passed = all(gates[name] == 0 for name in zero_gates) and all(
        bool(value) for name, value in gates.items() if name not in zero_gates
    )
    results = {
        "phase": "R6-M7",
        "claim": "DeepEP-style GPU Dispatch PASS" if passed else "DeepEP-style GPU Dispatch FAIL",
        "scope_stop": "ProgressiveEPHandle after expert-contiguous dispatch",
        "environment": {
            "world_size": WORLD_SIZE,
            "gpu": [torch.cuda.get_device_name(i) for i in range(WORLD_SIZE)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": "2.29.7", "compiled_architecture": "sm_70",
        },
        "execution_model": {
            "persistent_pipeline_launches_per_rank": 1,
            "post_completion_epilogue_launches_per_rank": 3,
            "total_job_level_cuda_launches_per_rank": 4,
            "persistent_blocks": 4,
            "roles": ["router", "descriptor_scheduler", "fused_dispatch", "lsa_wait"],
            "epilogue_kernels": ["count", "exclusive_scan", "scatter"],
            "producer_delay_cycles": args.producer_delay_cycles,
        },
        "layout": {
            "record_bytes": _layout().record_bytes,
            "peer_stride": _layout().peer_stride,
            "descriptor_stride": _layout().descriptor_stride,
            "region_bytes": _layout().region_bytes,
            "capacity_bytes": _layout().capacity_bytes,
            "receive_region_first": True, "gin_staging_region_second": True,
        },
        "gates": gates, "counters": counters, "cpu_audit": cpu_audit,
        "gin_runtime_status": "GIN_RUNTIME_NOT_AVAILABLE",
        "stop_rule": "dispatch complete; combine and tuning not implemented",
    }
    commit_equivalence = {
        "pass": not any((commit_divergence, peer_plan_divergence, shadow_divergence)),
        "descriptor_commit_divergence": commit_divergence,
        "commit_peer_plan_divergence": peer_plan_divergence,
        "m6_shadow_action_divergence": shadow_divergence,
        "rows": commit_rows,
        "offset_semantics": {
            "shadow_actions": "frozen M6 logical offsets",
            "commit_peer_plan": "M7 EP staging and direct remote receive offsets",
        },
    }
    payload_correctness = {
        "pass": not any((lost, duplicate, corruption, wrong_destination)),
        "lost": lost, "duplicate": duplicate, "corruption": corruption,
        "wrong_destination": wrong_destination, "rows": payload_rows,
    }
    expert_layout = {
        "pass": gates["expert_contiguous"] and gates["handle_pairing"],
        "rows": expert_rows,
        "handles": [rank["handle"] for rank in ranks],
    }
    _write_csv(args.output_dir / "dispatch_trace.csv", trace_rows)
    for name, value in (
        ("results.json", results),
        ("commit_equivalence.json", commit_equivalence),
        ("payload_correctness.json", payload_correctness),
        ("expert_layout.json", expert_layout),
    ):
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps({"claim": results["claim"], "gates": gates}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
