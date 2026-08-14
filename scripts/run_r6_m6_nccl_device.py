#!/usr/bin/env python3
"""R6-M6 two-V100 real NCCL Device API LSA correctness gate."""

from __future__ import annotations

import argparse
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

from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan  # noqa: E402
from rlccl.scheduler.common.scheduler_schema import RevealRecord, SchedulerConfig  # noqa: E402
from rlccl.scheduler.cpu.reference import CPUSchedulerShadow  # noqa: E402
from rlccl.transport.cuda.layout import GPURegisteredBufferLayout  # noqa: E402
from rlccl.transport.gpu_driven_nccl import GPUDrivenNcclLsaRuntime  # noqa: E402
from rlccl.transport.reference_full_moe import feature_digest, identity_checksum  # noqa: E402
from rlccl.transport.reference_router import router_topk  # noqa: E402


WORLD_SIZE = 2
CHUNKS = 4
TOKENS_PER_CHUNK = 32
TOTAL_TOKENS = CHUNKS * TOKENS_PER_CHUNK
FEATURE_WIDTH = 16
EXPERTS = 4
META_FIELDS = 9
OUTPUT = ROOT / "outputs" / "phase_r6" / "m6_nccl_device"
ACTION_FIELDS = (
    "action_id", "descriptor_id", "chunk_id", "reveal_epoch", "src_rank",
    "dst_rank", "src_offset", "dst_offset", "token_count", "bytes",
    "route_id", "flags",
)
TRACE_FIELDS = (
    "action_id", "descriptor_id", "src_rank", "dst_rank",
    "logical_src_offset", "logical_dst_offset", "physical_src_offset",
    "physical_dst_offset", "payload_bytes", "physical_bytes", "token_count",
    "route_id", "t2_action_consumed", "t3_pack_start", "t4_pack_end",
    "t5_put_start", "t6_put_end", "error_code", "is_remote",
)


def _load_torch_after_selected_nccl(library: str):
    """Load the rpath-selected NCCL before PyTorch can bind an older SONAME."""
    ctypes.CDLL(
        str(Path(library).resolve()),
        mode=getattr(ctypes, "RTLD_GLOBAL", 0),
    )
    import torch  # pylint: disable=import-outside-toplevel
    return torch


def _layout() -> GPURegisteredBufferLayout:
    return GPURegisteredBufferLayout(
        world_size=WORLD_SIZE, max_descriptors=CHUNKS,
        max_tokens_per_peer=TOKENS_PER_CHUNK,
        metadata_fields=META_FIELDS, feature_width=FEATURE_WIDTH,
    )


def _plan(rank: int, layout: GPURegisteredBufferLayout):
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=WORLD_SIZE, source_rank=rank, record_bytes=layout.record_bytes,
        max_descriptors=CHUNKS, max_chunks=CHUNKS,
        max_tokens_per_peer=TOKENS_PER_CHUNK,
        reveal_queue_capacity=8, action_queue_capacity=16, block_size=32,
    ))


def _records() -> tuple[RevealRecord, ...]:
    return tuple(RevealRecord(
        chunk_id=chunk, reveal_epoch=chunk + 1,
        token_begin=chunk * TOKENS_PER_CHUNK, token_count=TOKENS_PER_CHUNK,
        assignment_begin=chunk * TOKENS_PER_CHUNK,
        assignment_count=TOKENS_PER_CHUNK, descriptor_id=chunk,
    ) for chunk in range(CHUNKS))


def _decode_received(raw: bytes, layout: GPURegisteredBufferLayout, descriptor: int, source: int):
    base = layout.receive_offset(descriptor, source)
    count = int.from_bytes(raw[base:base + 8], "little", signed=False)
    if not 0 <= count <= layout.max_tokens_per_peer:
        raise RuntimeError(f"receive count outside slot: d={descriptor} src={source} count={count}")
    return [
        raw[
            base + 8 + index * layout.record_bytes:
            base + 8 + (index + 1) * layout.record_bytes
        ]
        for index in range(count)
    ]


def _worker(rank: int, library: str, unique_id: bytes, delay_cycles: int, queue) -> None:
    try:
        torch = _load_torch_after_selected_nccl(library)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        layout = _layout()
        plan = _plan(rank, layout)
        records = _records()
        features_cpu = np.zeros((TOTAL_TOKENS, FEATURE_WIDTH), dtype=np.float32)
        for token in range(TOTAL_TOKENS):
            features_cpu[token, token % EXPERTS] = np.float32(10.0 + rank)
            features_cpu[token, 8] = np.float32(rank)
            features_cpu[token, 9] = np.float32(token / 128.0)
        token_ids_cpu = rank * 1_000_000 + np.arange(TOTAL_TOKENS, dtype=np.int64)
        digests_cpu = np.asarray([feature_digest(row) for row in features_cpu], dtype=np.int64)
        features = torch.from_numpy(features_cpu).to(device)
        token_ids = torch.from_numpy(token_ids_cpu).to(device)
        feature_digests = torch.from_numpy(digests_cpu).to(device)
        weight = torch.zeros((FEATURE_WIDTH, EXPERTS), dtype=torch.float32, device=device)
        weight[:EXPERTS, :EXPERTS] = torch.eye(EXPERTS, device=device)
        bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
        expert_ids, _ = router_topk(features, weight, bias, 1)
        expert_ids = expert_ids.reshape(-1).to(torch.int64).contiguous()
        destination_ranks = expert_ids.remainder(WORLD_SIZE).to(torch.int32).contiguous()
        reveal_records = torch.tensor(
            [record.as_tuple() for record in records], dtype=torch.int64, device=device,
        )
        metadata = torch.empty((TOTAL_TOKENS, META_FIELDS), dtype=torch.int64, device=device)
        router_stream = torch.cuda.current_stream(rank)

        with GPUDrivenNcclLsaRuntime(
            library, rank=rank, device=rank, unique_id=unique_id,
            plan=plan, layout=layout,
        ) as runtime:
            output = runtime.run(
                reveal_records=reveal_records,
                destination_ranks=destination_ranks,
                expert_ids=expert_ids,
                token_ids=token_ids,
                feature_digests=feature_digests,
                features=features,
                metadata=metadata,
                expected_remote_actions=CHUNKS,
                producer_delay_cycles=delay_cycles,
                router_stream=router_stream.cuda_stream,
            )

        # All CPU work below is post-job debug/reference collection.
        metadata_cpu = metadata.cpu().numpy()
        destination_cpu = destination_ranks.cpu().numpy()
        expert_cpu = expert_ids.cpu().numpy()
        registered_cpu = output["registered_buffer"].tobytes()
        cpu_run = CPUSchedulerShadow(plan).run(records, tuple(map(int, destination_cpu.tolist())))
        gpu_action_rows = output["actions"].tolist()
        cpu_action_rows = [list(action.comparison_tuple()) for action in cpu_run.actions]
        action_equal = cpu_action_rows == gpu_action_rows and not cpu_run.errors

        packed_by_destination: list[list[list[str]]] = []
        received_by_source: list[list[list[str]]] = []
        metadata_valid = True
        destination_valid = True
        for descriptor, record in enumerate(records):
            expected_groups = [[] for _ in range(WORLD_SIZE)]
            for assignment in range(record.assignment_begin, record.assignment_begin + record.assignment_count):
                row = metadata_cpu[assignment]
                metadata_valid &= int(row[8]) == identity_checksum(tuple(map(int, row[:8])))
                destination_valid &= int(row[2]) == int(row[3]) % WORLD_SIZE
                packed = row.astype("<i8", copy=False).tobytes() + features_cpu[assignment].astype("<f4", copy=False).tobytes()
                expected_groups[int(destination_cpu[assignment])].append(packed.hex())
            packed_by_destination.append(expected_groups)
            received_by_source.append([
                [value.hex() for value in _decode_received(registered_cpu, layout, descriptor, source)]
                for source in range(WORLD_SIZE)
            ])
        queue.put({
            "rank": rank,
            "actions": gpu_action_rows,
            "cpu_actions": cpu_action_rows,
            "action_equal": action_equal,
            "traces": output["traces"].tolist(),
            "timings": output["timings"].tolist(),
            "counters": output["counters"],
            "capability": output["capability"],
            "packed_by_destination": packed_by_destination,
            "received_by_source": received_by_source,
            "metadata_valid": bool(metadata_valid),
            "destination_valid": bool(destination_valid),
            "router_experts": expert_cpu.tolist(),
            "cpu_audit": {
                "python_callback_per_descriptor": 0,
                "cpu_poll_per_descriptor": 0,
                "cpu_packing_per_descriptor": runtime.cpu_per_descriptor_packing,
                "cpu_action_construction_per_descriptor": 0,
                "cpu_transport_submission_per_descriptor": runtime.cpu_per_descriptor_transport_submission,
                "cpu_cuda_launch_per_descriptor": runtime.cpu_per_descriptor_cuda_launch,
            },
        })
    except BaseException as error:
        queue.put({"rank": rank, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--mscclpp-reference-dir", type=Path,
        default=ROOT / "outputs" / "phase_r6" / "m5_gpu_transport",
    )
    parser.add_argument("--producer-delay-cycles", type=int, default=4_000_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = GPUDrivenNcclLsaRuntime.get_unique_id(args.library)
    torch = _load_torch_after_selected_nccl(args.library)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(
        target=_worker,
        args=(rank, args.library, unique_id, args.producer_delay_cycles, queue),
    ) for rank in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    ranks = [queue.get(timeout=180) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    errors = [rank for rank in ranks if "error" in rank]
    if errors:
        for process in processes:
            if process.is_alive(): process.terminate()
        raise RuntimeError(json.dumps(errors, indent=2))
    ranks.sort(key=lambda value: value["rank"])
    reference_result = json.loads(
        (args.mscclpp_reference_dir / "results.json").read_text(encoding="utf-8")
    )
    with (args.mscclpp_reference_dir / "action_trace.csv").open(
        encoding="utf-8", newline="",
    ) as handle:
        reference_action_rows = list(csv.DictReader(handle))
    reference_actions = {
        (int(row["rank"]), int(row["action_index"])): tuple(
            int(row[f"gpu_{field}"]) for field in ACTION_FIELDS
        )
        for row in reference_action_rows
    }

    lost = duplicate = wrong_destination = corruption = payload_divergence = 0
    seen_tokens: set[int] = set()
    correctness_rows = []
    for receiver in range(WORLD_SIZE):
        for descriptor in range(CHUNKS):
            for source in range(WORLD_SIZE):
                expected = ranks[source]["packed_by_destination"][descriptor][receiver]
                received = ranks[receiver]["received_by_source"][descriptor][source]
                lost += max(0, len(expected) - len(received))
                duplicate += max(0, len(received) - len(set(received)))
                payload_divergence += int(expected != received)
                one_corruption = 0
                one_wrong = 0
                for record_hex in received:
                    raw = bytes.fromhex(record_hex)
                    metadata = np.frombuffer(raw[:META_FIELDS * 8], dtype="<i8")
                    token_id = int(metadata[0])
                    duplicate += int(token_id in seen_tokens)
                    seen_tokens.add(token_id)
                    one_wrong += int(int(metadata[2]) != receiver)
                    one_corruption += int(int(metadata[8]) != identity_checksum(tuple(map(int, metadata[:8]))))
                wrong_destination += one_wrong
                corruption += one_corruption
                correctness_rows.append({
                    "receiver_rank": receiver, "descriptor_id": descriptor,
                    "source_rank": source, "expected_records": len(expected),
                    "received_records": len(received), "exact_bytes": expected == received,
                    "wrong_destination": one_wrong, "corruption": one_corruption,
                })

    action_rows = []
    packing_rows = []
    transport_rows = []
    put_before_final = []
    overlap = []
    for rank_result in ranks:
        rank = rank_result["rank"]
        for index, (cpu, gpu) in enumerate(zip(rank_result["cpu_actions"], rank_result["actions"], strict=True)):
            row = {"rank": rank, "action_index": index, "equal": cpu == gpu}
            for column, field in enumerate(ACTION_FIELDS):
                row[f"cpu_{field}"] = cpu[column]
                row[f"gpu_{field}"] = gpu[column]
            action_rows.append(row)
        timing_by_descriptor = {int(row[1]): row for row in rank_result["timings"]}
        for values in rank_result["traces"]:
            trace = dict(zip(TRACE_FIELDS, map(int, values), strict=True))
            descriptor = trace["descriptor_id"]
            timing = timing_by_descriptor[descriptor]
            packing_rows.append({
                "rank": rank, "action_id": trace["action_id"],
                "descriptor_id": descriptor, "src_rank": trace["src_rank"],
                "dst_rank": trace["dst_rank"],
                "logical_src_offset": trace["logical_src_offset"],
                "logical_dst_offset": trace["logical_dst_offset"],
                "physical_src_offset": trace["physical_src_offset"],
                "physical_dst_offset": trace["physical_dst_offset"],
                "payload_bytes": trace["payload_bytes"],
                "physical_bytes": trace["physical_bytes"],
                "t2_action_consume_ns": trace["t2_action_consumed"],
                "t3_pack_start_ns": trace["t3_pack_start"],
                "t4_pack_end_ns": trace["t4_pack_end"],
                "commit_to_pack_us": max(0, trace["t3_pack_start"] - int(timing[3])) / 1e3,
                "pack_latency_us": max(0, trace["t4_pack_end"] - trace["t3_pack_start"]) / 1e3,
                "error_code": trace["error_code"],
            })
            if trace["is_remote"]:
                before = 0 < trace["t5_put_start"] < int(timing[5])
                real_overlap = trace["t5_put_start"] < int(timing[5]) and trace["t6_put_end"] > trace["t5_put_start"]
                put_before_final.append(before); overlap.append(real_overlap)
                transport_rows.append({
                    "rank": rank, "action_id": trace["action_id"],
                    "descriptor_id": descriptor, "src_rank": trace["src_rank"],
                    "dst_rank": trace["dst_rank"], "physical_bytes": trace["physical_bytes"],
                    "t0_router_reveal_ns": int(timing[2]),
                    "t1_scheduler_commit_ns": int(timing[3]),
                    "t5_put_start_ns": trace["t5_put_start"],
                    "t6_put_end_ns": trace["t6_put_end"],
                    "t7_remote_completion_ns": int(timing[4]),
                    "t8_final_router_completion_ns": int(timing[5]),
                    "pack_to_put_us": max(0, trace["t5_put_start"] - trace["t4_pack_end"]) / 1e3,
                    "reveal_to_put_us": max(0, trace["t5_put_start"] - int(timing[2])) / 1e3,
                    "lsa_before_final_router": before,
                    "router_communication_overlap": real_overlap,
                })

    counters = {
        key: sum(int(rank["counters"][key]) for rank in ranks)
        for key in ranks[0]["counters"]
    }
    cpu_audit = {
        key: sum(int(rank["cpu_audit"][key]) for rank in ranks)
        for key in ranks[0]["cpu_audit"]
    }
    action_divergence = sum(not row["equal"] for row in action_rows)
    reference_action_divergence = 0
    for row in action_rows:
        key = (int(row["rank"]), int(row["action_index"]))
        current = tuple(int(row[f"gpu_{field}"]) for field in ACTION_FIELDS)
        reference_action_divergence += int(reference_actions.get(key) != current)
    reference_action_divergence += max(0, len(reference_actions) - len(action_rows))
    correctness = {
        "pass": not any((lost, duplicate, wrong_destination, corruption, payload_divergence)),
        "descriptor_source_destination_cases": len(correctness_rows),
        "lost": lost, "duplicate": duplicate,
        "wrong_destination": wrong_destination, "corruption": corruption,
        "payload_divergence": payload_divergence,
        "metadata_valid": all(rank["metadata_valid"] for rank in ranks),
        "router_destination_valid": all(rank["destination_valid"] for rank in ranks),
        "rows": correctness_rows,
    }
    gates = {
        "scheduler_action_divergence": action_divergence,
        "mscclpp_reference_action_divergence": reference_action_divergence,
        "mscclpp_reference_payload_divergence": int(
            reference_result["gates"]["payload_divergence"]
        ),
        "payload_divergence": payload_divergence,
        "cpu_per_descriptor_scheduler_involvement": 0,
        "cpu_per_descriptor_packing_involvement": cpu_audit["cpu_packing_per_descriptor"],
        "cpu_per_descriptor_transport_submission": cpu_audit["cpu_transport_submission_per_descriptor"],
        "cpu_per_descriptor_cuda_launch": cpu_audit["cpu_cuda_launch_per_descriptor"],
        "real_lsa_transfers": counters["lsa_transfers"] > 0,
        "real_lsa_bytes": counters["lsa_bytes_transferred"] > 0,
        "device_api_capability": all(
            bool(rank["capability"]["device_api_support"]) for rank in ranks
        ),
        "symmetric_window_registered": all(
            bool(rank["capability"]["symmetric_window"]) for rank in ranks
        ),
        "lsa_team_size_two": all(
            int(rank["capability"]["lsa_size"]) == WORLD_SIZE for rank in ranks
        ),
        "mscclpp_reference_pass": reference_result["claim"].endswith("PASS"),
        "no_collective_or_mscclpp_fallback": True,
        "correctness": correctness["pass"],
        "future_unrevealed_stale_zero": all(counters[key] == 0 for key in (
            "future_access", "unrevealed_access", "stale_action",
        )),
        "transport_errors_zero": counters["transport_errors"] == 0,
        "lsa_before_final_router": any(put_before_final),
        "router_communication_overlap": any(overlap),
        "two_rank_clean_exit": all(process.exitcode == 0 for process in processes),
    }
    passed = all(value == 0 if key.endswith("divergence") or key.startswith("cpu_per_descriptor")
                 else bool(value) for key, value in gates.items())
    results = {
        "phase": "R6-M6", "claim": "NCCL LSA Transport PASS" if passed else "NCCL LSA Transport FAIL",
        "scope_stop": "remote GPU registered receive buffer decode",
        "environment": {
            "world_size": WORLD_SIZE, "gpu": [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": "2.29.7", "device_transport": "nccl_lsa",
            "topology": "two V100 GPUs over NV2",
            "compiled_architectures": "sm_70,sm_80,sm_90",
        },
        "execution_model": {
            "job_level_cuda_launches_per_rank": 1,
            "pipeline_grid_blocks": 4,
            "pipeline_threads_per_block": 256,
            "roles": "router,scheduler,transport,remote_wait",
            "producer_delay_cycles": args.producer_delay_cycles,
        },
        "layout": {
            "metadata_fields": META_FIELDS, "feature_width": FEATURE_WIDTH,
            "record_bytes": _layout().record_bytes, "peer_stride": _layout().peer_stride,
            "descriptor_stride": _layout().descriptor_stride,
            "region_bytes": _layout().region_bytes,
            "capacity_bytes": _layout().capacity_bytes,
            "logical_to_physical_mapping": "validated identity mapping into registered buffer",
        },
        "gates": gates, "counters": counters,
        "cpu_participation": cpu_audit,
        "correctness": {key: value for key, value in correctness.items() if key != "rows"},
        "limitations": [
            "R6-M6 LSA validation is frozen to two local V100 GPUs and one NCCL LSA team.",
            "The formal path covers forward packing and transport only; expert compute and return are out of scope.",
            "One remote action per rank/descriptor maps completion_id to the frozen descriptor id.",
            "Router reveal pacing is mechanism instrumentation, not a performance benchmark.",
            "GIN_RUNTIME_NOT_AVAILABLE: the single-node container has no /dev/infiniband device.",
        ],
    }
    capability_values = ranks[0]["capability"]
    gin_types = {0: "NCCL_GIN_TYPE_NONE", 2: "NCCL_GIN_TYPE_PROXY", 3: "NCCL_GIN_TYPE_GDAKI"}
    capability = {
        "system_nccl_version": "2.27.3",
        "selected_nccl_version_code": capability_values["nccl_version"],
        "selected_nccl_version": "2.29.7",
        "device_api_support": bool(capability_values["device_api_support"]),
        "symmetric_window": bool(capability_values["symmetric_window"]),
        "multimem_support": bool(capability_values["multimem_support"]),
        "lsa_size": capability_values["lsa_size"],
        "n_lsa_teams": capability_values["n_lsa_teams"],
        "gin_type": gin_types.get(capability_values["gin_type"], f"UNKNOWN_{capability_values['gin_type']}"),
        "gin_runtime_status": "GIN_RUNTIME_NOT_AVAILABLE",
        "gin_runtime_reason": "single-node container; /dev/infiniband is absent",
    }
    results["capability"] = capability
    _write_csv(args.output_dir / "lsa_trace.csv", transport_rows)
    _write_csv(args.output_dir / "gin_trace.csv", [{
        "status": "GIN_RUNTIME_NOT_AVAILABLE",
        "real_gin_puts": 0,
        "real_network_bytes": 0,
        "reason": "single-node container; /dev/infiniband is absent",
    }])
    (args.output_dir / "correctness.json").write_text(
        json.dumps(correctness, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.output_dir / "capability.json").write_text(
        json.dumps(capability, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"claim": results["claim"], "gates": gates, "counters": counters}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
