"""R6-M1: real progressive MSCCL++ forward vs PyTorch NCCL reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rlccl.envs.problem import TopologyInfo  # noqa: E402
from rlccl.scheduling.compiled_event_driven import (  # noqa: E402
    DynamicGuard,
    FastBinder,
    IncrementalState,
    StaticPlanCompiler,
)
from rlccl.transport.mscclpp_backend import (  # noqa: E402
    COUNT_HEADER_BYTES,
    RECORD_BYTES,
    MscclppCommittedAdapter,
    RegisteredBufferLayout,
    action_payload,
)
from rlccl.transport.mscclpp_native import MscclppNativeRuntime  # noqa: E402
from rlccl.transport.reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.transport.reference_a2av import (  # noqa: E402
    PAYLOAD_FIELDS,
    ProgressivePackingState,
    RouterAssignment,
    pack_destination_layout,
    verify_received_records,
)
from rlccl.uncertainty.observation import RevealedDemandToken, TruthTokenId  # noqa: E402


WORLD_SIZE = 2
CHUNKS = int(os.environ.get("R6_M1_CHUNKS", "4"))
TOKENS_PER_CHUNK = 32
D = 16
EXPERTS = 4


def _free_port() -> int:
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return int(value.getsockname()[1])


def _assignments(rank: int, chunk: int, experts: np.ndarray) -> tuple[RouterAssignment, ...]:
    return tuple(
        RouterAssignment(
            token_id=rank * 1_000_000 + chunk * TOKENS_PER_CHUNK + offset,
            source_rank=rank,
            destination_rank=int(expert) % WORLD_SIZE,
            expert_id=int(expert),
            chunk_id=chunk,
            chunk_offset=offset,
            payload_word=(rank + 1) * 100_000 + chunk * 1000 + offset,
        )
        for offset, expert in enumerate(experts.tolist())
    )


def _scheduler_topology() -> TopologyInfo:
    edges = tuple((source, target) for source in range(EXPERTS)
                  for target in range(EXPERTS) if source != target)
    return TopologyInfo(
        EXPERTS, len(edges), np.asarray(edges, dtype=np.int64),
        np.full(len(edges), TOKENS_PER_CHUNK, dtype=np.float64), [],
        name="r6-m1-complete-static-topology",
    )


def _control_tokens(rank: int, chunk: int, values: tuple[RouterAssignment, ...]):
    return tuple(
        RevealedDemandToken(
            token_id=TruthTokenId(f"r6-m1:rank{rank}:token{item.token_id}"),
            source=(int(item.expert_id) + 1) % EXPERTS,
            destination=int(item.expert_id),
            holders=((int(item.expert_id) + 1) % EXPERTS,),
        )
        for item in values
    )


def _expected_for_rank(all_records: list[list[list[int]]], rank: int, descriptor: int):
    return {
        int(row[0]): tuple(int(value) for value in row)
        for source in range(WORLD_SIZE)
        for row in all_records[source][descriptor]
        if int(row[2]) == rank
    }


def _worker_impl(rank: int, args: dict, queue) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{args['nccl_port']}",
        rank=rank, world_size=WORLD_SIZE,
    )
    device = torch.device("cuda", rank)
    generator = torch.Generator().manual_seed(61000 + rank)
    tokens = torch.randn(CHUNKS * TOKENS_PER_CHUNK, D, generator=generator).to(device)
    weight, bias = seed_router_params(D, EXPERTS, seed=20260813)
    weight, bias = weight.to(device), bias.to(device)
    packing = ProgressivePackingState(world_size=WORLD_SIZE, source_rank=rank, max_chunks=CHUNKS)
    topology = _scheduler_topology()
    plan = StaticPlanCompiler().compile(topology)
    state = IncrementalState(
        plan, max_tokens=CHUNKS * TOKENS_PER_CHUNK, max_chunks=CHUNKS,
        sequence_id=f"r6-m1-rank{rank}", sequence_step=8,
    )
    binder = FastBinder(plan)
    guard = DynamicGuard(plan)
    layout = RegisteredBufferLayout(
        world_size=WORLD_SIZE, max_descriptors=CHUNKS,
        max_tokens_per_peer_descriptor=TOKENS_PER_CHUNK,
    )
    adapter = MscclppCommittedAdapter(rank=rank, layout=layout)
    registered = torch.zeros(layout.capacity_bytes // 8, dtype=torch.int64, device=device)
    stream = torch.cuda.Stream(device=rank)
    descriptors = []
    packed_values = []
    final_router_ns = 0
    first_issue_ns = 0
    with MscclppNativeRuntime(
        args["library"], rank=rank, device=rank, buffer_ptr=registered.data_ptr(),
        buffer_bytes=layout.capacity_bytes,
        endpoint=f"lo:127.0.0.1:{args['mscclpp_port']}",
    ) as runtime:
        for chunk in range(CHUNKS):
            router_start = time.monotonic_ns()
            left = chunk * TOKENS_PER_CHUNK
            expert_ids, _ = router_topk(tokens[left:left + TOKENS_PER_CHUNK], weight, bias, 1)
            expert_host = expert_ids.cpu().numpy()
            router_ready = time.monotonic_ns()
            if chunk == CHUNKS - 1:
                final_router_ns = router_ready
            values = _assignments(rank, chunk, expert_host)
            controls = _control_tokens(rank, chunk, values)
            packing.mark_completed(chunk, values)
            packing.reveal(chunk)
            state.stage_ready_chunk(chunk, controls)
            state.consume_pending_chunk(chunk)
            state.stage = chunk + 1
            state.ratio = (chunk + 1) / CHUNKS
            bound = binder.step(state)
            guard_decision = guard.apply(
                state, bound.proposal, require_scheduler_semantics=True,
                expected_state_version=bound.state_version,
            )
            if not guard_decision.accepted or guard_decision.applied_actions != len(values):
                raise RuntimeError(f"DynamicGuard failed closed: {guard_decision}")
            packed = pack_destination_layout(packing.build_delta_layout((chunk,)))
            actions = adapter.commit_descriptor(
                packed, descriptor_id=chunk, guard_decision=guard_decision,
                completed_chunks=range(chunk + 1), revealed_chunks=range(chunk + 1),
            )
            records = torch.from_numpy(np.array(packed.records, copy=True)).to(device)
            for destination, count in enumerate(packed.sendcounts_tokens):
                base = layout.send_offset(chunk, destination) // 8
                registered[base] = int(count)
                if count:
                    offset = packed.offsets_tokens[destination]
                    registered[base + 1:base + 1 + count * PAYLOAD_FIELDS].copy_(
                        records[offset:offset + count].reshape(-1)
                    )
            action_ready_ns = time.monotonic_ns()
            # Packing writes happen on PyTorch's current stream; make the
            # independent MSCCL++ communication stream a real consumer.
            stream.wait_stream(torch.cuda.current_stream(rank))
            for action in actions:
                if action.is_remote:
                    if not first_issue_ns:
                        first_issue_ns = time.monotonic_ns()
                    runtime.issue(
                        dst_offset=action.dst_offset, src_offset=action.src_offset,
                        bytes=action.physical_bytes, stream=stream.cuda_stream,
                    )
                else:
                    src = action.src_offset // 8
                    dst = action.dst_offset // 8
                    words = action.physical_bytes // 8
                    registered[dst:dst + words].copy_(registered[src:src + words])
            runtime.wait(stream=stream.cuda_stream)
            runtime.synchronize(stream=stream.cuda_stream)
            recv_rows = []
            for source in range(WORLD_SIZE):
                base = layout.receive_offset(chunk, source) // 8
                count = int(registered[base].item())
                if not 0 <= count <= layout.max_tokens_per_peer_descriptor:
                    nearby = registered[base:base + 12].cpu().tolist()
                    raise RuntimeError(
                        f"receiver count outside registered slot: descriptor={chunk} "
                        f"source={source} count={count} nearby={nearby} "
                        f"local_sendcounts={list(packed.sendcounts_tokens)} "
                        f"receive_byte_offset={layout.receive_offset(chunk, source)}"
                    )
                if count:
                    recv_rows.extend(
                        registered[base + 1:base + 1 + count * PAYLOAD_FIELDS]
                        .reshape(count, PAYLOAD_FIELDS).cpu().tolist()
                    )
            packed_values.append(packed.records.tolist())
            descriptors.append({
                "descriptor_id": chunk,
                "chunk_ids": [chunk],
                "router_start_ns": router_start,
                "router_ready_ns": router_ready,
                "action_ready_ns": action_ready_ns,
                "put_issue_ns": first_issue_ns if chunk == 0 else action_ready_ns,
                "sendcounts_tokens": list(packed.sendcounts_tokens),
                "offsets_tokens": list(packed.offsets_tokens),
                "actions": [action_payload(action) for action in actions],
                "scheduler_actions": len(bound.proposal.actions),
                "guard_accepted": guard_decision.accepted,
                "guard_state_version": guard_decision.state_version,
                "mscclpp_received": recv_rows,
            })

        gathered_records = [None] * WORLD_SIZE
        dist.all_gather_object(gathered_records, packed_values)

        nccl_received = []
        for packed_rows in packed_values:
            array = np.asarray(packed_rows, dtype=np.int64).reshape((-1, PAYLOAD_FIELDS))
            counts = [sum(int(row[2]) == dst for row in packed_rows) for dst in range(WORLD_SIZE)]
            send = torch.from_numpy(array.reshape(-1).copy()).to(device)
            send_counts = torch.tensor(counts, dtype=torch.int64, device=device)
            recv_counts_device = torch.empty(WORLD_SIZE, dtype=torch.int64, device=device)
            dist.all_to_all_single(recv_counts_device, send_counts)
            recv_counts = [int(value) for value in recv_counts_device.cpu().tolist()]
            recv = torch.empty(sum(recv_counts) * PAYLOAD_FIELDS, dtype=torch.int64, device=device)
            dist.all_to_all_single(
                recv, send,
                output_split_sizes=[value * PAYLOAD_FIELDS for value in recv_counts],
                input_split_sizes=[value * PAYLOAD_FIELDS for value in counts],
            )
            nccl_received.append(recv.reshape(-1, PAYLOAD_FIELDS).cpu().tolist())

        correctness = []
        for descriptor in range(CHUNKS):
            expected = _expected_for_rank(gathered_records, rank, descriptor)
            mscclpp_array = np.asarray(
                descriptors[descriptor]["mscclpp_received"], dtype=np.int64,
            ).reshape((-1, PAYLOAD_FIELDS))
            nccl_array = np.asarray(nccl_received[descriptor], dtype=np.int64).reshape((-1, PAYLOAD_FIELDS))
            ms = verify_received_records(mscclpp_array, destination_rank=rank, expected_by_token=expected)
            nc = verify_received_records(nccl_array, destination_rank=rank, expected_by_token=expected)
            correctness.append({
                "descriptor_id": descriptor, "mscclpp": ms, "nccl": nc,
                "exact_multiset_match": ms["payload_multiset_digest"] == nc["payload_multiset_digest"],
            })
        counters = runtime.counters()
    queue.put({
        "rank": rank,
        "descriptors": descriptors,
        "correctness": correctness,
        "runtime": counters,
        "adapter": adapter.counters(),
        "scheduler": {
            "runtime_bfs_calls": binder.runtime_bfs_calls,
            "full_rebuild_count": state.full_rebuild_count,
            "revealed_count": state.revealed_count,
        },
        "first_issue_ns": first_issue_ns,
        "final_router_ns": final_router_ns,
        "action_before_final_router": 0 < first_issue_ns < final_router_ns,
    })
    dist.destroy_process_group()


def _worker(rank: int, args: dict, queue) -> None:
    try:
        _worker_impl(rank, args, queue)
    except BaseException as error:
        detail = {
            "rank": rank, "worker_error": repr(error),
            "traceback": traceback.format_exc(),
        }
        Path(f"/tmp/r6_m1_rank{rank}_error.json").write_text(
            json.dumps(detail, indent=2), encoding="utf-8",
        )
        queue.put(detail)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    queue = context.Queue()
    config = {
        "library": args.library,
        "nccl_port": _free_port(),
        "mscclpp_port": _free_port(),
    }
    processes = [context.Process(target=_worker, args=(rank, config, queue)) for rank in range(2)]
    for process in processes:
        process.start()
    ranks = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=10)
    errors = [value for value in ranks if "worker_error" in value]
    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(json.dumps(errors, indent=2))
    ranks.sort(key=lambda value: value["rank"])
    correctness_rows = [row for rank in ranks for row in rank["correctness"]]
    runtime = {
        key: sum(rank["runtime"][key] for rank in ranks)
        for key in ranks[0]["runtime"]
    }
    requirements = {
        "real_mscclpp_execution": runtime["mscclpp_put_calls"] > 0,
        "real_bytes_transferred": runtime["mscclpp_bytes_transferred"] > 0,
        "correctness": all(
            row["mscclpp"]["pass"] and row["nccl"]["pass"] and row["exact_multiset_match"]
            for row in correctness_rows
        ),
        "future_unrevealed_stale_zero": all(
            rank["adapter"][key] == 0 for rank in ranks
            for key in ("future_access", "unrevealed_access", "stale_action")
        ),
        "action_before_final_router": all(rank["action_before_final_router"] for rank in ranks),
        "two_rank_clean_exit": all(process.exitcode == 0 for process in processes),
    }
    result = {
        "schema_version": 1,
        "study": "R6-M1 progressive AICCL x real MSCCL++ MemoryChannel forward",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "R6_M1_PASS" if all(requirements.values()) else "R6_M1_FAIL",
        "pass": all(requirements.values()),
        "environment": {
            "world_size": 2, "devices": [torch.cuda.get_device_name(i) for i in range(2)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "mscclpp_version": "0.9.0", "transport": "CudaIpc",
            "channel": "MemoryChannel", "gpu_arch": "sm_70",
        },
        "requirements": requirements,
        "runtime": runtime,
        "correctness": {
            "descriptor_rank_cases": len(correctness_rows),
            "lost": sum(row["mscclpp"]["lost"] for row in correctness_rows),
            "duplicate": sum(row["mscclpp"]["duplicate"] for row in correctness_rows),
            "wrong_destination": sum(row["mscclpp"]["wrong_destination"] for row in correctness_rows),
            "corruption": sum(row["mscclpp"]["corruption"] for row in correctness_rows),
            "nccl_mscclpp_digest_divergence": sum(not row["exact_multiset_match"] for row in correctness_rows),
        },
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "requirements": requirements,
        "runtime": runtime, "correctness": result["correctness"],
    }, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
