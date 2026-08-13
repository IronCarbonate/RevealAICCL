"""R3-A0/C0: router-derived incremental variable-size A2Av-T0 correctness.

This is a reference substrate and correctness preparation run.  It does not
implement production MoE packing, AlltoAllv performance evaluation, expert
GEMM, combine, DeepEP, or formal R3 E2E.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rlccl.transport.reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.compiled_event_driven import (  # noqa: E402
    DynamicGuard,
    FastBinder,
    IncrementalState,
    StaticPlanCompiler,
    structural_signature,
)
from rlccl.transport.reference_a2av import (  # noqa: E402
    PAYLOAD_FIELDS,
    ProgressivePackingState,
    RouterAssignment,
    decode_records,
    pack_destination_layout,
    payload_multiset_digest,
    verify_received_records,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from rlccl.uncertainty.execution import commit_proposal  # noqa: E402
from rlccl.uncertainty.observation import RevealedDemandToken, TruthTokenId  # noqa: E402
from scripts.run_r2_f0_integrated import (  # noqa: E402
    CHUNKS,
    D,
    EXPERTS,
    PARTIAL_CHUNKS,
    SCHEDULER_TRIGGERS,
    TOP_K,
    WAIT_TIMEOUT_NS,
    _OracleWorld,
    _action_signature,
    _load_bridge_extension,
    _old_structural,
)


TOTAL_TOKENS = 4096
CONTROL_LIMIT_PER_CHUNK = 6
A2AV_NAME = "A2Av-T0"
DEFAULT_CHUNKS = (512,) * CHUNKS
CASE_ORDER = (
    "balanced",
    "skewed",
    "all_to_one_like",
    "zero_sized_pair",
    "empty_shard",
    "single_token_shard",
    "multiple_progressive_shards",
)
CASE_SPECS = {
    "balanced": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (0.0, 0.0, 0.0, 0.0)},
    "skewed": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (100.0, 0.0, 0.0, 0.0)},
    "all_to_one_like": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (100.0, 0.0, 50.0, 0.0)},
    "zero_sized_pair": {"chunk_sizes": DEFAULT_CHUNKS, "bias": (0.0, 100.0, 0.0, 50.0)},
    "empty_shard": {
        "chunk_sizes": (512, 512, 0, 512, 512, 512, 512, 1024),
        "bias": (0.0, 0.0, 0.0, 0.0),
    },
    "single_token_shard": {
        "chunk_sizes": (512, 1, 512, 512, 512, 512, 512, 1023),
        "bias": (0.0, 0.0, 0.0, 0.0),
    },
    "multiple_progressive_shards": {
        "chunk_sizes": (128, 256, 384, 512, 640, 768, 512, 896),
        "bias": (0.0, 0.0, 0.0, 0.0),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(array.max()),
    }


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _case_inputs(case_name: str, rank: int) -> dict[str, Any]:
    spec = CASE_SPECS[case_name]
    chunk_sizes = tuple(int(value) for value in spec["chunk_sizes"])
    if len(chunk_sizes) != CHUNKS or sum(chunk_sizes) != TOTAL_TOKENS:
        raise RuntimeError(f"invalid frozen chunk sizes for {case_name}")
    case_index = CASE_ORDER.index(case_name)
    rng = np.random.default_rng(30_000 + case_index * 101 + rank)
    tokens = rng.standard_normal((TOTAL_TOKENS, D)).astype(np.float32)
    topology_sources = np.arange(TOTAL_TOKENS, dtype=np.int64) % EXPERTS
    token_ids = (
        (case_index + 1) * 100_000_000
        + rank * 10_000_000
        + np.arange(TOTAL_TOKENS, dtype=np.int64)
    )
    first = tokens[:, 0].view(np.uint32).astype(np.uint64)
    second = tokens[:, 1].view(np.uint32).astype(np.uint64)
    payload_words = ((first << np.uint64(30)) ^ second ^ token_ids.astype(np.uint64))
    payload_words &= np.uint64((1 << 62) - 1)
    offsets = np.cumsum((0,) + chunk_sizes)
    return {
        "case": case_name,
        "case_index": case_index,
        "chunk_sizes": chunk_sizes,
        "chunk_offsets": tuple(int(value) for value in offsets),
        "tokens": tokens,
        "topology_sources": topology_sources,
        "token_ids": token_ids,
        "payload_words": payload_words.astype(np.int64),
        "bias_delta": np.asarray(spec["bias"], dtype=np.float32),
    }


def _make_assignments(
    *, case_data: dict[str, Any], rank: int, chunk: int, experts: np.ndarray,
) -> tuple[RouterAssignment, ...]:
    left, right = case_data["chunk_offsets"][chunk:chunk + 2]
    if len(experts) != right - left:
        raise ValueError("router output cardinality does not match active shard")
    return tuple(
        RouterAssignment(
            token_id=int(case_data["token_ids"][left + offset]),
            source_rank=rank,
            destination_rank=int(expert) % dist.get_world_size(),
            expert_id=int(expert),
            chunk_id=chunk,
            chunk_offset=offset,
            payload_word=int(case_data["payload_words"][left + offset]),
        )
        for offset, expert in enumerate(experts.tolist())
    )


def _control_tokens(
    *, case_name: str, rank: int, chunk: int, assignments: Sequence[RouterAssignment],
    case_data: dict[str, Any],
) -> tuple[RevealedDemandToken, ...]:
    left = int(case_data["chunk_offsets"][chunk])
    values = []
    for item in assignments[:CONTROL_LIMIT_PER_CHUNK]:
        source = int(case_data["topology_sources"][left + int(item.chunk_offset)])
        destination = int(item.expert_id)
        if source == destination:
            raise ValueError("router mask failed to exclude topology source")
        values.append(RevealedDemandToken(
            token_id=TruthTokenId(f"r3-a0:{case_name}:rank{rank}:token{item.token_id}"),
            source=source,
            destination=destination,
            holders=(source,),
        ))
    return tuple(values)


def _oracle_replay_variable(
    *, topology: Any, trial_id: str, chunk_tokens: Sequence[Sequence[RevealedDemandToken]],
    selected_signatures: Sequence[Any], action_signatures: Sequence[Any],
    decisions: Sequence[Any], compiled_state: IncrementalState,
) -> dict[str, int]:
    oracle = _OracleWorld(topology, trial_id)
    candidate_divergences = action_divergences = checker_divergences = 0
    legality = holder_divergences = 0
    for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
        additions = (
            tuple(chunk_tokens[trigger]) if trigger < PARTIAL_CHUNKS
            else tuple(chunk_tokens[6]) + tuple(chunk_tokens[7])
        )
        if additions:
            oracle.append_tokens(additions)
        final = trigger == CHUNKS - 1
        observation = oracle.observation(
            stage=CHUNKS if final else trigger + 1,
            ratio=1.0 if final else (trigger + 1) / CHUNKS,
            final=final,
        )
        _, old_selected, old_proposal = _old_structural(observation)
        candidate_divergences += int(
            structural_signature(old_selected) != selected_signatures[slot]
        )
        action_divergences += int(
            _action_signature(old_proposal) != action_signatures[slot]
        )
        try:
            old_commit = commit_proposal(oracle.world, observation, old_proposal)
            old_accepted = True
            legality += int(old_commit.legal)
        except Exception:
            old_commit = None
            old_accepted = False
        decision = decisions[slot]
        checker_divergences += int(
            old_accepted != bool(decision.accepted)
            or (
                old_commit is not None
                and (
                    int(old_commit.applied_actions) != int(decision.applied_actions)
                    or int(old_commit.state_version) != int(decision.state_version)
                )
            )
        )
    for ordinal in range(compiled_state.revealed_count):
        token_id = compiled_state.token_ids[ordinal]
        if token_id is None:
            holder_divergences += 1
            continue
        private = oracle.world._public_to_private[token_id]
        old_holders = tuple(int(value) for value in np.flatnonzero(oracle.world._possession[private]))
        new_holders = tuple(int(value) for value in np.flatnonzero(compiled_state.holders[ordinal]))
        holder_divergences += int(old_holders != new_holders)
    return {
        "candidate_divergences": candidate_divergences,
        "action_divergences": action_divergences,
        "checker_divergences": checker_divergences,
        "holder_divergences": holder_divergences,
        "legal": legality,
        "total": len(SCHEDULER_TRIGGERS),
    }


def _real_variable_alltoallv(
    *, packed: Any, device: torch.device, comm_stream: torch.cuda.Stream,
) -> tuple[np.ndarray, tuple[int, ...], dict[str, float | int]]:
    """Exchange delta counts, then submit one real uneven-split NCCL AlltoAllv."""

    world_size = dist.get_world_size()
    h2d_start = time.perf_counter_ns()
    send_flat = np.array(packed.records.reshape(-1), dtype=np.int64, copy=True)
    send_device = torch.from_numpy(send_flat).to(device)
    send_counts_device = torch.tensor(
        packed.sendcounts_tokens, dtype=torch.int64, device=device,
    )
    recv_counts_device = torch.empty(world_size, dtype=torch.int64, device=device)
    h2d_done = time.perf_counter_ns()

    count_call_host_ns = time.monotonic_ns()
    count_call = time.perf_counter_ns()
    with torch.cuda.stream(comm_stream):
        count_work = dist.all_to_all_single(
            recv_counts_device, send_counts_device, async_op=True,
        )
    count_return = time.perf_counter_ns()
    count_work.wait()
    recv_counts = tuple(int(value) for value in recv_counts_device.cpu().tolist())
    count_done = time.perf_counter_ns()

    output_elements = sum(recv_counts) * PAYLOAD_FIELDS
    recv_device = torch.empty(output_elements, dtype=torch.int64, device=device)
    input_splits = list(packed.sendcounts_elements)
    output_splits = [value * PAYLOAD_FIELDS for value in recv_counts]
    payload_call_host_ns = time.monotonic_ns()
    payload_call = time.perf_counter_ns()
    with torch.cuda.stream(comm_stream):
        payload_work = dist.all_to_all_single(
            recv_device,
            send_device,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            async_op=True,
        )
    payload_return = time.perf_counter_ns()
    payload_work.wait()
    payload_done = time.perf_counter_ns()
    received = recv_device.cpu().numpy().reshape((-1, PAYLOAD_FIELDS)).copy()
    copy_done = time.perf_counter_ns()
    return received, recv_counts, {
        "h2d_us": (h2d_done - h2d_start) / 1e3,
        "count_submit_us": (count_return - count_call) / 1e3,
        "count_completion_us": (count_done - count_call) / 1e3,
        "payload_submit_us": (payload_return - payload_call) / 1e3,
        "payload_completion_us": (payload_done - payload_call) / 1e3,
        "d2h_us": (copy_done - payload_done) / 1e3,
        "count_call_host_ns": count_call_host_ns,
        "payload_call_host_ns": payload_call_host_ns,
        "input_tokens": packed.total_tokens,
        "output_tokens": int(received.shape[0]),
    }


def _run_arm(
    *, arm: str, case_name: str, rank: int, topology: Any, plan: Any, bridge: Any,
    case_data: dict[str, Any], tokens_device: torch.Tensor,
    token_chunks: Sequence[torch.Tensor], mask_chunks: Sequence[torch.Tensor],
    weight: torch.Tensor, bias: torch.Tensor, router_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
) -> dict[str, Any]:
    if arm not in ("C", "D"):
        raise ValueError("arm must be early C or delayed D")
    delayed = arm == "D"
    trial_id = f"r3-a0-{case_name}-rank{rank}"
    total_controls = sum(min(value, CONTROL_LIMIT_PER_CHUNK) for value in case_data["chunk_sizes"])
    state = IncrementalState(
        plan,
        max_tokens=max(1, total_controls),
        max_chunks=CHUNKS,
        sequence_id=trial_id,
        sequence_step=8,
    )
    binder = FastBinder(plan)
    guard = DynamicGuard(plan)
    packing = ProgressivePackingState(
        world_size=dist.get_world_size(), source_rank=rank, max_chunks=CHUNKS,
    )
    bridge.reset_all()
    completion_events = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(CHUNKS)]
    host_indices = [
        torch.empty(size, dtype=torch.int64, pin_memory=True)
        for size in case_data["chunk_sizes"]
    ]
    host_numpy = [value.numpy() for value in host_indices]
    launch_host_ns = [0] * CHUNKS
    ready_host_ns = [0] * CHUNKS
    producer_error: list[BaseException | None] = [None]
    assignments_by_chunk: list[tuple[RouterAssignment, ...] | None] = [None] * CHUNKS
    control_by_chunk: list[tuple[RevealedDemandToken, ...] | None] = [None] * CHUNKS
    bounds: list[Any] = []
    decisions: list[Any] = []
    descriptors: list[dict[str, Any]] = []
    delayed_payloads: list[tuple[Any, dict[str, Any]]] = []
    all_sent_records: list[tuple[int, ...]] = []
    all_received_arrays: list[np.ndarray] = []

    def producer() -> None:
        try:
            torch.cuda.set_device(rank)
            with torch.inference_mode():
                for chunk in range(CHUNKS):
                    launch_host_ns[chunk] = time.monotonic_ns()
                    with torch.cuda.stream(router_stream):
                        start_events[chunk].record(router_stream)
                        indices, _ = router_topk(
                            token_chunks[chunk], weight, bias, TOP_K,
                            mask=mask_chunks[chunk],
                        )
                        host_indices[chunk].copy_(indices, non_blocking=True)
                        completion_events[chunk].record(router_stream)
                    bridge.arm(chunk, completion_events[chunk].cuda_event)
        except BaseException as error:
            producer_error[0] = error

    router_origin_ns = time.monotonic_ns()
    thread = threading.Thread(target=producer, name=f"r3-router-{case_name}-{arm}-rank{rank}")
    thread.start()

    def consume_chunk(chunk: int, *, reveal: bool) -> None:
        ready_host_ns[chunk] = int(bridge.wait_ready(chunk, WAIT_TIMEOUT_NS))
        assignments = _make_assignments(
            case_data=case_data, rank=rank, chunk=chunk, experts=host_numpy[chunk],
        )
        controls = _control_tokens(
            case_name=case_name,
            rank=rank,
            chunk=chunk,
            assignments=assignments,
            case_data=case_data,
        )
        assignments_by_chunk[chunk] = assignments
        control_by_chunk[chunk] = controls
        packing.mark_completed(chunk, assignments)
        state.stage_ready_chunk(chunk, controls)
        if reveal:
            packing.reveal(chunk)
            state.consume_pending_chunk(chunk)

    def schedule() -> None:
        bound = binder.step(state)
        decision = guard.apply(
            state,
            bound.proposal,
            require_scheduler_semantics=True,
            expected_state_version=bound.state_version,
        )
        if not decision.accepted:
            raise RuntimeError(f"compiled scheduler failed closed: {decision}")
        bounds.append(bound)
        decisions.append(decision)

    def build_descriptor(chunk_ids: tuple[int, ...]) -> tuple[Any, dict[str, Any]]:
        layout_start = time.perf_counter_ns()
        layout = packing.build_delta_layout(chunk_ids)
        layout_done = time.perf_counter_ns()
        packed = pack_destination_layout(layout)
        pack_done = time.perf_counter_ns()
        meta = {
            "descriptor_index": packing.descriptor_count - 1,
            "chunk_ids": list(chunk_ids),
            "sendcounts_tokens": list(packed.sendcounts_tokens),
            "offsets_tokens": list(packed.offsets_tokens),
            "token_count": packed.total_tokens,
            "bytes": packed.total_bytes,
            "count_offset_us": (layout_done - layout_start) / 1e3,
            "packing_us": (pack_done - layout_done) / 1e3,
            "payload_multiset_digest": payload_multiset_digest(decode_records(packed.records)),
            "revealed_bitmap": packing.revealed_bitmap,
            "completed_bitmap": packing.completed_bitmap,
            "built_host_ns": time.monotonic_ns(),
        }
        return packed, meta

    def communicate(packed: Any, meta: dict[str, Any]) -> None:
        received, recvcounts, timing = _real_variable_alltoallv(
            packed=packed, device=tokens_device.device, comm_stream=comm_stream,
        )
        meta["recvcounts_tokens"] = list(recvcounts)
        meta["communication"] = timing
        descriptors.append(meta)
        all_sent_records.extend(decode_records(packed.records))
        all_received_arrays.append(received)

    for chunk in range(PARTIAL_CHUNKS):
        consume_chunk(chunk, reveal=True)
        state.stage = chunk + 1
        state.ratio = (chunk + 1) / CHUNKS
        schedule()
        packed, meta = build_descriptor((chunk,))
        if delayed:
            delayed_payloads.append((packed, meta))
        else:
            communicate(packed, meta)

    consume_chunk(6, reveal=False)
    consume_chunk(7, reveal=False)
    final_router_host_ns = int(ready_host_ns[7])
    for chunk in (6, 7):
        packing.reveal(chunk)
        state.consume_pending_chunk(chunk)
    state.stage = CHUNKS
    state.ratio = 1.0
    schedule()
    packed, meta = build_descriptor((6, 7))
    if delayed:
        delayed_payloads.append((packed, meta))
        for one_packed, one_meta in delayed_payloads:
            communicate(one_packed, one_meta)
    else:
        communicate(packed, meta)

    thread.join(timeout=120.0)
    if thread.is_alive():
        raise TimeoutError("router producer did not terminate")
    if producer_error[0] is not None:
        raise RuntimeError("router producer failed") from producer_error[0]
    torch.cuda.synchronize(tokens_device.device)
    router_done_ns = time.monotonic_ns()
    router_chunk_us = [
        float(start.elapsed_time(end) * 1e3)
        for start, end in zip(start_events, completion_events, strict=True)
    ]

    gathered_sent: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_sent, all_sent_records)
    expected_by_token = {
        int(row[0]): tuple(int(value) for value in row)
        for rank_rows in gathered_sent for row in rank_rows
        if int(row[2]) == rank
    }
    received = (
        np.concatenate(all_received_arrays, axis=0)
        if all_received_arrays else np.empty((0, PAYLOAD_FIELDS), dtype=np.int64)
    )
    verify_start = time.perf_counter_ns()
    verification = verify_received_records(
        received, destination_rank=rank, expected_by_token=expected_by_token,
    )
    verify_done = time.perf_counter_ns()
    verification["unpack_verification_us"] = (verify_done - verify_start) / 1e3

    if any(value is None for value in control_by_chunk):
        raise RuntimeError("missing completed control chunk")
    selected_signatures = [structural_signature(bound.selected) for bound in bounds]
    action_signatures = [_action_signature(bound.proposal) for bound in bounds]
    oracle = _oracle_replay_variable(
        topology=topology,
        trial_id=trial_id,
        chunk_tokens=[tuple(value or ()) for value in control_by_chunk],
        selected_signatures=selected_signatures,
        action_signatures=action_signatures,
        decisions=decisions,
        compiled_state=state,
    )
    assignment_rows = [
        item.record() for values in assignments_by_chunk for item in (values or ())
    ]
    semantic = {
        "runtime_bfs_calls": binder.runtime_bfs_calls,
        "full_rebuild_count": state.full_rebuild_count,
        "unrevealed_execution": packing.unrevealed_execution,
        "future_access": packing.future_access_attempts,
        "duplicate_dispatch": packing.duplicate_dispatch,
        "stale_dispatch": packing.stale_dispatch,
        "candidate_divergences": oracle["candidate_divergences"],
        "action_divergences": oracle["action_divergences"],
        "checker_divergences": oracle["checker_divergences"],
        "holder_divergences": oracle["holder_divergences"],
        "legal": oracle["legal"],
        "total": oracle["total"],
        "token_integrity": bool(
            verification["pass"]
            and packing.dispatched_token_count == TOTAL_TOKENS
            and len(assignment_rows) == TOTAL_TOKENS
            and len({int(row[0]) for row in assignment_rows}) == TOTAL_TOKENS
        ),
    }
    if not (
        verification["pass"]
        and semantic["token_integrity"]
        and semantic["legal"] == semantic["total"]
        and all(semantic[key] == 0 for key in (
            "runtime_bfs_calls", "full_rebuild_count", "unrevealed_execution",
            "future_access", "duplicate_dispatch", "stale_dispatch",
            "candidate_divergences", "action_divergences", "checker_divergences",
            "holder_divergences",
        ))
    ):
        raise RuntimeError(f"A0/C0 correctness gate failed: {semantic}, {verification}")
    return {
        "arm": arm,
        "case": case_name,
        "rank": rank,
        "transport": A2AV_NAME,
        "router_assignment_digest": payload_multiset_digest(assignment_rows),
        "final_sent_payload_multiset_digest": payload_multiset_digest(all_sent_records),
        "final_received_payload_multiset_digest": verification["payload_multiset_digest"],
        "total_sent_tokens": len(all_sent_records),
        "total_sent_bytes": len(all_sent_records) * PAYLOAD_FIELDS * 8,
        "total_received_tokens": int(received.shape[0]),
        "chunk_sizes": list(case_data["chunk_sizes"]),
        "chunk_launch_host_ns": launch_host_ns,
        "chunk_ready_host_ns": ready_host_ns,
        "final_router_host_ns": final_router_host_ns,
        "router_origin_host_ns": router_origin_ns,
        "router_done_host_ns": router_done_ns,
        "router_final_latency_us": (final_router_host_ns - router_origin_ns) / 1e3,
        "router_chunk_cuda_us": router_chunk_us,
        "descriptors": descriptors,
        "scheduler_selected_signatures": [
            [list(item) for item in value] for value in selected_signatures
        ],
        "scheduler_action_signatures": [
            [list(item) for item in value] for value in action_signatures
        ],
        "topk_by_chunk_digests": [
            _array_digest(np.asarray(host_numpy[chunk], dtype=np.int64)) for chunk in range(CHUNKS)
        ],
        "verification": verification,
        "semantic": semantic,
        "progressive": {
            "completed_bitmap": packing.completed_bitmap,
            "revealed_bitmap": packing.revealed_bitmap,
            "dispatched_bitmap": packing.dispatched_bitmap,
            "descriptor_count": packing.descriptor_count,
            "first_descriptor_before_final_router_host": bool(
                descriptors and int(descriptors[0]["built_host_ns"]) < final_router_host_ns
            ),
            "first_payload_call_before_final_router_host": bool(
                descriptors
                and int(descriptors[0]["communication"]["payload_call_host_ns"])
                < final_router_host_ns
            ),
            "checkpoint8_descriptor_chunks": descriptors[-1]["chunk_ids"],
        },
    }


def _warm_variable_alltoallv(device: torch.device, rank: int) -> None:
    send_token_counts = (1, 2) if rank == 0 else (3, 1)
    send_counts = torch.tensor(send_token_counts, dtype=torch.int64, device=device)
    recv_counts = torch.empty(2, dtype=torch.int64, device=device)
    dist.all_to_all_single(recv_counts, send_counts)
    recv = tuple(int(value) for value in recv_counts.cpu().tolist())
    input_tensor = torch.arange(sum(send_token_counts) * PAYLOAD_FIELDS, device=device, dtype=torch.int64)
    output_tensor = torch.empty(sum(recv) * PAYLOAD_FIELDS, device=device, dtype=torch.int64)
    dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=[value * PAYLOAD_FIELDS for value in recv],
        input_split_sizes=[value * PAYLOAD_FIELDS for value in send_token_counts],
    )
    dist.barrier()


def _case_coverage(case_name: str, rank_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    early = [item["C"] for item in rank_results]
    pair_counts = [
        int(value)
        for result in early
        for descriptor in result["descriptors"]
        for value in descriptor["sendcounts_tokens"]
    ]
    destination_totals = [0, 0]
    descriptor_totals = []
    for result in early:
        for descriptor in result["descriptors"]:
            descriptor_totals.append(int(descriptor["token_count"]))
            for destination, count in enumerate(descriptor["sendcounts_tokens"]):
                destination_totals[destination] += int(count)
    total = sum(destination_totals)
    coverage = {
        "case": case_name,
        "pair_counts": pair_counts,
        "destination_totals": destination_totals,
        "zero_sized_pairs": sum(value == 0 for value in pair_counts),
        "distinct_pair_sizes": len(set(pair_counts)),
        "descriptor_totals": descriptor_totals,
    }
    if case_name == "balanced":
        coverage["pass"] = total > 0 and abs(destination_totals[0] - destination_totals[1]) / total < 0.10
    elif case_name == "skewed":
        coverage["pass"] = total > 0 and max(destination_totals) / total > 0.75
    elif case_name == "all_to_one_like":
        coverage["pass"] = destination_totals[0] == total and total > 0
    elif case_name == "zero_sized_pair":
        coverage["pass"] = destination_totals[1] == total and coverage["zero_sized_pairs"] > 0
    elif case_name == "empty_shard":
        coverage["pass"] = descriptor_totals.count(0) >= 2
    elif case_name == "single_token_shard":
        coverage["pass"] = descriptor_totals.count(1) >= 2
    elif case_name == "multiple_progressive_shards":
        coverage["pass"] = len(set(descriptor_totals)) >= 4
    else:
        raise AssertionError("unknown case")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASE_ORDER)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    selected_cases = tuple(args.case or CASE_ORDER)
    if not args.allow_smoke and selected_cases != CASE_ORDER:
        raise ValueError("canonical R3-A0/C0 requires all frozen coverage cases in order")

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R3-A0/C0 requires exactly two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler()
    plan = compiler.compile(topology)
    if compiler.compile_bfs_sources <= 0 or not plan.proof.valid:
        raise RuntimeError("compiled scheduler static proof unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension = _load_bridge_extension(args.output_dir / f"build_rank{rank}")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed[-(rank + 1)] if allowed != [-1] else -1
    bridge = extension.IntegratedEventBridge(CHUNKS, cpu_core, rank)
    router_stream = torch.cuda.Stream(device=device, priority=0)
    comm_stream = torch.cuda.Stream(device=device, priority=0)
    _warm_variable_alltoallv(device, rank)

    local_cases: list[dict[str, Any]] = []
    try:
        for case_name in selected_cases:
            case_data = _case_inputs(case_name, rank)
            tokens_device = torch.from_numpy(case_data["tokens"]).to(device)
            weight_cpu, bias_cpu = seed_router_params(D, EXPERTS, 20260805)
            bias_cpu = bias_cpu + torch.from_numpy(case_data["bias_delta"])
            weight, bias = weight_cpu.to(device), bias_cpu.to(device)
            mask = torch.zeros((TOTAL_TOKENS, EXPERTS), dtype=torch.bool, device=device)
            mask[
                torch.arange(TOTAL_TOKENS, device=device),
                torch.from_numpy(case_data["topology_sources"]).to(device),
            ] = True
            token_chunks = tuple(
                tokens_device.narrow(
                    0,
                    case_data["chunk_offsets"][chunk],
                    case_data["chunk_sizes"][chunk],
                )
                for chunk in range(CHUNKS)
            )
            mask_chunks = tuple(
                mask.narrow(
                    0,
                    case_data["chunk_offsets"][chunk],
                    case_data["chunk_sizes"][chunk],
                )
                for chunk in range(CHUNKS)
            )
            # Warm the same reference-router shape family; this is outside diagnostics.
            with torch.inference_mode(), torch.cuda.stream(router_stream):
                router_topk(token_chunks[0], weight, bias, TOP_K, mask=mask_chunks[0])
            torch.cuda.synchronize(device)
            arms = {}
            for arm in ("C", "D"):
                dist.barrier()
                arms[arm] = _run_arm(
                    arm=arm,
                    case_name=case_name,
                    rank=rank,
                    topology=topology,
                    plan=plan,
                    bridge=bridge,
                    case_data=case_data,
                    tokens_device=tokens_device,
                    token_chunks=token_chunks,
                    mask_chunks=mask_chunks,
                    weight=weight,
                    bias=bias,
                    router_stream=router_stream,
                    comm_stream=comm_stream,
                )
            local_cases.append({"case": case_name, "C": arms["C"], "D": arms["D"]})
            dist.barrier()
    finally:
        bridge.stop()

    local = {
        "rank": rank,
        "poller_cpu_core": cpu_core,
        "poller_pinned": bool(bridge.pinned),
        "cases": local_cases,
    }
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if rank == 0:
        assert gathered is not None
        by_rank_case = {
            (item["rank"], case["case"]): case
            for item in gathered for case in item["cases"]
        }
        equivalence = []
        for rank_index in range(world_size):
            for case_name in selected_cases:
                pair = by_rank_case[(rank_index, case_name)]
                c, d = pair["C"], pair["D"]
                c_descriptors = [{
                    key: value for key, value in descriptor.items()
                    if key in (
                        "chunk_ids", "sendcounts_tokens", "offsets_tokens",
                        "token_count", "bytes", "payload_multiset_digest",
                    )
                } for descriptor in c["descriptors"]]
                d_descriptors = [{
                    key: value for key, value in descriptor.items()
                    if key in (
                        "chunk_ids", "sendcounts_tokens", "offsets_tokens",
                        "token_count", "bytes", "payload_multiset_digest",
                    )
                } for descriptor in d["descriptors"]]
                equivalence.append({
                    "rank": rank_index,
                    "case": case_name,
                    "same_router_assignments": c["router_assignment_digest"] == d["router_assignment_digest"],
                    "same_topk": c["topk_by_chunk_digests"] == d["topk_by_chunk_digests"],
                    "same_final_payload_multiset": c["final_sent_payload_multiset_digest"] == d["final_sent_payload_multiset_digest"],
                    "same_total_bytes": c["total_sent_bytes"] == d["total_sent_bytes"],
                    "same_descriptors": c_descriptors == d_descriptors,
                    "same_scheduler_actions": c["scheduler_action_signatures"] == d["scheduler_action_signatures"],
                })
        equivalence_pass = all(
            all(row[key] for key in (
                "same_router_assignments", "same_topk", "same_final_payload_multiset",
                "same_total_bytes", "same_descriptors", "same_scheduler_actions",
            ))
            for row in equivalence
        )
        coverages = [
            _case_coverage(
                case_name,
                [by_rank_case[(rank_index, case_name)] for rank_index in range(world_size)],
            )
            for case_name in selected_cases
        ]
        all_arms = [
            pair[arm]
            for pair in by_rank_case.values()
            for arm in ("C", "D")
        ]
        early_arms = [pair["C"] for pair in by_rank_case.values()]
        semantic_rows = [item["semantic"] for item in all_arms]
        verification_rows = [item["verification"] for item in all_arms]
        descriptors = [value for item in all_arms for value in item["descriptors"]]
        early_descriptors = [value for item in early_arms for value in item["descriptors"]]
        pair_sizes = [
            int(count) for descriptor in early_descriptors
            for count in descriptor["sendcounts_tokens"]
        ]
        requirements = {
            "real_variable_size_alltoallv": bool(
                descriptors
                and any(len(set(value["sendcounts_tokens"])) > 1 for value in descriptors)
                and len(set(pair_sizes)) > 1
            ),
            "sendcounts_directly_router_derived": True,
            "incremental_delta_descriptors": all(
                item["progressive"]["descriptor_count"] == len(SCHEDULER_TRIGGERS)
                for item in all_arms
            ),
            "first_communication_does_not_require_final_counts": all(
                item["progressive"]["first_descriptor_before_final_router_host"]
                and item["progressive"]["first_payload_call_before_final_router_host"]
                for item in early_arms
            ),
            "checkpoint8_preserved": all(
                item["progressive"]["checkpoint8_descriptor_chunks"] == [6, 7]
                for item in all_arms
            ),
            "runtime_bfs_zero": all(item["runtime_bfs_calls"] == 0 for item in semantic_rows),
            "full_rebuild_zero": all(item["full_rebuild_count"] == 0 for item in semantic_rows),
            "unrevealed_execution_zero": all(item["unrevealed_execution"] == 0 for item in semantic_rows),
            "future_access_zero": all(item["future_access"] == 0 for item in semantic_rows),
            "duplicate_dispatch_zero": all(item["duplicate_dispatch"] == 0 for item in semantic_rows),
            "stale_dispatch_zero": all(item["stale_dispatch"] == 0 for item in semantic_rows),
            "scheduler_checker_divergence_zero": all(
                item[key] == 0 for item in semantic_rows
                for key in (
                    "candidate_divergences", "action_divergences",
                    "checker_divergences", "holder_divergences",
                )
            ),
            "legality_100pct": all(item["legal"] == item["total"] for item in semantic_rows),
            "token_payload_integrity_100pct": all(item["token_integrity"] for item in semantic_rows),
            "receive_lost_zero": all(item["lost"] == 0 for item in verification_rows),
            "receive_duplicate_zero": all(item["duplicate"] == 0 for item in verification_rows),
            "receive_wrong_destination_zero": all(item["wrong_destination"] == 0 for item in verification_rows),
            "receive_corruption_zero": all(item["corruption"] == 0 for item in verification_rows),
            "early_delayed_semantic_equivalence": equivalence_pass,
            "coverage_all_cases": all(item["pass"] for item in coverages),
        }
        result = {
            "schema_version": 1,
            "study": "R3-A0 real variable-size AlltoAllv substrate + R3-C0 correctness",
            "status": "R3_A0_C0_COMPLETE_PENDING_SUPERVISOR",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {
                "world_size": world_size,
                "devices": [torch.cuda.get_device_name(index) for index in range(world_size)],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "python": platform.python_version(),
            },
            "frozen_protocol": {
                "transport": A2AV_NAME,
                "primitive": "torch.distributed.all_to_all_single(async_op=True) with unequal input/output split sizes",
                "count_exchange": "per-descriptor delta token counts via NCCL all_to_all_single",
                "payload_record_fields": [
                    "token_id", "source_rank", "destination_rank", "expert_id",
                    "chunk_id", "chunk_offset", "payload_word", "checksum",
                ],
                "partial_current_only": True,
                "partial_shards_ratio": 0.75,
                "checkpoint8": True,
                "router_stream_priority": 0,
                "communication_stream_priority": 0,
                "cases": list(selected_cases),
                "tokens_per_rank_case": TOTAL_TOKENS,
                "performance_is_pass_condition": False,
            },
            "requirements": requirements,
            "pass": all(requirements.values()),
            "case_coverage": coverages,
            "early_delayed_equivalence": {
                "comparisons": len(equivalence),
                "pass": equivalence_pass,
                "details": equivalence,
            },
            "correctness": {
                "arm_rank_cases": len(all_arms),
                "legal_steps": sum(int(item["legal"]) for item in semantic_rows),
                "total_steps": sum(int(item["total"]) for item in semantic_rows),
                "sent_tokens": sum(int(item["total_sent_tokens"]) for item in all_arms),
                "sent_bytes": sum(int(item["total_sent_bytes"]) for item in all_arms),
                "received_tokens": sum(int(item["total_received_tokens"]) for item in all_arms),
                "lost": sum(int(item["lost"]) for item in verification_rows),
                "duplicate": sum(int(item["duplicate"]) for item in verification_rows),
                "wrong_destination": sum(int(item["wrong_destination"]) for item in verification_rows),
                "corruption": sum(int(item["corruption"]) for item in verification_rows),
                "unrevealed_execution": sum(int(item["unrevealed_execution"]) for item in semantic_rows),
                "future_access": sum(int(item["future_access"]) for item in semantic_rows),
                "duplicate_dispatch": sum(int(item["duplicate_dispatch"]) for item in semantic_rows),
                "stale_dispatch": sum(int(item["stale_dispatch"]) for item in semantic_rows),
            },
            "traffic_distribution": {
                "early_pair_token_counts": distribution(pair_sizes),
                "distinct_pair_sizes": len(set(pair_sizes)),
                "zero_sized_pairs": sum(value == 0 for value in pair_sizes),
                "nonzero_pair_min": min((value for value in pair_sizes if value > 0), default=0),
                "nonzero_pair_max": max(pair_sizes, default=0),
                "per_case": coverages,
            },
            "diagnostics": {
                "router_final_latency_us": distribution([
                    float(item["router_final_latency_us"]) for item in all_arms
                ]),
                "router_chunk_cuda_us": distribution([
                    float(value) for item in all_arms for value in item["router_chunk_cuda_us"]
                ]),
                "count_offset_construction_us": distribution([
                    float(value["count_offset_us"]) for value in descriptors
                ]),
                "reference_packing_us": distribution([
                    float(value["packing_us"]) for value in descriptors
                ]),
                "payload_h2d_us": distribution([
                    float(value["communication"]["h2d_us"]) for value in descriptors
                ]),
                "count_exchange_completion_us": distribution([
                    float(value["communication"]["count_completion_us"]) for value in descriptors
                ]),
                "alltoallv_submit_us": distribution([
                    float(value["communication"]["payload_submit_us"]) for value in descriptors
                ]),
                "alltoallv_completion_us": distribution([
                    float(value["communication"]["payload_completion_us"]) for value in descriptors
                ]),
                "unpack_verification_us": distribution([
                    float(item["verification"]["unpack_verification_us"]) for item in all_arms
                ]),
                "total_payload_bytes": sum(int(item["total_sent_bytes"]) for item in all_arms),
            },
            "rank_results": gathered,
            "forbidden_work": {
                "expert_gemm": False,
                "return_path_combine": False,
                "production_moe_runtime": False,
                "deepep": False,
                "pccl_production": False,
                "new_transport_variant": False,
                "scheduler_semantics_changed": False,
                "partial_shards_or_checkpoint_changed": False,
                "predictor_robust_adaptive": False,
                "artificial_router_delay": False,
                "benefit_selected_workload": False,
                "formal_r3_e2e": False,
                "r3_p0": False,
            },
            "next": "stop; Supervisor decides whether R3-P0 is authorized",
        }
        if not result["pass"]:
            raise RuntimeError(f"R3-A0/C0 gate failed: {requirements}")
        output = args.output_dir / "r3_a0_c0_results.json"
        output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "status": result["status"],
            "requirements": requirements,
            "correctness": result["correctness"],
            "traffic_distribution": result["traffic_distribution"],
            "diagnostics": result["diagnostics"],
            "output": str(output),
        }, indent=1))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
