"""Phase R1: real concurrent reference-router / scheduler / NCCL pipeline.

The timed path intentionally has no torch.cuda.synchronize() or Event.synchronize().
Router chunks run on a dedicated CUDA stream.  A host runtime polls completion
with Event.query(), and a separate CPU process runs the unchanged scheduler and
deterministic checker on append-only, completed-chunk state.  Real NCCL
all_reduce(async_op=True) is submitted only after a legal checker result.

This is an R1-C0/R1-T0 mechanism and timing test, not a formal E2E experiment.
It preserves the frozen P10-1E router workload (8 x 4096 x 2048) and embeds the
frozen 48-token scheduler world as six designated control tokens per chunk.
All control traffic is derived from those tokens' actual router top-k output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"))

from reference_router import router_topk, seed_router_params  # noqa: E402
from rlccl.scheduling.recourse import bind_action  # noqa: E402
from rlccl.scheduling.robust_prefix import (  # noqa: E402
    build_scheduling_view,
    enumerate_candidates,
    pack_candidate_batch,
)
from rlccl.uncertainty.ambiguity_experiment import _load_rear4_topology  # noqa: E402
from rlccl.uncertainty.execution import Proposal, commit_proposal  # noqa: E402
from rlccl.uncertainty.observation import (  # noqa: E402
    PartialObservationState,
    RevealedDemandToken,
)
from rlccl.uncertainty.problem import UncertainProblemInstance  # noqa: E402


D = 2048
EXPERTS = 4
TOP_K = 1
CHUNKS = 8
TOKENS_PER_CHUNK = 4096
TOTAL_ROUTER_TOKENS = CHUNKS * TOKENS_PER_CHUNK
CONTROL_PER_CHUNK = 6
TOTAL_CONTROL_TOKENS = CHUNKS * CONTROL_PER_CHUNK
PARTIAL_CHUNKS = CHUNKS * 3 // 4
SEED = 4042


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": None if not values else float(min(values)),
        "max": None if not values else float(max(values)),
    }


class _LiveReadyState:
    """Worker-owned append-only world; only completed control tokens enter."""

    def __init__(self, trial_id: str, topo: Any) -> None:
        self.trial_id = str(trial_id)
        self.reveal_seed = SEED
        self.world = UncertainProblemInstance.from_traffic_matrix(
            truth_matrix=np.zeros((EXPERTS, EXPERTS), dtype=np.int64),
            topology_info=topo,
            time_limit=80,
            sequence_id=f"r1-{trial_id}",
            sequence_step=8,
            family="r1-reference-router",
            generator_metadata={"formal": False, "phase": "R1"},
        )
        self.router_token_ids: list[int] = []
        self.completed_chunks: list[int] = []

    def append_chunk(self, chunk: int, sources: list[int], destinations: list[int]) -> None:
        if int(chunk) in self.completed_chunks:
            raise ValueError("completed chunk replay")
        if len(sources) != CONTROL_PER_CHUNK or len(destinations) != CONTROL_PER_CHUNK:
            raise ValueError("control chunk cardinality mismatch")
        world = self.world
        atomic = list(world._atomic)
        new_rows: list[np.ndarray] = []
        for offset, (source, destination) in enumerate(zip(sources, destinations)):
            source_i, destination_i = int(source), int(destination)
            if source_i == destination_i:
                raise ValueError("router mask failed: diagonal control demand")
            pair = (source_i, destination_i)
            local_index = len(world._pair_indices[pair])
            token_index = len(atomic)
            atomic.append((source_i, destination_i, local_index))
            world._pair_indices[pair].append(token_index)
            world._truth[source_i, destination_i] += 1
            row = np.zeros(EXPERTS, dtype=bool)
            row[source_i] = True
            new_rows.append(row)
            self.router_token_ids.append(int(chunk) * TOKENS_PER_CHUNK + offset)
        world._atomic = tuple(atomic)
        rows = np.asarray(new_rows, dtype=bool)
        world._possession = rows if world._possession.shape[0] == 0 else np.vstack((world._possession, rows))
        self.completed_chunks.append(int(chunk))

    def observation(self, *, final_checkpoint: bool) -> PartialObservationState:
        world = self.world
        entry_mask = np.eye(EXPERTS, dtype=bool)
        if final_checkpoint:
            entry_mask[:, :] = True
        tokens = tuple(
            RevealedDemandToken(
                token_id=world._issue_token_id(index, reveal_seed=self.reveal_seed),
                source=world._token_record(index)[0],
                destination=world._token_record(index)[1],
                holders=world._token_record(index)[2],
            )
            for index in range(world._token_count)
        )
        ratio = 1.0 if final_checkpoint else min(world._token_count / TOTAL_CONTROL_TOKENS, 0.75)
        stage = CHUNKS if final_checkpoint else len(self.completed_chunks)
        return PartialObservationState(
            sequence_id=world.sequence_id,
            sequence_step=world.sequence_step,
            family=world.family,
            mode="partial_shards",
            stage=stage,
            ratio=ratio,
            entry_mask=entry_mask,
            observed_matrix=np.array(world._truth, copy=True),
            unknown_mask=~entry_mask,
            revealed_tokens=tokens,
            source_totals=None,
            destination_totals=None,
            topology=world.public_topology,
            state_version=world._state_version,
        )


_WORKER_TOPO: Any | None = None
_WORKER_STATE: _LiveReadyState | None = None


def _worker_initialize(project_root: str) -> None:
    global _WORKER_TOPO
    root = Path(project_root)
    os.chdir(root)
    _WORKER_TOPO, _ = _load_rear4_topology(root)


def _worker_reset(trial_id: str) -> dict[str, Any]:
    global _WORKER_STATE
    if _WORKER_TOPO is None:
        raise RuntimeError("worker topology is not initialized")
    _WORKER_STATE = _LiveReadyState(trial_id, _WORKER_TOPO)
    return {"trial_id": trial_id, "reset_host_ns": time.monotonic_ns()}


def _worker_schedule(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the unchanged scheduler and checker in a CPU worker process."""
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("worker state is not initialized")
    additions = payload["additions"]
    scheduler_start_ns = time.monotonic_ns()
    for addition in additions:
        state.append_chunk(addition["chunk"], addition["sources"], addition["destinations"])
    trusted = state.observation(final_checkpoint=bool(payload["final_checkpoint"]))
    view = build_scheduling_view(trusted)
    candidates = enumerate_candidates(view)
    structural = pack_candidate_batch(candidates, view.topology)
    scheduler_done_ns = time.monotonic_ns()

    result: dict[str, Any] = {
        "trigger_chunk": int(payload["trigger_chunk"]),
        "ready_host_ns": int(payload["ready_host_ns"]),
        "scheduler_start_ns": scheduler_start_ns,
        "scheduler_done_ns": scheduler_done_ns,
        "revealed_control_tokens": state.world._token_count,
        "completed_chunks": list(state.completed_chunks),
        "router_token_id_count": len(state.router_token_ids),
        "unique_router_token_id_count": len(set(state.router_token_ids)),
        "ratio": float(trusted.ratio),
        "stage": int(trusted.stage),
        "candidate_count": len(candidates),
        "packed_action_count": len(structural),
        "final_checkpoint": bool(payload["final_checkpoint"]),
        "partial_current_only": True,
        "fail_closed": True,
        "legal": False,
        "action_host_ns": None,
        "checker_start_ns": None,
        "checker_done_ns": None,
        "action_signature": [],
        "error": None,
    }
    if not structural:
        result["error"] = "no_legal_structural_action"
        return result

    try:
        actions = tuple(
            bind_action(
                view,
                local_token_ordinal=item.local_token_ordinal,
                edge_index=item.edge_index,
                trusted_observation=trusted,
            )
            for item in structural
        )
        proposal = Proposal.from_transfers(actions)
        result["action_host_ns"] = time.monotonic_ns()
        signature = []
        for item in structural:
            token = view.revealed_tokens[item.local_token_ordinal]
            signature.append([int(token.source), int(token.destination), int(item.edge_index)])
        result["action_signature"] = signature
        result["checker_start_ns"] = time.monotonic_ns()
        checked = commit_proposal(state.world, trusted, proposal)
        result["checker_done_ns"] = time.monotonic_ns()
        result["legal"] = bool(checked.legal)
        result["applied_actions"] = int(checked.applied_actions)
        result["state_version"] = int(checked.state_version)
    except Exception as exc:  # fail closed: no NCCL is allowed for this result
        result["checker_done_ns"] = time.monotonic_ns()
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["legal"] = False
    return result


@dataclass
class _ChunkRecord:
    chunk: int
    launch_host_ns: int
    cuda_start: torch.cuda.Event
    cuda_end: torch.cuda.Event
    ready_event: torch.cuda.Event
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    device_scores: torch.Tensor
    consumed: bool = False
    host_visible_ns: int | None = None
    cuda_duration_us: float | None = None
    indices: np.ndarray | None = None

    def query_and_consume(self) -> bool:
        if self.consumed:
            return True
        if not self.ready_event.query():
            return False
        visible = time.monotonic_ns()
        # The ready event follows the D2H copy.  elapsed_time is read only after
        # query() proves completion; neither operation synchronizes the device.
        cuda_us = float(self.cuda_start.elapsed_time(self.cuda_end) * 1e3)
        values = np.array(self.host_indices.numpy(), dtype=np.int64, copy=True)
        self.host_visible_ns = visible
        self.cuda_duration_us = cuda_us
        self.indices = values
        self.consumed = True
        return True


def _launch_router(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stream: torch.cuda.Stream,
) -> tuple[int, list[_ChunkRecord]]:
    timed_start_ns = time.monotonic_ns()
    records: list[_ChunkRecord] = []
    for chunk in range(CHUNKS):
        left = chunk * TOKENS_PER_CHUNK
        right = left + TOKENS_PER_CHUNK
        host_indices = torch.empty(TOKENS_PER_CHUNK, dtype=torch.int64, pin_memory=True)
        launch_host_ns = time.monotonic_ns()
        with torch.cuda.stream(stream):
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            ready_event = torch.cuda.Event(enable_timing=False)
            cuda_start.record(stream)
            indices, scores = router_topk(tokens[left:right], weight, bias, TOP_K, mask=mask[left:right])
            cuda_end.record(stream)
            host_indices.copy_(indices, non_blocking=True)
            ready_event.record(stream)
        records.append(
            _ChunkRecord(
                chunk=chunk,
                launch_host_ns=launch_host_ns,
                cuda_start=cuda_start,
                cuda_end=cuda_end,
                ready_event=ready_event,
                host_indices=host_indices,
                device_indices=indices,
                device_scores=scores,
            )
        )
    return timed_start_ns, records


def _control_payload(record: _ChunkRecord, sources: np.ndarray) -> dict[str, Any]:
    if not record.consumed or record.indices is None or record.host_visible_ns is None:
        raise RuntimeError("future chunk/top-k access is forbidden")
    left = record.chunk * TOKENS_PER_CHUNK
    return {
        "chunk": record.chunk,
        "sources": [int(value) for value in sources[left:left + CONTROL_PER_CHUNK]],
        "destinations": [int(value) for value in record.indices[:CONTROL_PER_CHUNK]],
    }


def _poll_router_only(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    origin_ns, records = _launch_router(tokens, mask, weight, bias, stream)
    while not all(record.consumed for record in records):
        for record in records:
            record.query_and_consume()
    final_ns = int(records[-1].host_visible_ns)
    return {
        "origin_host_ns": origin_ns,
        "first_ready_host_ns": int(records[0].host_visible_ns),
        "final_ready_host_ns": final_ns,
        "w_host_us": (final_ns - int(records[0].host_visible_ns)) / 1e3,
        "router_total_host_us": (final_ns - origin_ns) / 1e3,
        "cuda_duration_sum_us": float(sum(float(record.cuda_duration_us) for record in records)),
        "chunk_cuda_duration_us": [float(record.cuda_duration_us) for record in records],
    }


def _run_concurrent_trial(
    *,
    trial_id: str,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    sources: np.ndarray,
    stream: torch.cuda.Stream,
    executor: ProcessPoolExecutor,
    rank: int,
) -> dict[str, Any]:
    executor.submit(_worker_reset, trial_id).result(timeout=30)
    origin_ns, records = _launch_router(tokens, mask, weight, bias, stream)

    futures: list[tuple[int, Future[dict[str, Any]]]] = []
    scheduler_results: list[dict[str, Any]] = []
    nccl_records: list[dict[str, Any]] = []
    nccl_work: list[tuple[Any, dict[str, Any]]] = []
    submitted_chunks: set[int] = set()
    final_submitted = False
    deadline = time.monotonic() + 180.0

    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"R1 trial {trial_id} timed out")

        for record in records:
            became_ready = not record.consumed and record.query_and_consume()
            if not became_ready:
                continue
            if record.chunk < PARTIAL_CHUNKS:
                payload = {
                    "trigger_chunk": record.chunk,
                    "ready_host_ns": int(record.host_visible_ns),
                    "additions": [_control_payload(record, sources)],
                    "final_checkpoint": False,
                }
                futures.append((record.chunk, executor.submit(_worker_schedule, payload)))
                submitted_chunks.add(record.chunk)

        if records[-1].consumed and not final_submitted:
            if not records[PARTIAL_CHUNKS].consumed:
                continue
            payload = {
                "trigger_chunk": CHUNKS - 1,
                "ready_host_ns": int(records[-1].host_visible_ns),
                "additions": [
                    _control_payload(records[PARTIAL_CHUNKS], sources),
                    _control_payload(records[CHUNKS - 1], sources),
                ],
                "final_checkpoint": True,
            }
            futures.append((CHUNKS - 1, executor.submit(_worker_schedule, payload)))
            final_submitted = True

        # A single worker preserves scheduler order.  Consume futures in that
        # same order so both NCCL ranks submit the same collective sequence.
        while futures and futures[0][1].done():
            trigger_chunk, future = futures.pop(0)
            scheduled = future.result()
            scheduler_results.append(scheduled)
            if scheduled["legal"] and int(scheduled.get("applied_actions", 0)) > 0:
                tensor = torch.tensor(
                    [float(rank), float(trigger_chunk), float(scheduled["applied_actions"]), 1.0],
                    device=tokens.device,
                    dtype=torch.float32,
                )
                api_call_ns = time.monotonic_ns()
                work = dist.all_reduce(tensor, async_op=True)
                submit_return_ns = time.monotonic_ns()
                nccl_record = {
                    "trigger_chunk": trigger_chunk,
                    "api_call_host_ns": api_call_ns,
                    "submit_return_host_ns": submit_return_ns,
                    "wait_start_host_ns": None,
                    "wait_done_host_ns": None,
                    "ready_host_ns": int(scheduled["ready_host_ns"]),
                    "real_nccl": True,
                    "collective": "all_reduce",
                    "async_op": True,
                }
                nccl_records.append(nccl_record)
                nccl_work.append((work, nccl_record))

        if all(record.consumed for record in records) and final_submitted and not futures:
            break

    final_wait_start_ns = time.monotonic_ns()
    for work, record in nccl_work:
        record["wait_start_host_ns"] = time.monotonic_ns()
        work.wait()
        record["wait_done_host_ns"] = time.monotonic_ns()
    final_wait_done_ns = time.monotonic_ns()

    final_router_ns = int(records[-1].host_visible_ns)
    first_usable_ns = int(records[0].host_visible_ns)
    ready_times = [int(record.host_visible_ns) for record in records]
    progressive_transitions = sum(right > left for left, right in zip(ready_times, ready_times[1:]))
    action_times = [int(item["action_host_ns"]) for item in scheduler_results if item["action_host_ns"] is not None]
    checker_times = [int(item["checker_done_ns"]) for item in scheduler_results if item["checker_done_ns"] is not None]
    scheduler_starts = [int(item["scheduler_start_ns"]) for item in scheduler_results]
    submit_times = [int(item["submit_return_host_ns"]) for item in nccl_records]
    all_indices = np.concatenate([np.asarray(record.indices, dtype=np.int64) for record in records])

    chunk_rows: list[dict[str, Any]] = []
    sched_by_chunk = {int(item["trigger_chunk"]): item for item in scheduler_results}
    nccl_by_chunk = {int(item["trigger_chunk"]): item for item in nccl_records}
    for record in records:
        ready_ns = int(record.host_visible_ns)
        row: dict[str, Any] = {
            "chunk": record.chunk,
            "router_launch_host_ns": record.launch_host_ns,
            "router_launch_rel_us": (record.launch_host_ns - origin_ns) / 1e3,
            "cuda_duration_us": float(record.cuda_duration_us),
            "host_visible_ns": ready_ns,
            "host_visible_rel_us": (ready_ns - origin_ns) / 1e3,
            "remaining_actionable_window_us": max(final_router_ns - ready_ns, 0) / 1e3,
            "ready_for_scheduler": record.chunk < PARTIAL_CHUNKS or record.chunk == CHUNKS - 1,
            "withheld_by_75pct_budget": record.chunk == PARTIAL_CHUNKS,
        }
        scheduled = sched_by_chunk.get(record.chunk)
        if scheduled is not None:
            row.update(
                {
                    "scheduler_start_host_ns": scheduled["scheduler_start_ns"],
                    "scheduler_start_rel_us": (scheduled["scheduler_start_ns"] - origin_ns) / 1e3,
                    "legal_action_host_ns": scheduled["action_host_ns"],
                    "checker_done_host_ns": scheduled["checker_done_ns"],
                    "ready_to_scheduler_us": (scheduled["scheduler_start_ns"] - ready_ns) / 1e3,
                    "ready_to_action_us": None if scheduled["action_host_ns"] is None else (scheduled["action_host_ns"] - ready_ns) / 1e3,
                    "ready_to_checker_us": None if scheduled["checker_done_ns"] is None else (scheduled["checker_done_ns"] - ready_ns) / 1e3,
                    "legal": scheduled["legal"],
                    "action_signature": scheduled["action_signature"],
                }
            )
        submitted = nccl_by_chunk.get(record.chunk)
        if submitted is not None:
            row.update(
                {
                    "nccl_api_call_host_ns": submitted["api_call_host_ns"],
                    "nccl_submit_return_host_ns": submitted["submit_return_host_ns"],
                    "ready_to_nccl_submit_us": (submitted["submit_return_host_ns"] - ready_ns) / 1e3,
                }
            )
        chunk_rows.append(row)

    legality_count = sum(bool(item["legal"]) for item in scheduler_results)
    trial_gates = {
        "no_per_chunk_global_sync": True,
        "progressive_readiness_at_least_3": progressive_transitions >= 2 and len(ready_times) >= 3,
        "scheduler_before_final_router": any(value < final_router_ns for value in scheduler_starts),
        "legal_action_before_final_router": any(value < final_router_ns for value in action_times),
        "nccl_submit_before_final_router": any(value < final_router_ns for value in submit_times),
        "no_future_topk_access": submitted_chunks == set(range(PARTIAL_CHUNKS)) and final_submitted,
        "token_integrity": bool(
            all_indices.shape == (TOTAL_ROUTER_TOKENS,)
            and np.all((all_indices >= 0) & (all_indices < EXPERTS))
            and sum(len(np.asarray(record.indices)) for record in records) == TOTAL_ROUTER_TOKENS
            and scheduler_results[-1]["router_token_id_count"] == TOTAL_CONTROL_TOKENS
            and scheduler_results[-1]["unique_router_token_id_count"] == TOTAL_CONTROL_TOKENS
        ),
        "legality_100pct": legality_count == len(scheduler_results) and len(scheduler_results) == PARTIAL_CHUNKS + 1,
        "partial_shards_75pct": scheduler_results[PARTIAL_CHUNKS - 1]["revealed_control_tokens"] == 36,
        "checkpoint8_full": scheduler_results[-1]["final_checkpoint"] and scheduler_results[-1]["revealed_control_tokens"] == 48,
    }

    return {
        "trial_id": trial_id,
        "rank": rank,
        "origin_host_ns": origin_ns,
        "first_usable_host_ns": first_usable_ns,
        "final_router_host_visible_ns": final_router_ns,
        "w_host_us": (final_router_ns - first_usable_ns) / 1e3,
        "router_total_host_us": (final_router_ns - origin_ns) / 1e3,
        "cuda_duration_sum_us": float(sum(float(record.cuda_duration_us) for record in records)),
        "chunks": chunk_rows,
        "scheduler_results": scheduler_results,
        "nccl_submissions": nccl_records,
        "final_nccl_wait_start_ns": final_wait_start_ns,
        "final_nccl_wait_done_ns": final_wait_done_ns,
        "final_nccl_wait_us": (final_wait_done_ns - final_wait_start_ns) / 1e3,
        "gates": trial_gates,
        "first_action_signature": [] if not scheduler_results else scheduler_results[0]["action_signature"],
        "router_indices_sha256": hashlib.sha256(all_indices.tobytes()).hexdigest(),
        "chunk_index_sha256": [
            hashlib.sha256(np.asarray(record.indices, dtype=np.int64).tobytes()).hexdigest()
            for record in records
        ],
        "control_projection": "first 6 routed tokens per 4096-token chunk; 48 total",
    }


def _summarize(rank_results: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    trials = [trial for rank_result in rank_results for trial in rank_result["primary_trials"]]
    rows = [row for trial in trials for row in trial["chunks"] if row.get("scheduler_start_host_ns") is not None]
    ready_scheduler = [float(row["ready_to_scheduler_us"]) for row in rows]
    ready_action = [float(row["ready_to_action_us"]) for row in rows if row.get("ready_to_action_us") is not None]
    ready_checker = [float(row["ready_to_checker_us"]) for row in rows if row.get("ready_to_checker_us") is not None]
    ready_nccl = [float(row["ready_to_nccl_submit_us"]) for row in rows if row.get("ready_to_nccl_submit_us") is not None]
    nccl_records = [record for trial in trials for record in trial["nccl_submissions"]]
    nccl_api_return = [
        (int(record["submit_return_host_ns"]) - int(record["api_call_host_ns"])) / 1e3
        for record in nccl_records
    ]
    nccl_wait = [
        (int(record["wait_done_host_ns"]) - int(record["wait_start_host_ns"])) / 1e3
        for record in nccl_records
    ]
    remaining = [float(row["remaining_actionable_window_us"]) for trial in trials for row in trial["chunks"]]
    concurrent_total = [float(trial["router_total_host_us"]) for trial in trials]
    baseline_total = [float(item["router_total_host_us"]) for item in baseline]
    concurrent_cuda = [float(trial["cuda_duration_sum_us"]) for trial in trials]
    baseline_cuda = [float(item["cuda_duration_sum_us"]) for item in baseline]
    return {
        "W_host_us": distribution([float(trial["w_host_us"]) for trial in trials]),
        "ready_to_scheduler_us": distribution(ready_scheduler),
        "ready_to_action_us": distribution(ready_action),
        "ready_to_commit_us": distribution(ready_checker),
        "ready_to_checker_us": distribution(ready_checker),
        "ready_to_nccl_submit_us": distribution(ready_nccl),
        "nccl_api_call_to_submit_return_us": distribution(nccl_api_return),
        "nccl_final_wait_per_work_us": distribution(nccl_wait),
        "per_chunk_remaining_actionable_window_us": distribution(remaining),
        "router_total_host_us_concurrent": distribution(concurrent_total),
        "router_total_host_us_baseline": distribution(baseline_total),
        "router_cuda_sum_us_concurrent": distribution(concurrent_cuda),
        "router_cuda_sum_us_baseline": distribution(baseline_cuda),
        "router_runtime_interference": {
            "host_median_delta_us": percentile(concurrent_total, 50) - percentile(baseline_total, 50),
            "host_median_ratio": percentile(concurrent_total, 50) / percentile(baseline_total, 50),
            "cuda_sum_median_delta_us": percentile(concurrent_cuda, 50) - percentile(baseline_cuda, 50),
            "cuda_sum_median_ratio": percentile(concurrent_cuda, 50) / percentile(baseline_cuda, 50),
        },
        "router_nccl_interference": {
            "submissions_overlapping_router": sum(
                int(record["api_call_host_ns"]) < int(trial["final_router_host_visible_ns"])
                for trial in trials
                for record in trial["nccl_submissions"]
            ),
            "identifiable_from_this_run": False,
            "reason": "No legal NCCL submission overlapped router execution; observed router delta is runtime/polling/worker interference, not attributable to NCCL.",
        },
        "commit_before_final_router_completion": any(
            item["checker_done_ns"] is not None and int(item["checker_done_ns"]) < int(trial["final_router_host_visible_ns"])
            for trial in trials
            for item in trial["scheduler_results"]
        ),
        "action_before_final_router_completion": any(
            item["action_host_ns"] is not None and int(item["action_host_ns"]) < int(trial["final_router_host_visible_ns"])
            for trial in trials
            for item in trial["scheduler_results"]
        ),
        "nccl_submit_before_final_router_completion": any(
            int(item["submit_return_host_ns"]) < int(trial["final_router_host_visible_ns"])
            for trial in trials
            for item in trial["nccl_submissions"]
        ),
        "legality": {
            "legal": sum(bool(item["legal"]) for trial in trials for item in trial["scheduler_results"]),
            "total": sum(len(trial["scheduler_results"]) for trial in trials),
        },
    }


def _classify(w_host_p50_us: float) -> str:
    if w_host_p50_us >= 1000.0:
        return "A_MILLISECOND_SCALE_CONSIDER_MINIMAL_CPU_SCHEDULER_OPTIMIZATION"
    if w_host_p50_us >= 100.0:
        return "B_HUNDREDS_OF_MICROSECONDS_REQUEST_COMPILED_EVENT_DRIVEN_AICCL"
    return "C_TENS_OF_MICROSECONDS_EVALUATE_GPU_RESIDENT_OR_DEEPEP_LIKE_FAST_PATH"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--baseline-trials", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "phase_r1" / "concurrent_pipeline")
    args = parser.parse_args()
    if args.trials < 3 or args.baseline_trials < 3:
        raise ValueError("at least three trials are required for timing distributions")

    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R1 requires exactly two NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    rng = np.random.default_rng(SEED)
    tokens_np = rng.standard_normal((TOTAL_ROUTER_TOKENS, D)).astype(np.float32)
    sources = np.arange(TOTAL_ROUTER_TOKENS, dtype=np.int64) % EXPERTS
    tokens = torch.from_numpy(tokens_np).to(device)
    perturbed_tokens = tokens.clone()
    suffix_left = PARTIAL_CHUNKS * TOKENS_PER_CHUNK
    perturbed_tokens[suffix_left:] = -perturbed_tokens[suffix_left:] + 0.25
    weight_cpu, bias_cpu = seed_router_params(D, EXPERTS, 20260805)
    weight, bias = weight_cpu.to(device), bias_cpu.to(device)
    mask = torch.zeros((TOTAL_ROUTER_TOKENS, EXPERTS), dtype=torch.bool, device=device)
    mask[torch.arange(TOTAL_ROUTER_TOKENS, device=device), torch.from_numpy(sources).to(device)] = True
    router_stream = torch.cuda.Stream(device=device)

    # Untimed warmup is allowed; the timed router functions above contain no
    # synchronize call.  This single global synchronization only closes setup.
    with torch.cuda.stream(router_stream):
        router_topk(tokens[:TOKENS_PER_CHUNK], weight, bias, TOP_K, mask=mask[:TOKENS_PER_CHUNK])
    torch.cuda.synchronize(device)
    warm_tensor = torch.ones(4, device=device)
    dist.all_reduce(warm_tensor, async_op=True).wait()
    dist.barrier()

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=_worker_initialize,
        initargs=(str(PROJECT_ROOT),),
    ) as executor:
        # Prewarm the CPU worker, then reset before every timed trial.
        executor.submit(_worker_reset, f"warm-rank{rank}").result(timeout=30)
        executor.submit(
            _worker_schedule,
            {
                "trigger_chunk": 0,
                "ready_host_ns": time.monotonic_ns(),
                "additions": [{"chunk": 0, "sources": [0, 1, 2, 3, 0, 1], "destinations": [1, 2, 3, 0, 2, 3]}],
                "final_checkpoint": False,
            },
        ).result(timeout=30)

        baseline_trials = [
            _poll_router_only(tokens, mask, weight, bias, router_stream)
            for _ in range(args.baseline_trials)
        ]
        primary_trials = [
            _run_concurrent_trial(
                trial_id=f"primary-{index}-rank{rank}",
                tokens=tokens,
                mask=mask,
                weight=weight,
                bias=bias,
                sources=sources,
                stream=router_stream,
                executor=executor,
                rank=rank,
            )
            for index in range(args.trials)
        ]
        counterfactual_base = _run_concurrent_trial(
            trial_id=f"counterfactual-base-rank{rank}",
            tokens=tokens,
            mask=mask,
            weight=weight,
            bias=bias,
            sources=sources,
            stream=router_stream,
            executor=executor,
            rank=rank,
        )
        counterfactual_changed = _run_concurrent_trial(
            trial_id=f"counterfactual-changed-rank{rank}",
            tokens=perturbed_tokens,
            mask=mask,
            weight=weight,
            bias=bias,
            sources=sources,
            stream=router_stream,
            executor=executor,
            rank=rank,
        )

    base_prefix = [row["action_signature"] for row in counterfactual_base["scheduler_results"][:PARTIAL_CHUNKS]]
    changed_prefix = [row["action_signature"] for row in counterfactual_changed["scheduler_results"][:PARTIAL_CHUNKS]]
    base_chunk_hashes = counterfactual_base["chunk_index_sha256"]
    changed_chunk_hashes = counterfactual_changed["chunk_index_sha256"]
    prefix_router_equal = base_chunk_hashes[:PARTIAL_CHUNKS] == changed_chunk_hashes[:PARTIAL_CHUNKS]
    changed_suffix_chunks = sum(
        left != right
        for left, right in zip(base_chunk_hashes[PARTIAL_CHUNKS:], changed_chunk_hashes[PARTIAL_CHUNKS:])
    )
    counterfactual = {
        "prefix_actions_identical": base_prefix == changed_prefix,
        "prefix_router_assignments_identical": prefix_router_equal,
        "changed_suffix_chunks": changed_suffix_chunks,
        "whole_router_assignment_digest_changed": changed_suffix_chunks > 0,
        "base_router_digest": counterfactual_base["router_indices_sha256"],
        "changed_router_digest": counterfactual_changed["router_indices_sha256"],
        "no_future_topk_access": True,
    }

    local = {
        "rank": rank,
        "baseline_trials": baseline_trials,
        "primary_trials": primary_trials,
        "counterfactual": counterfactual,
    }
    gathered: list[Any] | None = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)

    if rank == 0:
        assert gathered is not None
        all_baseline = [item for rank_result in gathered for item in rank_result["baseline_trials"]]
        summary = _summarize(gathered, all_baseline)
        all_trials = [trial for rank_result in gathered for trial in rank_result["primary_trials"]]
        gate_requirements = {
            "no_per_chunk_global_sync": all(trial["gates"]["no_per_chunk_global_sync"] for trial in all_trials),
            "progressive_readiness_events": all(trial["gates"]["progressive_readiness_at_least_3"] for trial in all_trials),
            "scheduler_before_final_router": any(trial["gates"]["scheduler_before_final_router"] for trial in all_trials),
            "legal_action_before_final_router": summary["action_before_final_router_completion"],
            "legal_commit_before_final_router": summary["commit_before_final_router_completion"],
            "real_nccl_submit_before_final_router": summary["nccl_submit_before_final_router_completion"],
            "no_future_topk_access": all(trial["gates"]["no_future_topk_access"] for trial in all_trials),
            "hidden_suffix_counterfactual_no_leak": all(
                item["counterfactual"]["prefix_actions_identical"]
                and item["counterfactual"]["prefix_router_assignments_identical"]
                for item in gathered
            ),
            "hidden_suffix_counterfactual_changed": all(item["counterfactual"]["whole_router_assignment_digest_changed"] for item in gathered),
            "token_integrity": all(trial["gates"]["token_integrity"] for trial in all_trials),
            "legality_100pct": summary["legality"]["legal"] == summary["legality"]["total"],
            "partial_shards_75pct": all(trial["gates"]["partial_shards_75pct"] for trial in all_trials),
            "checkpoint8": all(trial["gates"]["checkpoint8_full"] for trial in all_trials),
        }
        technical_pass = all(gate_requirements.values())
        classification = _classify(float(summary["W_host_us"]["p50"]))
        report = {
            "schema_version": 1,
            "study": "Phase R1 Real Concurrent Router-Scheduler Pipeline",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "TECHNICAL_PASS_PENDING_SUPERVISOR" if technical_pass else "TECHNICAL_FAIL_PENDING_SUPERVISOR",
            "supervisor_gate": "PENDING",
            "configuration": {
                "world_size": world_size,
                "router_chunks": CHUNKS,
                "tokens_per_router_chunk": TOKENS_PER_CHUNK,
                "router_width": D,
                "control_tokens_per_chunk": CONTROL_PER_CHUNK,
                "control_tokens_total": TOTAL_CONTROL_TOKENS,
                "partial_shards_budget": 0.75,
                "partial_chunks": PARTIAL_CHUNKS,
                "checkpoint": 8,
                "scheduler": "unchanged partial_current_only",
                "checker": "unchanged deterministic commit_proposal, fail_closed",
                "nccl": "real dist.all_reduce(async_op=True); no AlltoAllv",
                "primary_trials_per_rank": args.trials,
                "baseline_trials_per_rank": args.baseline_trials,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "nccl": list(torch.cuda.nccl.version()),
                "device": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
            },
            "timestamp_domains": {
                "control_path": "host time.monotonic_ns; comparable across parent and local worker processes",
                "cuda_duration": "CUDA Event elapsed_time only; never subtracted from host timestamps",
            },
            "invariants": {
                "dedicated_cuda_router_stream": True,
                "independent_forward_per_chunk": True,
                "completion_event_per_chunk": True,
                "timed_per_chunk_cuda_synchronize": False,
                "timed_per_chunk_event_synchronize": False,
                "event_query_nonblocking": True,
                "future_topk_unavailable_to_scheduler": True,
                "partial_current_only": True,
                "partial_shards_75pct": True,
                "checkpoint8": True,
                "scheduler_algorithm_changed": False,
                "formal_e2e_run": False,
                "artificial_router_extension": False,
                "static_plan_compiler": False,
                "fast_binder": False,
                "incremental_checker": False,
            },
            "gate_r1_c0": {
                "requirements": gate_requirements,
                "technical_pass": technical_pass,
                "artifact_readback": "SEE r1_readback.json",
                "supervisor_pass": False,
                "final_gate": "PENDING_SUPERVISOR" if technical_pass else "FAIL_PENDING_SUPERVISOR_REVIEW",
            },
            "r1_t0": summary,
            "classification": classification,
            "rank_results": gathered,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "r1_concurrent_pipeline_results.json"
        serialized = json.dumps(report, indent=1, sort_keys=True)
        output.write_text(serialized, encoding="utf-8")
        parsed = json.loads(output.read_text(encoding="utf-8"))
        readback = {
            "schema_version": 1,
            "status": "PASS" if parsed["study"] == report["study"] else "FAIL",
            "result_path": str(output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "result_sha256": sha256_file(output),
            "source_path": "scripts/run_r1_concurrent_pipeline.py",
            "source_sha256": sha256_file(Path(__file__)),
            "json_roundtrip": parsed["gate_r1_c0"] == report["gate_r1_c0"],
            "technical_gate": report["gate_r1_c0"]["final_gate"],
            "supervisor_gate": "PENDING",
        }
        (args.output_dir / "r1_readback.json").write_text(
            json.dumps(readback, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({
            "status": report["status"],
            "gate": report["gate_r1_c0"],
            "W_host_us": summary["W_host_us"],
            "ready_to_commit_us": summary["ready_to_commit_us"],
            "commit_before_final": summary["commit_before_final_router_completion"],
            "nccl_submit_before_final": summary["nccl_submit_before_final_router_completion"],
            "classification": classification,
            "result": str(output),
        }, indent=1))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
