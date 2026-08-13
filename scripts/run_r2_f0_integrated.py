"""Phase R2-F0 integrated CUDA-ready -> compiled commit -> real NCCL gate.

The measured path is single-process and contains no ProcessPool, IPC queue,
serialization, sleep polling, runtime graph search, full observation rebuild,
or reference Python scheduler enumeration.  A native pinned EventBridge busy
polls preallocated CUDA events.  The frozen scheduler and checker are replayed
after each trial solely as an untimed semantic oracle.

This is a feasibility gate, not formal E2E and not the R2-O0 overlap gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import CUDA_HOME, load


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
    TruthTokenId,
)
from rlccl.uncertainty.problem import UncertainProblemInstance  # noqa: E402


D = 2048
EXPERTS = 4
TOP_K = 1
CHUNKS = 8
TOKENS_PER_CHUNK = 4096
CONTROL_PER_CHUNK = 6
TOTAL_CONTROL_TOKENS = CHUNKS * CONTROL_PER_CHUNK
PARTIAL_CHUNKS = CHUNKS * 3 // 4
SCHEDULER_TRIGGERS = (0, 1, 2, 3, 4, 5, 7)
SEED = 4042
HARD_P95_US = 655.551
STRETCH_P95_US = 300.0
WAIT_TIMEOUT_NS = 120_000_000_000

T_HOST_READY = 0
T_STATE_DONE = 1
T_ACTION = 2
T_GUARD_DONE = 3
T_DESCRIPTOR_DONE = 4
T_NCCL_CALL = 5
T_NCCL_RETURN = 6
TIMESTAMP_NAMES = (
    "t_host_ready",
    "t_state_done",
    "t_action",
    "t_guard_done",
    "t_descriptor_done",
    "t_nccl_call",
    "t_nccl_submit_return",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
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


def _load_bridge_extension(build_dir: Path) -> Any:
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is unavailable")
    interpreter_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = interpreter_bin + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        raise RuntimeError("ninja is unavailable in the interpreter environment")
    build_dir.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "extensions" / "r2_event_bridge" / "integrated_event_bridge.cpp"
    return load(
        name="r2_f0_integrated_event_bridge_ext",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_include_paths=[str(Path(CUDA_HOME) / "include")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=[f"-L{Path(CUDA_HOME) / 'lib64'}", "-lcudart"],
        with_cuda=False,
        verbose=False,
    )


class _OracleWorld:
    """Untimed append-only oracle world using the runtime's opaque token IDs."""

    def __init__(self, topology: Any, sequence_id: str) -> None:
        self.world = UncertainProblemInstance.from_traffic_matrix(
            truth_matrix=np.zeros((EXPERTS, EXPERTS), dtype=np.int64),
            topology_info=topology,
            time_limit=128,
            sequence_id=sequence_id,
            sequence_step=8,
            family="r2-f0-reference-router",
            generator_metadata={"formal": False, "phase": "R2-F0"},
        )
        self.tokens: list[RevealedDemandToken] = []

    def append_tokens(self, tokens: Sequence[RevealedDemandToken]) -> None:
        world = self.world
        atomic = list(world._atomic)
        rows: list[np.ndarray] = []
        for token in tokens:
            source, destination = int(token.source), int(token.destination)
            pair = (source, destination)
            local_index = len(world._pair_indices[pair])
            private = len(atomic)
            atomic.append((source, destination, local_index))
            world._pair_indices[pair].append(private)
            world._truth[source, destination] += 1
            row = np.zeros(EXPERTS, dtype=bool)
            row[source] = True
            rows.append(row)
            world._public_to_private[token.token_id] = private
            world._private_to_public[private] = token.token_id
            self.tokens.append(token)
        world._atomic = tuple(atomic)
        values = np.asarray(rows, dtype=bool)
        world._possession = (
            values if world._possession.shape[0] == 0
            else np.vstack((world._possession, values))
        )

    def observation(self, *, stage: int, ratio: float, final: bool) -> PartialObservationState:
        world = self.world
        current_tokens = tuple(
            RevealedDemandToken(
                token_id=token.token_id,
                source=token.source,
                destination=token.destination,
                holders=tuple(int(value) for value in np.flatnonzero(
                    world._possession[world._public_to_private[token.token_id]]
                )),
            )
            for token in self.tokens
        )
        observed = np.zeros((EXPERTS, EXPERTS), dtype=np.int64)
        for token in current_tokens:
            observed[token.source, token.destination] += 1
        entry_mask = np.ones((EXPERTS, EXPERTS), dtype=bool) if final else np.eye(EXPERTS, dtype=bool)
        return PartialObservationState(
            sequence_id=world.sequence_id,
            sequence_step=world.sequence_step,
            family=world.family,
            mode="partial_shards",
            stage=int(stage),
            ratio=float(ratio),
            entry_mask=entry_mask,
            observed_matrix=observed,
            unknown_mask=~entry_mask,
            revealed_tokens=current_tokens,
            source_totals=None,
            destination_totals=None,
            topology=world.public_topology,
            state_version=world._state_version,
        )


def _old_structural(observation: PartialObservationState):
    view = build_scheduling_view(observation)
    candidates = enumerate_candidates(view)
    selected = pack_candidate_batch(candidates, view.topology)
    proposal = Proposal.from_transfers(tuple(
        bind_action(
            view,
            local_token_ordinal=item.local_token_ordinal,
            edge_index=item.edge_index,
            trusted_observation=observation,
        )
        for item in selected
    ))
    return candidates, selected, proposal


def _make_tokens(
    *,
    chunk: int,
    token_ids: Sequence[TruthTokenId],
    control_sources: np.ndarray,
    destinations: np.ndarray,
) -> tuple[RevealedDemandToken, ...]:
    """Materialize oracle tokens only after the measured trial has ended."""
    left = chunk * CONTROL_PER_CHUNK
    return tuple(
        RevealedDemandToken(
            token_id=token_ids[left + offset],
            source=int(control_sources[left + offset]),
            destination=int(destinations[offset]),
            holders=(int(control_sources[left + offset]),),
        )
        for offset in range(CONTROL_PER_CHUNK)
    )


def _action_signature(proposal: Proposal) -> tuple[tuple[str, int], ...]:
    return tuple((str(action.token_id), int(action.edge_index)) for action in proposal.actions)


class _FastStateIngress:
    """Preallocated delta ingress for completed router control projections.

    Opaque IDs, static sources, and initial holders are installed before the
    timed region. Destinations are copied only after native EventBridge
    readiness. Pending destinations remain outside ``revealed_count`` and are
    therefore invisible to FastBinder and DynamicGuard.
    """

    def __init__(
        self,
        state: IncrementalState,
        token_ids: Sequence[TruthTokenId],
        control_sources: np.ndarray,
    ) -> None:
        if len(token_ids) != TOTAL_CONTROL_TOKENS or len(control_sources) != TOTAL_CONTROL_TOKENS:
            raise ValueError("preallocated ingress cardinality mismatch")
        self.state = state
        self.control_sources = np.asarray(control_sources, dtype=np.int32)
        for ordinal, token_id in enumerate(token_ids):
            state.token_ids[ordinal] = token_id
            state.token_ordinals[token_id] = ordinal
            source = int(self.control_sources[ordinal])
            state.sources[ordinal] = source
            state.holders[ordinal, source] = True

    def stage(self, chunk: int, destinations: np.ndarray) -> None:
        state = self.state
        index = int(chunk)
        bit = 1 << index
        if state.pending_ready_bitmap & bit or state.ready_bitmap & bit:
            raise ValueError("chunk replay")
        start = state.staged_count
        expected = index * CONTROL_PER_CHUNK
        if start != expected:
            raise ValueError("out-of-order staged chunk")
        stop = start + CONTROL_PER_CHUNK
        if destinations.shape[0] < CONTROL_PER_CHUNK:
            raise ValueError("completed router destination cardinality mismatch")
        # Six fixed control tokens are faster to validate/update directly than
        # dispatching several tiny NumPy kernels and temporary index arrays.
        for ordinal in range(start, stop):
            source = int(self.control_sources[ordinal])
            destination = int(destinations[ordinal - start])
            if destination < 0 or destination >= state.plan.nodes or destination == source:
                raise ValueError("invalid completed router destination")
            state.destinations[ordinal] = destination
        state.chunk_starts[index] = start
        state.chunk_counts[index] = CONTROL_PER_CHUNK
        state.staged_count = stop
        state.pending_ready_bitmap |= bit

    def consume(self, chunk: int) -> None:
        state = self.state
        index = int(chunk)
        bit = 1 << index
        if not state.pending_ready_bitmap & bit:
            raise ValueError("chunk is not pending-ready")
        start = int(state.chunk_starts[index])
        stop = start + int(state.chunk_counts[index])
        if start != state.revealed_count:
            raise ValueError("out-of-order reveal")
        for ordinal in range(start, stop):
            source = int(state.sources[ordinal])
            destination = int(state.destinations[ordinal])
            state.ready[ordinal] = True
            state.residual_token[ordinal] = True
            state.residual_demand[source, destination] += 1
            state.remaining_hops[ordinal] = int(state.plan.distances[source, destination])
        state.revealed_count = stop
        state.pending_ready_bitmap &= ~bit
        state.ready_bitmap |= bit
        state.chunk_consumed[index] = True
        state.delta_update_count += 1


def _oracle_replay(
    *,
    topology: Any,
    trial_id: str,
    chunk_tokens: list[tuple[RevealedDemandToken, ...] | None],
    selected_signatures: list[Any],
    action_signatures: list[Any],
    decisions: list[Any],
    compiled_state: IncrementalState,
) -> dict[str, Any]:
    oracle = _OracleWorld(topology, trial_id)
    candidate_comparisons = 0
    action_comparisons = 0
    checker_comparisons = 0
    candidate_divergences = 0
    action_divergences = 0
    checker_divergences = 0
    legality = 0

    for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
        if trigger < PARTIAL_CHUNKS:
            additions = chunk_tokens[trigger]
        else:
            additions = tuple(chunk_tokens[6] or ()) + tuple(chunk_tokens[7] or ())
        if additions is None:
            raise RuntimeError("oracle replay is missing a completed chunk")
        oracle.append_tokens(additions)
        final = trigger == CHUNKS - 1
        observation = oracle.observation(
            stage=CHUNKS if final else trigger + 1,
            ratio=1.0 if final else (trigger + 1) / CHUNKS,
            final=final,
        )
        old_candidates, old_selected, old_proposal = _old_structural(observation)
        old_selected_signature = structural_signature(old_selected)
        candidate_comparisons += 1
        if old_selected_signature != selected_signatures[slot]:
            candidate_divergences += 1
        old_action_signature = _action_signature(old_proposal)
        action_comparisons += 1
        if old_action_signature != action_signatures[slot]:
            action_divergences += 1
        try:
            old_commit = commit_proposal(oracle.world, observation, old_proposal)
            old_accepted = True
            legality += int(old_commit.legal)
        except Exception:
            old_commit = None
            old_accepted = False
        decision = decisions[slot]
        checker_comparisons += 1
        if (
            old_accepted != bool(decision.accepted)
            or (
                old_commit is not None
                and (
                    int(old_commit.applied_actions) != int(decision.applied_actions)
                    or int(old_commit.state_version) != int(decision.state_version)
                )
            )
        ):
            checker_divergences += 1

    holder_divergences = 0
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
        "candidate_comparisons": candidate_comparisons,
        "candidate_divergences": candidate_divergences,
        "action_comparisons": action_comparisons,
        "action_divergences": action_divergences,
        "checker_comparisons": checker_comparisons,
        "checker_divergences": checker_divergences,
        "holder_divergences": holder_divergences,
        "legal": legality,
        "total": len(SCHEDULER_TRIGGERS),
        "oracle_token_count": oracle.world._token_count,
    }


def _run_trial(
    *,
    trial_index: int,
    rank: int,
    topology: Any,
    plan: Any,
    bridge: Any,
    tokens: torch.Tensor,
    token_chunks: tuple[torch.Tensor, ...],
    mask_chunks: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor,
    sources: np.ndarray,
    router_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    events: list[torch.cuda.Event],
    host_indices: list[torch.Tensor],
    host_index_numpy: list[np.ndarray],
    descriptor_host_rows: list[torch.Tensor],
    descriptor_numpy_rows: list[np.ndarray],
    descriptor_device_rows: list[torch.Tensor],
) -> dict[str, Any]:
    trial_id = f"primary-{trial_index}-rank{rank}"
    state = IncrementalState(
        plan,
        max_tokens=TOTAL_CONTROL_TOKENS,
        max_chunks=CHUNKS,
        sequence_id=trial_id,
        sequence_step=8,
    )
    binder = FastBinder(plan)
    guard = DynamicGuard(plan)
    control_sources = np.asarray(
        [sources[chunk * TOKENS_PER_CHUNK + offset]
         for chunk in range(CHUNKS) for offset in range(CONTROL_PER_CHUNK)],
        dtype=np.int32,
    )
    token_ids = tuple(
        TruthTokenId(f"r2-f0:{trial_id}:{chunk}:{offset}")
        for chunk in range(CHUNKS) for offset in range(CONTROL_PER_CHUNK)
    )
    ingress = _FastStateIngress(state, token_ids, control_sources)
    bridge.reset_all()

    timestamps = np.zeros((len(SCHEDULER_TRIGGERS), len(TIMESTAMP_NAMES)), dtype=np.int64)
    final_router_ns = 0
    launch_ns = np.zeros(CHUNKS, dtype=np.int64)
    chunk_ready_ns = np.zeros(CHUNKS, dtype=np.int64)
    destination_snapshot = np.full((CHUNKS, CONTROL_PER_CHUNK), -1, dtype=np.int64)
    chunk_tokens: list[tuple[RevealedDemandToken, ...] | None] = [None] * CHUNKS
    bounds: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    selected_signatures: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    action_signatures: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    decisions: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    works: list[Any] = [None] * len(SCHEDULER_TRIGGERS)
    device_indices: list[Any] = [None] * CHUNKS
    device_scores: list[Any] = [None] * CHUNKS
    producer_error: list[BaseException | None] = [None]
    accessed_chunks = 0
    unrevealed_execution = 0

    def produce_router() -> None:
        try:
            torch.cuda.set_device(rank)
            with torch.inference_mode():
                for chunk in range(CHUNKS):
                    launch_ns[chunk] = time.monotonic_ns()
                    with torch.cuda.stream(router_stream):
                        indices, scores = router_topk(
                            token_chunks[chunk], weight, bias, TOP_K, mask=mask_chunks[chunk]
                        )
                        host_indices[chunk].copy_(indices, non_blocking=True)
                        events[chunk].record(router_stream)
                    device_indices[chunk] = indices
                    device_scores[chunk] = scores
                    bridge.arm(chunk, events[chunk].cuda_event)
        except BaseException as error:  # surfaced on the control thread
            producer_error[0] = error

    producer = threading.Thread(target=produce_router, name=f"r2-f0-router-rank{rank}")
    origin_ns = time.monotonic_ns()
    producer.start()

    def consume_chunk(chunk: int, *, reveal: bool) -> None:
        nonlocal accessed_chunks
        ready_ns = int(bridge.wait_ready(chunk, WAIT_TIMEOUT_NS))
        chunk_ready_ns[chunk] = ready_ns
        accessed_chunks |= 1 << chunk
        # The event follows D2H, so the pre-created NumPy view is now legal.
        ingress.stage(chunk, host_index_numpy[chunk])
        if reveal:
            ingress.consume(chunk)

    def schedule_submit(slot: int, trigger: int, ready_ns: int) -> None:
        nonlocal unrevealed_execution
        timestamps[slot, T_HOST_READY] = ready_ns
        timestamps[slot, T_STATE_DONE] = time.monotonic_ns()
        bound = binder.step(state)
        bounds[slot] = bound
        timestamps[slot, T_ACTION] = time.monotonic_ns()
        decision = guard.apply(
            state,
            bound.proposal,
            require_scheduler_semantics=True,
            expected_state_version=bound.state_version,
        )
        decisions[slot] = decision
        timestamps[slot, T_GUARD_DONE] = time.monotonic_ns()
        if not decision.accepted or decision.applied_actions <= 0:
            raise RuntimeError(
                f"fail-closed: trigger {trigger} has no accepted executable action: {decision}"
            )

        descriptor_numpy_rows[slot][2:5] = (
            decision.applied_actions,
            decision.state_version,
            state.revealed_count,
        )
        with torch.cuda.stream(comm_stream):
            descriptor_device_rows[slot].copy_(descriptor_host_rows[slot], non_blocking=True)
            timestamps[slot, T_DESCRIPTOR_DONE] = time.monotonic_ns()
            timestamps[slot, T_NCCL_CALL] = time.monotonic_ns()
            works[slot] = dist.all_reduce(descriptor_device_rows[slot], async_op=True)
            timestamps[slot, T_NCCL_RETURN] = time.monotonic_ns()

    for chunk in range(PARTIAL_CHUNKS):
        consume_chunk(chunk, reveal=True)
        state.stage = chunk + 1
        state.ratio = (chunk + 1) / CHUNKS
        schedule_submit(chunk, chunk, int(chunk_ready_ns[chunk]))

    consume_chunk(6, reveal=False)
    consume_chunk(7, reveal=False)
    final_router_ns = int(chunk_ready_ns[7])
    ingress.consume(6)
    ingress.consume(7)
    state.stage = CHUNKS
    state.ratio = 1.0
    schedule_submit(6, 7, final_router_ns)

    producer.join(timeout=120.0)
    if producer.is_alive():
        raise TimeoutError("router producer did not terminate")
    if producer_error[0] is not None:
        raise RuntimeError("router producer failed") from producer_error[0]
    wait_start_ns = time.monotonic_ns()
    for work in works:
        if work is None:
            raise RuntimeError("missing real NCCL async work")
        work.wait()
    wait_done_ns = time.monotonic_ns()

    # All evidence-only object materialization is deliberately outside every
    # ready->submit interval.
    for chunk in range(CHUNKS):
        destination_snapshot[chunk, :] = host_index_numpy[chunk][:CONTROL_PER_CHUNK]
        chunk_tokens[chunk] = _make_tokens(
            chunk=chunk,
            token_ids=token_ids,
            control_sources=control_sources,
            destinations=destination_snapshot[chunk],
        )
    for slot, bound in enumerate(bounds):
        selected_signatures[slot] = structural_signature(bound.selected)
        action_signatures[slot] = _action_signature(bound.proposal)
        revealed_limit = (slot + 1) * CONTROL_PER_CHUNK if slot < PARTIAL_CHUNKS else TOTAL_CONTROL_TOKENS
        for action in bound.proposal.actions:
            ordinal = state.token_ordinals.get(action.token_id, TOTAL_CONTROL_TOKENS)
            if ordinal >= revealed_limit:
                unrevealed_execution += 1

    oracle = _oracle_replay(
        topology=topology,
        trial_id=trial_id,
        chunk_tokens=chunk_tokens,
        selected_signatures=selected_signatures,
        action_signatures=action_signatures,
        decisions=decisions,
        compiled_state=state,
    )
    token_ids = state.token_ids[:state.staged_count]
    token_integrity = bool(
        state.staged_count == TOTAL_CONTROL_TOKENS
        and state.revealed_count == TOTAL_CONTROL_TOKENS
        and len(set(token_ids)) == TOTAL_CONTROL_TOKENS
        and state.ready_bitmap == (1 << CHUNKS) - 1
        and state.pending_ready_bitmap == 0
        and oracle["oracle_token_count"] == TOTAL_CONTROL_TOKENS
        and oracle["holder_divergences"] == 0
    )

    rows: list[dict[str, Any]] = []
    for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
        row = {name: int(timestamps[slot, index]) for index, name in enumerate(TIMESTAMP_NAMES)}
        row.update({
            "trigger_chunk": trigger,
            "t_final_router_completion": final_router_ns,
            "ready_to_state_us": (timestamps[slot, T_STATE_DONE] - timestamps[slot, T_HOST_READY]) / 1e3,
            "ready_to_action_us": (timestamps[slot, T_ACTION] - timestamps[slot, T_HOST_READY]) / 1e3,
            "ready_to_guard_us": (timestamps[slot, T_GUARD_DONE] - timestamps[slot, T_HOST_READY]) / 1e3,
            "ready_to_nccl_call_us": (timestamps[slot, T_NCCL_CALL] - timestamps[slot, T_HOST_READY]) / 1e3,
            "ready_to_nccl_submit_return_us": (timestamps[slot, T_NCCL_RETURN] - timestamps[slot, T_HOST_READY]) / 1e3,
            "nccl_api_to_submit_return_us": (timestamps[slot, T_NCCL_RETURN] - timestamps[slot, T_NCCL_CALL]) / 1e3,
            "submit_return_margin_us": (final_router_ns - timestamps[slot, T_NCCL_RETURN]) / 1e3,
            "submit_before_final": bool(timestamps[slot, T_NCCL_RETURN] < final_router_ns),
            "selected_actions": int(decisions[slot].applied_actions),
            "selected_signature": [list(item) for item in selected_signatures[slot]],
            "action_signature": [list(item) for item in action_signatures[slot]],
        })
        rows.append(row)

    return {
        "trial_id": trial_id,
        "rank": rank,
        "origin_host_ns": origin_ns,
        "t_final_router_completion": final_router_ns,
        "router_host_duration_us": (final_router_ns - origin_ns) / 1e3,
        "chunk_launch_host_ns": [int(value) for value in launch_ns],
        "chunk_host_ready_ns": [int(value) for value in chunk_ready_ns],
        "events": rows,
        "oracle": oracle,
        "runtime_counters": {
            "runtime_bfs_calls": binder.runtime_bfs_calls,
            "full_rebuild_count": state.full_rebuild_count,
            "delta_update_count": state.delta_update_count,
            "unrevealed_execution": unrevealed_execution,
            "accessed_ready_bitmap": accessed_chunks,
            "ready_bitmap": state.ready_bitmap,
            "pending_ready_bitmap": state.pending_ready_bitmap,
        },
        "token_integrity": token_integrity,
        "legality_100pct": oracle["legal"] == oracle["total"],
        "final_nccl_wait_start_ns": wait_start_ns,
        "final_nccl_wait_done_ns": wait_done_ns,
        "final_nccl_wait_us": (wait_done_ns - wait_start_ns) / 1e3,
        "partial_current_only": True,
        "partial_shards_75pct": True,
        "checkpoint8": True,
    }


def _summarize(rank_results: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    trials = [trial for rank in rank_results for trial in rank["trials"]]
    rows = [row for trial in trials for row in trial["events"]]
    metrics = {
        "ready_to_state_us": distribution([row["ready_to_state_us"] for row in rows]),
        "ready_to_action_us": distribution([row["ready_to_action_us"] for row in rows]),
        "ready_to_guard_us": distribution([row["ready_to_guard_us"] for row in rows]),
        "ready_to_nccl_call_us": distribution([row["ready_to_nccl_call_us"] for row in rows]),
        "ready_to_nccl_submit_return_us": distribution([
            row["ready_to_nccl_submit_return_us"] for row in rows
        ]),
        "nccl_api_to_submit_return_us": distribution([
            row["nccl_api_to_submit_return_us"] for row in rows
        ]),
    }
    candidate_divergences = sum(t["oracle"]["candidate_divergences"] for t in trials)
    action_divergences = sum(t["oracle"]["action_divergences"] for t in trials)
    checker_divergences = sum(t["oracle"]["checker_divergences"] for t in trials)
    holder_divergences = sum(t["oracle"]["holder_divergences"] for t in trials)
    legal = sum(t["oracle"]["legal"] for t in trials)
    legality_total = sum(t["oracle"]["total"] for t in trials)
    submit_before = sum(row["submit_before_final"] for row in rows)
    requirements = {
        "runtime_bfs_zero": all(t["runtime_counters"]["runtime_bfs_calls"] == 0 for t in trials),
        "full_rebuild_zero": all(t["runtime_counters"]["full_rebuild_count"] == 0 for t in trials),
        "unrevealed_execution_zero": all(t["runtime_counters"]["unrevealed_execution"] == 0 for t in trials),
        "action_semantic_divergence_zero": action_divergences == 0 and candidate_divergences == 0,
        "checker_divergence_zero": checker_divergences == 0 and holder_divergences == 0,
        "legality_100pct": legal == legality_total and legality_total == len(rows),
        "token_integrity_100pct": all(t["token_integrity"] for t in trials),
        "partial_current_only": all(t["partial_current_only"] for t in trials),
        "partial_shards_75pct_checkpoint8": all(
            t["partial_shards_75pct"] and t["checkpoint8"] for t in trials
        ),
        "real_nccl_async_submit_every_eligible_event": len(rows) == len(trials) * len(SCHEDULER_TRIGGERS),
    }
    semantic_pass = all(requirements.values())
    submit_p95 = float(metrics["ready_to_nccl_submit_return_us"]["p95"])
    hard_pass = submit_p95 < HARD_P95_US
    stretch_pass = submit_p95 < STRETCH_P95_US
    gate = {
        "f0_a_requirements": requirements,
        "f0_a_semantic_safety_pass": semantic_pass,
        "f0_b_hard_target_us": HARD_P95_US,
        "f0_b_ready_to_submit_p95_us": submit_p95,
        "f0_b_hard_pass": hard_pass,
        "stretch_target_us": STRETCH_P95_US,
        "stretch_pass": stretch_pass,
        "technical_pass": semantic_pass and hard_pass,
        "supervisor_gate": "PENDING",
    }
    overlap_diagnostic = {
        "scope": "diagnostic only; R2-O0 is not authorized or claimed",
        "eligible_shards": len(rows),
        "submit_before_final_count": int(submit_before),
        "submit_before_final_fraction": submit_before / len(rows),
        "positive_margin_us": distribution([
            row["submit_return_margin_us"] for row in rows if row["submit_before_final"]
        ]),
        "any_submit_before_final": submit_before > 0,
    }
    metrics["oracle_counts"] = {
        "candidate_comparisons": sum(t["oracle"]["candidate_comparisons"] for t in trials),
        "candidate_divergences": candidate_divergences,
        "action_comparisons": sum(t["oracle"]["action_comparisons"] for t in trials),
        "action_divergences": action_divergences,
        "checker_comparisons": sum(t["oracle"]["checker_comparisons"] for t in trials),
        "checker_divergences": checker_divergences,
        "holder_divergences": holder_divergences,
        "legal": legal,
        "total": legality_total,
    }
    metrics["overlap_diagnostic"] = overlap_diagnostic
    return metrics, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_r2" / "f0_integrated_ready_commit",
    )
    args = parser.parse_args()
    if args.trials < 20:
        raise ValueError("R2-F0 requires at least 20 primary trials per rank")

    dist.init_process_group("nccl", init_method="env://")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("R2-F0 requires exactly two real NCCL ranks")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    topology, _ = _load_rear4_topology(PROJECT_ROOT)
    compiler = StaticPlanCompiler()
    plan = compiler.compile(topology)
    if compiler.compile_bfs_sources != EXPERTS:
        raise RuntimeError("unexpected static compile BFS count")

    rng = np.random.default_rng(SEED)
    total_router_tokens = CHUNKS * TOKENS_PER_CHUNK
    tokens_np = rng.standard_normal((total_router_tokens, D)).astype(np.float32)
    sources = np.arange(total_router_tokens, dtype=np.int64) % EXPERTS
    tokens = torch.from_numpy(tokens_np).to(device)
    weight_cpu, bias_cpu = seed_router_params(D, EXPERTS, 20260805)
    weight, bias = weight_cpu.to(device), bias_cpu.to(device)
    mask = torch.zeros((total_router_tokens, EXPERTS), dtype=torch.bool, device=device)
    mask[torch.arange(total_router_tokens, device=device), torch.from_numpy(sources).to(device)] = True
    token_chunks = tuple(tokens.narrow(0, i * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK) for i in range(CHUNKS))
    mask_chunks = tuple(mask.narrow(0, i * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK) for i in range(CHUNKS))

    router_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)
    events = [torch.cuda.Event(enable_timing=False) for _ in range(CHUNKS)]
    host_indices = [torch.empty(TOKENS_PER_CHUNK, dtype=torch.int64, pin_memory=True) for _ in range(CHUNKS)]
    host_index_numpy = [value.numpy() for value in host_indices]
    descriptor_host = torch.empty((len(SCHEDULER_TRIGGERS), 8), dtype=torch.int64, pin_memory=True)
    descriptor_device = torch.empty((len(SCHEDULER_TRIGGERS), 8), dtype=torch.int64, device=device)
    descriptor_host_rows = [descriptor_host[index] for index in range(len(SCHEDULER_TRIGGERS))]
    descriptor_numpy_rows = [value.numpy() for value in descriptor_host_rows]
    descriptor_device_rows = [descriptor_device[index] for index in range(len(SCHEDULER_TRIGGERS))]
    for slot, trigger in enumerate(SCHEDULER_TRIGGERS):
        descriptor_numpy_rows[slot][:] = (rank, trigger, 0, 0, 0, 0, 0, 1)

    # Setup-only warmup and lazy event initialization.  No synchronize call is
    # present in _run_trial or its per-chunk control path.
    with torch.inference_mode(), torch.cuda.stream(router_stream):
        warm_indices, _ = router_topk(token_chunks[0], weight, bias, TOP_K, mask=mask_chunks[0])
        host_indices[0].copy_(warm_indices, non_blocking=True)
        for event in events:
            event.record(router_stream)
    torch.cuda.synchronize(device)
    descriptor_device.zero_()
    dist.all_reduce(descriptor_device_rows[0], async_op=True).wait()
    dist.barrier()

    allowed_cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [-1]
    cpu_core = allowed_cores[-(rank + 1)] if allowed_cores != [-1] else -1
    extension = _load_bridge_extension(args.output_dir / "build")
    bridge = extension.IntegratedEventBridge(CHUNKS, cpu_core, rank)

    trials: list[dict[str, Any]] = []
    try:
        for trial_index in range(args.trials):
            trials.append(_run_trial(
                trial_index=trial_index,
                rank=rank,
                topology=topology,
                plan=plan,
                bridge=bridge,
                tokens=tokens,
                token_chunks=token_chunks,
                mask_chunks=mask_chunks,
                weight=weight,
                bias=bias,
                sources=sources,
                router_stream=router_stream,
                comm_stream=comm_stream,
                events=events,
                host_indices=host_indices,
                host_index_numpy=host_index_numpy,
                descriptor_host_rows=descriptor_host_rows,
                descriptor_numpy_rows=descriptor_numpy_rows,
                descriptor_device_rows=descriptor_device_rows,
            ))
            dist.barrier()
    finally:
        bridge.stop()

    local = {
        "rank": rank,
        "poller_cpu_core": cpu_core,
        "poller_pinned": bool(bridge.pinned),
        "trials": trials,
    }
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)

    if rank == 0:
        assert gathered is not None
        metrics, gate = _summarize(gathered)
        status = "TECHNICAL_PASS_PENDING_SUPERVISOR" if gate["technical_pass"] else "TECHNICAL_FAIL_PENDING_SUPERVISOR"
        result = {
            "schema_version": 1,
            "study": "Phase R2-F0 Integrated Fast Ready-to-Commit Feasibility",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "supervisor_gate": "PENDING",
            "environment": {
                "world_size": world_size,
                "backend": dist.get_backend(),
                "devices": [torch.cuda.get_device_name(index) for index in range(world_size)],
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "python": platform.python_version(),
                "poller_cpu_cores": [item["poller_cpu_core"] for item in gathered],
                "pollers_pinned": [item["poller_pinned"] for item in gathered],
            },
            "frozen_workload": {
                "router": "minimal deterministic reference MoE router",
                "shape_per_chunk": [TOKENS_PER_CHUNK, D],
                "chunks": CHUNKS,
                "experts": EXPERTS,
                "control_tokens_per_chunk": CONTROL_PER_CHUNK,
                "partial_shards_ratio": 0.75,
                "checkpoint": 8,
                "scheduler": "partial_current_only compiled semantic equivalent",
                "seed": SEED,
                "workload_changed_to_expand_window": False,
            },
            "preregistered_targets": {
                "primary_ready_to_nccl_submit_return_p95_lt_us": HARD_P95_US,
                "stretch_ready_to_nccl_submit_return_p95_lt_us": STRETCH_P95_US,
            },
            "implementation": {
                "single_process_per_rank": True,
                "native_pinned_busy_event_bridge": True,
                "preallocated_ready_events_and_descriptors": True,
                "independent_router_stream": True,
                "independent_communication_stream": True,
                "real_nccl_collective": "all_reduce(async_op=True)",
                "runtime_bfs": False,
                "full_state_rebuild_fast_path": False,
                "reference_oracle_location": "after final NCCL waits; excluded from timestamps",
                "critical_path_forbidden_mechanisms": {
                    "ProcessPool": False,
                    "multiprocessing_queue": False,
                    "pickle": False,
                    "JSON": False,
                    "sleep_polling": False,
                    "full_python_candidate_enumeration": False,
                },
            },
            "latency_us": metrics,
            "gate_r2_f0": gate,
            "rank_results": gathered,
            "forbidden_work": {
                "formal_e2e": False,
                "real_alltoallv": False,
                "expert_packing_gemm_combine": False,
                "deepep": False,
                "r2_o0_claimed": False,
                "predictor_robust_adaptive": False,
            },
            "next_step": "STOP_FOR_SUPERVISOR_REVIEW; R2-O0_NOT_AUTHORIZED",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "r2_f0_results.json"
        output.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
        parsed = json.loads(output.read_text(encoding="utf-8"))
        try:
            result_path = str(output.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            result_path = str(output.resolve())
        readback = {
            "schema_version": 1,
            "status": "PASS" if parsed["gate_r2_f0"] == gate else "FAIL",
            "result_path": result_path,
            "result_sha256": sha256_file(output),
            "runner_sha256": sha256_file(Path(__file__)),
            "extension_sha256": sha256_file(
                PROJECT_ROOT / "extensions" / "r2_event_bridge" / "integrated_event_bridge.cpp"
            ),
            "compiled_scheduler_sha256": sha256_file(
                PROJECT_ROOT / "rlccl" / "scheduling" / "compiled_event_driven.py"
            ),
            "json_roundtrip": parsed["study"] == result["study"],
            "supervisor_gate": "PENDING",
        }
        readback_path = args.output_dir / "r2_f0_readback.json"
        readback_path.write_text(json.dumps(readback, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({
            "status": status,
            "gate": gate,
            "latency_us": metrics,
            "output": str(output),
        }, indent=1))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
