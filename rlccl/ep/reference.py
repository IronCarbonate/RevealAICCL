"""CPU construction oracle for the M7 commit boundary and dispatch payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rlccl.ep.common.progressive_ep_schema import CommitPeerPlan, DescriptorCommit
from rlccl.ep.layout import ProgressiveDispatchLayout
from rlccl.scheduler.common.compiled_plan import CompiledPlanBlob
from rlccl.scheduler.common.scheduler_schema import CommittedAction, RevealRecord
from rlccl.scheduler.cpu.reference import CPUSchedulerShadow


@dataclass(frozen=True, slots=True)
class CommitReference:
    commits: tuple[DescriptorCommit, ...]
    peer_plans: tuple[CommitPeerPlan, ...]
    shadow_actions: tuple[CommittedAction, ...]
    destinations: tuple[int, ...]


def build_commit_reference(
    plan: CompiledPlanBlob, layout: ProgressiveDispatchLayout,
    records: Iterable[RevealRecord], topk_idx: np.ndarray,
    experts_per_rank: int,
) -> CommitReference:
    """Expand each reveal once while retaining the frozen M6 action shadow."""
    records = tuple(records)
    if topk_idx.ndim != 2:
        raise ValueError("topk_idx must have shape [tokens, topk]")
    num_topk = int(topk_idx.shape[1])
    flattened = tuple(map(int, topk_idx.reshape(-1).tolist()))
    destinations = tuple(expert // experts_per_rank for expert in flattened)
    shadow_run = CPUSchedulerShadow(plan).run(records, destinations)
    if shadow_run.errors:
        raise ValueError(f"reference scheduler rejected reveals: {shadow_run.errors}")
    actions = {(a.descriptor_id, a.dst_rank): a for a in shadow_run.actions}
    commits: list[DescriptorCommit] = []
    peers: list[CommitPeerPlan] = []
    for commit_id, record in enumerate(records):
        if record.assignment_count != record.token_count * num_topk:
            raise ValueError("assignment count is not token_count * num_topk")
        counts = [0] * layout.world_size
        for assignment in range(
            record.assignment_begin, record.assignment_begin + record.assignment_count,
        ):
            counts[destinations[assignment]] += 1
        mask = sum((1 << dst) for dst, count in enumerate(counts) if count)
        commits.append(DescriptorCommit(
            commit_id, record.descriptor_id, record.chunk_id,
            record.reveal_epoch, record.token_begin, record.token_count,
            record.assignment_begin, record.assignment_count, mask, 1,
        ))
        for dst, count in enumerate(counts):
            action = actions.get((record.descriptor_id, dst))
            peers.append(CommitPeerPlan(
                record.descriptor_id, dst, count,
                0 if action is None else action.route_id,
                0 if not count else layout.staging_offset(record.descriptor_id, dst),
                0 if not count else layout.recv_offset(record.descriptor_id, plan.config.source_rank),
                0 if not count else 1,
            ))
    return CommitReference(
        tuple(commits), tuple(peers), shadow_run.actions, destinations,
    )


def expected_rank_records(
    *, source_rank: int, destination_rank: int,
    records: Iterable[RevealRecord], x: np.ndarray,
    topk_idx: np.ndarray, topk_weights: np.ndarray,
    experts_per_rank: int,
) -> list[tuple[tuple[int, int, int, int, int, int, float], bytes]]:
    """Return the exact metadata/features that one source sends to one rank."""
    output = []
    num_topk = int(topk_idx.shape[1])
    for record in records:
        for token in range(record.token_begin, record.token_begin + record.token_count):
            for slot in range(num_topk):
                expert = int(topk_idx[token, slot])
                if expert // experts_per_rank != destination_rank:
                    continue
                meta = (
                    source_rank, token, expert, slot, record.descriptor_id,
                    record.reveal_epoch, float(np.float32(topk_weights[token, slot])),
                )
                payload = np.asarray(x[token], dtype="<f4").tobytes()
                output.append((meta, payload))
    return output


__all__ = ["CommitReference", "build_commit_reference", "expected_rank_records"]
