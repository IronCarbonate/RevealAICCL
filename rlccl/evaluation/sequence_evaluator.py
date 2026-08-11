"""Paired sequence evaluation utilities for AMR-AICCL policies."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
import torch

from ..envs.decoder import SlotDecoder
from ..envs.evaluator import evaluate_schedule
from ..envs.problem import compute_received_chunks
from ..traffic.context_views import mean_only_context


CONTEXT_MODES = ("baseline", "mean_only", "full", "shuffled")


def build_shuffled_context_map(
    problems: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    """Deterministically pair each problem with another sequence's context.

    Pairing first stays within the same family and sequence step.  If there is
    only one sequence for a family, it falls back to another family at the same
    step.  A donor from the same sequence is never accepted.
    """
    by_family_step: dict[tuple[str, int], list[tuple[str, Any]]] = defaultdict(list)
    by_step: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    for problem_id, problem in problems:
        family = str(problem.metadata.get("family", "unknown"))
        step = int(problem.sequence_step)
        entry = (problem_id, problem)
        by_family_step[(family, step)].append(entry)
        by_step[step].append(entry)

    result: dict[str, Any] = {}
    for problem_id, problem in problems:
        family = str(problem.metadata.get("family", "unknown"))
        step = int(problem.sequence_step)
        candidates = sorted(by_family_step[(family, step)], key=lambda item: item[0])
        if len({item[1].sequence_id for item in candidates}) < 2:
            candidates = sorted(by_step[step], key=lambda item: item[0])
        donors = [
            donor
            for _, donor in candidates
            if donor.sequence_id != problem.sequence_id
        ]
        if not donors:
            raise ValueError(
                "Shuffled-context evaluation requires at least two sequences "
                f"at sequence step {step}"
            )
        # The stable index spreads donors without relying on process-global RNG.
        donor = donors[sum(problem_id.encode("utf-8")) % len(donors)]
        result[problem_id] = donor.moment_context
    return result


def _selected_context(problem_id, problem, context_mode, shuffled_contexts):
    if context_mode == "baseline":
        return None
    if context_mode == "full":
        return problem.moment_context
    if context_mode == "mean_only":
        return mean_only_context(problem.moment_context)
    if context_mode == "shuffled":
        if shuffled_contexts is None or problem_id not in shuffled_contexts:
            raise ValueError("Missing shuffled context for problem")
        return shuffled_contexts[problem_id]
    raise ValueError(f"Unsupported context mode: {context_mode}")


def evaluate_sequence_policy(
    model,
    problems: Sequence[tuple[str, Any]],
    device: torch.device,
    *,
    context_mode: str,
    moment_max_entry: float = 8.0,
    shuffled_contexts: Mapping[str, Any] | None = None,
    warmup: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate one policy/context ablation over materialized problems."""
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unsupported context mode: {context_mode}")
    if context_mode != "baseline" and getattr(model, "global_moment_feat_dim", 0) <= 0:
        raise ValueError("Moment context modes require a moment-enabled model")

    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        if warmup and problems:
            warmup_id, warmup_problem = problems[0]
            warmup_context = _selected_context(
                warmup_id, warmup_problem, context_mode, shuffled_contexts
            )
            warmup_decoder = SlotDecoder(warmup_problem.topology_info)
            warmup_decoder.decode_slot(
                model,
                warmup_problem.initial_state.copy(),
                warmup_problem.demands.copy(),
                0,
                warmup_problem.T,
                train=False,
                moment_context=warmup_context,
                current_matrix=warmup_problem.traffic_matrix,
                moment_max_entry=moment_max_entry,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        for problem_id, problem in problems:
            topology = problem.topology_info
            decoder = SlotDecoder(topology)
            state = problem.initial_state.copy()
            demands = problem.demands.copy()
            schedule = []
            context = _selected_context(
                problem_id, problem, context_mode, shuffled_contexts
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            completion_steps = problem.T
            for slot in range(problem.T):
                slot_matrix, _, _, _, _, _ = decoder.decode_slot(
                    model,
                    state,
                    demands,
                    slot,
                    problem.T,
                    train=False,
                    moment_context=context,
                    current_matrix=problem.traffic_matrix,
                    moment_max_entry=moment_max_entry,
                )
                schedule.append(slot_matrix)
                received = compute_received_chunks(
                    slot_matrix, topology.edge_dst, topology.V
                )
                state = np.maximum(state, received)
                demands = demands * (1 - received)
                if not np.any(demands):
                    completion_steps = slot + 1
                    break
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            synthesis_ms = (time.perf_counter() - started) * 1000.0
            timeout = bool(np.any(demands))

            while len(schedule) < problem.T:
                schedule.append(np.zeros((problem.C, problem.E), dtype=np.int64))
            score, error = evaluate_schedule(schedule, problem)
            rows.append(
                {
                    "method": context_mode,
                    "problem_id": problem_id,
                    "sequence_id": problem.sequence_id,
                    "sequence_step": int(problem.sequence_step),
                    "family": str(problem.metadata.get("family", "unknown")),
                    "completion_steps": int(completion_steps),
                    "timeout": timeout,
                    "legal": error == "",
                    "evaluation_error": error,
                    "score": float(score),
                    "synthesis_ms": float(synthesis_ms),
                }
            )
    return rows
