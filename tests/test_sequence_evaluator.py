import numpy as np
import pytest


pytest.importorskip("torch")

from rlccl.envs.problem import TopologyInfo
from rlccl.evaluation import build_shuffled_context_map, cvar, summarize_rows
from rlccl.training import SequenceDatasetConfig, build_sequence_problems


def _problems():
    edges = np.asarray(
        [[u, v] for u in range(3) for v in range(3) if u != v], dtype=np.int64
    )
    topology = TopologyInfo(3, len(edges), edges, np.ones(len(edges)), [], name="full3")
    config = SequenceDatasetConfig(
        families=("smooth_ar",),
        num_sequences_per_family=2,
        sequence_length=8,
        window_size=4,
        min_history=2,
        mean_level=2.0,
        std_level=1.0,
        max_entry=8,
        epsilon_mean=0.3,
        epsilon_var=0.4,
        seed=17,
        time_limit=6,
    )
    return build_sequence_problems(topology, config)[0]


def test_shuffled_contexts_always_come_from_another_sequence_same_step():
    problems = _problems()
    mapping = build_shuffled_context_map(problems)
    by_identity = {id(problem.moment_context): problem for _, problem in problems}
    for problem_id, problem in problems:
        donor = by_identity[id(mapping[problem_id])]
        assert donor.sequence_id != problem.sequence_id
        assert donor.sequence_step == problem.sequence_step


def test_standard_metric_summary_and_cvar():
    rows = [
        {
            "completion_steps": value,
            "synthesis_ms": value / 10,
            "timeout": value == 4,
            "legal": True,
        }
        for value in (1, 2, 3, 4)
    ]
    summary = summarize_rows(rows)
    assert summary["completion_steps_mean"] == 2.5
    assert summary["legality_rate"] == 1.0
    assert summary["timeout_rate"] == 0.25
    assert cvar(np.asarray([1, 2, 3, 4]), 0.75) == 4.0
