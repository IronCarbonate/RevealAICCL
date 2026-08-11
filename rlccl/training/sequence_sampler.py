"""Deterministic, temporally ordered sequence dataset construction."""

from dataclasses import dataclass
from typing import Any

from ..envs.sequence_env import TrafficSequenceRunner
from ..traffic.process_generator import TrafficProcessConfig, generate_traffic_sequence


@dataclass(frozen=True)
class SequenceDatasetConfig:
    families: tuple[str, ...]
    num_sequences_per_family: int
    sequence_length: int
    window_size: int
    min_history: int
    mean_level: float
    std_level: float
    max_entry: int
    epsilon_mean: float
    epsilon_var: float
    seed: int
    time_limit: int = 20


def build_sequence_problems(topology_info, config: SequenceDatasetConfig):
    """Return materialized problems with contexts computed in temporal order."""
    if config.num_sequences_per_family <= 0:
        raise ValueError("num_sequences_per_family must be positive")
    problems = []
    sequences = []
    sequence_records: list[dict[str, Any]] = []
    for family_index, family in enumerate(config.families):
        for sequence_index in range(config.num_sequences_per_family):
            sequence_seed = config.seed + family_index * 10000 + sequence_index
            traffic_config = TrafficProcessConfig(
                num_nodes=topology_info.V,
                sequence_length=config.sequence_length,
                window_size=config.window_size,
                mean_level=config.mean_level,
                std_level=config.std_level,
                max_entry=config.max_entry,
                epsilon_mean=config.epsilon_mean,
                epsilon_var=config.epsilon_var,
                family=family,
                seed=sequence_seed,
                topology_name=topology_info.name or "unknown",
            )
            sequence = generate_traffic_sequence(traffic_config)
            runner = TrafficSequenceRunner(
                sequence,
                topology_info,
                time_limit=config.time_limit,
                min_history=config.min_history,
            )
            sequence_problem_ids = []
            for problem, _, metadata in runner:
                problem_id = f"{sequence.sequence_id}-step{metadata['sequence_step']}"
                problems.append((problem_id, problem))
                sequence_problem_ids.append(problem_id)
            sequences.append(sequence)
            sequence_records.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "family": family,
                    "seed": sequence_seed,
                    "problem_ids": sequence_problem_ids,
                }
            )
    return problems, sequences, sequence_records
