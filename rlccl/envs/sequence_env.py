"""Temporal runner for history-conditioned All-to-All-V problem sequences."""

from collections.abc import Iterator
from typing import Any

import numpy as np

from .problem import ProblemInstance, TopologyInfo
from ..traffic.matrix_utils import traffic_matrix_to_scenario
from ..traffic.moment_estimator import SlidingMomentEstimator
from ..traffic.types import MomentContext, TrafficSequence


class TrafficSequenceRunner:
    """Yield sequence problems in order with context from completed history only.

    The estimator update is intentionally after ``yield``.  Therefore the context
    for step ``t`` can only contain matrices from steps ``0..t-1``.
    """

    def __init__(
        self,
        sequence: TrafficSequence,
        topology_info: TopologyInfo,
        *,
        time_limit: int = 20,
        min_history: int | None = None,
        estimator_eps: float = 1e-6,
    ) -> None:
        if topology_info.V != sequence.mean_ref.shape[0]:
            raise ValueError("Traffic sequence and topology node counts do not match")
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        self.sequence = sequence
        self.topology_info = topology_info
        self.time_limit = int(time_limit)
        window_size = int(sequence.metadata.get("window_size", len(sequence.matrices)))
        self.min_history = int(min_history if min_history is not None else window_size)
        self.estimator_eps = float(estimator_eps)

    def __iter__(
        self,
    ) -> Iterator[tuple[ProblemInstance, MomentContext, dict[str, Any]]]:
        estimator = SlidingMomentEstimator(
            num_nodes=self.topology_info.V,
            window_size=int(
                self.sequence.metadata.get("window_size", len(self.sequence.matrices))
            ),
            min_history=self.min_history,
            eps=self.estimator_eps,
        )
        for step, matrix in enumerate(self.sequence.matrices):
            context = estimator.get_context(
                matrix,
                self.sequence.mean_ref,
                self.sequence.var_ref,
            )
            scenario = traffic_matrix_to_scenario(
                matrix,
                sequence_id=self.sequence.sequence_id,
                sequence_step=step,
                family=self.sequence.family,
            )
            problem = ProblemInstance(
                num_nodes=self.topology_info.V,
                num_chunks=scenario["C"],
                num_edges=self.topology_info.E,
                time_limit=self.time_limit,
                capacities=self.topology_info.capacities,
                topology=self.topology_info.edges,
                demands=np.asarray(scenario["demands"], dtype=np.int64).reshape(
                    scenario["C"], self.topology_info.V
                ),
                initial_state=np.asarray(scenario["initial_state"], dtype=np.int64).reshape(
                    scenario["C"], self.topology_info.V
                ),
                shared_constraints=self.topology_info.shared_constraints,
                topology_info=self.topology_info,
                traffic_matrix=matrix,
                scenario_type="all_to_all_v",
                sequence_id=self.sequence.sequence_id,
                sequence_step=step,
                moment_context=context,
                metadata={"family": self.sequence.family, **self.sequence.metadata},
            )
            metadata = {
                "sequence_id": self.sequence.sequence_id,
                "sequence_step": step,
                "family": self.sequence.family,
                "seed": self.sequence.seed,
            }
            yield problem, context, metadata
            estimator.update(matrix)
