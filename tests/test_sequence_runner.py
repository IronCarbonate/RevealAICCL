import numpy as np

from rlccl.envs.problem import TopologyInfo
from rlccl.envs.sequence_env import TrafficSequenceRunner
from rlccl.traffic.process_generator import TrafficProcessConfig, generate_traffic_sequence


def test_runner_yields_temporal_history_without_current_leakage():
    sequence = generate_traffic_sequence(
        TrafficProcessConfig(
            num_nodes=3,
            sequence_length=8,
            window_size=4,
            mean_level=2.0,
            std_level=1.0,
            max_entry=8,
            epsilon_mean=0.3,
            epsilon_var=0.4,
            family="alternating_burst",
            seed=7,
        )
    )
    edges = np.array(
        [[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]], dtype=np.int64
    )
    topology = TopologyInfo(3, len(edges), edges, np.ones(len(edges)), [], name="ring")
    rows = list(TrafficSequenceRunner(sequence, topology, time_limit=5, min_history=2))

    assert [metadata["sequence_step"] for _, _, metadata in rows] == list(range(8))
    first_problem, first_context, _ = rows[0]
    assert first_context.history_length == 0
    assert first_problem.moment_context is first_context
    np.testing.assert_array_equal(first_problem.traffic_matrix, sequence.matrices[0])

    _, second_context, _ = rows[1]
    assert second_context.history_length == 1
    np.testing.assert_allclose(second_context.mean_matrix, sequence.matrices[0])
    _, third_context, _ = rows[2]
    np.testing.assert_allclose(
        third_context.mean_matrix,
        np.stack(sequence.matrices[:2]).mean(axis=0),
    )


def test_separate_runners_do_not_share_estimator_history():
    config = TrafficProcessConfig(3, 8, 4, 2.0, 1.0, 8, 0.3, 0.4, "bimodal", 9)
    sequence = generate_traffic_sequence(config)
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    topology = TopologyInfo(3, 3, edges, np.ones(3), [])
    first_a = next(iter(TrafficSequenceRunner(sequence, topology, min_history=1)))[1]
    first_b = next(iter(TrafficSequenceRunner(sequence, topology, min_history=1)))[1]
    assert first_a.history_length == first_b.history_length == 0
