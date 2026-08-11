import json
from pathlib import Path

import numpy as np

from rlccl.envs import evaluator
from rlccl.envs.problem import ProblemInstance, TopologyInfo, compute_received_chunks
from rlccl.traffic.moment_estimator import SlidingMomentEstimator


def _two_node_problem():
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    topology = TopologyInfo(2, 2, edges, np.ones(2), [])
    return ProblemInstance(
        2,
        1,
        2,
        3,
        np.ones(2),
        edges,
        np.array([[0, 1]], dtype=np.int64),
        np.array([[1, 0]], dtype=np.int64),
        topology_info=topology,
        traffic_matrix=np.array([[0, 1], [0, 0]], dtype=np.int64),
        scenario_type="all_to_all_v",
        sequence_id="seq-a",
        sequence_step=0,
        metadata={"array": np.array([1, 2])},
    )


def test_evaluator_uses_authoritative_problem_types():
    assert evaluator.ProblemInstance is ProblemInstance
    assert evaluator.TopologyInfo is TopologyInfo


def test_problem_optional_fields_round_trip_as_json():
    problem = _two_node_problem()
    estimator = SlidingMomentEstimator(2, 2, 1)
    problem.moment_context = estimator.get_context(
        problem.traffic_matrix,
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    payload = problem.to_dict()
    json.dumps(payload)
    restored = ProblemInstance.from_dict(payload)
    assert restored.sequence_id == "seq-a"
    assert restored.sequence_step == 0
    assert restored.scenario_type == "all_to_all_v"
    np.testing.assert_array_equal(restored.traffic_matrix, problem.traffic_matrix)
    np.testing.assert_allclose(
        restored.moment_context.mean_matrix, problem.moment_context.mean_matrix
    )


def test_received_chunks_matches_legacy_dense_incidence():
    rng = np.random.default_rng(7)
    edge_dst = np.array([1, 2, 0, 2])
    schedule = rng.integers(0, 2, size=(6, 4))
    dense_d = np.zeros((4, 3), dtype=np.int64)
    dense_d[np.arange(4), edge_dst] = 1
    legacy = (schedule @ dense_d > 0).astype(np.int64)
    actual = compute_received_chunks(schedule, edge_dst, 3)
    np.testing.assert_array_equal(actual, legacy)


def test_evaluate_schedule_does_not_require_dense_d():
    problem = _two_node_problem()
    assert not hasattr(problem, "D")
    schedule = [np.array([[1, 0]], dtype=np.int64)]
    score, error = evaluator.evaluate_schedule(schedule, problem)
    assert error == ""
    assert score > -2


def test_topology_capacity_and_group_limits_share_normalization():
    topology_name = "Rear4GPU"
    path = (
        Path(evaluator._BASE_DIR)
        / "Data"
        / topology_name
        / "Topology"
        / "pipeline_topology_no_switch.json"
    )
    _, _, _, raw_capacities, raw_groups = evaluator.load_topology_from_json(path)
    loaded = evaluator.load_topology_info(topology_name)
    scale = float(np.min(raw_capacities))
    np.testing.assert_allclose(loaded.capacities, raw_capacities / scale)
    assert len(loaded.shared_constraints) == len(raw_groups)
    for (loaded_edges, loaded_limit), (raw_edges, raw_limit) in zip(
        loaded.shared_constraints, raw_groups
    ):
        assert list(loaded_edges) == list(raw_edges)
        assert loaded_limit == raw_limit / scale


def test_all_to_all_v_range_includes_zero_and_is_nonempty():
    scenario = evaluator.generate_all_to_all_v_scenario(8, np.random.default_rng(3))
    matrix = np.asarray(scenario["traffic_matrix"])
    off_diagonal = matrix[~np.eye(8, dtype=bool)]
    assert off_diagonal.min() == 0
    assert off_diagonal.max() <= 4
    assert matrix.sum() == scenario["C"] > 0


def test_stale_scenario_cache_is_regenerated_with_schema(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [{"stale": True}]}), encoding="utf-8")
    scenarios = evaluator.load_or_generate_scenarios(path, 3, 4, 11)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(scenarios) == 4
    assert payload["schema_version"] == evaluator.SCENARIO_SCHEMA_VERSION
    assert payload["generator_name"] == evaluator.LEGACY_GENERATOR_NAME
    assert payload["generator_config_hash"]
