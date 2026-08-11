import numpy as np

from rlccl.evaluation.partial_demand import build_partial_observation
from rlccl.traffic import traffic_matrix_to_scenario, validate_traffic_matrix


def _fixture():
    matrix = np.asarray(
        [[0, 2, 1, 0], [1, 0, 2, 1], [2, 1, 0, 1], [1, 2, 1, 0]],
        dtype=np.int64,
    )
    scenario = traffic_matrix_to_scenario(matrix)
    return matrix, np.asarray(scenario["demands"], dtype=np.int64)


def test_random_entry_observation_is_reproducible_and_never_invents_revealed_demand():
    matrix, demands = _fixture()
    left = build_partial_observation(
        matrix, demands, mode="random_entries", hide_ratio=0.50, seed=42
    )
    right = build_partial_observation(
        matrix, demands, mode="random_entries", hide_ratio=0.50, seed=42
    )
    validate_traffic_matrix(left.observed_matrix)
    assert np.array_equal(left.observed_matrix, right.observed_matrix)
    assert np.array_equal(left.observation_demands, right.observation_demands)
    assert np.all(left.observed_matrix <= matrix)
    for chunk in np.flatnonzero(left.revealed_chunk_mask):
        assert np.array_equal(left.observation_demands[chunk], demands[chunk])
    for chunk in np.flatnonzero(~left.revealed_chunk_mask):
        assert not np.any(left.observation_demands[chunk])


def test_source_total_observation_preserves_only_row_margins():
    matrix, demands = _fixture()
    observation = build_partial_observation(
        matrix, demands, mode="source_totals", hide_ratio=None, seed=42
    )
    validate_traffic_matrix(observation.observed_matrix)
    assert np.array_equal(observation.observed_matrix.sum(axis=1), matrix.sum(axis=1))
    assert observation.observation_demands.shape == demands.shape
    assert np.all(observation.observation_demands.sum(axis=1) == 1)


def test_source_destination_totals_use_only_feasible_margins():
    matrix, demands = _fixture()
    observation = build_partial_observation(
        matrix,
        demands,
        mode="source_destination_totals",
        hide_ratio=None,
        seed=42,
    )
    validate_traffic_matrix(observation.observed_matrix)
    assert np.array_equal(observation.observed_matrix.sum(axis=1), matrix.sum(axis=1))
    assert np.array_equal(observation.observed_matrix.sum(axis=0), matrix.sum(axis=0))
    assert np.all(observation.observation_demands.sum(axis=1) == 1)


def test_partial_shards_reveal_requested_fraction_without_changing_truth():
    matrix, demands = _fixture()
    original = demands.copy()
    observation = build_partial_observation(
        matrix, demands, mode="partial_shards", hide_ratio=0.25, seed=7
    )
    assert np.array_equal(demands, original)
    assert observation.revealed_chunk_mask.sum() == round(0.75 * len(demands))
    assert int(observation.observed_matrix.sum()) == observation.revealed_chunk_mask.sum()


def test_decoder_full_observation_keeps_default_policy_path_identical():
    torch = __import__("pytest").importorskip("torch")
    from rlccl.envs.decoder import SlotDecoder
    from rlccl.envs.evaluator import load_topology_info
    from rlccl.models import SlotLevelPolicy

    topology = load_topology_info("Rear4GPU")
    matrix, _ = _fixture()
    scenario = traffic_matrix_to_scenario(matrix)
    state = np.asarray(scenario["initial_state"], dtype=np.int64)
    demands = np.asarray(scenario["demands"], dtype=np.int64)
    torch.manual_seed(42)
    model = SlotLevelPolicy(hidden_dim=16)
    model.eval()
    default = SlotDecoder(topology).decode_slot(
        model, state.copy(), demands.copy(), 0, 20, train=False
    )[0]
    explicit = SlotDecoder(topology).decode_slot(
        model,
        state.copy(),
        demands.copy(),
        0,
        20,
        train=False,
        observation_demands=demands.astype(np.float32),
    )[0]
    assert np.array_equal(default, explicit)
