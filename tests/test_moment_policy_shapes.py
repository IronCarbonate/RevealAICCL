import numpy as np
import pytest


torch = pytest.importorskip("torch")

from rlccl.envs.decoder import (
    SlotDecoder,
    get_candidate_moment_features,
    get_candidate_moment_node_arrays,
    get_global_moment_features,
    get_moment_node_features,
    recompute_logp_slot,
)
from rlccl.envs.problem import TopologyInfo
from rlccl.envs.sequence_env import TrafficSequenceRunner
from rlccl.models import SlotLevelPolicy
from rlccl.traffic.context_views import mean_only_context
from rlccl.traffic.process_generator import (
    TrafficProcessConfig,
    generate_traffic_sequence,
)


def _topology():
    edges = np.asarray(
        [[u, v] for u in range(3) for v in range(3) if u != v], dtype=np.int64
    )
    return TopologyInfo(3, len(edges), edges, np.ones(len(edges)), [], name="full3")


def _problem():
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
            seed=71,
        )
    )
    return next(iter(TrafficSequenceRunner(sequence, _topology(), min_history=2)))[0]


def test_baseline_state_dict_has_no_moment_modules_and_loads_strictly():
    baseline = SlotLevelPolicy(hidden_dim=8)
    assert baseline.node_feat_dim == 5
    assert baseline.cand_feat_dim == 5
    assert baseline.global_moment_feat_dim == 0
    assert not hasattr(baseline, "moment_encoder")
    assert not any("moment" in key or "context_fusion" in key for key in baseline.state_dict())
    restored = SlotLevelPolicy(hidden_dim=8)
    restored.load_state_dict(baseline.state_dict(), strict=True)


def test_moment_feature_shapes_and_mean_ablation_are_finite():
    problem = _problem()
    context = problem.moment_context
    node = get_moment_node_features(context)
    global_features = get_global_moment_features(context, problem.traffic_matrix, 8)
    arrays = get_candidate_moment_node_arrays(context)
    topology = problem.topology_info
    candidate = get_candidate_moment_features(
        np.arange(topology.E), topology.edge_src, topology.edge_dst, arrays
    )
    assert node.shape == (problem.V, 7)
    assert global_features.shape == (8,)
    assert candidate.shape == (problem.E, 4)
    assert np.isfinite(node).all()
    assert np.isfinite(global_features).all()
    assert np.isfinite(candidate).all()

    mean_context = mean_only_context(context)
    assert not np.any(mean_context.var_matrix)
    assert not np.any(mean_context.current_send_z)
    assert np.any(mean_context.mean_matrix)


def test_moment_rollout_and_ppo_recompute_use_identical_features():
    torch.manual_seed(9)
    problem = _problem()
    model = SlotLevelPolicy(
        node_feat_dim=12,
        edge_feat_dim=2,
        cand_feat_dim=9,
        chunk_feat_dim=2,
        hidden_dim=8,
        global_moment_feat_dim=8,
    )
    decoder = SlotDecoder(problem.topology_info)
    _, old_logp, _, _, state_info, actions = decoder.decode_slot(
        model,
        problem.initial_state.copy(),
        problem.demands.copy(),
        0,
        problem.T,
        train=True,
        moment_context=problem.moment_context,
        current_matrix=problem.traffic_matrix,
        moment_max_entry=8,
    )
    new_logp, _, _ = recompute_logp_slot(
        model, state_info, actions, torch.device("cpu"), decoder.get_static_info()
    )
    assert state_info["node_feats"].shape[1] == 12
    assert state_info["global_moment_feats"].shape == (8,)
    assert "mean_matrix" not in state_info
    assert "var_matrix" not in state_info
    torch.testing.assert_close(new_logp.detach(), old_logp.detach(), rtol=1e-5, atol=1e-6)


def test_missing_context_uses_explicit_zero_moment_features():
    problem = _problem()
    model = SlotLevelPolicy(12, 2, 9, 2, hidden_dim=8, global_moment_feat_dim=8)
    decoder = SlotDecoder(problem.topology_info)
    output = decoder.decode_slot(
        model,
        problem.initial_state.copy(),
        problem.demands.copy(),
        0,
        problem.T,
        train=False,
        moment_context=None,
        current_matrix=problem.traffic_matrix,
    )
    state_info = output[4]
    assert state_info["moment_enabled"] is True
    assert not torch.any(state_info["global_moment_feats"])
    assert not np.any(state_info["node_moment_feats"])


def test_baseline_decoder_keeps_original_feature_widths():
    problem = _problem()
    model = SlotLevelPolicy(hidden_dim=8)
    decoder = SlotDecoder(problem.topology_info)
    output = decoder.decode_slot(
        model,
        problem.initial_state.copy(),
        problem.demands.copy(),
        0,
        problem.T,
        train=False,
        moment_context=problem.moment_context,
        current_matrix=problem.traffic_matrix,
    )
    state_info = output[4]
    assert state_info["node_feats"].shape[1] == 5
    assert state_info["global_moment_feats"] is None
