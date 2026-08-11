import inspect

import pytest


torch = pytest.importorskip("torch")

from rlccl.models import SlotLevelPolicy
from rlccl.training.ppo_trainer import train_epoch


def test_train_epoch_receives_persistent_optimizer():
    parameters = list(inspect.signature(train_epoch).parameters)
    assert parameters[:2] == ["model", "optimizer"]


def test_optimizer_state_round_trip():
    model = SlotLevelPolicy(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 0,
        "best_score": -1.0,
        "config": {},
    }
    restored_model = SlotLevelPolicy(hidden_dim=8)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=3e-4)
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert restored_optimizer.state_dict()["state"]


def test_baseline_policy_dimensions_are_unchanged():
    model = SlotLevelPolicy()
    assert model.node_encoder.in_features == 5
    assert model.edge_encoder.in_features == 2
    assert model.chunk_encoder[0].in_features == 2
    assert model.actor[0].in_features == 4 * 64 + 5 + 64
