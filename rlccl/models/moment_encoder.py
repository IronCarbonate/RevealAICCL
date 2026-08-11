"""Encoder for global history-only moment features."""

import torch
import torch.nn as nn


class MomentEncoder(nn.Module):
    """Encode a fixed-width global moment vector into policy context space."""

    def __init__(self, global_feat_dim: int, hidden_dim: int):
        super().__init__()
        if global_feat_dim <= 0 or hidden_dim <= 0:
            raise ValueError("global_feat_dim and hidden_dim must be positive")
        self.global_feat_dim = int(global_feat_dim)
        self.network = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, global_moment_feats: torch.Tensor) -> torch.Tensor:
        if global_moment_feats.shape[-1] != self.global_feat_dim:
            raise ValueError(
                f"Expected {self.global_feat_dim} global moment features, "
                f"got {global_moment_feats.shape[-1]}"
            )
        return self.network(global_moment_feats)
