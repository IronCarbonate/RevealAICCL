"""Frozen deterministic L2-R top-k router used by staged runtime gates."""

from __future__ import annotations

import torch


def seed_router_params(d: int, e: int, seed: int = 20260805):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(d, e, generator=generator) * 0.1).float()
    bias = (torch.randn(e, generator=generator) * 0.01).float()
    return weight, bias


def router_topk(
    tokens: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    k: int = 1, mask: torch.Tensor | None = None,
):
    """Exact frozen top-k: stable descending ties choose smaller expert IDs."""

    logits = tokens @ weight + bias
    if mask is not None:
        logits = logits.masked_fill(mask.bool(), float("-inf"))
    values, indices = torch.sort(logits, dim=-1, descending=True, stable=True)
    return indices[:, :k].squeeze(-1), values[:, :k].squeeze(-1)


__all__ = ["router_topk", "seed_router_params"]
