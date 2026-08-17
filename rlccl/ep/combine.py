"""Host mirrors and correctness oracle for R6-M8 handle-driven combine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReturnLayout:
    num_source_tokens: int
    num_topk: int
    hidden: int
    base_offset: int

    def __post_init__(self) -> None:
        if min(self.num_source_tokens, self.num_topk, self.hidden) <= 0:
            raise ValueError("return layout dimensions must be positive")
        if self.base_offset < 0 or self.base_offset % 16:
            raise ValueError("return layout base must be non-negative and aligned")

    @property
    def record_bytes(self) -> int:
        return 16 + self.hidden * 4

    @property
    def region_bytes(self) -> int:
        return self.num_source_tokens * self.num_topk * self.record_bytes

    @property
    def capacity_bytes(self) -> int:
        return 2 * self.region_bytes

    def slot_id(self, token: int, topk_slot: int) -> int:
        if not 0 <= token < self.num_source_tokens:
            raise IndexError("source token outside return layout")
        if not 0 <= topk_slot < self.num_topk:
            raise IndexError("top-k slot outside return layout")
        return token * self.num_topk + topk_slot

    def return_offset(self, token: int, topk_slot: int) -> int:
        return self.base_offset + self.slot_id(token, topk_slot) * self.record_bytes

    def staging_offset(self, token: int, topk_slot: int) -> int:
        return self.base_offset + self.region_bytes + self.slot_id(token, topk_slot) * self.record_bytes


def reference_moe_output(
    x: np.ndarray, topk_idx: np.ndarray, topk_weights: np.ndarray,
    expert_weights: np.ndarray,
) -> np.ndarray:
    """Expert GEMM plus deterministic k-ascending weighted reduction."""
    tokens, hidden = x.shape
    output = np.zeros_like(x, dtype=np.float32)
    for token in range(tokens):
        for slot in range(topk_idx.shape[1]):
            expert = int(topk_idx[token, slot])
            if expert < 0:
                continue
            expert_value = np.zeros(hidden, dtype=np.float32)
            for out in range(hidden):
                value = np.float32(0)
                for inner in range(hidden):
                    value = np.float32(value + np.float32(
                        expert_weights[expert, out, inner] * x[token, inner]
                    ))
                expert_value[out] = value
            output[token] = np.asarray(
                output[token] + np.float32(topk_weights[token, slot]) * expert_value,
                dtype=np.float32,
            )
    return output


__all__ = ["ReturnLayout", "reference_moe_output"]
