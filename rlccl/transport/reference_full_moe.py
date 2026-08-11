"""Reference full-MoE payload, expert, return, and combine correctness helpers.

This is intentionally not production packing.  Identity metadata and FP32
features/outputs use paired, identically split real A2Av payloads so integer
identity remains exact while expert tensors retain FP32 semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from rlccl.transport.reference_a2av import DestinationLayout


FORWARD_META_FIELDS = 9
RETURN_META_FIELDS = 8
_MODULUS = (1 << 63) - 25
_COEFFICIENTS = (1_000_003, 1_000_033, 1_000_037, 1_000_081,
                 1_000_099, 1_000_117, 1_000_121, 1_000_133)


def identity_checksum(prefix: Sequence[int]) -> int:
    if len(prefix) > len(_COEFFICIENTS):
        raise ValueError("identity checksum prefix too long")
    value = 0
    for item, coefficient in zip(prefix, _COEFFICIENTS, strict=False):
        value = (value + (int(item) % _MODULUS) * coefficient) % _MODULUS
    return int(value)


def feature_digest(row: np.ndarray) -> int:
    values = np.ascontiguousarray(np.asarray(row, dtype=np.float32))
    digest = hashlib.blake2b(values.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 62) - 1)


@dataclass(frozen=True, slots=True)
class ForwardPayload:
    sendcounts_tokens: tuple[int, ...]
    offsets_tokens: tuple[int, ...]
    metadata: np.ndarray
    features: np.ndarray

    @property
    def total_tokens(self) -> int:
        return int(self.metadata.shape[0])


@dataclass(frozen=True, slots=True)
class ReturnPayload:
    sendcounts_tokens: tuple[int, ...]
    offsets_tokens: tuple[int, ...]
    metadata: np.ndarray
    outputs: np.ndarray

    @property
    def total_tokens(self) -> int:
        return int(self.metadata.shape[0])


def pack_forward_payload(
    layout: DestinationLayout,
    *,
    features_by_token: Mapping[int, np.ndarray],
    original_position_by_token: Mapping[int, int],
) -> ForwardPayload:
    rows: list[tuple[int, ...]] = []
    features: list[np.ndarray] = []
    for assignments in layout.destination_lists:
        for item in assignments:
            token_id = int(item.token_id)
            feature = np.asarray(features_by_token[token_id], dtype=np.float32)
            prefix = (
                token_id, int(item.source_rank), int(item.destination_rank),
                int(item.expert_id), int(item.chunk_id), int(item.chunk_offset),
                int(original_position_by_token[token_id]), feature_digest(feature),
            )
            rows.append(prefix + (identity_checksum(prefix),))
            features.append(feature)
    metadata = np.asarray(rows, dtype=np.int64).reshape((-1, FORWARD_META_FIELDS))
    width = (
        int(features[0].shape[0]) if features
        else int(np.asarray(next(iter(features_by_token.values())), dtype=np.float32).shape[0])
    )
    feature_array = (
        np.asarray(features, dtype=np.float32).reshape((-1, width))
        if features else np.empty((0, 0), dtype=np.float32)
    )
    return ForwardPayload(
        sendcounts_tokens=layout.sendcounts_tokens,
        offsets_tokens=layout.offsets_tokens,
        metadata=np.ascontiguousarray(metadata),
        features=np.ascontiguousarray(feature_array),
    )


def verify_forward_payload(
    metadata: np.ndarray,
    features: np.ndarray,
    *,
    destination_rank: int,
    world_size: int,
    recvcounts_tokens: Sequence[int] | None = None,
) -> dict[str, int | bool]:
    meta = np.asarray(metadata, dtype=np.int64)
    values = np.asarray(features, dtype=np.float32)
    if meta.ndim != 2 or meta.shape[1] != FORWARD_META_FIELDS:
        raise ValueError("invalid forward metadata shape")
    if values.ndim != 2 or values.shape[0] != meta.shape[0]:
        raise ValueError("invalid forward feature shape")
    seen: set[int] = set()
    sender_by_row: list[int] | None = None
    if recvcounts_tokens is not None:
        if sum(int(value) for value in recvcounts_tokens) != meta.shape[0]:
            raise ValueError("forward recvcounts mismatch")
        sender_by_row = []
        for sender, count in enumerate(recvcounts_tokens):
            sender_by_row.extend([sender] * int(count))
    duplicate = wrong_source = wrong_destination = wrong_expert = corruption = 0
    for index, row in enumerate(meta):
        token_id = int(row[0])
        duplicate += int(token_id in seen)
        seen.add(token_id)
        if sender_by_row is not None:
            wrong_source += int(int(row[1]) != sender_by_row[index])
        wrong_destination += int(int(row[2]) != destination_rank)
        wrong_expert += int(int(row[3]) % world_size != destination_rank)
        corruption += int(identity_checksum(row[:8]) != int(row[8]))
        corruption += int(feature_digest(values[index]) != int(row[7]))
    return {
        "pass": not any((duplicate, wrong_source, wrong_destination, wrong_expert, corruption)),
        "tokens": int(meta.shape[0]), "duplicate": duplicate,
        "wrong_source": wrong_source,
        "wrong_destination": wrong_destination, "wrong_expert": wrong_expert,
        "corruption": corruption,
    }


def pack_return_payload(
    forward_metadata: np.ndarray,
    expert_outputs: np.ndarray,
    *,
    expert_rank: int,
    world_size: int,
) -> ReturnPayload:
    meta = np.asarray(forward_metadata, dtype=np.int64)
    outputs = np.asarray(expert_outputs, dtype=np.float32)
    if meta.shape[0] != outputs.shape[0]:
        raise ValueError("expert output cardinality mismatch")
    groups: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(world_size)]
    for row, output in zip(meta, outputs, strict=True):
        source = int(row[1])
        if not 0 <= source < world_size:
            raise ValueError("invalid original source")
        prefix = (
            int(row[0]), source, expert_rank, int(row[3]), int(row[6]),
            int(row[4]), int(row[5]),
        )
        return_row = np.asarray(prefix + (identity_checksum(prefix),), dtype=np.int64)
        groups[source].append((return_row, np.asarray(output, dtype=np.float32)))
    for group in groups:
        group.sort(key=lambda item: (int(item[0][0]), int(item[0][4])))
    counts = tuple(len(group) for group in groups)
    offsets, cursor = [], 0
    for count in counts:
        offsets.append(cursor); cursor += count
    flattened = [item for group in groups for item in group]
    return ReturnPayload(
        sendcounts_tokens=counts, offsets_tokens=tuple(offsets),
        metadata=np.ascontiguousarray(
            np.asarray([item[0] for item in flattened], dtype=np.int64)
            .reshape((-1, RETURN_META_FIELDS))
        ),
        outputs=np.ascontiguousarray(
            np.asarray([item[1] for item in flattened], dtype=np.float32)
            .reshape((-1, outputs.shape[1]))
        ),
    )


def verify_return_and_combine(
    metadata: np.ndarray,
    outputs: np.ndarray,
    *,
    origin_rank: int,
    recvcounts_tokens: Sequence[int],
    expected_expert_by_token: Mapping[int, int],
    expected_position_by_token: Mapping[int, int],
    expected_output_by_token: Mapping[int, np.ndarray],
    total_tokens: int,
    required_positions: Sequence[int] | None = None,
    atol: float = 2e-3,
    rtol: float = 2e-3,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    meta = np.asarray(metadata, dtype=np.int64)
    values = np.asarray(outputs, dtype=np.float32)
    if meta.ndim != 2 or meta.shape[1] != RETURN_META_FIELDS or values.shape[0] != meta.shape[0]:
        raise ValueError("invalid return payload")
    if sum(int(value) for value in recvcounts_tokens) != meta.shape[0]:
        raise ValueError("return recvcounts mismatch")
    sender_by_row: list[int] = []
    for sender, count in enumerate(recvcounts_tokens):
        sender_by_row.extend([sender] * int(count))
    combined = np.zeros((total_tokens, values.shape[1]), dtype=np.float32)
    filled = np.zeros(total_tokens, dtype=np.bool_)
    seen: set[int] = set()
    duplicate = wrong_destination = wrong_return = wrong_position = corruption = 0
    wrong_expert = expert_output_mismatch = 0
    for index, row in enumerate(meta):
        token_id = int(row[0]); position = int(row[4]); expert = int(row[3])
        duplicate += int(token_id in seen); seen.add(token_id)
        wrong_destination += int(int(row[1]) != origin_rank)
        wrong_return += int(int(row[2]) != sender_by_row[index])
        wrong_expert += int(expected_expert_by_token.get(token_id, -1) != expert)
        expected_position = expected_position_by_token.get(token_id, -1)
        wrong_position += int(position != expected_position or not 0 <= position < total_tokens)
        corruption += int(identity_checksum(row[:7]) != int(row[7]))
        expected_output = expected_output_by_token.get(token_id)
        if expected_output is None or not np.allclose(values[index], expected_output, atol=atol, rtol=rtol):
            expert_output_mismatch += 1
        if 0 <= position < total_tokens:
            if filled[position]:
                duplicate += 1
            combined[position] = values[index]
            filled[position] = True
    expected_tokens = set(int(value) for value in expected_expert_by_token)
    required = (
        tuple(int(value) for value in required_positions)
        if required_positions is not None else tuple(range(total_tokens))
    )
    missing_tokens = len(expected_tokens - seen)
    missing_positions = sum(not bool(filled[position]) for position in required)
    lost = max(missing_tokens, missing_positions)
    passed = not any((lost, duplicate, wrong_destination, wrong_return, wrong_position,
                      corruption, wrong_expert, expert_output_mismatch))
    return combined, {
        "pass": passed, "tokens": int(meta.shape[0]), "lost": lost,
        "duplicate": duplicate, "wrong_destination": wrong_destination,
        "wrong_return": wrong_return, "wrong_position": wrong_position,
        "wrong_expert": wrong_expert, "corruption": corruption,
        "expert_output_mismatch": expert_output_mismatch,
    }


def seed_reference_experts(
    input_dim: int, hidden_dim: int, output_dim: int, experts: int, seed: int,
) -> tuple[Any, Any, Any, Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale1, scale2 = input_dim ** -0.5, hidden_dim ** -0.5
    w1 = torch.randn(experts, input_dim, hidden_dim, generator=generator, dtype=torch.float32) * scale1
    b1 = torch.randn(experts, hidden_dim, generator=generator, dtype=torch.float32) * 0.01
    w2 = torch.randn(experts, hidden_dim, output_dim, generator=generator, dtype=torch.float32) * scale2
    b2 = torch.randn(experts, output_dim, generator=generator, dtype=torch.float32) * 0.01
    return w1, b1, w2, b2


def reference_expert_mlp(
    features: Any, expert_ids: np.ndarray, weights: tuple[Any, Any, Any, Any],
) -> tuple[Any, tuple[int, ...]]:
    import torch

    w1, b1, w2, b2 = weights
    output = torch.empty((features.shape[0], w2.shape[2]), dtype=torch.float32, device=features.device)
    counts = []
    for expert in range(w1.shape[0]):
        indices_np = np.flatnonzero(np.asarray(expert_ids) == expert)
        counts.append(int(indices_np.size))
        if not indices_np.size:
            continue
        indices = torch.from_numpy(indices_np.astype(np.int64)).to(features.device)
        batch = features.index_select(0, indices)
        hidden = torch.relu(batch @ w1[expert].to(features.device) + b1[expert].to(features.device))
        values = hidden @ w2[expert].to(features.device) + b2[expert].to(features.device)
        output.index_copy_(0, indices, values)
    return output, tuple(counts)
