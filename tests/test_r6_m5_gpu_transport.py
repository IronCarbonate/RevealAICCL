"""Portable offset and fail-closed contracts for R6-M5."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan
from rlccl.scheduler.common.scheduler_schema import CommittedAction, SchedulerConfig
from rlccl.scheduler.cpu.reference import CPUSchedulerShadow
from rlccl.scheduler.common.scheduler_schema import RevealRecord
from rlccl.transport.cuda.layout import GPURegisteredBufferLayout


def _layout() -> GPURegisteredBufferLayout:
    return GPURegisteredBufferLayout(
        world_size=2, max_descriptors=4, max_tokens_per_peer=32,
        metadata_fields=9, feature_width=16,
    )


def _actions():
    layout = _layout()
    plan = compile_rank_pair_plan(SchedulerConfig(
        world_size=2, source_rank=0, record_bytes=layout.record_bytes,
        max_descriptors=4, max_chunks=4, max_tokens_per_peer=32,
        reveal_queue_capacity=8, action_queue_capacity=16,
    ))
    result = CPUSchedulerShadow(plan).run(
        (RevealRecord(0, 1, 0, 4, 0, 4, 0),), (0, 1, 1, 0),
    )
    assert not result.errors
    return layout, result.actions


def test_m4_logical_offsets_map_uniquely_to_registered_physical_slots() -> None:
    layout, actions = _actions()
    physical = tuple(layout.map_action(action) for action in actions)
    assert [value.dst_rank for value in physical] == [0, 1]
    assert physical[0].physical_src_offset == layout.send_offset(0, 0)
    assert physical[1].physical_src_offset == layout.send_offset(0, 1)
    assert physical[1].physical_dst_offset == layout.receive_offset(0, 0)
    assert physical[1].physical_bytes == 8 + 2 * layout.record_bytes


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("token_count", 0, "token count"),
        ("bytes", 1, "byte count"),
        ("src_offset", 8, "unique physical slot"),
        ("dst_offset", 8, "unique physical slot"),
        ("descriptor_id", 99, "descriptor"),
        ("dst_rank", 99, "rank"),
    ],
)
def test_invalid_physical_mapping_fails_closed(field, value, message) -> None:
    layout, actions = _actions()
    with pytest.raises(ValueError, match=message):
        layout.map_action(replace(actions[0], **{field: value}))


def test_packing_layout_preserves_metadata_and_fp32_record_bytes() -> None:
    layout = _layout()
    assert layout.metadata_bytes == 9 * 8
    assert layout.feature_bytes == 16 * 4
    assert layout.record_bytes == 136
    assert layout.peer_stride == 8 + 32 * 136
    assert layout.capacity_bytes == 2 * 4 * 2 * layout.peer_stride
