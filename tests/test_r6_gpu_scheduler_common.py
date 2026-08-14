"""CPU-side contracts for the portable R6-M4 scheduler boundary."""

from __future__ import annotations

import pytest

from rlccl.scheduler.common.compiled_plan import (
    compile_rank_pair_plan, validate_compiled_plan,
)
from rlccl.scheduler.common.scheduler_schema import (
    RevealFlags, RevealRecord, SchedulerConfig, SchedulerErrorCode,
)
from rlccl.scheduler.cpu.reference import CPUSchedulerShadow


def _plan(*, action_capacity: int = 32):
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=4, source_rank=0, record_bytes=64,
        max_descriptors=8, max_chunks=8, max_tokens_per_peer=16,
        reveal_queue_capacity=8, action_queue_capacity=action_capacity,
    ))


def test_compiled_plan_is_contiguous_pointer_free_and_checksum_protected() -> None:
    plan = _plan()
    assert validate_compiled_plan(plan)
    corrupted = bytearray(plan.data)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        validate_compiled_plan(bytes(corrupted))
    assert len(plan.rank_pair_to_route) == 16
    assert all(route.route_id == index for index, route in enumerate(plan.route_templates))


def test_cpu_shadow_emits_deterministic_destination_order_and_offsets() -> None:
    plan = _plan()
    records = (
        RevealRecord(0, 1, 0, 4, 0, 4, 0),
        RevealRecord(1, 2, 4, 3, 4, 3, 1),
    )
    result = CPUSchedulerShadow(plan).run(records, (3, 1, 3, 2, 1, 1, 2))
    assert not result.errors
    assert [(a.descriptor_id, a.dst_rank, a.token_count) for a in result.actions] == [
        (0, 1, 1), (0, 2, 1), (0, 3, 2), (1, 1, 2), (1, 2, 1),
    ]
    assert [a.action_id for a in result.actions] == list(range(5))
    assert result.revealed_count == result.committed_count


@pytest.mark.parametrize(
    ("flag", "code"),
    [
        (RevealFlags.INJECT_FUTURE, SchedulerErrorCode.FUTURE_DEMAND),
        (RevealFlags.INJECT_UNREVEALED, SchedulerErrorCode.UNREVEALED_DEMAND),
        (RevealFlags.INJECT_STALE_ACTION, SchedulerErrorCode.STALE_REVEAL),
        (RevealFlags.INJECT_DUPLICATE_ACTION, SchedulerErrorCode.DUPLICATE_DESCRIPTOR),
        (RevealFlags.INJECT_OFFSET_OVERFLOW, SchedulerErrorCode.OFFSET_OVERFLOW),
        (RevealFlags.INJECT_INVALID_RANK, SchedulerErrorCode.INVALID_SOURCE_RANK),
        (RevealFlags.INJECT_INVALID_ROUTE, SchedulerErrorCode.INVALID_ROUTE),
        (RevealFlags.INJECT_ZERO_TOKEN_ACTION, SchedulerErrorCode.ZERO_TOKEN_ACTION),
        (RevealFlags.INJECT_BYTES_OVERFLOW, SchedulerErrorCode.BYTES_OVERFLOW),
    ],
)
def test_cpu_shadow_fault_injection_is_fail_closed(flag, code) -> None:
    plan = _plan()
    record = RevealRecord(0, 1, 0, 1, 0, 1, 0, int(flag))
    result = CPUSchedulerShadow(plan).run((record,), (1,))
    assert not result.actions
    assert result.errors[0].error_code == int(code)
    assert sum(result.committed_count) == 0


def test_stale_reveal_and_action_queue_overflow_are_fail_closed() -> None:
    plan = _plan(action_capacity=1)
    overflow = CPUSchedulerShadow(plan).run(
        (RevealRecord(0, 1, 0, 2, 0, 2, 0),), (1, 2),
    )
    assert not overflow.actions
    assert overflow.errors[0].error_code == int(SchedulerErrorCode.ACTION_QUEUE_OVERFLOW)
    assert sum(overflow.committed_count) == 0

    stale = CPUSchedulerShadow(_plan()).run((
        RevealRecord(0, 2, 0, 1, 0, 1, 0),
        RevealRecord(1, 1, 1, 1, 1, 1, 1),
    ), (1, 1))
    assert len(stale.actions) == 1
    assert stale.errors[0].error_code == int(SchedulerErrorCode.STALE_REVEAL)
