"""Python initialization/debug facade for the persistent CUDA scheduler.

The runtime scheduler itself is entirely device resident.  Python participates
only in plan upload, one job-level launch, and post-job evidence collection.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from ..common.compiled_plan import CompiledPlanBlob, validate_compiled_plan
from ..common.scheduler_schema import (
    CommittedAction, DeviceSchedulerError, RevealRecord,
)


_EXTENSION: Any | None = None


def load_gpu_scheduler_extension(*, verbose: bool = False) -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    import torch
    from torch.utils.cpp_extension import load

    if not torch.cuda.is_available():
        raise RuntimeError("R6-M4 CUDA backend requires an available CUDA device")
    source_dir = Path(__file__).resolve().parent
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;8.0;9.0")
    _EXTENSION = load(
        name="rlccl_r6_m4_gpu_scheduler",
        sources=[
            str(source_dir / "gpu_scheduler_bindings.cpp"),
            str(source_dir / "gpu_scheduler.cu"),
        ],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2", "-std=c++17", "--ptxas-options=-v"],
        verbose=verbose,
    )
    return _EXTENSION


@dataclass(frozen=True, slots=True)
class GPUSchedulerRun:
    actions: tuple[CommittedAction, ...]
    errors: tuple[DeviceSchedulerError, ...]
    timings: tuple[tuple[int, ...], ...]
    counters: tuple[int, ...]
    revealed_count: tuple[int, ...]
    committed_count: tuple[int, ...]


class GPUSchedulerBackend:
    """Job-level CUDA backend; no per-descriptor CPU scheduler calls."""

    def __init__(self, plan: CompiledPlanBlob, *, extension: Any | None = None) -> None:
        validate_compiled_plan(plan)
        self.plan = plan
        self.extension = extension or load_gpu_scheduler_extension()
        self.cpu_per_descriptor_scheduler_calls = 0
        self.cpu_per_descriptor_action_construction = 0
        self.cpu_scheduler_kernel_launches = 0

    def run(
        self,
        records: Sequence[RevealRecord],
        dst_ranks: Sequence[int],
        *,
        producer_delay_cycles: int = 0,
        reveal_queue_capacity: int | None = None,
        action_queue_capacity: int | None = None,
    ) -> GPUSchedulerRun:
        import torch

        rows = [item.as_tuple() for item in records]
        record_cpu = torch.tensor(rows, dtype=torch.int64).reshape((-1, 8))
        dst_cpu = torch.tensor(tuple(dst_ranks), dtype=torch.int32)
        # CPU-originating compatibility/debug path.  Formal Router integration
        # calls run_device() with the top-k output already resident on GPU.
        record_device = record_cpu.cuda()
        dst_device = dst_cpu.cuda()
        plan_tensor = torch.frombuffer(bytearray(self.plan.data), dtype=torch.uint8)
        return self.run_device(
            record_device, dst_device,
            producer_delay_cycles=producer_delay_cycles,
            reveal_queue_capacity=reveal_queue_capacity,
            action_queue_capacity=action_queue_capacity,
            plan_tensor=plan_tensor,
        )

    def run_device(
        self,
        reveal_records: Any,
        dst_ranks: Any,
        *,
        producer_delay_cycles: int = 0,
        reveal_queue_capacity: int | None = None,
        action_queue_capacity: int | None = None,
        plan_tensor: Any | None = None,
    ) -> GPUSchedulerRun:
        """Consume Router-resident CUDA tensors on the current Router stream."""
        import torch

        if not reveal_records.is_cuda or reveal_records.dtype != torch.int64:
            raise ValueError("reveal_records must be a CUDA int64 tensor")
        if not dst_ranks.is_cuda or dst_ranks.dtype != torch.int32:
            raise ValueError("dst_ranks must be a CUDA int32 tensor")
        if plan_tensor is None:
            plan_tensor = torch.frombuffer(bytearray(self.plan.data), dtype=torch.uint8)
        output = self.extension.run_gpu_scheduler(
            plan_tensor, reveal_records.contiguous(), dst_ranks.contiguous(),
            self.plan.config.source_rank,
            int(reveal_queue_capacity or self.plan.config.reveal_queue_capacity),
            int(action_queue_capacity or self.plan.config.action_queue_capacity),
            int(producer_delay_cycles),
        )
        action_rows = output["actions"].tolist()
        error_rows = output["errors"].tolist()
        state = output["state"].tolist()
        return GPUSchedulerRun(
            actions=tuple(CommittedAction(*map(int, row)) for row in action_rows),
            errors=tuple(DeviceSchedulerError(*map(int, row)) for row in error_rows),
            timings=tuple(tuple(map(int, row)) for row in output["timings"].tolist()),
            counters=tuple(map(int, output["counters"].tolist())),
            revealed_count=tuple(map(int, state[0])),
            committed_count=tuple(map(int, state[1])),
        )


__all__ = ["GPUSchedulerBackend", "GPUSchedulerRun", "load_gpu_scheduler_extension"]
