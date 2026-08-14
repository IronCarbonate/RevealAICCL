#!/usr/bin/env python3
"""Render the R6-M4 report from formal machine-readable artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "phase_r6" / "m4_gpu_scheduler"
REPORT = ROOT / "docs" / "phase_r6" / "R6_M4_GPU_SCHEDULER_REPORT.md"


def main() -> None:
    results = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    legality = json.loads((OUTPUT / "legality_tests.json").read_text(encoding="utf-8"))
    with (OUTPUT / "scheduler_timeline.csv").open(encoding="utf-8", newline="") as handle:
        timing = list(csv.DictReader(handle))
    reveal_to_commit = [float(row["reveal_to_commit_us"]) for row in timing if float(row["reveal_to_commit_us"]) > 0]
    gates = results["gates"]
    model = results["execution_model"]
    scenario_lines = "\n".join(
        f"- `{name}`: {'PASS' if value['pass'] else 'FAIL'}, "
        f"CPU/GPU actions {value['cpu_action_count']}/{value['gpu_action_count']}"
        for name, value in results["scenario_results"].items()
    )
    legality_lines = "\n".join(
        f"- `{name}`: {'PASS' if value['pass'] else 'FAIL'} "
        f"(CPU={value['cpu_code']}, GPU={value['gpu_code']}, expected={value['expected_code']})"
        for name, value in legality.items()
    )
    latency = (
        f"min={min(reveal_to_commit):.3f} us, max={max(reveal_to_commit):.3f} us, "
        f"mean={sum(reveal_to_commit) / len(reveal_to_commit):.3f} us"
        if reveal_to_commit else "no committed-action samples"
    )
    text = f"""# R6-M4 GPU Scheduler Report

## Result

**{results['claim']}**

This claim stops at `DeviceActionQueue`. It is not a claim of GPU-driven
communication; packing and transport are explicitly outside R6-M4.

## 1. Modified and added files

The existing CPU scheduler and transport implementation were not modified.
R6-M4 adds:

- `rlccl/scheduler/common/scheduler_ir.h`: portable POD IR and error codes;
- `rlccl/scheduler/common/scheduler_schema.py`: host schema/configuration;
- `rlccl/scheduler/common/compiled_plan.py`: contiguous plan serialization,
  validation, and checksum;
- `rlccl/scheduler/cpu/reference.py`: deterministic CPU shadow adapter;
- `rlccl/scheduler/cuda/gpu_scheduler.cuh`: device queue/state definitions;
- `rlccl/scheduler/cuda/gpu_scheduler.cu`: producer, persistent scheduler,
  FastBinder, DynamicGuard, action publication, instrumentation;
- `rlccl/scheduler/cuda/gpu_scheduler_bindings.cpp`: job-level binding;
- `rlccl/scheduler/cuda/gpu_scheduler_backend.py`: initialization/debug facade;
- `scripts/run_r6_m4_gpu_scheduler.py`: formal gates and artifacts;
- `tests/test_r6_gpu_scheduler_common.py`: portable CPU contracts.

## 2. New structures

The common boundary is `RevealRecord -> CompiledPlanBlob -> CommittedAction`.
All common records use fixed-width integers and logical byte offsets. CUDA
queues/state are separate and contain the only device pointers.

## 3. Logic moved to GPU

Runtime reveal consumption, flattened pair demand updates, deterministic route
lookup, token-count/offset binding, every dynamic legality check, action ID
allocation, committed-count updates, and CommittedAction queue publication run
inside `gpu_scheduler_progress_kernel`.

## 4. Logic retained on CPU

Topology/static-plan compilation, blob serialization, job initialization, the
single persistent-kernel launch, job completion, CPU shadow validation, and
debug artifact collection remain on CPU. Runtime graph search remains absent.

## 5. StaticPlan upload and checksum

The host serializes header, route templates, rank-pair lookup, capacities, and
legality flags into one blob and uploads it once. CPU checksum:
`{gates['cpu_plan_checksum']}`. GPU uploaded/recomputed checksum:
`{gates['gpu_uploaded_plan_checksum']}`. Match: `{gates['plan_checksum_match']}`.

## 6. RevealQueue

The fixed-capacity device ring uses monotonic head/tail counters. The producer
writes the entire record before a device-scope release store to tail. The
single consumer performs an acquire load of tail before reading the record.
The producer consumes Router-resident destination arrays on the current Router
stream; EventBridge and Python do not participate in scheduler correctness.

## 7. IncrementalState

`revealed_count`, `committed_count`, `next_send_offset`, and
`next_recv_offset` are flat `[src * world_size + dst]` arrays. Descriptor
epochs, last reveal epoch, and next action ID are GPU resident. The frozen
single-consumer model owns all state writes, avoiding unnecessary atomics.

## 8. FastBinder

FastBinder counts revealed assignments by destination, performs a single
`rank_pair_to_route[src * world_size + dst]` lookup, assigns deterministic
destination order, computes logical registered-region offsets, and produces
candidate actions. Runtime BFS/DFS/topology planning is zero.

## 9. DynamicGuard

The GPU guard is fail closed for revealed/future demand, duplicate descriptor,
stale epoch, rank/count validity, logical offset and byte bounds, route/template
identity, descriptor range, assignment range, and queue capacity. A reveal is
preflighted as a batch; no action from the failing reveal is published.

{legality_lines}

## 10. ActionQueue

After every candidate for a reveal passes, actions enter a fixed-capacity
device ring in deterministic destination order. Each complete action write is
followed by a device-scope release publication of tail. Transport may consume
only this common `CommittedAction` in a later phase.

## 11. Persistent scheduler

Launch geometry is grid `{model['scheduler_grid_size']}`, block
`{model['scheduler_block_size']}`. It uses one strict consumer lane in one warp.
Measured attributes: `{model['registers_per_thread']}` registers/thread,
`{model['static_shared_memory_bytes']}` static shared-memory bytes, and
`{model['local_memory_bytes']}` local-memory bytes. It does not require
cooperative launch or generation-specific scheduling primitives.

## 12. CPU/GPU action equivalence

Aggregate divergence: `{gates['cpu_gpu_action_divergence']}`.

{scenario_lines}

Every comparison covers count/order and all 12 common action fields.

## 13. Legality result

All injected cases fail closed: `{gates['legality_fail_closed']}`. Detailed
codes are stored in `legality_tests.json`.

## 14. CPU per-descriptor involvement

Python callback: 0; CPU poll: 0; CPU scheduler: 0; CPU action construction: 0;
per-descriptor scheduler kernel launch: 0. CPU participation occurs only at the
job boundaries and in optional post-job shadow/debug collection.

## 15. Router-to-commit timing

Observed `reveal_to_commit`: {latency}. Per-record T0-T5 values and derived
mechanism timings are in the two timeline CSVs. These measurements are not a
performance benchmark.

## 16. Commit before final Router completion

Gate result: `{gates['commit_before_final_router']}`. The multi-shard producer
continues publishing later Router completions while the persistent scheduler
commits earlier revealed shards.

## 17. Volta/Ampere/Hopper compatibility

The extension compiled target list is `{model['compiled_architectures']}` and
the formal execution target is V100/sm_70. The kernel uses one block, one warp,
CUDA device-scope atomics, and no Hopper-only feature.

## 18. Current limitations

""" + "\n".join(f"- {item}" for item in results["limitations"]) + """

## 19. Next-phase transport boundary

R6-M4 ends at `DeviceActionQueue`. The next phase may add GPU packing and a
device transport consumer for NVIDIA, while a sibling Ascend branch may consume
the same host schema with an Ascend-specific queue/state/kernel. Scheduler IR
must remain unaware of NCCL, MSCCL++, NVSHMEM, DeepEP, or device pointers.

## Artifact index

- `outputs/phase_r6/m4_gpu_scheduler/results.json`
- `outputs/phase_r6/m4_gpu_scheduler/action_comparison.csv`
- `outputs/phase_r6/m4_gpu_scheduler/reveal_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/scheduler_timeline.csv`
- `outputs/phase_r6/m4_gpu_scheduler/legality_tests.json`
"""
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
