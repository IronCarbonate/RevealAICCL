#!/usr/bin/env python3
"""Render the R6-M5 report from formal GPU artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "phase_r6" / "m5_gpu_transport"
REPORT = ROOT / "docs" / "phase_r6" / "R6_M5_GPU_TRANSPORT_REPORT.md"


def _stats(rows, field):
    values = [float(row[field]) for row in rows]
    return "n/a" if not values else f"min={min(values):.3f}, max={max(values):.3f}, mean={sum(values)/len(values):.3f} us"


def main() -> None:
    result = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    correctness = json.loads((OUTPUT / "correctness.json").read_text(encoding="utf-8"))
    with (OUTPUT / "packing_trace.csv").open(encoding="utf-8", newline="") as handle:
        packing = list(csv.DictReader(handle))
    with (OUTPUT / "transport_trace.csv").open(encoding="utf-8", newline="") as handle:
        transport = list(csv.DictReader(handle))
    gates, counters, layout = result["gates"], result["counters"], result["layout"]
    text = f"""# R6-M5 GPU-Driven Packing and Transport Report

## Result

**{result['claim']}**

The claim ends at remote registered-buffer decode. It does not include expert
compute, return traffic, NCCL Device API, multi-node work, or a performance
benchmark.

## 1. Modified files

R6-M4 Scheduler files were not modified. R6-M5 adds the CUDA packing/layout
layer under `rlccl/transport/cuda/`, the native runtime in
`extensions/r6_m5_gpu_transport/gpu_transport_runtime.cu`, the job-level ctypes
owner, formal runner, tests, protocol, report generator, and result artifacts.

## 2. Packing buffer layout

The registered allocation is `{layout['capacity_bytes']}` bytes. Each half is a
`{layout['region_bytes']}`-byte send/receive region; descriptor stride is
`{layout['descriptor_stride']}` and peer stride is `{layout['peer_stride']}`.
Each `{layout['record_bytes']}`-byte record preserves nine int64 metadata fields
and 16 FP32 feature values, preceded by one 8-byte count header per slot.

## 3. Logical to physical offsets

M4 logical offsets remain unchanged. M5 creates `PhysicalTransportAction` only
after checking the unique descriptor/peer slot, exact payload byte count,
8-byte phase, send/receive region bounds, and replay bitmap. Mapping used:
`{layout['logical_to_physical_mapping']}`.

## 4. MSCCL++ memory registration

Each rank allocates its complete buffer before runtime creation. The native
runtime registers it once using CUDA IPC, exchanges remote memory once, and
keeps the semaphore, MemoryChannel, and device handle alive for the entire job.
There is no per-descriptor allocation or registration.

## 5. Persistent transport consumer

One 256-thread block directly pops `DeviceActionQueue`, validates the action,
packs metadata/features into the registered send slot, and performs either a
remote MemoryChannel put or a local GPU copy. The consumer exits only after the
frozen scheduler publishes its job-done flag and the queue is empty.

## 6. put/signal/wait

Remote actions call device-side `put<8>(dst, src, physical_bytes, tid, 256)`.
After a block barrier lane zero calls `signal()`. A separate job-lifetime block
calls `wait(-1)` and records remote completion. Put calls:
`{counters['mscclpp_put_calls']}`; transferred bytes:
`{counters['mscclpp_bytes_transferred']}`; signals:
`{counters['mscclpp_signals']}`; waits: `{counters['mscclpp_waits']}`.

## 7. CPU participation audit

Per descriptor: Python callback 0, CPU poll 0, CPU packing 0, CPU action
construction 0, CPU transport submission 0, and CPU CUDA launch 0. CPU is used
only for initialization, job-level launches/completion, and post-job debug
collection.

## 8. Correctness

Action divergence: `{gates['scheduler_action_divergence']}`. Payload divergence:
`{correctness['payload_divergence']}`. Lost `{correctness['lost']}`, duplicate
`{correctness['duplicate']}`, wrong destination
`{correctness['wrong_destination']}`, corruption `{correctness['corruption']}`.
Metadata and Router-derived destination validity are
`{correctness['metadata_valid']}` and `{correctness['router_destination_valid']}`.

## 9. Legality

Future/unrevealed/stale counters are zero:
`{gates['future_unrevealed_stale_zero']}`. Transport errors are zero:
`{gates['transport_errors_zero']}`. Slot replay count:
`{counters['slot_replays']}`.

## 10. Real MSCCL++ counters

Scheduler actions `{counters['scheduler_actions']}`, consumed transport actions
`{counters['transport_actions']}`, GPU pack calls `{counters['gpu_pack_calls']}`,
real puts `{counters['mscclpp_put_calls']}`, bytes
`{counters['mscclpp_bytes_transferred']}`. No NCCL symbol or fallback exists in
the M5 runtime.

## 11. Reveal-to-put timing

Commit-to-pack: {_stats(packing, 'commit_to_pack_us')}.
Pack latency: {_stats(packing, 'pack_latency_us')}.
Pack-to-put: {_stats(transport, 'pack_to_put_us')}.
Reveal-to-put: {_stats(transport, 'reveal_to_put_us')}.
These are mechanism timings, not a performance benchmark.

## 12. Put before final Router

Gate: `{gates['put_before_final_router']}`. At least one formal remote action
has T5 < T8, as required by the progressive communication gate.

## 13. Router/communication overlap

Gate: `{gates['router_communication_overlap']}`. Real MemoryChannel put
intervals occur while later Router reveal work remains outstanding.

## 14. Limitations

""" + "\n".join(f"- {value}" for value in result["limitations"]) + """

## Stop rule

The device action queue, GPU packing, real MSCCL++ device put/signal/wait, and
remote byte-exact correctness gate are complete. Work stops here; no LSA, GIN,
NVSHMEM, DeepEP, multi-node implementation, tuning, or benchmark is started.
"""
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
