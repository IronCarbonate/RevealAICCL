#!/usr/bin/env python3
"""Render the evidence-backed R6-M6 NCCL Device Transport report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "phase_r6" / "m6_nccl_device"
REPORT = ROOT / "docs" / "phase_r6" / "R6_M6_NCCL_DEVICE_TRANSPORT_REPORT.md"


def _json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def main() -> int:
    results = _json("results.json")
    capability = _json("capability.json")
    correctness = _json("correctness.json")
    with (OUTPUT / "lsa_trace.csv").open(encoding="utf-8", newline="") as handle:
        trace = list(csv.DictReader(handle))
    progressive = sum(row["lsa_before_final_router"] == "True" for row in trace)
    counters = results["counters"]
    gates = results["gates"]
    cpu = results["cpu_participation"]
    layout = results["layout"]
    text = f"""# R6-M6 NCCL Device API Transport Report

## Result

**{results['claim']}**

The validated scope ends at remote registered receive-buffer decode. The LSA
path is real NCCL Device API traffic; GIN is not claimed as a runtime PASS.

## 1. NCCL version and Device API capability

The server system/PyTorch NCCL is `{capability['system_nccl_version']}`, which is
too old for the required `ncclCommQueryProperties`. Validation therefore uses
an independently selected NCCL `{capability['selected_nccl_version']}` build
(version code `{capability['selected_nccl_version_code']}`). Its queried
properties report `deviceApiSupport={str(capability['device_api_support']).lower()}`,
`nLsaTeams={capability['n_lsa_teams']}`, LSA team size
`{capability['lsa_size']}`, `multimemSupport={str(capability['multimem_support']).lower()}`,
and `ginType={capability['gin_type']}`.

## 2. Frozen boundary and modified surface

Router, RevealQueue, IncrementalState, FastBinder, DynamicGuard,
CommittedAction, DeviceActionQueue, GPU packing, descriptor/record layout, and
reveal/chunk policy remain frozen. M6 adds only the backend-neutral transport
contract, NCCL capability wrapper, NCCL native runtime, GIN compile surface,
formal runner/tests, reports, and results. M4/M5 sources are unchanged.

## 3. Real NCCL APIs used

Job-level host setup calls `ncclGetUniqueId`, `ncclCommInitRank`,
`ncclCommQueryProperties`, `ncclMemAlloc`, `ncclCommWindowRegister` with
`NCCL_WIN_COLL_SYMMETRIC`, and `ncclDevCommCreate`. Device LSA calls
`ncclGetPeerPointer`. Teardown calls `ncclDevCommDestroy`,
`ncclCommWindowDeregister`, `ncclMemFree`, and `ncclCommDestroy`. No NCCL
collective, MSCCL++, or host-staged fallback exists in the M6 runtime.

## 4. Symmetric window layout

Each rank owns one `{layout['capacity_bytes']}`-byte NCCL allocation and one
collectively registered symmetric window. The frozen region size is
`{layout['region_bytes']}`, descriptor stride `{layout['descriptor_stride']}`,
peer stride `{layout['peer_stride']}`, and record size `{layout['record_bytes']}`.
Every record preserves `{layout['metadata_fields']}` int64 metadata fields and
`{layout['feature_width']}` FP32 features. Logical/physical mapping is:
`{layout['logical_to_physical_mapping']}`.

## 5. LSA implementation

The four-block job kernel keeps the frozen router, scheduler, transport, and
remote-wait roles. For every remote physical action, the transport block packs
the existing slot, evaluates
`ncclGetPeerPointer(window, physical_dst_offset, dst_rank)`, and GPU threads
copy the exact packed bytes by load/store. It executed
`{counters['lsa_transfers']}` real peer transfers totaling
`{counters['lsa_bytes_transferred']}` bytes, alongside
`{counters['gpu_pack_calls']}` frozen GPU pack calls.

## 6. LSA completion

Completion ID is the frozen descriptor ID. The sender publishes
`ncclLsaBarrierSession<ncclCoopCta>::arrive(cuda::memory_order_release)` only
after the peer stores finish. The receiver uses
`wait(cuda::memory_order_acquire)` before post-job decode. Formal counters show
`{counters['lsa_arrives']}` arrives and `{counters['lsa_waits']}` waits. Thus
the protocol is payload visibility -> release publication -> acquire wait ->
decode, rather than assuming that issuing a store is completion.

## 7. Action to DeviceTransport/API mapping

`PhysicalTransportAction` maps one-to-one to
`put(peer, dst_offset, src_offset, bytes, completion_id)`: `peer=dst_rank`,
`dst_offset=physical_dst_offset`, `src_offset=physical_src_offset`,
`bytes=physical_bytes`, and `completion_id=descriptor_id`. Transport does not
reschedule, reread Router top-k, change rank/bytes/order, or merge/split an
action. Explicit backend values are `mscclpp`, `nccl_lsa`, and `nccl_gin`.

## 8. MSCCL++ reference equivalence

The same deterministic input and frozen M5 artifacts were used as reference.
Scheduler action divergence is `{gates['scheduler_action_divergence']}`;
MSCCL++ reference action divergence is
`{gates['mscclpp_reference_action_divergence']}` and reference payload
divergence is `{gates['mscclpp_reference_payload_divergence']}`. M5 reference
PASS is `{gates['mscclpp_reference_pass']}`.

## 9. CPU participation audit

Per descriptor: Python callback `{cpu['python_callback_per_descriptor']}`, CPU
poll `{cpu['cpu_poll_per_descriptor']}`, action construction
`{cpu['cpu_action_construction_per_descriptor']}`, packing
`{cpu['cpu_packing_per_descriptor']}`, transport submission
`{cpu['cpu_transport_submission_per_descriptor']}`, and CUDA launch
`{cpu['cpu_cuda_launch_per_descriptor']}`. CPU work is limited to job-level
communicator/window/devComm setup, one pipeline launch per rank, completion,
and post-job evidence collection.

## 10. Correctness

Across `{correctness['descriptor_source_destination_cases']}`
descriptor/source/destination cases: payload divergence
`{correctness['payload_divergence']}`, lost `{correctness['lost']}`, duplicate
`{correctness['duplicate']}`, wrong destination
`{correctness['wrong_destination']}`, and corruption
`{correctness['corruption']}`. Metadata validity is
`{correctness['metadata_valid']}` and Router destination validity is
`{correctness['router_destination_valid']}`.

## 11. Legality

Future access `{counters['future_access']}`, unrevealed access
`{counters['unrevealed_access']}`, stale action `{counters['stale_action']}`,
slot replay `{counters['slot_replays']}`, and transport errors
`{counters['transport_errors']}`. The two-rank clean-exit and legality gates
are both true.

## 12. Progressive evidence

`{progressive}` of `{len(trace)}` real remote LSA actions have
`communication_start < final_router_completion`; the remaining final action
per rank begins at/after that rank's final reveal. Therefore both
`lsa_before_final_router` and `router_communication_overlap` gates are
`{gates['lsa_before_final_router']}` and
`{gates['router_communication_overlap']}`. Trace times are mechanism evidence,
not a performance benchmark.

## 13. GIN integration and runtime status

Only after LSA PASS, the NCCL 2.29.7 header-backed GIN surface was compiled for
sm70/sm80/sm90. `ncclDevCommRequirements_t` requests one context, per-completion
signals/counters, and `NCCL_GIN_CONNECTION_FULL`. `ncclGin::put` directly uses
the action's peer/destination/source/bytes. Remote payload visibility maps to
`ncclGin_SignalInc` plus `waitSignal`; sender source reuse independently maps
to `ncclGin_CounterInc` plus `waitCounter`.

Runtime status is **{capability['gin_runtime_status']}**: queried GIN type is
`{capability['gin_type']}` and `{capability['gin_runtime_reason']}`. Hence real
GIN puts and network bytes are both zero, as recorded in `gin_trace.csv`; no
collective or MSCCL++ path is relabeled as GIN.

## 14. Blockers and limitations

- The validation host is one node with two V100 GPUs over NV2 and no exposed
  `/dev/infiniband`; true multi-node RDMA GIN execution cannot be performed.
- LSA validation is limited to two local ranks and one LSA team.
- The scope is forward packing/transport only; expert compute and return traffic
  remain out of scope.
- One remote action per rank/descriptor gives deterministic completion mapping.
- No tuning, coalescing, chunk changes, or formal performance benchmark was done.

## Stop rule

DeviceTransport abstraction, real NCCL LSA, byte-exact correctness, legality,
progressive overlap, GIN compilation, and capability detection are complete.
Because the environment lacks real GIN capability, work stops at
`GIN_RUNTIME_NOT_AVAILABLE` as required.
"""
    REPORT.write_text(text, encoding="utf-8")
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
