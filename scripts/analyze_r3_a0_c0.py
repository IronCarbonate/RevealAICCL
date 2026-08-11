"""Independent read-back and summary recomputation for R3-A0/C0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CASE_ORDER = (
    "balanced",
    "skewed",
    "all_to_one_like",
    "zero_sized_pair",
    "empty_shard",
    "single_token_shard",
    "multiple_progressive_shards",
)
EXPECTED_GROUPS = ([0], [1], [2], [3], [4], [5], [6, 7])
ZERO_KEYS = (
    "runtime_bfs_calls",
    "full_rebuild_count",
    "unrevealed_execution",
    "future_access",
    "duplicate_dispatch",
    "stale_dispatch",
    "candidate_divergences",
    "action_divergences",
    "checker_divergences",
    "holder_divergences",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(array.max()),
    }


def _semantic_descriptor(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key] for key in (
            "chunk_ids", "sendcounts_tokens", "offsets_tokens", "token_count",
            "bytes", "payload_multiset_digest",
        )
    }


def _coverage(case_name: str, early: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pair_counts: list[int] = []
    destination_totals = [0, 0]
    descriptor_totals: list[int] = []
    for arm in early:
        for descriptor in arm["descriptors"]:
            descriptor_totals.append(int(descriptor["token_count"]))
            for destination, value in enumerate(descriptor["sendcounts_tokens"]):
                count = int(value)
                pair_counts.append(count)
                destination_totals[destination] += count
    total = sum(destination_totals)
    result = {
        "case": case_name,
        "pair_counts": pair_counts,
        "destination_totals": destination_totals,
        "zero_sized_pairs": sum(value == 0 for value in pair_counts),
        "distinct_pair_sizes": len(set(pair_counts)),
        "descriptor_totals": descriptor_totals,
    }
    if case_name == "balanced":
        passed = total > 0 and abs(destination_totals[0] - destination_totals[1]) / total < 0.10
    elif case_name == "skewed":
        passed = total > 0 and max(destination_totals) / total > 0.75
    elif case_name == "all_to_one_like":
        passed = destination_totals[0] == total and total > 0
    elif case_name == "zero_sized_pair":
        passed = destination_totals[1] == total and result["zero_sized_pairs"] > 0
    elif case_name == "empty_shard":
        passed = descriptor_totals.count(0) >= 2
    elif case_name == "single_token_shard":
        passed = descriptor_totals.count(1) >= 2
    elif case_name == "multiple_progressive_shards":
        passed = len(set(descriptor_totals)) >= 4
    else:
        raise ValueError("unexpected case")
    result["pass"] = passed
    return result


def analyze(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "R3_A0_C0_COMPLETE_PENDING_SUPERVISOR" or not result["pass"]:
        raise ValueError("canonical result is not complete/pass")
    ranks = sorted(result["rank_results"], key=lambda item: int(item["rank"]))
    if [int(item["rank"]) for item in ranks] != [0, 1]:
        raise ValueError("canonical artifact must contain exactly ranks 0 and 1")
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    descriptor_checks = 0
    semantic_checks = 0
    early_before_final_checks = 0
    delayed_after_final_checks = 0
    for rank_item in ranks:
        if [item["case"] for item in rank_item["cases"]] != list(CASE_ORDER):
            raise ValueError("case order/coverage mismatch")
        rank = int(rank_item["rank"])
        for pair in rank_item["cases"]:
            by_key[(rank, pair["case"])] = pair
            for arm_name in ("C", "D"):
                arm = pair[arm_name]
                if arm["transport"] != "A2Av-T0" or int(arm["total_sent_tokens"]) != 4096:
                    raise ValueError("transport/token cardinality mismatch")
                if int(arm["total_sent_bytes"]) != 4096 * 64:
                    raise ValueError("payload byte accounting mismatch")
                if len(arm["descriptors"]) != 7:
                    raise ValueError("incremental descriptor count mismatch")
                if [item["chunk_ids"] for item in arm["descriptors"]] != list(EXPECTED_GROUPS):
                    raise ValueError("partial75/checkpoint8 descriptor sequence mismatch")
                token_sum = 0
                for index, descriptor in enumerate(arm["descriptors"]):
                    counts = tuple(int(value) for value in descriptor["sendcounts_tokens"])
                    offsets = tuple(int(value) for value in descriptor["offsets_tokens"])
                    if len(counts) != 2 or offsets != (0, counts[0]):
                        raise ValueError("count/offset structure mismatch")
                    if int(descriptor["descriptor_index"]) != index:
                        raise ValueError("descriptor order mismatch")
                    if int(descriptor["token_count"]) != sum(counts):
                        raise ValueError("sendcount sum mismatch")
                    if int(descriptor["bytes"]) != sum(counts) * 64:
                        raise ValueError("descriptor byte accounting mismatch")
                    token_sum += sum(counts)
                    descriptor_checks += 1
                if token_sum != 4096:
                    raise ValueError("delta descriptors lose or duplicate source tokens")
                payload_calls = [
                    int(value["communication"]["payload_call_host_ns"])
                    for value in arm["descriptors"]
                ]
                final_router = int(arm["final_router_host_ns"])
                if arm_name == "C":
                    if payload_calls[0] >= final_router:
                        raise ValueError("early arm first payload call waited for final counts")
                    early_before_final_checks += 1
                else:
                    if any(value < final_router for value in payload_calls):
                        raise ValueError("delayed arm submitted before final router completion")
                    delayed_after_final_checks += len(payload_calls)
                semantic = arm["semantic"]
                if any(int(semantic[key]) != 0 for key in ZERO_KEYS):
                    raise ValueError("semantic zero invariant failed")
                if int(semantic["legal"]) != int(semantic["total"]) or not semantic["token_integrity"]:
                    raise ValueError("legality/token integrity failed")
                verification = arm["verification"]
                if not verification["pass"] or any(
                    int(verification[key]) != 0
                    for key in ("lost", "duplicate", "wrong_destination", "corruption", "unexpected")
                ):
                    raise ValueError("receive verification failed")
                semantic_checks += 1

    # Independently validate the count exchange's source/destination transpose.
    split_transpose_checks = 0
    for case_name in CASE_ORDER:
        for arm_name in ("C", "D"):
            arms = [by_key[(rank, case_name)][arm_name] for rank in (0, 1)]
            for descriptor_index in range(7):
                sends = [
                    tuple(int(value) for value in arm["descriptors"][descriptor_index]["sendcounts_tokens"])
                    for arm in arms
                ]
                for destination in (0, 1):
                    observed = tuple(
                        int(value) for value in arms[destination]["descriptors"][descriptor_index]["recvcounts_tokens"]
                    )
                    expected = (sends[0][destination], sends[1][destination])
                    if observed != expected:
                        raise ValueError("AlltoAllv recv split is not sendcount transpose")
                    split_transpose_checks += 1

    equivalence: list[dict[str, Any]] = []
    for rank in (0, 1):
        for case_name in CASE_ORDER:
            pair = by_key[(rank, case_name)]
            c, d = pair["C"], pair["D"]
            row = {
                "rank": rank,
                "case": case_name,
                "same_router_assignments": c["router_assignment_digest"] == d["router_assignment_digest"],
                "same_topk": c["topk_by_chunk_digests"] == d["topk_by_chunk_digests"],
                "same_final_payload_multiset": c["final_sent_payload_multiset_digest"] == d["final_sent_payload_multiset_digest"],
                "same_total_bytes": c["total_sent_bytes"] == d["total_sent_bytes"],
                "same_descriptors": [
                    _semantic_descriptor(value) for value in c["descriptors"]
                ] == [
                    _semantic_descriptor(value) for value in d["descriptors"]
                ],
                "same_scheduler_actions": c["scheduler_action_signatures"] == d["scheduler_action_signatures"],
            }
            row["pass"] = all(value for key, value in row.items() if key.startswith("same_"))
            equivalence.append(row)
    if not all(item["pass"] for item in equivalence):
        raise ValueError("early/delayed equivalence mismatch")

    coverage = [
        _coverage(case_name, [by_key[(rank, case_name)]["C"] for rank in (0, 1)])
        for case_name in CASE_ORDER
    ]
    if coverage != result["case_coverage"] or not all(item["pass"] for item in coverage):
        raise ValueError("case coverage recomputation mismatch")
    all_arms = [
        by_key[(rank, case_name)][arm]
        for rank in (0, 1) for case_name in CASE_ORDER for arm in ("C", "D")
    ]
    descriptors = [value for arm in all_arms for value in arm["descriptors"]]
    early = [by_key[(rank, case_name)]["C"] for rank in (0, 1) for case_name in CASE_ORDER]
    early_descriptors = [value for arm in early for value in arm["descriptors"]]
    pair_sizes = [int(value) for descriptor in early_descriptors for value in descriptor["sendcounts_tokens"]]
    diagnostics = {
        "router_final_latency_us": distribution([float(item["router_final_latency_us"]) for item in all_arms]),
        "router_chunk_cuda_us": distribution([float(value) for item in all_arms for value in item["router_chunk_cuda_us"]]),
        "count_offset_construction_us": distribution([float(value["count_offset_us"]) for value in descriptors]),
        "reference_packing_us": distribution([float(value["packing_us"]) for value in descriptors]),
        "payload_h2d_us": distribution([float(value["communication"]["h2d_us"]) for value in descriptors]),
        "count_exchange_completion_us": distribution([float(value["communication"]["count_completion_us"]) for value in descriptors]),
        "alltoallv_submit_us": distribution([float(value["communication"]["payload_submit_us"]) for value in descriptors]),
        "alltoallv_completion_us": distribution([float(value["communication"]["payload_completion_us"]) for value in descriptors]),
        "unpack_verification_us": distribution([float(item["verification"]["unpack_verification_us"]) for item in all_arms]),
        "total_payload_bytes": sum(int(item["total_sent_bytes"]) for item in all_arms),
    }
    if diagnostics != result["diagnostics"]:
        raise ValueError("diagnostic recomputation mismatch")
    traffic_summary = {
        "early_pair_token_counts": distribution(pair_sizes),
        "distinct_pair_sizes": len(set(pair_sizes)),
        "zero_sized_pairs": sum(value == 0 for value in pair_sizes),
        "nonzero_pair_min": min(value for value in pair_sizes if value > 0),
        "nonzero_pair_max": max(pair_sizes),
        "per_case": coverage,
    }
    if traffic_summary != result["traffic_distribution"]:
        raise ValueError("traffic distribution recomputation mismatch")
    return {
        "schema_version": 1,
        "study": "R3-A0/C0 independent read-back",
        "status": "PASS",
        "canonical_result": str(result_path),
        "canonical_result_sha256": sha256_file(result_path),
        "environment": result["environment"],
        "checks": {
            "descriptor_structure_checks": descriptor_checks,
            "semantic_arm_rank_case_checks": semantic_checks,
            "split_transpose_checks": split_transpose_checks,
            "early_delayed_equivalence_checks": len(equivalence),
            "case_coverage_checks": len(coverage),
            "early_first_payload_call_before_final_checks": early_before_final_checks,
            "delayed_payload_calls_after_final_checks": delayed_after_final_checks,
            "diagnostics_exact_recompute": True,
            "traffic_distribution_exact_recompute": True,
            "all_pass": True,
        },
        "early_delayed_equivalence": equivalence,
        "case_coverage": coverage,
        "traffic_distribution": traffic_summary,
        "diagnostics": diagnostics,
        "correctness": result["correctness"],
        "requirements": result["requirements"],
        "recommendation": "ELIGIBLE_FOR_SUPERVISOR_REVIEW_OF_R3_P0; NOT AUTOMATICALLY_AUTHORIZED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readback = analyze(args.result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(readback, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": readback["status"],
        "checks": readback["checks"],
        "result_sha256": readback["canonical_result_sha256"],
        "output": str(args.output),
    }, indent=1))


if __name__ == "__main__":
    main()
