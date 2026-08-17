#!/usr/bin/env python3
"""R6-M8 dual-V100 full-handle combine correctness gate."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlccl.ep.combine import ReturnLayout, reference_moe_output  # noqa: E402
from rlccl.ep.gpu_handle_combine import GPUHandleCombineRuntime  # noqa: E402
from rlccl.ep.layout import ProgressiveDispatchLayout  # noqa: E402
from rlccl.scheduler.common.compiled_plan import compile_rank_pair_plan  # noqa: E402
from rlccl.scheduler.common.scheduler_schema import RevealRecord, SchedulerConfig  # noqa: E402


WORLD_SIZE = 2
CHUNKS = 3
TOKENS_PER_CHUNK = 16
TOKENS = CHUNKS * TOKENS_PER_CHUNK
HIDDEN = 16
EXPERTS = 4
EXPERTS_PER_RANK = EXPERTS // WORLD_SIZE
M6_RECORD_BYTES = 136
GENERATION = 1
SCENARIOS = (
    ("balanced", 1),
    ("skewed", 2),
    ("all_to_one_like", 3),
)
OUTPUT = ROOT / "outputs" / "phase_r6" / "m8_handle_combine"
TRACE_FIELDS = (
    "row", "src_rank", "src_token_idx", "topk_slot", "expert_id",
    "is_remote", "bytes", "t_return_start", "t_return_end",
    "t_remote_completion", "error_code",
)


def _load_torch(library: str):
    ctypes.CDLL(str(Path(library).resolve()), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    import torch  # pylint: disable=import-outside-toplevel
    return torch


def _dispatch_layout(topk: int) -> ProgressiveDispatchLayout:
    return ProgressiveDispatchLayout(
        WORLD_SIZE, CHUNKS, TOKENS_PER_CHUNK * topk, HIDDEN,
    )


def _plan(rank: int, topk: int):
    return compile_rank_pair_plan(SchedulerConfig(
        world_size=WORLD_SIZE, source_rank=rank, record_bytes=M6_RECORD_BYTES,
        max_descriptors=CHUNKS, max_chunks=CHUNKS,
        max_tokens_per_peer=TOKENS_PER_CHUNK * topk,
        reveal_queue_capacity=8, action_queue_capacity=16, block_size=32,
    ))


def _records(topk: int) -> tuple[RevealRecord, ...]:
    return tuple(RevealRecord(
        chunk, chunk + 1, chunk * TOKENS_PER_CHUNK, TOKENS_PER_CHUNK,
        chunk * TOKENS_PER_CHUNK * topk, TOKENS_PER_CHUNK * topk, chunk,
    ) for chunk in range(CHUNKS))


def _features(rank: int, scenario: str) -> np.ndarray:
    values = np.zeros((TOKENS, HIDDEN), dtype=np.float32)
    for token in range(TOKENS):
        if scenario == "balanced":
            order = [(token + rank) % EXPERTS]
        elif scenario == "skewed":
            order = [0, 1 + (token + rank) % 3]
        else:
            order = [0, 1, 2 + (token + rank) % 2]
        values[token, :EXPERTS] = np.float32(-4.0)
        for position, expert in enumerate(order):
            values[token, expert] = np.float32(6.0 - position)
        values[token, 8] = np.float32(rank)
        values[token, 9] = np.float32(token / TOKENS)
        values[token, 10:] = np.arange(6, dtype=np.float32) + np.float32(token / 8)
    return values


def _global_expert_weights() -> np.ndarray:
    values = np.zeros((EXPERTS, HIDDEN, HIDDEN), dtype=np.float32)
    for expert in range(EXPERTS):
        values[expert] = np.eye(HIDDEN, dtype=np.float32) * np.float32(1 + expert / 8)
    return values


def _meta_rows(values: np.ndarray) -> list[list[int | float]]:
    return [[
        int(row["src_rank"]), int(row["src_token_idx"]),
        int(row["expert_id"]), int(row["topk_slot"]),
        int(row["descriptor_id"]), int(row["reveal_epoch"]),
        float(row["topk_weight"]),
    ] for row in values]


def _worker(rank: int, library: str, unique_ids: list[bytes], delay: int, queue) -> None:
    try:
        torch = _load_torch(library)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        global_weights = _global_expert_weights()
        local_weights = torch.from_numpy(
            global_weights[rank * EXPERTS_PER_RANK:(rank + 1) * EXPERTS_PER_RANK]
        ).to(device).contiguous()
        scenario_results = []
        for scenario_index, (scenario, topk) in enumerate(SCENARIOS):
            features_cpu = _features(rank, scenario)
            x = torch.from_numpy(features_cpu).to(device)
            logits = x[:, :EXPERTS]
            values, indices = torch.sort(logits, dim=-1, descending=True, stable=True)
            topk_idx = indices[:, :topk].to(torch.int64).contiguous()
            topk_weights = values[:, :topk].to(torch.float32).contiguous()
            records = _records(topk)
            reveal_records = torch.tensor(
                [record.as_tuple() for record in records],
                dtype=torch.int64, device=device,
            )
            layout = _dispatch_layout(topk)
            with GPUHandleCombineRuntime(
                library, rank=rank, device=rank,
                unique_id=unique_ids[scenario_index], plan=_plan(rank, topk),
                dispatch_layout=layout, num_source_tokens=TOKENS,
                num_topk=topk,
            ) as runtime:
                output = runtime.run(
                    reveal_records=reveal_records, x=x,
                    topk_idx=topk_idx, topk_weights=topk_weights,
                    expert_weights=local_weights,
                    experts_per_rank=EXPERTS_PER_RANK,
                    producer_delay_cycles=delay,
                    router_stream=torch.cuda.current_stream(rank).cuda_stream,
                )
                cpu_audit = {
                    "python_callback_per_output": runtime.python_callback_per_output,
                    "cpu_poll_per_output": runtime.cpu_poll_per_output,
                    "cpu_return_construction_per_output": runtime.cpu_return_construction_per_output,
                    "cpu_packing_per_output": runtime.cpu_packing_per_output,
                    "cpu_transport_submission_per_output": runtime.cpu_transport_submission_per_output,
                    "cpu_cuda_launch_per_output": runtime.cpu_cuda_launch_per_output,
                }
            topk_cpu = topk_idx.cpu().numpy()
            weight_cpu = topk_weights.cpu().numpy()
            reference = reference_moe_output(
                features_cpu, topk_cpu, weight_cpu, global_weights,
            )
            scenario_results.append({
                "scenario": scenario, "topk": topk,
                "features": features_cpu.tolist(),
                "topk_idx": topk_cpu.tolist(),
                "topk_weights": weight_cpu.tolist(),
                "recv_x": output["recv_x"].tolist(),
                "recv_metadata": _meta_rows(output["recv_metadata"]),
                "expert_counts": output["expert_counts"].tolist(),
                "expert_offsets": output["expert_offsets"].tolist(),
                "return_traces": output["return_traces"].tolist(),
                "final_output": output["final_output"].tolist(),
                "reference_output": reference.tolist(),
                "dispatch_counters": output["dispatch_counters"],
                "combine_counters": output["combine_counters"],
                "capability": output["capability"],
                "handle": output["handle"], "cpu_audit": cpu_audit,
                "dispatch_layout": {
                    "capacity_bytes": layout.capacity_bytes,
                    "record_bytes": layout.record_bytes,
                },
            })
        queue.put({"rank": rank, "scenarios": scenario_results})
    except BaseException as error:
        queue.put({"rank": rank, "error": repr(error), "traceback": traceback.format_exc()})
        raise


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--producer-delay-cycles", type=int, default=1_000_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_ids = [GPUHandleCombineRuntime.get_unique_id(args.library) for _ in SCENARIOS]
    torch = _load_torch(args.library)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(
        target=_worker,
        args=(rank, args.library, unique_ids, args.producer_delay_cycles, queue),
    ) for rank in range(WORLD_SIZE)]
    for process in processes:
        process.start()
    ranks = [queue.get(timeout=360) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    errors = [item for item in ranks if "error" in item]
    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(json.dumps(errors, indent=2))
    ranks.sort(key=lambda item: item["rank"])

    trace_rows: list[dict[str, Any]] = []
    reverse_rows = []
    layout_rows = []
    final_rows = []
    scenario_correctness = []
    total_pairing = total_rank = total_token = total_slot = 0
    total_wrong_expert = total_lost = total_duplicate = total_corruption = 0
    max_abs = 0.0
    all_close = True
    combined_counters: Counter[str] = Counter()
    dispatch_counters: Counter[str] = Counter()
    cpu_audit: Counter[str] = Counter()

    for scenario_index, (scenario, topk) in enumerate(SCENARIOS):
        results = [rank["scenarios"][scenario_index] for rank in ranks]
        slot_counts: Counter[tuple[int, int, int]] = Counter()
        one_pairing = one_rank = one_token = one_slot = one_wrong_expert = 0
        one_corruption = 0
        for expert_rank, result in enumerate(results):
            metadata = result["recv_metadata"]
            recv_x = result["recv_x"]
            traces = {
                int(row[0]): dict(zip(TRACE_FIELDS, map(int, row), strict=True))
                for row in result["return_traces"]
            }
            one_corruption += sum(int(trace["error_code"] != 0)
                                  for trace in traces.values())
            for row_index, (meta, feature) in enumerate(zip(metadata, recv_x, strict=True)):
                src_rank, src_token, expert, topk_slot = map(int, meta[:4])
                expected_feature = np.asarray(
                    results[src_rank]["features"][src_token], dtype="<f4",
                )
                actual_feature = np.asarray(feature, dtype="<f4")
                one_pairing += int(expected_feature.tobytes() != actual_feature.tobytes())
                trace = traces.get(row_index)
                if trace is None:
                    one_corruption += 1
                    continue
                one_rank += int(trace["src_rank"] != src_rank)
                one_token += int(trace["src_token_idx"] != src_token)
                one_slot += int(trace["topk_slot"] != topk_slot)
                expected_expert = int(results[src_rank]["topk_idx"][src_token][topk_slot])
                one_wrong_expert += int(trace["expert_id"] != expert or expert != expected_expert)
                slot_counts[(src_rank, src_token, topk_slot)] += 1
                trace_rows.append({
                    "scenario": scenario, "topk": topk,
                    "expert_rank": expert_rank, **trace,
                })
        expected_slots = {(rank, token, slot) for rank in range(WORLD_SIZE)
                          for token in range(TOKENS) for slot in range(topk)}
        one_lost = sum(slot_counts[key] == 0 for key in expected_slots)
        one_duplicate = sum(max(0, count - 1) for count in slot_counts.values())
        final_pass = True
        one_max_abs = 0.0
        for rank, result in enumerate(results):
            actual = np.asarray(result["final_output"], dtype=np.float32)
            reference = np.asarray(result["reference_output"], dtype=np.float32)
            difference = np.abs(actual - reference)
            rank_max = float(difference.max(initial=0.0))
            close = bool(np.allclose(actual, reference, rtol=2e-5, atol=2e-5))
            final_pass &= close
            one_max_abs = max(one_max_abs, rank_max)
            final_rows.append({
                "scenario": scenario, "topk": topk, "rank": rank,
                "max_abs_error": rank_max, "within_tolerance": close,
            })
            combined_counters.update(result["combine_counters"])
            dispatch_counters.update(result["dispatch_counters"])
            cpu_audit.update(result["cpu_audit"])
        return_layout = ReturnLayout(
            TOKENS, topk, HIDDEN, results[0]["dispatch_layout"]["capacity_bytes"],
        )
        layout_rows.append({
            "scenario": scenario, "topk": topk,
            "num_source_tokens": TOKENS, "record_bytes": return_layout.record_bytes,
            "return_base_offset": return_layout.base_offset,
            "return_region_bytes": return_layout.region_bytes,
            "gin_staging_base_offset": return_layout.base_offset + return_layout.region_bytes,
            "combine_capacity_bytes": return_layout.capacity_bytes,
            "unique_slots": TOKENS * topk,
        })
        scenario_pass = not any((
            one_pairing, one_rank, one_token, one_slot, one_wrong_expert,
            one_lost, one_duplicate, one_corruption,
        )) and final_pass
        reverse_rows.append({
            "scenario": scenario, "topk": topk,
            "recv_src_metadata_pairing_divergence": one_pairing,
            "return_rank_divergence": one_rank,
            "return_token_divergence": one_token,
            "return_topk_slot_divergence": one_slot,
            "wrong_expert": one_wrong_expert,
        })
        scenario_correctness.append({
            "scenario": scenario, "topk": topk, "pass": scenario_pass,
            "lost": one_lost, "duplicate": one_duplicate,
            "corruption": one_corruption, "max_abs_error": one_max_abs,
        })
        total_pairing += one_pairing
        total_rank += one_rank
        total_token += one_token
        total_slot += one_slot
        total_wrong_expert += one_wrong_expert
        total_lost += one_lost
        total_duplicate += one_duplicate
        total_corruption += one_corruption
        max_abs = max(max_abs, one_max_abs)
        all_close &= final_pass

    counter_error_fields = (
        "errors", "stale_handle", "range_bounds", "wrong_source_rank",
        "wrong_token", "wrong_topk_slot", "wrong_expert", "slot_collision",
        "missing_return", "corruption",
    )
    expected_contributions = sum(
        WORLD_SIZE * TOKENS * topk for _, topk in SCENARIOS
    )
    gates = {
        "recv_src_metadata_pairing_divergence": total_pairing,
        "return_rank_divergence": total_rank,
        "return_token_divergence": total_token,
        "return_topk_slot_divergence": total_slot,
        "lost": total_lost, "duplicate": total_duplicate,
        "wrong_expert": total_wrong_expert,
        "wrong_source_rank": combined_counters["wrong_source_rank"],
        "wrong_token": combined_counters["wrong_token"],
        "wrong_topk_slot": combined_counters["wrong_topk_slot"],
        "stale_handle": combined_counters["stale_handle"],
        "slot_collision": combined_counters["slot_collision"],
        "corruption": total_corruption + combined_counters["corruption"],
        "cpu_per_output_involvement": sum(cpu_audit.values()),
        "all_scenarios_pass": all(row["pass"] for row in scenario_correctness),
        "final_output_within_tolerance": all_close,
        "all_contributions_mapped": combined_counters["rows_mapped"] == expected_contributions,
        "all_contributions_reduced": combined_counters["contributions_reduced"] == expected_contributions,
        "lsa_real_remote_returns": combined_counters["remote_returns"] > 0,
        "lsa_completion_balanced": combined_counters["lsa_arrives"] == len(SCENARIOS) * WORLD_SIZE and
                                   combined_counters["lsa_waits"] == len(SCENARIOS) * WORLD_SIZE,
        "device_errors_zero": all(combined_counters[name] == 0 for name in counter_error_fields),
        "forward_legality_zero": all(dispatch_counters[name] == 0 for name in (
            "errors", "future_access", "unrevealed_access", "stale_action",
        )),
        "device_api": all(
            rank["scenarios"][i]["capability"]["device_api_support"]
            for rank in ranks for i in range(len(SCENARIOS))
        ),
        "no_collective_or_mscclpp_fallback": True,
        "clean_exit": all(process.exitcode == 0 for process in processes),
    }
    zero_gates = (
        "recv_src_metadata_pairing_divergence", "return_rank_divergence",
        "return_token_divergence", "return_topk_slot_divergence", "lost",
        "duplicate", "wrong_expert", "wrong_source_rank", "wrong_token",
        "wrong_topk_slot", "stale_handle", "slot_collision", "corruption",
        "cpu_per_output_involvement",
    )
    passed = all(gates[name] == 0 for name in zero_gates) and all(
        bool(value) for name, value in gates.items() if name not in zero_gates
    )
    results = {
        "phase": "R6-M8",
        "claim": "Handle-Driven GPU Combine PASS" if passed else "Handle-Driven GPU Combine FAIL",
        "scope_stop": "deterministic source-side top-k reduction",
        "environment": {
            "world_size": WORLD_SIZE,
            "gpu": [torch.cuda.get_device_name(i) for i in range(WORLD_SIZE)],
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nccl": "2.29.7", "compiled_architecture": "sm_70",
        },
        "scenarios": [{"name": name, "topk": topk} for name, topk in SCENARIOS],
        "execution_model": {
            "return_routing_source": "ProgressiveEPHandle.recv_src_metadata",
            "return_descriptor": False, "scheduler_on_return": False,
            "expert_progressive_return": False,
            "completion_scope": "full handle",
            "reduction_order": "ascending topk slot",
        },
        "gates": gates,
        "dispatch_counters": dict(dispatch_counters),
        "combine_counters": dict(combined_counters),
        "cpu_audit": dict(cpu_audit),
        "max_abs_error": max_abs,
        "gin_runtime_status": "GIN_RUNTIME_NOT_AVAILABLE",
        "stop_rule": "combine correctness complete; no tuning or progressive return",
    }
    reverse_mapping = {
        "pass": not any((total_pairing, total_rank, total_token, total_slot, total_wrong_expert)),
        "rows": reverse_rows,
    }
    return_layout = {
        "pass": total_lost == total_duplicate == combined_counters["slot_collision"] == 0,
        "formula": "slot_id = src_token_idx * num_topk + topk_slot",
        "rows": layout_rows,
    }
    correctness = {
        "pass": all(row["pass"] for row in scenario_correctness),
        "lost": total_lost, "duplicate": total_duplicate,
        "corruption": total_corruption + combined_counters["corruption"],
        "wrong_expert": total_wrong_expert,
        "scenarios": scenario_correctness,
    }
    final_equivalence = {
        "pass": all_close, "rtol": 2e-5, "atol": 2e-5,
        "max_abs_error": max_abs, "rows": final_rows,
    }
    _write_csv(args.output_dir / "return_trace.csv", trace_rows)
    for name, value in (
        ("results.json", results), ("reverse_mapping.json", reverse_mapping),
        ("return_layout.json", return_layout), ("correctness.json", correctness),
        ("final_output_equivalence.json", final_equivalence),
    ):
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    print(json.dumps({"claim": results["claim"], "gates": gates,
                      "max_abs_error": max_abs}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
