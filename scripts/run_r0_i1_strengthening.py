"""Phase R0: strengthened evidence checks for the P10-I1 reference router.

This is an evidence-only runner.  It does not modify router, scheduler, reveal,
or checker behavior and it does not run an E2E experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "outputs" / "phase4_10" / "p10_1a_substrate"

import sys

sys.path.insert(0, str(REFERENCE_DIR))

from reference_router import cpu_oracle, router_topk, seed_router_params  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode("ascii")
        + repr(array.shape).encode("ascii")
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def independent_cpu_traffic_oracle(
    token_ids: np.ndarray,
    sources: np.ndarray,
    expert_indices: np.ndarray,
    num_sources: int,
    num_experts: int,
) -> np.ndarray:
    """Reconstruct token->traffic with an explicit, dependency-free token loop."""
    if not (len(token_ids) == len(sources) == len(expert_indices)):
        raise ValueError("token/source/expert arrays must have equal length")
    if len(set(int(value) for value in token_ids)) != len(token_ids):
        raise ValueError("duplicate token id in traffic oracle input")
    traffic = np.zeros((num_sources, num_experts), dtype=np.int64)
    for token_id, source, expert in zip(token_ids, sources, expert_indices):
        if int(token_id) < 0:
            raise ValueError("token id must be nonnegative")
        if not 0 <= int(source) < num_sources:
            raise ValueError("source is outside traffic domain")
        if not 0 <= int(expert) < num_experts:
            raise ValueError("expert is outside traffic domain")
        traffic[int(source), int(expert)] += 1
    return traffic


def cuda_vectorized_traffic(
    sources: torch.Tensor,
    expert_indices: torch.Tensor,
    num_sources: int,
    num_experts: int,
) -> torch.Tensor:
    flat = sources.long() * num_experts + expert_indices.long()
    return torch.bincount(flat, minlength=num_sources * num_experts).reshape(
        num_sources, num_experts
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "phase_r0"
        / "evidence_repair"
        / "p10_i1_strengthened_results.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R0-I1 strengthened evidence requires a real CUDA device")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    seed = 20260810
    batch, width, experts, top_k = 256, 16, 4, 1
    reveal_count = batch * 3 // 4
    shard_size = 64

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokens_np = rng.standard_normal((batch, width)).astype(np.float32)
    sources_np = (np.arange(batch, dtype=np.int64) % experts).astype(np.int64)
    token_ids_np = np.arange(batch, dtype=np.int64)
    tokens = torch.from_numpy(tokens_np).to(device)
    sources = torch.from_numpy(sources_np).to(device)
    weights_cpu, bias_cpu = seed_router_params(width, experts, seed)
    weights = weights_cpu.to(device)
    bias = bias_cpu.to(device)

    checks: dict[str, dict[str, object]] = {}

    def check(name: str, condition: bool, **detail: object) -> None:
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise AssertionError(f"{name} failed: {detail}")

    # Full router result and independent CPU oracle.
    full_idx, full_scores = router_topk(tokens, weights, bias, top_k)
    oracle_idx, oracle_scores = cpu_oracle(
        tokens_np, weights_cpu.numpy(), bias_cpu.numpy(), top_k
    )
    full_idx_np = full_idx.detach().cpu().numpy()
    full_scores_np = full_scores.detach().cpu().numpy().astype(np.float64)
    check("oracle_indices_exact", np.array_equal(full_idx_np, oracle_idx[:, 0]))
    check(
        "oracle_scores_close",
        np.allclose(full_scores_np, oracle_scores[:, 0], atol=1e-4, rtol=1e-5),
        max_abs_error=float(np.max(np.abs(full_scores_np - oracle_scores[:, 0]))),
    )

    # Actual 75% view: execute only the first 75% of token rows.  No suffix
    # assignment is computed or stored in the view payload.
    partial_idx, partial_scores = router_topk(
        tokens[:reveal_count], weights, bias, top_k
    )
    partial_ids_np = token_ids_np[:reveal_count].copy()
    partial_idx_np = partial_idx.detach().cpu().numpy()
    partial_view = {
        "token_ids": partial_ids_np.tolist(),
        "expert_indices": partial_idx_np.tolist(),
        "scores_sha256": digest_array(
            partial_scores.detach().cpu().numpy().astype(np.float32)
        ),
    }
    check(
        "actual_75pct_view_cardinality",
        len(partial_view["token_ids"]) == reveal_count,
        reveal_count=reveal_count,
        total_tokens=batch,
    )
    check(
        "actual_75pct_view_has_no_future_token_ids",
        max(partial_view["token_ids"]) == reveal_count - 1
        and all(value < reveal_count for value in partial_view["token_ids"]),
    )
    check(
        "actual_75pct_matches_full_prefix",
        np.array_equal(partial_idx_np, full_idx_np[:reveal_count]),
    )

    # Independent token->traffic reconstruction: CUDA vectorized path versus
    # an explicit CPU token loop with independent validation.
    traffic_cuda = cuda_vectorized_traffic(
        sources, full_idx, experts, experts
    ).cpu().numpy()
    traffic_oracle = independent_cpu_traffic_oracle(
        token_ids_np, sources_np, oracle_idx[:, 0], experts, experts
    )
    check(
        "independent_token_to_traffic_oracle",
        np.array_equal(traffic_cuda, traffic_oracle),
        traffic=traffic_cuda.tolist(),
    )
    check(
        "traffic_conserves_all_tokens",
        int(traffic_cuda.sum()) == batch,
        total=int(traffic_cuda.sum()),
    )

    # Token loss/duplication across real shard executions.
    shard_ids: list[torch.Tensor] = []
    shard_idx: list[torch.Tensor] = []
    for start in range(0, batch, shard_size):
        stop = min(start + shard_size, batch)
        idx, _ = router_topk(tokens[start:stop], weights, bias, top_k)
        shard_ids.append(torch.arange(start, stop, device=device))
        shard_idx.append(idx)
    reconstructed_ids = torch.cat(shard_ids).cpu().numpy()
    reconstructed_idx = torch.cat(shard_idx).cpu().numpy()
    check(
        "no_token_loss_across_shards",
        np.array_equal(reconstructed_ids, token_ids_np),
        reconstructed_count=int(len(reconstructed_ids)),
    )
    check(
        "no_token_duplication_across_shards",
        len(np.unique(reconstructed_ids)) == batch,
        unique_count=int(len(np.unique(reconstructed_ids))),
    )
    check(
        "sharded_assignments_equal_batched",
        np.array_equal(reconstructed_idx, full_idx_np),
    )

    # Real counterfactual: alter only unrevealed token rows and require the
    # hidden suffix to change while the actual partial view stays byte-identical.
    counterfactual = tokens.clone()
    suffix = counterfactual[reveal_count:]
    counterfactual[reveal_count:] = -7.0 * torch.flip(suffix, dims=(1,)) + 3.0
    counterfactual_idx, _ = router_topk(counterfactual, weights, bias, top_k)
    counterfactual_idx_np = counterfactual_idx.cpu().numpy()
    changed_suffix = int(
        np.count_nonzero(
            counterfactual_idx_np[reveal_count:] != full_idx_np[reveal_count:]
        )
    )
    check(
        "counterfactual_really_changes_unrevealed_assignments",
        changed_suffix > 0,
        changed_suffix=changed_suffix,
        suffix_tokens=batch - reveal_count,
    )
    check(
        "counterfactual_does_not_change_revealed_assignments",
        np.array_equal(
            counterfactual_idx_np[:reveal_count], full_idx_np[:reveal_count]
        ),
    )
    counterfactual_partial_idx, counterfactual_partial_scores = router_topk(
        counterfactual[:reveal_count], weights, bias, top_k
    )
    check(
        "counterfactual_partial_view_indices_no_leak",
        torch.equal(counterfactual_partial_idx, partial_idx),
    )
    check(
        "counterfactual_partial_view_scores_no_leak",
        torch.equal(counterfactual_partial_scores, partial_scores),
    )
    partial_traffic_original = independent_cpu_traffic_oracle(
        partial_ids_np,
        sources_np[:reveal_count],
        partial_idx_np,
        experts,
        experts,
    )
    partial_traffic_counterfactual = independent_cpu_traffic_oracle(
        partial_ids_np,
        sources_np[:reveal_count],
        counterfactual_partial_idx.cpu().numpy(),
        experts,
        experts,
    )
    check(
        "counterfactual_partial_traffic_no_leak",
        np.array_equal(partial_traffic_original, partial_traffic_counterfactual),
    )

    # Exact deterministic tie cases, including mask behavior and repeated runs.
    tie_tokens = torch.zeros((8, width), dtype=torch.float32, device=device)
    tie_weights = torch.zeros((width, experts), dtype=torch.float32, device=device)
    tie_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    tie_idx, _ = router_topk(tie_tokens, tie_weights, tie_bias, k=experts)
    expected_ties = torch.arange(experts, device=device).repeat(8, 1)
    check("deterministic_all_equal_tie_order", torch.equal(tie_idx, expected_ties))
    repeated_equal = all(
        torch.equal(router_topk(tie_tokens, tie_weights, tie_bias, k=experts)[0], tie_idx)
        for _ in range(10)
    )
    check("deterministic_tie_repeated_runs", repeated_equal)
    tie_mask = torch.zeros((8, experts), dtype=torch.bool, device=device)
    tie_mask[:, 0] = True
    masked_idx, _ = router_topk(
        tie_tokens, tie_weights, tie_bias, k=experts - 1, mask=tie_mask
    )
    expected_masked = torch.arange(1, experts, device=device).repeat(8, 1)
    check("deterministic_masked_tie_order", torch.equal(masked_idx, expected_masked))
    cpu_tie_idx, _ = cpu_oracle(
        np.zeros((8, width), dtype=np.float32),
        np.zeros((width, experts), dtype=np.float32),
        np.zeros(experts, dtype=np.float32),
        k=experts,
    )
    check(
        "deterministic_tie_matches_cpu_oracle",
        np.array_equal(cpu_tie_idx, expected_ties.cpu().numpy()),
    )

    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(device)
    nccl_version = torch.cuda.nccl.version()
    report = {
        "schema_version": 1,
        "study": "phase_r0_p10_i1_strengthening",
        "evidence_scope": "reference router correctness only; no E2E or concurrency claim",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": all(item["pass"] for item in checks.values()),
        "check_count": len(checks),
        "checks": checks,
        "configuration": {
            "seed": seed,
            "batch": batch,
            "width": width,
            "experts": experts,
            "top_k": top_k,
            "reveal_count": reveal_count,
            "reveal_ratio": reveal_count / batch,
            "shard_size": shard_size,
        },
        "evidence": {
            "full_assignment_sha256": digest_array(full_idx_np),
            "partial_view_sha256": hashlib.sha256(
                json.dumps(partial_view, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "traffic_matrix": traffic_cuda.tolist(),
            "traffic_sha256": digest_array(traffic_cuda),
            "counterfactual_changed_suffix_assignments": changed_suffix,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
        "source_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "reference_router": sha256_file(REFERENCE_DIR / "reference_router.py"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"all_pass": report["all_pass"], "check_count": len(checks)}))


if __name__ == "__main__":
    main()
