"""Two-process real MSCCL++ MemoryChannel smoke test for R6-M1."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rlccl.transport.mscclpp_native import MscclppNativeRuntime  # noqa: E402


def _worker(rank: int, library: str, endpoint: str, queue: mp.Queue) -> None:
    torch.cuda.set_device(rank)
    stream = torch.cuda.Stream(device=rank)
    buffer = torch.full((64,), -1, dtype=torch.int64, device=rank)
    source = torch.arange(8, dtype=torch.int64, device=rank) + rank * 1000
    buffer[:8].copy_(source)
    torch.cuda.synchronize(rank)
    started = time.monotonic_ns()
    try:
        with MscclppNativeRuntime(
            library, rank=rank, device=rank, buffer_ptr=buffer.data_ptr(),
            buffer_bytes=buffer.numel() * buffer.element_size(), endpoint=endpoint,
        ) as runtime:
            runtime.issue(dst_offset=8 * 8, src_offset=0, bytes=8 * 8,
                          stream=stream.cuda_stream)
            runtime.wait(stream=stream.cuda_stream)
            runtime.synchronize(stream=stream.cuda_stream)
            received = buffer[8:16].cpu().tolist()
            expected = (torch.arange(8, dtype=torch.int64) + (rank ^ 1) * 1000).tolist()
            queue.put({
                "rank": rank,
                "pass": received == expected,
                "received": received,
                "expected": expected,
                "elapsed_us": (time.monotonic_ns() - started) / 1e3,
                "counters": runtime.counters(),
            })
    except BaseException as error:
        queue.put({"rank": rank, "pass": False, "error": repr(error)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=50561)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    queue = context.Queue()
    endpoint = f"lo:127.0.0.1:{args.port}"
    processes = [
        context.Process(target=_worker, args=(rank, args.library, endpoint, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=120)
    result = {
        "schema_version": 1,
        "test": "real MSCCL++ CudaIpc MemoryChannel bidirectional put/signal/wait",
        "pid": os.getpid(),
        "pass": all(item.get("pass") for item in results)
                and all(process.exitcode == 0 for process in processes),
        "ranks": sorted(results, key=lambda item: item["rank"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
