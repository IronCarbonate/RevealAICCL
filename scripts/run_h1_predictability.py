#!/usr/bin/env python3
"""Run the frozen H1 predictability experiment or its deterministic toy test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rlccl.prediction.experiment import run_formal_experiment, run_toy_experiment


DEFAULT_OUTPUT_DIR = "outputs/h1_predictability"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen H1 predictability pipeline.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--formal",
        action="store_true",
        help="Run the exact frozen 75-sequence formal experiment.",
    )
    mode.add_argument(
        "--toy",
        action="store_true",
        help="Run the deterministic small end-to-end smoke experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.formal and not args.toy:
        parser.error("choose an explicit execution mode: --toy or --formal")

    if args.formal:
        result = run_formal_experiment(output_dir=args.output_dir)
    else:
        result = run_toy_experiment(output_dir=args.output_dir)

    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
