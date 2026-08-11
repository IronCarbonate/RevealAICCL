"""Explicit, import-safe command line entry point for Phase 3B artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rlccl.uncertainty.ambiguity_experiment import (
    run_formal_experiment as _run_formal,
    run_toy_experiment,
)


DEFAULT_OUTPUT_DIRECTORY = Path("outputs/phase3b_ambiguity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 3B ambiguity experiment",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
        help="artifact destination (default: outputs/phase3b_ambiguity)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--toy",
        action="store_true",
        help="write deterministic synthetic smoke artifacts only",
    )
    mode.add_argument(
        "--formal",
        action="store_true",
        help="run the frozen formal path after separate supervisory approval",
    )
    return parser


def run_formal_experiment(output_directory: str | Path) -> dict[str, object]:
    """Forward to the explicit formal path without changing frozen inputs."""

    return _run_formal(output_directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.toy and not args.formal:
        parser.error("one explicit mode is required: --toy or --formal")
    destination = Path(args.output_dir)
    if args.toy:
        run_toy_experiment(destination)
    else:
        run_formal_experiment(destination)
    return 0


__all__ = ["build_parser", "main", "run_formal_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())
