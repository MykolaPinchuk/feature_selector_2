"""Command-line entry point for running experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from fs_xgb.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-selection experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Path to experiment YAML config (defaults to fs_xgb/config/default_config.yaml).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where experiment outputs will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args.config, results_root=args.results_dir)
    print(f"Experiment finished. Results stored in: {result.run_dir}")


if __name__ == "__main__":
    main()
