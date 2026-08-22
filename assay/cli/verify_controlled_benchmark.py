"""Verify the local package exported by the controlled Colab benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.benchmark.artifact import (
    ControlledBenchmarkArtifactError,
    load_controlled_benchmark_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify checksums, feature order, and XGBoost model loading."
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--skip-model-load", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        package = load_controlled_benchmark_package(
            arguments.run_directory,
            load_model=not arguments.skip_model_load,
        )
    except (ControlledBenchmarkArtifactError, OSError, ValueError) as error:
        print(f"Controlled benchmark verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(package.report.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
