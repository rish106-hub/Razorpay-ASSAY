"""Verify a portable solvency model package exported by Colab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assay.solvency.artifact import load_solvency_model_package


def main() -> None:
    """Parse arguments and print checksum-verified package evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    package = load_solvency_model_package(
        args.run_directory, load_model=args.load_model
    )
    print(json.dumps(package.report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
