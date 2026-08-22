"""Run the bounded merchant-solvency candidate search in Colab or locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assay.solvency.model_training import (
    SolvencyModelTrainingConfig,
    train_solvency_model,
)


def main() -> None:
    """Train from a checksum-pinned fit artifact and print export evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-data", required=True, type=Path)
    parser.add_argument("--fit-sha256", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = train_solvency_model(
        SolvencyModelTrainingConfig(
            fit_data_path=args.fit_data,
            expected_fit_sha256=args.fit_sha256,
            output_directory=args.output_directory,
            device=args.device,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
