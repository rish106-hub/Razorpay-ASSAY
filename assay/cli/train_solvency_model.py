"""Run the bounded merchant-solvency candidate search in Colab or locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from assay.solvency.feature_sets import (
    DEFAULT_SOLVENCY_FEATURE_SET_NAME,
    SOLVENCY_FEATURE_SETS,
)
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
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_SOLVENCY_FEATURE_SET_NAME,
        choices=sorted(SOLVENCY_FEATURE_SETS),
        help=(
            "Named feature contract to fit. baseline_with_status keeps the "
            "outcome-adjacent MCA company_status column; baseline_no_status "
            "drops it."
        ),
    )
    args = parser.parse_args()
    report = train_solvency_model(
        SolvencyModelTrainingConfig(
            fit_data_path=args.fit_data,
            expected_fit_sha256=args.fit_sha256,
            output_directory=args.output_directory,
            device=args.device,
            feature_set_name=args.feature_set,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
