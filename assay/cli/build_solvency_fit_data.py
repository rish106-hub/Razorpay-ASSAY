"""Build the sampled train-and-validation payload for a local solvency fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.solvency.fit_data import (
    SolvencyFitDataBuilder,
    SolvencyFitDataConfig,
    SolvencyFitDataError,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data-report", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = SolvencyFitDataBuilder(
            SolvencyFitDataConfig(
                model_data_report_path=arguments.model_data_report
            )
        ).run()
    except (ImmutableArtifactError, SolvencyFitDataError, ValueError) as error:
        print(f"Solvency fit-data build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "split_evidence": {
                    split_name: evidence.model_dump()
                    for split_name, evidence in report.split_evidence.items()
                },
                "artifact": report.fit_data_artifact.model_dump(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
