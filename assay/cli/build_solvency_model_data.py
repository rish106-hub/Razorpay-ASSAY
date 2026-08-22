"""Build the frozen solvency modeling handoff for Colab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.solvency.training_data import (
    SolvencyTrainingDataBuilder,
    SolvencyTrainingDataConfig,
    SolvencyTrainingDataError,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-report", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = SolvencyTrainingDataBuilder(
            SolvencyTrainingDataConfig(
                observation_report_path=arguments.observation_report
            )
        ).run()
    except (ImmutableArtifactError, SolvencyTrainingDataError, ValueError) as error:
        print(f"Solvency model-data build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "split_decision": report.split_decision,
                "split_evidence": {
                    split_name: evidence.model_dump()
                    for split_name, evidence in report.split_evidence.items()
                },
                "artifact": report.model_data_artifact.model_dump(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
