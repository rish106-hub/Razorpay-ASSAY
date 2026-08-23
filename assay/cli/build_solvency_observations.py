"""Build exact-CIN MCA-to-IBBI merchant-solvency observations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.solvency.observations import SolvencyObservationError
from assay.solvency.runner import (
    SolvencyObservationRunConfig,
    SolvencyObservationRunError,
    SolvencyObservationRunner,
)


def _parse_date(raw_date: str) -> date:
    return date.fromisoformat(raw_date)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mca-report", type=Path, required=True)
    parser.add_argument("--ibbi-report", type=Path, required=True)
    parser.add_argument("--outcome-window-end", type=_parse_date, required=True)
    parser.add_argument("--temporal-holdout-start", type=_parse_date, required=True)
    arguments = parser.parse_args()
    try:
        report = SolvencyObservationRunner(
            SolvencyObservationRunConfig(
                mca_report_path=arguments.mca_report,
                ibbi_report_path=arguments.ibbi_report,
                outcome_window_end=arguments.outcome_window_end,
                temporal_holdout_start=arguments.temporal_holdout_start,
            )
        ).run()
    except (
        ImmutableArtifactError,
        SolvencyObservationError,
        SolvencyObservationRunError,
        ValueError,
    ) as error:
        print(f"Solvency observation build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "evaluation_eligible_rows": report.evaluation_eligible_rows,
                "observed_outcome_rows": report.observed_outcome_rows,
                "unseen_geography_positive_rows": (
                    report.unseen_geography_positive_rows
                ),
                "temporal_holdout_positive_rows": (
                    report.temporal_holdout_positive_rows
                ),
                "positive_base_rate": report.positive_base_rate,
                "training_decision": report.training_decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
