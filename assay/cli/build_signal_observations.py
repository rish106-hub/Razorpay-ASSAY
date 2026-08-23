"""Build leakage-safe shared-address signal observations."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.signals.observations import SignalObservationError
from assay.signals.runner import (
    SignalObservationRunConfig,
    SignalObservationRunError,
    SignalObservationRunner,
)


def _iso_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build as-of shared-address and incorporation-cohort signals."
    )
    parser.add_argument("--mca-report", type=Path, required=True)
    parser.add_argument("--nse-report", type=Path, required=True)
    parser.add_argument("--linkage-report", type=Path, required=True)
    parser.add_argument("--outcome-window-end", type=_iso_date, required=True)
    parser.add_argument("--shared-address-minimum-companies", type=int, default=2)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = SignalObservationRunner(
            SignalObservationRunConfig(
                mca_report_path=arguments.mca_report,
                nse_report_path=arguments.nse_report,
                linkage_report_path=arguments.linkage_report,
                outcome_window_end=arguments.outcome_window_end,
                shared_address_minimum_companies=(
                    arguments.shared_address_minimum_companies
                ),
            )
        ).run()
    except (
        ImmutableArtifactError,
        SignalObservationError,
        SignalObservationRunError,
        OSError,
        ValueError,
    ) as error:
        print(f"Signal observation build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "company_rows": report.company_rows,
                "evaluation_eligible_rows": report.evaluation_eligible_rows,
                "observed_adverse_outcome_rows": (
                    report.observed_adverse_outcome_rows
                ),
                "shared_address_signal_rows": report.shared_address_signal_rows,
                "address_registration_month_cohort_signal_rows": (
                    report.address_registration_month_cohort_signal_rows
                ),
                "quality_status": report.quality_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

