"""Apply a completed linkage review workbook to a new decision run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.linkage.review_application import (
    ReviewedLinkageApplicationConfig,
    ReviewedLinkageRunError,
    ReviewedLinkageRunner,
)
from assay.linkage.review_ingestion import LinkageReviewIngestionError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and apply completed human entity-match reviews."
    )
    parser.add_argument("--linkage-report", type=Path, required=True)
    parser.add_argument("--review-report", type=Path, required=True)
    parser.add_argument("--completed-review-csv", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = ReviewedLinkageRunner(
            ReviewedLinkageApplicationConfig(
                linkage_report_path=arguments.linkage_report,
                review_report_path=arguments.review_report,
                completed_review_csv_path=arguments.completed_review_csv,
            )
        ).run()
    except (
        ImmutableArtifactError,
        LinkageReviewIngestionError,
        ReviewedLinkageRunError,
        OSError,
        ValueError,
    ) as error:
        print(f"Linkage review application failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "outcome_eligible_rows": report.outcome_eligible_rows,
                "accepted_reviewed_event_rows": (
                    report.review_application.accepted_reviewed_event_rows
                ),
                "unresolved_review_event_rows": (
                    report.review_application.unresolved_review_event_rows
                ),
                "review_quality_status": report.review_quality_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

