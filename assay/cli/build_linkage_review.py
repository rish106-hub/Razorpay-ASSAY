"""Build a deterministic MCA-to-NSE linkage review workbook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.linkage.review import LinkageReviewError, LinkageReviewSamplingConfig
from assay.linkage.review_runner import (
    LinkageReviewRunConfig,
    LinkageReviewRunError,
    LinkageReviewRunner,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stratified entity-linkage review workbook."
    )
    parser.add_argument("--linkage-report", type=Path, required=True)
    parser.add_argument("--nse-report", type=Path, required=True)
    parser.add_argument("--strict-name-events", type=int, default=200)
    parser.add_argument("--legal-name-events", type=int, default=200)
    parser.add_argument("--ambiguous-events", type=int, default=100)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = LinkageReviewRunner(
            LinkageReviewRunConfig(
                linkage_report_path=arguments.linkage_report,
                nse_report_path=arguments.nse_report,
                sampling=LinkageReviewSamplingConfig(
                    strict_name_review_events=arguments.strict_name_events,
                    legal_name_review_events=arguments.legal_name_events,
                    ambiguous_review_events=arguments.ambiguous_events,
                ),
            )
        ).run()
    except (
        ImmutableArtifactError,
        LinkageReviewError,
        LinkageReviewRunError,
        OSError,
        ValueError,
    ) as error:
        print(f"Linkage review build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "review_rows": report.review_rows,
                "review_event_rows": report.review_event_rows,
                "stratum_event_counts": report.stratum_event_counts,
                "csv_path": report.csv_artifact.path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

