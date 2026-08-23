"""Link explicit canonical MCA and NSE snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.linkage.entity_resolution import (
    EntityLinkageConfig,
    EntityLinkageError,
    EntityLinker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build conservative MCA-to-NSE entity-match decisions."
    )
    parser.add_argument("--mca-report", type=Path, required=True)
    parser.add_argument("--nse-report", type=Path, required=True)
    parser.add_argument(
        "--max-candidates-per-event",
        type=int,
        default=25,
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = EntityLinker(
            EntityLinkageConfig(
                mca_report_path=arguments.mca_report,
                nse_report_path=arguments.nse_report,
                max_persisted_candidates_per_event=(
                    arguments.max_candidates_per_event
                ),
            )
        ).run()
    except (
        EntityLinkageError,
        ImmutableArtifactError,
        OSError,
        ValueError,
    ) as error:
        print(f"Entity linkage failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "adverse_event_rows": report.adverse_event_rows,
                "accepted_exact_cin_rows": report.accepted_exact_cin_rows,
                "pending_exact_name_review_rows": (
                    report.pending_exact_name_review_rows
                ),
                "ambiguous_rows": report.ambiguous_rows,
                "unmatched_rows": report.unmatched_rows,
                "outcome_eligible_rows": report.outcome_eligible_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

