"""Publish a deterministic backend evidence-readiness contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.reporting.readiness import (
    EvidenceReadinessConfig,
    EvidenceReadinessError,
    EvidenceReadinessReporter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a frontend-safe merchant-risk evidence report."
    )
    parser.add_argument("--mca-report", type=Path, required=True)
    parser.add_argument("--nse-report", type=Path, required=True)
    parser.add_argument("--linkage-report", type=Path, required=True)
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--minimum-positive-outcomes", type=int, default=20)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = EvidenceReadinessReporter(
            EvidenceReadinessConfig(
                mca_report_path=arguments.mca_report,
                nse_report_path=arguments.nse_report,
                linkage_report_path=arguments.linkage_report,
                observation_report_path=arguments.observation_report,
                minimum_positive_outcomes_per_required_slice=(
                    arguments.minimum_positive_outcomes
                ),
            )
        ).run()
    except (EvidenceReadinessError, OSError, ValueError) as error:
        print(f"Evidence-readiness build failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report_id": report.report_id,
                "model_training_decision": report.model_readiness.decision,
                "model_readiness_status": report.model_readiness.status,
                "available_positive_outcomes": (
                    report.model_readiness.available_overall_positive_outcomes
                ),
                "positive_outcome_shortfall": (
                    report.model_readiness.positive_outcome_shortfall_before_holdouts
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
