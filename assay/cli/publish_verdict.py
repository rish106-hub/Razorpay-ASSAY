"""Publish an ASSAY SHIP, DO_NOT_SHIP, or RE_UNIT verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.evaluation.verdict import VerdictError
from assay.evaluation.verdict_runner import (
    VerdictPublicationConfig,
    VerdictPublicationError,
    VerdictPublisher,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a deterministic merchant-risk signal verdict."
    )
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument(
        "--false-positive-review-cost-budget-inr",
        type=float,
        required=True,
        help="Approved maximum false-positive review spend per evaluation slice.",
    )
    parser.add_argument("--minimum-conservative-lift", type=float, default=1.25)
    parser.add_argument(
        "--maximum-group-concentration-p-value",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum-review-capacity-expansion",
        type=float,
        default=2.0,
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        publication = VerdictPublisher(
            VerdictPublicationConfig(
                evaluation_report_path=arguments.evaluation_report,
                false_positive_review_cost_budget_inr=(
                    arguments.false_positive_review_cost_budget_inr
                ),
                minimum_conservative_lift=arguments.minimum_conservative_lift,
                maximum_group_concentration_p_value=(
                    arguments.maximum_group_concentration_p_value
                ),
                maximum_review_capacity_expansion=(
                    arguments.maximum_review_capacity_expansion
                ),
            )
        ).run()
    except (OSError, ValueError, VerdictError, VerdictPublicationError) as error:
        print(f"Verdict publication failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": publication.run_id,
                "verdict": publication.verdict.verdict,
                "selected_signal_name": (
                    publication.verdict.selected_signal_name
                ),
                "reason_codes": publication.verdict.reason_codes,
                "report_path": publication.report_path,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
