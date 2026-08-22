"""Evaluate ASSAY signals on overall, temporal, and unseen-state slices."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.evaluation.concentration import GroupConcentrationError
from assay.evaluation.metrics import SignalEvaluationError
from assay.evaluation.runner import (
    EvaluationRunConfig,
    EvaluationRunError,
    EvaluationRunner,
)


def _iso_date(raw_value: str) -> date:
    try:
        return date.fromisoformat(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected date in YYYY-MM-DD format") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run imbalance-aware ASSAY signal evaluation."
    )
    parser.add_argument("--observation-report", type=Path, required=True)
    parser.add_argument("--temporal-holdout-start", type=_iso_date, required=True)
    parser.add_argument("--review-capacity", type=float, default=0.01)
    parser.add_argument(
        "--false-positive-review-cost-inr",
        type=float,
        required=True,
        help="Measured analyst handling cost per false-positive review.",
    )
    parser.add_argument("--null-simulations", type=int, default=1_000)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = EvaluationRunner(
            EvaluationRunConfig(
                observation_report_path=arguments.observation_report,
                temporal_holdout_start=arguments.temporal_holdout_start,
                requested_review_capacity=arguments.review_capacity,
                review_cost_inr_per_false_positive=(
                    arguments.false_positive_review_cost_inr
                ),
                null_simulations=arguments.null_simulations,
            )
        ).run()
    except (
        EvaluationRunError,
        GroupConcentrationError,
        ImmutableArtifactError,
        SignalEvaluationError,
        OSError,
        ValueError,
    ) as error:
        print(f"Signal evaluation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_id": report.run_id,
                "geography_holdout_states": report.geography_holdout_states,
                "required_slice_positive_outcomes": (
                    report.required_slice_positive_outcomes
                ),
                "evidence_status": report.evidence_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
