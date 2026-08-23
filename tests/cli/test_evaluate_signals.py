from __future__ import annotations

import pytest

from assay.cli.evaluate_signals import _parser


def test_evaluation_cli_requires_measured_false_positive_cost() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--observation-report",
                "observation.json",
                "--temporal-holdout-start",
                "2026-01-01",
            ]
        )

    arguments = parser.parse_args(
        [
            "--observation-report",
            "observation.json",
            "--temporal-holdout-start",
            "2026-01-01",
            "--false-positive-review-cost-inr",
            "275.50",
        ]
    )
    assert arguments.false_positive_review_cost_inr == 275.50
