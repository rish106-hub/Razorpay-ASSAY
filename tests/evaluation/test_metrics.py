from __future__ import annotations

import polars as pl

from assay.evaluation.metrics import evaluate_signal, wilson_interval


def test_signal_metrics_report_lift_and_tie_aware_review_capacity() -> None:
    observations = pl.DataFrame(
        {
            "evaluation_eligible": [True] * 10,
            "observed_adverse_outcome": [True, True] + [False] * 8,
            "shared_address_signal": [True, True, True, True] + [False] * 6,
            "shared_address_company_count": [4, 3, 3, 3, 1, 1, 1, 1, 1, 1],
        }
    )

    report = evaluate_signal(
        observations,
        slice_name="all_eligible_entities",
        signal_name="shared_address",
        score_column="shared_address_company_count",
        signal_column="shared_address_signal",
        requested_review_capacity=0.2,
        review_cost_inr_per_false_positive=250.0,
    )

    assert report.base_rate == 0.2
    assert report.precision == 0.5
    assert report.recall == 1.0
    assert report.lift == 2.5
    assert report.reviewed_rows == 4
    assert report.realized_review_capacity == 0.4
    assert report.review_precision == 0.5
    assert report.false_positive_review_cost_inr == 500.0


def test_wilson_interval_handles_zero_successes() -> None:
    interval = wilson_interval(0, 100)

    assert interval is not None
    assert interval.lower == 0.0
    assert 0.03 < interval.upper < 0.04
