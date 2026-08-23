from __future__ import annotations

import polars as pl
import pytest

from assay.solvency.evaluation import (
    SOLVENCY_REVIEW_CAPACITIES,
    SolvencyHoldoutEvaluationError,
    evaluate_solvency_split,
)


class _FixedScorePackage:
    def score(self, merchant_feature_frame: pl.DataFrame) -> pl.Series:
        return merchant_feature_frame["fixed_score"].rename(
            "cirp_public_announcement_score"
        )


def test_evaluate_solvency_split_reports_review_capacity() -> None:
    frame = pl.DataFrame(
        {
            "dataset_split": ["temporal_test"] * 1000,
            "target": [1, 1, 0, 0] + [0] * 996,
            "fixed_score": [0.9, 0.8, 0.7, 0.6] + [0.0] * 996,
        }
    )

    metrics = evaluate_solvency_split(
        _FixedScorePackage(),  # type: ignore[arg-type]
        frame,
        dataset_split="temporal_test",
    )

    assert metrics.pr_auc == pytest.approx(1.0)
    assert len(metrics.review_capacities) == len(SOLVENCY_REVIEW_CAPACITIES)
    assert metrics.review_capacities[0].review_rows == 1
    assert metrics.review_capacities[0].precision == pytest.approx(1.0)
    assert metrics.review_capacities[0].score_threshold == pytest.approx(0.9)
    assert metrics.review_capacities[1].review_rows == 5
    assert metrics.review_capacities[1].recall == pytest.approx(1.0)
    assert metrics.review_capacities[1].score_threshold == pytest.approx(0.0)
    assert metrics.review_capacities[2].review_rows == 20
    assert metrics.review_capacities[2].score_threshold == pytest.approx(0.0)


def test_evaluate_solvency_split_rejects_split_contamination() -> None:
    frame = pl.DataFrame(
        {
            "dataset_split": ["temporal_test", "training"],
            "target": [1, 0],
            "fixed_score": [0.9, 0.1],
        }
    )

    with pytest.raises(SolvencyHoldoutEvaluationError, match="another split"):
        evaluate_solvency_split(
            _FixedScorePackage(),  # type: ignore[arg-type]
            frame,
            dataset_split="temporal_test",
        )


def test_evaluate_solvency_split_reports_company_status_concentration() -> None:
    """The status breakdown has to survive `company_status` leaving the model."""

    frame = pl.DataFrame(
        {
            "dataset_split": ["temporal_test"] * 1000,
            "company_status": ["Under CIRP"] * 4 + ["Active"] * 996,
            "target": [1, 1, 1, 0] + [1] + [0] * 995,
            "fixed_score": [0.9, 0.8, 0.7, 0.6] + [0.5] + [0.0] * 995,
        }
    )

    metrics = evaluate_solvency_split(
        _FixedScorePackage(),  # type: ignore[arg-type]
        frame,
        dataset_split="temporal_test",
    )

    slices = {
        slice_metrics.company_status: slice_metrics
        for slice_metrics in metrics.status_slices
    }
    assert set(slices) == {"Active", "Under CIRP"}
    assert slices["Under CIRP"].rows == 4
    assert slices["Under CIRP"].positive_rows == 3
    assert slices["Under CIRP"].base_rate == pytest.approx(0.75)
    assert slices["Active"].rows == 996
    assert slices["Active"].positive_rows == 1
    assert metrics.status_slices[0].company_status == "Active"


def test_status_slices_leave_pr_auc_unset_without_a_positive_outcome() -> None:
    frame = pl.DataFrame(
        {
            "dataset_split": ["geography_test"] * 4,
            "state_code": ["kerala"] * 4,
            "company_status": ["Active", "Active", "Strike Off", "Strike Off"],
            "target": [1, 0, 0, 0],
            "fixed_score": [0.9, 0.2, 0.1, 0.05],
        }
    )

    metrics = evaluate_solvency_split(
        _FixedScorePackage(),  # type: ignore[arg-type]
        frame,
        dataset_split="geography_test",
    )

    slices = {
        slice_metrics.company_status: slice_metrics
        for slice_metrics in metrics.status_slices
    }
    assert slices["Strike Off"].positive_rows == 0
    assert slices["Strike Off"].base_rate == pytest.approx(0.0)
    assert slices["Strike Off"].pr_auc is None
    assert slices["Active"].pr_auc == pytest.approx(1.0)


def test_status_slices_are_omitted_when_the_column_is_absent() -> None:
    frame = pl.DataFrame(
        {
            "dataset_split": ["temporal_test"] * 4,
            "target": [1, 0, 0, 0],
            "fixed_score": [0.9, 0.2, 0.1, 0.05],
        }
    )

    metrics = evaluate_solvency_split(
        _FixedScorePackage(),  # type: ignore[arg-type]
        frame,
        dataset_split="temporal_test",
    )

    assert metrics.status_slices == ()
