from __future__ import annotations

import polars as pl
import pytest

from assay.solvency.evaluation import (
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
    assert metrics.review_capacities[0].review_rows == 1
    assert metrics.review_capacities[0].precision == pytest.approx(1.0)
    assert metrics.review_capacities[1].review_rows == 5
    assert metrics.review_capacities[1].recall == pytest.approx(1.0)


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
