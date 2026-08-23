from __future__ import annotations

from pathlib import Path

import pytest

from assay.serving.bands import (
    RISK_BAND_ORDER,
    RiskBandError,
    RiskBandScale,
)
from assay.solvency.evaluation import (
    SolvencyHoldoutEvaluationReport,
    SolvencyReviewCapacityMetrics,
    SolvencySplitMetrics,
)

BANNED_TERMS = ("fraud", "fraudulent", "reject", "block", "decline")


def _capacity(
    capacity_fraction: float,
    score_threshold: float,
    precision: float,
    recall: float,
    lift: float,
) -> SolvencyReviewCapacityMetrics:
    return SolvencyReviewCapacityMetrics(
        capacity_fraction=capacity_fraction,
        review_rows=max(1, int(100_000 * capacity_fraction)),
        score_threshold=score_threshold,
        precision=precision,
        recall=recall,
        lift=lift,
    )


def _legacy_capacity(
    capacity_fraction: float,
) -> SolvencyReviewCapacityMetrics:
    """Build a schema-1.0.0 capacity record that never stored a threshold."""

    return SolvencyReviewCapacityMetrics.model_construct(
        capacity_fraction=capacity_fraction,
        review_rows=max(1, int(100_000 * capacity_fraction)),
        precision=0.3771,
        recall=0.3415,
        lift=341.41,
    )


def _measured_capacities() -> tuple[SolvencyReviewCapacityMetrics, ...]:
    return (
        _capacity(0.001, 0.038758847, 0.377104377, 0.341463414, 341.411677),
        _capacity(0.005, 0.007238911, 0.088888888, 0.402439024, 80.475609),
        _capacity(0.02, 0.003305471, 0.029461279, 0.533536585, 26.672787),
    )


def _report(
    capacities: tuple[SolvencyReviewCapacityMetrics, ...],
    *,
    dataset_split: str = "temporal_test",
) -> SolvencyHoldoutEvaluationReport:
    split = SolvencySplitMetrics(
        dataset_split=dataset_split,
        rows=100_000,
        positive_rows=110,
        base_rate=0.0011,
        diagnostic_accuracy_at_0_5=0.999,
        pr_auc=0.34280,
        roc_auc=0.93781,
        brier_score=0.000758,
        score_minimum=0.000003,
        score_maximum=0.878938,
        score_mean=0.000693,
        review_capacities=capacities,
    )
    return SolvencyHoldoutEvaluationReport(
        label_boundary="cirp_public_announcement_outcome_not_fraud",
        model_sha256="a" * 64,
        model_data_path=Path("solvency_model_data.parquet"),
        holdouts_are_unsampled=True,
        evaluation_decision="EVALUATED_NOT_PRODUCTION_READY",
        deployment_blockers=("Legal clearance is outstanding.",),
        split_metrics={dataset_split: split},
    )


def test_scale_derives_every_band_from_measured_review_capacities() -> None:
    scale = RiskBandScale.from_evaluation(_report(_measured_capacities()))

    assert scale.source_split == "temporal_test"
    assert scale.source_rows == 100_000
    assert scale.source_positive_rows == 110
    assert scale.source_base_rate == pytest.approx(0.0011)
    assert tuple(item.band for item in scale.thresholds) == RISK_BAND_ORDER
    elevated, watch, standard, low_signal = scale.thresholds
    assert elevated.minimum_score == pytest.approx(0.038758847)
    assert elevated.review_capacity_fraction == pytest.approx(0.001)
    assert elevated.holdout_precision == pytest.approx(0.377104377)
    assert elevated.holdout_recall == pytest.approx(0.341463414)
    assert elevated.holdout_lift == pytest.approx(341.411677)
    assert watch.minimum_score == pytest.approx(0.007238911)
    assert watch.review_capacity_fraction == pytest.approx(0.005)
    assert standard.minimum_score == pytest.approx(0.003305471)
    assert standard.review_capacity_fraction == pytest.approx(0.02)
    assert low_signal.minimum_score == 0.0
    assert low_signal.review_capacity_fraction is None
    assert low_signal.holdout_precision is None
    assert low_signal.holdout_recall is None
    assert low_signal.holdout_lift is None


def test_scale_thresholds_are_strictly_descending() -> None:
    scale = RiskBandScale.from_evaluation(_report(_measured_capacities()))

    minimum_scores = [item.minimum_score for item in scale.thresholds]

    assert minimum_scores == sorted(minimum_scores, reverse=True)
    assert len(set(minimum_scores)) == len(minimum_scores)


def test_interpretations_describe_a_queue_and_never_a_verdict() -> None:
    scale = RiskBandScale.from_evaluation(_report(_measured_capacities()))

    for threshold in scale.thresholds:
        wording = threshold.interpretation.lower()
        assert not any(term in wording for term in BANNED_TERMS)
    assert "Top 0.1%" in scale.thresholds[0].interpretation
    assert "37.7%" in scale.thresholds[0].interpretation
    assert "Top 2%" in scale.thresholds[2].interpretation
    assert "not evidence of solvency" in scale.thresholds[3].interpretation


def test_band_for_returns_the_band_at_and_below_each_threshold() -> None:
    scale = RiskBandScale.from_evaluation(_report(_measured_capacities()))

    assert scale.band_for(1.0).band == "ELEVATED_REVIEW"
    assert scale.band_for(0.038758847).band == "ELEVATED_REVIEW"
    assert scale.band_for(0.038758846).band == "WATCH"
    assert scale.band_for(0.007238911).band == "WATCH"
    assert scale.band_for(0.007238910).band == "STANDARD"
    assert scale.band_for(0.003305471).band == "STANDARD"
    assert scale.band_for(0.003305470).band == "LOW_SIGNAL"
    assert scale.band_for(0.0).band == "LOW_SIGNAL"


@pytest.mark.parametrize("score", [-0.000001, 1.000001, 2.0])
def test_band_for_rejects_a_score_outside_the_unit_interval(
    score: float,
) -> None:
    scale = RiskBandScale.from_evaluation(_report(_measured_capacities()))

    with pytest.raises(RiskBandError, match=r"within \[0, 1\]"):
        scale.band_for(score)


def test_missing_source_split_is_rejected() -> None:
    report = _report(_measured_capacities(), dataset_split="geography_test")

    with pytest.raises(RiskBandError, match="temporal_test"):
        RiskBandScale.from_evaluation(report)


def test_missing_review_capacity_is_rejected_by_fraction() -> None:
    measured = _measured_capacities()
    report = _report((measured[0], measured[1]))

    with pytest.raises(RiskBandError, match="fraction 0.02"):
        RiskBandScale.from_evaluation(report)


def test_report_without_score_thresholds_asks_for_regeneration() -> None:
    report = _report((_legacy_capacity(0.001), _legacy_capacity(0.005)))

    with pytest.raises(RiskBandError, match="schema 1.1.0"):
        RiskBandScale.from_evaluation(report)


def test_non_monotonic_capacity_thresholds_are_rejected() -> None:
    report = _report(
        (
            _capacity(0.001, 0.004, 0.377104377, 0.341463414, 341.411677),
            _capacity(0.005, 0.009, 0.088888888, 0.402439024, 80.475609),
            _capacity(0.02, 0.003305471, 0.029461279, 0.533536585, 26.672787),
        )
    )

    with pytest.raises(RiskBandError, match="strictly descending"):
        RiskBandScale.from_evaluation(report)
