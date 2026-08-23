"""Imbalance-aware metrics for merchant-risk signal audits."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm
from sklearn.metrics import average_precision_score

EVALUATION_METRICS_SCHEMA_VERSION = "1.0.0"


class SignalEvaluationError(RuntimeError):
    """A signal cannot be evaluated under the declared metric contract."""


class ConfidenceInterval(BaseModel):
    """Two-sided interval with its method declared."""

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    method: str


class SignalMetricReport(BaseModel):
    """Policy and fixed-review-capacity metrics for one evaluation slice."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = EVALUATION_METRICS_SCHEMA_VERSION
    slice_name: str = Field(min_length=1)
    signal_name: str = Field(min_length=1)
    score_column: str = Field(min_length=1)
    signal_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    population_rows: int = Field(ge=0)
    positive_outcome_rows: int = Field(ge=0)
    base_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    base_rate_interval: ConfidenceInterval | None = None
    flagged_rows: int = Field(ge=0)
    true_positive_rows: int = Field(ge=0)
    false_positive_rows: int = Field(ge=0)
    false_negative_rows: int = Field(ge=0)
    true_negative_rows: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    precision_interval: ConfidenceInterval | None = None
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_interval: ConfidenceInterval | None = None
    lift: float | None = Field(default=None, ge=0.0)
    average_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    requested_review_capacity: float = Field(gt=0.0, le=1.0)
    realized_review_capacity: float | None = Field(default=None, ge=0.0, le=1.0)
    review_score_threshold: float | None = None
    reviewed_rows: int = Field(ge=0)
    reviewed_true_positive_rows: int = Field(ge=0)
    review_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    review_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    review_lift: float | None = Field(default=None, ge=0.0)
    false_positive_review_cost_inr: float = Field(ge=0.0)
    review_cost_inr_per_false_positive: float = Field(ge=0.0)
    metric_status: str


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval | None:
    """Calculate a Wilson score interval for a binomial proportion."""

    if trials == 0:
        return None
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    z_score = float(norm.ppf(1 - (1 - confidence_level) / 2))
    proportion = successes / trials
    denominator = 1 + (z_score**2 / trials)
    centre = proportion + (z_score**2 / (2 * trials))
    margin = z_score * math.sqrt(
        (proportion * (1 - proportion) / trials)
        + (z_score**2 / (4 * trials**2))
    )
    lower_bound = max(0.0, (centre - margin) / denominator)
    upper_bound = min(1.0, (centre + margin) / denominator)
    return ConfidenceInterval(
        lower=0.0 if lower_bound < 1e-15 else lower_bound,
        upper=1.0 if upper_bound > 1 - 1e-15 else upper_bound,
        confidence_level=confidence_level,
        method="wilson_score",
    )


def evaluate_signal(
    observation_frame: pl.DataFrame,
    *,
    slice_name: str,
    signal_name: str,
    score_column: str,
    signal_column: str,
    target_column: str = "observed_adverse_outcome",
    eligibility_column: str = "evaluation_eligible",
    requested_review_capacity: float = 0.01,
    review_cost_inr_per_false_positive: float = 100.0,
) -> SignalMetricReport:
    """Evaluate a signal without using misleading majority-class accuracy."""

    if not 0 < requested_review_capacity <= 1:
        raise SignalEvaluationError("Review capacity must be in the interval (0, 1].")
    required_columns = {
        score_column,
        signal_column,
        target_column,
        eligibility_column,
    }
    if missing_columns := required_columns - set(observation_frame.columns):
        raise SignalEvaluationError(
            f"Observation frame is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    evaluation_frame = observation_frame.filter(pl.col(eligibility_column)).select(
        pl.col(score_column).cast(pl.Float64),
        pl.col(signal_column).cast(pl.Boolean),
        pl.col(target_column).cast(pl.Boolean),
    )
    population_rows = evaluation_frame.height
    if population_rows == 0:
        return SignalMetricReport(
            slice_name=slice_name,
            signal_name=signal_name,
            score_column=score_column,
            signal_column=signal_column,
            target_column=target_column,
            population_rows=0,
            positive_outcome_rows=0,
            flagged_rows=0,
            true_positive_rows=0,
            false_positive_rows=0,
            false_negative_rows=0,
            true_negative_rows=0,
            requested_review_capacity=requested_review_capacity,
            reviewed_rows=0,
            reviewed_true_positive_rows=0,
            false_positive_review_cost_inr=0.0,
            review_cost_inr_per_false_positive=(
                review_cost_inr_per_false_positive
            ),
            metric_status="empty_evaluation_slice",
        )
    if evaluation_frame[score_column].null_count() > 0:
        raise SignalEvaluationError("Signal scores cannot be null.")

    target = evaluation_frame[target_column]
    signal = evaluation_frame[signal_column]
    positive_outcome_rows = int(target.sum())
    flagged_rows = int(signal.sum())
    true_positive_rows = evaluation_frame.filter(
        pl.col(signal_column) & pl.col(target_column)
    ).height
    false_positive_rows = flagged_rows - true_positive_rows
    false_negative_rows = positive_outcome_rows - true_positive_rows
    true_negative_rows = (
        population_rows
        - true_positive_rows
        - false_positive_rows
        - false_negative_rows
    )
    base_rate = positive_outcome_rows / population_rows
    precision = true_positive_rows / flagged_rows if flagged_rows else None
    recall = (
        true_positive_rows / positive_outcome_rows
        if positive_outcome_rows
        else None
    )
    lift = precision / base_rate if precision is not None and base_rate > 0 else None

    requested_review_rows = max(
        1,
        math.ceil(population_rows * requested_review_capacity),
    )
    sorted_scores = evaluation_frame[score_column].sort(descending=True)
    review_score_threshold = float(sorted_scores[requested_review_rows - 1])
    reviewed_frame = evaluation_frame.filter(
        pl.col(score_column) >= review_score_threshold
    )
    reviewed_rows = reviewed_frame.height
    reviewed_true_positive_rows = int(reviewed_frame[target_column].sum())
    review_precision = reviewed_true_positive_rows / reviewed_rows
    review_recall = (
        reviewed_true_positive_rows / positive_outcome_rows
        if positive_outcome_rows
        else None
    )
    review_lift = review_precision / base_rate if base_rate > 0 else None
    average_precision = (
        float(
            average_precision_score(
                np.asarray(target, dtype=np.int8),
                np.asarray(evaluation_frame[score_column], dtype=np.float64),
            )
        )
        if positive_outcome_rows
        else None
    )
    metric_status = (
        "passed"
        if positive_outcome_rows > 0 and flagged_rows > 0
        else "insufficient_positive_or_flagged_rows"
    )
    return SignalMetricReport(
        slice_name=slice_name,
        signal_name=signal_name,
        score_column=score_column,
        signal_column=signal_column,
        target_column=target_column,
        population_rows=population_rows,
        positive_outcome_rows=positive_outcome_rows,
        base_rate=base_rate,
        base_rate_interval=wilson_interval(
            positive_outcome_rows,
            population_rows,
        ),
        flagged_rows=flagged_rows,
        true_positive_rows=true_positive_rows,
        false_positive_rows=false_positive_rows,
        false_negative_rows=false_negative_rows,
        true_negative_rows=true_negative_rows,
        precision=precision,
        precision_interval=wilson_interval(true_positive_rows, flagged_rows),
        recall=recall,
        recall_interval=wilson_interval(
            true_positive_rows,
            positive_outcome_rows,
        ),
        lift=lift,
        average_precision=average_precision,
        requested_review_capacity=requested_review_capacity,
        realized_review_capacity=reviewed_rows / population_rows,
        review_score_threshold=review_score_threshold,
        reviewed_rows=reviewed_rows,
        reviewed_true_positive_rows=reviewed_true_positive_rows,
        review_precision=review_precision,
        review_recall=review_recall,
        review_lift=review_lift,
        false_positive_review_cost_inr=(
            false_positive_rows * review_cost_inr_per_false_positive
        ),
        review_cost_inr_per_false_positive=review_cost_inr_per_false_positive,
        metric_status=metric_status,
    )
