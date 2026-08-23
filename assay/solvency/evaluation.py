"""Evaluate the solvency model once on frozen, unsampled holdouts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from assay.solvency.artifact import SolvencyModelPackage
from assay.solvency.feature_sets import (
    DEFAULT_SOLVENCY_FEATURE_SET_NAME,
    SOLVENCY_OUTCOME_ADJACENT_FEATURE,
)

SOLVENCY_HOLDOUT_EVALUATION_SCHEMA_VERSION = "1.2.0"
SOLVENCY_REVIEW_CAPACITIES = (0.001, 0.005, 0.02)
SOLVENCY_HOLDOUT_SPLITS = ("geography_test", "temporal_test")
SOLVENCY_STATE_SLICE_COLUMN = "state_code"
SOLVENCY_STATUS_SLICE_COLUMN = SOLVENCY_OUTCOME_ADJACENT_FEATURE


class SolvencyHoldoutEvaluationError(RuntimeError):
    """Frozen solvency holdout data failed its evaluation contract."""


class SolvencyReviewCapacityMetrics(BaseModel):
    """Operational retrieval metrics at a fixed review queue size."""

    model_config = ConfigDict(frozen=True)
    capacity_fraction: float = Field(gt=0, le=1)
    review_rows: int = Field(gt=0)
    score_threshold: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    lift: float = Field(ge=0)


class SolvencyStateSliceMetrics(BaseModel):
    """Generalisation evidence for one unseen-state test slice."""

    model_config = ConfigDict(frozen=True)
    state_code: str
    rows: int = Field(gt=0)
    positive_rows: int = Field(gt=0)
    base_rate: float = Field(gt=0, le=1)
    pr_auc: float = Field(ge=0, le=1)


class SolvencyStatusSliceMetrics(BaseModel):
    """Outcome concentration inside one MCA `company_status` value.

    This slice is reported whether or not `company_status` is a model feature.
    A status such as `Under CIRP` marks a company already inside the insolvency
    resolution process, so a slice that is a handful of rows and almost all
    positives is the shape of outcome contamination, not of a learned signal.
    `pr_auc` is undefined for a slice with no positive outcome and is left
    unset rather than reported as zero.
    """

    model_config = ConfigDict(frozen=True)
    company_status: str
    rows: int = Field(gt=0)
    positive_rows: int = Field(ge=0)
    base_rate: float = Field(ge=0, le=1)
    pr_auc: float | None = Field(default=None, ge=0, le=1)


class SolvencySplitMetrics(BaseModel):
    """Probability and retrieval metrics for one untouched holdout."""

    model_config = ConfigDict(frozen=True)
    dataset_split: str
    rows: int = Field(gt=0)
    positive_rows: int = Field(gt=0)
    base_rate: float = Field(gt=0, le=1)
    diagnostic_accuracy_at_0_5: float = Field(ge=0, le=1)
    pr_auc: float = Field(ge=0, le=1)
    roc_auc: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0, le=1)
    score_minimum: float = Field(ge=0, le=1)
    score_maximum: float = Field(ge=0, le=1)
    score_mean: float = Field(ge=0, le=1)
    review_capacities: tuple[SolvencyReviewCapacityMetrics, ...]
    state_slices: tuple[SolvencyStateSliceMetrics, ...] = ()
    status_slices: tuple[SolvencyStatusSliceMetrics, ...] = ()


class SolvencyHoldoutEvaluationReport(BaseModel):
    """Auditable full-holdout evidence with an explicit claim boundary."""

    model_config = ConfigDict(frozen=True)
    schema_version: str = SOLVENCY_HOLDOUT_EVALUATION_SCHEMA_VERSION
    label_boundary: str
    feature_set_name: str = DEFAULT_SOLVENCY_FEATURE_SET_NAME
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_data_path: Path
    holdouts_are_unsampled: bool
    evaluation_decision: str
    deployment_blockers: tuple[str, ...]
    split_metrics: dict[str, SolvencySplitMetrics]


def calculate_review_capacity_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    capacity_fraction: float,
) -> SolvencyReviewCapacityMetrics:
    """Calculate stable top-queue threshold, precision, recall, and lift."""

    review_rows = max(1, math.ceil(len(targets) * capacity_fraction))
    selected_indices = np.argsort(-scores, kind="stable")[:review_rows]
    selected_targets = targets[selected_indices]
    precision = float(selected_targets.mean())
    recall = float(selected_targets.sum() / targets.sum())
    base_rate = float(targets.mean())
    return SolvencyReviewCapacityMetrics(
        capacity_fraction=capacity_fraction,
        review_rows=review_rows,
        score_threshold=float(scores[selected_indices].min()),
        precision=precision,
        recall=recall,
        lift=precision / base_rate,
    )


def _grouped_slice_evidence(
    holdout_frame: pl.DataFrame,
    scores: np.ndarray,
    slice_column: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Split one holdout into name-sorted (value, targets, scores) groups."""

    scored_frame = holdout_frame.select(slice_column, "target").with_columns(
        pl.Series("score", scores)
    )
    grouped: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            str(group_key[0]),
            group_frame["target"].to_numpy(),
            group_frame["score"].to_numpy(),
        )
        for group_key, group_frame in scored_frame.group_by(
            slice_column, maintain_order=True
        )
    ]
    return sorted(grouped, key=lambda group: group[0])


def calculate_state_slice_metrics(
    holdout_frame: pl.DataFrame,
    scores: np.ndarray,
) -> tuple[SolvencyStateSliceMetrics, ...]:
    """Measure generalisation per unseen state and refuse a flat slice."""

    state_metrics: list[SolvencyStateSliceMetrics] = []
    for state_code, targets, state_scores in _grouped_slice_evidence(
        holdout_frame, scores, SOLVENCY_STATE_SLICE_COLUMN
    ):
        positive_rows = int(targets.sum())
        if positive_rows == 0 or positive_rows == len(targets):
            raise SolvencyHoldoutEvaluationError(
                f"Unseen-state slice is not evaluable: {state_code}."
            )
        state_metrics.append(
            SolvencyStateSliceMetrics(
                state_code=state_code,
                rows=len(targets),
                positive_rows=positive_rows,
                base_rate=float(targets.mean()),
                pr_auc=float(average_precision_score(targets, state_scores)),
            )
        )
    return tuple(state_metrics)


def calculate_status_slice_metrics(
    holdout_frame: pl.DataFrame,
    scores: np.ndarray,
) -> tuple[SolvencyStatusSliceMetrics, ...]:
    """Expose how much of the outcome sits inside one MCA status value."""

    status_metrics: list[SolvencyStatusSliceMetrics] = []
    for status_value, targets, status_scores in _grouped_slice_evidence(
        holdout_frame, scores, SOLVENCY_STATUS_SLICE_COLUMN
    ):
        positive_rows = int(targets.sum())
        status_metrics.append(
            SolvencyStatusSliceMetrics(
                company_status=status_value,
                rows=len(targets),
                positive_rows=positive_rows,
                base_rate=float(targets.mean()),
                pr_auc=(
                    float(average_precision_score(targets, status_scores))
                    if positive_rows > 0
                    else None
                ),
            )
        )
    return tuple(status_metrics)


def evaluate_solvency_split(
    package: SolvencyModelPackage,
    holdout_frame: pl.DataFrame,
    *,
    dataset_split: str,
) -> SolvencySplitMetrics:
    """Score one entity-disjoint holdout without tuning on its outcomes."""

    if holdout_frame.is_empty():
        raise SolvencyHoldoutEvaluationError(
            f"Solvency holdout is empty: {dataset_split}."
        )
    if holdout_frame["dataset_split"].unique().to_list() != [dataset_split]:
        raise SolvencyHoldoutEvaluationError(
            "Solvency holdout contains rows from another split."
        )
    targets = holdout_frame["target"].cast(pl.Int8).to_numpy()
    if int(targets.sum()) == 0 or int(targets.sum()) == len(targets):
        raise SolvencyHoldoutEvaluationError(
            "Solvency holdout must contain positive and negative outcomes."
        )
    scores = package.score(holdout_frame).to_numpy()
    review_capacities = tuple(
        calculate_review_capacity_metrics(targets, scores, capacity)
        for capacity in SOLVENCY_REVIEW_CAPACITIES
    )
    state_slices: tuple[SolvencyStateSliceMetrics, ...] = ()
    if dataset_split == "geography_test":
        state_slices = calculate_state_slice_metrics(holdout_frame, scores)
    status_slices: tuple[SolvencyStatusSliceMetrics, ...] = ()
    if SOLVENCY_STATUS_SLICE_COLUMN in holdout_frame.columns:
        status_slices = calculate_status_slice_metrics(holdout_frame, scores)
    return SolvencySplitMetrics(
        dataset_split=dataset_split,
        rows=holdout_frame.height,
        positive_rows=int(targets.sum()),
        base_rate=float(targets.mean()),
        diagnostic_accuracy_at_0_5=float(
            ((scores >= 0.5).astype(np.int8) == targets).mean()
        ),
        pr_auc=float(average_precision_score(targets, scores)),
        roc_auc=float(roc_auc_score(targets, scores)),
        brier_score=float(brier_score_loss(targets, scores)),
        score_minimum=float(scores.min()),
        score_maximum=float(scores.max()),
        score_mean=float(scores.mean()),
        review_capacities=review_capacities,
        state_slices=state_slices,
        status_slices=status_slices,
    )


def evaluate_solvency_holdouts(
    package: SolvencyModelPackage,
    model_data_path: Path,
) -> SolvencyHoldoutEvaluationReport:
    """Read and evaluate the two frozen full-population test partitions."""

    if not model_data_path.is_file():
        raise SolvencyHoldoutEvaluationError(
            f"Solvency model data is missing: {model_data_path}."
        )
    if SOLVENCY_STATUS_SLICE_COLUMN not in pl.scan_parquet(
        model_data_path
    ).collect_schema().names():
        raise SolvencyHoldoutEvaluationError(
            "Solvency model data must carry "
            f"{SOLVENCY_STATUS_SLICE_COLUMN} so that outcome concentration "
            "stays reportable even when it is not a model feature."
        )
    split_metrics: dict[str, SolvencySplitMetrics] = {}
    for dataset_split in SOLVENCY_HOLDOUT_SPLITS:
        holdout_frame = (
            pl.scan_parquet(model_data_path)
            .filter(pl.col("dataset_split") == dataset_split)
            .collect()
        )
        split_metrics[dataset_split] = evaluate_solvency_split(
            package, holdout_frame, dataset_split=dataset_split
        )
    return SolvencyHoldoutEvaluationReport(
        label_boundary=package.metrics.label_boundary,
        feature_set_name=package.metrics.feature_set_name,
        model_sha256=package.report.model_sha256,
        model_data_path=model_data_path,
        holdouts_are_unsampled=True,
        evaluation_decision="EVALUATED_NOT_PRODUCTION_READY",
        deployment_blockers=(
            "IBBI row-level reuse and deployment terms require legal approval.",
            "Merchant/payment telemetry has not been joined to this public-data baseline.",
            "Risk operations has not supplied a false-positive cost or review budget.",
        ),
        split_metrics=split_metrics,
    )
