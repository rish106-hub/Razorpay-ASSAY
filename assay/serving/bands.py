"""Derive an analyst triage scale from frozen holdout review capacities.

A band is not an invented constant. Every band boundary is the score a company
had to reach to enter a fixed-size review queue on the frozen prospective-time
holdout, and every band carries the retrieval metrics that queue actually
measured. The scale describes a queue position derived from public-data
evidence. It is not a fraud signal, and it is not an automated merchant
decision.
"""

from __future__ import annotations

from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assay.solvency.evaluation import (
    SOLVENCY_HOLDOUT_EVALUATION_SCHEMA_VERSION,
    SolvencyHoldoutEvaluationReport,
    SolvencyReviewCapacityMetrics,
)

RISK_BAND_SOURCE_SPLIT = "temporal_test"
RISK_BAND_SOURCE_SPLIT_DESCRIPTION = "prospective-time holdout"
RISK_BAND_ORDER = ("ELEVATED_REVIEW", "WATCH", "STANDARD", "LOW_SIGNAL")
RISK_BAND_CAPACITY_FRACTIONS = {
    "ELEVATED_REVIEW": 0.001,
    "WATCH": 0.005,
    "STANDARD": 0.02,
}
LOW_SIGNAL_INTERPRETATION = (
    "Outside every measured review-capacity queue on the "
    f"{RISK_BAND_SOURCE_SPLIT_DESCRIPTION}; no elevated public insolvency "
    "signal was found, and an absence of signal is not evidence of solvency."
)


class RiskBandError(RuntimeError):
    """A risk-band scale could not be derived from frozen holdout evidence."""


def _format_share(capacity_fraction: float) -> str:
    return f"{capacity_fraction * 100:.3g}%"


def _format_rate(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _queue_interpretation(capacity: SolvencyReviewCapacityMetrics) -> str:
    """Describe one capacity queue without asserting anything about a company."""

    return (
        f"Top {_format_share(capacity.capacity_fraction)} of the "
        f"{RISK_BAND_SOURCE_SPLIT_DESCRIPTION} queue; "
        f"{_format_rate(capacity.precision)} of the companies in this queue "
        "had a public IBBI CIRP announcement in the outcome window, covering "
        f"{_format_rate(capacity.recall)} of the announcements observed in "
        "that holdout."
    )


class RiskBandThreshold(BaseModel):
    """One queue boundary and the holdout evidence measured at it."""

    model_config = ConfigDict(frozen=True)

    band: str
    minimum_score: float = Field(ge=0, le=1)
    review_capacity_fraction: float | None
    holdout_precision: float | None
    holdout_recall: float | None
    holdout_lift: float | None
    interpretation: str


class RiskBandScale(BaseModel):
    """An ordered triage scale traceable to one frozen holdout split."""

    model_config = ConfigDict(frozen=True)

    source_split: str
    source_rows: int
    source_positive_rows: int
    source_base_rate: float
    thresholds: tuple[RiskBandThreshold, ...]

    @model_validator(mode="after")
    def _require_strictly_descending_thresholds(self) -> RiskBandScale:
        minimum_scores = [item.minimum_score for item in self.thresholds]
        for higher, lower in pairwise(minimum_scores):
            if higher <= lower:
                raise RiskBandError(
                    "Risk band thresholds must be strictly descending: "
                    f"{minimum_scores}."
                )
        return self

    @classmethod
    def from_evaluation(
        cls, report: SolvencyHoldoutEvaluationReport
    ) -> RiskBandScale:
        """Read the frozen source split and turn its queues into bands."""

        split = report.split_metrics.get(RISK_BAND_SOURCE_SPLIT)
        if split is None:
            raise RiskBandError(
                "Holdout evaluation report has no "
                f"{RISK_BAND_SOURCE_SPLIT} split to derive risk bands from."
            )
        for measured_capacity in split.review_capacities:
            if getattr(measured_capacity, "score_threshold", None) is None:
                raise RiskBandError(
                    "Holdout evaluation report records no review-capacity "
                    "score thresholds. Regenerate the evaluation artifact at "
                    f"schema {SOLVENCY_HOLDOUT_EVALUATION_SCHEMA_VERSION}."
                )
        capacities = {
            measured_capacity.capacity_fraction: measured_capacity
            for measured_capacity in split.review_capacities
        }
        thresholds: list[RiskBandThreshold] = []
        for band in RISK_BAND_ORDER[:-1]:
            capacity_fraction = RISK_BAND_CAPACITY_FRACTIONS[band]
            capacity = capacities.get(capacity_fraction)
            if capacity is None:
                raise RiskBandError(
                    "Holdout evaluation report has no review capacity at "
                    f"fraction {capacity_fraction} required by risk band "
                    f"{band}."
                )
            thresholds.append(
                RiskBandThreshold(
                    band=band,
                    minimum_score=capacity.score_threshold,
                    review_capacity_fraction=capacity_fraction,
                    holdout_precision=capacity.precision,
                    holdout_recall=capacity.recall,
                    holdout_lift=capacity.lift,
                    interpretation=_queue_interpretation(capacity),
                )
            )
        thresholds.append(
            RiskBandThreshold(
                band=RISK_BAND_ORDER[-1],
                minimum_score=0.0,
                review_capacity_fraction=None,
                holdout_precision=None,
                holdout_recall=None,
                holdout_lift=None,
                interpretation=LOW_SIGNAL_INTERPRETATION,
            )
        )
        return cls(
            source_split=split.dataset_split,
            source_rows=split.rows,
            source_positive_rows=split.positive_rows,
            source_base_rate=split.base_rate,
            thresholds=tuple(thresholds),
        )

    def band_for(self, score: float) -> RiskBandThreshold:
        """Return the highest queue band a single score qualifies for."""

        if not 0.0 <= score <= 1.0:
            raise RiskBandError(
                f"Risk band score must be within [0, 1]: {score}."
            )
        for threshold in self.thresholds:
            if score >= threshold.minimum_score:
                return threshold
        raise RiskBandError("Risk band scale has no baseline band.")
