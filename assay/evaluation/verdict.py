"""Deterministic ASSAY verdicts from frozen evaluation evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from assay.evaluation.metrics import SignalMetricReport
from assay.evaluation.runner import EvaluationRunReport

ASSAY_VERDICT_SCHEMA_VERSION = "1.0.0"
AssayVerdict = Literal["SHIP", "DO_NOT_SHIP", "RE_UNIT"]
REQUIRED_VERDICT_SLICES = frozenset(
    {"overall", "geography_holdout", "temporal_holdout"}
)


class VerdictError(RuntimeError):
    """Evaluation evidence cannot be mapped to a deterministic verdict."""


class VerdictThresholds(BaseModel):
    """Frozen evidence and merchant risk-operations acceptance thresholds."""

    model_config = ConfigDict(frozen=True)

    minimum_conservative_lift: float = Field(default=1.25, gt=1.0)
    maximum_group_concentration_p_value: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
    )
    maximum_review_capacity_expansion: float = Field(default=2.0, ge=1.0)
    false_positive_review_cost_budget_inr: float = Field(gt=0.0)


class SignalVerdictEvidence(BaseModel):
    """Pass/fail evidence for one signal unit across required slices."""

    model_config = ConfigDict(frozen=True)

    signal_name: str
    passed: bool
    failed_checks: tuple[str, ...]
    conservative_lift_by_slice: dict[str, float | None]


class VerdictReport(BaseModel):
    """Final deterministic verdict and its machine-readable reasons."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = ASSAY_VERDICT_SCHEMA_VERSION
    evaluation_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: AssayVerdict
    selected_signal_name: str | None
    reason_codes: tuple[str, ...]
    thresholds: VerdictThresholds
    signal_evidence: tuple[SignalVerdictEvidence, ...]


def _conservative_lift(metric: SignalMetricReport) -> float | None:
    if metric.precision_interval is None or metric.base_rate_interval is None:
        return None
    if metric.base_rate_interval.upper <= 0:
        return None
    return metric.precision_interval.lower / metric.base_rate_interval.upper


def _evaluate_signal_unit(
    evaluation_report: EvaluationRunReport,
    *,
    signal_name: str,
    group_column: str,
    thresholds: VerdictThresholds,
) -> SignalVerdictEvidence:
    metrics = {
        metric.slice_name: metric
        for metric in evaluation_report.signal_metrics
        if metric.signal_name == signal_name
        and metric.slice_name in REQUIRED_VERDICT_SLICES
    }
    concentration_tests = {
        concentration.slice_name: concentration
        for concentration in evaluation_report.concentration_tests
        if concentration.group_column == group_column
        and concentration.slice_name in REQUIRED_VERDICT_SLICES
    }
    failed_checks: list[str] = []
    conservative_lift_by_slice: dict[str, float | None] = {}
    for slice_name in sorted(REQUIRED_VERDICT_SLICES):
        metric = metrics.get(slice_name)
        if metric is None:
            failed_checks.append(f"{slice_name}:missing_metric")
            conservative_lift_by_slice[slice_name] = None
            continue
        conservative_lift = _conservative_lift(metric)
        conservative_lift_by_slice[slice_name] = conservative_lift
        if conservative_lift is None:
            failed_checks.append(f"{slice_name}:missing_confidence_interval")
        elif conservative_lift < thresholds.minimum_conservative_lift:
            failed_checks.append(f"{slice_name}:conservative_lift_below_threshold")
        if metric.realized_review_capacity is None:
            failed_checks.append(f"{slice_name}:missing_review_capacity")
        elif (
            metric.realized_review_capacity
            > metric.requested_review_capacity
            * thresholds.maximum_review_capacity_expansion
        ):
            failed_checks.append(f"{slice_name}:review_capacity_tie_expansion")
        if (
            metric.false_positive_review_cost_inr
            > thresholds.false_positive_review_cost_budget_inr
        ):
            failed_checks.append(f"{slice_name}:false_positive_cost_over_budget")
        concentration = concentration_tests.get(slice_name)
        if concentration is None:
            failed_checks.append(f"{slice_name}:missing_concentration_test")
        elif concentration.monte_carlo_p_value is None:
            failed_checks.append(f"{slice_name}:insufficient_concentration_evidence")
        elif (
            concentration.monte_carlo_p_value
            > thresholds.maximum_group_concentration_p_value
        ):
            failed_checks.append(f"{slice_name}:concentration_not_significant")
    return SignalVerdictEvidence(
        signal_name=signal_name,
        passed=not failed_checks,
        failed_checks=tuple(failed_checks),
        conservative_lift_by_slice=conservative_lift_by_slice,
    )


def decide_verdict(
    evaluation_report: EvaluationRunReport,
    thresholds: VerdictThresholds,
) -> VerdictReport:
    """Return SHIP, RE_UNIT, or DO_NOT_SHIP without model-generated judgment."""

    shared_address_evidence = _evaluate_signal_unit(
        evaluation_report,
        signal_name="shared_address",
        group_column="address_group_key",
        thresholds=thresholds,
    )
    cohort_evidence = _evaluate_signal_unit(
        evaluation_report,
        signal_name="address_registration_month_cohort",
        group_column="address_registration_month_cohort_key",
        thresholds=thresholds,
    )
    signal_evidence = (shared_address_evidence, cohort_evidence)
    if evaluation_report.evidence_status != "evaluation_ready":
        return VerdictReport(
            evaluation_run_id=evaluation_report.run_id,
            verdict="DO_NOT_SHIP",
            selected_signal_name=None,
            reason_codes=("insufficient_held_out_positive_outcomes",),
            thresholds=thresholds,
            signal_evidence=signal_evidence,
        )
    if shared_address_evidence.passed:
        return VerdictReport(
            evaluation_run_id=evaluation_report.run_id,
            verdict="SHIP",
            selected_signal_name="shared_address",
            reason_codes=("shared_address_passed_all_required_slices",),
            thresholds=thresholds,
            signal_evidence=signal_evidence,
        )
    if cohort_evidence.passed:
        return VerdictReport(
            evaluation_run_id=evaluation_report.run_id,
            verdict="RE_UNIT",
            selected_signal_name="address_registration_month_cohort",
            reason_codes=(
                "entity_signal_failed",
                "cohort_signal_passed_all_required_slices",
            ),
            thresholds=thresholds,
            signal_evidence=signal_evidence,
        )
    return VerdictReport(
        evaluation_run_id=evaluation_report.run_id,
        verdict="DO_NOT_SHIP",
        selected_signal_name=None,
        reason_codes=("no_signal_unit_passed_all_required_slices",),
        thresholds=thresholds,
        signal_evidence=signal_evidence,
    )
