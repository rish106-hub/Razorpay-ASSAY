from __future__ import annotations

from datetime import date

from assay.evaluation.concentration import GroupConcentrationReport
from assay.evaluation.metrics import ConfidenceInterval, SignalMetricReport
from assay.evaluation.runner import EvaluationRunReport
from assay.evaluation.verdict import VerdictThresholds, decide_verdict


def test_verdict_blocks_training_when_holdout_evidence_is_insufficient() -> None:
    evaluation_report = EvaluationRunReport(
        run_id="a" * 64,
        observation_run_id="b" * 64,
        temporal_holdout_start=date(2026, 1, 1),
        geography_holdout_states=("MH",),
        development_states=("DL",),
        minimum_positive_outcomes_per_required_slice=20,
        signal_metrics=(),
        concentration_tests=(),
        required_slice_positive_outcomes={
            "overall": 10,
            "geography_holdout": 2,
            "temporal_holdout": 1,
        },
        evidence_status="insufficient_held_out_positive_outcomes",
    )

    verdict = decide_verdict(
        evaluation_report,
        VerdictThresholds(false_positive_review_cost_budget_inr=1_000_000),
    )

    assert verdict.verdict == "DO_NOT_SHIP"
    assert verdict.reason_codes == ("insufficient_held_out_positive_outcomes",)


def test_verdict_reunits_when_only_the_cohort_signal_passes() -> None:
    slices = ("overall", "geography_holdout", "temporal_holdout")
    base_interval = ConfidenceInterval(
        lower=0.01,
        upper=0.02,
        confidence_level=0.95,
        method="wilson_score",
    )
    metrics = tuple(
        SignalMetricReport.model_construct(
            slice_name=slice_name,
            signal_name=signal_name,
            precision_interval=ConfidenceInterval(
                lower=0.01 if signal_name == "shared_address" else 0.04,
                upper=0.06,
                confidence_level=0.95,
                method="wilson_score",
            ),
            base_rate_interval=base_interval,
            realized_review_capacity=0.01,
            requested_review_capacity=0.01,
            false_positive_review_cost_inr=100_000,
        )
        for signal_name in (
            "shared_address",
            "address_registration_month_cohort",
        )
        for slice_name in slices
    )
    concentrations = tuple(
        GroupConcentrationReport.model_construct(
            slice_name=slice_name,
            group_column=group_column,
            monte_carlo_p_value=0.01,
        )
        for group_column in (
            "address_group_key",
            "address_registration_month_cohort_key",
        )
        for slice_name in slices
    )
    evaluation_report = EvaluationRunReport(
        run_id="a" * 64,
        observation_run_id="b" * 64,
        temporal_holdout_start=date(2026, 1, 1),
        geography_holdout_states=("MH",),
        development_states=("DL",),
        minimum_positive_outcomes_per_required_slice=20,
        signal_metrics=metrics,
        concentration_tests=concentrations,
        required_slice_positive_outcomes={slice_name: 25 for slice_name in slices},
        evidence_status="evaluation_ready",
    )

    verdict = decide_verdict(
        evaluation_report,
        VerdictThresholds(false_positive_review_cost_budget_inr=1_000_000),
    )

    assert verdict.verdict == "RE_UNIT"
    assert verdict.selected_signal_name == "address_registration_month_cohort"
