from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from assay.api.app import create_app
from assay.reporting.readiness import (
    EvidenceReadinessReport,
    LinkageCoverageSummary,
    ModelReadinessSummary,
    SignalPopulationSummary,
    SourceEvidenceSummary,
)
from assay.solvency.evaluation import (
    SolvencyHoldoutEvaluationReport,
    SolvencyReviewCapacityMetrics,
    SolvencySplitMetrics,
)


def _write_evidence_report(report_path: Path) -> EvidenceReadinessReport:
    report = EvidenceReadinessReport(
        report_id="a" * 64,
        source_evidence=SourceEvidenceSummary(
            mca_source_snapshot_id="b" * 64,
            nse_source_snapshot_id="c" * 64,
            entity_linkage_run_id="d" * 64,
            signal_observation_run_id="e" * 64,
            mca_quality_status="passed_with_quarantine",
            signal_observation_quality_status="passed",
            nse_provenance_complete_assets=0,
        ),
        linkage_coverage=LinkageCoverageSummary(
            adverse_event_rows=16_165,
            accepted_exact_cin_rows=40,
            pending_exact_name_review_rows=2_134,
            ambiguous_rows=58,
            unmatched_rows=13_933,
            exact_cin_coverage_rate=40 / 16_165,
            human_review_required=True,
        ),
        signal_population=SignalPopulationSummary(
            company_rows=3_674_314,
            evaluation_eligible_rows=2_920_186,
            observed_adverse_outcome_rows=8,
            shared_address_signal_rows=466_414,
            shared_address_signal_rate=466_414 / 2_920_186,
            address_registration_month_cohort_signal_rows=174_025,
            address_registration_month_cohort_signal_rate=174_025 / 2_920_186,
        ),
        model_readiness=ModelReadinessSummary(
            decision="DO_NOT_TRAIN",
            status="blocked_insufficient_entity_outcomes",
            available_overall_positive_outcomes=8,
            minimum_positive_outcomes_per_required_slice=20,
            positive_outcome_shortfall_before_holdouts=12,
            reason_codes=("insufficient_overall_positive_outcomes",),
        ),
        limitations=("Regulatory outcomes are not generic fraud labels.",),
    )
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def _write_solvency_evaluation(
    report_path: Path,
) -> SolvencyHoldoutEvaluationReport:
    capacity = SolvencyReviewCapacityMetrics(
        capacity_fraction=0.001,
        review_rows=1,
        precision=0.5,
        recall=0.25,
        lift=100.0,
    )
    split = SolvencySplitMetrics(
        dataset_split="temporal_test",
        rows=1_000,
        positive_rows=5,
        base_rate=0.005,
        diagnostic_accuracy_at_0_5=0.99,
        pr_auc=0.4,
        roc_auc=0.9,
        brier_score=0.004,
        score_minimum=0.0,
        score_maximum=0.8,
        score_mean=0.01,
        review_capacities=(capacity,),
    )
    report = SolvencyHoldoutEvaluationReport(
        label_boundary="cirp_public_announcement_outcome_not_fraud",
        model_sha256="f" * 64,
        model_data_path=Path("model-data.parquet"),
        holdouts_are_unsampled=True,
        evaluation_decision="EVALUATED_NOT_PRODUCTION_READY",
        deployment_blockers=("Payment telemetry is missing.",),
        split_metrics={"temporal_test": split},
    )
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report


def test_api_serves_one_immutable_evidence_report(tmp_path: Path) -> None:
    report_path = tmp_path / "evidence.json"
    expected_report = _write_evidence_report(report_path)

    with TestClient(create_app(report_path)) as client:
        health_response = client.get("/healthz")
        evidence_response = client.get("/v1/evidence/readiness")
        readiness_response = client.get("/v1/model/readiness")

    assert health_response.status_code == 200
    assert health_response.json()["report_id"] == expected_report.report_id
    assert evidence_response.status_code == 200
    assert evidence_response.headers["etag"] == f'"{expected_report.report_id}"'
    assert evidence_response.json()["model_readiness"]["decision"] == (
        "DO_NOT_TRAIN"
    )
    assert readiness_response.json()["positive_outcome_shortfall_before_holdouts"] == 12


def test_api_allows_configured_frontend_origin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = tmp_path / "evidence.json"
    _write_evidence_report(report_path)
    monkeypatch.setenv("ASSAY_ALLOWED_ORIGINS", "https://assay.example")

    with TestClient(create_app(report_path)) as client:
        response = client.options(
            "/v1/evidence/readiness",
            headers={
                "Origin": "https://assay.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://assay.example"
    )


def test_api_serves_optional_solvency_evaluation(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    evaluation_path = tmp_path / "solvency.json"
    _write_evidence_report(evidence_path)
    expected = _write_solvency_evaluation(evaluation_path)

    with TestClient(create_app(evidence_path, evaluation_path)) as client:
        health_response = client.get("/healthz")
        evaluation_response = client.get("/v1/models/solvency/evaluation")

    assert health_response.json()["solvency_evaluation_available"] is True
    assert evaluation_response.status_code == 200
    assert evaluation_response.json()["model_sha256"] == expected.model_sha256
    assert evaluation_response.headers["etag"].startswith('"')


def test_api_returns_503_when_solvency_evaluation_is_not_configured(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "evidence.json"
    _write_evidence_report(evidence_path)

    with TestClient(create_app(evidence_path)) as client:
        response = client.get("/v1/models/solvency/evaluation")

    assert response.status_code == 503
