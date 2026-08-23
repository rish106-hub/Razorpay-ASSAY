"""End-to-end contract tests for the public merchant-risk assessment API.

Every artifact under test is synthetic and built in a temporary directory: the
production model package and merchant index are gitignored, so nothing here may
reach for them. No assertion names a real company, a real CIN, or a real score.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assay.api.app import (
    EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE,
    MERCHANT_INDEX_REPORT_PATH_ENVIRONMENT_VARIABLE,
    SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE,
    SOLVENCY_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE,
    AssayApiConfigurationError,
    create_app,
)
from assay.reporting.readiness import (
    EvidenceReadinessReport,
    LinkageCoverageSummary,
    ModelReadinessSummary,
    SignalPopulationSummary,
    SourceEvidenceSummary,
)
from assay.serving.assessment import MERCHANT_RISK_SCORE_NAME
from assay.serving.bands import RISK_BAND_ORDER
from tests.serving.model_package_fixture import (
    AMBIGUOUS_NAME,
    HIGH_RISK_CIN,
    LEGAL_SUFFIX_QUERY,
    LOW_RISK_CIN,
    LOW_RISK_NAME,
    UNINDEXED_CIN,
    ServingFixture,
    write_serving_fixture,
)

pytest.importorskip("xgboost")

# Pinned so a future rename of a served field fails loudly instead of silently
# dropping evidence a reviewer depends on.
ASSESSMENT_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "generated_at",
        "score_name",
        "label_boundary",
        "model_status",
        "identity",
        "identity_match_method",
        "cirp_public_announcement_score",
        "risk_band",
        "risk_band_evidence",
        "explanation_factors",
        "evidence_sources",
        "confidence",
        "permitted_use",
        "limitations",
    }
)
SERVING_ENVIRONMENT_VARIABLES = (
    EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE,
    SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE,
    MERCHANT_INDEX_REPORT_PATH_ENVIRONMENT_VARIABLE,
    SOLVENCY_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE,
)


def _write_evidence_report(report_path: Path) -> Path:
    """Write the minimum evidence artifact the API requires to start."""

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
            adverse_event_rows=100,
            accepted_exact_cin_rows=10,
            pending_exact_name_review_rows=20,
            ambiguous_rows=5,
            unmatched_rows=65,
            exact_cin_coverage_rate=0.1,
            human_review_required=True,
        ),
        signal_population=SignalPopulationSummary(
            company_rows=1_000,
            evaluation_eligible_rows=900,
            observed_adverse_outcome_rows=8,
            shared_address_signal_rows=100,
            shared_address_signal_rate=100 / 900,
            address_registration_month_cohort_signal_rows=40,
            address_registration_month_cohort_signal_rate=40 / 900,
        ),
        model_readiness=ModelReadinessSummary(
            decision="DO_NOT_TRAIN",
            status="blocked_insufficient_entity_outcomes",
            available_overall_positive_outcomes=8,
            minimum_positive_outcomes_per_required_slice=20,
            positive_outcome_shortfall_before_holdouts=12,
            reason_codes=("insufficient_overall_positive_outcomes",),
        ),
        limitations=("Public insolvency risk is not merchant misconduct.",),
    )
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    return report_path


@pytest.fixture(autouse=True)
def _isolate_serving_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's exported artifact paths reach the test app."""

    for environment_variable in SERVING_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)


@pytest.fixture(name="artifacts", scope="session")
def artifacts_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> ServingFixture:
    return write_serving_fixture(tmp_path_factory.mktemp("serving"))


@pytest.fixture(name="evidence_path")
def evidence_path_fixture(tmp_path: Path) -> Path:
    return _write_evidence_report(tmp_path / "evidence.json")


@pytest.fixture(name="client")
def client_fixture(
    evidence_path: Path,
    artifacts: ServingFixture,
) -> Iterator[TestClient]:
    application = create_app(
        report_path=evidence_path,
        solvency_evaluation_path=artifacts.evaluation_path,
        merchant_index_report_path=artifacts.index_report_path,
        solvency_model_directory=artifacts.model_directory,
    )
    with TestClient(application) as client:
        yield client


def test_assessment_by_cin_returns_an_evidence_backed_score(
    client: TestClient,
) -> None:
    response = client.post("/v1/risk/assess", json={"cin": LOW_RISK_CIN})

    assert response.status_code == 200
    body = response.json()
    assert body["score_name"] == MERCHANT_RISK_SCORE_NAME
    assert 0.0 <= body["cirp_public_announcement_score"] <= 1.0
    assert body["risk_band"] in RISK_BAND_ORDER
    assert body["model_status"] == "EVALUATED_NOT_PRODUCTION_READY"
    assert body["identity"]["cin"] == LOW_RISK_CIN
    assert body["identity_match_method"] == "exact_cin"
    assert body["explanation_factors"]
    for factor in body["explanation_factors"]:
        assert factor["feature"]
        assert factor["direction"] in {"increases_risk", "decreases_risk"}
        assert isinstance(factor["contribution"], float)
        assert factor["evidence"]
    assert {source["source_id"] for source in body["evidence_sources"]} == {
        "mca_company_master",
        "ibbi_cirp_public_announcements",
        "merchant_risk_index",
        "solvency_model_package",
    }
    assert body["confidence"]["identity_match_method"] == "exact_cin"
    assert body["confidence"]["reasons"]
    assert body["permitted_use"]
    assert body["limitations"]


def test_assessment_response_keeps_its_published_top_level_contract(
    client: TestClient,
) -> None:
    response = client.post("/v1/risk/assess", json={"cin": LOW_RISK_CIN})

    assert response.status_code == 200
    assert set(response.json()) == ASSESSMENT_RESPONSE_KEYS


def test_unique_company_name_resolves_through_strict_normalisation(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/risk/assess", json={"company_name": LOW_RISK_NAME.lower()}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["identity_match_method"] == "normalized_strict_name"
    assert body["identity"]["cin"] == LOW_RISK_CIN


def test_name_without_a_legal_suffix_resolves_through_the_legal_form(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/risk/assess", json={"company_name": LEGAL_SUFFIX_QUERY}
    )

    assert response.status_code == 200
    assert response.json()["identity_match_method"] == "normalized_legal_name"


def test_malformed_cin_is_rejected_with_actionable_guidance(
    client: TestClient,
) -> None:
    response = client.post("/v1/risk/assess", json={"cin": "NOT-A-REAL-CIN"})

    assert response.status_code == 422
    body = response.json()
    assert {"outcome", "detail", "guidance"} <= set(body)
    assert body["outcome"] == "invalid_identifier"
    assert body["guidance"]


def test_well_formed_cin_outside_the_frozen_population_is_not_scored(
    client: TestClient,
) -> None:
    response = client.post("/v1/risk/assess", json={"cin": UNINDEXED_CIN})

    assert response.status_code == 404
    body = response.json()
    assert body["outcome"] == "no_match"
    assert body["candidates"] == []
    assert MERCHANT_RISK_SCORE_NAME not in json.dumps(body)


def test_ambiguous_company_name_returns_candidates_and_never_a_score(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/risk/assess", json={"company_name": AMBIGUOUS_NAME}
    )

    assert response.status_code == 409
    body = response.json()
    assert body["outcome"] == "ambiguous_name"
    assert len(body["candidates"]) > 1
    assert all(candidate["cin"] for candidate in body["candidates"])
    assert set(body) == {"outcome", "detail", "guidance", "candidates"}
    assert MERCHANT_RISK_SCORE_NAME not in json.dumps(body)


def test_request_without_an_identifier_or_a_name_is_rejected(
    client: TestClient,
) -> None:
    response = client.post("/v1/risk/assess", json={})

    assert response.status_code == 422


def test_request_with_an_unknown_field_is_rejected(
    client: TestClient,
) -> None:
    response = client.post(
        "/v1/risk/assess",
        json={"cin": LOW_RISK_CIN, "merchant_id": "placeholder"},
    )

    assert response.status_code == 422


def test_higher_scoring_company_lands_in_a_stricter_review_band(
    client: TestClient,
) -> None:
    elevated = client.post(
        "/v1/risk/assess", json={"cin": HIGH_RISK_CIN}
    ).json()
    baseline = client.post("/v1/risk/assess", json={"cin": LOW_RISK_CIN}).json()

    assert (
        elevated["cirp_public_announcement_score"]
        > baseline["cirp_public_announcement_score"]
    )
    assert RISK_BAND_ORDER.index(elevated["risk_band"]) < RISK_BAND_ORDER.index(
        baseline["risk_band"]
    )
    assert (
        elevated["risk_band_evidence"]["minimum_score"]
        > baseline["risk_band_evidence"]["minimum_score"]
    )


def test_risk_bands_expose_one_strictly_descending_measured_scale(
    client: TestClient,
) -> None:
    response = client.get("/v1/risk/bands")

    assert response.status_code == 200
    thresholds = response.json()["thresholds"]
    assert [threshold["band"] for threshold in thresholds] == list(
        RISK_BAND_ORDER
    )
    minimum_scores = [threshold["minimum_score"] for threshold in thresholds]
    assert all(higher > lower for higher, lower in pairwise(minimum_scores))


def test_assessment_endpoints_report_503_when_not_configured(
    evidence_path: Path,
) -> None:
    with TestClient(create_app(report_path=evidence_path)) as client:
        health_response = client.get("/healthz")
        assess_response = client.post(
            "/v1/risk/assess", json={"cin": LOW_RISK_CIN}
        )
        bands_response = client.get("/v1/risk/bands")

    assert health_response.json()["merchant_assessment_available"] is False
    assert health_response.json()["merchant_index_run_id"] is None
    assert assess_response.status_code == 503
    assert bands_response.status_code == 503


def test_index_without_a_model_directory_fails_startup(
    evidence_path: Path,
    artifacts: ServingFixture,
) -> None:
    application = create_app(
        report_path=evidence_path,
        solvency_evaluation_path=artifacts.evaluation_path,
        merchant_index_report_path=artifacts.index_report_path,
    )

    with (
        pytest.raises(AssayApiConfigurationError, match="requires both"),
        TestClient(application),
    ):
        pass


def test_index_and_model_without_holdout_evidence_fail_startup(
    evidence_path: Path,
    artifacts: ServingFixture,
) -> None:
    application = create_app(
        report_path=evidence_path,
        merchant_index_report_path=artifacts.index_report_path,
        solvency_model_directory=artifacts.model_directory,
    )

    with (
        pytest.raises(AssayApiConfigurationError, match="frozen holdout"),
        TestClient(application),
    ):
        pass
