"""Serve immutable merchant-risk evidence without recalculating decisions."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from assay.artifacts.parquet import ImmutableArtifactError, sha256_file
from assay.reporting.readiness import (
    EvidenceReadinessReport,
    ModelReadinessSummary,
)
from assay.serving.assessment import (
    MerchantAssessmentError,
    MerchantRiskAssessment,
    MerchantRiskAssessor,
)
from assay.serving.bands import RiskBandError, RiskBandScale
from assay.serving.lookup import (
    MerchantCandidate,
    MerchantLookupError,
    MerchantLookupResult,
    MerchantRiskIndexStore,
)
from assay.solvency.artifact import (
    SolvencyModelArtifactError,
    load_solvency_model_package,
)
from assay.solvency.evaluation import SolvencyHoldoutEvaluationReport

EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE = "ASSAY_EVIDENCE_REPORT_PATH"
SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE = (
    "ASSAY_SOLVENCY_EVALUATION_PATH"
)
MERCHANT_INDEX_REPORT_PATH_ENVIRONMENT_VARIABLE = (
    "ASSAY_MERCHANT_INDEX_REPORT_PATH"
)
SOLVENCY_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE = "ASSAY_SOLVENCY_MODEL_DIRECTORY"
ALLOWED_ORIGINS_ENVIRONMENT_VARIABLE = "ASSAY_ALLOWED_ORIGINS"

LOOKUP_OUTCOME_STATUS_CODES = {
    "invalid_identifier": 422,
    "no_match": 404,
    "ambiguous_name": 409,
}
LOOKUP_OUTCOME_GUIDANCE = {
    "invalid_identifier": (
        "Submit a 21-character MCA Corporate Identity Number, or submit the "
        "registered company name instead."
    ),
    "no_match": (
        "This identity is outside the frozen public population this service "
        "can assess. Route the merchant to manual KYB review."
    ),
    "ambiguous_name": (
        "Several public companies normalise to this name. Resubmit with the "
        "exact CIN from the merchant's incorporation documents."
    ),
}


class AssayApiConfigurationError(RuntimeError):
    """The read-only evidence API is missing a valid startup artifact."""


class HealthResponse(BaseModel):
    """Operational state and the immutable report currently being served."""

    model_config = ConfigDict(frozen=True)

    status: str
    service: str = "assay-evidence-api"
    report_id: str
    schema_version: str
    solvency_evaluation_available: bool
    merchant_assessment_available: bool
    merchant_index_run_id: str | None


class MerchantAssessmentRequest(BaseModel):
    """A KYB lookup by public identifier or registered company name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cin: str | None = Field(default=None, max_length=32)
    company_name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def require_one_identifier(self) -> MerchantAssessmentRequest:
        if not (self.cin or "").strip() and not (self.company_name or "").strip():
            raise ValueError("provide either cin or company_name")
        return self


class MerchantLookupProblem(BaseModel):
    """An identity that could not be resolved to exactly one public company."""

    model_config = ConfigDict(frozen=True)

    outcome: str
    detail: str
    guidance: str
    candidates: tuple[MerchantCandidate, ...] = ()


def _lookup_problem(lookup: MerchantLookupResult) -> MerchantLookupProblem:
    return MerchantLookupProblem(
        outcome=lookup.outcome,
        detail=lookup.detail,
        guidance=LOOKUP_OUTCOME_GUIDANCE.get(
            lookup.outcome, "Route this merchant to manual KYB review."
        ),
        candidates=lookup.candidates,
    )


def _configured_report_path(explicit_report_path: Path | None) -> Path:
    if explicit_report_path is not None:
        return explicit_report_path
    configured_value = os.getenv(EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE)
    if not configured_value:
        raise AssayApiConfigurationError(
            f"{EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE} is required."
        )
    return Path(configured_value)


def _optional_path(
    explicit_path: Path | None,
    environment_variable: str,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    configured_value = os.getenv(environment_variable)
    return Path(configured_value) if configured_value else None


def _load_evidence_report(report_path: Path) -> EvidenceReadinessReport:
    if not report_path.is_file():
        raise AssayApiConfigurationError(
            f"Evidence-readiness artifact is missing: {report_path}."
        )
    try:
        return EvidenceReadinessReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise AssayApiConfigurationError(
            f"Evidence-readiness artifact is invalid: {report_path}."
        ) from error


def _load_optional_solvency_evaluation(
    explicit_evaluation_path: Path | None,
) -> tuple[SolvencyHoldoutEvaluationReport, str] | None:
    evaluation_path = _optional_path(
        explicit_evaluation_path,
        SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE,
    )
    if evaluation_path is None:
        return None
    if not evaluation_path.is_file():
        raise AssayApiConfigurationError(
            f"Solvency-evaluation artifact is missing: {evaluation_path}."
        )
    try:
        report = SolvencyHoldoutEvaluationReport.model_validate_json(
            evaluation_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise AssayApiConfigurationError(
            f"Solvency-evaluation artifact is invalid: {evaluation_path}."
        ) from error
    return report, sha256_file(evaluation_path)


def _load_optional_assessor(
    explicit_index_report_path: Path | None,
    explicit_model_directory: Path | None,
    evaluation: tuple[SolvencyHoldoutEvaluationReport, str] | None,
) -> MerchantRiskAssessor | None:
    index_report_path = _optional_path(
        explicit_index_report_path,
        MERCHANT_INDEX_REPORT_PATH_ENVIRONMENT_VARIABLE,
    )
    model_directory = _optional_path(
        explicit_model_directory,
        SOLVENCY_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE,
    )
    if index_report_path is None and model_directory is None:
        return None
    if index_report_path is None or model_directory is None:
        raise AssayApiConfigurationError(
            "Merchant assessment requires both "
            f"{MERCHANT_INDEX_REPORT_PATH_ENVIRONMENT_VARIABLE} and "
            f"{SOLVENCY_MODEL_DIRECTORY_ENVIRONMENT_VARIABLE}."
        )
    if evaluation is None:
        raise AssayApiConfigurationError(
            "Merchant assessment requires "
            f"{SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE}; a score is not "
            "served without its frozen holdout evidence."
        )
    evaluation_report, _ = evaluation
    try:
        store = MerchantRiskIndexStore.from_report_path(index_report_path)
        package = load_solvency_model_package(model_directory, load_model=True)
        band_scale = RiskBandScale.from_evaluation(evaluation_report)
        return MerchantRiskAssessor(
            store=store,
            package=package,
            band_scale=band_scale,
            evaluation=evaluation_report,
        )
    except (
        ImmutableArtifactError,
        MerchantAssessmentError,
        MerchantLookupError,
        RiskBandError,
        SolvencyModelArtifactError,
    ) as error:
        raise AssayApiConfigurationError(
            f"Merchant assessment could not be configured: {error}"
        ) from error


def _allowed_origins() -> list[str]:
    configured_origins = os.getenv(
        ALLOWED_ORIGINS_ENVIRONMENT_VARIABLE,
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    ]


def create_app(
    report_path: Path | None = None,
    solvency_evaluation_path: Path | None = None,
    merchant_index_report_path: Path | None = None,
    solvency_model_directory: Path | None = None,
) -> FastAPI:
    """Create an API that loads one validated report once at process startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.evidence_report = _load_evidence_report(
            _configured_report_path(report_path)
        )
        evaluation = _load_optional_solvency_evaluation(solvency_evaluation_path)
        application.state.solvency_evaluation = evaluation
        application.state.merchant_assessor = _load_optional_assessor(
            merchant_index_report_path,
            solvency_model_directory,
            evaluation,
        )
        yield

    application = FastAPI(
        title="ASSAY Public Merchant Risk API",
        version="1.1.0",
        description=(
            "Read-only public-data merchant risk intelligence. The API returns "
            "a public insolvency-announcement score with its evidence for "
            "analyst triage. It never actions a merchant and uses no Razorpay "
            "payment data."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )

    def current_report(request: Request) -> EvidenceReadinessReport:
        return cast("EvidenceReadinessReport", request.app.state.evidence_report)

    def current_solvency_evaluation(
        request: Request,
    ) -> tuple[SolvencyHoldoutEvaluationReport, str] | None:
        return cast(
            "tuple[SolvencyHoldoutEvaluationReport, str] | None",
            request.app.state.solvency_evaluation,
        )

    def current_assessor(request: Request) -> MerchantRiskAssessor:
        assessor = cast(
            "MerchantRiskAssessor | None", request.app.state.merchant_assessor
        )
        if assessor is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Merchant assessment is not configured. The service needs "
                    "a merchant index, a verified model package, and frozen "
                    "holdout evidence."
                ),
            )
        return assessor

    @application.get("/healthz", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        evidence_report = current_report(request)
        assessor = cast(
            "MerchantRiskAssessor | None", request.app.state.merchant_assessor
        )
        return HealthResponse(
            status="ok",
            report_id=evidence_report.report_id,
            schema_version=evidence_report.schema_version,
            solvency_evaluation_available=(
                current_solvency_evaluation(request) is not None
            ),
            merchant_assessment_available=assessor is not None,
            merchant_index_run_id=(
                None if assessor is None else assessor.index_report.run_id
            ),
        )

    @application.get(
        "/v1/evidence/readiness",
        response_model=EvidenceReadinessReport,
    )
    def evidence_readiness(
        request: Request,
        response: Response,
    ) -> EvidenceReadinessReport:
        evidence_report = current_report(request)
        response.headers["ETag"] = f'"{evidence_report.report_id}"'
        response.headers["Cache-Control"] = "public, max-age=300"
        return evidence_report

    @application.get(
        "/v1/model/readiness",
        response_model=ModelReadinessSummary,
    )
    def model_readiness(request: Request) -> ModelReadinessSummary:
        return current_report(request).model_readiness

    @application.get(
        "/v1/models/solvency/evaluation",
        response_model=SolvencyHoldoutEvaluationReport,
    )
    def solvency_evaluation(
        request: Request,
        response: Response,
    ) -> SolvencyHoldoutEvaluationReport:
        evaluation = current_solvency_evaluation(request)
        if evaluation is None:
            raise HTTPException(
                status_code=503,
                detail="Solvency evaluation artifact is not configured.",
            )
        report, evaluation_sha256 = evaluation
        response.headers["ETag"] = f'"{evaluation_sha256}"'
        response.headers["Cache-Control"] = "public, max-age=300"
        return report

    @application.get("/v1/risk/bands", response_model=RiskBandScale)
    def risk_bands(request: Request) -> RiskBandScale:
        return current_assessor(request).band_scale

    @application.post(
        "/v1/risk/assess",
        response_model=MerchantRiskAssessment,
        responses={
            404: {"model": MerchantLookupProblem},
            409: {"model": MerchantLookupProblem},
            422: {"model": MerchantLookupProblem},
        },
    )
    def assess_merchant_risk(
        request: Request,
        assessment_request: MerchantAssessmentRequest,
    ) -> MerchantRiskAssessment | JSONResponse:
        assessor = current_assessor(request)
        try:
            outcome = assessor.assess(
                cin=assessment_request.cin,
                company_name=assessment_request.company_name,
            )
        except MerchantLookupError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (MerchantAssessmentError, SolvencyModelArtifactError) as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        if outcome.assessment is None:
            problem = _lookup_problem(outcome.lookup)
            return JSONResponse(
                status_code=LOOKUP_OUTCOME_STATUS_CODES.get(
                    outcome.lookup.outcome, 404
                ),
                content=problem.model_dump(mode="json"),
            )
        return outcome.assessment

    return application


app = create_app()
