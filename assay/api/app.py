"""Serve immutable merchant-risk evidence without recalculating decisions."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from assay.artifacts.parquet import sha256_file
from assay.reporting.readiness import (
    EvidenceReadinessReport,
    ModelReadinessSummary,
)
from assay.solvency.evaluation import SolvencyHoldoutEvaluationReport

EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE = "ASSAY_EVIDENCE_REPORT_PATH"
SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE = (
    "ASSAY_SOLVENCY_EVALUATION_PATH"
)
ALLOWED_ORIGINS_ENVIRONMENT_VARIABLE = "ASSAY_ALLOWED_ORIGINS"


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


def _configured_report_path(explicit_report_path: Path | None) -> Path:
    if explicit_report_path is not None:
        return explicit_report_path
    configured_value = os.getenv(EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE)
    if not configured_value:
        raise AssayApiConfigurationError(
            f"{EVIDENCE_REPORT_PATH_ENVIRONMENT_VARIABLE} is required."
        )
    return Path(configured_value)


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
    evaluation_path = explicit_evaluation_path
    if evaluation_path is None:
        configured_value = os.getenv(
            SOLVENCY_EVALUATION_PATH_ENVIRONMENT_VARIABLE
        )
        if configured_value:
            evaluation_path = Path(configured_value)
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
) -> FastAPI:
    """Create an API that loads one validated report once at process startup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.evidence_report = _load_evidence_report(
            _configured_report_path(report_path)
        )
        application.state.solvency_evaluation = (
            _load_optional_solvency_evaluation(solvency_evaluation_path)
        )
        yield

    application = FastAPI(
        title="ASSAY Evidence API",
        version="1.0.0",
        description=(
            "Read-only merchant-risk evidence. The API does not score merchants "
            "or alter deterministic training decisions."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
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

    @application.get("/healthz", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        evidence_report = current_report(request)
        return HealthResponse(
            status="ok",
            report_id=evidence_report.report_id,
            schema_version=evidence_report.schema_version,
            solvency_evaluation_available=(
                current_solvency_evaluation(request) is not None
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

    return application


app = create_app()
