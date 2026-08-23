"""Publish deterministic signal verdicts from immutable evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from assay.evaluation.runner import EvaluationRunReport
from assay.evaluation.verdict import (
    ASSAY_VERDICT_SCHEMA_VERSION,
    VerdictReport,
    VerdictThresholds,
    decide_verdict,
)


class VerdictPublicationError(RuntimeError):
    """A deterministic verdict could not be validated or published."""


class VerdictPublicationConfig(BaseModel):
    """Explicit evidence and risk-operations thresholds for one verdict."""

    model_config = ConfigDict(frozen=True)

    evaluation_report_path: Path
    false_positive_review_cost_budget_inr: float = Field(gt=0.0)
    minimum_conservative_lift: float = Field(default=1.25, gt=1.0)
    maximum_group_concentration_p_value: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
    )
    maximum_review_capacity_expansion: float = Field(default=2.0, ge=1.0)
    project_root: Path = Path(__file__).resolve().parents[2]
    generated_root: Path = Path("data/generated/verdict")


class VerdictPublicationReport(BaseModel):
    """Checksum-addressable wrapper around the deterministic verdict contract."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = ASSAY_VERDICT_SCHEMA_VERSION
    verdict: VerdictReport
    report_path: str


class VerdictPublisher:
    """Apply frozen thresholds and publish one immutable verdict artifact."""

    def __init__(self, config: VerdictPublicationConfig) -> None:
        self._config = config

    def run(self) -> VerdictPublicationReport:
        evaluation_report_path = self._resolve(
            self._config.evaluation_report_path
        )
        if not evaluation_report_path.is_file():
            raise VerdictPublicationError(
                f"Evaluation report is missing: {evaluation_report_path}."
            )
        try:
            evaluation_report = EvaluationRunReport.model_validate_json(
                evaluation_report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise VerdictPublicationError("Evaluation report is invalid.") from error
        thresholds = VerdictThresholds(
            minimum_conservative_lift=self._config.minimum_conservative_lift,
            maximum_group_concentration_p_value=(
                self._config.maximum_group_concentration_p_value
            ),
            maximum_review_capacity_expansion=(
                self._config.maximum_review_capacity_expansion
            ),
            false_positive_review_cost_budget_inr=(
                self._config.false_positive_review_cost_budget_inr
            ),
        )
        verdict = decide_verdict(evaluation_report, thresholds)
        run_id = self._run_id(evaluation_report, thresholds)
        report_path = self._resolve(self._config.generated_root) / (
            f"schema-{ASSAY_VERDICT_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if report_path.exists():
            raise VerdictPublicationError("Verdict report already exists.")
        publication = VerdictPublicationReport(
            run_id=run_id,
            verdict=verdict,
            report_path=str(report_path),
        )
        self._write_report(report_path, publication)
        return publication

    @staticmethod
    def _run_id(
        evaluation_report: EvaluationRunReport,
        thresholds: VerdictThresholds,
    ) -> str:
        payload = json.dumps(
            {
                "evaluation_run_id": evaluation_report.run_id,
                "thresholds": thresholds.model_dump(mode="json"),
                "schema_version": ASSAY_VERDICT_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _write_report(
        report_path: Path,
        publication: VerdictPublicationReport,
    ) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    publication.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
            )
            report_file.flush()
            os.fsync(report_file.fileno())
