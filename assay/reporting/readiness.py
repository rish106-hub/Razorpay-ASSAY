"""Build a frontend-safe evidence-readiness report from immutable run reports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.signals.runner import SignalObservationRunReport

EVIDENCE_READINESS_SCHEMA_VERSION = "1.0.0"
ModelTrainingDecision = Literal["TRAIN", "DO_NOT_TRAIN"]
ReportModel = TypeVar("ReportModel", bound=BaseModel)


class EvidenceReadinessError(RuntimeError):
    """The evidence chain is incomplete, inconsistent, or already published."""


class EvidenceReadinessConfig(BaseModel):
    """Explicit upstream evidence and frozen training sufficiency threshold."""

    model_config = ConfigDict(frozen=True)

    mca_report_path: Path
    nse_report_path: Path
    linkage_report_path: Path
    observation_report_path: Path
    project_root: Path = Path(__file__).resolve().parents[2]
    generated_root: Path = Path("data/generated/evidence_readiness")
    minimum_positive_outcomes_per_required_slice: int = Field(default=20, ge=1)


class SourceEvidenceSummary(BaseModel):
    """Source snapshot IDs and quality states exposed to downstream consumers."""

    model_config = ConfigDict(frozen=True)

    mca_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    nse_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_linkage_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    signal_observation_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mca_quality_status: str
    signal_observation_quality_status: str
    nse_provenance_complete_assets: int = Field(ge=0)


class LinkageCoverageSummary(BaseModel):
    """Measured label coverage without converting name candidates into labels."""

    model_config = ConfigDict(frozen=True)

    adverse_event_rows: int = Field(ge=0)
    accepted_exact_cin_rows: int = Field(ge=0)
    pending_exact_name_review_rows: int = Field(ge=0)
    ambiguous_rows: int = Field(ge=0)
    unmatched_rows: int = Field(ge=0)
    exact_cin_coverage_rate: float = Field(ge=0, le=1)
    human_review_required: bool


class SignalPopulationSummary(BaseModel):
    """Risk-set size and deterministic signal fire rates."""

    model_config = ConfigDict(frozen=True)

    company_rows: int = Field(ge=0)
    evaluation_eligible_rows: int = Field(ge=0)
    observed_adverse_outcome_rows: int = Field(ge=0)
    shared_address_signal_rows: int = Field(ge=0)
    shared_address_signal_rate: float = Field(ge=0, le=1)
    address_registration_month_cohort_signal_rows: int = Field(ge=0)
    address_registration_month_cohort_signal_rate: float = Field(ge=0, le=1)


class ModelReadinessSummary(BaseModel):
    """A deterministic gate that prevents training on inadequate entity labels."""

    model_config = ConfigDict(frozen=True)

    decision: ModelTrainingDecision
    status: str
    available_overall_positive_outcomes: int = Field(ge=0)
    minimum_positive_outcomes_per_required_slice: int = Field(ge=1)
    positive_outcome_shortfall_before_holdouts: int = Field(ge=0)
    controlled_benchmark_is_transferable: bool = False
    reason_codes: tuple[str, ...]


class EvidenceReadinessReport(BaseModel):
    """Versioned backend contract for the static report and frontend."""

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = EVIDENCE_READINESS_SCHEMA_VERSION
    source_evidence: SourceEvidenceSummary
    linkage_coverage: LinkageCoverageSummary
    signal_population: SignalPopulationSummary
    model_readiness: ModelReadinessSummary
    limitations: tuple[str, ...]


class EvidenceReadinessReporter:
    """Validate one evidence chain and publish a deterministic JSON artifact."""

    def __init__(self, config: EvidenceReadinessConfig) -> None:
        self._config = config

    def run(self) -> EvidenceReadinessReport:
        mca_report = self._load_report(
            self._config.mca_report_path,
            McaCanonicalisationReport,
        )
        nse_report = self._load_report(
            self._config.nse_report_path,
            NseCanonicalisationReport,
        )
        linkage_report = self._load_report(
            self._config.linkage_report_path,
            EntityLinkageReport,
        )
        observation_report = self._load_report(
            self._config.observation_report_path,
            SignalObservationRunReport,
        )
        self._validate_evidence_chain(
            mca_report,
            nse_report,
            linkage_report,
            observation_report,
        )
        report_id = self._report_id(
            mca_report,
            nse_report,
            linkage_report,
            observation_report,
        )
        report_path = self._resolve(self._config.generated_root) / (
            f"schema-{EVIDENCE_READINESS_SCHEMA_VERSION}/report-{report_id}.json"
        )
        if report_path.exists():
            raise EvidenceReadinessError(
                "Evidence-readiness report already exists for this evidence chain."
            )

        observed_positive_outcomes = observation_report.observed_adverse_outcome_rows
        minimum_positive_outcomes = (
            self._config.minimum_positive_outcomes_per_required_slice
        )
        training_is_ready = observed_positive_outcomes >= minimum_positive_outcomes
        reason_codes = (
            ("held_out_evaluation_required",)
            if training_is_ready
            else (
                "insufficient_overall_positive_outcomes",
                "human_linkage_review_incomplete",
                "held_out_slices_not_evaluable",
            )
        )
        report = EvidenceReadinessReport(
            report_id=report_id,
            source_evidence=SourceEvidenceSummary(
                mca_source_snapshot_id=mca_report.source_snapshot_id,
                nse_source_snapshot_id=nse_report.source_snapshot_id,
                entity_linkage_run_id=linkage_report.run_id,
                signal_observation_run_id=observation_report.run_id,
                mca_quality_status=mca_report.quality_status,
                signal_observation_quality_status=observation_report.quality_status,
                nse_provenance_complete_assets=(
                    nse_report.provenance_complete_assets
                ),
            ),
            linkage_coverage=LinkageCoverageSummary(
                adverse_event_rows=linkage_report.adverse_event_rows,
                accepted_exact_cin_rows=linkage_report.accepted_exact_cin_rows,
                pending_exact_name_review_rows=(
                    linkage_report.pending_exact_name_review_rows
                ),
                ambiguous_rows=linkage_report.ambiguous_rows,
                unmatched_rows=linkage_report.unmatched_rows,
                exact_cin_coverage_rate=self._safe_rate(
                    linkage_report.accepted_exact_cin_rows,
                    linkage_report.adverse_event_rows,
                ),
                human_review_required=(
                    linkage_report.pending_exact_name_review_rows > 0
                    or linkage_report.ambiguous_rows > 0
                ),
            ),
            signal_population=SignalPopulationSummary(
                company_rows=observation_report.company_rows,
                evaluation_eligible_rows=observation_report.evaluation_eligible_rows,
                observed_adverse_outcome_rows=observed_positive_outcomes,
                shared_address_signal_rows=(
                    observation_report.shared_address_signal_rows
                ),
                shared_address_signal_rate=self._safe_rate(
                    observation_report.shared_address_signal_rows,
                    observation_report.evaluation_eligible_rows,
                ),
                address_registration_month_cohort_signal_rows=(
                    observation_report.address_registration_month_cohort_signal_rows
                ),
                address_registration_month_cohort_signal_rate=self._safe_rate(
                    observation_report.address_registration_month_cohort_signal_rows,
                    observation_report.evaluation_eligible_rows,
                ),
            ),
            model_readiness=ModelReadinessSummary(
                decision="TRAIN" if training_is_ready else "DO_NOT_TRAIN",
                status=(
                    "held_out_evaluation_required"
                    if training_is_ready
                    else "blocked_insufficient_entity_outcomes"
                ),
                available_overall_positive_outcomes=observed_positive_outcomes,
                minimum_positive_outcomes_per_required_slice=(
                    minimum_positive_outcomes
                ),
                positive_outcome_shortfall_before_holdouts=max(
                    minimum_positive_outcomes - observed_positive_outcomes,
                    0,
                ),
                reason_codes=reason_codes,
            ),
            limitations=(
                "MCA company-master facts are frozen at 2023-11-03.",
                "NSE adverse regulatory outcomes are not generic fraud labels.",
                "Unique name matches require completed human review.",
                "The simulated transaction benchmark cannot validate entity risk.",
            ),
        )
        self._write_report(report_path, report)
        return report

    def _validate_evidence_chain(
        self,
        mca_report: McaCanonicalisationReport,
        nse_report: NseCanonicalisationReport,
        linkage_report: EntityLinkageReport,
        observation_report: SignalObservationRunReport,
    ) -> None:
        if linkage_report.mca_source_snapshot_id != mca_report.source_snapshot_id:
            raise EvidenceReadinessError("Linkage and MCA snapshots do not match.")
        if linkage_report.nse_source_snapshot_id != nse_report.source_snapshot_id:
            raise EvidenceReadinessError("Linkage and NSE snapshots do not match.")
        if observation_report.mca_source_snapshot_id != mca_report.source_snapshot_id:
            raise EvidenceReadinessError("Observations and MCA snapshots do not match.")
        if observation_report.nse_source_snapshot_id != nse_report.source_snapshot_id:
            raise EvidenceReadinessError("Observations and NSE snapshots do not match.")
        if observation_report.entity_linkage_run_id != linkage_report.run_id:
            raise EvidenceReadinessError("Observations and linkage runs do not match.")

    def _report_id(
        self,
        mca_report: McaCanonicalisationReport,
        nse_report: NseCanonicalisationReport,
        linkage_report: EntityLinkageReport,
        observation_report: SignalObservationRunReport,
    ) -> str:
        payload = json.dumps(
            {
                "mca_source_snapshot_id": mca_report.source_snapshot_id,
                "nse_source_snapshot_id": nse_report.source_snapshot_id,
                "entity_linkage_run_id": linkage_report.run_id,
                "signal_observation_run_id": observation_report.run_id,
                "minimum_positive_outcomes_per_required_slice": (
                    self._config.minimum_positive_outcomes_per_required_slice
                ),
                "schema_version": EVIDENCE_READINESS_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_report(
        self,
        configured_path: Path,
        model: type[ReportModel],
    ) -> ReportModel:
        report_path = self._resolve(configured_path)
        if not report_path.is_file():
            raise EvidenceReadinessError(f"Evidence report is missing: {report_path}.")
        return model.model_validate_json(report_path.read_text(encoding="utf-8"))

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _safe_rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    @staticmethod
    def _write_report(report_path: Path, report: EvidenceReadinessReport) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    report.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
            )
            report_file.flush()
            os.fsync(report_file.fileno())
