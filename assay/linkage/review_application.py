"""Publish reviewed entity-match decisions as a new immutable linkage run."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import (
    ParquetArtifact,
    sha256_file,
    verify_parquet_artifact,
    write_immutable_parquet,
)
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.linkage.review_ingestion import (
    LinkageReviewApplicationSummary,
    validate_and_apply_linkage_reviews,
)
from assay.linkage.review_runner import LinkageReviewRunReport

REVIEWED_LINKAGE_SCHEMA_VERSION = "1.1.0"
REVIEWED_LINKAGE_POLICY_VERSION = "cin_then_exact_name_plus_human_review_v1"


class ReviewedLinkageRunError(RuntimeError):
    """A reviewed linkage decision artifact could not be published safely."""


class ReviewedLinkageApplicationConfig(BaseModel):
    """Explicit original artifacts and completed human-review workbook."""

    model_config = ConfigDict(frozen=True)

    linkage_report_path: Path
    review_report_path: Path
    completed_review_csv_path: Path
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/entity_match_reviewed")
    generated_report_root: Path = Path("data/generated/entity_linkage_reviewed")
    parquet_compression: str = "zstd"


class ReviewedEntityLinkageReport(EntityLinkageReport):
    """Entity-linkage report extended with validated reviewer provenance."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = REVIEWED_LINKAGE_SCHEMA_VERSION
    policy_version: str = REVIEWED_LINKAGE_POLICY_VERSION
    original_entity_linkage_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    linkage_review_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_review_csv_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_ids: tuple[str, ...]
    review_application: LinkageReviewApplicationSummary
    review_quality_status: str


class ReviewedLinkageRunner:
    """Validate a completed CSV and issue a new linkage decision run."""

    def __init__(self, config: ReviewedLinkageApplicationConfig) -> None:
        self._config = config

    def run(self) -> ReviewedEntityLinkageReport:
        linkage_report = EntityLinkageReport.model_validate_json(
            self._read_report(self._config.linkage_report_path)
        )
        review_report = LinkageReviewRunReport.model_validate_json(
            self._read_report(self._config.review_report_path)
        )
        if review_report.entity_linkage_run_id != linkage_report.run_id:
            raise ReviewedLinkageRunError(
                "Review workbook and linkage report reference different runs."
            )
        completed_review_csv_path = self._resolve(
            self._config.completed_review_csv_path
        )
        if not completed_review_csv_path.is_file():
            raise ReviewedLinkageRunError(
                f"Completed review CSV is missing: {completed_review_csv_path}."
            )
        completed_review_csv_sha256 = sha256_file(completed_review_csv_path)
        original_review_frame = pl.read_parquet(
            verify_parquet_artifact(
                review_report.parquet_artifact,
                self._config.project_root,
            )
        )
        completed_review_frame = pl.read_csv(
            completed_review_csv_path,
            infer_schema=False,
            null_values=[],
        )
        original_decision_frame = pl.read_parquet(
            verify_parquet_artifact(
                linkage_report.decision_artifact,
                self._config.project_root,
            )
        )
        revised_decision_frame, application_summary = (
            validate_and_apply_linkage_reviews(
                original_review_frame,
                completed_review_frame,
                original_decision_frame,
            )
        )
        run_payload = json.dumps(
            {
                "original_entity_linkage_run_id": linkage_report.run_id,
                "linkage_review_run_id": review_report.run_id,
                "completed_review_csv_sha256": completed_review_csv_sha256,
                "schema_version": REVIEWED_LINKAGE_SCHEMA_VERSION,
                "policy_version": REVIEWED_LINKAGE_POLICY_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        run_id = hashlib.sha256(run_payload).hexdigest()
        run_directory = self._resolve(self._config.curated_root) / (
            f"schema-{REVIEWED_LINKAGE_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{REVIEWED_LINKAGE_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise ReviewedLinkageRunError("Reviewed linkage run already exists.")
        decision_artifact = self._publish_decisions(
            revised_decision_frame,
            run_directory,
        )
        reviewer_ids = tuple(
            sorted(
                reviewer_id
                for reviewer_id in completed_review_frame["reviewer_id"]
                .str.strip_chars()
                .unique()
                .to_list()
                if reviewer_id
            )
        )
        accepted_exact_cin_rows = revised_decision_frame.filter(
            (pl.col("review_status") == "accepted")
            & (pl.col("match_method") == "exact_cin")
        ).height
        report = ReviewedEntityLinkageReport(
            run_id=run_id,
            mca_source_snapshot_id=linkage_report.mca_source_snapshot_id,
            nse_source_snapshot_id=linkage_report.nse_source_snapshot_id,
            nse_provenance_complete_assets=(
                linkage_report.nse_provenance_complete_assets
            ),
            adverse_event_rows=revised_decision_frame.height,
            accepted_exact_cin_rows=accepted_exact_cin_rows,
            pending_exact_name_review_rows=revised_decision_frame.filter(
                pl.col("review_status") == "pending_review"
            ).height,
            ambiguous_rows=revised_decision_frame.filter(
                pl.col("review_status") == "ambiguous"
            ).height,
            unmatched_rows=revised_decision_frame.filter(
                pl.col("review_status") == "unmatched"
            ).height,
            identifier_collision_rows=revised_decision_frame.filter(
                (pl.col("match_method") == "exact_cin")
                & (pl.col("candidate_count") > 1)
            ).height,
            persisted_candidate_rows=linkage_report.persisted_candidate_rows,
            truncated_candidate_event_rows=revised_decision_frame.filter(
                pl.col("candidate_set_truncated")
            ).height,
            outcome_eligible_rows=revised_decision_frame.filter(
                pl.col("outcome_eligible")
            ).height,
            candidate_artifact=linkage_report.candidate_artifact,
            decision_artifact=decision_artifact,
            original_entity_linkage_run_id=linkage_report.run_id,
            linkage_review_run_id=review_report.run_id,
            completed_review_csv_sha256=completed_review_csv_sha256,
            reviewer_ids=reviewer_ids,
            review_application=application_summary,
            review_quality_status=(
                "identifier_discrepancy_detected"
                if application_summary.identifier_audit_discrepancy_rows > 0
                else "passed"
            ),
        )
        self._write_report(report_path, report)
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    def _read_report(self, configured_path: Path) -> str:
        report_path = self._resolve(configured_path)
        if not report_path.is_file():
            raise ReviewedLinkageRunError(f"Run report is missing: {report_path}.")
        return report_path.read_text(encoding="utf-8")

    def _publish_decisions(
        self,
        decision_frame: pl.DataFrame,
        run_directory: Path,
    ) -> ParquetArtifact:
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=run_directory.parent,
            prefix=".assay-reviewed-linkage-",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            temporary_artifact = write_immutable_parquet(
                decision_frame,
                temporary_directory / "entity_match_decision.parquet",
                self._config.parquet_compression,
            )
            os.rename(temporary_directory, run_directory)
        published_path = run_directory / "entity_match_decision.parquet"
        return ParquetArtifact(
            path=str(published_path),
            rows=temporary_artifact.rows,
            bytes=published_path.stat().st_size,
            sha256=sha256_file(published_path),
        )

    @staticmethod
    def _write_report(
        report_path: Path,
        report: ReviewedEntityLinkageReport,
    ) -> None:
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
