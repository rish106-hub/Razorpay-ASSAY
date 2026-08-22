"""Publish checksum-addressed Parquet and CSV linkage review workbooks."""

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
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.linkage.review import (
    LINKAGE_REVIEW_SCHEMA_VERSION,
    LinkageReviewSamplingConfig,
    build_linkage_review_sample,
)


class LinkageReviewRunError(RuntimeError):
    """A linkage review workbook could not be published safely."""


class ReviewFileArtifact(BaseModel):
    """Checksum metadata for a non-Parquet reviewer-facing artifact."""

    model_config = ConfigDict(frozen=True)

    path: str
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)


class LinkageReviewRunConfig(BaseModel):
    """Explicit linkage inputs and bounded reviewer workload."""

    model_config = ConfigDict(frozen=True)

    linkage_report_path: Path
    nse_report_path: Path
    project_root: Path = Path(__file__).resolve().parents[2]
    generated_root: Path = Path("data/generated/linkage_review")
    sampling: LinkageReviewSamplingConfig = LinkageReviewSamplingConfig()
    parquet_compression: str = "zstd"


class LinkageReviewRunReport(BaseModel):
    """Review sample coverage and immutable workbook artifacts."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = LINKAGE_REVIEW_SCHEMA_VERSION
    entity_linkage_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    nse_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_rows: int = Field(ge=0)
    review_event_rows: int = Field(ge=0)
    stratum_event_counts: dict[str, int]
    parquet_artifact: ParquetArtifact
    csv_artifact: ReviewFileArtifact
    instructions_artifact: ReviewFileArtifact


REVIEW_INSTRUCTIONS = """# ASSAY entity-linkage review

Review one candidate row at a time. Use only these values in `review_decision`:

- `MATCH`: the NSE subject and MCA company are the same legal entity.
- `NO_MATCH`: they are different legal entities.
- `UNSURE`: the evidence is not enough for a defensible decision.

Fill `reviewer_id`, `reviewed_at` in ISO 8601 format, and `reviewer_notes`.
For ambiguous events, review every candidate shown for that event. Do not edit
pipeline columns, IDs, match scores, source fields, or sampling fields. An
exact name is evidence, not proof. Regulatory action is not a fraud label.
"""


def _escape_spreadsheet_formula(raw_value: str | None) -> str | None:
    if raw_value and raw_value.startswith(("=", "+", "-", "@")):
        return f"'{raw_value}"
    return raw_value


def _spreadsheet_safe_frame(review_frame: pl.DataFrame) -> pl.DataFrame:
    string_columns = [
        column_name
        for column_name, column_type in review_frame.schema.items()
        if column_type == pl.String
    ]
    return review_frame.with_columns(
        [
            pl.col(column_name)
            .map_elements(_escape_spreadsheet_formula, return_dtype=pl.String)
            .alias(column_name)
            for column_name in string_columns
        ]
    )


class LinkageReviewRunner:
    """Build a deterministic review sample without altering linkage decisions."""

    def __init__(self, config: LinkageReviewRunConfig) -> None:
        self._config = config

    def run(self) -> LinkageReviewRunReport:
        linkage_report = self._load_report(
            self._config.linkage_report_path,
            EntityLinkageReport,
        )
        nse_report = self._load_report(
            self._config.nse_report_path,
            NseCanonicalisationReport,
        )
        if linkage_report.nse_source_snapshot_id != nse_report.source_snapshot_id:
            raise LinkageReviewRunError(
                "Linkage and NSE reports reference different source snapshots."
            )
        run_payload = json.dumps(
            {
                "entity_linkage_run_id": linkage_report.run_id,
                "nse_source_snapshot_id": nse_report.source_snapshot_id,
                "sampling": self._config.sampling.model_dump(mode="json"),
                "schema_version": LINKAGE_REVIEW_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        run_id = hashlib.sha256(run_payload).hexdigest()
        run_directory = self._resolve(self._config.generated_root) / (
            f"schema-{LINKAGE_REVIEW_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = run_directory.parent / f"run-{run_id}.report.json"
        if run_directory.exists() or report_path.exists():
            raise LinkageReviewRunError("Linkage review run output already exists.")

        candidate_frame = pl.read_parquet(
            verify_parquet_artifact(
                linkage_report.candidate_artifact,
                self._config.project_root,
            )
        )
        decision_frame = pl.read_parquet(
            verify_parquet_artifact(
                linkage_report.decision_artifact,
                self._config.project_root,
            )
        )
        adverse_frame = pl.concat(
            [
                pl.scan_parquet(
                    verify_parquet_artifact(part, self._config.project_root)
                ).select(
                    "adverse_event_id",
                    "event_source_authority",
                    "event_date",
                    "order_particulars",
                    "entity_name_raw",
                    "entity_name_normalized_strict",
                    "entity_name_normalized_legal",
                    "pan_raw",
                    "din_cin_raw",
                    "cin_candidate",
                )
                for part in nse_report.parts
            ],
            how="vertical",
        ).collect(engine="streaming")
        review_frame = build_linkage_review_sample(
            candidate_frame,
            decision_frame,
            adverse_frame,
            self._config.sampling,
        )
        artifacts = self._publish_review_artifacts(review_frame, run_directory)
        stratum_event_counts = {
            row["review_stratum"]: row["len"]
            for row in review_frame.group_by("review_stratum")
            .agg(pl.col("adverse_event_id").n_unique().alias("len"))
            .to_dicts()
        }
        report = LinkageReviewRunReport(
            run_id=run_id,
            entity_linkage_run_id=linkage_report.run_id,
            nse_source_snapshot_id=nse_report.source_snapshot_id,
            review_rows=review_frame.height,
            review_event_rows=review_frame["adverse_event_id"].n_unique(),
            stratum_event_counts=stratum_event_counts,
            parquet_artifact=artifacts[0],
            csv_artifact=artifacts[1],
            instructions_artifact=artifacts[2],
        )
        self._write_report(report_path, report)
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    def _load_report(self, configured_path: Path, report_model: type[BaseModel]):
        report_path = self._resolve(configured_path)
        if not report_path.is_file():
            raise LinkageReviewRunError(f"Run report is missing: {report_path}.")
        return report_model.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )

    def _publish_review_artifacts(
        self,
        review_frame: pl.DataFrame,
        run_directory: Path,
    ) -> tuple[ParquetArtifact, ReviewFileArtifact, ReviewFileArtifact]:
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=run_directory.parent,
            prefix=".assay-linkage-review-",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            temporary_parquet_artifact = write_immutable_parquet(
                review_frame,
                temporary_directory / "linkage_review.parquet",
                self._config.parquet_compression,
            )
            csv_path = temporary_directory / "linkage_review.csv"
            _spreadsheet_safe_frame(review_frame).write_csv(csv_path)
            with csv_path.open("rb") as csv_file:
                os.fsync(csv_file.fileno())
            instructions_path = temporary_directory / "REVIEW_INSTRUCTIONS.md"
            with instructions_path.open("xb") as instructions_file:
                instructions_file.write(REVIEW_INSTRUCTIONS.encode("utf-8"))
                instructions_file.flush()
                os.fsync(instructions_file.fileno())
            os.rename(temporary_directory, run_directory)

        published_parquet_path = run_directory / "linkage_review.parquet"
        published_csv_path = run_directory / "linkage_review.csv"
        published_instructions_path = run_directory / "REVIEW_INSTRUCTIONS.md"
        parquet_artifact = ParquetArtifact(
            path=str(published_parquet_path),
            rows=temporary_parquet_artifact.rows,
            bytes=published_parquet_path.stat().st_size,
            sha256=sha256_file(published_parquet_path),
        )
        csv_artifact = ReviewFileArtifact(
            path=str(published_csv_path),
            bytes=published_csv_path.stat().st_size,
            sha256=sha256_file(published_csv_path),
            media_type="text/csv",
        )
        instructions_artifact = ReviewFileArtifact(
            path=str(published_instructions_path),
            bytes=published_instructions_path.stat().st_size,
            sha256=sha256_file(published_instructions_path),
            media_type="text/markdown",
        )
        return parquet_artifact, csv_artifact, instructions_artifact

    @staticmethod
    def _write_report(report_path: Path, report: LinkageReviewRunReport) -> None:
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

