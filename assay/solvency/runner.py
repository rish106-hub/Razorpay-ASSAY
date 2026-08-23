"""Publish reproducible MCA-to-IBBI merchant-solvency observations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from assay.artifacts.parquet import (
    ParquetArtifact,
    sha256_file,
    verify_parquet_artifact,
    write_immutable_parquet,
)
from assay.canonicalisation.ibbi import IbbiCanonicalisationReport
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.solvency.observations import (
    SOLVENCY_OBSERVATION_SCHEMA_VERSION,
    SolvencyObservationConfig,
    build_solvency_observations,
)

ReportModel = TypeVar("ReportModel", bound=BaseModel)


class SolvencyObservationRunError(RuntimeError):
    """A merchant-solvency observation run failed reproducibility checks."""


class SolvencyObservationRunConfig(BaseModel):
    """Explicit snapshots and frozen model-readiness slices."""

    model_config = ConfigDict(frozen=True)

    mca_report_path: Path
    ibbi_report_path: Path
    outcome_window_end: date
    temporal_holdout_start: date
    unseen_geography_states: tuple[str, ...] = Field(
        default=("karnataka", "kerala", "tamil nadu", "telangana"),
        min_length=1,
    )
    minimum_positive_outcomes_per_slice: int = Field(default=20, ge=20)
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/solvency_observation")
    generated_report_root: Path = Path("data/generated/solvency_observation")
    parquet_compression: str = "zstd"

    @model_validator(mode="after")
    def require_holdout_inside_outcome_window(self) -> SolvencyObservationRunConfig:
        if self.temporal_holdout_start > self.outcome_window_end:
            raise ValueError("the temporal holdout must start inside the outcome window")
        return self


class SolvencyObservationRunReport(BaseModel):
    """Population, slice coverage, and training gate for one solvency target."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = SOLVENCY_OBSERVATION_SCHEMA_VERSION
    mca_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ibbi_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    outcome_window_start: date
    outcome_window_end: date
    temporal_holdout_start: date
    unseen_geography_states: tuple[str, ...]
    company_rows: int = Field(ge=0)
    evaluation_eligible_rows: int = Field(ge=0)
    excluded_prior_outcome_rows: int = Field(ge=0)
    observed_outcome_rows: int = Field(ge=0)
    unseen_geography_positive_rows: int = Field(ge=0)
    temporal_holdout_positive_rows: int = Field(ge=0)
    positive_base_rate: float = Field(ge=0, le=1)
    minimum_positive_outcomes_per_slice: int = Field(ge=20)
    training_decision: str
    quality_status: str
    observation_artifact: ParquetArtifact


class SolvencyObservationRunner:
    """Build and publish one exact-CIN solvency observation population."""

    def __init__(self, config: SolvencyObservationRunConfig) -> None:
        self._config = config

    def run(self) -> SolvencyObservationRunReport:
        mca_report = self._load_report(
            self._config.mca_report_path,
            McaCanonicalisationReport,
        )
        ibbi_report = self._load_report(
            self._config.ibbi_report_path,
            IbbiCanonicalisationReport,
        )
        if not mca_report.acquisition_complete or mca_report.quality_status not in {
            "passed",
            "passed_with_quarantine",
        }:
            raise SolvencyObservationRunError(
                "Solvency observations require a complete quality-passed MCA snapshot."
            )
        if self._config.outcome_window_end <= mca_report.snapshot_as_of:
            raise SolvencyObservationRunError(
                "The solvency outcome window must follow the MCA feature cutoff."
            )
        if self._config.temporal_holdout_start <= mca_report.snapshot_as_of:
            raise SolvencyObservationRunError(
                "The temporal holdout must follow the MCA feature cutoff."
            )
        run_id = self._run_id(mca_report, ibbi_report)
        observation_config = SolvencyObservationConfig(
            run_id=run_id,
            feature_cutoff=mca_report.snapshot_as_of,
            outcome_window_end=self._config.outcome_window_end,
        )
        run_directory = self._resolve(self._config.curated_root) / (
            f"schema-{SOLVENCY_OBSERVATION_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{SOLVENCY_OBSERVATION_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise SolvencyObservationRunError(
                "Solvency-observation run output already exists."
            )
        company_frame = self._read_parts(
            mca_report.company_parts,
            (
                "company_snapshot_id",
                "source_snapshot_id",
                "snapshot_as_of",
                "legal_entity_identifier",
                "legal_entity_identifier_type",
                "legal_entity_identifier_is_valid_format",
                "cin",
                "company_name",
                "company_status",
                "registration_date",
                "state_code",
                "roc_code",
                "company_category",
                "company_subcategory",
                "company_class",
                "listing_status",
                "company_origin",
                "authorised_capital_inr",
                "paid_up_capital_inr",
                "nic_code",
            ),
        )
        address_frame = self._read_parts(
            mca_report.address_parts,
            (
                "legal_entity_identifier",
                "source_snapshot_id",
                "snapshot_as_of",
                "address_group_key",
            ),
        )
        cirp_announcement_frame = pl.read_parquet(
            verify_parquet_artifact(
                ibbi_report.event_part,
                self._config.project_root,
            ),
            columns=["solvency_event_id", "cin", "event_date"],
        )
        observation_frame = build_solvency_observations(
            company_frame,
            address_frame,
            cirp_announcement_frame,
            observation_config,
        )
        observation_artifact = self._publish_observations(
            observation_frame,
            run_directory,
        )
        eligible_frame = observation_frame.filter(pl.col("evaluation_eligible"))
        positive_frame = eligible_frame.filter(
            pl.col("observed_cirp_public_announcement")
        )
        unseen_geography_positive_rows = positive_frame.filter(
            pl.col("state_code").is_in(self._config.unseen_geography_states)
        ).height
        temporal_holdout_positive_rows = positive_frame.filter(
            pl.col("first_cirp_announcement_date")
            >= self._config.temporal_holdout_start
        ).height
        observed_outcome_rows = positive_frame.height
        slice_positive_counts = (
            observed_outcome_rows,
            unseen_geography_positive_rows,
            temporal_holdout_positive_rows,
        )
        ready_to_train = all(
            positive_count
            >= self._config.minimum_positive_outcomes_per_slice
            for positive_count in slice_positive_counts
        )
        report = SolvencyObservationRunReport(
            run_id=run_id,
            mca_source_snapshot_id=mca_report.source_snapshot_id,
            ibbi_source_snapshot_id=ibbi_report.source_snapshot_id,
            feature_cutoff=mca_report.snapshot_as_of,
            outcome_window_start=observation_config.outcome_window_start,
            outcome_window_end=observation_config.outcome_window_end,
            temporal_holdout_start=self._config.temporal_holdout_start,
            unseen_geography_states=self._config.unseen_geography_states,
            company_rows=observation_frame.height,
            evaluation_eligible_rows=eligible_frame.height,
            excluded_prior_outcome_rows=observation_frame.filter(
                pl.col("has_prior_cirp_announcement")
            ).height,
            observed_outcome_rows=observed_outcome_rows,
            unseen_geography_positive_rows=unseen_geography_positive_rows,
            temporal_holdout_positive_rows=temporal_holdout_positive_rows,
            positive_base_rate=(
                observed_outcome_rows / eligible_frame.height
                if eligible_frame.height
                else 0.0
            ),
            minimum_positive_outcomes_per_slice=(
                self._config.minimum_positive_outcomes_per_slice
            ),
            training_decision="READY_TO_TRAIN" if ready_to_train else "DO_NOT_TRAIN",
            quality_status=(
                "passed_training_gate"
                if ready_to_train
                else "insufficient_slice_outcomes"
            ),
            observation_artifact=observation_artifact,
        )
        self._write_report(report_path, report)
        return report

    def _run_id(
        self,
        mca_report: McaCanonicalisationReport,
        ibbi_report: IbbiCanonicalisationReport,
    ) -> str:
        payload = {
            "mca_source_snapshot_id": mca_report.source_snapshot_id,
            "ibbi_source_snapshot_id": ibbi_report.source_snapshot_id,
            "feature_cutoff": mca_report.snapshot_as_of.isoformat(),
            "outcome_window_end": self._config.outcome_window_end.isoformat(),
            "temporal_holdout_start": (
                self._config.temporal_holdout_start.isoformat()
            ),
            "unseen_geography_states": self._config.unseen_geography_states,
            "minimum_positive_outcomes_per_slice": (
                self._config.minimum_positive_outcomes_per_slice
            ),
            "schema_version": SOLVENCY_OBSERVATION_SCHEMA_VERSION,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    def _load_report(
        self,
        configured_path: Path,
        report_model: type[ReportModel],
    ) -> ReportModel:
        report_path = self._resolve(configured_path)
        if not report_path.is_file():
            raise SolvencyObservationRunError(f"Run report is missing: {report_path}.")
        return report_model.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )

    def _read_parts(
        self,
        artifacts: tuple[ParquetArtifact, ...],
        columns: tuple[str, ...],
    ) -> pl.DataFrame:
        if not artifacts:
            raise SolvencyObservationRunError("Required canonical artifacts are empty.")
        return pl.concat(
            [
                pl.scan_parquet(
                    verify_parquet_artifact(artifact, self._config.project_root)
                ).select(columns)
                for artifact in artifacts
            ],
            how="vertical",
        ).collect(engine="streaming")

    def _publish_observations(
        self,
        observation_frame: pl.DataFrame,
        run_directory: Path,
    ) -> ParquetArtifact:
        run_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=run_directory.parent,
            prefix=".assay-solvency-observations-",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            temporary_artifact = write_immutable_parquet(
                observation_frame,
                temporary_directory / "solvency_observation.parquet",
                self._config.parquet_compression,
            )
            os.rename(temporary_directory, run_directory)
        published_path = run_directory / "solvency_observation.parquet"
        return ParquetArtifact(
            path=str(published_path),
            rows=temporary_artifact.rows,
            bytes=published_path.stat().st_size,
            sha256=sha256_file(published_path),
        )

    @staticmethod
    def _write_report(
        report_path: Path,
        report: SolvencyObservationRunReport,
    ) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    report.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                ).encode()
            )
            report_file.flush()
            os.fsync(report_file.fileno())
