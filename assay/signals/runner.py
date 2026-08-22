"""Publish reproducible signal-observation datasets from explicit run reports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import (
    ParquetArtifact,
    sha256_file,
    verify_parquet_artifact,
    write_immutable_parquet,
)
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.signals.observations import (
    SIGNAL_OBSERVATION_SCHEMA_VERSION,
    build_signal_observations,
    default_signal_observation_config,
)


class SignalObservationRunError(RuntimeError):
    """A reproducible signal-observation run could not be completed."""


class SignalObservationRunConfig(BaseModel):
    """Explicit upstream reports and outcome window for one derived dataset."""

    model_config = ConfigDict(frozen=True)

    mca_report_path: Path
    nse_report_path: Path
    linkage_report_path: Path
    outcome_window_end: date
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/signal_observation")
    generated_report_root: Path = Path("data/generated/signal_observation")
    shared_address_minimum_companies: int = Field(default=2, ge=2)
    parquet_compression: str = "zstd"


class SignalObservationRunReport(BaseModel):
    """Population and label coverage for one observation dataset."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = SIGNAL_OBSERVATION_SCHEMA_VERSION
    mca_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    nse_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity_linkage_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    outcome_window_start: date
    outcome_window_end: date
    company_rows: int = Field(ge=0)
    evaluation_eligible_rows: int = Field(ge=0)
    observed_adverse_outcome_rows: int = Field(ge=0)
    excluded_prior_outcome_rows: int = Field(ge=0)
    shared_address_signal_rows: int = Field(ge=0)
    address_registration_month_cohort_signal_rows: int = Field(ge=0)
    nse_provenance_complete_assets: int = Field(ge=0)
    quality_status: str
    observation_artifact: ParquetArtifact


class SignalObservationRunner:
    """Resolve, validate, derive, and publish signal observations."""

    def __init__(self, config: SignalObservationRunConfig) -> None:
        self._config = config

    def run(self) -> SignalObservationRunReport:
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
        accepted_mca_quality_statuses = {"passed", "passed_with_quarantine"}
        if (
            not mca_report.acquisition_complete
            or mca_report.quality_status not in accepted_mca_quality_statuses
        ):
            raise SignalObservationRunError(
                "Signal construction requires a complete, quality-passed MCA snapshot."
            )
        if linkage_report.mca_source_snapshot_id != mca_report.source_snapshot_id:
            raise SignalObservationRunError(
                "Entity-linkage and MCA source snapshots do not match."
            )
        if linkage_report.nse_source_snapshot_id != nse_report.source_snapshot_id:
            raise SignalObservationRunError(
                "Entity-linkage and NSE source snapshots do not match."
            )
        if self._config.outcome_window_end <= mca_report.snapshot_as_of:
            raise SignalObservationRunError(
                "Outcome window end must be after the MCA feature cutoff."
            )

        run_payload = json.dumps(
            {
                "mca_source_snapshot_id": mca_report.source_snapshot_id,
                "nse_source_snapshot_id": nse_report.source_snapshot_id,
                "entity_linkage_run_id": linkage_report.run_id,
                "feature_cutoff": mca_report.snapshot_as_of.isoformat(),
                "outcome_window_end": self._config.outcome_window_end.isoformat(),
                "shared_address_minimum_companies": (
                    self._config.shared_address_minimum_companies
                ),
                "schema_version": SIGNAL_OBSERVATION_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        run_id = hashlib.sha256(run_payload).hexdigest()
        observation_config = default_signal_observation_config(
            run_id=run_id,
            feature_cutoff=mca_report.snapshot_as_of,
            outcome_window_end=self._config.outcome_window_end,
        ).model_copy(
            update={
                "shared_address_minimum_companies": (
                    self._config.shared_address_minimum_companies
                )
            }
        )
        run_directory = self._resolve(self._config.curated_root) / (
            f"schema-{SIGNAL_OBSERVATION_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{SIGNAL_OBSERVATION_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise SignalObservationRunError(
                "Signal-observation run output already exists."
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
                "registration_date",
                "state_code",
                "nic_code",
                "paid_up_capital_inr",
            ),
        )
        address_frame = self._read_parts(
            mca_report.address_parts,
            (
                "company_address_snapshot_id",
                "source_snapshot_id",
                "legal_entity_identifier",
                "address_normalized",
                "address_group_key",
                "snapshot_as_of",
            ),
        )
        adverse_frame = self._read_parts(
            nse_report.parts,
            ("adverse_event_id", "event_date"),
        )
        match_decision_frame = pl.read_parquet(
            verify_parquet_artifact(
                linkage_report.decision_artifact,
                self._config.project_root,
            ),
            columns=[
                "adverse_event_id",
                "accepted_company_snapshot_id",
                "review_status",
                "outcome_eligible",
            ],
        )
        observation_frame = build_signal_observations(
            company_frame,
            address_frame,
            adverse_frame,
            match_decision_frame,
            observation_config,
        )
        observation_artifact = self._publish_observations(
            observation_frame,
            run_directory,
        )
        eligible_frame = observation_frame.filter(pl.col("evaluation_eligible"))
        observed_outcome_rows = eligible_frame.filter(
            pl.col("observed_adverse_outcome")
        ).height
        quality_status = (
            "passed"
            if observed_outcome_rows > 0
            else "insufficient_eligible_outcomes"
        )
        report = SignalObservationRunReport(
            run_id=run_id,
            mca_source_snapshot_id=mca_report.source_snapshot_id,
            nse_source_snapshot_id=nse_report.source_snapshot_id,
            entity_linkage_run_id=linkage_report.run_id,
            feature_cutoff=observation_config.feature_cutoff,
            outcome_window_start=observation_config.outcome_window_start,
            outcome_window_end=observation_config.outcome_window_end,
            company_rows=observation_frame.height,
            evaluation_eligible_rows=eligible_frame.height,
            observed_adverse_outcome_rows=observed_outcome_rows,
            excluded_prior_outcome_rows=observation_frame.filter(
                pl.col("has_prior_adverse_outcome")
            ).height,
            shared_address_signal_rows=eligible_frame.filter(
                pl.col("shared_address_signal")
            ).height,
            address_registration_month_cohort_signal_rows=eligible_frame.filter(
                pl.col("address_registration_month_cohort_signal")
            ).height,
            nse_provenance_complete_assets=nse_report.provenance_complete_assets,
            quality_status=quality_status,
            observation_artifact=observation_artifact,
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
            raise SignalObservationRunError(f"Run report is missing: {report_path}.")
        return report_model.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )

    def _read_parts(
        self,
        artifacts: tuple[ParquetArtifact, ...],
        columns: tuple[str, ...],
    ) -> pl.DataFrame:
        if not artifacts:
            raise SignalObservationRunError("Required canonical artifacts are empty.")
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
            prefix=".assay-observations-",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            temporary_artifact = write_immutable_parquet(
                observation_frame,
                temporary_directory / "signal_observation.parquet",
                self._config.parquet_compression,
            )
            os.rename(temporary_directory, run_directory)
        published_path = run_directory / "signal_observation.parquet"
        return ParquetArtifact(
            path=str(published_path),
            rows=temporary_artifact.rows,
            bytes=published_path.stat().st_size,
            sha256=sha256_file(published_path),
        )

    @staticmethod
    def _write_report(
        report_path: Path,
        report: SignalObservationRunReport,
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
