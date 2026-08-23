"""Freeze compact, entity-disjoint training splits for solvency modeling."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import (
    ParquetArtifact,
    verify_parquet_artifact,
    write_immutable_parquet,
)
from assay.solvency.feature_sets import BASELINE_WITH_STATUS
from assay.solvency.runner import SolvencyObservationRunReport

SOLVENCY_MODEL_DATA_SCHEMA_VERSION = "1.0.0"

# The frozen model-data Parquet always carries the widest feature set, so a
# narrower named set can be fitted from the same bytes without re-cutting a
# split. These aliases keep the published column contract stable while
# `assay.solvency.feature_sets` owns the definition.
SOLVENCY_NUMERIC_FEATURES = BASELINE_WITH_STATUS.numeric_features
SOLVENCY_CATEGORICAL_FEATURES = BASELINE_WITH_STATUS.categorical_features
SOLVENCY_MODEL_FEATURES = BASELINE_WITH_STATUS.model_features


SOLVENCY_FEATURE_SOURCE_COLUMNS = (
    "company_age_days",
    "log_authorised_capital_inr",
    "log_paid_up_capital_inr",
    "paid_to_authorised_capital_ratio",
    "shared_address_company_count",
    "address_registration_month_company_count",
    *SOLVENCY_CATEGORICAL_FEATURES,
)


def solvency_feature_expressions() -> tuple[pl.Expr, ...]:
    """Return the single frozen train-and-serve feature derivation."""

    return (
        (pl.col("company_age_days") / 365.25)
        .cast(pl.Float32)
        .alias("company_age_years"),
        pl.col("paid_to_authorised_capital_ratio")
        .clip(0.0, 10.0)
        .cast(pl.Float32),
        pl.col("shared_address_company_count")
        .cast(pl.Float64)
        .log1p()
        .cast(pl.Float32)
        .alias("log_shared_address_company_count"),
        pl.col("address_registration_month_company_count")
        .cast(pl.Float64)
        .log1p()
        .cast(pl.Float32)
        .alias("log_address_registration_month_company_count"),
        pl.col("log_authorised_capital_inr").cast(pl.Float32),
        pl.col("log_paid_up_capital_inr").cast(pl.Float32),
        *[
            pl.col(feature_name).fill_null("UNKNOWN").cast(pl.String)
            for feature_name in SOLVENCY_CATEGORICAL_FEATURES
        ],
    )


class SolvencyTrainingDataError(RuntimeError):
    """A frozen solvency modeling dataset failed its split contract."""


class SolvencyTrainingDataConfig(BaseModel):
    """Explicit validation, temporal, geography, and control partitions."""

    model_config = ConfigDict(frozen=True)

    observation_report_path: Path
    validation_outcome_start: date = date(2025, 7, 1)
    temporal_test_outcome_start: date = date(2026, 1, 1)
    unseen_geography_states: tuple[str, ...] = (
        "karnataka",
        "kerala",
        "tamil nadu",
        "telangana",
    )
    negative_validation_percent: int = Field(default=15, ge=1, le=40)
    negative_temporal_test_percent: int = Field(default=15, ge=1, le=40)
    split_hash_seed: int = Field(default=106, ge=0)
    minimum_positive_outcomes_per_split: int = Field(default=20, ge=20)
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/solvency_model_data")
    generated_report_root: Path = Path("data/generated/solvency_model_data")
    parquet_compression: str = "zstd"


class SolvencySplitEvidence(BaseModel):
    """Rows and base rate for one entity-disjoint modeling partition."""

    model_config = ConfigDict(frozen=True)

    rows: int = Field(gt=0)
    positive_rows: int = Field(ge=0)
    positive_rate: float = Field(ge=0, le=1)


class SolvencyTrainingDataReport(BaseModel):
    """Checksummed split evidence handed to the controlled Colab runner."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = SOLVENCY_MODEL_DATA_SCHEMA_VERSION
    observation_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    validation_outcome_start: date
    temporal_test_outcome_start: date
    unseen_geography_states: tuple[str, ...]
    split_hash_seed: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    split_evidence: dict[str, SolvencySplitEvidence]
    split_decision: str
    model_data_artifact: ParquetArtifact


def build_solvency_model_data(
    observation_frame: pl.DataFrame,
    config: SolvencyTrainingDataConfig,
) -> pl.DataFrame:
    """Derive compact features and split entities without label overlap."""

    required_columns = {
        "company_snapshot_id",
        "cin",
        "evaluation_eligible",
        "observed_cirp_public_announcement",
        "first_cirp_announcement_date",
        "company_age_days",
        "log_authorised_capital_inr",
        "log_paid_up_capital_inr",
        "paid_to_authorised_capital_ratio",
        "shared_address_company_count",
        "address_registration_month_company_count",
        *SOLVENCY_CATEGORICAL_FEATURES,
    }
    if missing_columns := required_columns - set(observation_frame.columns):
        raise SolvencyTrainingDataError(
            "Solvency observations are missing: "
            f"{', '.join(sorted(missing_columns))}."
        )
    eligible_frame = observation_frame.filter(pl.col("evaluation_eligible"))
    if eligible_frame["company_snapshot_id"].n_unique() != eligible_frame.height:
        raise SolvencyTrainingDataError(
            "Solvency modeling data must contain one row per company."
        )
    control_hash_bucket = (
        pl.col("company_snapshot_id").hash(config.split_hash_seed) % 100
    )
    temporal_negative_boundary = config.negative_temporal_test_percent
    validation_negative_boundary = (
        temporal_negative_boundary + config.negative_validation_percent
    )
    target = pl.col("observed_cirp_public_announcement")
    event_date = pl.col("first_cirp_announcement_date")
    split = (
        pl.when(pl.col("state_code").is_in(config.unseen_geography_states))
        .then(pl.lit("geography_test"))
        .when(target & (event_date >= config.temporal_test_outcome_start))
        .then(pl.lit("temporal_test"))
        .when(target & (event_date >= config.validation_outcome_start))
        .then(pl.lit("validation"))
        .when(target)
        .then(pl.lit("training"))
        .when(control_hash_bucket < temporal_negative_boundary)
        .then(pl.lit("temporal_test"))
        .when(control_hash_bucket < validation_negative_boundary)
        .then(pl.lit("validation"))
        .otherwise(pl.lit("training"))
    )
    return eligible_frame.with_columns(
        *solvency_feature_expressions(),
        target.cast(pl.Int8).alias("target"),
        split.alias("dataset_split"),
    ).select(
        "company_snapshot_id",
        "cin",
        "dataset_split",
        "target",
        "first_cirp_announcement_date",
        *SOLVENCY_MODEL_FEATURES,
    )


class SolvencyTrainingDataBuilder:
    """Validate, derive, checksum, and report one Colab modeling handoff."""

    def __init__(self, config: SolvencyTrainingDataConfig) -> None:
        self._config = config

    def run(self) -> SolvencyTrainingDataReport:
        observation_report_path = self._resolve(
            self._config.observation_report_path
        )
        observation_report = SolvencyObservationRunReport.model_validate_json(
            observation_report_path.read_text(encoding="utf-8")
        )
        if observation_report.training_decision != "READY_TO_TRAIN":
            raise SolvencyTrainingDataError(
                "Solvency observations have not passed the training gate."
            )
        if (
            observation_report.unseen_geography_states
            != self._config.unseen_geography_states
        ):
            raise SolvencyTrainingDataError(
                "Modeling and observation unseen-geography contracts differ."
            )
        if (
            self._config.validation_outcome_start
            >= self._config.temporal_test_outcome_start
        ):
            raise SolvencyTrainingDataError(
                "Validation outcomes must precede temporal-test outcomes."
            )
        observation_path = verify_parquet_artifact(
            observation_report.observation_artifact,
            self._config.project_root,
        )
        observation_frame = pl.read_parquet(observation_path)
        model_frame = build_solvency_model_data(observation_frame, self._config)
        run_payload = {
            "observation_run_id": observation_report.run_id,
            "validation_outcome_start": (
                self._config.validation_outcome_start.isoformat()
            ),
            "temporal_test_outcome_start": (
                self._config.temporal_test_outcome_start.isoformat()
            ),
            "unseen_geography_states": self._config.unseen_geography_states,
            "negative_validation_percent": (
                self._config.negative_validation_percent
            ),
            "negative_temporal_test_percent": (
                self._config.negative_temporal_test_percent
            ),
            "split_hash_seed": self._config.split_hash_seed,
            "schema_version": SOLVENCY_MODEL_DATA_SCHEMA_VERSION,
        }
        run_id = hashlib.sha256(
            json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        run_directory = self._resolve(self._config.curated_root) / (
            f"schema-{SOLVENCY_MODEL_DATA_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{SOLVENCY_MODEL_DATA_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise SolvencyTrainingDataError(
                "Solvency modeling dataset output already exists."
            )
        model_data_artifact = write_immutable_parquet(
            model_frame,
            run_directory / "solvency_model_data.parquet",
            self._config.parquet_compression,
        )
        split_evidence: dict[str, SolvencySplitEvidence] = {}
        for split_name in (
            "training",
            "validation",
            "geography_test",
            "temporal_test",
        ):
            split_frame = model_frame.filter(
                pl.col("dataset_split") == split_name
            )
            positive_rows = int(split_frame["target"].sum())
            split_evidence[split_name] = SolvencySplitEvidence(
                rows=split_frame.height,
                positive_rows=positive_rows,
                positive_rate=positive_rows / split_frame.height,
            )
        split_decision = (
            "READY_FOR_COLAB"
            if all(
                evidence.positive_rows
                >= self._config.minimum_positive_outcomes_per_split
                for evidence in split_evidence.values()
            )
            else "DO_NOT_TRAIN"
        )
        report = SolvencyTrainingDataReport(
            run_id=run_id,
            observation_run_id=observation_report.run_id,
            feature_cutoff=observation_report.feature_cutoff,
            validation_outcome_start=self._config.validation_outcome_start,
            temporal_test_outcome_start=self._config.temporal_test_outcome_start,
            unseen_geography_states=self._config.unseen_geography_states,
            split_hash_seed=self._config.split_hash_seed,
            numeric_features=SOLVENCY_NUMERIC_FEATURES,
            categorical_features=SOLVENCY_CATEGORICAL_FEATURES,
            split_evidence=split_evidence,
            split_decision=split_decision,
            model_data_artifact=model_data_artifact,
        )
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
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path
