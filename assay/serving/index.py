"""Freeze a CIN- and name-addressable merchant lookup index for serving.

The index is a projection of one frozen solvency observation run. It never adds
a feature, never recomputes an outcome, and never stores the training label. It
exists so that a KYB analyst request can resolve a public identifier to the
exact as-of feature row the evaluated model was built on.
"""

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
from assay.contracts.names import (
    COMPANY_NAME_NORMALISATION_VERSION,
    legal_name_expression,
    strict_name_expression,
)
from assay.solvency.runner import SolvencyObservationRunReport
from assay.solvency.training_data import (
    SOLVENCY_FEATURE_SOURCE_COLUMNS,
    SOLVENCY_MODEL_FEATURES,
    solvency_feature_expressions,
)

MERCHANT_INDEX_SCHEMA_VERSION = "1.0.0"
MERCHANT_INDEX_IDENTITY_COLUMNS = (
    "cin",
    "company_name",
    "company_name_normalized_strict",
    "company_name_normalized_legal",
    "strict_name_company_count",
    "registration_date",
    "authorised_capital_inr",
    "paid_up_capital_inr",
    "shared_address_company_count",
    "address_registration_month_company_count",
    "nic_code",
)
MERCHANT_INDEX_COLUMNS = (
    *MERCHANT_INDEX_IDENTITY_COLUMNS,
    *SOLVENCY_MODEL_FEATURES,
)


class MerchantIndexError(RuntimeError):
    """A merchant lookup index failed its frozen serving contract."""


class MerchantIndexConfig(BaseModel):
    """Inputs and immutable output locations for one index build."""

    model_config = ConfigDict(frozen=True)

    observation_report_path: Path
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/merchant_risk_index")
    generated_report_root: Path = Path("data/generated/merchant_risk_index")
    parquet_compression: str = "zstd"


class MerchantRiskIndexReport(BaseModel):
    """Checksummed provenance for the served merchant lookup index."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = MERCHANT_INDEX_SCHEMA_VERSION
    observation_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mca_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    ibbi_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    name_normalisation_version: str = COMPANY_NAME_NORMALISATION_VERSION
    label_boundary: str = "cirp_public_announcement_outcome_not_fraud"
    indexed_companies: int = Field(gt=0)
    distinct_strict_names: int = Field(gt=0)
    companies_sharing_a_strict_name: int = Field(ge=0)
    index_artifact: ParquetArtifact


def build_merchant_risk_index(observation_frame: pl.DataFrame) -> pl.DataFrame:
    """Project eligible companies into a CIN-sorted, name-searchable index."""

    required_columns = {
        "cin",
        "company_name",
        "evaluation_eligible",
        "nic_code",
        "authorised_capital_inr",
        "paid_up_capital_inr",
        "registration_date",
        *SOLVENCY_FEATURE_SOURCE_COLUMNS,
    }
    if missing_columns := required_columns - set(observation_frame.columns):
        raise MerchantIndexError(
            "Solvency observations are missing: "
            f"{', '.join(sorted(missing_columns))}."
        )
    eligible_frame = observation_frame.filter(
        pl.col("evaluation_eligible") & pl.col("cin").is_not_null()
    )
    if eligible_frame.is_empty():
        raise MerchantIndexError("Solvency observations contain no eligible CIN.")
    if eligible_frame["cin"].n_unique() != eligible_frame.height:
        raise MerchantIndexError("Merchant index requires one row per CIN.")

    strict_name = strict_name_expression("company_name")
    indexed_frame = eligible_frame.with_columns(
        *solvency_feature_expressions(),
        strict_name.alias("company_name_normalized_strict"),
        legal_name_expression(strict_name).alias(
            "company_name_normalized_legal"
        ),
        pl.col("authorised_capital_inr").cast(pl.Float64),
        pl.col("paid_up_capital_inr").cast(pl.Float64),
        pl.col("nic_code").fill_null(""),
    ).with_columns(
        pl.len()
        .over("company_name_normalized_strict")
        .cast(pl.Int32)
        .alias("strict_name_company_count")
    )
    return indexed_frame.select(MERCHANT_INDEX_COLUMNS).sort("cin")


class MerchantRiskIndexBuilder:
    """Validate a frozen observation run and publish one immutable index."""

    def __init__(self, config: MerchantIndexConfig) -> None:
        self._config = config

    def run(self) -> MerchantRiskIndexReport:
        observation_report_path = self._resolve(
            self._config.observation_report_path
        )
        observation_report = SolvencyObservationRunReport.model_validate_json(
            observation_report_path.read_text(encoding="utf-8")
        )
        observation_path = verify_parquet_artifact(
            observation_report.observation_artifact,
            self._config.project_root,
        )
        index_frame = build_merchant_risk_index(pl.read_parquet(observation_path))
        run_payload = {
            "observation_run_id": observation_report.run_id,
            "schema_version": MERCHANT_INDEX_SCHEMA_VERSION,
            "name_normalisation_version": COMPANY_NAME_NORMALISATION_VERSION,
            "columns": list(MERCHANT_INDEX_COLUMNS),
        }
        run_id = hashlib.sha256(
            json.dumps(
                run_payload, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        index_path = self._resolve(self._config.curated_root) / (
            f"schema-{MERCHANT_INDEX_SCHEMA_VERSION}/run-{run_id}"
            "/merchant_risk_index.parquet"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{MERCHANT_INDEX_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if index_path.exists() or report_path.exists():
            raise MerchantIndexError("Merchant index output already exists.")
        index_artifact = write_immutable_parquet(
            index_frame,
            index_path,
            self._config.parquet_compression,
        )
        report = MerchantRiskIndexReport(
            run_id=run_id,
            observation_run_id=observation_report.run_id,
            mca_source_snapshot_id=observation_report.mca_source_snapshot_id,
            ibbi_source_snapshot_id=observation_report.ibbi_source_snapshot_id,
            feature_cutoff=observation_report.feature_cutoff,
            indexed_companies=index_frame.height,
            distinct_strict_names=index_frame[
                "company_name_normalized_strict"
            ].n_unique(),
            companies_sharing_a_strict_name=int(
                (index_frame["strict_name_company_count"] > 1).sum()
            ),
            index_artifact=index_artifact,
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
