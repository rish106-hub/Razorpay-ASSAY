"""Canonicalise official NSE regulatory actions without creating fraud labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.canonicalisation.mca import (
    CanonicalPartArtifact,
    _write_immutable_parquet,
)
from assay.contracts.provenance import MerchantRiskAsset, load_and_validate_manifests

NSE_ADVERSE_EVENT_SCHEMA_VERSION = "1.0.0"
NSE_SOURCE_IDS = frozenset(
    {"nse_sebi_debarred", "nse_other_authorities_debarred"}
)
NSE_REQUIRED_COLUMNS = frozenset(
    {
        "Order Date",
        "Order Particulars",
        "Entity / Individual Name",
        "PAN",
        "DIN / CIN",
        "Symbol ",
        "Period",
    }
)


class NseCanonicalisationError(RuntimeError):
    """A deterministic official NSE canonicalisation failure."""


class NseCanonicalisationConfig(BaseModel):
    """Paths for one official adverse-regulatory snapshot."""

    model_config = ConfigDict(frozen=True)

    project_root: Path = Path(__file__).resolve().parents[2]
    source_manifest_path: Path = Path("data/manifests/sources.yaml")
    acquisition_manifest_path: Path = Path("data/manifests/acquired.yaml")
    curated_root: Path = Path("data/curated/adverse_event")
    generated_report_root: Path = Path("data/generated/nse_canonicalisation")
    parquet_compression: str = "zstd"


class NseCanonicalisationReport(BaseModel):
    """Coverage and provenance metrics for official regulatory events."""

    model_config = ConfigDict(frozen=True)

    source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = NSE_ADVERSE_EVENT_SCHEMA_VERSION
    adverse_event_rows: int = Field(ge=0)
    sebi_event_rows: int = Field(ge=0)
    other_authority_event_rows: int = Field(ge=0)
    cin_candidate_rows: int = Field(ge=0)
    din_candidate_rows: int = Field(ge=0)
    pan_candidate_rows: int = Field(ge=0)
    missing_entity_name_rows: int = Field(ge=0)
    missing_event_date_rows: int = Field(ge=0)
    provenance_complete_assets: int = Field(ge=0)
    parts: tuple[CanonicalPartArtifact, ...]


def _normalised_name_expression(source_column: str) -> pl.Expr:
    return (
        pl.col(source_column)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _canonicalise_adverse_frame(
    source_frame: pl.DataFrame,
    asset: MerchantRiskAsset,
    source_snapshot_id: str,
) -> pl.DataFrame:
    identifier_normalized = (
        pl.col("DIN / CIN")
        .fill_null("")
        .str.strip_chars()
        .str.to_uppercase()
    )
    pan_normalized = (
        pl.col("PAN").fill_null("").str.strip_chars().str.to_uppercase()
    )
    strict_entity_name = _normalised_name_expression("Entity / Individual Name")
    legal_entity_name = (
        strict_entity_name.str.replace_all(
            r"\b(PRIVATE|PVT|LIMITED|LTD|LLP|COMPANY|CO)\b", " "
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    source_authority = (
        "SEBI"
        if asset.source_id == "nse_sebi_debarred"
        else "OTHER_AUTHORITY"
    )
    return source_frame.select(
        pl.concat_str(
            [
                pl.lit(asset.source_id),
                pl.lit(asset.sha256),
                pl.col("source_row_number").cast(pl.String),
            ],
            separator=":",
        ).alias("adverse_event_id"),
        pl.lit(asset.source_id).alias("source_id"),
        pl.col("source_row_number").cast(pl.String).alias("source_record_id"),
        pl.lit(asset.sha256).alias("source_file_sha256"),
        pl.lit(source_snapshot_id).alias("source_snapshot_id"),
        pl.lit(asset.retrieved_at).alias("observed_at"),
        pl.lit(NSE_ADVERSE_EVENT_SCHEMA_VERSION).alias("schema_version"),
        pl.lit(source_authority).alias("event_source_authority"),
        pl.lit("regulatory_debarment_or_action").alias("event_type"),
        pl.lit(True).alias("adverse_regulatory_outcome"),
        pl.col("Order Date").cast(pl.Date).alias("event_date"),
        pl.col("Order Particulars").fill_null("").alias("order_particulars"),
        pl.col("Entity / Individual Name").fill_null("").alias("entity_name_raw"),
        strict_entity_name.alias("entity_name_normalized_strict"),
        legal_entity_name.alias("entity_name_normalized_legal"),
        pl.col("PAN").fill_null("").alias("pan_raw"),
        pl.when(pan_normalized.str.contains(r"^[A-Z]{5}[0-9]{4}[A-Z]$"))
        .then(pan_normalized)
        .otherwise(None)
        .alias("pan_candidate"),
        pl.col("DIN / CIN").fill_null("").alias("din_cin_raw"),
        pl.when(identifier_normalized.str.contains(r"^[A-Z0-9]{21}$"))
        .then(identifier_normalized)
        .otherwise(None)
        .alias("cin_candidate"),
        pl.when(identifier_normalized.str.contains(r"^[0-9]{8}$"))
        .then(identifier_normalized)
        .otherwise(None)
        .alias("din_candidate"),
        pl.col("Symbol ").fill_null("").str.strip_chars().alias("symbol"),
        pl.col("Period").fill_null("").alias("period_raw"),
        pl.col("NSE Circular No. (For Debarment)")
        .fill_null("")
        .alias("nse_debarment_circular"),
        pl.col("Date of NSE circular")
        .cast(pl.Date, strict=False)
        .alias("nse_debarment_circular_date"),
        pl.col("NSE Circular No. (For Revocation)")
        .fill_null("")
        .alias("nse_revocation_circular"),
        pl.col("Date of NSE circular. (For Revocation)")
        .cast(pl.Date, strict=False)
        .alias("nse_revocation_circular_date"),
    )


class NseCanonicaliser:
    """Create immutable adverse-event Parquet from official NSE files."""

    def __init__(self, config: NseCanonicalisationConfig) -> None:
        self._config = config

    def run(self) -> NseCanonicalisationReport:
        source_manifest_path = self._resolve(self._config.source_manifest_path)
        acquisition_manifest_path = self._resolve(
            self._config.acquisition_manifest_path
        )
        _, acquisition_manifest = load_and_validate_manifests(
            source_manifest_path,
            acquisition_manifest_path,
        )
        nse_assets = tuple(
            asset
            for asset in acquisition_manifest.assets
            if asset.source_id in NSE_SOURCE_IDS
        )
        if {asset.source_id for asset in nse_assets} != NSE_SOURCE_IDS:
            raise NseCanonicalisationError(
                "Both official NSE source assets are required."
            )
        source_snapshot_payload = json.dumps(
            [asset.model_dump(mode="json") for asset in nse_assets],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        source_snapshot_id = hashlib.sha256(source_snapshot_payload).hexdigest()
        snapshot_directory = self._resolve(self._config.curated_root) / (
            f"snapshot-{source_snapshot_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"snapshot-{source_snapshot_id}.report.json"
        )
        if snapshot_directory.exists() or report_path.exists():
            raise NseCanonicalisationError(
                "NSE canonical snapshot output already exists."
            )

        parts: list[CanonicalPartArtifact] = []
        canonical_frames: list[pl.DataFrame] = []
        for asset in nse_assets:
            raw_asset_path = self._resolve(Path(asset.local_path))
            self._verify_asset(asset, raw_asset_path)
            excel_engine = (
                "openpyxl"
                if asset.format == "xlsx_container_with_xls_extension"
                else "xlrd"
            )
            source_pandas_frame = pd.read_excel(
                raw_asset_path,
                sheet_name=asset.sheet,
                engine=excel_engine,
                dtype=str,
            ).fillna("")
            if len(source_pandas_frame) != asset.rows:
                raise NseCanonicalisationError(
                    f"NSE row count does not match manifest for {asset.id}."
                )
            missing_columns = NSE_REQUIRED_COLUMNS - set(source_pandas_frame.columns)
            if missing_columns:
                missing_column_list = ", ".join(sorted(missing_columns))
                raise NseCanonicalisationError(
                    f"NSE asset {asset.id} is missing columns: {missing_column_list}."
                )
            source_pandas_frame["Order Date"] = pd.to_datetime(
                source_pandas_frame["Order Date"],
                format="%d-%b-%Y",
                errors="coerce",
            )
            for circular_date_column in (
                "Date of NSE circular",
                "Date of NSE circular. (For Revocation)",
            ):
                source_pandas_frame[circular_date_column] = pd.to_datetime(
                    source_pandas_frame[circular_date_column],
                    format="mixed",
                    dayfirst=True,
                    errors="coerce",
                )
            source_frame = pl.from_pandas(source_pandas_frame).with_columns(
                pl.int_range(2, len(source_pandas_frame) + 2, eager=True).alias(
                    "source_row_number"
                )
            )
            canonical_frame = _canonicalise_adverse_frame(
                source_frame,
                asset=asset,
                source_snapshot_id=source_snapshot_id,
            )
            canonical_frames.append(canonical_frame)
            parts.append(
                _write_immutable_parquet(
                    canonical_frame,
                    snapshot_directory / f"part-{asset.source_id}.parquet",
                    self._config.parquet_compression,
                )
            )

        combined_frame = pl.concat(canonical_frames, how="vertical")
        report = NseCanonicalisationReport(
            source_snapshot_id=source_snapshot_id,
            adverse_event_rows=combined_frame.height,
            sebi_event_rows=combined_frame.filter(
                pl.col("event_source_authority") == "SEBI"
            ).height,
            other_authority_event_rows=combined_frame.filter(
                pl.col("event_source_authority") == "OTHER_AUTHORITY"
            ).height,
            cin_candidate_rows=combined_frame["cin_candidate"].is_not_null().sum(),
            din_candidate_rows=combined_frame["din_candidate"].is_not_null().sum(),
            pan_candidate_rows=combined_frame["pan_candidate"].is_not_null().sum(),
            missing_entity_name_rows=(
                combined_frame["entity_name_normalized_strict"] == ""
            ).sum(),
            missing_event_date_rows=combined_frame["event_date"].is_null().sum(),
            provenance_complete_assets=sum(
                asset.provenance_status == "complete" for asset in nse_assets
            ),
            parts=tuple(parts),
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    report.model_dump(mode="json"), indent=2, sort_keys=True
                ).encode("utf-8")
            )
            report_file.flush()
            os.fsync(report_file.fileno())
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _verify_asset(asset: MerchantRiskAsset, raw_asset_path: Path) -> None:
        if not raw_asset_path.is_file():
            raise NseCanonicalisationError(f"NSE raw asset is missing: {asset.id}.")
        if raw_asset_path.stat().st_size != asset.bytes:
            raise NseCanonicalisationError(
                f"NSE raw asset size mismatch: {asset.id}."
            )
        raw_asset_sha256 = hashlib.sha256(raw_asset_path.read_bytes()).hexdigest()
        if raw_asset_sha256 != asset.sha256:
            raise NseCanonicalisationError(
                f"NSE raw asset checksum mismatch: {asset.id}."
            )
