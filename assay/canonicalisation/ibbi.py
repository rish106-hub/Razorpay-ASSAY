"""Canonicalise official IBBI CIRP announcements as solvency outcomes."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import ParquetArtifact, write_immutable_parquet
from assay.contracts.identifiers import INDIAN_CIN_PATTERN
from assay.contracts.provenance import MerchantRiskAsset, load_and_validate_manifests

IBBI_SOURCE_ID = "ibbi_cirp_public_announcements"
IBBI_SOLVENCY_EVENT_SCHEMA_VERSION = "1.0.0"
IBBI_CIRP_ANNOUNCEMENT_TYPE = (
    "Public Announcement of Corporate Insolvency Resolution Process"
)
IBBI_REQUIRED_COLUMNS = (
    "Announcement Type",
    "Date of Announcement",
    "Last date of Submission",
    "Name of Corporate Debtor",
    "CIN No.",
    "Name of Applicant",
    "Name of Insolvency Professional",
    "Address of Insolvency Professional",
    "Remarks",
)
_CIN_PATTERN = re.compile(INDIAN_CIN_PATTERN)


class IbbiCanonicalisationError(RuntimeError):
    """An official IBBI solvency-outcome snapshot failed validation."""


class IbbiCanonicalisationConfig(BaseModel):
    """Immutable locations and the pre-outcome MCA feature cutoff."""

    model_config = ConfigDict(frozen=True)

    project_root: Path = Path(__file__).resolve().parents[2]
    source_manifest_path: Path = Path("data/manifests/sources.yaml")
    acquisition_manifest_path: Path = Path("data/manifests/acquired.yaml")
    curated_root: Path = Path("data/curated/solvency_event")
    generated_report_root: Path = Path("data/generated/ibbi_canonicalisation")
    mca_feature_cutoff: date = date(2023, 11, 3)
    parquet_compression: str = "zstd"


class IbbiCanonicalisationReport(BaseModel):
    """Quality evidence for exact-CIN merchant-solvency observations."""

    model_config = ConfigDict(frozen=True)

    source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = IBBI_SOLVENCY_EVENT_SCHEMA_VERSION
    source_rows: int = Field(ge=0)
    solvency_event_rows: int = Field(ge=0)
    unique_cin_count: int = Field(ge=0)
    post_mca_cutoff_rows: int = Field(ge=0)
    post_mca_cutoff_unique_cin_count: int = Field(ge=0)
    duplicate_cin_date_rows: int = Field(ge=0)
    invalid_cin_rows: int = Field(ge=0)
    missing_announcement_date_rows: int = Field(ge=0)
    malformed_tsv_rows: int = Field(ge=0)
    provenance_complete: bool
    event_part: ParquetArtifact
    quarantine_part: ParquetArtifact


def _normalise_entity_name(entity_name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", entity_name.upper())).strip()


def _parse_date(raw_date: str) -> date | None:
    try:
        day, month, year = (int(part) for part in raw_date.strip().split("-"))
        return date(year, month, day)
    except ValueError:
        return None


class IbbiCanonicaliser:
    """Publish CIRP public announcements without converting them to fraud labels."""

    def __init__(self, config: IbbiCanonicalisationConfig) -> None:
        self._config = config

    def run(self) -> IbbiCanonicalisationReport:
        _, acquisition_manifest = load_and_validate_manifests(
            self._resolve(self._config.source_manifest_path),
            self._resolve(self._config.acquisition_manifest_path),
        )
        ibbi_assets = tuple(
            asset
            for asset in acquisition_manifest.assets
            if asset.source_id == IBBI_SOURCE_ID
        )
        if len(ibbi_assets) != 1:
            raise IbbiCanonicalisationError(
                "Exactly one official IBBI CIRP announcement asset is required."
            )
        asset = ibbi_assets[0]
        raw_asset_path = self._resolve(Path(asset.local_path))
        self._verify_asset(asset, raw_asset_path)
        source_snapshot_id = hashlib.sha256(
            json.dumps(
                asset.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        versioned_snapshot_name = (
            f"schema-{IBBI_SOLVENCY_EVENT_SCHEMA_VERSION}/"
            f"snapshot-{source_snapshot_id}"
        )
        snapshot_directory = (
            self._resolve(self._config.curated_root) / versioned_snapshot_name
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"{versioned_snapshot_name}.report.json"
        )
        if snapshot_directory.exists() or report_path.exists():
            raise IbbiCanonicalisationError(
                "IBBI canonical snapshot output already exists."
            )

        source_rows, event_records, quarantine_records = self._parse_export(
            asset,
            raw_asset_path,
            source_snapshot_id,
        )
        if source_rows != asset.rows:
            raise IbbiCanonicalisationError(
                "IBBI row count does not match the acquisition manifest."
            )
        event_frame = pl.DataFrame(event_records)
        quarantine_frame = pl.DataFrame(
            quarantine_records,
            schema={
                "source_row_number": pl.Int64,
                "quarantine_reason": pl.String,
                "raw_field_count": pl.Int64,
                "raw_row_json": pl.String,
            },
        )
        event_part = write_immutable_parquet(
            event_frame,
            snapshot_directory / "part-cirp-public-announcements.parquet",
            self._config.parquet_compression,
        )
        quarantine_part = write_immutable_parquet(
            quarantine_frame,
            snapshot_directory / "quarantine-malformed-tsv.parquet",
            self._config.parquet_compression,
        )
        unique_cin_count = event_frame["cin"].n_unique()
        post_cutoff_frame = event_frame.filter(
            pl.col("event_date") > self._config.mca_feature_cutoff
        )
        duplicate_cin_date_rows = event_frame.height - event_frame.unique(
            subset=["cin", "event_date"]
        ).height
        report = IbbiCanonicalisationReport(
            source_snapshot_id=source_snapshot_id,
            source_rows=source_rows,
            solvency_event_rows=event_frame.height,
            unique_cin_count=unique_cin_count,
            post_mca_cutoff_rows=post_cutoff_frame.height,
            post_mca_cutoff_unique_cin_count=post_cutoff_frame["cin"].n_unique(),
            duplicate_cin_date_rows=duplicate_cin_date_rows,
            invalid_cin_rows=sum(
                record["quarantine_reason"] == "invalid_cin"
                for record in quarantine_records
            ),
            missing_announcement_date_rows=sum(
                record["quarantine_reason"] == "missing_announcement_date"
                for record in quarantine_records
            ),
            malformed_tsv_rows=sum(
                record["quarantine_reason"] == "malformed_field_count"
                for record in quarantine_records
            ),
            provenance_complete=asset.provenance_status == "complete",
            event_part=event_part,
            quarantine_part=quarantine_part,
        )
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
        return report

    def _parse_export(
        self,
        asset: MerchantRiskAsset,
        raw_asset_path: Path,
        source_snapshot_id: str,
    ) -> tuple[int, list[dict[str, object]], list[dict[str, object]]]:
        event_records: list[dict[str, object]] = []
        quarantine_records: list[dict[str, object]] = []
        with raw_asset_path.open(encoding="utf-8", newline="") as export_file:
            reader = csv.reader(export_file, delimiter="\t")
            header = tuple(value.strip() for value in next(reader))
            if header != IBBI_REQUIRED_COLUMNS:
                raise IbbiCanonicalisationError(
                    "IBBI export columns do not match the frozen contract."
                )
            source_rows = 0
            for source_row_number, raw_row in enumerate(reader, start=2):
                source_rows += 1
                if len(raw_row) != len(IBBI_REQUIRED_COLUMNS):
                    quarantine_records.append(
                        {
                            "source_row_number": source_row_number,
                            "quarantine_reason": "malformed_field_count",
                            "raw_field_count": len(raw_row),
                            "raw_row_json": json.dumps(raw_row, ensure_ascii=True),
                        }
                    )
                    continue
                row = dict(
                    zip(
                        IBBI_REQUIRED_COLUMNS,
                        (value.strip() for value in raw_row),
                        strict=True,
                    )
                )
                cin = row["CIN No."].upper()
                event_date = _parse_date(row["Date of Announcement"])
                quarantine_reason = None
                if not _CIN_PATTERN.fullmatch(cin):
                    quarantine_reason = "invalid_cin"
                elif event_date is None:
                    quarantine_reason = "missing_announcement_date"
                elif row["Announcement Type"] != IBBI_CIRP_ANNOUNCEMENT_TYPE:
                    quarantine_reason = "unexpected_announcement_type"
                if quarantine_reason is not None:
                    quarantine_records.append(
                        {
                            "source_row_number": source_row_number,
                            "quarantine_reason": quarantine_reason,
                            "raw_field_count": len(raw_row),
                            "raw_row_json": json.dumps(raw_row, ensure_ascii=True),
                        }
                    )
                    continue
                event_records.append(
                    {
                        "solvency_event_id": hashlib.sha256(
                            f"{asset.sha256}:{source_row_number}".encode()
                        ).hexdigest(),
                        "source_id": asset.source_id,
                        "source_record_id": str(source_row_number),
                        "source_file_sha256": asset.sha256,
                        "source_snapshot_id": source_snapshot_id,
                        "observed_at": asset.retrieved_at,
                        "schema_version": IBBI_SOLVENCY_EVENT_SCHEMA_VERSION,
                        "event_source_authority": "IBBI",
                        "event_type": "cirp_public_announcement",
                        "cirp_public_announcement_outcome": True,
                        "event_date": event_date,
                        "event_date_semantics": "public_announcement_date",
                        "cin": cin,
                        "corporate_debtor_name_raw": row[
                            "Name of Corporate Debtor"
                        ],
                        "corporate_debtor_name_normalized": _normalise_entity_name(
                            row["Name of Corporate Debtor"]
                        ),
                        "claims_submission_deadline": _parse_date(
                            row["Last date of Submission"]
                        ),
                        "applicant_name": row["Name of Applicant"],
                        "insolvency_professional_name": row[
                            "Name of Insolvency Professional"
                        ],
                        "insolvency_professional_address": row[
                            "Address of Insolvency Professional"
                        ],
                        "remarks": row["Remarks"],
                    }
                )
        return source_rows, event_records, quarantine_records

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _verify_asset(asset: MerchantRiskAsset, raw_asset_path: Path) -> None:
        if not raw_asset_path.is_file():
            raise IbbiCanonicalisationError(
                f"IBBI raw asset is missing: {asset.id}."
            )
        if raw_asset_path.stat().st_size != asset.bytes:
            raise IbbiCanonicalisationError(
                f"IBBI raw asset size mismatch: {asset.id}."
            )
        asset_sha256 = hashlib.sha256(raw_asset_path.read_bytes()).hexdigest()
        if asset_sha256 != asset.sha256:
            raise IbbiCanonicalisationError(
                f"IBBI raw asset checksum mismatch: {asset.id}."
            )
