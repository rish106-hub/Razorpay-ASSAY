"""Canonicalise immutable MCA pages into versioned merchant entity snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.acquisition.mca import McaAcquisitionCheckpoint

MCA_CANONICAL_SCHEMA_VERSION = "1.0.0"
MCA_SOURCE_DATA_THROUGH = date(2023, 11, 3)
MCA_CANONICAL_SOURCE_FIELDS = frozenset(
    {
        "AuthorizedCapital",
        "CIN",
        "CompanyCategory",
        "CompanyClass",
        "CompanyIndian/Foreign Company",
        "CompanyIndustrialClassification",
        "CompanyName",
        "CompanyROCcode",
        "CompanyRegistrationdate_date",
        "CompanyStateCode",
        "CompanyStatus",
        "CompanySubCategory",
        "Listingstatus",
        "PaidupCapital",
        "Registered_Office_Address",
        "nic_code",
    }
)


class McaCanonicalisationError(RuntimeError):
    """A deterministic MCA canonicalisation failure."""


class McaCanonicalisationConfig(BaseModel):
    """Paths and evidence boundaries for one MCA source snapshot."""

    model_config = ConfigDict(frozen=True)

    raw_directory: Path = Path("data/raw/mca_company_master/pagesize-10000")
    staged_root: Path = Path("data/staged/mca_company_master")
    curated_company_root: Path = Path("data/curated/company_snapshot")
    curated_address_root: Path = Path("data/curated/company_address_snapshot")
    generated_report_root: Path = Path("data/generated/mca_canonicalisation")
    snapshot_as_of: date = MCA_SOURCE_DATA_THROUGH
    allow_incomplete_acquisition: bool = False
    parquet_compression: str = "zstd"


class CanonicalPartArtifact(BaseModel):
    """Checksum and row count for one immutable Parquet artifact."""

    model_config = ConfigDict(frozen=True)

    path: str
    rows: int = Field(ge=0)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class McaCanonicalisationReport(BaseModel):
    """Quality evidence for one canonical MCA snapshot run."""

    model_config = ConfigDict(frozen=True)

    source_id: str = "mca_company_master"
    source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = MCA_CANONICAL_SCHEMA_VERSION
    snapshot_as_of: date
    acquisition_complete: bool
    raw_page_count: int = Field(ge=0)
    raw_record_count: int = Field(ge=0)
    company_snapshot_rows: int = Field(ge=0)
    address_snapshot_rows: int = Field(ge=0)
    duplicate_cin_rows: int = Field(ge=0)
    invalid_cin_rows: int = Field(ge=0)
    missing_registration_date_rows: int = Field(ge=0)
    blank_address_rows: int = Field(ge=0)
    quality_status: str
    staged_parts: tuple[CanonicalPartArtifact, ...]
    company_parts: tuple[CanonicalPartArtifact, ...]
    address_parts: tuple[CanonicalPartArtifact, ...]


def _sha256_file(artifact_path: Path) -> str:
    merchant_risk_digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for artifact_chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            merchant_risk_digest.update(artifact_chunk)
    return merchant_risk_digest.hexdigest()


def _write_immutable_parquet(
    merchant_risk_frame: pl.DataFrame,
    artifact_path: Path,
    compression: str,
) -> CanonicalPartArtifact:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise McaCanonicalisationError(
            f"Canonical artifact already exists: {artifact_path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=artifact_path.parent,
            prefix=".assay-parquet-",
            suffix=".parquet",
            delete=False,
        ) as temporary_artifact:
            temporary_path = Path(temporary_artifact.name)
        merchant_risk_frame.write_parquet(
            temporary_path,
            compression=compression,
            statistics=True,
        )
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.link(temporary_path, artifact_path)
    except FileExistsError as error:
        raise McaCanonicalisationError(
            f"Canonical artifact publish collision: {artifact_path}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return CanonicalPartArtifact(
        path=str(artifact_path),
        rows=merchant_risk_frame.height,
        bytes=artifact_path.stat().st_size,
        sha256=_sha256_file(artifact_path),
    )


def _canonical_company_frame(
    staged_frame: pl.DataFrame,
    source_snapshot_id: str,
    snapshot_as_of: date,
) -> pl.DataFrame:
    merchant_entity_frame = staged_frame.select(
        pl.concat_str(
            [pl.lit(source_snapshot_id), pl.col("CIN")], separator=":"
        ).alias("company_snapshot_id"),
        pl.lit("mca_company_master").alias("source_id"),
        pl.col("CIN").str.strip_chars().str.to_uppercase().alias("source_record_id"),
        pl.col("source_file_sha256"),
        pl.lit(source_snapshot_id).alias("source_snapshot_id"),
        pl.lit(snapshot_as_of).cast(pl.Date).alias("snapshot_as_of"),
        pl.col("observed_at"),
        pl.lit(MCA_CANONICAL_SCHEMA_VERSION).alias("schema_version"),
        pl.col("CIN").str.strip_chars().str.to_uppercase().alias("cin"),
        pl.col("CompanyName").str.strip_chars().alias("company_name"),
        pl.col("CompanyStatus").str.strip_chars().alias("company_status"),
        pl.col("CompanyRegistrationdate_date")
        .str.to_date(format="%Y-%m-%d", strict=False)
        .alias("registration_date"),
        pl.col("CompanyStateCode").str.strip_chars().alias("state_code"),
        pl.col("CompanyROCcode").str.strip_chars().alias("roc_code"),
        pl.col("CompanyCategory").str.strip_chars().alias("company_category"),
        pl.col("CompanySubCategory")
        .str.strip_chars()
        .alias("company_subcategory"),
        pl.col("CompanyClass").str.strip_chars().alias("company_class"),
        pl.col("Listingstatus").str.strip_chars().alias("listing_status"),
        pl.col("CompanyIndian/Foreign Company")
        .str.strip_chars()
        .alias("company_origin"),
        pl.col("AuthorizedCapital")
        .cast(pl.Decimal(precision=38, scale=2), strict=False)
        .alias("authorised_capital_inr"),
        pl.col("PaidupCapital")
        .cast(pl.Decimal(precision=38, scale=2), strict=False)
        .alias("paid_up_capital_inr"),
        pl.col("nic_code").str.strip_chars().alias("nic_code"),
        pl.col("CompanyIndustrialClassification")
        .str.strip_chars()
        .alias("industrial_classification"),
        pl.col("CIN")
        .str.strip_chars()
        .str.to_uppercase()
        .str.contains(r"^[A-Z0-9]{21}$")
        .alias("cin_is_valid_format"),
        pl.lit(None, dtype=pl.Date).alias("valid_from"),
        pl.lit(None, dtype=pl.Date).alias("valid_to"),
    )
    return merchant_entity_frame


def _canonical_address_frame(
    staged_frame: pl.DataFrame,
    source_snapshot_id: str,
    snapshot_as_of: date,
) -> pl.DataFrame:
    address_normalized = (
        pl.col("Registered_Office_Address")
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    return staged_frame.select(
        pl.concat_str(
            [pl.lit(source_snapshot_id), pl.col("CIN")], separator=":"
        ).alias("company_address_snapshot_id"),
        pl.lit("mca_company_master").alias("source_id"),
        pl.col("CIN").str.strip_chars().str.to_uppercase().alias("source_record_id"),
        pl.col("source_file_sha256"),
        pl.lit(source_snapshot_id).alias("source_snapshot_id"),
        pl.lit(snapshot_as_of).cast(pl.Date).alias("snapshot_as_of"),
        pl.col("observed_at"),
        pl.lit(MCA_CANONICAL_SCHEMA_VERSION).alias("schema_version"),
        pl.col("CIN").str.strip_chars().str.to_uppercase().alias("cin"),
        pl.col("Registered_Office_Address").alias("registered_office_address_raw"),
        address_normalized.alias("address_normalized"),
        address_normalized.alias("address_group_key"),
        pl.lit("india_address_v1").alias("address_normalization_version"),
        pl.lit(None, dtype=pl.Date).alias("valid_from"),
        pl.lit(None, dtype=pl.Date).alias("valid_to"),
    )


class McaCanonicaliser:
    """Convert a complete MCA acquisition checkpoint into Parquet snapshots."""

    def __init__(self, config: McaCanonicalisationConfig) -> None:
        self._config = config

    def run(self) -> McaCanonicalisationReport:
        checkpoint_path = self._config.raw_directory / "checkpoint.json"
        if not checkpoint_path.is_file():
            raise McaCanonicalisationError("MCA checkpoint is missing.")
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint = McaAcquisitionCheckpoint.model_validate_json(checkpoint_bytes)
        acquisition_complete = (
            checkpoint.total_records is not None
            and checkpoint.next_offset >= checkpoint.total_records
        )
        if not acquisition_complete and not self._config.allow_incomplete_acquisition:
            raise McaCanonicalisationError(
                "MCA acquisition is incomplete; canonicalisation is blocked."
            )
        source_snapshot_id = hashlib.sha256(checkpoint_bytes).hexdigest()
        snapshot_directory_name = f"snapshot-{source_snapshot_id}"
        staged_directory = self._config.staged_root / snapshot_directory_name
        company_directory = (
            self._config.curated_company_root / snapshot_directory_name
        )
        address_directory = (
            self._config.curated_address_root / snapshot_directory_name
        )
        report_path = (
            self._config.generated_report_root
            / f"{snapshot_directory_name}.report.json"
        )
        for output_path in (
            staged_directory,
            company_directory,
            address_directory,
            report_path,
        ):
            if output_path.exists():
                raise McaCanonicalisationError(
                    f"Canonical snapshot output already exists: {output_path}"
                )

        staged_parts: list[CanonicalPartArtifact] = []
        company_parts: list[CanonicalPartArtifact] = []
        address_parts: list[CanonicalPartArtifact] = []
        expected_offset = 0

        for page_artifact in checkpoint.pages:
            if page_artifact.offset != expected_offset:
                raise McaCanonicalisationError(
                    "MCA checkpoint page offsets are not contiguous."
                )
            raw_page_path = self._config.raw_directory / Path(
                page_artifact.raw_path
            ).name
            if not raw_page_path.is_file():
                raise McaCanonicalisationError("MCA raw page is missing.")
            if _sha256_file(raw_page_path) != page_artifact.sha256:
                raise McaCanonicalisationError("MCA raw page checksum mismatch.")

            raw_page_payload = json.loads(raw_page_path.read_bytes())
            merchant_entity_records = raw_page_payload.get("records")
            if not isinstance(merchant_entity_records, list):
                raise McaCanonicalisationError("MCA raw page records are invalid.")
            if merchant_entity_records:
                source_fields = set(merchant_entity_records[0])
                missing_source_fields = MCA_CANONICAL_SOURCE_FIELDS - source_fields
                if missing_source_fields:
                    missing_field_list = ", ".join(sorted(missing_source_fields))
                    raise McaCanonicalisationError(
                        "MCA raw page is missing canonical fields: "
                        f"{missing_field_list}."
                    )
            staged_frame = pl.from_dicts(
                merchant_entity_records,
                infer_schema_length=None,
            ).with_columns(
                pl.int_range(
                    page_artifact.offset,
                    page_artifact.offset + len(merchant_entity_records),
                    eager=True,
                ).alias("source_row_number"),
                pl.lit(page_artifact.sha256).alias("source_file_sha256"),
                pl.lit(page_artifact.retrieved_at).alias("observed_at"),
            )
            company_frame = _canonical_company_frame(
                staged_frame,
                source_snapshot_id=source_snapshot_id,
                snapshot_as_of=self._config.snapshot_as_of,
            )
            address_frame = _canonical_address_frame(
                staged_frame,
                source_snapshot_id=source_snapshot_id,
                snapshot_as_of=self._config.snapshot_as_of,
            )
            part_name = f"part-offset-{page_artifact.offset:010d}.parquet"
            staged_parts.append(
                _write_immutable_parquet(
                    staged_frame,
                    staged_directory / part_name,
                    self._config.parquet_compression,
                )
            )
            company_parts.append(
                _write_immutable_parquet(
                    company_frame,
                    company_directory / part_name,
                    self._config.parquet_compression,
                )
            )
            address_parts.append(
                _write_immutable_parquet(
                    address_frame,
                    address_directory / part_name,
                    self._config.parquet_compression,
                )
            )
            expected_offset += page_artifact.record_count

        quality_metrics = self._audit_snapshot(company_directory, address_directory)
        quality_status = (
            "passed"
            if quality_metrics["duplicate_cin_rows"] == 0
            and quality_metrics["invalid_cin_rows"] == 0
            else "failed"
        )
        report = McaCanonicalisationReport(
            source_snapshot_id=source_snapshot_id,
            snapshot_as_of=self._config.snapshot_as_of,
            acquisition_complete=acquisition_complete,
            raw_page_count=len(checkpoint.pages),
            raw_record_count=checkpoint.next_offset,
            quality_status=quality_status,
            staged_parts=tuple(staged_parts),
            company_parts=tuple(company_parts),
            address_parts=tuple(address_parts),
            **quality_metrics,
        )
        self._write_report(report, report_path)
        return report

    @staticmethod
    def _audit_snapshot(
        company_directory: Path,
        address_directory: Path,
    ) -> dict[str, int]:
        company_glob = str(company_directory / "*.parquet")
        address_glob = str(address_directory / "*.parquet")
        with duckdb.connect() as merchant_risk_database:
            company_metrics = merchant_risk_database.execute(
                """
                SELECT
                    count(*) AS company_snapshot_rows,
                    count(*) - count(DISTINCT cin) AS duplicate_cin_rows,
                    count(*) FILTER (WHERE NOT cin_is_valid_format) AS invalid_cin_rows,
                    count(*) FILTER (WHERE registration_date IS NULL)
                        AS missing_registration_date_rows
                FROM read_parquet(?)
                """,
                [company_glob],
            ).fetchone()
            address_metrics = merchant_risk_database.execute(
                """
                SELECT
                    count(*) AS address_snapshot_rows,
                    count(*) FILTER (WHERE address_normalized = '') AS blank_address_rows
                FROM read_parquet(?)
                """,
                [address_glob],
            ).fetchone()
        if company_metrics is None or address_metrics is None:
            raise McaCanonicalisationError("Canonical quality audit returned no rows.")
        return {
            "company_snapshot_rows": int(company_metrics[0]),
            "duplicate_cin_rows": int(company_metrics[1]),
            "invalid_cin_rows": int(company_metrics[2]),
            "missing_registration_date_rows": int(company_metrics[3]),
            "address_snapshot_rows": int(address_metrics[0]),
            "blank_address_rows": int(address_metrics[1]),
        }

    @staticmethod
    def _write_report(
        report: McaCanonicalisationReport,
        report_path: Path,
    ) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = json.dumps(
            report.model_dump(mode="json"), indent=2, sort_keys=True
        ).encode("utf-8")
        with report_path.open("xb") as report_file:
            report_file.write(report_payload)
            report_file.flush()
            os.fsync(report_file.fileno())
