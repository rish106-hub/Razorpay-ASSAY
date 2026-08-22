"""Conservative MCA-to-NSE entity linkage with reviewable match decisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import duckdb
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

ENTITY_LINKAGE_SCHEMA_VERSION = "1.0.0"
ENTITY_LINKAGE_POLICY_VERSION = "cin_then_exact_name_v1"


class EntityLinkageError(RuntimeError):
    """A deterministic merchant entity-linkage failure."""


class EntityLinkageConfig(BaseModel):
    """Explicit snapshot inputs and immutable linkage output paths."""

    model_config = ConfigDict(frozen=True)

    mca_report_path: Path
    nse_report_path: Path
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/entity_match")
    generated_report_root: Path = Path("data/generated/entity_linkage")
    max_persisted_candidates_per_event: int = Field(default=25, ge=1, le=100)
    parquet_compression: str = "zstd"


class EntityLinkageReport(BaseModel):
    """Coverage, ambiguity, and eligibility evidence for one linkage run."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = ENTITY_LINKAGE_SCHEMA_VERSION
    policy_version: str = ENTITY_LINKAGE_POLICY_VERSION
    mca_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    nse_source_snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    nse_provenance_complete_assets: int = Field(ge=0)
    adverse_event_rows: int = Field(ge=0)
    accepted_exact_cin_rows: int = Field(ge=0)
    pending_exact_name_review_rows: int = Field(ge=0)
    ambiguous_rows: int = Field(ge=0)
    unmatched_rows: int = Field(ge=0)
    identifier_collision_rows: int = Field(ge=0)
    persisted_candidate_rows: int = Field(ge=0)
    truncated_candidate_event_rows: int = Field(ge=0)
    outcome_eligible_rows: int = Field(ge=0)
    candidate_artifact: ParquetArtifact
    decision_artifact: ParquetArtifact


def _strict_name_expression(source_column: str) -> pl.Expr:
    return (
        pl.col(source_column)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _legal_name_expression(source_column: str) -> pl.Expr:
    return (
        _strict_name_expression(source_column)
        .str.replace_all(r"\b(PRIVATE|PVT|LIMITED|LTD|LLP|COMPANY|CO)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _candidate_frame(
    adverse_frame: pl.DataFrame,
    company_frame: pl.DataFrame,
    *,
    adverse_key: str,
    company_key: str,
    match_method: str,
    match_score: float,
    run_id: str,
    max_persisted_candidates_per_event: int,
) -> pl.DataFrame:
    joined_frame = adverse_frame.join(
        company_frame,
        left_on=adverse_key,
        right_on=company_key,
        how="inner",
    ).filter(pl.col(adverse_key).fill_null("") != "")
    if joined_frame.is_empty():
        return _empty_candidate_frame()

    ranked_frame = joined_frame.with_columns(
        pl.len().over("adverse_event_id").alias("candidate_count"),
        pl.col("company_snapshot_id")
        .rank(method="ordinal")
        .over("adverse_event_id")
        .cast(pl.Int32)
        .alias("candidate_rank"),
    ).filter(pl.col("candidate_rank") <= max_persisted_candidates_per_event)
    return ranked_frame.select(
        pl.concat_str(
            [
                pl.lit(run_id),
                pl.col("adverse_event_id"),
                pl.col("company_snapshot_id"),
                pl.lit(match_method),
            ],
            separator=":",
        ).alias("entity_match_candidate_id"),
        pl.lit(run_id).alias("run_id"),
        pl.col("adverse_event_id"),
        pl.col("company_snapshot_id"),
        pl.col("company_cin"),
        pl.col("company_name"),
        pl.lit(match_method).alias("match_method"),
        pl.lit(match_score).cast(pl.Float64).alias("match_score"),
        pl.col("candidate_rank"),
        pl.col("candidate_count").cast(pl.Int32),
        (pl.col("candidate_count") > max_persisted_candidates_per_event).alias(
            "candidate_set_truncated"
        ),
    )


def _empty_candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entity_match_candidate_id": pl.String,
            "run_id": pl.String,
            "adverse_event_id": pl.String,
            "company_snapshot_id": pl.String,
            "company_cin": pl.String,
            "company_name": pl.String,
            "match_method": pl.String,
            "match_score": pl.Float64,
            "candidate_rank": pl.Int32,
            "candidate_count": pl.Int32,
            "candidate_set_truncated": pl.Boolean,
        }
    )


def _prefilter_linkage_companies(
    company_parquet_paths: list[str],
    adverse_frame: pl.DataFrame,
) -> pl.DataFrame:
    """Use an out-of-core semi-join before building the in-memory match registry."""

    strict_company_name_sql = """
        trim(
            regexp_replace(
                regexp_replace(
                    upper(coalesce(company_name, '')),
                    '[^A-Z0-9]+',
                    ' ',
                    'g'
                ),
                '\\s+',
                ' ',
                'g'
            )
        )
    """
    legal_company_name_sql = f"""
        trim(
            regexp_replace(
                regexp_replace(
                    {strict_company_name_sql},
                    '\\b(PRIVATE|PVT|LIMITED|LTD|LLP|COMPANY|CO)\\b',
                    ' ',
                    'g'
                ),
                '\\s+',
                ' ',
                'g'
            )
        )
    """
    query = f"""
        WITH merchant_registry AS (
            SELECT
                company_snapshot_id,
                upper(coalesce(cin, '')) AS cin,
                company_name,
                {strict_company_name_sql} AS company_name_strict,
                {legal_company_name_sql} AS company_name_legal
            FROM read_parquet(?)
        ),
        matched_companies AS (
            SELECT merchant_registry.company_snapshot_id
            FROM merchant_registry
            SEMI JOIN adverse_registry
                ON merchant_registry.cin = adverse_registry.cin_candidate
            WHERE merchant_registry.cin <> ''

            UNION

            SELECT merchant_registry.company_snapshot_id
            FROM merchant_registry
            SEMI JOIN adverse_registry
                ON merchant_registry.company_name_strict =
                    adverse_registry.entity_name_normalized_strict
            WHERE merchant_registry.company_name_strict <> ''

            UNION

            SELECT merchant_registry.company_snapshot_id
            FROM merchant_registry
            SEMI JOIN adverse_registry
                ON merchant_registry.company_name_legal =
                    adverse_registry.entity_name_normalized_legal
            WHERE merchant_registry.company_name_legal <> ''
        )
        SELECT
            merchant_registry.company_snapshot_id,
            nullif(merchant_registry.cin, '') AS cin,
            merchant_registry.company_name
        FROM merchant_registry
        INNER JOIN matched_companies USING (company_snapshot_id)
        ORDER BY merchant_registry.company_snapshot_id
    """
    with (
        tempfile.TemporaryDirectory(prefix="assay-linkage-duckdb-") as spill_path,
        duckdb.connect(
            config={
                "memory_limit": "4GB",
                "temp_directory": spill_path,
                "threads": "4",
            }
        ) as merchant_risk_database,
    ):
        merchant_risk_database.register("adverse_registry", adverse_frame)
        return merchant_risk_database.execute(
            query,
            [company_parquet_paths],
        ).pl()


def build_entity_linkage_frames(
    company_frame: pl.DataFrame,
    adverse_frame: pl.DataFrame,
    *,
    run_id: str,
    max_persisted_candidates_per_event: int = 25,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build candidates and one deterministic decision per adverse event."""

    required_company_columns = {
        "company_snapshot_id",
        "cin",
        "company_name",
    }
    required_adverse_columns = {
        "adverse_event_id",
        "cin_candidate",
        "entity_name_normalized_strict",
        "entity_name_normalized_legal",
    }
    if missing := required_company_columns - set(company_frame.columns):
        raise EntityLinkageError(
            f"Company snapshot is missing columns: {', '.join(sorted(missing))}."
        )
    if missing := required_adverse_columns - set(adverse_frame.columns):
        raise EntityLinkageError(
            f"Adverse-event snapshot is missing columns: {', '.join(sorted(missing))}."
        )
    if adverse_frame["adverse_event_id"].n_unique() != adverse_frame.height:
        raise EntityLinkageError("Adverse-event IDs must be unique before linkage.")

    merchant_registry = company_frame.select(
        "company_snapshot_id",
        pl.col("cin").fill_null("").str.to_uppercase().alias("cin"),
        pl.col("cin").fill_null("").str.to_uppercase().alias("company_cin"),
        "company_name",
        _strict_name_expression("company_name").alias("company_name_strict"),
        _legal_name_expression("company_name").alias("company_name_legal"),
    )
    adverse_registry = adverse_frame.select(
        "adverse_event_id",
        pl.col("cin_candidate").fill_null("").alias("cin_candidate"),
        "entity_name_normalized_strict",
        "entity_name_normalized_legal",
    )

    exact_cin_candidates = _candidate_frame(
        adverse_registry,
        merchant_registry,
        adverse_key="cin_candidate",
        company_key="cin",
        match_method="exact_cin",
        match_score=1.0,
        run_id=run_id,
        max_persisted_candidates_per_event=max_persisted_candidates_per_event,
    )
    cin_candidate_event_ids = exact_cin_candidates.select("adverse_event_id").unique()
    name_eligible_events = adverse_registry.join(
        cin_candidate_event_ids,
        on="adverse_event_id",
        how="anti",
    )
    exact_name_candidates = _candidate_frame(
        name_eligible_events,
        merchant_registry,
        adverse_key="entity_name_normalized_strict",
        company_key="company_name_strict",
        match_method="exact_strict_name",
        match_score=0.95,
        run_id=run_id,
        max_persisted_candidates_per_event=max_persisted_candidates_per_event,
    )
    strict_candidate_event_ids = exact_name_candidates.select(
        "adverse_event_id"
    ).unique()
    legal_name_eligible_events = name_eligible_events.join(
        strict_candidate_event_ids,
        on="adverse_event_id",
        how="anti",
    )
    exact_legal_name_candidates = _candidate_frame(
        legal_name_eligible_events,
        merchant_registry,
        adverse_key="entity_name_normalized_legal",
        company_key="company_name_legal",
        match_method="exact_legal_name",
        match_score=0.90,
        run_id=run_id,
        max_persisted_candidates_per_event=max_persisted_candidates_per_event,
    )
    candidate_frame = pl.concat(
        [
            exact_cin_candidates,
            exact_name_candidates,
            exact_legal_name_candidates,
        ],
        how="vertical",
    ).sort(["adverse_event_id", "candidate_rank"])

    candidate_summary = candidate_frame.filter(pl.col("candidate_rank") == 1).select(
        "adverse_event_id",
        "match_method",
        "candidate_count",
        "candidate_set_truncated",
        pl.when(pl.col("candidate_count") == 1)
        .then(pl.col("company_snapshot_id"))
        .otherwise(None)
        .alias("proposed_company_snapshot_id"),
        pl.when(
            (pl.col("match_method") == "exact_cin")
            & (pl.col("candidate_count") == 1)
        )
        .then(pl.col("company_snapshot_id"))
        .otherwise(None)
        .alias("accepted_company_snapshot_id"),
    )
    decision_frame = adverse_registry.select("adverse_event_id").join(
        candidate_summary,
        on="adverse_event_id",
        how="left",
    ).with_columns(
        pl.lit(run_id).alias("run_id"),
        pl.when(pl.col("match_method").is_null())
        .then(pl.lit("unmatched"))
        .when(pl.col("candidate_count") > 1)
        .then(pl.lit("ambiguous"))
        .when(pl.col("match_method") == "exact_cin")
        .then(pl.lit("accepted"))
        .otherwise(pl.lit("pending_review"))
        .alias("review_status"),
        (
            (pl.col("match_method") == "exact_cin")
            & (pl.col("candidate_count") == 1)
        )
        .fill_null(False)
        .alias("outcome_eligible"),
    ).select(
        pl.concat_str(
            [pl.lit(run_id), pl.col("adverse_event_id")], separator=":"
        ).alias("entity_match_decision_id"),
        "run_id",
        "adverse_event_id",
        pl.col("match_method").fill_null("none"),
        pl.col("candidate_count").fill_null(0).cast(pl.Int32),
        pl.col("candidate_set_truncated").fill_null(False),
        "proposed_company_snapshot_id",
        "accepted_company_snapshot_id",
        "review_status",
        "outcome_eligible",
        pl.lit(ENTITY_LINKAGE_SCHEMA_VERSION).alias("schema_version"),
        pl.lit(ENTITY_LINKAGE_POLICY_VERSION).alias("policy_version"),
    ).sort("adverse_event_id")
    return candidate_frame, decision_frame


class EntityLinker:
    """Link explicit MCA and NSE snapshots and publish immutable audit artifacts."""

    def __init__(self, config: EntityLinkageConfig) -> None:
        self._config = config

    def run(self) -> EntityLinkageReport:
        mca_report_path = self._resolve(self._config.mca_report_path)
        nse_report_path = self._resolve(self._config.nse_report_path)
        mca_report = McaCanonicalisationReport.model_validate_json(
            mca_report_path.read_text(encoding="utf-8")
        )
        nse_report = NseCanonicalisationReport.model_validate_json(
            nse_report_path.read_text(encoding="utf-8")
        )
        accepted_mca_quality_statuses = {"passed", "passed_with_quarantine"}
        if (
            not mca_report.acquisition_complete
            or mca_report.quality_status not in accepted_mca_quality_statuses
        ):
            raise EntityLinkageError(
                "MCA linkage requires a complete, quality-passed canonical snapshot."
            )

        run_payload = json.dumps(
            {
                "mca_source_snapshot_id": mca_report.source_snapshot_id,
                "nse_source_snapshot_id": nse_report.source_snapshot_id,
                "policy_version": ENTITY_LINKAGE_POLICY_VERSION,
                "max_persisted_candidates_per_event": (
                    self._config.max_persisted_candidates_per_event
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        run_id = hashlib.sha256(run_payload).hexdigest()
        run_directory = self._resolve(self._config.curated_root) / f"run-{run_id}"
        report_path = self._resolve(self._config.generated_report_root) / (
            f"run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise EntityLinkageError("Entity-linkage run output already exists.")

        company_parquet_paths = [
            str(verify_parquet_artifact(part, self._config.project_root))
            for part in mca_report.company_parts
        ]
        adverse_frame = pl.concat(
            [
                pl.scan_parquet(
                    verify_parquet_artifact(part, self._config.project_root)
                ).select(
                    "adverse_event_id",
                    "cin_candidate",
                    "entity_name_normalized_strict",
                    "entity_name_normalized_legal",
                )
                for part in nse_report.parts
            ],
            how="vertical",
        ).collect(engine="streaming")
        company_frame = _prefilter_linkage_companies(
            company_parquet_paths,
            adverse_frame,
        )
        candidate_frame, decision_frame = build_entity_linkage_frames(
            company_frame,
            adverse_frame,
            run_id=run_id,
            max_persisted_candidates_per_event=(
                self._config.max_persisted_candidates_per_event
            ),
        )

        run_directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=run_directory.parent,
            prefix=".assay-linkage-",
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            temporary_candidate_artifact = write_immutable_parquet(
                candidate_frame,
                temporary_directory / "entity_match_candidate.parquet",
                self._config.parquet_compression,
            )
            temporary_decision_artifact = write_immutable_parquet(
                decision_frame,
                temporary_directory / "entity_match_decision.parquet",
                self._config.parquet_compression,
            )
            os.rename(temporary_directory, run_directory)

        candidate_artifact = self._published_artifact(
            temporary_candidate_artifact,
            run_directory / "entity_match_candidate.parquet",
        )
        decision_artifact = self._published_artifact(
            temporary_decision_artifact,
            run_directory / "entity_match_decision.parquet",
        )
        report = EntityLinkageReport(
            run_id=run_id,
            mca_source_snapshot_id=mca_report.source_snapshot_id,
            nse_source_snapshot_id=nse_report.source_snapshot_id,
            nse_provenance_complete_assets=nse_report.provenance_complete_assets,
            adverse_event_rows=decision_frame.height,
            accepted_exact_cin_rows=decision_frame.filter(
                pl.col("review_status") == "accepted"
            ).height,
            pending_exact_name_review_rows=decision_frame.filter(
                pl.col("review_status") == "pending_review"
            ).height,
            ambiguous_rows=decision_frame.filter(
                pl.col("review_status") == "ambiguous"
            ).height,
            unmatched_rows=decision_frame.filter(
                pl.col("review_status") == "unmatched"
            ).height,
            identifier_collision_rows=decision_frame.filter(
                (pl.col("match_method") == "exact_cin")
                & (pl.col("candidate_count") > 1)
            ).height,
            persisted_candidate_rows=candidate_frame.height,
            truncated_candidate_event_rows=decision_frame.filter(
                pl.col("candidate_set_truncated")
            ).height,
            outcome_eligible_rows=decision_frame.filter(
                pl.col("outcome_eligible")
            ).height,
            candidate_artifact=candidate_artifact,
            decision_artifact=decision_artifact,
        )
        self._write_report(report_path, report)
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _published_artifact(
        temporary_artifact: ParquetArtifact,
        published_path: Path,
    ) -> ParquetArtifact:
        return ParquetArtifact(
            path=str(published_path),
            rows=temporary_artifact.rows,
            bytes=published_path.stat().st_size,
            sha256=sha256_file(published_path),
        )

    @staticmethod
    def _write_report(report_path: Path, report: EntityLinkageReport) -> None:
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
