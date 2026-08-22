"""Deterministic human-review samples for MCA-to-NSE entity candidates."""

from __future__ import annotations

import hashlib

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

LINKAGE_REVIEW_SCHEMA_VERSION = "1.0.0"


class LinkageReviewError(RuntimeError):
    """A linkage review sample cannot satisfy its audit contract."""


class LinkageReviewSamplingConfig(BaseModel):
    """Per-stratum event limits for deterministic reviewer workload."""

    model_config = ConfigDict(frozen=True)

    identifier_audit_events: int = Field(default=100, ge=0, le=1_000)
    strict_name_review_events: int = Field(default=200, ge=0, le=5_000)
    legal_name_review_events: int = Field(default=200, ge=0, le=5_000)
    ambiguous_review_events: int = Field(default=100, ge=0, le=1_000)
    maximum_candidates_per_ambiguous_event: int = Field(default=5, ge=2, le=25)
    sampling_seed: int = Field(default=106, ge=0)


def _sampling_hash(event_id: str, seed: int) -> str:
    sampling_payload = f"{seed}:{event_id}".encode()
    return hashlib.sha256(sampling_payload).hexdigest()


def _require_columns(
    frame: pl.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    if missing_columns := required_columns - set(frame.columns):
        raise LinkageReviewError(
            f"{frame_name} is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )


def build_linkage_review_sample(
    candidate_frame: pl.DataFrame,
    decision_frame: pl.DataFrame,
    adverse_frame: pl.DataFrame,
    config: LinkageReviewSamplingConfig,
) -> pl.DataFrame:
    """Sample review events by match method without outcome-based selection."""

    _require_columns(
        candidate_frame,
        {
            "entity_match_candidate_id",
            "run_id",
            "adverse_event_id",
            "company_snapshot_id",
            "company_cin",
            "company_name",
            "match_method",
            "match_score",
            "candidate_rank",
            "candidate_count",
            "candidate_set_truncated",
        },
        "entity-match candidates",
    )
    _require_columns(
        decision_frame,
        {
            "adverse_event_id",
            "review_status",
            "outcome_eligible",
        },
        "entity-match decisions",
    )
    _require_columns(
        adverse_frame,
        {
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
        },
        "adverse events",
    )
    if decision_frame["adverse_event_id"].n_unique() != decision_frame.height:
        raise LinkageReviewError("Entity-match decisions must be unique by event.")

    event_candidates = candidate_frame.filter(
        pl.col("candidate_rank") == 1
    ).select(
        "adverse_event_id",
        "match_method",
        "candidate_count",
    ).join(
        decision_frame.select(
            "adverse_event_id",
            "review_status",
            "outcome_eligible",
        ),
        on="adverse_event_id",
        how="inner",
        validate="1:1",
    ).with_columns(
        pl.when(
            (pl.col("match_method") == "exact_cin")
            & (pl.col("review_status") == "accepted")
        )
        .then(pl.lit("identifier_audit"))
        .when(
            (pl.col("match_method") == "exact_strict_name")
            & (pl.col("review_status") == "pending_review")
        )
        .then(pl.lit("strict_name_review"))
        .when(
            (pl.col("match_method") == "exact_legal_name")
            & (pl.col("review_status") == "pending_review")
        )
        .then(pl.lit("legal_name_review"))
        .when(pl.col("review_status") == "ambiguous")
        .then(pl.lit("ambiguous_review"))
        .otherwise(None)
        .alias("review_stratum")
    ).filter(pl.col("review_stratum").is_not_null())
    if event_candidates.is_empty():
        raise LinkageReviewError("No linkage candidates are eligible for review.")

    event_hash_frame = pl.DataFrame(
        {
            "adverse_event_id": event_candidates["adverse_event_id"],
            "sampling_hash": [
                _sampling_hash(event_id, config.sampling_seed)
                for event_id in event_candidates["adverse_event_id"]
            ],
        }
    )
    event_candidates = event_candidates.join(
        event_hash_frame,
        on="adverse_event_id",
        how="left",
        validate="1:1",
    )
    stratum_limits = {
        "identifier_audit": config.identifier_audit_events,
        "strict_name_review": config.strict_name_review_events,
        "legal_name_review": config.legal_name_review_events,
        "ambiguous_review": config.ambiguous_review_events,
    }
    sampled_events = pl.concat(
        [
            event_candidates.filter(pl.col("review_stratum") == stratum)
            .sort("sampling_hash")
            .head(event_limit)
            for stratum, event_limit in stratum_limits.items()
            if event_limit > 0
        ],
        how="vertical",
    )
    sampled_candidates = candidate_frame.join(
        sampled_events.select(
            "adverse_event_id",
            "review_stratum",
            "sampling_hash",
            "review_status",
            "outcome_eligible",
        ),
        on="adverse_event_id",
        how="inner",
        validate="m:1",
    ).filter(
        (pl.col("review_stratum") != "ambiguous_review")
        | (
            pl.col("candidate_rank")
            <= config.maximum_candidates_per_ambiguous_event
        )
    ).join(
        adverse_frame.select(
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
        ),
        on="adverse_event_id",
        how="left",
        validate="m:1",
    )
    return sampled_candidates.select(
        pl.concat_str(
            [
                pl.col("entity_match_candidate_id"),
                pl.lit(LINKAGE_REVIEW_SCHEMA_VERSION),
            ],
            separator=":",
        ).alias("linkage_review_row_id"),
        "entity_match_candidate_id",
        "run_id",
        "review_stratum",
        "sampling_hash",
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
        "company_snapshot_id",
        "company_cin",
        "company_name",
        "match_method",
        "match_score",
        "candidate_rank",
        "candidate_count",
        "candidate_set_truncated",
        pl.col("review_status").alias("pipeline_review_status"),
        pl.col("outcome_eligible").alias("pipeline_outcome_eligible"),
        pl.lit("").alias("review_decision"),
        pl.lit("").alias("reviewer_id"),
        pl.lit("").alias("reviewed_at"),
        pl.lit("").alias("reviewer_notes"),
        pl.lit(LINKAGE_REVIEW_SCHEMA_VERSION).alias("review_schema_version"),
    ).sort(["review_stratum", "sampling_hash", "candidate_rank"])
