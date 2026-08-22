from __future__ import annotations

import polars as pl
import pytest

from assay.linkage.review_ingestion import (
    LinkageReviewIngestionError,
    validate_and_apply_linkage_reviews,
)


def _original_review_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "linkage_review_row_id": ["review-1", "review-2"],
            "adverse_event_id": ["event-1", "event-1"],
            "review_stratum": ["ambiguous_review", "ambiguous_review"],
            "company_snapshot_id": ["company-1", "company-2"],
            "match_method": ["exact_legal_name", "exact_legal_name"],
            "candidate_set_truncated": [False, False],
            "review_decision": ["", ""],
            "reviewer_id": ["", ""],
            "reviewed_at": ["", ""],
            "reviewer_notes": ["", ""],
        }
    )


def _decision_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "adverse_event_id": ["event-1"],
            "match_method": ["exact_legal_name"],
            "candidate_count": [2],
            "proposed_company_snapshot_id": [None],
            "accepted_company_snapshot_id": [None],
            "review_status": ["ambiguous"],
            "outcome_eligible": [False],
        }
    )


def test_completed_review_accepts_one_match_only_after_other_candidates_rejected() -> None:
    completed_frame = _original_review_frame().with_columns(
        pl.Series("review_decision", ["MATCH", "NO_MATCH"]),
        pl.Series("reviewer_id", ["risk-ops-1", "risk-ops-1"]),
        pl.Series(
            "reviewed_at",
            ["2026-08-22T20:00:00+05:30", "2026-08-22T20:01:00+05:30"],
        ),
    )

    revised_decisions, summary = validate_and_apply_linkage_reviews(
        _original_review_frame(),
        completed_frame,
        _decision_frame(),
    )

    assert summary.accepted_reviewed_event_rows == 1
    assert revised_decisions["review_status"].item() == "accepted_reviewed"
    assert revised_decisions["accepted_company_snapshot_id"].item() == "company-1"
    assert revised_decisions["outcome_eligible"].item() is True


def test_completed_review_rejects_protected_column_edits() -> None:
    completed_frame = _original_review_frame().with_columns(
        pl.when(pl.col("linkage_review_row_id") == "review-1")
        .then(pl.lit("different-company"))
        .otherwise(pl.col("company_snapshot_id"))
        .alias("company_snapshot_id"),
        pl.lit("NO_MATCH").alias("review_decision"),
        pl.lit("risk-ops-1").alias("reviewer_id"),
        pl.lit("2026-08-22T20:00:00+05:30").alias("reviewed_at"),
    )

    with pytest.raises(LinkageReviewIngestionError, match="Protected review column"):
        validate_and_apply_linkage_reviews(
            _original_review_frame(),
            completed_frame,
            _decision_frame(),
        )
