"""Validate completed linkage reviews and apply them to match decisions."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.linkage.review import LINKAGE_REVIEW_SCHEMA_VERSION

ALLOWED_REVIEW_DECISIONS = frozenset({"MATCH", "NO_MATCH", "UNSURE"})
REVIEWER_COLUMNS = frozenset(
    {"review_decision", "reviewer_id", "reviewed_at", "reviewer_notes"}
)


class LinkageReviewIngestionError(RuntimeError):
    """A completed review workbook is invalid or was altered outside review fields."""


class LinkageReviewApplicationSummary(BaseModel):
    """Outcome counts after applying one complete reviewer workbook."""

    model_config = ConfigDict(frozen=True)

    reviewed_event_rows: int = Field(ge=0)
    accepted_reviewed_event_rows: int = Field(ge=0)
    rejected_reviewed_event_rows: int = Field(ge=0)
    unresolved_review_event_rows: int = Field(ge=0)
    identifier_audit_discrepancy_rows: int = Field(ge=0)


def _canonical_cell(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    if isinstance(raw_value, (date, datetime)):
        return raw_value.isoformat()
    return str(raw_value)


def _spreadsheet_cell(raw_value: Any) -> str:
    canonical_value = _canonical_cell(raw_value)
    if canonical_value.startswith(("=", "+", "-", "@")):
        return f"'{canonical_value}"
    return canonical_value


def _parse_reviewed_at(raw_value: str) -> datetime:
    try:
        reviewed_at = datetime.fromisoformat(raw_value)
    except ValueError as error:
        raise LinkageReviewIngestionError(
            "reviewed_at must use ISO 8601 format."
        ) from error
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise LinkageReviewIngestionError(
            "reviewed_at must include an explicit timezone offset."
        )
    return reviewed_at


def validate_and_apply_linkage_reviews(
    original_review_frame: pl.DataFrame,
    completed_review_frame: pl.DataFrame,
    decision_frame: pl.DataFrame,
) -> tuple[pl.DataFrame, LinkageReviewApplicationSummary]:
    """Reject workbook tampering and apply only fully resolved candidate sets."""

    if "linkage_review_row_id" not in original_review_frame.columns:
        raise LinkageReviewIngestionError(
            "Original review artifact is missing linkage_review_row_id."
        )
    if set(completed_review_frame.columns) != set(original_review_frame.columns):
        raise LinkageReviewIngestionError(
            "Completed workbook columns must exactly match the original workbook."
        )
    if completed_review_frame.height != original_review_frame.height:
        raise LinkageReviewIngestionError(
            "Completed workbook row count does not match the original workbook."
        )
    original_rows = {
        row["linkage_review_row_id"]: row
        for row in original_review_frame.to_dicts()
    }
    completed_rows = {
        row["linkage_review_row_id"]: row
        for row in completed_review_frame.to_dicts()
    }
    if len(original_rows) != original_review_frame.height:
        raise LinkageReviewIngestionError("Original review row IDs must be unique.")
    if len(completed_rows) != completed_review_frame.height:
        raise LinkageReviewIngestionError("Completed review row IDs must be unique.")
    if set(completed_rows) != set(original_rows):
        raise LinkageReviewIngestionError(
            "Completed workbook review row IDs do not match the original workbook."
        )

    protected_columns = set(original_review_frame.columns) - REVIEWER_COLUMNS
    reviewed_rows_by_event: dict[str, list[dict[str, Any]]] = {}
    for review_row_id, original_row in original_rows.items():
        completed_row = completed_rows[review_row_id]
        for protected_column in protected_columns:
            if _spreadsheet_cell(original_row[protected_column]) != _canonical_cell(
                completed_row[protected_column]
            ):
                raise LinkageReviewIngestionError(
                    f"Protected review column was modified: {protected_column}."
                )
        review_decision = _canonical_cell(
            completed_row["review_decision"]
        ).strip().upper()
        reviewer_id = _canonical_cell(completed_row["reviewer_id"]).strip()
        reviewed_at_raw = _canonical_cell(completed_row["reviewed_at"]).strip()
        if review_decision not in ALLOWED_REVIEW_DECISIONS:
            raise LinkageReviewIngestionError(
                "review_decision must be MATCH, NO_MATCH, or UNSURE."
            )
        if not reviewer_id:
            raise LinkageReviewIngestionError("reviewer_id is required for every row.")
        _parse_reviewed_at(reviewed_at_raw)
        reviewed_row = dict(original_row)
        reviewed_row.update(
            {
                "review_decision": review_decision,
                "reviewer_id": reviewer_id,
                "reviewed_at": reviewed_at_raw,
                "reviewer_notes": _canonical_cell(
                    completed_row["reviewer_notes"]
                ).strip(),
            }
        )
        reviewed_rows_by_event.setdefault(
            reviewed_row["adverse_event_id"], []
        ).append(reviewed_row)

    decision_rows = {
        row["adverse_event_id"]: row for row in decision_frame.to_dicts()
    }
    if len(decision_rows) != decision_frame.height:
        raise LinkageReviewIngestionError(
            "Entity-match decisions must be unique by adverse event."
        )
    accepted_reviewed_event_rows = 0
    rejected_reviewed_event_rows = 0
    unresolved_review_event_rows = 0
    identifier_audit_discrepancy_rows = 0
    for adverse_event_id, reviewed_rows in reviewed_rows_by_event.items():
        if adverse_event_id not in decision_rows:
            raise LinkageReviewIngestionError(
                "Review workbook references an unknown adverse event."
            )
        decision_row = decision_rows[adverse_event_id]
        matched_rows = [
            row for row in reviewed_rows if row["review_decision"] == "MATCH"
        ]
        all_other_rows_rejected = all(
            row["review_decision"] == "NO_MATCH"
            for row in reviewed_rows
            if row not in matched_rows
        )
        candidate_set_truncated = any(
            bool(row["candidate_set_truncated"]) for row in reviewed_rows
        )
        review_stratum = reviewed_rows[0]["review_stratum"]
        if (
            len(matched_rows) == 1
            and all_other_rows_rejected
            and not candidate_set_truncated
        ):
            matched_row = matched_rows[0]
            decision_row.update(
                {
                    "match_method": f"reviewed_{matched_row['match_method']}",
                    "candidate_count": len(reviewed_rows),
                    "proposed_company_snapshot_id": matched_row[
                        "company_snapshot_id"
                    ],
                    "accepted_company_snapshot_id": matched_row[
                        "company_snapshot_id"
                    ],
                    "review_status": "accepted_reviewed",
                    "outcome_eligible": True,
                    "review_source_schema_version": (
                        LINKAGE_REVIEW_SCHEMA_VERSION
                    ),
                }
            )
            accepted_reviewed_event_rows += 1
        elif not matched_rows and all(
            row["review_decision"] == "NO_MATCH" for row in reviewed_rows
        ):
            decision_row.update(
                {
                    "accepted_company_snapshot_id": None,
                    "review_status": "rejected_reviewed",
                    "outcome_eligible": False,
                    "review_source_schema_version": (
                        LINKAGE_REVIEW_SCHEMA_VERSION
                    ),
                }
            )
            rejected_reviewed_event_rows += 1
            if review_stratum == "identifier_audit":
                identifier_audit_discrepancy_rows += 1
        else:
            decision_row.update(
                {
                    "accepted_company_snapshot_id": None,
                    "review_status": "unresolved_review",
                    "outcome_eligible": False,
                    "review_source_schema_version": (
                        LINKAGE_REVIEW_SCHEMA_VERSION
                    ),
                }
            )
            unresolved_review_event_rows += 1

    revised_decision_frame = pl.DataFrame(list(decision_rows.values())).sort(
        "adverse_event_id"
    )
    summary = LinkageReviewApplicationSummary(
        reviewed_event_rows=len(reviewed_rows_by_event),
        accepted_reviewed_event_rows=accepted_reviewed_event_rows,
        rejected_reviewed_event_rows=rejected_reviewed_event_rows,
        unresolved_review_event_rows=unresolved_review_event_rows,
        identifier_audit_discrepancy_rows=identifier_audit_discrepancy_rows,
    )
    return revised_decision_frame, summary
