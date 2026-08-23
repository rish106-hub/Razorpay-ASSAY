from __future__ import annotations

from datetime import date

import polars as pl

from assay.linkage.review import (
    LinkageReviewSamplingConfig,
    build_linkage_review_sample,
)


def test_review_sample_keeps_ambiguous_candidate_sets_together() -> None:
    candidate_frame = pl.DataFrame(
        {
            "entity_match_candidate_id": ["candidate-1", "candidate-2", "candidate-3"],
            "run_id": ["a" * 64] * 3,
            "adverse_event_id": ["event-exact", "event-ambiguous", "event-ambiguous"],
            "company_snapshot_id": ["company-1", "company-2", "company-3"],
            "company_cin": ["cin-1", "cin-2", "cin-3"],
            "company_name": ["Exact Limited", "Shared Limited", "Shared Pvt Ltd"],
            "match_method": ["exact_strict_name", "exact_legal_name", "exact_legal_name"],
            "match_score": [0.95, 0.9, 0.9],
            "candidate_rank": [1, 1, 2],
            "candidate_count": [1, 2, 2],
            "candidate_set_truncated": [False, False, False],
        }
    )
    decision_frame = pl.DataFrame(
        {
            "adverse_event_id": ["event-exact", "event-ambiguous"],
            "review_status": ["pending_review", "ambiguous"],
            "outcome_eligible": [False, False],
        }
    )
    adverse_frame = pl.DataFrame(
        {
            "adverse_event_id": ["event-exact", "event-ambiguous"],
            "event_source_authority": ["SEBI", "SEBI"],
            "event_date": [date(2024, 1, 1), date(2024, 2, 1)],
            "order_particulars": ["Order 1", "Order 2"],
            "entity_name_raw": ["Exact Limited", "Shared"],
            "entity_name_normalized_strict": ["EXACT LIMITED", "SHARED"],
            "entity_name_normalized_legal": ["EXACT", "SHARED"],
            "pan_raw": ["", ""],
            "din_cin_raw": ["", ""],
            "cin_candidate": [None, None],
        }
    )

    review_frame = build_linkage_review_sample(
        candidate_frame,
        decision_frame,
        adverse_frame,
        LinkageReviewSamplingConfig(
            identifier_audit_events=0,
            strict_name_review_events=1,
            legal_name_review_events=0,
            ambiguous_review_events=1,
        ),
    )

    assert review_frame.filter(pl.col("adverse_event_id") == "event-exact").height == 1
    assert review_frame.filter(
        pl.col("adverse_event_id") == "event-ambiguous"
    ).height == 2
    assert review_frame["review_decision"].unique().to_list() == [""]
