from __future__ import annotations

from datetime import date

import polars as pl

from assay.signals.observations import (
    SignalObservationConfig,
    build_signal_observations,
)


def test_signal_observations_exclude_prior_outcomes_and_preserve_as_of_cutoff() -> None:
    snapshot_id = "a" * 64
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": ["company-1", "company-2", "company-3"],
            "source_snapshot_id": [snapshot_id] * 3,
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
            "cin": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
                "U99999MH2021PTC999999",
            ],
            "company_name": ["A Limited", "B Limited", "C Limited"],
            "registration_date": [
                date(2020, 1, 1),
                date(2020, 1, 12),
                date(2021, 5, 1),
            ],
            "state_code": ["DL", "DL", "MH"],
            "nic_code": ["64990", "64990", "62010"],
            "paid_up_capital_inr": [100_000, 100_000, 50_000],
            "cin_is_valid_format": [True, True, True],
        }
    )
    address_frame = pl.DataFrame(
        {
            "company_address_snapshot_id": ["address-1", "address-2", "address-3"],
            "source_snapshot_id": [snapshot_id] * 3,
            "source_record_id": company_frame["cin"],
            "address_normalized": ["12 RISK ROAD", "12 RISK ROAD", "9 SAFE ROAD"],
            "address_group_key": ["12 RISK ROAD", "12 RISK ROAD", "9 SAFE ROAD"],
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
        }
    )
    adverse_frame = pl.DataFrame(
        {
            "adverse_event_id": ["prior-event", "window-event"],
            "event_date": [date(2023, 10, 1), date(2024, 4, 1)],
        }
    )
    decision_frame = pl.DataFrame(
        {
            "adverse_event_id": ["prior-event", "window-event"],
            "accepted_company_snapshot_id": ["company-1", "company-2"],
            "review_status": ["accepted", "accepted_reviewed"],
            "outcome_eligible": [True, True],
        }
    )
    config = SignalObservationConfig(
        run_id="b" * 64,
        feature_cutoff=date(2023, 11, 3),
        outcome_window_start=date(2023, 11, 4),
        outcome_window_end=date(2026, 8, 22),
    )

    observations = build_signal_observations(
        company_frame,
        address_frame,
        adverse_frame,
        decision_frame,
        config,
    )
    by_company = {
        row["company_snapshot_id"]: row for row in observations.to_dicts()
    }

    assert by_company["company-1"]["has_prior_adverse_outcome"] is True
    assert by_company["company-1"]["evaluation_eligible"] is False
    assert by_company["company-2"]["observed_adverse_outcome"] is True
    assert by_company["company-2"]["label_match_policy"] == (
        "accepted_unique_exact_cin_or_completed_human_review"
    )
    assert by_company["company-2"]["evaluation_eligible"] is True
    assert by_company["company-2"]["shared_address_company_count"] == 2
    assert by_company["company-2"]["shared_address_signal"] is True
    assert by_company["company-2"][
        "address_registration_month_cohort_company_count"
    ] == 2
    assert by_company["company-3"]["observed_adverse_outcome"] is False
