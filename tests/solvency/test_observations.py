from __future__ import annotations

from datetime import date

import polars as pl

from assay.solvency.observations import (
    SolvencyObservationConfig,
    build_solvency_observations,
)


def test_solvency_observations_exclude_prior_cirp_and_freeze_features() -> None:
    snapshot_id = "a" * 64
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": ["company-1", "company-2", "company-3"],
            "source_snapshot_id": [snapshot_id] * 3,
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
            "legal_entity_identifier": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
                "U99999MH2021PTC999999",
            ],
            "legal_entity_identifier_type": ["CIN"] * 3,
            "legal_entity_identifier_is_valid_format": [True] * 3,
            "cin": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
                "U99999MH2021PTC999999",
            ],
            "company_name": ["A Limited", "B Limited", "C Limited"],
            "company_status": ["Active"] * 3,
            "registration_date": [
                date(2020, 1, 1),
                date(2020, 1, 12),
                date(2021, 5, 1),
            ],
            "state_code": ["delhi", "delhi", "maharashtra"],
            "roc_code": ["roc delhi", "roc delhi", "roc mumbai"],
            "company_category": ["company limited by shares"] * 3,
            "company_subcategory": ["non-government company"] * 3,
            "company_class": ["private", "private", "public"],
            "listing_status": ["unlisted", "unlisted", "listed"],
            "company_origin": ["indian"] * 3,
            "authorised_capital_inr": [200_000, 200_000, 100_000],
            "paid_up_capital_inr": [100_000, 100_000, 50_000],
            "nic_code": ["64990", "64990", "62010"],
        }
    )
    address_frame = pl.DataFrame(
        {
            "legal_entity_identifier": company_frame["legal_entity_identifier"],
            "source_snapshot_id": [snapshot_id] * 3,
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
            "address_group_key": ["12 RISK ROAD", "12 RISK ROAD", "9 SAFE ROAD"],
        }
    )
    cirp_frame = pl.DataFrame(
        {
            "solvency_event_id": ["prior", "window"],
            "cin": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
            ],
            "event_date": [date(2023, 10, 1), date(2025, 4, 1)],
        }
    )
    config = SolvencyObservationConfig(
        run_id="b" * 64,
        feature_cutoff=date(2023, 11, 3),
        outcome_window_end=date(2026, 8, 21),
    )

    observations = build_solvency_observations(
        company_frame,
        address_frame,
        cirp_frame,
        config,
    )
    by_company = {
        row["company_snapshot_id"]: row for row in observations.to_dicts()
    }

    assert by_company["company-1"]["has_prior_cirp_announcement"] is True
    assert by_company["company-1"]["evaluation_eligible"] is False
    assert by_company["company-2"]["evaluation_eligible"] is True
    assert by_company["company-2"]["observed_cirp_public_announcement"] is True
    assert by_company["company-2"]["first_cirp_announcement_date"] == date(
        2025,
        4,
        1,
    )
    assert by_company["company-2"]["shared_address_company_count"] == 2
    assert by_company["company-2"][
        "address_registration_month_company_count"
    ] == 2
    assert by_company["company-2"]["paid_to_authorised_capital_ratio"] == 0.5
    assert by_company["company-2"]["nic_division"] == "64"
    assert by_company["company-3"]["observed_cirp_public_announcement"] is False
    assert by_company["company-3"]["target_name"] == (
        "cirp_public_announcement_outcome"
    )
