from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from assay.serving.index import (
    MERCHANT_INDEX_COLUMNS,
    MerchantIndexError,
    build_merchant_risk_index,
)
from assay.solvency.training_data import (
    SolvencyTrainingDataConfig,
    build_solvency_model_data,
)


def _observation_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_snapshot_id": ["snapshot-1", "snapshot-2", "snapshot-3"],
            "cin": [
                "U72900KA2019PTC000002",
                "U72900MH2015PTC000001",
                "U72900DL2011PTC000003",
            ],
            "company_name": [
                "Acme Retail Private Limited",
                "acme retail pvt ltd",
                "Zephyr Logistics Limited",
            ],
            "evaluation_eligible": [True, True, False],
            "observed_cirp_public_announcement": [False, True, False],
            "first_cirp_announcement_date": [None, date(2026, 2, 1), None],
            "company_age_days": [1_500, 3_000, 4_500],
            "log_authorised_capital_inr": [13.8, 15.4, 12.0],
            "log_paid_up_capital_inr": [13.1, 14.9, 11.5],
            "paid_to_authorised_capital_ratio": [0.5, 20.0, 1.0],
            "shared_address_company_count": [3, 0, 7],
            "address_registration_month_company_count": [2, 0, 4],
            "registration_date": [
                date(2019, 9, 20),
                date(2015, 8, 15),
                date(2011, 6, 1),
            ],
            "authorised_capital_inr": [1_000_000.0, 5_000_000.0, 200_000.0],
            "paid_up_capital_inr": [500_000.0, 4_900_000.0, 200_000.0],
            "nic_code": ["72900", None, "49221"],
            "state_code": ["karnataka", "maharashtra", "delhi"],
            "roc_code": ["RoC-Bangalore", "RoC-Mumbai", "RoC-Delhi"],
            "company_status": ["ACTIVE", "ACTIVE", "STRIKE OFF"],
            "company_category": ["Company limited by Shares"] * 3,
            "company_subcategory": ["Non-govt company"] * 3,
            "company_class": ["Private", "Private", "Public"],
            "listing_status": ["Unlisted", "Unlisted", "Listed"],
            "company_origin": ["Indian", "Indian", "Indian"],
            "nic_division": ["72", None, "49"],
        }
    )


def test_index_keeps_only_eligible_companies_and_flags_name_collisions() -> None:
    index_frame = build_merchant_risk_index(_observation_frame())

    assert index_frame.columns == list(MERCHANT_INDEX_COLUMNS)
    assert index_frame["cin"].to_list() == [
        "U72900KA2019PTC000002",
        "U72900MH2015PTC000001",
    ]
    assert index_frame["company_name_normalized_strict"].to_list() == [
        "ACME RETAIL PRIVATE LIMITED",
        "ACME RETAIL PVT LTD",
    ]
    assert index_frame["company_name_normalized_legal"].unique().to_list() == [
        "ACME RETAIL"
    ]
    assert index_frame["strict_name_company_count"].to_list() == [1, 1]
    assert index_frame["nic_division"].to_list() == ["72", "UNKNOWN"]


def test_index_features_match_the_frozen_training_derivation() -> None:
    observation_frame = _observation_frame()
    model_frame = build_solvency_model_data(
        observation_frame,
        SolvencyTrainingDataConfig(
            observation_report_path="unused.json",  # type: ignore[arg-type]
        ),
    )
    index_frame = build_merchant_risk_index(observation_frame)

    feature_columns = [
        column
        for column in model_frame.columns
        if column not in {"company_snapshot_id", "dataset_split", "target",
                          "first_cirp_announcement_date"}
    ]
    assert model_frame.sort("cin").select(feature_columns).equals(
        index_frame.sort("cin").select(feature_columns)
    )


def test_index_rejects_observations_without_required_columns() -> None:
    with pytest.raises(MerchantIndexError, match="missing"):
        build_merchant_risk_index(_observation_frame().drop("nic_division"))


def test_index_rejects_duplicate_company_identifiers() -> None:
    duplicated = pl.concat(
        [_observation_frame(), _observation_frame().head(1)],
        how="vertical",
    )
    with pytest.raises(MerchantIndexError, match="one row per CIN"):
        build_merchant_risk_index(duplicated)
