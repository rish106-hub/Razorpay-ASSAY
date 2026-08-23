from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from assay.solvency.training_data import (
    EXPANDED_SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_CATEGORICAL_FEATURES,
    SolvencyTrainingDataConfig,
    build_solvency_model_data,
)


def test_solvency_model_splits_prioritise_geography_then_event_time() -> None:
    company_ids = ["geography", "temporal", "validation", "training"]
    observation_frame = pl.DataFrame(
        {
            "company_snapshot_id": company_ids,
            "cin": [
                "U12345KA2020PTC123456",
                "U12345DL2020PTC123457",
                "U12345DL2020PTC123458",
                "U12345DL2020PTC123459",
            ],
            "evaluation_eligible": [True] * 4,
            "observed_cirp_public_announcement": [True] * 4,
            "first_cirp_announcement_date": [
                date(2024, 1, 1),
                date(2026, 2, 1),
                date(2025, 8, 1),
                date(2024, 2, 1),
            ],
            "company_age_days": [1000] * 4,
            "log_authorised_capital_inr": [12.0] * 4,
            "log_paid_up_capital_inr": [11.0] * 4,
            "paid_to_authorised_capital_ratio": [0.5] * 4,
            "shared_address_company_count": [1] * 4,
            "address_registration_month_company_count": [1] * 4,
            "address_cluster_registration_span_days": [0] * 4,
            "address_cluster_registration_month_entropy": [0.0] * 4,
            "address_cluster_max_month_share": [1.0] * 4,
            "address_cluster_nic_division_distinct": [1] * 4,
            "address_cluster_authorised_capital_cv": [0.0] * 4,
            "address_cluster_distinct_name_head_ratio": [1.0] * 4,
            "registrar_year_cohort_company_count": [1] * 4,
            "address_cluster_cohort_peer_count": [0] * 4,
            "address_cluster_roc_serial_min_gap": [-1] * 4,
            "cin_record_disagreement_count": [0] * 4,
            **{
                feature_name: [False] * 4
                for feature_name in EXPANDED_SOLVENCY_CATEGORICAL_FEATURES
            },
            **{
                feature_name: [
                    "karnataka" if feature_name == "state_code" else "VALUE",
                    "delhi" if feature_name == "state_code" else "VALUE",
                    "delhi" if feature_name == "state_code" else "VALUE",
                    "delhi" if feature_name == "state_code" else "VALUE",
                ]
                for feature_name in SOLVENCY_CATEGORICAL_FEATURES
            },
        }
    )
    config = SolvencyTrainingDataConfig(
        observation_report_path=Path("unused.json")
    )

    model_frame = build_solvency_model_data(observation_frame, config)
    splits = dict(
        model_frame.select("company_snapshot_id", "dataset_split").iter_rows()
    )

    assert splits == {
        "geography": "geography_test",
        "temporal": "temporal_test",
        "validation": "validation",
        "training": "training",
    }
