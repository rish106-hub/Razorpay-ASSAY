from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from assay.artifacts.parquet import write_immutable_parquet
from assay.serving.index import (
    MERCHANT_INDEX_COLUMNS,
    MerchantRiskIndexReport,
    build_merchant_risk_index,
)
from assay.serving.lookup import (
    MerchantLookupError,
    MerchantRiskIndexStore,
)

FEATURE_CUTOFF = date(2025, 1, 1)
PLACEHOLDER_ID = "a" * 64


def _observation_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_snapshot_id": [f"snapshot-{index}" for index in range(1, 6)],
            "cin": [
                "U72900KA2019PTC000002",
                "U72900MH2015PTC000001",
                "U72900DL2011PTC000003",
                "U72900DL2012PTC000004",
                "U72900TN2018PTC000005",
            ],
            "company_name": [
                "Acme Retail Private Limited",
                "Zephyr Logistics Limited",
                "Orion Traders Private Limited",
                "Orion Traders Private Limited",
                "Ineligible Ventures Private Limited",
            ],
            "evaluation_eligible": [True, True, True, True, False],
            "observed_cirp_public_announcement": [False] * 5,
            "first_cirp_announcement_date": [None] * 5,
            "company_age_days": [1_500, 3_000, 4_500, 4_400, 2_000],
            "log_authorised_capital_inr": [13.8, 15.4, 12.0, 12.1, 11.0],
            "log_paid_up_capital_inr": [13.1, 14.9, 11.5, 11.6, 10.5],
            "paid_to_authorised_capital_ratio": [0.5, 20.0, 1.0, 1.0, 0.9],
            "shared_address_company_count": [3, 0, 7, 5, 1],
            "address_registration_month_company_count": [2, 0, 4, 3, 1],
            "registration_date": [
                date(2019, 9, 20),
                date(2015, 8, 15),
                date(2011, 6, 1),
                date(2012, 3, 4),
                date(2018, 1, 9),
            ],
            "authorised_capital_inr": [
                1_000_000.0,
                5_000_000.0,
                200_000.0,
                210_000.0,
                100_000.0,
            ],
            "paid_up_capital_inr": [
                500_000.0,
                4_900_000.0,
                200_000.0,
                205_000.0,
                90_000.0,
            ],
            "nic_code": ["72900", None, "49221", "49221", "72900"],
            "state_code": [
                "karnataka",
                "maharashtra",
                "delhi",
                "delhi",
                "tamil nadu",
            ],
            "roc_code": [
                "RoC-Bangalore",
                "RoC-Mumbai",
                "RoC-Delhi",
                "RoC-Delhi",
                "RoC-Chennai",
            ],
            "company_status": ["ACTIVE"] * 4 + ["STRIKE OFF"],
            "company_category": ["Company limited by Shares"] * 5,
            "company_subcategory": ["Non-govt company"] * 5,
            "company_class": ["Private"] * 5,
            "listing_status": ["Unlisted"] * 5,
            "company_origin": ["Indian"] * 5,
            "nic_division": ["72", None, "49", "49", "72"],
        }
    )


def _publish_index(tmp_path: Path) -> Path:
    """Write a tiny index plus a matching report and return the report path."""

    index_frame = build_merchant_risk_index(_observation_frame())
    index_artifact = write_immutable_parquet(
        index_frame,
        tmp_path / "merchant_risk_index.parquet",
        "zstd",
    )
    report = MerchantRiskIndexReport(
        run_id=PLACEHOLDER_ID,
        observation_run_id=PLACEHOLDER_ID,
        mca_source_snapshot_id=PLACEHOLDER_ID,
        ibbi_source_snapshot_id=PLACEHOLDER_ID,
        feature_cutoff=FEATURE_CUTOFF,
        indexed_companies=index_frame.height,
        distinct_strict_names=index_frame[
            "company_name_normalized_strict"
        ].n_unique(),
        companies_sharing_a_strict_name=int(
            (index_frame["strict_name_company_count"] > 1).sum()
        ),
        index_artifact=index_artifact,
    )
    report_path = tmp_path / "merchant_risk_index.report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


@pytest.fixture(name="store")
def store_fixture(tmp_path: Path) -> MerchantRiskIndexStore:
    return MerchantRiskIndexStore.from_report_path(_publish_index(tmp_path))


def test_store_reports_the_index_it_serves(
    store: MerchantRiskIndexStore,
) -> None:
    assert store.report.indexed_companies == 4
    assert store.report.feature_cutoff == FEATURE_CUTOFF


def test_exact_cin_resolves_to_one_frozen_feature_row(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(cin=" u72900ka2019ptc000002 ")

    assert result.outcome == "matched"
    assert result.match_method == "exact_cin"
    assert result.matched_row is not None
    assert result.matched_row.height == 1
    assert result.matched_row.columns == list(MERCHANT_INDEX_COLUMNS)
    assert result.matched_row["company_name"][0] == "Acme Retail Private Limited"
    assert result.candidates == ()


def test_malformed_cin_never_falls_back_to_name_matching(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(
        cin="NOTACIN",
        company_name="Acme Retail Private Limited",
    )

    assert result.outcome == "invalid_identifier"
    assert result.match_method is None
    assert result.matched_row is None
    assert result.candidates == ()
    assert "21 characters" in result.detail


def test_well_formed_cin_outside_the_index_explains_the_cutoff(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(cin="U99999WB2020PTC999999")

    assert result.outcome == "no_match"
    assert result.match_method is None
    assert result.matched_row is None
    assert "feature cutoff 2025-01-01" in result.detail
    assert "LLPIN" in result.detail
    assert "CIRP" in result.detail


def test_unique_strict_name_matches_without_a_legal_fallback(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(company_name="zephyr  logistics limited")

    assert result.outcome == "matched"
    assert result.match_method == "normalized_strict_name"
    assert result.matched_row is not None
    assert result.matched_row["cin"][0] == "U72900MH2015PTC000001"


def test_missing_legal_suffix_resolves_through_the_legal_form(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(company_name="Acme Retail")

    assert result.outcome == "matched"
    assert result.match_method == "normalized_legal_name"
    assert result.matched_row is not None
    assert result.matched_row["cin"][0] == "U72900KA2019PTC000002"


def test_shared_name_returns_candidates_and_never_a_matched_row(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(company_name="Orion Traders Private Limited")

    assert result.outcome == "ambiguous_name"
    assert result.match_method is None
    assert result.matched_row is None
    assert [candidate.cin for candidate in result.candidates] == [
        "U72900DL2011PTC000003",
        "U72900DL2012PTC000004",
    ]
    assert result.candidates[0].registration_date == date(2011, 6, 1)
    assert result.candidates[0].state_code == "delhi"
    assert "Resubmit the request with a CIN" in result.detail


def test_unknown_name_reports_both_normalised_forms_tried(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(company_name="Nonesuch Holdings Private Limited")

    assert result.outcome == "no_match"
    assert result.matched_row is None
    assert "'NONESUCH HOLDINGS PRIVATE LIMITED'" in result.detail
    assert "'NONESUCH HOLDINGS'" in result.detail


def test_unindexable_name_reports_no_match(
    store: MerchantRiskIndexStore,
) -> None:
    result = store.lookup(company_name="!!! ---")

    assert result.outcome == "no_match"
    assert "no indexable characters" in result.detail


def test_ineligible_company_is_absent_from_the_served_population(
    store: MerchantRiskIndexStore,
) -> None:
    assert store.lookup(cin="U72900TN2018PTC000005").outcome == "no_match"


def test_lookup_without_an_identifier_or_a_name_is_rejected(
    store: MerchantRiskIndexStore,
) -> None:
    with pytest.raises(MerchantLookupError, match="CIN or a company name"):
        store.lookup(cin="   ", company_name="  ")


def test_checksum_verification_rejects_a_tampered_index(tmp_path: Path) -> None:
    report_path = _publish_index(tmp_path)
    index_path = tmp_path / "merchant_risk_index.parquet"
    index_path.write_bytes(index_path.read_bytes() + b"tampered")

    with pytest.raises(MerchantLookupError, match="size mismatch"):
        MerchantRiskIndexStore.from_report_path(report_path)


def test_missing_index_artifact_is_rejected_without_checksums(
    tmp_path: Path,
) -> None:
    report_path = _publish_index(tmp_path)
    (tmp_path / "merchant_risk_index.parquet").unlink()

    with pytest.raises(MerchantLookupError, match="missing"):
        MerchantRiskIndexStore.from_report_path(
            report_path, verify_checksum=False
        )


def test_invalid_report_document_is_rejected(tmp_path: Path) -> None:
    report_path = tmp_path / "not-a-report.json"
    report_path.write_text('{"run_id": "short"}', encoding="utf-8")

    with pytest.raises(MerchantLookupError, match="missing or invalid"):
        MerchantRiskIndexStore.from_report_path(report_path)
