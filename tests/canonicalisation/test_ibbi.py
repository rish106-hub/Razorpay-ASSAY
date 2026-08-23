from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from assay.canonicalisation.ibbi import (
    IBBI_CIRP_ANNOUNCEMENT_TYPE,
    IBBI_REQUIRED_COLUMNS,
    IbbiCanonicalisationConfig,
    IbbiCanonicaliser,
)
from assay.contracts.provenance import MerchantRiskAsset


def _asset() -> MerchantRiskAsset:
    return MerchantRiskAsset.model_validate(
        {
            "id": "ibbi_test",
            "source_id": "ibbi_cirp_public_announcements",
            "source_url": "https://ibbi.gov.in/public-announcement",
            "publisher": "Insolvency and Bankruptcy Board of India",
            "retrieved_at": datetime(2026, 8, 22, 20, 58, tzinfo=UTC),
            "extract_date": date(2026, 8, 21),
            "licence_or_terms": "unknown_IBBI_website_reuse_terms",
            "intended_use": "merchant_solvency_outcome_validation",
            "restrictions": ["insolvency_is_not_fraud_ground_truth"],
            "provenance_status": "incomplete",
            "local_path": "data/raw/ibbi.tsv",
            "sha256": "a" * 64,
            "bytes": 1,
            "format": "tsv_export_with_xlsx_response_name",
            "status": "schema_audited",
            "sheet": "tab_separated_export",
            "rows": 3,
            "columns": IBBI_REQUIRED_COLUMNS,
        }
    )


def test_ibbi_export_preserves_solvency_semantics_and_quarantines_rows(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "ibbi.tsv"
    valid_row = (
        IBBI_CIRP_ANNOUNCEMENT_TYPE,
        "21-08-2026",
        "02-09-2026",
        "Merchant Evidence Pvt. Ltd.",
        "U12345DL2020PTC123456",
        "Secured Creditor",
        "Resolution Professional",
        "New Delhi",
        "",
    )
    invalid_cin_row = (*valid_row[:4], "U001234", *valid_row[5:])
    malformed_row = valid_row[:-1]
    export_path.write_text(
        "\t".join(IBBI_REQUIRED_COLUMNS)
        + "\n"
        + "\t".join(valid_row)
        + "\n"
        + "\t".join(invalid_cin_row)
        + "\n"
        + "\t".join(malformed_row)
        + "\n",
        encoding="utf-8",
    )
    canonicaliser = IbbiCanonicaliser(
        IbbiCanonicalisationConfig(project_root=tmp_path)
    )

    source_rows, event_records, quarantine_records = canonicaliser._parse_export(
        _asset(),
        export_path,
        "b" * 64,
    )

    assert source_rows == 3
    assert len(event_records) == 1
    assert event_records[0]["cin"] == "U12345DL2020PTC123456"
    assert event_records[0]["event_date"] == date(2026, 8, 21)
    assert event_records[0]["event_date_semantics"] == (
        "public_announcement_date"
    )
    assert event_records[0]["corporate_debtor_name_normalized"] == (
        "MERCHANT EVIDENCE PVT LTD"
    )
    assert {record["quarantine_reason"] for record in quarantine_records} == {
        "invalid_cin",
        "malformed_field_count",
    }
