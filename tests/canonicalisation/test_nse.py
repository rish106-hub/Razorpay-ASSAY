from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl

from assay.canonicalisation.nse import _canonicalise_adverse_frame
from assay.contracts.provenance import MerchantRiskAsset


def test_adverse_event_canonicalisation_preserves_source_semantics() -> None:
    source_frame = pl.DataFrame(
        {
            "Order Date": [date(2026, 8, 21)],
            "Order Particulars": ["Regulatory order"],
            "Entity / Individual Name": ["Merchant Evidence Pvt. Ltd."],
            "PAN": ["ABCDE1234F"],
            "DIN / CIN": ["U12345DL2020PTC123456"],
            "Symbol ": ["ASSAY"],
            "Period": ["Six months"],
            "NSE Circular No. (For Debarment)": ["NSE/TEST/1"],
            "Date of NSE circular": [date(2026, 8, 22)],
            "NSE Circular No. (For Revocation)": [""],
            "Date of NSE circular. (For Revocation)": [None],
            "source_row_number": [2],
        }
    )
    asset = MerchantRiskAsset.model_validate(
        {
            "id": "nse_sebi_test",
            "source_id": "nse_sebi_debarred",
            "source_url": "https://example.com/nse.xls",
            "publisher": "National Stock Exchange of India Limited",
            "retrieved_at": datetime(2026, 8, 22, 15, 28, tzinfo=UTC),
            "extract_date": "unknown",
            "licence_or_terms": "unknown_source_specific_terms",
            "intended_use": "external_adverse_outcome_validation",
            "restrictions": ["debarment_is_not_fraud_ground_truth"],
            "provenance_status": "incomplete",
            "local_path": "data/raw/nse.xls",
            "sha256": "a" * 64,
            "bytes": 1,
            "format": "xls",
            "status": "schema_validated",
            "sheet": "Sheet1",
            "rows": 1,
            "columns": list(source_frame.columns),
        }
    )

    adverse_frame = _canonicalise_adverse_frame(
        source_frame,
        asset=asset,
        source_snapshot_id="b" * 64,
    )

    assert adverse_frame["event_source_authority"].item() == "SEBI"
    assert adverse_frame["adverse_regulatory_outcome"].item() is True
    assert adverse_frame["cin_candidate"].item() == "U12345DL2020PTC123456"
    assert adverse_frame["pan_candidate"].item() == "ABCDE1234F"
    assert adverse_frame["entity_name_normalized_legal"].item() == (
        "MERCHANT EVIDENCE"
    )
