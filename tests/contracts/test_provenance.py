from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from assay.contracts.provenance import (
    MerchantRiskAsset,
    load_and_validate_manifests,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_manifests_satisfy_provenance_schema() -> None:
    source_manifest, acquisition_manifest = load_and_validate_manifests(
        PROJECT_ROOT / "data/manifests/sources.yaml",
        PROJECT_ROOT / "data/manifests/acquired.yaml",
    )

    assert len(source_manifest.sources) >= 1
    assert len(acquisition_manifest.assets) >= 1


def test_unknown_terms_cannot_be_marked_provenance_complete() -> None:
    with pytest.raises(ValidationError, match="unknown provenance"):
        MerchantRiskAsset.model_validate(
            {
                "id": "merchant_risk_asset",
                "source_id": "merchant_risk_source",
                "source_url": "https://example.com/source.csv",
                "publisher": "Merchant Risk Publisher",
                "retrieved_at": "2026-08-22T12:00:00+05:30",
                "extract_date": "unknown",
                "licence_or_terms": "unknown_source_specific_terms",
                "intended_use": "adverse_outcome_validation",
                "restrictions": ["not_fraud_ground_truth"],
                "provenance_status": "complete",
                "local_path": "data/raw/source.csv",
                "sha256": "a" * 64,
                "bytes": 1,
                "format": "csv",
                "status": "schema_validated",
                "sheet": "not_applicable",
                "rows": 1,
                "columns": ["merchant_id"],
            }
        )
