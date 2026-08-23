from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from assay.acquisition.mca import McaAcquisitionCheckpoint, McaPageArtifact
from assay.canonicalisation.mca import (
    McaCanonicalisationConfig,
    McaCanonicaliser,
    _canonical_company_frame,
)


def _write_complete_acquisition(raw_directory: Path) -> None:
    merchant_records = [
        {
            "AuthorizedCapital": "100000.00",
            "CIN": "U12345DL2020PTC123456",
            "CompanyCategory": "Company limited by Shares",
            "CompanyClass": "Private",
            "CompanyIndian/Foreign Company": "Indian",
            "CompanyIndustrialClassification": "Financial intermediation",
            "CompanyName": "Merchant Evidence Private Limited",
            "CompanyROCcode": "RoC-Delhi",
            "CompanyRegistrationdate_date": "2020-01-02",
            "CompanyStateCode": "Delhi",
            "CompanyStatus": "Active",
            "CompanySubCategory": "Non-govt company",
            "Listingstatus": "Unlisted",
            "PaidupCapital": "50000.00",
            "Registered_Office_Address": " 12, Risk Road; New Delhi ",
            "nic_code": "64990",
        }
    ]
    raw_payload = json.dumps(
        {
            "status": "ok",
            "total": 1,
            "count": 1,
            "records": merchant_records,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    raw_sha256 = hashlib.sha256(raw_payload).hexdigest()
    raw_page_name = f"page-offset-0000000000-sha256-{raw_sha256[:16]}.json"
    raw_directory.mkdir(parents=True)
    (raw_directory / raw_page_name).write_bytes(raw_payload)
    checkpoint = McaAcquisitionCheckpoint(
        resource_id="4dbe5667-7b6b-41d7-82af-211562424d9a",
        page_size=1,
        next_offset=1,
        total_records=1,
        pages=(
            McaPageArtifact(
                resource_id="4dbe5667-7b6b-41d7-82af-211562424d9a",
                offset=0,
                requested_limit=1,
                record_count=1,
                total_records=1,
                retrieved_at=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
                raw_path=raw_page_name,
                sha256=raw_sha256,
                byte_count=len(raw_payload),
            ),
        ),
    )
    (raw_directory / "checkpoint.json").write_text(
        checkpoint.model_dump_json(indent=2), encoding="utf-8"
    )


def test_canonicaliser_writes_versioned_company_and_address_snapshots(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw"
    _write_complete_acquisition(raw_directory)
    config = McaCanonicalisationConfig(
        raw_directory=raw_directory,
        staged_root=tmp_path / "staged",
        curated_company_root=tmp_path / "company",
        curated_address_root=tmp_path / "address",
        generated_report_root=tmp_path / "reports",
    )

    report = McaCanonicaliser(config).run()

    assert report.acquisition_complete is True
    assert report.company_snapshot_rows == 1
    assert report.address_snapshot_rows == 1
    assert report.duplicate_legal_entity_identifier_rows == 0
    assert report.invalid_legal_entity_identifier_rows == 0
    assert report.cin_rows == 1
    assert report.llpin_rows == 0
    assert report.fcrn_rows == 0
    assert report.quality_status == "passed"

    company_frame = pl.read_parquet(report.company_parts[0].path)
    address_frame = pl.read_parquet(report.address_parts[0].path)
    assert company_frame["registration_date"].item().isoformat() == "2020-01-02"
    assert str(company_frame["paid_up_capital_inr"].item()) == "50000.00"
    assert company_frame["snapshot_as_of"].item().isoformat() == "2023-11-03"
    assert address_frame["address_normalized"].item() == "12 RISK ROAD NEW DELHI"
    assert address_frame["address_group_key"].item() == "12 RISK ROAD NEW DELHI"


def test_company_frame_classifies_mca_legal_entity_identifiers() -> None:
    staged_frame = pl.DataFrame(
        {
            "CIN": ["U12345DL2020PTC123456", "AAA-1111", "F01234", "TEST1"],
            "source_file_sha256": ["a" * 64] * 4,
            "observed_at": [datetime(2026, 8, 22, 12, 0, tzinfo=UTC)] * 4,
            "CompanyName": ["Entity"] * 4,
            "CompanyStatus": ["Active"] * 4,
            "CompanyRegistrationdate_date": ["2020-01-02"] * 4,
            "CompanyStateCode": ["Delhi"] * 4,
            "CompanyROCcode": ["RoC-Delhi"] * 4,
            "CompanyCategory": ["Company limited by Shares"] * 4,
            "CompanySubCategory": ["Non-govt company"] * 4,
            "CompanyClass": ["Private"] * 4,
            "Listingstatus": ["Unlisted"] * 4,
            "CompanyIndian/Foreign Company": ["Indian"] * 4,
            "AuthorizedCapital": ["100000.00"] * 4,
            "PaidupCapital": ["50000.00"] * 4,
            "nic_code": ["64990"] * 4,
            "CompanyIndustrialClassification": ["Financial intermediation"] * 4,
        }
    )

    company_frame = _canonical_company_frame(
        staged_frame,
        source_snapshot_id="b" * 64,
        snapshot_as_of=date(2023, 11, 3),
    )

    assert company_frame["legal_entity_identifier_type"].to_list() == [
        "CIN",
        "LLPIN",
        "FCRN",
        "UNKNOWN",
    ]
    assert company_frame["legal_entity_identifier_is_valid_format"].to_list() == [
        True,
        True,
        True,
        False,
    ]
    assert company_frame["cin"].to_list() == [
        "U12345DL2020PTC123456",
        None,
        None,
        None,
    ]
