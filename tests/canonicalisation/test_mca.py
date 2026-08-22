from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from assay.acquisition.mca import McaAcquisitionCheckpoint, McaPageArtifact
from assay.canonicalisation.mca import McaCanonicalisationConfig, McaCanonicaliser


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
    assert report.duplicate_cin_rows == 0
    assert report.invalid_cin_rows == 0
    assert report.quality_status == "passed"

    company_frame = pl.read_parquet(report.company_parts[0].path)
    address_frame = pl.read_parquet(report.address_parts[0].path)
    assert company_frame["registration_date"].item().isoformat() == "2020-01-02"
    assert str(company_frame["paid_up_capital_inr"].item()) == "50000.00"
    assert company_frame["snapshot_as_of"].item().isoformat() == "2023-11-03"
    assert address_frame["address_normalized"].item() == "12 RISK ROAD NEW DELHI"
    assert address_frame["address_group_key"].item() == "12 RISK ROAD NEW DELHI"
