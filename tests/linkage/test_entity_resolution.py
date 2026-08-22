from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from assay.artifacts.parquet import write_immutable_parquet
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import (
    EntityLinkageConfig,
    EntityLinker,
    build_entity_linkage_frames,
)


def test_linkage_accepts_only_unique_exact_cin_matches() -> None:
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": ["company-1", "company-2", "company-3"],
            "cin": [
                "U12345DL2020PTC123456",
                "U54321MH2020PTC654321",
                "U99999KA2020PTC999999",
            ],
            "company_name": [
                "Merchant Evidence Private Limited",
                "Shared Name Limited",
                "Shared Name Private Limited",
            ],
        }
    )
    adverse_frame = pl.DataFrame(
        {
            "adverse_event_id": [
                "event-cin",
                "event-name",
                "event-ambiguous",
                "event-unmatched",
            ],
            "cin_candidate": [
                "U12345DL2020PTC123456",
                None,
                None,
                None,
            ],
            "entity_name_normalized_strict": [
                "MERCHANT EVIDENCE PRIVATE LIMITED",
                "MERCHANT EVIDENCE PRIVATE LIMITED",
                "SHARED NAME",
                "NO SUCH MERCHANT",
            ],
            "entity_name_normalized_legal": [
                "MERCHANT EVIDENCE",
                "MERCHANT EVIDENCE",
                "SHARED NAME",
                "NO SUCH MERCHANT",
            ],
        }
    )

    candidates, decisions = build_entity_linkage_frames(
        company_frame,
        adverse_frame,
        run_id="a" * 64,
    )
    decisions_by_event = {
        row["adverse_event_id"]: row for row in decisions.to_dicts()
    }

    assert decisions_by_event["event-cin"]["review_status"] == "accepted"
    assert decisions_by_event["event-cin"]["outcome_eligible"] is True
    assert decisions_by_event["event-name"]["review_status"] == "pending_review"
    assert decisions_by_event["event-name"]["outcome_eligible"] is False
    assert decisions_by_event["event-ambiguous"]["review_status"] == "ambiguous"
    assert decisions_by_event["event-unmatched"]["review_status"] == "unmatched"
    assert decisions_by_event["event-unmatched"]["outcome_eligible"] is False
    assert candidates.filter(pl.col("adverse_event_id") == "event-cin").height == 1


def test_entity_linker_publishes_checksum_verified_artifacts(tmp_path: Path) -> None:
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": ["company-1"],
            "cin": ["U12345DL2020PTC123456"],
            "company_name": ["Merchant Evidence Private Limited"],
        }
    )
    adverse_frame = pl.DataFrame(
        {
            "adverse_event_id": ["event-1"],
            "cin_candidate": ["U12345DL2020PTC123456"],
            "entity_name_normalized_strict": [
                "MERCHANT EVIDENCE PRIVATE LIMITED"
            ],
            "entity_name_normalized_legal": ["MERCHANT EVIDENCE"],
        }
    )
    company_artifact = write_immutable_parquet(
        company_frame,
        tmp_path / "company.parquet",
        "zstd",
    )
    adverse_artifact = write_immutable_parquet(
        adverse_frame,
        tmp_path / "adverse.parquet",
        "zstd",
    )
    mca_report = McaCanonicalisationReport(
        source_snapshot_id="a" * 64,
        snapshot_as_of=date(2023, 11, 3),
        acquisition_complete=True,
        raw_page_count=1,
        raw_record_count=1,
        company_snapshot_rows=1,
        address_snapshot_rows=1,
        duplicate_cin_rows=0,
        invalid_cin_rows=0,
        missing_registration_date_rows=0,
        blank_address_rows=0,
        quality_status="passed",
        staged_parts=(),
        company_parts=(company_artifact,),
        address_parts=(),
    )
    nse_report = NseCanonicalisationReport(
        source_snapshot_id="b" * 64,
        adverse_event_rows=1,
        sebi_event_rows=1,
        other_authority_event_rows=0,
        cin_candidate_rows=1,
        din_candidate_rows=0,
        pan_candidate_rows=1,
        missing_entity_name_rows=0,
        missing_event_date_rows=0,
        provenance_complete_assets=0,
        parts=(adverse_artifact,),
    )
    mca_report_path = tmp_path / "mca-report.json"
    nse_report_path = tmp_path / "nse-report.json"
    mca_report_path.write_text(mca_report.model_dump_json(), encoding="utf-8")
    nse_report_path.write_text(nse_report.model_dump_json(), encoding="utf-8")

    report = EntityLinker(
        EntityLinkageConfig(
            project_root=tmp_path,
            mca_report_path=mca_report_path,
            nse_report_path=nse_report_path,
        )
    ).run()

    assert report.accepted_exact_cin_rows == 1
    assert report.outcome_eligible_rows == 1
    assert Path(report.candidate_artifact.path).is_file()
    assert Path(report.decision_artifact.path).is_file()
