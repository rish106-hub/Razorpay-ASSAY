from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from assay.artifacts.parquet import write_immutable_parquet
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.signals.runner import (
    SignalObservationRunConfig,
    SignalObservationRunner,
)


def _write_report(report_path: Path, report: BaseModel) -> None:
    report_path.write_text(report.model_dump_json(), encoding="utf-8")


def test_signal_observation_runner_validates_and_publishes_inputs(
    tmp_path: Path,
) -> None:
    company_snapshot_id = "company-1"
    source_snapshot_id = "a" * 64
    company_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "company_snapshot_id": [company_snapshot_id],
                "source_snapshot_id": [source_snapshot_id],
                "snapshot_as_of": [date(2023, 11, 3)],
                "legal_entity_identifier": ["U12345DL2020PTC123456"],
                "legal_entity_identifier_type": ["CIN"],
                "legal_entity_identifier_is_valid_format": [True],
                "cin": ["U12345DL2020PTC123456"],
                "company_name": ["Merchant Evidence Limited"],
                "registration_date": [date(2020, 1, 1)],
                "state_code": ["DL"],
                "nic_code": ["64990"],
                "paid_up_capital_inr": [100_000],
            }
        ),
        tmp_path / "company.parquet",
        "zstd",
    )
    address_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "company_address_snapshot_id": ["address-1"],
                "source_snapshot_id": [source_snapshot_id],
                "legal_entity_identifier": ["U12345DL2020PTC123456"],
                "address_normalized": ["12 RISK ROAD"],
                "address_group_key": ["12 RISK ROAD"],
                "snapshot_as_of": [date(2023, 11, 3)],
            }
        ),
        tmp_path / "address.parquet",
        "zstd",
    )
    adverse_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "adverse_event_id": ["event-1"],
                "event_date": [date(2024, 1, 1)],
            }
        ),
        tmp_path / "adverse.parquet",
        "zstd",
    )
    decision_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "adverse_event_id": ["event-1"],
                "accepted_company_snapshot_id": [company_snapshot_id],
                "review_status": ["accepted"],
                "outcome_eligible": [True],
            }
        ),
        tmp_path / "decision.parquet",
        "zstd",
    )
    mca_report = McaCanonicalisationReport(
        source_snapshot_id=source_snapshot_id,
        snapshot_as_of=date(2023, 11, 3),
        acquisition_complete=True,
        raw_page_count=1,
        raw_record_count=1,
        company_snapshot_rows=1,
        address_snapshot_rows=1,
        duplicate_legal_entity_identifier_rows=0,
        invalid_legal_entity_identifier_rows=0,
        invalid_legal_entity_identifier_rate=0,
        max_invalid_legal_entity_identifier_rate=0.0001,
        cin_rows=1,
        llpin_rows=0,
        fcrn_rows=0,
        missing_registration_date_rows=0,
        blank_address_rows=0,
        quality_status="passed",
        staged_parts=(),
        company_parts=(company_artifact,),
        address_parts=(address_artifact,),
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
    linkage_report = EntityLinkageReport(
        run_id="c" * 64,
        mca_source_snapshot_id=source_snapshot_id,
        nse_source_snapshot_id=nse_report.source_snapshot_id,
        nse_provenance_complete_assets=0,
        adverse_event_rows=1,
        accepted_exact_cin_rows=1,
        pending_exact_name_review_rows=0,
        ambiguous_rows=0,
        unmatched_rows=0,
        identifier_collision_rows=0,
        persisted_candidate_rows=1,
        truncated_candidate_event_rows=0,
        outcome_eligible_rows=1,
        candidate_artifact=decision_artifact,
        decision_artifact=decision_artifact,
    )
    mca_report_path = tmp_path / "mca.json"
    nse_report_path = tmp_path / "nse.json"
    linkage_report_path = tmp_path / "linkage.json"
    _write_report(mca_report_path, mca_report)
    _write_report(nse_report_path, nse_report)
    _write_report(linkage_report_path, linkage_report)

    report = SignalObservationRunner(
        SignalObservationRunConfig(
            project_root=tmp_path,
            mca_report_path=mca_report_path,
            nse_report_path=nse_report_path,
            linkage_report_path=linkage_report_path,
            outcome_window_end=date(2026, 8, 22),
        )
    ).run()

    assert report.company_rows == 1
    assert report.evaluation_eligible_rows == 1
    assert report.observed_adverse_outcome_rows == 1
    assert Path(report.observation_artifact.path).is_file()
