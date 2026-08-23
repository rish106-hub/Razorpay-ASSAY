from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

from assay.artifacts.parquet import write_immutable_parquet
from assay.canonicalisation.ibbi import IbbiCanonicalisationReport
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.solvency.observations import SOLVENCY_OBSERVATION_SCHEMA_VERSION
from assay.solvency.runner import (
    SolvencyObservationRunConfig,
    SolvencyObservationRunError,
    SolvencyObservationRunner,
)

FEATURE_CUTOFF = date(2023, 11, 3)
MCA_SNAPSHOT_ID = "a" * 64
IBBI_SNAPSHOT_ID = "b" * 64
SHAPE_COLUMNS = (
    "address_cluster_registration_span_days",
    "address_cluster_registration_month_entropy",
    "address_cluster_max_month_share",
    "address_cluster_nic_division_distinct",
    "address_cluster_authorised_capital_cv",
    "address_cluster_distinct_name_head_ratio",
)


def _write_report(report_path: Path, report: BaseModel) -> None:
    report_path.write_text(report.model_dump_json(), encoding="utf-8")


def _run_config(tmp_path: Path) -> SolvencyObservationRunConfig:
    identifiers = [
        "U11111DL2020PTC111111",
        "U22222DL2020PTC222222",
        "U33333MH2021PTC333333",
    ]
    company_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "company_snapshot_id": ["company-1", "company-2", "company-3"],
                "source_snapshot_id": [MCA_SNAPSHOT_ID] * 3,
                "snapshot_as_of": [FEATURE_CUTOFF] * 3,
                "legal_entity_identifier": identifiers,
                "legal_entity_identifier_type": ["CIN"] * 3,
                "legal_entity_identifier_is_valid_format": [True] * 3,
                "cin": identifiers,
                "company_name": [
                    "Batch Traders One Private Limited",
                    "Batch Traders Two Private Limited",
                    "Unrelated Foods Private Limited",
                ],
                "company_status": ["Active"] * 3,
                "registration_date": [
                    date(2020, 1, 6),
                    date(2020, 1, 9),
                    date(2021, 5, 1),
                ],
                "state_code": ["delhi", "delhi", "maharashtra"],
                "roc_code": ["roc delhi", "roc delhi", "roc mumbai"],
                "company_category": ["company limited by shares"] * 3,
                "company_subcategory": ["non-government company"] * 3,
                "company_class": ["private"] * 3,
                "listing_status": ["unlisted"] * 3,
                "company_origin": ["indian"] * 3,
                "authorised_capital_inr": [100_000, 100_000, 750_000],
                "paid_up_capital_inr": [100_000, 100_000, 250_000],
                "nic_code": ["64990", "64990", "10101"],
            }
        ),
        tmp_path / "company.parquet",
        "zstd",
    )
    address_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "legal_entity_identifier": identifiers,
                "source_snapshot_id": [MCA_SNAPSHOT_ID] * 3,
                "snapshot_as_of": [FEATURE_CUTOFF] * 3,
                "address_group_key": [
                    "12 RISK ROAD",
                    "12 RISK ROAD",
                    "9 SAFE ROAD",
                ],
            }
        ),
        tmp_path / "address.parquet",
        "zstd",
    )
    event_artifact = write_immutable_parquet(
        pl.DataFrame(
            {
                "solvency_event_id": ["event-1"],
                "cin": [identifiers[1]],
                "event_date": [date(2025, 4, 1)],
            }
        ),
        tmp_path / "cirp_event.parquet",
        "zstd",
    )
    quarantine_artifact = write_immutable_parquet(
        pl.DataFrame({"quarantine_reason": ["invalid_cin"]}),
        tmp_path / "cirp_quarantine.parquet",
        "zstd",
    )
    mca_report_path = tmp_path / "mca.json"
    ibbi_report_path = tmp_path / "ibbi.json"
    _write_report(
        mca_report_path,
        McaCanonicalisationReport(
            source_snapshot_id=MCA_SNAPSHOT_ID,
            snapshot_as_of=FEATURE_CUTOFF,
            acquisition_complete=True,
            raw_page_count=1,
            raw_record_count=3,
            company_snapshot_rows=3,
            address_snapshot_rows=3,
            duplicate_legal_entity_identifier_rows=0,
            invalid_legal_entity_identifier_rows=0,
            invalid_legal_entity_identifier_rate=0,
            max_invalid_legal_entity_identifier_rate=0.0001,
            cin_rows=3,
            llpin_rows=0,
            fcrn_rows=0,
            missing_registration_date_rows=0,
            blank_address_rows=0,
            quality_status="passed",
            staged_parts=(),
            company_parts=(company_artifact,),
            address_parts=(address_artifact,),
        ),
    )
    _write_report(
        ibbi_report_path,
        IbbiCanonicalisationReport(
            source_snapshot_id=IBBI_SNAPSHOT_ID,
            source_rows=1,
            solvency_event_rows=1,
            unique_cin_count=1,
            post_mca_cutoff_rows=1,
            post_mca_cutoff_unique_cin_count=1,
            duplicate_cin_date_rows=0,
            invalid_cin_rows=0,
            missing_announcement_date_rows=0,
            malformed_tsv_rows=0,
            provenance_complete=True,
            event_part=event_artifact,
            quarantine_part=quarantine_artifact,
        ),
    )
    return SolvencyObservationRunConfig(
        project_root=tmp_path,
        mca_report_path=mca_report_path,
        ibbi_report_path=ibbi_report_path,
        outcome_window_end=date(2026, 8, 21),
        temporal_holdout_start=date(2025, 1, 1),
    )


def test_solvency_observation_runner_publishes_address_cluster_shape(
    tmp_path: Path,
) -> None:
    config = _run_config(tmp_path)

    report = SolvencyObservationRunner(config).run()
    published_frame = pl.read_parquet(report.observation_artifact.path)

    assert report.schema_version == SOLVENCY_OBSERVATION_SCHEMA_VERSION
    assert f"schema-{SOLVENCY_OBSERVATION_SCHEMA_VERSION}" in (
        report.observation_artifact.path
    )
    assert set(SHAPE_COLUMNS) <= set(published_frame.columns)
    assert published_frame["shared_address_company_count"].to_list() == [2, 2, 1]
    assert published_frame["address_cluster_registration_span_days"].to_list() == [
        3,
        3,
        0,
    ]
    assert published_frame["address_cluster_max_month_share"].to_list() == [
        1.0,
        1.0,
        1.0,
    ]
    assert published_frame["address_cluster_authorised_capital_cv"].to_list() == [
        0.0,
        0.0,
        0.0,
    ]
    assert published_frame[
        "address_cluster_distinct_name_head_ratio"
    ].to_list() == [0.5, 0.5, 1.0]
    assert report.company_rows == 3
    assert report.evaluation_eligible_rows == 3
    assert report.observed_outcome_rows == 1
    assert report.minimum_positive_outcomes_per_slice == 20
    assert report.training_decision == "DO_NOT_TRAIN"


def test_solvency_observation_runner_run_id_is_reproducible(tmp_path: Path) -> None:
    config = _run_config(tmp_path)
    report = SolvencyObservationRunner(config).run()

    assert f"run-{report.run_id}" in report.observation_artifact.path
    with pytest.raises(SolvencyObservationRunError):
        SolvencyObservationRunner(config).run()
