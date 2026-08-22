from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import BaseModel

from assay.artifacts.parquet import ParquetArtifact
from assay.canonicalisation.mca import McaCanonicalisationReport
from assay.canonicalisation.nse import NseCanonicalisationReport
from assay.linkage.entity_resolution import EntityLinkageReport
from assay.reporting.readiness import (
    EvidenceReadinessConfig,
    EvidenceReadinessReporter,
)
from assay.signals.runner import SignalObservationRunReport


def _write_report(path: Path, report: BaseModel) -> None:
    path.write_text(report.model_dump_json(), encoding="utf-8")


def _artifact(path: str) -> ParquetArtifact:
    return ParquetArtifact(path=path, rows=1, bytes=1, sha256="d" * 64)


def test_readiness_report_blocks_training_when_entity_outcomes_are_sparse(
    tmp_path: Path,
) -> None:
    mca_snapshot_id = "a" * 64
    nse_snapshot_id = "b" * 64
    linkage_run_id = "c" * 64
    observation_run_id = "e" * 64
    mca_report = McaCanonicalisationReport(
        source_snapshot_id=mca_snapshot_id,
        snapshot_as_of=date(2023, 11, 3),
        acquisition_complete=True,
        raw_page_count=368,
        raw_record_count=3_674_314,
        company_snapshot_rows=3_674_314,
        address_snapshot_rows=3_674_314,
        duplicate_legal_entity_identifier_rows=0,
        invalid_legal_entity_identifier_rows=59,
        invalid_legal_entity_identifier_rate=59 / 3_674_314,
        max_invalid_legal_entity_identifier_rate=0.0001,
        cin_rows=3_123_706,
        llpin_rows=545_206,
        fcrn_rows=5_343,
        missing_registration_date_rows=0,
        blank_address_rows=0,
        quality_status="passed_with_quarantine",
        staged_parts=(),
        company_parts=(),
        address_parts=(),
    )
    nse_report = NseCanonicalisationReport(
        source_snapshot_id=nse_snapshot_id,
        adverse_event_rows=16_165,
        sebi_event_rows=11_721,
        other_authority_event_rows=4_444,
        cin_candidate_rows=42,
        din_candidate_rows=3,
        pan_candidate_rows=13_215,
        missing_entity_name_rows=0,
        missing_event_date_rows=0,
        provenance_complete_assets=0,
        parts=(),
    )
    linkage_report = EntityLinkageReport(
        run_id=linkage_run_id,
        mca_source_snapshot_id=mca_snapshot_id,
        nse_source_snapshot_id=nse_snapshot_id,
        nse_provenance_complete_assets=0,
        adverse_event_rows=16_165,
        accepted_exact_cin_rows=40,
        pending_exact_name_review_rows=2_134,
        ambiguous_rows=58,
        unmatched_rows=13_933,
        identifier_collision_rows=0,
        persisted_candidate_rows=2_251,
        truncated_candidate_event_rows=0,
        outcome_eligible_rows=40,
        candidate_artifact=_artifact("candidate.parquet"),
        decision_artifact=_artifact("decision.parquet"),
    )
    observation_report = SignalObservationRunReport(
        run_id=observation_run_id,
        mca_source_snapshot_id=mca_snapshot_id,
        nse_source_snapshot_id=nse_snapshot_id,
        entity_linkage_run_id=linkage_run_id,
        feature_cutoff=date(2023, 11, 3),
        outcome_window_start=date(2023, 11, 4),
        outcome_window_end=date(2026, 8, 22),
        company_rows=3_674_314,
        evaluation_eligible_rows=2_920_186,
        observed_adverse_outcome_rows=8,
        excluded_prior_outcome_rows=32,
        shared_address_signal_rows=466_414,
        address_registration_month_cohort_signal_rows=174_025,
        nse_provenance_complete_assets=0,
        quality_status="passed",
        observation_artifact=_artifact("observations.parquet"),
    )
    paths = {
        "mca": tmp_path / "mca.json",
        "nse": tmp_path / "nse.json",
        "linkage": tmp_path / "linkage.json",
        "observation": tmp_path / "observation.json",
    }
    _write_report(paths["mca"], mca_report)
    _write_report(paths["nse"], nse_report)
    _write_report(paths["linkage"], linkage_report)
    _write_report(paths["observation"], observation_report)

    report = EvidenceReadinessReporter(
        EvidenceReadinessConfig(
            project_root=tmp_path,
            mca_report_path=paths["mca"],
            nse_report_path=paths["nse"],
            linkage_report_path=paths["linkage"],
            observation_report_path=paths["observation"],
            generated_root=tmp_path / "generated",
        )
    ).run()

    assert report.model_readiness.decision == "DO_NOT_TRAIN"
    assert report.model_readiness.positive_outcome_shortfall_before_holdouts == 12
    assert report.linkage_coverage.human_review_required is True
    assert report.signal_population.shared_address_signal_rate == (
        466_414 / 2_920_186
    )
    report_path = (
        tmp_path
        / "generated"
        / "schema-1.0.0"
        / f"report-{report.report_id}.json"
    )
    assert report_path.is_file()
