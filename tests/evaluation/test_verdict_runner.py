from __future__ import annotations

from datetime import date
from pathlib import Path

from assay.evaluation.runner import EvaluationRunReport
from assay.evaluation.verdict_runner import (
    VerdictPublicationConfig,
    VerdictPublisher,
)


def test_verdict_publisher_writes_do_not_ship_for_insufficient_evidence(
    tmp_path: Path,
) -> None:
    evaluation_report = EvaluationRunReport(
        run_id="a" * 64,
        observation_run_id="b" * 64,
        temporal_holdout_start=date(2026, 1, 1),
        geography_holdout_states=("Delhi",),
        development_states=("Maharashtra",),
        minimum_positive_outcomes_per_required_slice=20,
        signal_metrics=(),
        concentration_tests=(),
        required_slice_positive_outcomes={
            "overall": 8,
            "geography_holdout": 1,
            "temporal_holdout": 2,
        },
        evidence_status="insufficient_held_out_positive_outcomes",
    )
    evaluation_report_path = tmp_path / "evaluation.json"
    evaluation_report_path.write_text(
        evaluation_report.model_dump_json(),
        encoding="utf-8",
    )

    publication = VerdictPublisher(
        VerdictPublicationConfig(
            project_root=tmp_path,
            evaluation_report_path=evaluation_report_path,
            generated_root=tmp_path / "verdicts",
            false_positive_review_cost_budget_inr=100_000,
        )
    ).run()

    assert publication.verdict.verdict == "DO_NOT_SHIP"
    assert publication.verdict.reason_codes == (
        "insufficient_held_out_positive_outcomes",
    )
    assert Path(publication.report_path).is_file()
