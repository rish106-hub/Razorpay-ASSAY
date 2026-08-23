"""Publish the curated demo scenario set for the merchant-risk endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.serving.assessment import MerchantAssessmentError, MerchantRiskAssessor
from assay.serving.bands import RiskBandError, RiskBandScale
from assay.serving.demo import DemoScenarioError, curate_demo_scenarios
from assay.serving.lookup import MerchantLookupError, MerchantRiskIndexStore
from assay.solvency.artifact import (
    SolvencyModelArtifactError,
    load_solvency_model_package,
)
from assay.solvency.evaluation import SolvencyHoldoutEvaluationReport


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        evaluation = SolvencyHoldoutEvaluationReport.model_validate_json(
            arguments.evaluation_report.read_text(encoding="utf-8")
        )
        scenario_set = curate_demo_scenarios(
            MerchantRiskAssessor(
                store=MerchantRiskIndexStore.from_report_path(
                    arguments.index_report
                ),
                package=load_solvency_model_package(arguments.model_directory),
                band_scale=RiskBandScale.from_evaluation(evaluation),
                evaluation=evaluation,
            )
        )
    except (
        DemoScenarioError,
        ImmutableArtifactError,
        MerchantAssessmentError,
        MerchantLookupError,
        OSError,
        RiskBandError,
        SolvencyModelArtifactError,
        ValueError,
    ) as error:
        print(f"Demo scenario build failed: {error}", file=sys.stderr)
        return 1
    payload = json.dumps(
        scenario_set.model_dump(mode="json"), indent=2, sort_keys=True
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as output_file:
        output_file.write(payload + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
