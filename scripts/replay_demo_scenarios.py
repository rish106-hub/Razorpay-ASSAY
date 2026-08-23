"""Replay the curated demo scenarios against the real serving artifacts.

The committed scenario file records what the service returned when it was
curated. This script proves the frozen index, model package, and band evidence
still reproduce those exact answers. It needs the local generated artifacts, so
it is a reproduction check, not part of the portable test suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from assay.api.app import create_app

SCORE_TOLERANCE = 1e-8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--evidence-report", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--index-report", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    arguments = parser.parse_args()

    scenarios = json.loads(arguments.scenarios.read_text(encoding="utf-8"))[
        "scenarios"
    ]
    application = create_app(
        report_path=arguments.evidence_report,
        solvency_evaluation_path=arguments.evaluation_report,
        merchant_index_report_path=arguments.index_report,
        solvency_model_directory=arguments.model_directory,
    )
    failures = 0
    with TestClient(application) as client:
        for scenario in scenarios:
            response = client.post("/v1/risk/assess", json=scenario["request"])
            body = response.json()
            matched = response.status_code == scenario["expected_http_status"]
            if scenario["expected_risk_band"] is not None:
                matched = (
                    matched
                    and body.get("risk_band") == scenario["expected_risk_band"]
                    and abs(
                        body["cirp_public_announcement_score"]
                        - scenario["observed_score"]
                    )
                    < SCORE_TOLERANCE
                )
            else:
                matched = (
                    matched
                    and body.get("outcome") == scenario["expected_outcome"]
                )
            failures += not matched
            verdict = "PASS" if matched else "FAIL"
            observed = body.get("risk_band") or body.get("outcome")
            print(
                f"{verdict} {scenario['scenario_id']:<26} "
                f"http={response.status_code} observed={observed}"
            )
    print(f"scenarios={len(scenarios)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
