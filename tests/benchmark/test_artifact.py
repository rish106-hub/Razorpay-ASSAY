from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from assay.benchmark.artifact import (
    ControlledBenchmarkArtifactError,
    load_controlled_benchmark_package,
)


def _write_package(run_directory: Path) -> None:
    run_directory.mkdir()
    model_bytes = b"model-placeholder"
    metrics_payload = {
        "benchmark_revision": "a" * 40,
        "claim_boundary": "controlled_simulated_transaction_fraud_only",
        "features": ["TX_AMOUNT", "TX_DURING_NIGHT"],
        "best_iteration": 12,
        "seed": 106,
    }
    metrics_bytes = json.dumps(metrics_payload, separators=(",", ":")).encode()
    (run_directory / "transaction_fraud_xgboost.json").write_bytes(model_bytes)
    (run_directory / "metrics.json").write_bytes(metrics_bytes)
    manifest_payload = {
        "benchmark_revision": "a" * 40,
        "claim_boundary": "controlled_simulated_transaction_fraud_only",
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )


def test_package_verifies_checksums_without_loading_optional_model(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    _write_package(run_directory)

    package = load_controlled_benchmark_package(
        run_directory,
        load_model=False,
    )

    assert package.report.model_loaded is False
    assert package.report.feature_count == 2
    assert package.report.claim_boundary == (
        "controlled_simulated_transaction_fraud_only"
    )


def test_package_rejects_changed_model_bytes(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    _write_package(run_directory)
    (run_directory / "transaction_fraud_xgboost.json").write_bytes(b"changed")

    with pytest.raises(
        ControlledBenchmarkArtifactError,
        match="model checksum mismatch",
    ):
        load_controlled_benchmark_package(run_directory, load_model=False)
