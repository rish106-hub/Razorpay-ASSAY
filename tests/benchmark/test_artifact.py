from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from assay.benchmark.artifact import (
    TRANSACTION_VELOCITY_MODEL_FEATURES,
    TRANSACTION_VELOCITY_RAW_FEATURES,
    ControlledBenchmarkArtifactError,
    engineer_transaction_velocity_features,
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


def test_transaction_velocity_contract_builds_features_in_frozen_order() -> None:
    raw_frame = pl.DataFrame(
        {
            "TX_AMOUNT": [200.0],
            "TX_DURING_WEEKEND": [0],
            "TX_DURING_NIGHT": [1],
            "CUSTOMER_ID_NB_TX_1DAY_WINDOW": [2.0],
            "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW": [100.0],
            "CUSTOMER_ID_NB_TX_7DAY_WINDOW": [7.0],
            "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW": [80.0],
            "CUSTOMER_ID_NB_TX_30DAY_WINDOW": [30.0],
            "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW": [50.0],
            "TERMINAL_ID_NB_TX_1DAY_WINDOW": [3.0],
            "TERMINAL_ID_RISK_1DAY_WINDOW": [0.4],
            "TERMINAL_ID_NB_TX_7DAY_WINDOW": [14.0],
            "TERMINAL_ID_RISK_7DAY_WINDOW": [0.2],
            "TERMINAL_ID_NB_TX_30DAY_WINDOW": [30.0],
            "TERMINAL_ID_RISK_30DAY_WINDOW": [0.1],
        }
    )
    engineered_frame = engineer_transaction_velocity_features(
        raw_frame,
        raw_features=TRANSACTION_VELOCITY_RAW_FEATURES,
        expected_features=TRANSACTION_VELOCITY_MODEL_FEATURES,
    )

    assert tuple(engineered_frame.columns) == TRANSACTION_VELOCITY_MODEL_FEATURES
    assert engineered_frame["TERMINAL_RISK_MAX"].item() == pytest.approx(0.4)
    assert engineered_frame["AMOUNT_X_NIGHT"].item() == 200.0
    assert engineered_frame["CUSTOMER_AMOUNT_TO_AVG_1D"].item() == (
        pytest.approx(200.0 / 100.001)
    )


def test_transaction_velocity_contract_rejects_missing_raw_feature() -> None:
    with pytest.raises(
        ControlledBenchmarkArtifactError,
        match="missing: TX_AMOUNT",
    ):
        engineer_transaction_velocity_features(
            pl.DataFrame({"TX_DURING_NIGHT": [1.0]}),
            raw_features=("TX_AMOUNT", "TX_DURING_NIGHT"),
            expected_features=("TX_AMOUNT", "TX_DURING_NIGHT"),
        )


def test_transaction_velocity_contract_rejects_non_finite_value() -> None:
    with pytest.raises(
        ControlledBenchmarkArtifactError,
        match="non-finite",
    ):
        engineer_transaction_velocity_features(
            pl.DataFrame({"TX_AMOUNT": [float("inf")]}),
            raw_features=("TX_AMOUNT",),
            expected_features=("TX_AMOUNT",),
        )
