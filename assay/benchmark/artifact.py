"""Verify and load the isolated Colab transaction-fraud benchmark artifact."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import sha256_file

CONTROLLED_BENCHMARK_CLAIM_BOUNDARY = "controlled_simulated_transaction_fraud_only"


class ControlledBenchmarkArtifactError(RuntimeError):
    """A controlled benchmark artifact failed provenance or model validation."""


class ControlledBenchmarkManifest(BaseModel):
    """Checksums and claim boundary exported by the Colab training run."""

    model_config = ConfigDict(frozen=True)

    benchmark_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    claim_boundary: str
    metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ControlledBenchmarkMetrics(BaseModel):
    """Minimum reproducibility fields required before model loading."""

    model_config = ConfigDict(frozen=True, extra="allow")

    benchmark_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    claim_boundary: str
    features: tuple[str, ...] = Field(min_length=1)
    best_iteration: int = Field(ge=0)
    seed: int = Field(ge=0)


class ControlledBenchmarkArtifactReport(BaseModel):
    """Validated local package state for audit logs and release checks."""

    model_config = ConfigDict(frozen=True)

    benchmark_revision: str
    claim_boundary: str
    feature_count: int = Field(gt=0)
    model_sha256: str
    metrics_sha256: str
    model_loaded: bool
    xgboost_version: str | None


class ControlledBenchmarkPackage:
    """Checksum-verified model package and its frozen feature contract."""

    def __init__(
        self,
        *,
        run_directory: Path,
        manifest: ControlledBenchmarkManifest,
        metrics: ControlledBenchmarkMetrics,
        booster: Any | None,
        xgboost_version: str | None,
    ) -> None:
        self.run_directory = run_directory
        self.manifest = manifest
        self.metrics = metrics
        self._booster = booster
        self.xgboost_version = xgboost_version

    @property
    def report(self) -> ControlledBenchmarkArtifactReport:
        return ControlledBenchmarkArtifactReport(
            benchmark_revision=self.manifest.benchmark_revision,
            claim_boundary=self.manifest.claim_boundary,
            feature_count=len(self.metrics.features),
            model_sha256=self.manifest.model_sha256,
            metrics_sha256=self.manifest.metrics_sha256,
            model_loaded=self._booster is not None,
            xgboost_version=self.xgboost_version,
        )

    def score(self, transaction_feature_frame: pl.DataFrame) -> pl.Series:
        """Return scores only; operating decisions remain outside this benchmark."""

        if self._booster is None:
            raise ControlledBenchmarkArtifactError(
                "The benchmark model was not loaded; scoring is unavailable."
            )
        expected_features = list(self.metrics.features)
        missing_features = set(expected_features) - set(
            transaction_feature_frame.columns
        )
        if missing_features:
            raise ControlledBenchmarkArtifactError(
                "Transaction feature frame is missing: "
                f"{', '.join(sorted(missing_features))}."
            )
        feature_frame = transaction_feature_frame.select(expected_features).cast(
            pl.Float32,
            strict=True,
        )
        if feature_frame.null_count().sum_horizontal().item() > 0:
            raise ControlledBenchmarkArtifactError(
                "Transaction feature frame contains null values."
            )
        scores = self._booster.inplace_predict(feature_frame.to_numpy())
        return pl.Series("controlled_transaction_fraud_score", scores)


def load_controlled_benchmark_package(
    run_directory: Path,
    *,
    load_model: bool = True,
) -> ControlledBenchmarkPackage:
    """Verify checksums and optionally load the XGBoost booster on CPU."""

    manifest_path = run_directory / "manifest.json"
    metrics_path = run_directory / "metrics.json"
    model_path = run_directory / "transaction_fraud_xgboost.json"
    for required_path in (manifest_path, metrics_path, model_path):
        if not required_path.is_file():
            raise ControlledBenchmarkArtifactError(
                f"Controlled benchmark artifact is missing: {required_path}."
            )
    try:
        manifest = ControlledBenchmarkManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        metrics = ControlledBenchmarkMetrics.model_validate_json(
            metrics_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark manifest or metrics are invalid."
        ) from error
    if manifest.claim_boundary != CONTROLLED_BENCHMARK_CLAIM_BOUNDARY:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark claim boundary is invalid."
        )
    if metrics.claim_boundary != manifest.claim_boundary:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark claim boundaries do not match."
        )
    if metrics.benchmark_revision != manifest.benchmark_revision:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark revisions do not match."
        )
    if sha256_file(metrics_path) != manifest.metrics_sha256:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark metrics checksum mismatch."
        )
    if sha256_file(model_path) != manifest.model_sha256:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark model checksum mismatch."
        )

    booster = None
    xgboost_version = None
    if load_model:
        try:
            xgboost = importlib.import_module("xgboost")
        except ImportError as error:
            raise ControlledBenchmarkArtifactError(
                "Install the gpu optional dependency to load the benchmark model."
            ) from error
        booster = xgboost.Booster()
        booster.load_model(model_path)
        booster.set_param({"device": "cpu"})
        if tuple(booster.feature_names or ()) != metrics.features:
            raise ControlledBenchmarkArtifactError(
                "Controlled benchmark model feature order does not match metrics."
            )
        xgboost_version = xgboost.__version__

    return ControlledBenchmarkPackage(
        run_directory=run_directory,
        manifest=manifest,
        metrics=metrics,
        booster=booster,
        xgboost_version=xgboost_version,
    )
