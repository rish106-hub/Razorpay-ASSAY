"""Verify and load the isolated Colab transaction-fraud benchmark artifact."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import sha256_file

CONTROLLED_BENCHMARK_CLAIM_BOUNDARY = "controlled_simulated_transaction_fraud_only"
RAW_FEATURE_CONTRACT_VERSION = "raw_v1"
TRANSACTION_VELOCITY_FEATURE_CONTRACT_VERSION = "transaction_velocity_v2"
SUPPORTED_FEATURE_CONTRACT_VERSIONS = frozenset(
    {
        RAW_FEATURE_CONTRACT_VERSION,
        TRANSACTION_VELOCITY_FEATURE_CONTRACT_VERSION,
    }
)
TRANSACTION_VELOCITY_RAW_FEATURES = (
    "TX_AMOUNT",
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",
    "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",
    "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",
    "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",
    "TERMINAL_ID_NB_TX_1DAY_WINDOW",
    "TERMINAL_ID_RISK_1DAY_WINDOW",
    "TERMINAL_ID_NB_TX_7DAY_WINDOW",
    "TERMINAL_ID_RISK_7DAY_WINDOW",
    "TERMINAL_ID_NB_TX_30DAY_WINDOW",
    "TERMINAL_ID_RISK_30DAY_WINDOW",
)
TRANSACTION_VELOCITY_ENGINEERED_FEATURES = (
    "CUSTOMER_AMOUNT_TO_AVG_1D",
    "CUSTOMER_AMOUNT_TO_AVG_7D",
    "CUSTOMER_AMOUNT_TO_AVG_30D",
    "CUSTOMER_TX_ACCEL_1D_TO_7D",
    "CUSTOMER_TX_ACCEL_7D_TO_30D",
    "TERMINAL_TX_ACCEL_1D_TO_7D",
    "TERMINAL_TX_ACCEL_7D_TO_30D",
    "TERMINAL_RISK_MAX",
    "TERMINAL_RISK_SLOPE_1D_30D",
    "AMOUNT_X_TERMINAL_RISK_7D",
    "AMOUNT_X_NIGHT",
)
TRANSACTION_VELOCITY_MODEL_FEATURES = (
    *TRANSACTION_VELOCITY_RAW_FEATURES,
    *TRANSACTION_VELOCITY_ENGINEERED_FEATURES,
)


class ControlledBenchmarkArtifactError(RuntimeError):
    """A controlled benchmark artifact failed provenance or model validation."""


class ControlledBenchmarkManifest(BaseModel):
    """Checksums and claim boundary exported by the Colab training run."""

    model_config = ConfigDict(frozen=True)

    benchmark_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_version: str = "1.0.0"
    claim_boundary: str
    feature_contract_version: str = RAW_FEATURE_CONTRACT_VERSION
    metrics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ControlledBenchmarkMetrics(BaseModel):
    """Minimum reproducibility fields required before model loading."""

    model_config = ConfigDict(frozen=True, extra="allow")

    benchmark_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_version: str = "1.0.0"
    claim_boundary: str
    feature_contract_version: str = RAW_FEATURE_CONTRACT_VERSION
    raw_features: tuple[str, ...] | None = None
    features: tuple[str, ...] = Field(min_length=1)
    best_iteration: int = Field(ge=0)
    seed: int = Field(ge=0)


class ControlledBenchmarkArtifactReport(BaseModel):
    """Validated local package state for audit logs and release checks."""

    model_config = ConfigDict(frozen=True)

    benchmark_revision: str
    artifact_version: str
    claim_boundary: str
    feature_contract_version: str
    raw_feature_count: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    model_sha256: str
    metrics_sha256: str
    model_loaded: bool
    xgboost_version: str | None


def _require_finite_features(feature_frame: pl.DataFrame) -> None:
    if feature_frame.null_count().sum_horizontal().item() > 0:
        raise ControlledBenchmarkArtifactError(
            "Transaction feature frame contains null values."
        )
    finite_columns = feature_frame.select(
        pl.all().is_finite().all(),
    ).row(0)
    if not all(finite_columns):
        raise ControlledBenchmarkArtifactError(
            "Transaction feature frame contains non-finite values."
        )


def engineer_transaction_velocity_features(
    transaction_feature_frame: pl.DataFrame,
    *,
    raw_features: tuple[str, ...],
    expected_features: tuple[str, ...],
) -> pl.DataFrame:
    """Apply the exact raw-to-model feature contract selected in Colab."""

    missing_features = set(raw_features) - set(transaction_feature_frame.columns)
    if missing_features:
        raise ControlledBenchmarkArtifactError(
            "Transaction feature frame is missing: "
            f"{', '.join(sorted(missing_features))}."
        )
    try:
        raw_feature_frame = transaction_feature_frame.select(raw_features).cast(
            pl.Float32,
            strict=True,
        )
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise ControlledBenchmarkArtifactError(
            "Transaction feature frame contains non-numeric values."
        ) from error
    _require_finite_features(raw_feature_frame)
    epsilon = 1e-3
    engineered_frame = raw_feature_frame.with_columns(
        (
            pl.col("TX_AMOUNT")
            / (pl.col("CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW") + epsilon)
        ).alias("CUSTOMER_AMOUNT_TO_AVG_1D"),
        (
            pl.col("TX_AMOUNT")
            / (pl.col("CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW") + epsilon)
        ).alias("CUSTOMER_AMOUNT_TO_AVG_7D"),
        (
            pl.col("TX_AMOUNT")
            / (pl.col("CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW") + epsilon)
        ).alias("CUSTOMER_AMOUNT_TO_AVG_30D"),
        (
            pl.col("CUSTOMER_ID_NB_TX_1DAY_WINDOW")
            / (pl.col("CUSTOMER_ID_NB_TX_7DAY_WINDOW") / 7.0 + epsilon)
        ).alias("CUSTOMER_TX_ACCEL_1D_TO_7D"),
        (
            pl.col("CUSTOMER_ID_NB_TX_7DAY_WINDOW")
            / (
                pl.col("CUSTOMER_ID_NB_TX_30DAY_WINDOW") * 7.0 / 30.0
                + epsilon
            )
        ).alias("CUSTOMER_TX_ACCEL_7D_TO_30D"),
        (
            pl.col("TERMINAL_ID_NB_TX_1DAY_WINDOW")
            / (pl.col("TERMINAL_ID_NB_TX_7DAY_WINDOW") / 7.0 + epsilon)
        ).alias("TERMINAL_TX_ACCEL_1D_TO_7D"),
        (
            pl.col("TERMINAL_ID_NB_TX_7DAY_WINDOW")
            / (
                pl.col("TERMINAL_ID_NB_TX_30DAY_WINDOW") * 7.0 / 30.0
                + epsilon
            )
        ).alias("TERMINAL_TX_ACCEL_7D_TO_30D"),
        pl.max_horizontal(
            "TERMINAL_ID_RISK_1DAY_WINDOW",
            "TERMINAL_ID_RISK_7DAY_WINDOW",
            "TERMINAL_ID_RISK_30DAY_WINDOW",
        ).alias("TERMINAL_RISK_MAX"),
        (
            pl.col("TERMINAL_ID_RISK_1DAY_WINDOW")
            - pl.col("TERMINAL_ID_RISK_30DAY_WINDOW")
        ).alias("TERMINAL_RISK_SLOPE_1D_30D"),
        (
            pl.col("TX_AMOUNT") * pl.col("TERMINAL_ID_RISK_7DAY_WINDOW")
        ).alias("AMOUNT_X_TERMINAL_RISK_7D"),
        (pl.col("TX_AMOUNT") * pl.col("TX_DURING_NIGHT")).alias(
            "AMOUNT_X_NIGHT"
        ),
    )
    missing_engineered_features = set(expected_features) - set(
        engineered_frame.columns
    )
    if missing_engineered_features:
        raise ControlledBenchmarkArtifactError(
            "Unsupported engineered feature contract fields: "
            f"{', '.join(sorted(missing_engineered_features))}."
        )
    feature_frame = engineered_frame.select(expected_features).cast(pl.Float32)
    _require_finite_features(feature_frame)
    return feature_frame


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
            artifact_version=self.manifest.artifact_version,
            claim_boundary=self.manifest.claim_boundary,
            feature_contract_version=self.manifest.feature_contract_version,
            raw_feature_count=len(
                self.metrics.raw_features or self.metrics.features
            ),
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
        if (
            self.metrics.feature_contract_version
            == TRANSACTION_VELOCITY_FEATURE_CONTRACT_VERSION
        ):
            if self.metrics.raw_features is None:
                raise ControlledBenchmarkArtifactError(
                    "The transaction-velocity contract is missing raw features."
                )
            feature_frame = engineer_transaction_velocity_features(
                transaction_feature_frame,
                raw_features=self.metrics.raw_features,
                expected_features=self.metrics.features,
            )
        else:
            expected_features = list(self.metrics.features)
            missing_features = set(expected_features) - set(
                transaction_feature_frame.columns
            )
            if missing_features:
                raise ControlledBenchmarkArtifactError(
                    "Transaction feature frame is missing: "
                    f"{', '.join(sorted(missing_features))}."
                )
            try:
                feature_frame = transaction_feature_frame.select(
                    expected_features
                ).cast(pl.Float32, strict=True)
            except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
                raise ControlledBenchmarkArtifactError(
                    "Transaction feature frame contains non-numeric values."
                ) from error
            _require_finite_features(feature_frame)
        scores = self._booster.inplace_predict(feature_frame.to_numpy())
        score_series = pl.Series("controlled_transaction_fraud_score", scores)
        if not score_series.is_finite().all() or not score_series.is_between(
            0.0,
            1.0,
            closed="both",
        ).all():
            raise ControlledBenchmarkArtifactError(
                "Controlled benchmark produced an invalid probability."
            )
        return score_series


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
    if metrics.artifact_version != manifest.artifact_version:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark artifact versions do not match."
        )
    if metrics.feature_contract_version != manifest.feature_contract_version:
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark feature contracts do not match."
        )
    if (
        metrics.feature_contract_version
        not in SUPPORTED_FEATURE_CONTRACT_VERSIONS
    ):
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark feature contract is unsupported."
        )
    if (
        metrics.feature_contract_version
        == TRANSACTION_VELOCITY_FEATURE_CONTRACT_VERSION
        and metrics.raw_features != TRANSACTION_VELOCITY_RAW_FEATURES
    ):
        raise ControlledBenchmarkArtifactError(
            "The transaction-velocity raw feature contract is invalid."
        )
    if (
        metrics.feature_contract_version
        == TRANSACTION_VELOCITY_FEATURE_CONTRACT_VERSION
        and metrics.features != TRANSACTION_VELOCITY_MODEL_FEATURES
    ):
        raise ControlledBenchmarkArtifactError(
            "The transaction-velocity model feature contract is invalid."
        )
    if len(set(metrics.features)) != len(metrics.features):
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark model features must be unique."
        )
    if metrics.raw_features and len(set(metrics.raw_features)) != len(
        metrics.raw_features
    ):
        raise ControlledBenchmarkArtifactError(
            "Controlled benchmark raw features must be unique."
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
