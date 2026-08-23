"""Verify and score the portable merchant-solvency model package."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from assay.artifacts.parquet import sha256_file
from assay.solvency.training_data import (
    SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_MODEL_FEATURES,
    SOLVENCY_NUMERIC_FEATURES,
)

SOLVENCY_MODEL_CLAIM_BOUNDARY = "cirp_public_announcement_outcome_not_fraud"
REQUIRED_SOLVENCY_ARTIFACTS = frozenset(
    {
        "portable_preprocessor.json",
        "selection_metrics.json",
        "solvency_xgboost.json",
    }
)


class SolvencyModelArtifactError(RuntimeError):
    """A solvency model artifact failed its frozen package contract."""


class SolvencyArtifactDigest(BaseModel):
    """Expected size and digest for one Colab-exported artifact."""

    model_config = ConfigDict(frozen=True)

    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SolvencySelectionWinner(BaseModel):
    """Validation-selected candidate recorded by the Colab run."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str = Field(min_length=1)
    best_iteration: int = Field(ge=0)
    validation_weighted_pr_auc: float = Field(ge=0, le=1)


class SolvencySelectionMetrics(BaseModel):
    """Minimum model-selection evidence required for package loading."""

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: str
    model_family: str
    label_boundary: str
    fit_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    random_seed: int = Field(ge=0)
    model_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    selection_rule: str
    winner: SolvencySelectionWinner


class PortableSolvencyPreprocessor(BaseModel):
    """Version-neutral median and one-hot encoding contract."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    numeric_features: tuple[str, ...]
    numeric_medians: tuple[float, ...]
    categorical_features: tuple[str, ...]
    categorical_imputer_values: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]
    infrequent_categories: tuple[tuple[str, ...], ...]
    handle_unknown: str
    min_frequency: int = Field(ge=1)
    transformed_feature_names: tuple[str, ...]
    label_boundary: str

    def transformed_feature_owners(self) -> tuple[str, ...]:
        """Map every transformed column back to the raw feature that owns it."""

        owners: list[str] = list(self.numeric_features)
        categorical_contract = zip(
            self.categorical_features,
            self.categories,
            self.infrequent_categories,
            strict=True,
        )
        for feature_name, categories, infrequent in categorical_contract:
            infrequent_set = set(infrequent)
            frequent_count = sum(
                1 for category in categories if category not in infrequent_set
            )
            owners.extend(
                [feature_name] * (frequent_count + bool(infrequent))
            )
        if len(owners) != len(self.transformed_feature_names):
            raise SolvencyModelArtifactError(
                "Portable preprocessing width does not match its feature names."
            )
        return tuple(owners)

    def transform(self, merchant_feature_frame: pl.DataFrame) -> sparse.csr_matrix:
        """Apply the exact Colab preprocessing without loading a pickle."""

        required_features = set(self.numeric_features) | set(
            self.categorical_features
        )
        missing_features = required_features - set(merchant_feature_frame.columns)
        if missing_features:
            raise SolvencyModelArtifactError(
                "Merchant feature frame is missing: "
                f"{', '.join(sorted(missing_features))}."
            )
        try:
            numeric_values = merchant_feature_frame.select(
                self.numeric_features
            ).cast(pl.Float32, strict=True).to_numpy()
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise SolvencyModelArtifactError(
                "Merchant numeric features contain non-numeric values."
            ) from error
        numeric_values = np.array(numeric_values, dtype=np.float32, copy=True)
        numeric_medians = np.asarray(self.numeric_medians, dtype=np.float32)
        if numeric_values.shape[1] != len(numeric_medians):
            raise SolvencyModelArtifactError(
                "Merchant numeric feature width does not match the model contract."
            )
        missing_rows, missing_columns = np.where(np.isnan(numeric_values))
        numeric_values[missing_rows, missing_columns] = numeric_medians[
            missing_columns
        ]
        if not np.isfinite(numeric_values).all():
            raise SolvencyModelArtifactError(
                "Merchant numeric features contain infinite values."
            )

        blocks: list[sparse.csr_matrix] = [sparse.csr_matrix(numeric_values)]
        categorical_contract = zip(
            self.categorical_features,
            self.categorical_imputer_values,
            self.categories,
            self.infrequent_categories,
            strict=True,
        )
        for feature_name, imputer_value, categories, infrequent in (
            categorical_contract
        ):
            values = [
                imputer_value if value is None else str(value)
                for value in merchant_feature_frame[feature_name].to_list()
            ]
            infrequent_set = set(infrequent)
            frequent_categories = [
                category
                for category in categories
                if category not in infrequent_set
            ]
            category_columns = {
                category: index
                for index, category in enumerate(frequent_categories)
            }
            width = len(frequent_categories) + bool(infrequent)
            encoded_rows: list[int] = []
            encoded_columns: list[int] = []
            for row_index, value in enumerate(values):
                column_index = category_columns.get(value)
                if column_index is None and value in infrequent_set:
                    column_index = width - 1
                if column_index is not None:
                    encoded_rows.append(row_index)
                    encoded_columns.append(column_index)
            blocks.append(
                sparse.csr_matrix(
                    (
                        np.ones(len(encoded_rows), dtype=np.float32),
                        (encoded_rows, encoded_columns),
                    ),
                    shape=(merchant_feature_frame.height, width),
                )
            )
        transformed = sparse.hstack(blocks, format="csr")
        if transformed.shape[1] != len(self.transformed_feature_names):
            raise SolvencyModelArtifactError(
                "Portable preprocessing width does not match its feature names."
            )
        return cast(sparse.csr_matrix, transformed)


class SolvencyModelArtifactReport(BaseModel):
    """Validated package evidence for audit and release checks."""

    model_config = ConfigDict(frozen=True)

    schema_version: str
    label_boundary: str
    model_family: str
    selected_candidate: str
    validation_weighted_pr_auc: float
    raw_feature_count: int = Field(gt=0)
    transformed_feature_count: int = Field(gt=0)
    model_sha256: str
    portable_preprocessor_sha256: str
    selection_metrics_sha256: str
    model_loaded: bool
    xgboost_version: str | None


class SolvencyModelPackage:
    """Checksum-verified solvency model and portable feature transformer."""

    def __init__(
        self,
        *,
        run_directory: Path,
        manifest: dict[str, SolvencyArtifactDigest],
        metrics: SolvencySelectionMetrics,
        preprocessor: PortableSolvencyPreprocessor,
        booster: Any | None,
        xgboost_version: str | None,
    ) -> None:
        self.run_directory = run_directory
        self.manifest = manifest
        self.metrics = metrics
        self.preprocessor = preprocessor
        self._booster = booster
        self.xgboost_version = xgboost_version

    @property
    def report(self) -> SolvencyModelArtifactReport:
        return SolvencyModelArtifactReport(
            schema_version=self.metrics.schema_version,
            label_boundary=self.metrics.label_boundary,
            model_family=self.metrics.model_family,
            selected_candidate=self.metrics.winner.name,
            validation_weighted_pr_auc=(
                self.metrics.winner.validation_weighted_pr_auc
            ),
            raw_feature_count=len(self.metrics.model_features),
            transformed_feature_count=len(
                self.preprocessor.transformed_feature_names
            ),
            model_sha256=self.manifest["solvency_xgboost.json"].sha256,
            portable_preprocessor_sha256=self.manifest[
                "portable_preprocessor.json"
            ].sha256,
            selection_metrics_sha256=self.manifest[
                "selection_metrics.json"
            ].sha256,
            model_loaded=self._booster is not None,
            xgboost_version=self.xgboost_version,
        )

    def score(self, merchant_feature_frame: pl.DataFrame) -> pl.Series:
        """Return a CIRP-announcement score without making an action decision."""

        if self._booster is None:
            raise SolvencyModelArtifactError(
                "The solvency model was not loaded; scoring is unavailable."
            )
        transformed = self.preprocessor.transform(merchant_feature_frame)
        scores = self._booster.inplace_predict(
            transformed,
            iteration_range=(0, self.metrics.winner.best_iteration + 1),
        )
        score_series = pl.Series(
            "cirp_public_announcement_score",
            np.asarray(scores, dtype=np.float64),
        )
        if not score_series.is_finite().all() or not score_series.is_between(
            0.0,
            1.0,
            closed="both",
        ).all():
            raise SolvencyModelArtifactError(
                "Solvency model produced an invalid probability."
            )
        return score_series


    def score_with_contributions(
        self,
        merchant_feature_frame: pl.DataFrame,
    ) -> tuple[pl.Series, pl.DataFrame]:
        """Return scores plus per-raw-feature log-odds contributions.

        Contributions are exact tree SHAP values from the same iteration range
        the selected classifier used. They explain the model, not the merchant.
        """

        if self._booster is None:
            raise SolvencyModelArtifactError(
                "The solvency model was not loaded; scoring is unavailable."
            )
        scores = self.score(merchant_feature_frame)
        xgboost = importlib.import_module("xgboost")
        transformed = self.preprocessor.transform(merchant_feature_frame)
        raw_contributions = self._booster.predict(
            xgboost.DMatrix(transformed),
            pred_contribs=True,
            iteration_range=(0, self.metrics.winner.best_iteration + 1),
        )
        contribution_matrix = np.asarray(raw_contributions, dtype=np.float64)
        owners = self.preprocessor.transformed_feature_owners()
        if contribution_matrix.shape[1] != len(owners) + 1:
            raise SolvencyModelArtifactError(
                "Solvency contribution width does not match the feature contract."
            )
        owner_array = np.asarray(owners)
        contribution_columns = {
            feature_name: pl.Series(
                feature_name,
                contribution_matrix[:, : len(owners)][
                    :, owner_array == feature_name
                ].sum(axis=1),
            )
            for feature_name in self.metrics.model_features
        }
        contribution_columns["model_bias"] = pl.Series(
            "model_bias", contribution_matrix[:, -1]
        )
        return scores, pl.DataFrame(contribution_columns)


def _load_json_model(path: Path, model: type[BaseModel]) -> BaseModel:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SolvencyModelArtifactError(
            f"Solvency artifact is invalid: {path}."
        ) from error


def load_solvency_model_package(
    run_directory: Path,
    *,
    load_model: bool = True,
) -> SolvencyModelPackage:
    """Verify the Colab export and optionally load its XGBoost booster."""

    manifest_path = run_directory / "manifest.json"
    metrics_path = run_directory / "selection_metrics.json"
    preprocessor_path = run_directory / "portable_preprocessor.json"
    model_path = run_directory / "solvency_xgboost.json"
    for required_path in (
        manifest_path,
        metrics_path,
        preprocessor_path,
        model_path,
    ):
        if not required_path.is_file():
            raise SolvencyModelArtifactError(
                f"Solvency model artifact is missing: {required_path}."
            )
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {
            name: SolvencyArtifactDigest.model_validate(digest)
            for name, digest in raw_manifest.items()
        }
    except (OSError, ValueError, TypeError) as error:
        raise SolvencyModelArtifactError(
            "Solvency model manifest is invalid."
        ) from error
    if not REQUIRED_SOLVENCY_ARTIFACTS.issubset(manifest):
        raise SolvencyModelArtifactError(
            "Solvency model manifest is missing required artifact digests."
        )
    for artifact_name, digest in manifest.items():
        artifact_path = run_directory / artifact_name
        if not artifact_path.is_file():
            raise SolvencyModelArtifactError(
                f"Manifest artifact is missing: {artifact_path}."
            )
        if artifact_path.stat().st_size != digest.bytes:
            raise SolvencyModelArtifactError(
                f"Solvency artifact size mismatch: {artifact_name}."
            )
        if sha256_file(artifact_path) != digest.sha256:
            raise SolvencyModelArtifactError(
                f"Solvency artifact checksum mismatch: {artifact_name}."
            )

    metrics = _load_json_model(
        metrics_path,
        SolvencySelectionMetrics,
    )
    preprocessor = _load_json_model(
        preprocessor_path,
        PortableSolvencyPreprocessor,
    )
    assert isinstance(metrics, SolvencySelectionMetrics)
    assert isinstance(preprocessor, PortableSolvencyPreprocessor)
    if metrics.label_boundary != SOLVENCY_MODEL_CLAIM_BOUNDARY:
        raise SolvencyModelArtifactError("Solvency claim boundary is invalid.")
    if preprocessor.label_boundary != metrics.label_boundary:
        raise SolvencyModelArtifactError(
            "Solvency preprocessing and selection claim boundaries differ."
        )
    if metrics.model_features != SOLVENCY_MODEL_FEATURES:
        raise SolvencyModelArtifactError(
            "Solvency model feature contract is invalid."
        )
    if metrics.numeric_features != SOLVENCY_NUMERIC_FEATURES:
        raise SolvencyModelArtifactError(
            "Solvency numeric feature contract is invalid."
        )
    if metrics.categorical_features != SOLVENCY_CATEGORICAL_FEATURES:
        raise SolvencyModelArtifactError(
            "Solvency categorical feature contract is invalid."
        )
    if preprocessor.numeric_features != metrics.numeric_features:
        raise SolvencyModelArtifactError(
            "Solvency numeric preprocessing contract differs from selection."
        )
    if preprocessor.categorical_features != metrics.categorical_features:
        raise SolvencyModelArtifactError(
            "Solvency categorical preprocessing contract differs from selection."
        )

    booster = None
    xgboost_version = None
    if load_model:
        try:
            xgboost = importlib.import_module("xgboost")
        except ImportError as error:
            raise SolvencyModelArtifactError(
                "Install the gpu optional dependency to load the solvency model."
            ) from error
        booster = xgboost.Booster()
        booster.load_model(model_path)
        booster.set_param({"device": "cpu"})
        if booster.num_features() != len(
            preprocessor.transformed_feature_names
        ):
            raise SolvencyModelArtifactError(
                "Solvency booster width does not match portable preprocessing."
            )
        xgboost_version = xgboost.__version__
    return SolvencyModelPackage(
        run_directory=run_directory,
        manifest=manifest,
        metrics=metrics,
        preprocessor=preprocessor,
        booster=booster,
        xgboost_version=xgboost_version,
    )
