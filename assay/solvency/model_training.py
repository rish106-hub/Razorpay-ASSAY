"""Compute-bounded Colab training for the public merchant-solvency baseline."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from assay.artifacts.parquet import sha256_file
from assay.solvency.artifact import SOLVENCY_MODEL_CLAIM_BOUNDARY
from assay.solvency.training_data import (
    SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_MODEL_FEATURES,
    SOLVENCY_NUMERIC_FEATURES,
)


class SolvencyModelTrainingError(RuntimeError):
    """A Colab fit artifact or candidate run broke its frozen contract."""


class SolvencyCandidateConfig(BaseModel):
    """One bounded XGBoost candidate declared before validation scoring."""

    model_config = ConfigDict(frozen=True)
    name: str
    max_depth: int = Field(ge=1, le=12)
    min_child_weight: int = Field(ge=1)


class SolvencyModelTrainingConfig(BaseModel):
    """Frozen sample restoration and candidate-search configuration."""

    model_config = ConfigDict(frozen=True)
    fit_data_path: Path
    output_directory: Path
    expected_fit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    random_seed: int = 106
    full_training_negative_rows: int = 1_385_206
    full_validation_negative_rows: int = 297_516
    sampled_training_negative_rows: int = 48_200
    sampled_validation_negative_rows: int = 46_600
    expected_training_positive_rows: int = 964
    expected_validation_positive_rows: int = 233
    device: str = "cuda"
    candidates: tuple[SolvencyCandidateConfig, ...] = (
        SolvencyCandidateConfig(
            name="depth_4_regularised", max_depth=4, min_child_weight=5
        ),
        SolvencyCandidateConfig(
            name="depth_6_regularised", max_depth=6, min_child_weight=5
        ),
        SolvencyCandidateConfig(
            name="depth_8_regularised", max_depth=8, min_child_weight=10
        ),
        SolvencyCandidateConfig(
            name="depth_6_conservative", max_depth=6, min_child_weight=20
        ),
    )


def build_population_restoration_weights(
    targets: np.ndarray,
    *,
    full_negative_rows: int,
    sampled_negative_rows: int,
) -> np.ndarray:
    """Restore sampled controls to their full-split population mass."""

    if sampled_negative_rows <= 0 or full_negative_rows < sampled_negative_rows:
        raise SolvencyModelTrainingError(
            "Negative sampling counts cannot restore the full population."
        )
    negative_weight = full_negative_rows / sampled_negative_rows
    return np.where(targets == 1, 1.0, negative_weight).astype(np.float64)


def build_solvency_preprocessor() -> ColumnTransformer:
    """Create the frozen numeric and categorical feature transformer."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(SOLVENCY_NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                list(SOLVENCY_CATEGORICAL_FEATURES),
            ),
        ],
        sparse_threshold=0.1,
    )


def portable_preprocessor_payload(preprocessor: ColumnTransformer) -> dict[str, Any]:
    """Export sklearn preprocessing as a version-neutral JSON contract."""

    numeric_imputer = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    if not isinstance(numeric_imputer, SimpleImputer) or not isinstance(
        categorical_pipeline, Pipeline
    ):
        raise SolvencyModelTrainingError("Unexpected solvency preprocessor shape.")
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    categorical_encoder = categorical_pipeline.named_steps["encoder"]
    if not isinstance(categorical_imputer, SimpleImputer) or not isinstance(
        categorical_encoder, OneHotEncoder
    ):
        raise SolvencyModelTrainingError(
            "Unexpected solvency categorical preprocessor shape."
        )
    return {
        "schema_version": "1.0.0",
        "numeric_features": SOLVENCY_NUMERIC_FEATURES,
        "numeric_medians": [
            float(value) for value in numeric_imputer.statistics_
        ],
        "categorical_features": SOLVENCY_CATEGORICAL_FEATURES,
        "categorical_imputer_values": [
            str(value) for value in categorical_imputer.statistics_
        ],
        "categories": [
            [str(value) for value in values.tolist()]
            for values in categorical_encoder.categories_
        ],
        "infrequent_categories": [
            [] if values is None else [str(value) for value in values.tolist()]
            for values in categorical_encoder.infrequent_categories_
        ],
        "handle_unknown": categorical_encoder.handle_unknown,
        "min_frequency": categorical_encoder.min_frequency,
        "transformed_feature_names": [
            str(value) for value in preprocessor.get_feature_names_out().tolist()
        ],
        "label_boundary": SOLVENCY_MODEL_CLAIM_BOUNDARY,
    }


def _weighted_capacity_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    capacity_fraction: float,
) -> dict[str, float | int]:
    order = np.argsort(-scores, kind="stable")
    cutoff = capacity_fraction * float(weights.sum())
    selected_rows = int(
        np.searchsorted(np.cumsum(weights[order]), cutoff, side="left") + 1
    )
    selected_indices = order[:selected_rows]
    selected_mass = float(weights[selected_indices].sum())
    selected_positive_mass = float(
        np.sum(weights[selected_indices] * targets[selected_indices])
    )
    total_positive_mass = float(np.sum(weights * targets))
    precision = selected_positive_mass / selected_mass
    base_rate = total_positive_mass / float(weights.sum())
    return {
        "capacity_fraction": capacity_fraction,
        "selected_sample_rows": selected_rows,
        "selected_population_mass": selected_mass,
        "precision": precision,
        "recall": selected_positive_mass / total_positive_mass,
        "lift": precision / base_rate,
    }


def train_solvency_model(config: SolvencyModelTrainingConfig) -> dict[str, Any]:
    """Fit candidates, select on validation only, and export checksummed bytes."""

    if sha256_file(config.fit_data_path) != config.expected_fit_sha256:
        raise SolvencyModelTrainingError("Solvency fit-data checksum mismatch.")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    frame = pd.read_parquet(config.fit_data_path)
    training = frame.loc[frame["dataset_split"].eq("training")].copy()
    validation = frame.loc[frame["dataset_split"].eq("validation")].copy()
    if int(training["target"].sum()) != config.expected_training_positive_rows:
        raise SolvencyModelTrainingError("Unexpected training-positive count.")
    if int(validation["target"].sum()) != config.expected_validation_positive_rows:
        raise SolvencyModelTrainingError("Unexpected validation-positive count.")
    preprocessor = build_solvency_preprocessor()
    training_features = preprocessor.fit_transform(
        training[list(SOLVENCY_MODEL_FEATURES)]
    )
    validation_features = preprocessor.transform(
        validation[list(SOLVENCY_MODEL_FEATURES)]
    )
    training_targets = training["target"].to_numpy(dtype=np.int8)
    validation_targets = validation["target"].to_numpy(dtype=np.int8)
    training_weights = build_population_restoration_weights(
        training_targets,
        full_negative_rows=config.full_training_negative_rows,
        sampled_negative_rows=config.sampled_training_negative_rows,
    )
    validation_weights = build_population_restoration_weights(
        validation_targets,
        full_negative_rows=config.full_validation_negative_rows,
        sampled_negative_rows=config.sampled_validation_negative_rows,
    )
    try:
        xgboost = importlib.import_module("xgboost")
    except ImportError as error:
        raise SolvencyModelTrainingError(
            "Install the gpu optional dependency before model training."
        ) from error
    candidate_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}
    for candidate in config.candidates:
        started = time.time()
        model = xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device=config.device,
            n_estimators=1400,
            learning_rate=0.035,
            max_depth=candidate.max_depth,
            min_child_weight=candidate.min_child_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.25,
            reg_lambda=12.0,
            max_delta_step=1.0,
            random_state=config.random_seed,
            n_jobs=2,
            early_stopping_rounds=80,
        )
        model.fit(
            training_features,
            training_targets,
            sample_weight=training_weights,
            eval_set=[(validation_features, validation_targets)],
            sample_weight_eval_set=[validation_weights],
            verbose=False,
        )
        scores = model.predict_proba(validation_features)[:, 1]
        candidate_results.append(
            {
                **candidate.model_dump(),
                "best_iteration": int(model.best_iteration),
                "validation_weighted_pr_auc": float(
                    average_precision_score(
                        validation_targets,
                        scores,
                        sample_weight=validation_weights,
                    )
                ),
                "validation_weighted_roc_auc": float(
                    roc_auc_score(
                        validation_targets,
                        scores,
                        sample_weight=validation_weights,
                    )
                ),
                "validation_weighted_brier": float(
                    brier_score_loss(
                        validation_targets,
                        scores,
                        sample_weight=validation_weights,
                    )
                ),
                "review_capacity_0_1pct": _weighted_capacity_metrics(
                    validation_targets, scores, validation_weights, 0.001
                ),
                "elapsed_seconds": time.time() - started,
            }
        )
        trained_models[candidate.name] = model
    winner_result = max(
        candidate_results,
        key=lambda result: (
            result["validation_weighted_pr_auc"],
            result["review_capacity_0_1pct"]["precision"],
        ),
    )
    winner = trained_models[winner_result["name"]]
    config.output_directory.mkdir(parents=True, exist_ok=False)
    winner.save_model(config.output_directory / "solvency_xgboost.json")
    joblib.dump(
        preprocessor,
        config.output_directory / "solvency_preprocessor.joblib",
        compress=3,
    )
    (config.output_directory / "portable_preprocessor.json").write_text(
        json.dumps(portable_preprocessor_payload(preprocessor), indent=2, sort_keys=True)
    )
    metrics = {
        "schema_version": "1.0.0",
        "model_family": "merchant_solvency_cirp_public_announcement",
        "label_boundary": SOLVENCY_MODEL_CLAIM_BOUNDARY,
        "fit_payload_sha256": config.expected_fit_sha256,
        "random_seed": config.random_seed,
        "model_features": SOLVENCY_MODEL_FEATURES,
        "numeric_features": SOLVENCY_NUMERIC_FEATURES,
        "categorical_features": SOLVENCY_CATEGORICAL_FEATURES,
        "selection_rule": (
            "maximum_weighted_validation_pr_auc_then_precision_at_0_1pct_capacity"
        ),
        "winner": winner_result,
        "candidates": candidate_results,
        "runtime": {
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (config.output_directory / "selection_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )
    manifest = {}
    for artifact_path in sorted(config.output_directory.iterdir()):
        manifest[artifact_path.name] = {
            "bytes": artifact_path.stat().st_size,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
    (config.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return {"winner": winner_result, "manifest": manifest}
