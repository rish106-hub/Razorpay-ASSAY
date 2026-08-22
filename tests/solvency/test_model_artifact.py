from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from assay.solvency.artifact import (
    PortableSolvencyPreprocessor,
    SolvencyModelArtifactError,
    load_solvency_model_package,
)
from assay.solvency.model_training import build_population_restoration_weights
from assay.solvency.training_data import (
    SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_MODEL_FEATURES,
    SOLVENCY_NUMERIC_FEATURES,
)


def _write_package(run_directory: Path) -> None:
    metrics = {
        "schema_version": "1.0.0",
        "model_family": "merchant_solvency_cirp_public_announcement",
        "label_boundary": "cirp_public_announcement_outcome_not_fraud",
        "fit_payload_sha256": "a" * 64,
        "random_seed": 106,
        "model_features": SOLVENCY_MODEL_FEATURES,
        "numeric_features": SOLVENCY_NUMERIC_FEATURES,
        "categorical_features": SOLVENCY_CATEGORICAL_FEATURES,
        "selection_rule": "validation_pr_auc",
        "winner": {
            "name": "depth_4_regularised",
            "best_iteration": 10,
            "validation_weighted_pr_auc": 0.7,
        },
    }
    preprocessor = {
        "schema_version": "1.0.0",
        "numeric_features": SOLVENCY_NUMERIC_FEATURES,
        "numeric_medians": [1.0] * len(SOLVENCY_NUMERIC_FEATURES),
        "categorical_features": SOLVENCY_CATEGORICAL_FEATURES,
        "categorical_imputer_values": ["UNKNOWN"]
        * len(SOLVENCY_CATEGORICAL_FEATURES),
        "categories": [["KNOWN", "RARE"]]
        * len(SOLVENCY_CATEGORICAL_FEATURES),
        "infrequent_categories": [["RARE"]]
        * len(SOLVENCY_CATEGORICAL_FEATURES),
        "handle_unknown": "ignore",
        "min_frequency": 10,
        "transformed_feature_names": [
            *[f"numeric__{name}" for name in SOLVENCY_NUMERIC_FEATURES],
            *[
                encoded
                for feature in SOLVENCY_CATEGORICAL_FEATURES
                for encoded in (
                    f"categorical__{feature}_KNOWN",
                    f"categorical__{feature}_infrequent_sklearn",
                )
            ],
        ],
        "label_boundary": "cirp_public_announcement_outcome_not_fraud",
    }
    artifacts = {
        "selection_metrics.json": json.dumps(metrics, sort_keys=True).encode(),
        "portable_preprocessor.json": json.dumps(
            preprocessor, sort_keys=True
        ).encode(),
        "solvency_xgboost.json": b"model-placeholder",
    }
    manifest = {}
    for name, payload in artifacts.items():
        (run_directory / name).write_bytes(payload)
        manifest[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (run_directory / "manifest.json").write_text(json.dumps(manifest))


def test_load_solvency_model_package_verifies_portable_contract(
    tmp_path: Path,
) -> None:
    _write_package(tmp_path)

    package = load_solvency_model_package(tmp_path, load_model=False)

    assert package.report.selected_candidate == "depth_4_regularised"
    assert package.report.raw_feature_count == 15
    assert package.report.transformed_feature_count == 24
    assert package.report.model_loaded is False


def test_load_solvency_model_package_rejects_tampered_model(
    tmp_path: Path,
) -> None:
    _write_package(tmp_path)
    (tmp_path / "solvency_xgboost.json").write_bytes(b"changed")

    with pytest.raises(SolvencyModelArtifactError, match="size mismatch"):
        load_solvency_model_package(tmp_path, load_model=False)


def test_portable_preprocessor_imputes_and_routes_infrequent_categories() -> None:
    contract = PortableSolvencyPreprocessor(
        schema_version="1.0.0",
        numeric_features=("merchant_age",),
        numeric_medians=(4.0,),
        categorical_features=("merchant_state",),
        categorical_imputer_values=("KNOWN",),
        categories=(("KNOWN", "RARE"),),
        infrequent_categories=(("RARE",),),
        handle_unknown="ignore",
        min_frequency=10,
        transformed_feature_names=(
            "numeric__merchant_age",
            "categorical__merchant_state_KNOWN",
            "categorical__merchant_state_infrequent_sklearn",
        ),
        label_boundary="cirp_public_announcement_outcome_not_fraud",
    )
    frame = pl.DataFrame(
        {
            "merchant_age": [None, 2.0, 3.0, 4.0],
            "merchant_state": [None, "KNOWN", "RARE", "UNSEEN"],
        }
    )

    transformed = contract.transform(frame).toarray()

    assert transformed.tolist() == [
        [4.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 0.0, 1.0],
        [4.0, 0.0, 0.0],
    ]


def test_solvency_package_scores_only_through_best_iteration() -> None:
    contract = PortableSolvencyPreprocessor(
        schema_version="1.0.0",
        numeric_features=("merchant_age",),
        numeric_medians=(4.0,),
        categorical_features=(),
        categorical_imputer_values=(),
        categories=(),
        infrequent_categories=(),
        handle_unknown="ignore",
        min_frequency=10,
        transformed_feature_names=("numeric__merchant_age",),
        label_boundary="cirp_public_announcement_outcome_not_fraud",
    )

    class _Booster:
        def __init__(self) -> None:
            self.iteration_range: tuple[int, int] | None = None

        def inplace_predict(
            self,
            _: object,
            *,
            iteration_range: tuple[int, int],
        ) -> list[float]:
            self.iteration_range = iteration_range
            return [0.25]

    booster = _Booster()
    metrics = {
        "schema_version": "1.0.0",
        "model_family": "merchant_solvency_cirp_public_announcement",
        "label_boundary": "cirp_public_announcement_outcome_not_fraud",
        "fit_payload_sha256": "a" * 64,
        "random_seed": 106,
        "model_features": ["merchant_age"],
        "numeric_features": ["merchant_age"],
        "categorical_features": [],
        "selection_rule": "validation_pr_auc",
        "winner": {
            "name": "shallow",
            "best_iteration": 7,
            "validation_weighted_pr_auc": 0.7,
        },
    }
    from assay.solvency.artifact import (
        SolvencyArtifactDigest,
        SolvencyModelPackage,
        SolvencySelectionMetrics,
    )

    package = SolvencyModelPackage(
        run_directory=Path("."),
        manifest={
            name: SolvencyArtifactDigest(bytes=1, sha256="a" * 64)
            for name in (
                "solvency_xgboost.json",
                "portable_preprocessor.json",
                "selection_metrics.json",
            )
        },
        metrics=SolvencySelectionMetrics.model_validate(metrics),
        preprocessor=contract,
        booster=booster,
        xgboost_version="test",
    )

    scores = package.score(pl.DataFrame({"merchant_age": [2.0]}))

    assert scores.to_list() == [0.25]
    assert booster.iteration_range == (0, 8)


def test_population_restoration_weights_preserve_positive_mass() -> None:
    weights = build_population_restoration_weights(
        np.array([1, 0, 0], dtype=np.int8),
        full_negative_rows=10,
        sampled_negative_rows=2,
    )

    assert weights.tolist() == [1.0, 5.0, 5.0]
