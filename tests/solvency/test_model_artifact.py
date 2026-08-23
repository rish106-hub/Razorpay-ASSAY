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
from assay.solvency.feature_sets import (
    BASELINE_NO_STATUS,
    BASELINE_WITH_STATUS,
    SolvencyFeatureSet,
)
from assay.solvency.model_training import build_population_restoration_weights
from assay.solvency.training_data import SOLVENCY_MODEL_FEATURES

FROZEN_COLAB_PACKAGE = Path("data/generated/solvency_model/run-67e0cc4f177e35f4")


def _write_package(
    run_directory: Path,
    *,
    feature_set: SolvencyFeatureSet = BASELINE_WITH_STATUS,
    metrics_feature_set_name: str | None = None,
    preprocessor_feature_set_name: str | None = None,
) -> None:
    """Write one checksum-valid package for the named feature contract.

    A `None` feature-set name omits the key entirely, which is how every
    artifact exported before the feature-set contract existed looks on disk.
    """

    numeric_features = feature_set.numeric_features
    categorical_features = feature_set.categorical_features
    metrics = {
        "schema_version": "1.0.0",
        "model_family": "merchant_solvency_cirp_public_announcement",
        "label_boundary": "cirp_public_announcement_outcome_not_fraud",
        "fit_payload_sha256": "a" * 64,
        "random_seed": 106,
        "model_features": feature_set.model_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "selection_rule": "validation_pr_auc",
        "winner": {
            "name": "depth_4_regularised",
            "best_iteration": 10,
            "validation_weighted_pr_auc": 0.7,
        },
    }
    if metrics_feature_set_name is not None:
        metrics["feature_set_name"] = metrics_feature_set_name
    preprocessor = {
        "schema_version": "1.0.0",
        "numeric_features": numeric_features,
        "numeric_medians": [1.0] * len(numeric_features),
        "categorical_features": categorical_features,
        "categorical_imputer_values": ["UNKNOWN"] * len(categorical_features),
        "categories": [["KNOWN", "RARE"]] * len(categorical_features),
        "infrequent_categories": [["RARE"]] * len(categorical_features),
        "handle_unknown": "ignore",
        "min_frequency": 10,
        "transformed_feature_names": [
            *[f"numeric__{name}" for name in numeric_features],
            *[
                encoded
                for feature in categorical_features
                for encoded in (
                    f"categorical__{feature}_KNOWN",
                    f"categorical__{feature}_infrequent_sklearn",
                )
            ],
        ],
        "label_boundary": "cirp_public_announcement_outcome_not_fraud",
    }
    if preprocessor_feature_set_name is not None:
        preprocessor["feature_set_name"] = preprocessor_feature_set_name
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


def test_package_without_a_feature_set_name_defaults_to_the_baseline(
    tmp_path: Path,
) -> None:
    """Artifacts frozen before the contract existed must keep loading."""

    _write_package(tmp_path)

    package = load_solvency_model_package(tmp_path, load_model=False)

    assert package.preprocessor.feature_set_name == "baseline_with_status"
    assert package.metrics.feature_set_name == "baseline_with_status"
    assert package.report.feature_set_name == "baseline_with_status"


@pytest.mark.skipif(
    not (FROZEN_COLAB_PACKAGE / "manifest.json").is_file(),
    reason="The frozen Colab model package is gitignored and not present.",
)
def test_frozen_colab_package_still_loads_under_the_feature_set_contract() -> None:
    package = load_solvency_model_package(FROZEN_COLAB_PACKAGE, load_model=False)

    assert package.report.feature_set_name == "baseline_with_status"
    assert package.metrics.model_features == SOLVENCY_MODEL_FEATURES


def test_load_solvency_model_package_accepts_the_no_status_feature_set(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path,
        feature_set=BASELINE_NO_STATUS,
        metrics_feature_set_name="baseline_no_status",
        preprocessor_feature_set_name="baseline_no_status",
    )

    package = load_solvency_model_package(tmp_path, load_model=False)

    assert package.report.feature_set_name == "baseline_no_status"
    assert package.report.raw_feature_count == 14
    assert "company_status" not in package.metrics.model_features


def test_load_solvency_model_package_rejects_disagreeing_feature_sets(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path,
        feature_set=BASELINE_NO_STATUS,
        metrics_feature_set_name="baseline_with_status",
        preprocessor_feature_set_name="baseline_no_status",
    )

    with pytest.raises(SolvencyModelArtifactError, match="feature sets differ"):
        load_solvency_model_package(tmp_path, load_model=False)


def test_load_solvency_model_package_rejects_an_unknown_feature_set(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path,
        metrics_feature_set_name="baseline_invented",
        preprocessor_feature_set_name="baseline_invented",
    )

    with pytest.raises(
        SolvencyModelArtifactError, match="Unknown solvency feature set"
    ):
        load_solvency_model_package(tmp_path, load_model=False)


def test_load_solvency_model_package_rejects_a_mismatched_column_list(
    tmp_path: Path,
) -> None:
    _write_package(
        tmp_path,
        feature_set=BASELINE_NO_STATUS,
        metrics_feature_set_name="baseline_with_status",
        preprocessor_feature_set_name="baseline_with_status",
    )

    with pytest.raises(
        SolvencyModelArtifactError, match="invalid for baseline_with_status"
    ):
        load_solvency_model_package(tmp_path, load_model=False)


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
