"""Build a miniature but genuinely valid merchant-risk serving package.

The real model package and the 115 MB merchant index are gitignored, so no test
may depend on them. This helper writes the same artifacts in miniature: a
checksum-valid solvency model package with a real XGBoost booster, a small
merchant lookup index published through the immutable-Parquet writer, and a
frozen holdout evaluation report whose recorded model digest matches the
booster that was actually written.

Every company, identifier, capital figure, and address count below is an
obvious synthetic placeholder invented for this test suite. None of it is a
real merchant, a real MCA record, or a real IBBI outcome, and none of it comes
from Razorpay data.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from assay.artifacts.parquet import sha256_file, write_immutable_parquet
from assay.serving.index import (
    MerchantRiskIndexReport,
    build_merchant_risk_index,
)
from assay.solvency.artifact import (
    SOLVENCY_MODEL_CLAIM_BOUNDARY,
    PortableSolvencyPreprocessor,
)
from assay.solvency.evaluation import (
    SOLVENCY_REVIEW_CAPACITIES,
    SolvencyHoldoutEvaluationReport,
    SolvencyReviewCapacityMetrics,
    SolvencySplitMetrics,
)
from assay.solvency.training_data import (
    SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_MODEL_FEATURES,
    SOLVENCY_NUMERIC_FEATURES,
)

PLACEHOLDER_SHA256 = "a" * 64
FEATURE_CUTOFF = date(2026, 1, 1)
RANDOM_SEED = 106
BOOSTING_ROUNDS = 40
BEST_ITERATION = BOOSTING_ROUNDS - 1
TRAINING_ROWS = 240
MODEL_ARTIFACT_NAMES = (
    "portable_preprocessor.json",
    "selection_metrics.json",
    "solvency_xgboost.json",
)

# Synthetic placeholder identities. Not real companies, not real CINs.
LOW_RISK_CIN = "U72900KA2019PTC000001"
LEGAL_SUFFIX_CIN = "U72900MH2015PTC000002"
AMBIGUOUS_CINS = ("U72900KA2011PTC000003", "U72900MH2012PTC000004")
HIGH_RISK_CIN = "U46900MH2024PTC000005"
UNINDEXED_CIN = "U99999WB2020PTC999999"

LOW_RISK_NAME = "Placeholder Alpha Retail Private Limited"
LEGAL_SUFFIX_NAME = "Placeholder Beta Logistics Limited"
LEGAL_SUFFIX_QUERY = "Placeholder Beta Logistics"
AMBIGUOUS_NAME = "Placeholder Gamma Traders Private Limited"
HIGH_RISK_NAME = "Placeholder Omega Distressed Holdings Private Limited"
UNINDEXED_NAME = "Placeholder Nonesuch Ventures Private Limited"

# One or two categories per categorical feature keeps the transformed matrix
# 19 columns wide, so training a real booster stays cheap.
SYNTHETIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "state_code": ("karnataka", "maharashtra"),
    "roc_code": ("RoC-Bangalore", "RoC-Mumbai"),
    "company_status": ("ACTIVE", "UNDER LIQUIDATION"),
    "company_category": ("Company limited by Shares",),
    "company_subcategory": ("Non-govt company",),
    "company_class": ("Private",),
    "listing_status": ("Unlisted",),
    "company_origin": ("Indian",),
    "nic_division": ("46", "72"),
}
SYNTHETIC_NUMERIC_MEDIANS = (8.0, 13.8, 12.0, 0.4, 1.1, 0.7)

# Thresholds deliberately straddle the synthetic scores: the placeholder
# distressed company clears the top queue, every benign placeholder does not.
DEFAULT_BAND_SCORE_THRESHOLDS: dict[float, float] = {
    0.001: 0.5,
    0.005: 0.2,
    0.02: 0.05,
}
BAND_QUEUE_EVIDENCE: dict[float, tuple[int, float, float, float]] = {
    0.001: (20, 0.5, 0.1, 100.0),
    0.005: (100, 0.25, 0.25, 50.0),
    0.02: (400, 0.08, 0.4, 16.0),
}


@dataclass(frozen=True)
class ServingFixture:
    """Paths to one coherent set of synthetic serving artifacts."""

    model_directory: Path
    index_report_path: Path
    evaluation_path: Path
    model_sha256: str


def transformed_feature_names() -> tuple[str, ...]:
    """Name every column the portable preprocessor actually emits."""

    names = [f"numeric__{name}" for name in SOLVENCY_NUMERIC_FEATURES]
    for feature_name in SOLVENCY_CATEGORICAL_FEATURES:
        names.extend(
            f"categorical__{feature_name}_{category}"
            for category in SYNTHETIC_CATEGORIES[feature_name]
        )
    return tuple(names)


def build_portable_preprocessor() -> PortableSolvencyPreprocessor:
    """Return the exact preprocessing contract the fixture package ships."""

    return PortableSolvencyPreprocessor(
        schema_version="1.0.0",
        numeric_features=SOLVENCY_NUMERIC_FEATURES,
        numeric_medians=SYNTHETIC_NUMERIC_MEDIANS,
        categorical_features=SOLVENCY_CATEGORICAL_FEATURES,
        categorical_imputer_values=tuple(
            SYNTHETIC_CATEGORIES[feature_name][0]
            for feature_name in SOLVENCY_CATEGORICAL_FEATURES
        ),
        categories=tuple(
            SYNTHETIC_CATEGORIES[feature_name]
            for feature_name in SOLVENCY_CATEGORICAL_FEATURES
        ),
        infrequent_categories=tuple(
            () for _ in SOLVENCY_CATEGORICAL_FEATURES
        ),
        handle_unknown="ignore",
        min_frequency=10,
        transformed_feature_names=transformed_feature_names(),
        label_boundary=SOLVENCY_MODEL_CLAIM_BOUNDARY,
    )


def _synthetic_training_frame() -> tuple[pl.DataFrame, np.ndarray]:
    """Draw separable synthetic rows so the booster learns a real signal.

    The invented outcome fires for companies crowded onto one registered
    address or incorporated only months before the cutoff. It is a fixture
    convention, not a claim about how public insolvency actually behaves.
    """

    generator = np.random.default_rng(RANDOM_SEED)
    shared_address = generator.uniform(0.0, 6.5, TRAINING_ROWS)
    company_age_years = generator.uniform(0.2, 40.0, TRAINING_ROWS)
    target = (
        (shared_address >= 3.0) | (company_age_years <= 1.5)
    ).astype(np.int8)
    columns: dict[str, Any] = {
        "company_age_years": company_age_years,
        "log_authorised_capital_inr": generator.uniform(
            10.0, 18.0, TRAINING_ROWS
        ),
        "log_paid_up_capital_inr": generator.uniform(9.0, 17.0, TRAINING_ROWS),
        "paid_to_authorised_capital_ratio": generator.uniform(
            0.0, 1.0, TRAINING_ROWS
        ),
        "log_shared_address_company_count": shared_address,
        "log_address_registration_month_company_count": generator.uniform(
            0.0, 4.0, TRAINING_ROWS
        ),
    }
    for feature_name in SOLVENCY_CATEGORICAL_FEATURES:
        categories = SYNTHETIC_CATEGORIES[feature_name]
        picks = generator.integers(0, len(categories), TRAINING_ROWS)
        columns[feature_name] = [categories[index] for index in picks]
    return pl.DataFrame(columns).select(SOLVENCY_MODEL_FEATURES), target


def write_solvency_model_package(model_directory: Path) -> Path:
    """Write a checksum-valid, loadable, genuinely trained model package."""

    model_directory.mkdir(parents=True, exist_ok=True)
    preprocessor = build_portable_preprocessor()
    (model_directory / "portable_preprocessor.json").write_text(
        json.dumps(preprocessor.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    xgboost = importlib.import_module("xgboost")
    training_frame, target = _synthetic_training_frame()
    training_matrix = xgboost.DMatrix(
        preprocessor.transform(training_frame), label=target
    )
    booster = xgboost.train(
        {
            "objective": "binary:logistic",
            "eta": 0.3,
            "max_depth": 3,
            "min_child_weight": 1.0,
            "seed": RANDOM_SEED,
            "device": "cpu",
            "verbosity": 0,
        },
        training_matrix,
        num_boost_round=BOOSTING_ROUNDS,
    )
    booster.save_model(model_directory / "solvency_xgboost.json")

    metrics = {
        "schema_version": "1.0.0",
        "model_family": "merchant_solvency_cirp_public_announcement",
        "label_boundary": SOLVENCY_MODEL_CLAIM_BOUNDARY,
        "fit_payload_sha256": PLACEHOLDER_SHA256,
        "random_seed": RANDOM_SEED,
        "model_features": list(SOLVENCY_MODEL_FEATURES),
        "numeric_features": list(SOLVENCY_NUMERIC_FEATURES),
        "categorical_features": list(SOLVENCY_CATEGORICAL_FEATURES),
        "selection_rule": "validation_weighted_pr_auc",
        "winner": {
            "name": "synthetic_depth_3",
            "best_iteration": BEST_ITERATION,
            "validation_weighted_pr_auc": 0.72,
        },
    }
    (model_directory / "selection_metrics.json").write_text(
        json.dumps(metrics, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        artifact_name: {
            "bytes": (model_directory / artifact_name).stat().st_size,
            "sha256": sha256_file(model_directory / artifact_name),
        }
        for artifact_name in MODEL_ARTIFACT_NAMES
    }
    (model_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return model_directory


def model_sha256(model_directory: Path) -> str:
    """Return the digest the evaluation report has to agree with."""

    return sha256_file(model_directory / "solvency_xgboost.json")


def synthetic_observation_frame() -> pl.DataFrame:
    """Five obviously synthetic placeholder companies, one clearly distressed."""

    return pl.DataFrame(
        {
            "company_snapshot_id": [
                f"placeholder-snapshot-{index}" for index in range(1, 6)
            ],
            "cin": [
                LOW_RISK_CIN,
                LEGAL_SUFFIX_CIN,
                AMBIGUOUS_CINS[0],
                AMBIGUOUS_CINS[1],
                HIGH_RISK_CIN,
            ],
            "company_name": [
                LOW_RISK_NAME,
                LEGAL_SUFFIX_NAME,
                AMBIGUOUS_NAME,
                AMBIGUOUS_NAME,
                HIGH_RISK_NAME,
            ],
            "evaluation_eligible": [True] * 5,
            "company_age_days": [2_190, 3_800, 5_200, 5_000, 200],
            "log_authorised_capital_inr": [13.8, 15.4, 12.0, 12.1, 11.5],
            "log_paid_up_capital_inr": [13.1, 14.9, 11.5, 11.6, 6.9],
            "paid_to_authorised_capital_ratio": [0.5, 0.8, 1.0, 0.95, 0.01],
            "shared_address_company_count": [2, 1, 4, 3, 480],
            "address_registration_month_company_count": [1, 0, 2, 2, 60],
            "registration_date": [
                date(2019, 9, 20),
                date(2015, 8, 15),
                date(2011, 6, 1),
                date(2012, 3, 4),
                date(2024, 11, 4),
            ],
            "authorised_capital_inr": [
                1_000_000.0,
                5_000_000.0,
                200_000.0,
                210_000.0,
                100_000.0,
            ],
            "paid_up_capital_inr": [
                500_000.0,
                4_000_000.0,
                200_000.0,
                199_500.0,
                1_000.0,
            ],
            "nic_code": ["72900", "72900", "46900", "46900", "46900"],
            "state_code": [
                "karnataka",
                "maharashtra",
                "karnataka",
                "maharashtra",
                "maharashtra",
            ],
            "roc_code": [
                "RoC-Bangalore",
                "RoC-Mumbai",
                "RoC-Bangalore",
                "RoC-Mumbai",
                "RoC-Mumbai",
            ],
            "company_status": [
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "ACTIVE",
                "UNDER LIQUIDATION",
            ],
            "company_category": ["Company limited by Shares"] * 5,
            "company_subcategory": ["Non-govt company"] * 5,
            "company_class": ["Private"] * 5,
            "listing_status": ["Unlisted"] * 5,
            "company_origin": ["Indian"] * 5,
            "nic_division": ["72", "72", "46", "46", "46"],
        }
    )


def write_merchant_risk_index(index_root: Path) -> Path:
    """Publish the synthetic index Parquet and return its report path."""

    index_frame = build_merchant_risk_index(synthetic_observation_frame())
    index_artifact = write_immutable_parquet(
        index_frame,
        index_root / "merchant_risk_index.parquet",
        "zstd",
    )
    report = MerchantRiskIndexReport(
        run_id=PLACEHOLDER_SHA256,
        observation_run_id=PLACEHOLDER_SHA256,
        mca_source_snapshot_id="b" * 64,
        ibbi_source_snapshot_id="c" * 64,
        feature_cutoff=FEATURE_CUTOFF,
        indexed_companies=index_frame.height,
        distinct_strict_names=index_frame[
            "company_name_normalized_strict"
        ].n_unique(),
        companies_sharing_a_strict_name=int(
            (index_frame["strict_name_company_count"] > 1).sum()
        ),
        index_artifact=index_artifact,
    )
    report_path = index_root / "merchant_risk_index.report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def write_solvency_evaluation_report(
    report_path: Path,
    booster_sha256: str,
    *,
    score_thresholds: dict[float, float] | None = None,
) -> SolvencyHoldoutEvaluationReport:
    """Write frozen holdout evidence the risk bands can be derived from."""

    thresholds = score_thresholds or DEFAULT_BAND_SCORE_THRESHOLDS
    review_capacities = []
    for capacity_fraction in SOLVENCY_REVIEW_CAPACITIES:
        review_rows, precision, recall, lift = BAND_QUEUE_EVIDENCE[
            capacity_fraction
        ]
        review_capacities.append(
            SolvencyReviewCapacityMetrics(
                capacity_fraction=capacity_fraction,
                review_rows=review_rows,
                score_threshold=thresholds[capacity_fraction],
                precision=precision,
                recall=recall,
                lift=lift,
            )
        )
    split = SolvencySplitMetrics(
        dataset_split="temporal_test",
        rows=20_000,
        positive_rows=100,
        base_rate=0.005,
        diagnostic_accuracy_at_0_5=0.99,
        pr_auc=0.21,
        roc_auc=0.86,
        brier_score=0.005,
        score_minimum=0.0,
        score_maximum=0.95,
        score_mean=0.01,
        review_capacities=tuple(review_capacities),
    )
    report = SolvencyHoldoutEvaluationReport(
        label_boundary=SOLVENCY_MODEL_CLAIM_BOUNDARY,
        model_sha256=booster_sha256,
        model_data_path=Path("synthetic/solvency_model_data.parquet"),
        holdouts_are_unsampled=True,
        evaluation_decision="EVALUATED_NOT_PRODUCTION_READY",
        deployment_blockers=(
            "Synthetic fixture evidence; no production claim is made.",
            "Risk operations has not supplied a review budget.",
        ),
        split_metrics={"temporal_test": split},
    )
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def write_serving_fixture(root: Path) -> ServingFixture:
    """Write a model package, a merchant index, and matching holdout evidence."""

    model_directory = write_solvency_model_package(root / "model")
    index_report_path = write_merchant_risk_index(root / "index")
    booster_sha256 = model_sha256(model_directory)
    evaluation_path = root / "solvency_evaluation.json"
    write_solvency_evaluation_report(evaluation_path, booster_sha256)
    return ServingFixture(
        model_directory=model_directory,
        index_report_path=index_report_path,
        evaluation_path=evaluation_path,
        model_sha256=booster_sha256,
    )
