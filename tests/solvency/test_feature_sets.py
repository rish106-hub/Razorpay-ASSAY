from __future__ import annotations

import pytest

from assay.solvency.feature_sets import (
    ADDRESS_CLUSTER_SHAPE_FEATURES,
    BASELINE_NO_STATUS,
    BASELINE_NUMERIC_FEATURES,
    BASELINE_WITH_STATUS,
    CIN_STRUCTURE_CATEGORICAL_FEATURES,
    CIN_STRUCTURE_NUMERIC_FEATURES,
    DEFAULT_SOLVENCY_FEATURE_SET_NAME,
    EXPANDED_NO_STATUS,
    ROC_SERIAL_ADJACENCY_FEATURES,
    SOLVENCY_FEATURE_SETS,
    SOLVENCY_OUTCOME_ADJACENT_FEATURE,
    SolvencyFeatureSetError,
    resolve_feature_set,
)
from assay.solvency.training_data import (
    SOLVENCY_CATEGORICAL_FEATURES,
    SOLVENCY_MODEL_DATA_FEATURES,
    SOLVENCY_MODEL_FEATURES,
    SOLVENCY_NUMERIC_FEATURES,
    expanded_solvency_feature_expressions,
)


def test_baseline_with_status_reproduces_the_published_column_contract() -> None:
    assert BASELINE_WITH_STATUS.numeric_features == (
        "company_age_years",
        "log_authorised_capital_inr",
        "log_paid_up_capital_inr",
        "paid_to_authorised_capital_ratio",
        "log_shared_address_company_count",
        "log_address_registration_month_company_count",
    )
    assert BASELINE_WITH_STATUS.categorical_features == (
        "state_code",
        "roc_code",
        "company_status",
        "company_category",
        "company_subcategory",
        "company_class",
        "listing_status",
        "company_origin",
        "nic_division",
    )
    assert BASELINE_WITH_STATUS.model_features == (
        *BASELINE_WITH_STATUS.numeric_features,
        *BASELINE_WITH_STATUS.categorical_features,
    )
    assert len(BASELINE_WITH_STATUS.model_features) == 15


def test_training_data_constants_alias_the_default_feature_set() -> None:
    assert DEFAULT_SOLVENCY_FEATURE_SET_NAME == BASELINE_WITH_STATUS.name
    assert SOLVENCY_NUMERIC_FEATURES == BASELINE_WITH_STATUS.numeric_features
    assert SOLVENCY_CATEGORICAL_FEATURES == (
        BASELINE_WITH_STATUS.categorical_features
    )
    assert SOLVENCY_MODEL_FEATURES == BASELINE_WITH_STATUS.model_features


def test_baseline_no_status_drops_only_the_outcome_adjacent_column() -> None:
    assert BASELINE_NO_STATUS.numeric_features == (
        BASELINE_WITH_STATUS.numeric_features
    )
    assert SOLVENCY_OUTCOME_ADJACENT_FEATURE not in (
        BASELINE_NO_STATUS.categorical_features
    )
    assert BASELINE_NO_STATUS.categorical_features == tuple(
        feature_name
        for feature_name in BASELINE_WITH_STATUS.categorical_features
        if feature_name != SOLVENCY_OUTCOME_ADJACENT_FEATURE
    )
    assert len(BASELINE_NO_STATUS.model_features) == 14
    assert BASELINE_WITH_STATUS.includes_outcome_adjacent_status is True
    assert BASELINE_NO_STATUS.includes_outcome_adjacent_status is False


def test_resolve_feature_set_returns_the_registered_contract() -> None:
    for name, feature_set in SOLVENCY_FEATURE_SETS.items():
        assert resolve_feature_set(name) is feature_set


def test_resolve_feature_set_names_the_valid_alternatives() -> None:
    with pytest.raises(SolvencyFeatureSetError) as error:
        resolve_feature_set("baseline_without_status")

    message = str(error.value)
    assert "baseline_without_status" in message
    assert "baseline_no_status" in message
    assert "baseline_with_status" in message


def test_feature_sets_are_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        BASELINE_WITH_STATUS.name = "renamed"  # type: ignore[misc]


def test_expanded_no_status_extends_the_honest_baseline_only() -> None:
    """The expanded set adds cluster shape and CIN structure, never status."""

    assert EXPANDED_NO_STATUS.numeric_features[: len(BASELINE_NUMERIC_FEATURES)] == (
        BASELINE_NUMERIC_FEATURES
    )
    assert EXPANDED_NO_STATUS.numeric_features[len(BASELINE_NUMERIC_FEATURES) :] == (
        *ADDRESS_CLUSTER_SHAPE_FEATURES,
        *ROC_SERIAL_ADJACENCY_FEATURES,
        *CIN_STRUCTURE_NUMERIC_FEATURES,
    )
    assert EXPANDED_NO_STATUS.categorical_features == (
        *BASELINE_NO_STATUS.categorical_features,
        *CIN_STRUCTURE_CATEGORICAL_FEATURES,
    )
    assert EXPANDED_NO_STATUS.includes_outcome_adjacent_status is False
    assert len(EXPANDED_NO_STATUS.model_features) == 28


def test_model_data_carries_the_union_of_every_frozen_feature_set() -> None:
    """A narrower set must be fittable by selection, never by a new split."""

    carried = set(SOLVENCY_MODEL_DATA_FEATURES)
    for feature_set in SOLVENCY_FEATURE_SETS.values():
        assert set(feature_set.model_features) <= carried
    assert len(SOLVENCY_MODEL_DATA_FEATURES) == len(set(SOLVENCY_MODEL_DATA_FEATURES))
    assert SOLVENCY_MODEL_DATA_FEATURES[: len(SOLVENCY_MODEL_FEATURES)] != ()


def test_expanded_expressions_cover_every_added_column() -> None:
    """Each expanded feature has exactly one derivation, named after it."""

    derived_names = [
        expression.meta.output_name()
        for expression in expanded_solvency_feature_expressions()
    ]
    expanded_names = [
        feature_name
        for feature_name in EXPANDED_NO_STATUS.model_features
        if feature_name not in set(BASELINE_NO_STATUS.model_features)
    ]

    assert sorted(derived_names) == sorted(expanded_names)
