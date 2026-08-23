"""Name and version the competing solvency model feature contracts.

Every solvency metric has to be reportable twice: once for the feature set that
keeps `company_status`, and once for the feature set that removes it. The two
are not the same claim. MCA reports `Under CIRP` for a company that is already
inside the insolvency resolution process, so a pre-cutoff snapshot carrying that
value leaks the outcome it is meant to predict. The value is temporally legal
and it is still outcome contamination.

Naming the feature list, freezing it, and writing the name into every exported
artifact is what makes the difference auditable instead of a guess about which
published number was real.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SOLVENCY_FEATURE_SET_SCHEMA_VERSION = "1.0.0"

#: MCA status is a snapshot field, not a discovered pattern. `Under CIRP` means
#: the resolution process has already started, so the column is treated as an
#: outcome-adjacent feature that a named set either accepts or drops.
SOLVENCY_OUTCOME_ADJACENT_FEATURE = "company_status"

BASELINE_NUMERIC_FEATURES = (
    "company_age_years",
    "log_authorised_capital_inr",
    "log_paid_up_capital_inr",
    "paid_to_authorised_capital_ratio",
    "log_shared_address_company_count",
    "log_address_registration_month_company_count",
)
BASELINE_CATEGORICAL_FEATURES = (
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


class SolvencyFeatureSetError(RuntimeError):
    """A solvency feature set was requested under a name that is not frozen."""


class SolvencyFeatureSet(BaseModel):
    """One named, ordered numeric-then-categorical model feature contract."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    numeric_features: tuple[str, ...] = Field(min_length=1)
    categorical_features: tuple[str, ...] = Field(min_length=1)

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return every model column, numeric first, in declared order."""

        return (*self.numeric_features, *self.categorical_features)

    @property
    def includes_outcome_adjacent_status(self) -> bool:
        """State whether this set keeps the outcome-adjacent MCA status column."""

        return SOLVENCY_OUTCOME_ADJACENT_FEATURE in self.categorical_features


BASELINE_WITH_STATUS = SolvencyFeatureSet(
    name="baseline_with_status",
    numeric_features=BASELINE_NUMERIC_FEATURES,
    categorical_features=BASELINE_CATEGORICAL_FEATURES,
)
BASELINE_NO_STATUS = SolvencyFeatureSet(
    name="baseline_no_status",
    numeric_features=BASELINE_NUMERIC_FEATURES,
    categorical_features=tuple(
        feature_name
        for feature_name in BASELINE_CATEGORICAL_FEATURES
        if feature_name != SOLVENCY_OUTCOME_ADJACENT_FEATURE
    ),
)

DEFAULT_SOLVENCY_FEATURE_SET_NAME = BASELINE_WITH_STATUS.name
SOLVENCY_FEATURE_SETS: dict[str, SolvencyFeatureSet] = {
    feature_set.name: feature_set
    for feature_set in (BASELINE_WITH_STATUS, BASELINE_NO_STATUS)
}


def resolve_feature_set(name: str) -> SolvencyFeatureSet:
    """Look up one frozen feature set and name the alternatives on failure."""

    feature_set = SOLVENCY_FEATURE_SETS.get(name)
    if feature_set is None:
        raise SolvencyFeatureSetError(
            f"Unknown solvency feature set: {name}. Valid names are "
            f"{', '.join(sorted(SOLVENCY_FEATURE_SETS))}."
        )
    return feature_set
