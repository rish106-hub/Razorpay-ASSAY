"""Build leakage-safe merchant-solvency observations from MCA and IBBI."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

SOLVENCY_OBSERVATION_SCHEMA_VERSION = "1.0.0"


class SolvencyObservationError(RuntimeError):
    """A merchant-solvency observation contract failed validation."""


class SolvencyObservationConfig(BaseModel):
    """Frozen feature and CIRP public-announcement outcome windows."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    outcome_window_end: date

    @model_validator(mode="after")
    def require_forward_outcome_window(self) -> SolvencyObservationConfig:
        if self.outcome_window_end <= self.feature_cutoff:
            raise ValueError("the solvency outcome window must follow the feature cutoff")
        return self

    @property
    def outcome_window_start(self) -> date:
        return self.feature_cutoff + timedelta(days=1)


def _require_columns(
    frame: pl.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    if missing_columns := required_columns - set(frame.columns):
        missing_column_list = ", ".join(sorted(missing_columns))
        raise SolvencyObservationError(
            f"{frame_name} is missing columns: {missing_column_list}."
        )


def build_solvency_observations(
    company_frame: pl.DataFrame,
    address_frame: pl.DataFrame,
    cirp_announcement_frame: pl.DataFrame,
    config: SolvencyObservationConfig,
) -> pl.DataFrame:
    """Create one pre-outcome feature row per MCA company with an exact-CIN label."""

    _require_columns(
        company_frame,
        {
            "company_snapshot_id",
            "source_snapshot_id",
            "snapshot_as_of",
            "legal_entity_identifier",
            "legal_entity_identifier_type",
            "legal_entity_identifier_is_valid_format",
            "cin",
            "company_name",
            "company_status",
            "registration_date",
            "state_code",
            "roc_code",
            "company_category",
            "company_subcategory",
            "company_class",
            "listing_status",
            "company_origin",
            "authorised_capital_inr",
            "paid_up_capital_inr",
            "nic_code",
        },
        "MCA company snapshot",
    )
    _require_columns(
        address_frame,
        {
            "legal_entity_identifier",
            "source_snapshot_id",
            "snapshot_as_of",
            "address_group_key",
        },
        "MCA address snapshot",
    )
    _require_columns(
        cirp_announcement_frame,
        {"solvency_event_id", "cin", "event_date"},
        "IBBI CIRP announcement snapshot",
    )
    if company_frame["company_snapshot_id"].n_unique() != company_frame.height:
        raise SolvencyObservationError(
            "MCA company snapshot must contain one row per company_snapshot_id."
        )
    if address_frame["legal_entity_identifier"].n_unique() != address_frame.height:
        raise SolvencyObservationError(
            "MCA address snapshot must contain one row per legal identifier."
        )
    if company_frame["snapshot_as_of"].unique().to_list() != [
        config.feature_cutoff
    ]:
        raise SolvencyObservationError(
            "MCA company snapshot date must equal the feature cutoff."
        )
    if address_frame["snapshot_as_of"].unique().to_list() != [
        config.feature_cutoff
    ]:
        raise SolvencyObservationError(
            "MCA address snapshot date must equal the feature cutoff."
        )

    address_features = address_frame.select(
        "legal_entity_identifier",
        pl.col("address_group_key").fill_null(""),
    ).with_columns(
        pl.when(pl.col("address_group_key") != "")
        .then(pl.len().over("address_group_key"))
        .otherwise(0)
        .cast(pl.Int32)
        .alias("shared_address_company_count")
    )
    merchant_features = company_frame.join(
        address_features,
        on="legal_entity_identifier",
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("address_group_key").fill_null(""),
        pl.col("shared_address_company_count").fill_null(0),
        (
            pl.lit(config.feature_cutoff)
            - pl.col("registration_date")
        ).dt.total_days().cast(pl.Int32).alias("company_age_days"),
        pl.col("nic_code").fill_null("").str.slice(0, 2).alias("nic_division"),
        pl.col("authorised_capital_inr")
        .fill_null(0)
        .cast(pl.Float64)
        .log1p()
        .alias("log_authorised_capital_inr"),
        pl.col("paid_up_capital_inr")
        .fill_null(0)
        .cast(pl.Float64)
        .log1p()
        .alias("log_paid_up_capital_inr"),
        (
            pl.col("paid_up_capital_inr").fill_null(0).cast(pl.Float64)
            / pl.col("authorised_capital_inr")
            .cast(pl.Float64)
            .replace(0.0, None)
        )
        .fill_null(0.0)
        .alias("paid_to_authorised_capital_ratio"),
        pl.concat_str(
            [
                pl.col("address_group_key").fill_null(""),
                pl.col("registration_date").dt.strftime("%Y-%m"),
            ],
            separator=":",
        ).alias("address_registration_month_key"),
    ).with_columns(
        pl.when(pl.col("address_group_key") != "")
        .then(pl.len().over("address_registration_month_key"))
        .otherwise(0)
        .cast(pl.Int32)
        .alias("address_registration_month_company_count")
    )

    prior_cirp = cirp_announcement_frame.filter(
        pl.col("event_date") <= config.feature_cutoff
    ).select("cin").unique().with_columns(
        pl.lit(True).alias("has_prior_cirp_announcement")
    )
    window_cirp = cirp_announcement_frame.filter(
        pl.col("event_date").is_between(
            config.outcome_window_start,
            config.outcome_window_end,
            closed="both",
        )
    ).group_by("cin").agg(
        pl.len().cast(pl.Int32).alias("cirp_announcement_count"),
        pl.col("event_date").min().alias("first_cirp_announcement_date"),
    )
    return merchant_features.join(prior_cirp, on="cin", how="left").join(
        window_cirp,
        on="cin",
        how="left",
    ).with_columns(
        pl.col("has_prior_cirp_announcement").fill_null(False),
        pl.col("cirp_announcement_count").fill_null(0),
    ).with_columns(
        (pl.col("cirp_announcement_count") > 0).alias(
            "observed_cirp_public_announcement"
        ),
        (
            pl.col("legal_entity_identifier_type").eq("CIN")
            & pl.col("legal_entity_identifier_is_valid_format")
            & pl.col("registration_date").is_not_null()
            & (pl.col("registration_date") <= config.feature_cutoff)
            & ~pl.col("has_prior_cirp_announcement")
        ).alias("evaluation_eligible"),
        pl.lit(config.run_id).alias("run_id"),
        pl.lit(config.feature_cutoff).alias("feature_cutoff"),
        pl.lit(config.outcome_window_start).alias("outcome_window_start"),
        pl.lit(config.outcome_window_end).alias("outcome_window_end"),
        pl.lit("cirp_public_announcement_outcome").alias("target_name"),
        pl.lit("exact_valid_cin").alias("label_match_policy"),
        pl.lit(SOLVENCY_OBSERVATION_SCHEMA_VERSION).alias("schema_version"),
    )
