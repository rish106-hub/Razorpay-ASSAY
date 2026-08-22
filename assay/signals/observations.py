"""Build as-of signal observations and narrow adverse-outcome labels."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

SIGNAL_OBSERVATION_SCHEMA_VERSION = "1.0.0"


class SignalObservationError(RuntimeError):
    """A deterministic signal-observation contract failure."""


class SignalObservationConfig(BaseModel):
    """Frozen observation and outcome windows for one signal dataset."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: date
    outcome_window_start: date
    outcome_window_end: date
    shared_address_minimum_companies: int = Field(default=2, ge=2)

    @model_validator(mode="after")
    def require_non_overlapping_windows(self) -> SignalObservationConfig:
        if self.outcome_window_start <= self.feature_cutoff:
            raise ValueError("the outcome window must start after the feature cutoff")
        if self.outcome_window_end < self.outcome_window_start:
            raise ValueError("the outcome window end cannot precede its start")
        return self


def default_signal_observation_config(
    *,
    run_id: str,
    feature_cutoff: date,
    outcome_window_end: date,
) -> SignalObservationConfig:
    """Create a next-day outcome window without hidden overlap."""

    return SignalObservationConfig(
        run_id=run_id,
        feature_cutoff=feature_cutoff,
        outcome_window_start=feature_cutoff + timedelta(days=1),
        outcome_window_end=outcome_window_end,
    )


def _require_columns(
    frame: pl.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    if missing_columns := required_columns - set(frame.columns):
        missing_column_list = ", ".join(sorted(missing_columns))
        raise SignalObservationError(
            f"{frame_name} is missing columns: {missing_column_list}."
        )


def _require_unique(frame: pl.DataFrame, column: str, frame_name: str) -> None:
    if frame[column].n_unique() != frame.height:
        raise SignalObservationError(
            f"{frame_name} must contain one row per {column}."
        )


def build_signal_observations(
    company_frame: pl.DataFrame,
    address_frame: pl.DataFrame,
    adverse_frame: pl.DataFrame,
    match_decision_frame: pl.DataFrame,
    config: SignalObservationConfig,
) -> pl.DataFrame:
    """Build company and address-cohort signals before the outcome window."""

    _require_columns(
        company_frame,
        {
            "company_snapshot_id",
            "source_snapshot_id",
            "snapshot_as_of",
            "cin",
            "company_name",
            "registration_date",
            "state_code",
            "nic_code",
            "paid_up_capital_inr",
            "cin_is_valid_format",
        },
        "company snapshot",
    )
    _require_columns(
        address_frame,
        {
            "company_address_snapshot_id",
            "source_snapshot_id",
            "source_record_id",
            "address_normalized",
            "address_group_key",
            "snapshot_as_of",
        },
        "company address snapshot",
    )
    _require_columns(
        adverse_frame,
        {"adverse_event_id", "event_date"},
        "adverse-event snapshot",
    )
    _require_columns(
        match_decision_frame,
        {
            "adverse_event_id",
            "accepted_company_snapshot_id",
            "review_status",
            "outcome_eligible",
        },
        "entity-match decision",
    )
    _require_unique(company_frame, "company_snapshot_id", "company snapshot")
    _require_unique(
        address_frame,
        "company_address_snapshot_id",
        "company address snapshot",
    )
    _require_unique(
        match_decision_frame,
        "adverse_event_id",
        "entity-match decision",
    )

    company_snapshot_dates = company_frame["snapshot_as_of"].unique().to_list()
    address_snapshot_dates = address_frame["snapshot_as_of"].unique().to_list()
    if company_snapshot_dates != [config.feature_cutoff]:
        raise SignalObservationError(
            "Company snapshot_as_of must equal the declared feature cutoff."
        )
    if address_snapshot_dates != [config.feature_cutoff]:
        raise SignalObservationError(
            "Address snapshot_as_of must equal the declared feature cutoff."
        )
    company_source_snapshot_ids = company_frame["source_snapshot_id"].unique()
    address_source_snapshot_ids = address_frame["source_snapshot_id"].unique()
    if (
        len(company_source_snapshot_ids) != 1
        or len(address_source_snapshot_ids) != 1
        or company_source_snapshot_ids.item() != address_source_snapshot_ids.item()
    ):
        raise SignalObservationError(
            "Company and address rows must come from the same source snapshot."
        )

    address_signals = address_frame.select(
        pl.col("source_record_id").alias("cin"),
        "address_normalized",
        "address_group_key",
    ).with_columns(
        pl.when(pl.col("address_group_key").fill_null("") != "")
        .then(pl.len().over("address_group_key"))
        .otherwise(0)
        .cast(pl.Int32)
        .alias("shared_address_company_count")
    )
    merchant_population = company_frame.join(
        address_signals,
        on="cin",
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("address_normalized").fill_null(""),
        pl.col("address_group_key").fill_null(""),
        pl.col("shared_address_company_count").fill_null(0),
        (
            pl.col("shared_address_company_count")
            >= config.shared_address_minimum_companies
        ).alias("shared_address_signal"),
        pl.when(
            (pl.col("address_group_key").fill_null("") != "")
            & pl.col("registration_date").is_not_null()
        )
        .then(
            pl.concat_str(
                [
                    pl.col("address_group_key"),
                    pl.col("registration_date").dt.strftime("%Y-%m"),
                ],
                separator=":",
            )
        )
        .otherwise(None)
        .alias("address_registration_month_cohort_key"),
    ).with_columns(
        pl.when(pl.col("address_registration_month_cohort_key").is_not_null())
        .then(pl.len().over("address_registration_month_cohort_key"))
        .otherwise(0)
        .cast(pl.Int32)
        .alias("address_registration_month_cohort_company_count")
    ).with_columns(
        (
            pl.col("address_registration_month_cohort_company_count")
            >= config.shared_address_minimum_companies
        ).alias("address_registration_month_cohort_signal")
    )

    outcome_eligible_statuses = set(
        match_decision_frame.filter(pl.col("outcome_eligible"))[
            "review_status"
        ].to_list()
    )
    allowed_outcome_eligible_statuses = {"accepted", "accepted_reviewed"}
    if not outcome_eligible_statuses <= allowed_outcome_eligible_statuses:
        raise SignalObservationError(
            "Outcome-eligible matches contain an unapproved review status."
        )
    label_match_policy = (
        "accepted_unique_exact_cin_or_completed_human_review"
        if "accepted_reviewed" in outcome_eligible_statuses
        else "accepted_unique_exact_cin_only"
    )
    accepted_events = match_decision_frame.filter(
        pl.col("outcome_eligible")
        & pl.col("accepted_company_snapshot_id").is_not_null()
    ).join(
        adverse_frame.select("adverse_event_id", "event_date"),
        on="adverse_event_id",
        how="inner",
        validate="1:1",
    )
    prior_outcomes = accepted_events.filter(
        pl.col("event_date") <= config.feature_cutoff
    ).select(
        pl.col("accepted_company_snapshot_id").alias("company_snapshot_id")
    ).unique().with_columns(pl.lit(True).alias("has_prior_adverse_outcome"))
    window_outcomes = accepted_events.filter(
        pl.col("event_date").is_between(
            config.outcome_window_start,
            config.outcome_window_end,
            closed="both",
        )
    ).group_by("accepted_company_snapshot_id").agg(
        pl.len().cast(pl.Int32).alias("adverse_event_count"),
        pl.col("event_date").min().alias("first_adverse_event_date"),
    ).rename({"accepted_company_snapshot_id": "company_snapshot_id"})

    return merchant_population.join(
        prior_outcomes,
        on="company_snapshot_id",
        how="left",
    ).join(
        window_outcomes,
        on="company_snapshot_id",
        how="left",
    ).with_columns(
        pl.col("has_prior_adverse_outcome").fill_null(False),
        pl.col("adverse_event_count").fill_null(0),
    ).with_columns(
        (pl.col("adverse_event_count") > 0).alias("observed_adverse_outcome"),
        (
            pl.col("cin_is_valid_format")
            & pl.col("registration_date").is_not_null()
            & (pl.col("registration_date") <= config.feature_cutoff)
            & ~pl.col("has_prior_adverse_outcome")
        ).alias("evaluation_eligible"),
        pl.lit(config.run_id).alias("run_id"),
        pl.lit(config.feature_cutoff).cast(pl.Date).alias("feature_cutoff"),
        pl.lit(config.outcome_window_start)
        .cast(pl.Date)
        .alias("outcome_window_start"),
        pl.lit(config.outcome_window_end)
        .cast(pl.Date)
        .alias("outcome_window_end"),
        pl.lit("observed_nse_adverse_regulatory_outcome").alias("target_name"),
        pl.lit(label_match_policy).alias("label_match_policy"),
        pl.lit(SIGNAL_OBSERVATION_SCHEMA_VERSION).alias("schema_version"),
    ).select(
        pl.concat_str(
            [pl.lit(config.run_id), pl.col("company_snapshot_id")], separator=":"
        ).alias("signal_observation_id"),
        "run_id",
        "company_snapshot_id",
        "source_snapshot_id",
        "cin",
        "company_name",
        "registration_date",
        "state_code",
        "nic_code",
        "paid_up_capital_inr",
        "address_normalized",
        "address_group_key",
        "shared_address_company_count",
        "shared_address_signal",
        "address_registration_month_cohort_key",
        "address_registration_month_cohort_company_count",
        "address_registration_month_cohort_signal",
        "has_prior_adverse_outcome",
        "adverse_event_count",
        "first_adverse_event_date",
        "observed_adverse_outcome",
        "evaluation_eligible",
        "feature_cutoff",
        "outcome_window_start",
        "outcome_window_end",
        "target_name",
        "label_match_policy",
        "schema_version",
    ).sort("company_snapshot_id")
