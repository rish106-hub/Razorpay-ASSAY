"""Build leakage-safe merchant-solvency observations from MCA and IBBI."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from assay.contracts.cin import (
    cin_decode_expressions,
    cin_disagreement_expressions,
)

SOLVENCY_OBSERVATION_SCHEMA_VERSION = "1.2.0"

#: Neutral value for the ROC-serial gap of a company that has no comparable
#: peer. A real gap is a non-negative count of serial positions, so -1 cannot
#: collide with one and never has to be imputed away.
NO_COMPARABLE_ROC_SERIAL_PEER = -1

_ROC_SERIAL_WORKING_COLUMNS = (
    "cin_roc_serial",
    "cin_registrar_year_cohort_key",
)

ADDRESS_CLUSTER_SHAPE_PRECISION = 6
_ADDRESS_CLUSTER_SHAPE_WORKING_COLUMNS = (
    "cluster_registration_span_days",
    "cluster_nic_division_distinct",
    "cluster_authorised_capital_mean",
    "cluster_authorised_capital_std",
    "cluster_distinct_name_head_count",
    "cluster_named_company_count",
    "cluster_dated_company_count",
    "cluster_max_month_company_share",
    "cluster_month_entropy_nats",
)


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


def _normalised_name_head() -> pl.Expr:
    """Take the first alphanumeric token of a company name, lowercased.

    A name that carries no ASCII alphanumeric token yields an empty head, which
    the cluster aggregates treat as unobserved rather than as a shared stem.
    """

    return (
        pl.col("company_name")
        .fill_null("")
        .str.to_lowercase()
        .str.extract(r"([a-z0-9]+)", 1)
        .fill_null("")
        .alias("company_name_head")
    )


def _address_cluster_shape_frame(merchant_frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate one address-cluster shape row per non-empty address group key.

    Cluster shape separates the two populations that a plain shared-address count
    collapses together. A registered-office service provider hosts companies
    incorporated across many years, many months, and many industries. A shell
    factory incorporates its companies inside one narrow window, on templated
    capital, under one name stem. Both look identical to a count.

    Every input column here is read from the pre-cutoff MCA company snapshot
    only: address group key, registration date, NIC division, authorised
    capital, and company name. No IBBI outcome column, no post-cutoff row, and
    no label-derived value participates, so these features cannot leak the CIRP
    public-announcement target. The cluster population is the set of company
    snapshot rows sharing an address group key, which is the same population the
    existing shared_address_company_count measures whenever every address row
    resolves to a company row.

    Each statistic is computed over the rows where its input is observed, so an
    unknown registration date, NIC code, capital, or name removes that row from
    that statistic instead of inventing a value for it. When no row in a cluster
    observes an input, the statistic is undefined and the caller substitutes the
    neutral single-company value: span 0 days, entropy 0.0, largest-month share
    1.0, capital coefficient of variation 0.0, and distinct name-head ratio 1.0.

    An empty or missing address group key stays "no cluster" exactly as the
    frozen shared_address_company_count treats it, and the caller emits the same
    neutral single-company values for it, because a company with no observed
    shared address behaves like a cluster of one. The two populations remain
    separable downstream: only a genuine cluster carries a non-zero
    shared_address_company_count.
    """

    clustered_frame = merchant_frame.filter(pl.col("address_group_key") != "")
    month_shape_frame = (
        clustered_frame.filter(pl.col("registration_date").is_not_null())
        .select(
            "address_group_key",
            pl.col("registration_date")
            .dt.strftime("%Y-%m")
            .alias("cluster_registration_month"),
        )
        .group_by("address_group_key", "cluster_registration_month")
        .agg(pl.len().alias("month_company_count"))
        .with_columns(
            (
                pl.col("month_company_count")
                / pl.col("month_company_count").sum().over("address_group_key")
            ).alias("month_company_share")
        )
        .group_by("address_group_key")
        .agg(
            pl.col("month_company_count")
            .sum()
            .cast(pl.Int64)
            .alias("cluster_dated_company_count"),
            pl.col("month_company_share")
            .max()
            .alias("cluster_max_month_company_share"),
            (
                -(
                    pl.col("month_company_share")
                    * pl.col("month_company_share").log()
                ).sum()
            ).alias("cluster_month_entropy_nats"),
        )
    )
    return (
        clustered_frame.select(
            "address_group_key",
            "registration_date",
            "nic_division",
            "authorised_capital_inr",
            _normalised_name_head(),
        )
        .group_by("address_group_key")
        .agg(
            (
                pl.col("registration_date").max()
                - pl.col("registration_date").min()
            )
            .dt.total_days()
            .alias("cluster_registration_span_days"),
            pl.col("nic_division")
            .filter(pl.col("nic_division") != "")
            .n_unique()
            .cast(pl.Int64)
            .alias("cluster_nic_division_distinct"),
            pl.col("authorised_capital_inr")
            .cast(pl.Float64)
            .mean()
            .alias("cluster_authorised_capital_mean"),
            pl.col("authorised_capital_inr")
            .cast(pl.Float64)
            .std(ddof=0)
            .alias("cluster_authorised_capital_std"),
            pl.col("company_name_head")
            .filter(pl.col("company_name_head") != "")
            .n_unique()
            .cast(pl.Int64)
            .alias("cluster_distinct_name_head_count"),
            (pl.col("company_name_head") != "")
            .sum()
            .cast(pl.Int64)
            .alias("cluster_named_company_count"),
        )
        .join(
            month_shape_frame,
            on="address_group_key",
            how="left",
            validate="1:1",
        )
    )


def _roc_serial_adjacency_frame(merchant_frame: pl.DataFrame) -> pl.DataFrame:
    """Measure ROC-serial adjacency per company, one row per company snapshot.

    Segment 6 of a CIN is allocated sequentially by a registrar office, so two
    companies filed in the same batch carry near-adjacent serials. Serials are
    only comparable inside one registrar-year cohort: serial 4021 from Karnataka
    2010 and serial 4022 from Maharashtra 2019 are unrelated numbers.

    Adjacency is therefore measured inside the intersection of an address
    cluster and a registrar-year cohort. Two companies at one address, in one
    registrar's one year, with consecutive serials, were filed together. That is
    the closest available proxy for the director/agent graph this project has
    deliberately not acquired, and it is a proxy, not a replacement.

    Three columns are emitted:

    ``registrar_year_cohort_company_count``
        Size of the whole registrar-year cohort. It is the denominator: a gap of
        five inside a cohort of two hundred means something a gap of five inside
        a cohort of fifty thousand does not.
    ``address_cluster_cohort_peer_count``
        Other companies sharing both the address cluster and the cohort, so
        having a comparable serial at all.
    ``address_cluster_roc_serial_min_gap``
        Smallest absolute serial distance to such a peer, or
        :data:`NO_COMPARABLE_ROC_SERIAL_PEER` when there is none.

    Every input is a pre-cutoff MCA column - the identifier itself, the address
    group key, and the registration date already inside the identifier - so no
    IBBI outcome and no post-cutoff row participates. The minimum gap is read
    off sorted neighbours inside each group, never from all pairs, so the
    466,414 shared-address rows stay tractable.
    """

    cohort_counts = merchant_frame.filter(
        pl.col("cin_registrar_year_cohort_key").is_not_null()
    ).group_by("cin_registrar_year_cohort_key").agg(
        pl.len().cast(pl.Int32).alias("registrar_year_cohort_company_count")
    )
    comparable = (
        merchant_frame.filter(
            (pl.col("address_group_key") != "")
            & pl.col("cin_registrar_year_cohort_key").is_not_null()
            & pl.col("cin_roc_serial").is_not_null()
        )
        .select(
            "company_snapshot_id",
            "address_group_key",
            "cin_registrar_year_cohort_key",
            "cin_roc_serial",
        )
        .sort(
            "address_group_key",
            "cin_registrar_year_cohort_key",
            "cin_roc_serial",
        )
    )
    cluster_cohort = ("address_group_key", "cin_registrar_year_cohort_key")
    adjacency = comparable.select(
        "company_snapshot_id",
        (pl.len().over(cluster_cohort) - 1)
        .cast(pl.Int32)
        .alias("address_cluster_cohort_peer_count"),
        pl.min_horizontal(
            pl.col("cin_roc_serial")
            .diff()
            .over(cluster_cohort)
            .abs(),
            pl.col("cin_roc_serial")
            .diff(-1)
            .over(cluster_cohort)
            .abs(),
        )
        .cast(pl.Int32)
        .alias("address_cluster_roc_serial_min_gap"),
    )
    return merchant_frame.select(
        "company_snapshot_id", "cin_registrar_year_cohort_key"
    ).join(
        cohort_counts,
        on="cin_registrar_year_cohort_key",
        how="left",
    ).join(
        adjacency,
        on="company_snapshot_id",
        how="left",
    ).select(
        "company_snapshot_id",
        pl.col("registrar_year_cohort_company_count").fill_null(0).cast(pl.Int32),
        pl.col("address_cluster_cohort_peer_count").fill_null(0).cast(pl.Int32),
        pl.col("address_cluster_roc_serial_min_gap")
        .fill_null(NO_COMPARABLE_ROC_SERIAL_PEER)
        .cast(pl.Int32),
    )


def build_solvency_observations(
    company_frame: pl.DataFrame,
    address_frame: pl.DataFrame,
    cirp_announcement_frame: pl.DataFrame,
    config: SolvencyObservationConfig,
) -> pl.DataFrame:
    """Create one pre-outcome feature row per MCA company with an exact-CIN label.

    Alongside the two frozen shared-address counts, each row carries the shape of
    its address cluster: registration span, registration-month entropy, largest
    registration-month share, distinct NIC divisions, authorised-capital
    coefficient of variation, and distinct name-head ratio. Counts alone cannot
    tell a registered-office service provider from a shell factory, because both
    can host the same number of companies at one address; the shape statistics
    can. See _address_cluster_shape_frame for the leakage argument and the
    edge-case contract.

    Each row also carries what its own identifier says about it. The CIN encodes
    an incorporation year, an NIC division, a listing letter, and a registrar
    state, all of which the MCA record also stores in separate columns; the four
    tri-state disagreement flags and their count expose where the identifier and
    the record contradict each other. The decoded ROC serial supplies three
    adjacency columns - see _roc_serial_adjacency_frame - which is the nearest
    proxy available for filing-agent batches without director data.

    Both additions are derived only from pre-cutoff MCA columns, so neither can
    leak the CIRP public-announcement target.
    """

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
        *cin_decode_expressions(),
        *cin_disagreement_expressions(),
    ).with_columns(
        pl.when(pl.col("address_group_key") != "")
        .then(pl.len().over("address_registration_month_key"))
        .otherwise(0)
        .cast(pl.Int32)
        .alias("address_registration_month_company_count")
    )

    shaped_features = merchant_features.join(
        _address_cluster_shape_frame(merchant_features),
        on="address_group_key",
        how="left",
        validate="m:1",
        maintain_order="left",
    ).with_columns(
        pl.col("cluster_registration_span_days")
        .fill_null(0)
        .cast(pl.Int32)
        .alias("address_cluster_registration_span_days"),
        pl.when(pl.col("cluster_dated_company_count") > 1)
        .then(
            (
                pl.col("cluster_month_entropy_nats")
                / pl.col("cluster_dated_company_count").cast(pl.Float64).log()
            ).abs()
        )
        .otherwise(0.0)
        .fill_null(0.0)
        .round(ADDRESS_CLUSTER_SHAPE_PRECISION)
        .alias("address_cluster_registration_month_entropy"),
        pl.col("cluster_max_month_company_share")
        .fill_null(1.0)
        .round(ADDRESS_CLUSTER_SHAPE_PRECISION)
        .alias("address_cluster_max_month_share"),
        pl.when(pl.col("address_group_key") != "")
        .then(pl.col("cluster_nic_division_distinct").fill_null(0))
        .otherwise((pl.col("nic_division") != "").cast(pl.Int64))
        .cast(pl.Int32)
        .alias("address_cluster_nic_division_distinct"),
        pl.when(pl.col("cluster_authorised_capital_mean") > 0.0)
        .then(
            pl.col("cluster_authorised_capital_std")
            / pl.col("cluster_authorised_capital_mean")
        )
        .otherwise(0.0)
        .fill_null(0.0)
        .round(ADDRESS_CLUSTER_SHAPE_PRECISION)
        .alias("address_cluster_authorised_capital_cv"),
        pl.when(pl.col("cluster_named_company_count") > 0)
        .then(
            pl.col("cluster_distinct_name_head_count").cast(pl.Float64)
            / pl.col("cluster_named_company_count").cast(pl.Float64)
        )
        .otherwise(1.0)
        .fill_null(1.0)
        .round(ADDRESS_CLUSTER_SHAPE_PRECISION)
        .alias("address_cluster_distinct_name_head_ratio"),
    ).drop(_ADDRESS_CLUSTER_SHAPE_WORKING_COLUMNS)

    adjacent_features = shaped_features.join(
        _roc_serial_adjacency_frame(shaped_features),
        on="company_snapshot_id",
        how="left",
        validate="1:1",
        maintain_order="left",
    ).drop(_ROC_SERIAL_WORKING_COLUMNS)

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
    return adjacent_features.join(prior_cirp, on="cin", how="left").join(
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
