"""Group-level adverse-outcome concentration against a fixed-label null."""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class GroupConcentrationError(RuntimeError):
    """A group concentration test cannot satisfy its evidence contract."""


class GroupConcentrationReport(BaseModel):
    """Positive-pair concentration and a reproducible permutation null."""

    model_config = ConfigDict(frozen=True)

    slice_name: str = Field(min_length=1)
    group_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    grouped_population_rows: int = Field(ge=0)
    group_rows: int = Field(ge=0)
    positive_outcome_rows: int = Field(ge=0)
    observed_same_group_positive_pairs: int = Field(ge=0)
    expected_same_group_positive_pairs: float = Field(ge=0.0)
    concentration_ratio: float | None = Field(default=None, ge=0.0)
    monte_carlo_p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    null_simulations: int = Field(ge=0)
    random_seed: int = Field(ge=0)
    test_status: str


def evaluate_group_concentration(
    observation_frame: pl.DataFrame,
    *,
    slice_name: str,
    group_column: str,
    target_column: str = "observed_adverse_outcome",
    eligibility_column: str = "evaluation_eligible",
    null_simulations: int = 1_000,
    random_seed: int = 106,
) -> GroupConcentrationReport:
    """Test whether fixed positive labels cluster more than random allocation."""

    if null_simulations < 100:
        raise GroupConcentrationError("At least 100 null simulations are required.")
    required_columns = {group_column, target_column, eligibility_column}
    if missing_columns := required_columns - set(observation_frame.columns):
        raise GroupConcentrationError(
            f"Observation frame is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    grouped_frame = observation_frame.filter(
        pl.col(eligibility_column)
        & pl.col(group_column).is_not_null()
        & (pl.col(group_column) != "")
    ).select(
        pl.col(group_column).cast(pl.String),
        pl.col(target_column).cast(pl.Boolean),
    )
    grouped_population_rows = grouped_frame.height
    if grouped_population_rows == 0:
        return GroupConcentrationReport(
            slice_name=slice_name,
            group_column=group_column,
            target_column=target_column,
            grouped_population_rows=0,
            group_rows=0,
            positive_outcome_rows=0,
            observed_same_group_positive_pairs=0,
            expected_same_group_positive_pairs=0.0,
            null_simulations=0,
            random_seed=random_seed,
            test_status="empty_grouped_population",
        )

    group_summary = grouped_frame.group_by(group_column).agg(
        pl.len().alias("group_size"),
        pl.col(target_column).sum().alias("positive_count"),
    )
    group_sizes = np.asarray(group_summary["group_size"], dtype=np.int64)
    positive_counts = np.asarray(group_summary["positive_count"], dtype=np.int64)
    positive_outcome_rows = int(positive_counts.sum())
    observed_positive_pairs = int(
        np.sum(positive_counts * (positive_counts - 1) // 2)
    )
    possible_group_pairs = int(np.sum(group_sizes * (group_sizes - 1) // 2))
    population_pairs = grouped_population_rows * (grouped_population_rows - 1) / 2
    positive_pairs = positive_outcome_rows * (positive_outcome_rows - 1) / 2
    expected_positive_pairs = (
        possible_group_pairs * positive_pairs / population_pairs
        if population_pairs > 0
        else 0.0
    )
    concentration_ratio = (
        observed_positive_pairs / expected_positive_pairs
        if expected_positive_pairs > 0
        else None
    )
    if positive_outcome_rows < 2 or possible_group_pairs == 0:
        return GroupConcentrationReport(
            slice_name=slice_name,
            group_column=group_column,
            target_column=target_column,
            grouped_population_rows=grouped_population_rows,
            group_rows=group_summary.height,
            positive_outcome_rows=positive_outcome_rows,
            observed_same_group_positive_pairs=observed_positive_pairs,
            expected_same_group_positive_pairs=expected_positive_pairs,
            concentration_ratio=concentration_ratio,
            null_simulations=0,
            random_seed=random_seed,
            test_status="insufficient_positive_pairs",
        )

    entity_group_indices = np.repeat(
        np.arange(group_summary.height, dtype=np.int32),
        group_sizes,
    )
    random_generator = np.random.default_rng(random_seed)
    simulated_pairs = np.empty(null_simulations, dtype=np.int64)
    for simulation_index in range(null_simulations):
        selected_entity_indices = random_generator.choice(
            grouped_population_rows,
            size=positive_outcome_rows,
            replace=False,
            shuffle=False,
        )
        simulated_positive_counts = np.bincount(
            entity_group_indices[selected_entity_indices],
            minlength=group_summary.height,
        )
        simulated_pairs[simulation_index] = np.sum(
            simulated_positive_counts * (simulated_positive_counts - 1) // 2
        )
    monte_carlo_p_value = float(
        (1 + np.count_nonzero(simulated_pairs >= observed_positive_pairs))
        / (null_simulations + 1)
    )
    return GroupConcentrationReport(
        slice_name=slice_name,
        group_column=group_column,
        target_column=target_column,
        grouped_population_rows=grouped_population_rows,
        group_rows=group_summary.height,
        positive_outcome_rows=positive_outcome_rows,
        observed_same_group_positive_pairs=observed_positive_pairs,
        expected_same_group_positive_pairs=expected_positive_pairs,
        concentration_ratio=concentration_ratio,
        monte_carlo_p_value=monte_carlo_p_value,
        null_simulations=null_simulations,
        random_seed=random_seed,
        test_status="passed",
    )

