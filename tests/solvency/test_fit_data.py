from __future__ import annotations

import polars as pl
import pytest

from assay.solvency.fit_data import (
    SolvencyFitDataConfig,
    SolvencyFitDataError,
    build_solvency_fit_data,
)


def _model_frame(
    *,
    training_negatives: int = 40,
    validation_negatives: int = 20,
    training_positives: int = 3,
    validation_positives: int = 2,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, negative_rows, positive_rows in (
        ("training", training_negatives, training_positives),
        ("validation", validation_negatives, validation_positives),
        ("geography_test", 15, 4),
        ("temporal_test", 15, 4),
    ):
        for index in range(positive_rows):
            rows.append(
                {
                    "company_snapshot_id": f"{split_name}-positive-{index}",
                    "dataset_split": split_name,
                    "target": 1,
                }
            )
        for index in range(negative_rows):
            rows.append(
                {
                    "company_snapshot_id": f"{split_name}-negative-{index}",
                    "dataset_split": split_name,
                    "target": 0,
                }
            )
    return pl.DataFrame(rows, schema_overrides={"target": pl.Int8})


def _config(**overrides: object) -> SolvencyFitDataConfig:
    return SolvencyFitDataConfig(
        model_data_report_path="unused.json",  # type: ignore[arg-type]
        sampled_training_negative_rows=10,
        sampled_validation_negative_rows=5,
        **overrides,  # type: ignore[arg-type]
    )


def test_fit_data_keeps_every_positive_and_samples_only_negatives() -> None:
    fit_frame, split_evidence = build_solvency_fit_data(_model_frame(), _config())

    counts = {
        (row["dataset_split"], row["target"]): row["rows"]
        for row in fit_frame.group_by("dataset_split", "target")
        .agg(pl.len().alias("rows"))
        .to_dicts()
    }
    assert counts == {
        ("training", 1): 3,
        ("training", 0): 10,
        ("validation", 1): 2,
        ("validation", 0): 5,
    }
    assert split_evidence["training"].full_negative_rows == 40
    assert split_evidence["training"].negative_restoration_weight == 4.0
    assert split_evidence["validation"].negative_restoration_weight == 4.0


def test_fit_data_never_writes_a_holdout_row() -> None:
    """A holdout that passed through a sampler would not be a holdout."""

    fit_frame, _ = build_solvency_fit_data(_model_frame(), _config())

    assert sorted(fit_frame["dataset_split"].unique().to_list()) == [
        "training",
        "validation",
    ]


def test_fit_data_selection_is_deterministic_and_seed_dependent() -> None:
    """The same model data and seed must always yield the same payload."""

    first, _ = build_solvency_fit_data(_model_frame(), _config())
    again, _ = build_solvency_fit_data(_model_frame(), _config())
    shuffled_source, _ = build_solvency_fit_data(
        _model_frame().reverse(), _config()
    )
    other_seed, _ = build_solvency_fit_data(
        _model_frame(), _config(sample_hash_seed=7)
    )

    assert first.equals(again)
    assert first.equals(shuffled_source)
    assert not first.equals(other_seed)


def test_fit_data_refuses_to_sample_more_negatives_than_exist() -> None:
    with pytest.raises(SolvencyFitDataError, match="fewer than the frozen sample"):
        build_solvency_fit_data(
            _model_frame(training_negatives=4),
            _config(),
        )


def test_fit_data_refuses_an_unsampleable_holdout_split() -> None:
    with pytest.raises(SolvencyFitDataError, match="never sampled"):
        _config().sampled_negative_rows("geography_test")


def test_fit_data_requires_the_split_and_target_columns() -> None:
    with pytest.raises(SolvencyFitDataError, match="missing"):
        build_solvency_fit_data(
            _model_frame().drop("dataset_split"),
            _config(),
        )
