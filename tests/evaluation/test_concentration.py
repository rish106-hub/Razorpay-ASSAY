from __future__ import annotations

import polars as pl

from assay.evaluation.concentration import evaluate_group_concentration


def test_concentration_test_detects_positive_pair_clustering() -> None:
    observations = pl.DataFrame(
        {
            "evaluation_eligible": [True] * 20,
            "address_group_key": [
                *("GROUP-A" for _ in range(5)),
                *("GROUP-B" for _ in range(5)),
                *("GROUP-C" for _ in range(5)),
                *("GROUP-D" for _ in range(5)),
            ],
            "observed_adverse_outcome": [True, True, True, True, False]
            + [False] * 15,
        }
    )

    report = evaluate_group_concentration(
        observations,
        slice_name="all_eligible_entities",
        group_column="address_group_key",
        null_simulations=2_000,
        random_seed=106,
    )

    assert report.observed_same_group_positive_pairs == 6
    assert report.concentration_ratio is not None
    assert report.concentration_ratio > 4
    assert report.monte_carlo_p_value is not None
    assert report.monte_carlo_p_value < 0.05
