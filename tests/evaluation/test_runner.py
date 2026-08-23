from __future__ import annotations

from datetime import date

import polars as pl

from assay.evaluation.runner import build_evaluation_slices


def test_evaluation_slices_keep_states_disjoint_and_remove_prior_risk() -> None:
    observations = pl.DataFrame(
        {
            "state_code": ["DL", "MH", "KA", "TN"],
            "evaluation_eligible": [True] * 4,
            "first_adverse_event_date": [
                date(2025, 1, 1),
                date(2026, 2, 1),
                None,
                None,
            ],
            "observed_adverse_outcome": [True, True, False, False],
            "outcome_window_start": [date(2023, 11, 4)] * 4,
            "outcome_window_end": [date(2026, 8, 22)] * 4,
        }
    )

    slices, holdout_states, development_states = build_evaluation_slices(
        observations,
        temporal_holdout_start=date(2026, 1, 1),
        geography_holdout_modulus=2,
        geography_holdout_remainder=0,
    )

    assert set(holdout_states).isdisjoint(development_states)
    assert set(holdout_states) | set(development_states) == {"DL", "MH", "KA", "TN"}
    temporal_holdout = slices["temporal_holdout"]
    assert temporal_holdout.filter(pl.col("slice_eligible")).height == 3
    assert temporal_holdout.filter(pl.col("slice_target")).height == 1
    assert slices["temporal_development"].filter(pl.col("slice_target")).height == 1

