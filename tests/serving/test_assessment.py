from __future__ import annotations

import polars as pl
import pytest

from assay.serving.assessment import (
    MAXIMUM_EXPLANATION_FACTORS,
    MerchantAssessmentError,
    build_explanation_factors,
)


def _feature_row() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cin": ["U72900KA2019PTC000002"],
            "company_name": ["Acme Retail Private Limited"],
            "authorised_capital_inr": [1_000_000.0],
            "paid_up_capital_inr": [500_000.0],
            "shared_address_company_count": [37],
            "address_registration_month_company_count": [12],
            "company_age_years": [4.1],
            "paid_to_authorised_capital_ratio": [0.5],
            "company_status": ["ACTIVE"],
            "state_code": ["karnataka"],
        }
    )


def _contributions(**values: float) -> pl.DataFrame:
    payload: dict[str, list[float]] = {
        "company_age_years": [0.0],
        "log_authorised_capital_inr": [0.0],
        "log_paid_up_capital_inr": [0.0],
        "paid_to_authorised_capital_ratio": [0.0],
        "log_shared_address_company_count": [0.0],
        "log_address_registration_month_company_count": [0.0],
        "company_status": [0.0],
        "state_code": [0.0],
        "model_bias": [-4.2],
    }
    for feature_name, value in values.items():
        payload[feature_name] = [value]
    return pl.DataFrame(payload)


def test_factors_rank_by_absolute_contribution_and_name_the_direction() -> None:
    factors = build_explanation_factors(
        _feature_row(),
        _contributions(
            log_shared_address_company_count=1.4,
            company_status=-0.9,
            company_age_years=0.2,
        ),
    )

    assert [factor.feature for factor in factors] == [
        "log_shared_address_company_count",
        "company_status",
        "company_age_years",
    ]
    assert factors[0].direction == "increases_risk"
    assert factors[0].value == "37"
    assert "raises" in factors[0].evidence
    assert factors[1].direction == "decreases_risk"
    assert factors[1].value == "ACTIVE"
    assert "lowers" in factors[1].evidence


def test_factors_render_capital_features_in_rupees() -> None:
    factors = build_explanation_factors(
        _feature_row(),
        _contributions(log_authorised_capital_inr=0.8),
    )

    assert factors[0].label == "Authorised capital"
    assert factors[0].value == "INR 1,000,000"


def test_factors_exclude_features_the_model_did_not_move() -> None:
    factors = build_explanation_factors(
        _feature_row(),
        _contributions(company_age_years=0.5),
    )

    assert [factor.feature for factor in factors] == ["company_age_years"]


def test_factors_are_capped_at_the_configured_maximum() -> None:
    factors = build_explanation_factors(
        _feature_row(),
        _contributions(
            company_age_years=0.9,
            log_authorised_capital_inr=0.8,
            log_paid_up_capital_inr=0.7,
            paid_to_authorised_capital_ratio=0.6,
            log_shared_address_company_count=0.5,
            log_address_registration_month_company_count=0.4,
            company_status=0.3,
            state_code=0.2,
        ),
    )

    assert len(factors) == MAXIMUM_EXPLANATION_FACTORS


def test_factors_reject_a_multi_row_request() -> None:
    doubled = pl.concat([_feature_row(), _feature_row()], how="vertical")
    with pytest.raises(MerchantAssessmentError, match="exactly one"):
        build_explanation_factors(doubled, _contributions())
