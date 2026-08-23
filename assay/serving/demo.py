"""Curate a reproducible demo scenario set for the merchant-risk endpoint.

Every scenario is a real public MCA company drawn from the frozen serving index,
plus the failure paths an analyst tool must handle: an invalid identifier, a
well-formed identifier outside the assessable population, and a company name
that several public entities share. Nothing here is a synthetic merchant and
nothing here is a Razorpay account.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.serving.assessment import MerchantRiskAssessor
from assay.serving.bands import RISK_BAND_ORDER

DEMO_SCENARIO_SCHEMA_VERSION = "1.0.0"
DEMO_SAMPLE_ROWS = 400_000
DEMO_SAMPLE_SEED = 106
UNASSESSABLE_DEMO_CIN = "U74999DL9999PTC999999"
INVALID_DEMO_CIN = "NOT-A-REAL-CIN"


class DemoScenarioError(RuntimeError):
    """A demo scenario set could not be curated from the frozen index."""


class DemoScenario(BaseModel):
    """One reproducible request an analyst or reviewer can replay."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    title: str
    why_it_matters: str
    request: dict[str, str]
    expected_http_status: int = Field(ge=200, le=599)
    expected_outcome: str
    expected_risk_band: str | None
    observed_score: float | None


class DemoScenarioSet(BaseModel):
    """Curated public examples pinned to one index and one model."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = DEMO_SCENARIO_SCHEMA_VERSION
    merchant_index_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_cutoff: str
    sampled_rows: int = Field(gt=0)
    sample_seed: int
    provenance: str = (
        "Scenarios are real public MCA companies selected by score band from "
        "the frozen serving index. No synthetic or Razorpay merchant data is "
        "included."
    )
    scenarios: tuple[DemoScenario, ...]


def _band_rationale(band: str) -> str:
    return {
        "ELEVATED_REVIEW": (
            "Shows the top public-insolvency review queue an analyst would "
            "work first during enhanced due diligence."
        ),
        "WATCH": (
            "Shows a merchant worth periodic monitoring rather than immediate "
            "enhanced due diligence."
        ),
        "STANDARD": (
            "Shows the wider queue that only justifies routine KYB checks."
        ),
        "LOW_SIGNAL": (
            "Shows the majority case: no elevated public insolvency signal, "
            "which is not evidence of solvency."
        ),
    }[band]


def curate_demo_scenarios(
    assessor: MerchantRiskAssessor,
    *,
    sample_rows: int = DEMO_SAMPLE_ROWS,
    sample_seed: int = DEMO_SAMPLE_SEED,
) -> DemoScenarioSet:
    """Score a deterministic sample and pick one company per risk band."""

    index_report = assessor.index_report
    index_path = Path(index_report.index_artifact.path)
    sampled = (
        pl.scan_parquet(index_path)
        .filter(
            pl.col("cin").hash(sample_seed) % 1_000_000
            < max(1, round(sample_rows / index_report.indexed_companies * 1_000_000))
        )
        .collect()
    )
    if sampled.is_empty():
        raise DemoScenarioError("Demo sampling produced no candidate companies.")
    scores = assessor.score_frame(sampled)
    scored = sampled.select("cin", "company_name").with_columns(
        pl.Series("score", scores)
    )
    scenarios: list[DemoScenario] = []
    for band in RISK_BAND_ORDER:
        threshold = next(
            item
            for item in assessor.band_scale.thresholds
            if item.band == band
        )
        higher_minimums = [
            item.minimum_score
            for item in assessor.band_scale.thresholds
            if item.minimum_score > threshold.minimum_score
        ]
        upper_bound = min(higher_minimums) if higher_minimums else 1.01
        band_frame = scored.filter(
            (pl.col("score") >= threshold.minimum_score)
            & (pl.col("score") < upper_bound)
        ).sort("score", descending=True)
        # The median of a band is representative; the edges are boundary cases.
        if band_frame.is_empty():
            continue
        row = band_frame.row(band_frame.height // 2, named=True)
        observed_band = assessor.band_scale.band_for(float(row["score"])).band
        if observed_band != band:
            raise DemoScenarioError(
                f"Demo scenario for {band} scored into {observed_band}."
            )
        scenarios.append(
            DemoScenario(
                scenario_id=f"band_{band.lower()}",
                title=f"{band} public company",
                why_it_matters=_band_rationale(band),
                request={"cin": str(row["cin"])},
                expected_http_status=200,
                expected_outcome="matched",
                expected_risk_band=band,
                observed_score=round(float(row["score"]), 8),
            )
        )
    ambiguous_frame = (
        pl.scan_parquet(index_path)
        .filter(pl.col("strict_name_company_count") > 1)
        .select("company_name")
        .head(1)
        .collect()
    )
    if not ambiguous_frame.is_empty():
        scenarios.append(
            DemoScenario(
                scenario_id="ambiguous_company_name",
                title="Company name shared by several public entities",
                why_it_matters=(
                    "Name-only KYB input must never silently resolve to one "
                    "entity; the caller is asked for a CIN instead."
                ),
                request={"company_name": str(ambiguous_frame["company_name"][0])},
                expected_http_status=409,
                expected_outcome="ambiguous_name",
                expected_risk_band=None,
                observed_score=None,
            )
        )
    scenarios.extend(
        (
            DemoScenario(
                scenario_id="invalid_identifier",
                title="Malformed CIN",
                why_it_matters=(
                    "An onboarding form can submit a typo; the service must "
                    "reject the identifier instead of guessing a company."
                ),
                request={"cin": INVALID_DEMO_CIN},
                expected_http_status=422,
                expected_outcome="invalid_identifier",
                expected_risk_band=None,
                observed_score=None,
            ),
            DemoScenario(
                scenario_id="unassessable_company",
                title="Well-formed CIN outside the frozen population",
                why_it_matters=(
                    "Companies registered after the feature cutoff, LLPs, and "
                    "companies already in CIRP are not assessable. The service "
                    "says so instead of returning a stale or invented score."
                ),
                request={"cin": UNASSESSABLE_DEMO_CIN},
                expected_http_status=404,
                expected_outcome="no_match",
                expected_risk_band=None,
                observed_score=None,
            ),
        )
    )
    return DemoScenarioSet(
        merchant_index_run_id=index_report.run_id,
        feature_cutoff=index_report.feature_cutoff.isoformat(),
        sampled_rows=sampled.height,
        sample_seed=sample_seed,
        scenarios=tuple(scenarios),
    )
