"""Assemble one evidence-backed public merchant-risk assessment.

The assessment answers a KYB or enhanced-due-diligence question: given a public
Indian company identifier, where should an analyst place this merchant in a
review queue, and what public evidence puts it there. It resolves identity,
scores the frozen as-of feature row, maps the score to a measured review-capacity
band, and attaches the model contributions and source provenance behind it.

It never decides a merchant outcome. The score is
`cirp_public_announcement_score`: the modelled probability of a future public
IBBI CIRP announcement. It is not a fraud score, not a payment-abuse score, and
not an onboarding rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.serving.bands import RiskBandScale, RiskBandThreshold
from assay.serving.index import MerchantRiskIndexReport
from assay.serving.lookup import MerchantLookupResult, MerchantRiskIndexStore
from assay.solvency.artifact import SolvencyModelPackage
from assay.solvency.evaluation import SolvencyHoldoutEvaluationReport
from assay.solvency.training_data import SOLVENCY_NUMERIC_FEATURES

MERCHANT_ASSESSMENT_SCHEMA_VERSION = "1.0.0"
MERCHANT_RISK_SCORE_NAME = "cirp_public_announcement_score"
MAXIMUM_EXPLANATION_FACTORS = 6
MODERATE_CONFIDENCE_MAXIMUM_SNAPSHOT_AGE_DAYS = 400

FEATURE_LABELS: dict[str, str] = {
    "company_age_years": "Company age at the feature cutoff",
    "log_authorised_capital_inr": "Authorised capital",
    "log_paid_up_capital_inr": "Paid-up capital",
    "paid_to_authorised_capital_ratio": "Paid-up to authorised capital ratio",
    "log_shared_address_company_count": (
        "Companies sharing the registered address"
    ),
    "log_address_registration_month_company_count": (
        "Companies registered at that address in the same month"
    ),
    "state_code": "Registered state",
    "roc_code": "Registrar of Companies",
    "company_status": "MCA company status",
    "company_category": "Company category",
    "company_subcategory": "Company subcategory",
    "company_class": "Company class",
    "listing_status": "Listing status",
    "company_origin": "Indian or foreign origin",
    "nic_division": "NIC industry division",
}
DISPLAY_VALUE_COLUMNS: dict[str, str] = {
    "log_authorised_capital_inr": "authorised_capital_inr",
    "log_paid_up_capital_inr": "paid_up_capital_inr",
    "log_shared_address_company_count": "shared_address_company_count",
    "log_address_registration_month_company_count": (
        "address_registration_month_company_count"
    ),
}
PERMITTED_USES = (
    "KYB enrichment during merchant onboarding.",
    "Analyst triage ordering of an enhanced-due-diligence queue.",
    "Evidence-backed context for a manual underwriting review.",
    "Risk-operations monitoring of an existing public-company portfolio.",
)
STANDING_LIMITATIONS = (
    (
        "The score estimates a future public IBBI CIRP announcement. It is "
        "not a fraud, payment-abuse, chargeback, or regulatory-debarment "
        "score."
    ),
    "No Razorpay merchant, payment, or settlement data was used or joined.",
    "Public insolvency risk is not evidence of merchant misconduct.",
    (
        "The model is EVALUATED_NOT_PRODUCTION_READY and must not drive an "
        "automated onboarding, hold, or termination action."
    ),
    (
        "Every output requires a human risk-operations reviewer before any "
        "merchant-facing decision."
    ),
)


class MerchantAssessmentError(RuntimeError):
    """A merchant assessment could not be produced from frozen evidence."""


class MerchantIdentity(BaseModel):
    """Public MCA identity of the company an analyst asked about."""

    model_config = ConfigDict(frozen=True)

    cin: str
    company_name: str
    company_status: str
    registration_date: date | None
    state_code: str
    roc_code: str
    company_class: str
    listing_status: str
    company_origin: str
    nic_code: str
    nic_division: str
    authorised_capital_inr: float | None
    paid_up_capital_inr: float | None


class ExplanationFactor(BaseModel):
    """One model feature and the direction it pushed this merchant's score."""

    model_config = ConfigDict(frozen=True)

    feature: str
    label: str
    value: str
    direction: str
    contribution: float
    evidence: str


class EvidenceSource(BaseModel):
    """One immutable public artifact standing behind the assessment."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    description: str
    reference: str
    as_of: date | None


class AssessmentConfidence(BaseModel):
    """Honest, deterministic confidence in this specific assessment."""

    model_config = ConfigDict(frozen=True)

    level: str
    identity_match_method: str
    feature_snapshot_date: date
    feature_snapshot_age_days: int = Field(ge=0)
    band_holdout_precision: float | None
    reasons: tuple[str, ...]


class MerchantRiskAssessment(BaseModel):
    """Evidence-backed public risk intelligence for one merchant identity."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = MERCHANT_ASSESSMENT_SCHEMA_VERSION
    generated_at: datetime
    score_name: str = MERCHANT_RISK_SCORE_NAME
    label_boundary: str
    model_status: str
    identity: MerchantIdentity
    identity_match_method: str
    cirp_public_announcement_score: float = Field(ge=0, le=1)
    risk_band: str
    risk_band_evidence: RiskBandThreshold
    explanation_factors: tuple[ExplanationFactor, ...]
    evidence_sources: tuple[EvidenceSource, ...]
    confidence: AssessmentConfidence
    permitted_use: tuple[str, ...] = PERMITTED_USES
    limitations: tuple[str, ...] = STANDING_LIMITATIONS


@dataclass(frozen=True)
class MerchantAssessmentOutcome:
    """A resolved lookup and, when identity is unambiguous, its assessment."""

    lookup: MerchantLookupResult
    assessment: MerchantRiskAssessment | None


def _display_value(row: dict[str, Any], feature_name: str) -> str:
    display_column = DISPLAY_VALUE_COLUMNS.get(feature_name, feature_name)
    value = row.get(display_column)
    if value is None:
        return "unavailable"
    if display_column in {"authorised_capital_inr", "paid_up_capital_inr"}:
        return f"INR {float(value):,.0f}"
    if feature_name == "company_age_years":
        return f"{float(value):.1f} years"
    if feature_name == "paid_to_authorised_capital_ratio":
        return f"{float(value):.2f}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _factor_evidence(
    label: str,
    display_value: str,
    direction: str,
    feature_name: str,
) -> str:
    movement = (
        "raises" if direction == "increases_risk" else "lowers"
    )
    if feature_name in SOLVENCY_NUMERIC_FEATURES:
        return (
            f"{label} of {display_value} {movement} the modelled CIRP "
            "announcement probability for this merchant."
        )
    return (
        f"{label} recorded as {display_value} {movement} the modelled CIRP "
        "announcement probability for this merchant."
    )


def build_explanation_factors(
    feature_row: pl.DataFrame,
    contributions: pl.DataFrame,
    *,
    maximum_factors: int = MAXIMUM_EXPLANATION_FACTORS,
) -> tuple[ExplanationFactor, ...]:
    """Rank the model's own log-odds contributions for one merchant row."""

    if feature_row.height != 1 or contributions.height != 1:
        raise MerchantAssessmentError(
            "Explanations require exactly one merchant feature row."
        )
    row = feature_row.row(0, named=True)
    contribution_row = contributions.row(0, named=True)
    ranked = sorted(
        (
            (feature_name, float(value))
            for feature_name, value in contribution_row.items()
            if feature_name != "model_bias"
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    factors: list[ExplanationFactor] = []
    for feature_name, contribution in ranked[:maximum_factors]:
        if contribution == 0.0:
            continue
        direction = (
            "increases_risk" if contribution > 0 else "decreases_risk"
        )
        label = FEATURE_LABELS.get(feature_name, feature_name)
        display_value = _display_value(row, feature_name)
        factors.append(
            ExplanationFactor(
                feature=feature_name,
                label=label,
                value=display_value,
                direction=direction,
                contribution=contribution,
                evidence=_factor_evidence(
                    label, display_value, direction, feature_name
                ),
            )
        )
    return tuple(factors)


def _identity(row: dict[str, Any]) -> MerchantIdentity:
    authorised_capital = row.get("authorised_capital_inr")
    paid_up_capital = row.get("paid_up_capital_inr")
    return MerchantIdentity(
        cin=str(row["cin"]),
        company_name=str(row["company_name"]),
        company_status=str(row["company_status"]),
        registration_date=row.get("registration_date"),
        state_code=str(row["state_code"]),
        roc_code=str(row["roc_code"]),
        company_class=str(row["company_class"]),
        listing_status=str(row["listing_status"]),
        company_origin=str(row["company_origin"]),
        nic_code=str(row.get("nic_code") or ""),
        nic_division=str(row["nic_division"]),
        authorised_capital_inr=(
            None if authorised_capital is None else float(authorised_capital)
        ),
        paid_up_capital_inr=(
            None if paid_up_capital is None else float(paid_up_capital)
        ),
    )


class MerchantRiskAssessor:
    """Resolve, score, band, and explain one public merchant identity."""

    def __init__(
        self,
        *,
        store: MerchantRiskIndexStore,
        package: SolvencyModelPackage,
        band_scale: RiskBandScale,
        evaluation: SolvencyHoldoutEvaluationReport,
    ) -> None:
        model_sha256 = package.report.model_sha256
        if evaluation.model_sha256 != model_sha256:
            raise MerchantAssessmentError(
                "Loaded model and holdout evaluation describe different models."
            )
        if evaluation.label_boundary != package.metrics.label_boundary:
            raise MerchantAssessmentError(
                "Loaded model and holdout evaluation claim different boundaries."
            )
        self._store = store
        self._package = package
        self._band_scale = band_scale
        self._evaluation = evaluation
        self._evidence_sources = self._build_evidence_sources(
            store.report, model_sha256
        )

    @property
    def index_report(self) -> MerchantRiskIndexReport:
        return self._store.report

    @property
    def band_scale(self) -> RiskBandScale:
        return self._band_scale

    def score_frame(self, feature_frame: pl.DataFrame) -> pl.Series:
        """Score an index-shaped frame without producing an assessment."""

        scores, _ = self._package.score_with_contributions(feature_frame)
        return scores

    def assess(
        self,
        *,
        cin: str | None = None,
        company_name: str | None = None,
        now: datetime | None = None,
    ) -> MerchantAssessmentOutcome:
        """Return a lookup outcome and, on a unique identity, its assessment."""

        lookup = self._store.lookup(cin=cin, company_name=company_name)
        if lookup.outcome != "matched" or lookup.matched_row is None:
            return MerchantAssessmentOutcome(lookup=lookup, assessment=None)
        feature_row = lookup.matched_row
        scores, contributions = self._package.score_with_contributions(
            feature_row
        )
        score = float(scores[0])
        band = self._band_scale.band_for(score)
        generated_at = now or datetime.now(tz=UTC)
        match_method = lookup.match_method or "exact_cin"
        assessment = MerchantRiskAssessment(
            generated_at=generated_at,
            label_boundary=self._package.metrics.label_boundary,
            model_status=self._evaluation.evaluation_decision,
            identity=_identity(feature_row.row(0, named=True)),
            identity_match_method=match_method,
            cirp_public_announcement_score=score,
            risk_band=band.band,
            risk_band_evidence=band,
            explanation_factors=build_explanation_factors(
                feature_row, contributions
            ),
            evidence_sources=self._evidence_sources,
            confidence=self._confidence(
                match_method=match_method,
                band=band,
                generated_at=generated_at,
            ),
            limitations=(
                *STANDING_LIMITATIONS,
                *self._evaluation.deployment_blockers,
            ),
        )
        return MerchantAssessmentOutcome(lookup=lookup, assessment=assessment)

    def _confidence(
        self,
        *,
        match_method: str,
        band: RiskBandThreshold,
        generated_at: datetime,
    ) -> AssessmentConfidence:
        feature_cutoff = self._store.report.feature_cutoff
        snapshot_age_days = max(
            0, (generated_at.date() - feature_cutoff).days
        )
        reasons: list[str] = []
        if match_method != "exact_cin":
            reasons.append(
                "Identity was resolved from a normalised company name, not an "
                "exact CIN. An analyst must confirm the entity."
            )
        if snapshot_age_days > MODERATE_CONFIDENCE_MAXIMUM_SNAPSHOT_AGE_DAYS:
            reasons.append(
                f"The MCA feature snapshot is {snapshot_age_days} days old, so "
                "later filings, status changes, or capital changes are absent."
            )
        if band.holdout_precision is None:
            reasons.append(
                "This merchant falls below every measured review-capacity "
                "queue, so no queue-level precision applies."
            )
        level = "moderate" if not reasons else "low"
        if not reasons:
            reasons.append(
                "Exact-CIN identity on a current public snapshot with measured "
                "queue-level precision."
            )
        return AssessmentConfidence(
            level=level,
            identity_match_method=match_method,
            feature_snapshot_date=feature_cutoff,
            feature_snapshot_age_days=snapshot_age_days,
            band_holdout_precision=band.holdout_precision,
            reasons=tuple(reasons),
        )

    @staticmethod
    def _build_evidence_sources(
        index_report: MerchantRiskIndexReport,
        model_sha256: str,
    ) -> tuple[EvidenceSource, ...]:
        return (
            EvidenceSource(
                source_id="mca_company_master",
                description=(
                    "MCA company master register published on data.gov.in, "
                    "canonicalised into an as-of company snapshot."
                ),
                reference=index_report.mca_source_snapshot_id,
                as_of=index_report.feature_cutoff,
            ),
            EvidenceSource(
                source_id="ibbi_cirp_public_announcements",
                description=(
                    "Official IBBI CIRP public-announcement export used only "
                    "as the outcome the model was trained and tested against."
                ),
                reference=index_report.ibbi_source_snapshot_id,
                as_of=None,
            ),
            EvidenceSource(
                source_id="merchant_risk_index",
                description=(
                    "Immutable serving projection of the frozen leakage-safe "
                    "solvency observation population."
                ),
                reference=index_report.run_id,
                as_of=index_report.feature_cutoff,
            ),
            EvidenceSource(
                source_id="solvency_model_package",
                description=(
                    "Checksum-verified XGBoost package scored with its "
                    "selected early-stopping iteration range."
                ),
                reference=model_sha256,
                as_of=None,
            ),
        )
