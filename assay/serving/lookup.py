"""Resolve one caller identity query against the frozen merchant risk index.

The index is a single CIN-sorted Parquet artifact covering every eligible
public company, so serving scans it lazily and pushes the CIN or normalised
name predicate down instead of holding millions of rows in memory. The store
answers identity and nothing else: it hands back the frozen feature row a
model consumer may score later, never a decision, and it keeps an ambiguous
name distinct from a match so a name collision stays a human judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

from assay.artifacts.parquet import (
    ImmutableArtifactError,
    verify_parquet_artifact,
)
from assay.contracts.identifiers import INDIAN_CIN_PATTERN
from assay.contracts.names import normalise_query_name
from assay.serving.index import (
    MERCHANT_INDEX_COLUMNS,
    MerchantRiskIndexReport,
)

MERCHANT_LOOKUP_OUTCOMES = (
    "matched",
    "no_match",
    "ambiguous_name",
    "invalid_identifier",
)
MERCHANT_MATCH_METHODS = (
    "exact_cin",
    "normalized_strict_name",
    "normalized_legal_name",
)
MAXIMUM_AMBIGUOUS_CANDIDATES = 10
MERCHANT_CANDIDATE_COLUMNS = (
    "cin",
    "company_name",
    "state_code",
    "company_status",
    "registration_date",
)

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MerchantLookupError(RuntimeError):
    """A merchant lookup input or index artifact failed its contract."""


class MerchantCandidate(BaseModel):
    """One public company that a caller query could refer to."""

    model_config = ConfigDict(frozen=True)

    cin: str
    company_name: str
    state_code: str
    company_status: str
    registration_date: date | None


@dataclass(frozen=True)
class MerchantLookupResult:
    """The outcome of one identity query, with its evidence and its limits."""

    outcome: str
    match_method: str | None
    matched_row: pl.DataFrame | None
    candidates: tuple[MerchantCandidate, ...]
    detail: str


def merchant_candidates(frame: pl.DataFrame) -> tuple[MerchantCandidate, ...]:
    """Project index rows to the fields an analyst needs to disambiguate."""

    return tuple(
        MerchantCandidate(
            cin=str(row["cin"]),
            company_name=str(row["company_name"]),
            state_code=str(row["state_code"]),
            company_status=str(row["company_status"]),
            registration_date=row["registration_date"],
        )
        for row in frame.select(
            list(MERCHANT_CANDIDATE_COLUMNS)
        ).iter_rows(named=True)
    )


class MerchantRiskIndexStore:
    """Serve identity lookups from one verified, immutable index artifact."""

    def __init__(
        self,
        report: MerchantRiskIndexReport,
        index_path: Path,
    ) -> None:
        self._report = report
        self._index_path = index_path

    @classmethod
    def from_report_path(
        cls,
        report_path: Path,
        *,
        project_root: Path | None = None,
        verify_checksum: bool = True,
    ) -> MerchantRiskIndexStore:
        """Bind a validated index report to its checksum-verified Parquet."""

        try:
            report = MerchantRiskIndexReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise MerchantLookupError(
                f"Merchant index report is missing or invalid: {report_path}."
            ) from error
        resolved_root = (
            project_root if project_root is not None else _DEFAULT_PROJECT_ROOT
        )
        if verify_checksum:
            try:
                index_path = verify_parquet_artifact(
                    report.index_artifact, resolved_root
                )
            except ImmutableArtifactError as error:
                raise MerchantLookupError(str(error)) from error
        else:
            index_path = Path(report.index_artifact.path)
            if not index_path.is_absolute():
                index_path = resolved_root / index_path
            if not index_path.is_file():
                raise MerchantLookupError(
                    f"Merchant index artifact is missing: {index_path}."
                )
        return cls(report, index_path)

    @property
    def report(self) -> MerchantRiskIndexReport:
        """Provenance for the index every lookup in this process reads."""

        return self._report

    def lookup(
        self,
        *,
        cin: str | None = None,
        company_name: str | None = None,
    ) -> MerchantLookupResult:
        """Resolve a query to at most one frozen public company record."""

        requested_cin = (cin or "").strip().upper()
        requested_name = (company_name or "").strip()
        if not requested_cin and not requested_name:
            raise MerchantLookupError("A CIN or a company name is required.")
        if requested_cin:
            return self._lookup_by_cin(requested_cin)
        return self._lookup_by_name(requested_name)

    def _lookup_by_cin(self, requested_cin: str) -> MerchantLookupResult:
        if re.fullmatch(INDIAN_CIN_PATTERN, requested_cin) is None:
            return MerchantLookupResult(
                outcome="invalid_identifier",
                match_method=None,
                matched_row=None,
                candidates=(),
                detail=(
                    f"'{requested_cin}' is not a well-formed MCA Corporate "
                    "Identity Number. A CIN is 21 characters: a listing "
                    "status letter, a five-digit industry code, a two-letter "
                    "state code, a four-digit registration year, a "
                    "three-letter ownership code, and a six-digit "
                    "registration number. No company-name fallback was "
                    "attempted for a malformed identifier."
                ),
            )
        matched_frame = self._matching_rows(pl.col("cin") == requested_cin)
        if matched_frame.is_empty():
            return MerchantLookupResult(
                outcome="no_match",
                match_method=None,
                matched_row=None,
                candidates=(),
                detail=(
                    f"CIN {requested_cin} is not in the frozen eligible "
                    "population as of the index feature cutoff "
                    f"{self._report.feature_cutoff.isoformat()}. A real CIN "
                    "can be absent because the company was registered after "
                    "that cutoff, because the identifier is an LLPIN or an "
                    "FCRN rather than a CIN, or because the company was "
                    "excluded when a public CIRP announcement predates the "
                    "cutoff."
                ),
            )
        return self._single_match(
            matched_frame,
            "exact_cin",
            (
                f"CIN {requested_cin} resolves to exactly one public company "
                "record in the frozen eligible population as of "
                f"{self._report.feature_cutoff.isoformat()}."
            ),
        )

    def _lookup_by_name(self, requested_name: str) -> MerchantLookupResult:
        strict_name, legal_name = normalise_query_name(requested_name)
        if not strict_name:
            return MerchantLookupResult(
                outcome="no_match",
                match_method=None,
                matched_row=None,
                candidates=(),
                detail=(
                    "The submitted company name contains no indexable "
                    "characters after normalisation, so no public company "
                    "record could be looked up."
                ),
            )
        strict_frame = self._matching_rows(
            pl.col("company_name_normalized_strict") == strict_name
        )
        if strict_frame.height == 1:
            return self._single_match(
                strict_frame,
                "normalized_strict_name",
                (
                    "One public company record carries the normalised name "
                    f"'{strict_name}'. This is a name match, not an "
                    "identifier match; the analyst must confirm identity "
                    "before relying on the record."
                ),
            )
        if strict_frame.height > 1:
            return self._ambiguous(strict_frame, strict_name)
        if legal_name:
            legal_frame = self._matching_rows(
                pl.col("company_name_normalized_legal") == legal_name
            )
            if legal_frame.height == 1:
                return self._single_match(
                    legal_frame,
                    "normalized_legal_name",
                    (
                        "One public company record carries the normalised "
                        f"name '{legal_name}' once Indian legal-form "
                        "suffixes are removed from both sides. This is a "
                        "name match, not an identifier match; the analyst "
                        "must confirm identity before relying on the record."
                    ),
                )
            if legal_frame.height > 1:
                return self._ambiguous(legal_frame, legal_name)
        tried_forms = f"'{strict_name}'"
        if legal_name and legal_name != strict_name:
            tried_forms = f"{tried_forms} and '{legal_name}'"
        return MerchantLookupResult(
            outcome="no_match",
            match_method=None,
            matched_row=None,
            candidates=(),
            detail=(
                "No public company record in the frozen eligible population "
                f"carries the normalised name {tried_forms} as of the index "
                f"feature cutoff {self._report.feature_cutoff.isoformat()}."
            ),
        )

    def _ambiguous(
        self,
        frame: pl.DataFrame,
        normalised_name: str,
    ) -> MerchantLookupResult:
        listed_frame = frame.head(MAXIMUM_AMBIGUOUS_CANDIDATES)
        return MerchantLookupResult(
            outcome="ambiguous_name",
            match_method=None,
            matched_row=None,
            candidates=merchant_candidates(listed_frame),
            detail=(
                f"{frame.height} public company records in the frozen "
                "eligible population share the normalised name "
                f"'{normalised_name}'. This response lists "
                f"{listed_frame.height} of them, sorted by CIN. Resubmit the "
                "request with a CIN; the analyst must confirm identity "
                "before relying on any of these records."
            ),
        )

    def _single_match(
        self,
        frame: pl.DataFrame,
        match_method: str,
        detail: str,
    ) -> MerchantLookupResult:
        if frame.height != 1:
            raise MerchantLookupError(
                f"Merchant index returned {frame.height} rows for a single "
                f"{match_method} match."
            )
        return MerchantLookupResult(
            outcome="matched",
            match_method=match_method,
            matched_row=frame,
            candidates=(),
            detail=detail,
        )

    def _matching_rows(self, predicate: pl.Expr) -> pl.DataFrame:
        try:
            return (
                pl.scan_parquet(self._index_path)
                .filter(predicate)
                .select(list(MERCHANT_INDEX_COLUMNS))
                .sort("cin")
                .collect()
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MerchantLookupError(
                f"Merchant index is unreadable: {self._index_path}."
            ) from error
