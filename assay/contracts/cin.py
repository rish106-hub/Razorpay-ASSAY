"""Structural decoding of MCA Corporate Identity Numbers.

A CIN is a composite key, not an opaque identifier. Its twenty-one characters
carry the listing status, the NIC industry code, the registrar-office code, the
year of incorporation, the ownership class, and the registrar's sequential
registration serial:

    U 72900 KA 2010 PTC 123456
    | |     |  |    |   |
    | |     |  |    |   +-- segment 6: ROC registration serial (6 digits)
    | |     |  |    +------ segment 5: ownership class (3 letters)
    | |     |  +----------- segment 4: year of incorporation (4 digits)
    | |     +-------------- segment 3: registrar / state code (2 letters)
    | +-------------------- segment 2: NIC industry code (5 digits)
    +---------------------- segment 1: listing status, ``L`` or ``U``

Two properties of that structure are exploitable and are the reason this module
exists.

Internal disagreement
    The MCA company record carries its own ``registration_date``, ``nic_code``,
    ``listing_status``, and ``state_code`` columns, populated independently of
    the identifier. Where the embedded segment and the record's own column
    disagree, the entity's registration history is not a straight line -
    re-registration, migration between registrars, or a filing correction.
    :func:`cin_disagreement_expressions` surfaces those four comparisons as
    explicit tri-state flags: ``True`` disagrees, ``False`` agrees, ``null``
    undecidable.

ROC serial adjacency
    Segment 6 is allocated sequentially by a registrar office. Two companies
    with near-adjacent serials inside the same registrar-year cohort were
    filed in the same batch, most often through the same filing agent. This
    module exposes the decoded serial and a stable
    ``cin_registrar_year_cohort_key`` so an adjacency feature can be built on
    top; it does not build that feature itself.

Both a pure-Python path (:func:`decode_cin`, for one identifier in an API
request or a test) and a Polars-expression path
(:func:`cin_decode_expressions`, for a full columnar pass over the snapshot)
are provided. The two derive every segment from the same frozen
:data:`CIN_SEGMENT_SPECS` offsets and the same
:data:`assay.contracts.identifiers.INDIAN_CIN_PATTERN` validation, so they
agree by construction.

Nothing here raises on a malformed identifier. The snapshot mixes CINs with
LLPINs, FCRNs, and a quarantined malformed tail; every one of those decodes to
nulls.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from types import MappingProxyType
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.contracts.identifiers import INDIAN_CIN_PATTERN

CIN_STRUCTURE_CONTRACT_VERSION = "cin_structure_v1"

CIN_LENGTH = 21

#: Sentinel returned for a registrar code that is absent from
#: :data:`CIN_STATE_LETTERS_TO_STATE_NAME`. Real government data contains codes
#: outside any published list, so an unmapped code is reported rather than
#: guessed at or raised on.
CIN_UNKNOWN_STATE_NAME = "UNKNOWN"


class CinSegmentSpec(BaseModel):
    """Fixed position of one CIN segment, shared by both decode paths."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    offset: int = Field(ge=0, lt=CIN_LENGTH)
    width: int = Field(gt=0, le=CIN_LENGTH)


CIN_SEGMENT_SPECS: tuple[CinSegmentSpec, ...] = (
    CinSegmentSpec(name="listing_letter", offset=0, width=1),
    CinSegmentSpec(name="nic_code", offset=1, width=5),
    CinSegmentSpec(name="state_letters", offset=6, width=2),
    CinSegmentSpec(name="incorporation_year", offset=8, width=4),
    CinSegmentSpec(name="ownership_class", offset=12, width=3),
    CinSegmentSpec(name="roc_serial", offset=15, width=6),
)

_SEGMENT_BY_NAME: Mapping[str, CinSegmentSpec] = MappingProxyType(
    {segment.name: segment for segment in CIN_SEGMENT_SPECS}
)

#: Segment 3 of a CIN to the state name spelling used by the canonical MCA
#: ``state_code`` column.
#:
#: Segment 3 is a registrar-office code, not strictly a state code: ``PN`` is
#: the Pune registrar, ``TZ`` the Coimbatore registrar, and ``MR`` and ``ME``
#: are Mumbai registrars. Each entry therefore names the state the registrar
#: sits in, in the record's own lowercase vocabulary, so the two are directly
#: comparable.
#:
#: Sourced by cross-tabulating segment 3 against the canonical ``state_code``
#: and ``roc_code`` columns over all 3,123,764 CIN rows of the MCA
#: company-master snapshot, then keeping the published meaning of each code
#: rather than the observed majority. ``CT`` is retained as Chhattisgarh even
#: though its single occurrence in the snapshot carries ``ROC Gwalior`` and a
#: ``madhya pradesh`` state, because that mismatch is exactly the kind of
#: post-reorganisation disagreement this contract is built to surface.
#:
#: Deliberately absent, and therefore resolving to
#: :data:`CIN_UNKNOWN_STATE_NAME`: ``DH`` (2 rows), ``LA`` (1 row), and ``RH``
#: (1 row). None has a defensible published meaning consistent with its
#: registrar, and guessing at four rows would corrupt the disagreement signal.
CIN_STATE_LETTERS_TO_STATE_NAME: Mapping[str, str] = MappingProxyType(
    {
        "AN": "andaman and nicobar islands",
        "AP": "andhra pradesh",
        "AR": "arunachal pradesh",
        "AS": "assam",
        "BR": "bihar",
        "CH": "chandigarh",
        "CT": "chattisgarh",
        "DD": "daman and diu",
        "DL": "delhi",
        "DN": "dadra & nagar haveli",
        "DC": "delhi",
        "GA": "goa",
        "GJ": "gujarat",
        "HP": "himachal pradesh",
        "HR": "haryana",
        "JH": "jharkhand",
        "JK": "jammu & kashmir",
        "KA": "karnataka",
        "KL": "kerala",
        "LD": "lakshadweep",
        "LH": "ladakh",
        "ME": "maharashtra",
        "MH": "maharashtra",
        "ML": "meghalaya",
        "MN": "manipur",
        "MP": "madhya pradesh",
        "MR": "maharashtra",
        "MZ": "mizoram",
        "NL": "nagaland",
        "OD": "orissa",
        "OR": "orissa",
        "PB": "punjab",
        "PN": "maharashtra",
        "PY": "pondicherry",
        "RJ": "rajasthan",
        "SK": "sikkim",
        "TG": "telangana",
        "TN": "tamil nadu",
        "TR": "tripura",
        "TS": "telangana",
        "TZ": "tamil nadu",
        "UP": "uttar pradesh",
        "UR": "uttarakhand",
        "UT": "uttarakhand",
        "UW": "uttar pradesh",
        "WB": "west bengal",
        "WR": "west bengal",
    }
)

#: Columns emitted by :func:`cin_decode_expressions`, in order.
CIN_DECODED_COLUMNS: tuple[str, ...] = (
    "cin_listing_letter",
    "cin_is_listed",
    "cin_nic_code",
    "cin_nic_division",
    "cin_state_letters",
    "cin_state_name",
    "cin_incorporation_year",
    "cin_ownership_class",
    "cin_roc_serial",
    "cin_registrar_year_cohort_key",
)

#: Columns emitted by :func:`cin_disagreement_expressions`, in order.
CIN_DISAGREEMENT_COLUMNS: tuple[str, ...] = (
    "cin_year_disagrees_with_record",
    "cin_nic_division_disagrees_with_record",
    "cin_listing_disagrees_with_record",
    "cin_state_disagrees_with_record",
    "cin_record_disagreement_count",
)

_CIN_EXPRESSION = re.compile(INDIAN_CIN_PATTERN)
_TWO_DIGIT_EXPRESSION = r"^[0-9]{2}$"
_LISTED_RECORD_VALUE = "listed"
_UNLISTED_RECORD_VALUE = "unlisted"
_COHORT_KEY_SEPARATOR = ":"


class DecodedCin(BaseModel):
    """Every segment of one structurally valid CIN, decoded."""

    model_config = ConfigDict(frozen=True)

    cin: str = Field(pattern=INDIAN_CIN_PATTERN)
    listing_letter: Literal["L", "U"]
    is_listed: bool
    nic_code: str = Field(pattern=r"^[0-9]{5}$")
    nic_division: str = Field(pattern=_TWO_DIGIT_EXPRESSION)
    state_letters: str = Field(pattern=r"^[A-Z]{2}$")
    state_name: str = Field(min_length=1)
    incorporation_year: int = Field(ge=0, le=9999)
    ownership_class: str = Field(pattern=r"^[A-Z]{3}$")
    roc_serial: int = Field(ge=0, le=999999)
    registrar_year_cohort_key: str = Field(min_length=1)


class CinRecordDisagreement(BaseModel):
    """Tri-state comparison of a CIN's segments against the record's columns.

    Each flag is ``True`` when the segment and the record's own column
    disagree, ``False`` when they agree, and ``None`` when the comparison
    cannot be made - the identifier is not a CIN, the record's column is
    blank, or the registrar code is unmapped.
    """

    model_config = ConfigDict(frozen=True)

    year_disagrees: bool | None
    nic_division_disagrees: bool | None
    listing_disagrees: bool | None
    state_disagrees: bool | None
    disagreement_count: int = Field(ge=0, le=4)


def normalise_cin(identifier: str | None) -> str | None:
    """Return the identifier as a structurally valid CIN, or ``None``.

    Trims surrounding whitespace and upper-cases before validating against
    :data:`assay.contracts.identifiers.INDIAN_CIN_PATTERN`. LLPINs, FCRNs, and
    malformed identifiers return ``None`` rather than raising.
    """

    if identifier is None:
        return None
    candidate = identifier.strip().upper()
    if _CIN_EXPRESSION.fullmatch(candidate) is None:
        return None
    return candidate


def _segment(valid_cin: str, name: str) -> str:
    spec = _SEGMENT_BY_NAME[name]
    return valid_cin[spec.offset : spec.offset + spec.width]


def cin_state_name(state_letters: str) -> str:
    """Map segment 3 to the record's state-name vocabulary.

    Returns :data:`CIN_UNKNOWN_STATE_NAME` for a registrar code outside
    :data:`CIN_STATE_LETTERS_TO_STATE_NAME`.
    """

    return CIN_STATE_LETTERS_TO_STATE_NAME.get(
        state_letters, CIN_UNKNOWN_STATE_NAME
    )


def cin_registrar_year_cohort_key(
    state_letters: str,
    incorporation_year_digits: str,
    ownership_class: str,
) -> str:
    """Build the cohort within which ROC serial adjacency is meaningful.

    A registrar allocates segment 6 sequentially, so adjacency only carries
    co-incorporation meaning inside one registrar, one year, and one ownership
    class. The year is kept as its four raw digits so the key stays stable for
    the malformed years the pattern admits.
    """

    return _COHORT_KEY_SEPARATOR.join(
        (state_letters, incorporation_year_digits, ownership_class)
    )


def decode_cin(identifier: str | None) -> DecodedCin | None:
    """Decode one identifier, or return ``None`` if it is not a valid CIN."""

    valid_cin = normalise_cin(identifier)
    if valid_cin is None:
        return None
    listing_letter = _segment(valid_cin, "listing_letter")
    nic_code = _segment(valid_cin, "nic_code")
    state_letters = _segment(valid_cin, "state_letters")
    incorporation_year_digits = _segment(valid_cin, "incorporation_year")
    ownership_class = _segment(valid_cin, "ownership_class")
    roc_serial_digits = _segment(valid_cin, "roc_serial")
    return DecodedCin(
        cin=valid_cin,
        listing_letter="L" if listing_letter == "L" else "U",
        is_listed=listing_letter == "L",
        nic_code=nic_code,
        nic_division=nic_code[:2],
        state_letters=state_letters,
        state_name=cin_state_name(state_letters),
        incorporation_year=int(incorporation_year_digits),
        ownership_class=ownership_class,
        roc_serial=int(roc_serial_digits),
        registrar_year_cohort_key=cin_registrar_year_cohort_key(
            state_letters,
            incorporation_year_digits,
            ownership_class,
        ),
    )


def _record_listing_is_listed(listing_status: str | None) -> bool | None:
    if listing_status is None:
        return None
    normalised = listing_status.strip().lower()
    if normalised == _LISTED_RECORD_VALUE:
        return True
    if normalised == _UNLISTED_RECORD_VALUE:
        return False
    return None


def _record_nic_division(nic_code: str | None) -> str | None:
    if nic_code is None:
        return None
    division = nic_code.strip()[:2]
    if re.fullmatch(_TWO_DIGIT_EXPRESSION, division) is None:
        return None
    return division


def _record_state_name(state_code: str | None) -> str | None:
    if state_code is None:
        return None
    normalised = state_code.strip().lower()
    return normalised or None


def assess_cin_record_disagreement(
    identifier: str | None,
    *,
    registration_date: date | None = None,
    nic_code: str | None = None,
    listing_status: str | None = None,
    state_code: str | None = None,
) -> CinRecordDisagreement:
    """Compare one CIN's segments against that record's own columns."""

    decoded = decode_cin(identifier)
    year_disagrees: bool | None = None
    nic_division_disagrees: bool | None = None
    listing_disagrees: bool | None = None
    state_disagrees: bool | None = None
    if decoded is not None:
        if registration_date is not None:
            year_disagrees = decoded.incorporation_year != registration_date.year
        record_division = _record_nic_division(nic_code)
        if record_division is not None:
            nic_division_disagrees = decoded.nic_division != record_division
        record_is_listed = _record_listing_is_listed(listing_status)
        if record_is_listed is not None:
            listing_disagrees = decoded.is_listed != record_is_listed
        record_state = _record_state_name(state_code)
        if (
            record_state is not None
            and decoded.state_name != CIN_UNKNOWN_STATE_NAME
        ):
            state_disagrees = decoded.state_name != record_state
    flags = (
        year_disagrees,
        nic_division_disagrees,
        listing_disagrees,
        state_disagrees,
    )
    return CinRecordDisagreement(
        year_disagrees=year_disagrees,
        nic_division_disagrees=nic_division_disagrees,
        listing_disagrees=listing_disagrees,
        state_disagrees=state_disagrees,
        disagreement_count=sum(1 for flag in flags if flag is True),
    )


def _valid_cin_expression(cin_column: str) -> pl.Expr:
    normalised = (
        pl.col(cin_column).cast(pl.String).str.strip_chars().str.to_uppercase()
    )
    return (
        pl.when(normalised.str.contains(INDIAN_CIN_PATTERN))
        .then(normalised)
        .otherwise(None)
    )


def _segment_expression(valid_cin: pl.Expr, name: str) -> pl.Expr:
    spec = _SEGMENT_BY_NAME[name]
    return valid_cin.str.slice(spec.offset, spec.width)


def cin_decode_expressions(cin_column: str = "cin") -> tuple[pl.Expr, ...]:
    """Polars expressions decoding every CIN segment for a columnar pass.

    Emits :data:`CIN_DECODED_COLUMNS`. Every expression is null for a row whose
    identifier is not a structurally valid CIN, so LLPINs, FCRNs, malformed
    identifiers, and nulls pass through without raising.
    """

    valid_cin = _valid_cin_expression(cin_column)
    listing_letter = _segment_expression(valid_cin, "listing_letter")
    nic_code = _segment_expression(valid_cin, "nic_code")
    state_letters = _segment_expression(valid_cin, "state_letters")
    incorporation_year_digits = _segment_expression(
        valid_cin, "incorporation_year"
    )
    ownership_class = _segment_expression(valid_cin, "ownership_class")
    roc_serial_digits = _segment_expression(valid_cin, "roc_serial")
    state_name = (
        pl.when(state_letters.is_not_null())
        .then(
            state_letters.replace_strict(
                dict(CIN_STATE_LETTERS_TO_STATE_NAME),
                default=CIN_UNKNOWN_STATE_NAME,
                return_dtype=pl.String,
            )
        )
        .otherwise(None)
    )
    return (
        listing_letter.alias("cin_listing_letter"),
        (listing_letter == "L").alias("cin_is_listed"),
        nic_code.alias("cin_nic_code"),
        nic_code.str.slice(0, 2).alias("cin_nic_division"),
        state_letters.alias("cin_state_letters"),
        state_name.alias("cin_state_name"),
        incorporation_year_digits.cast(pl.Int32, strict=False).alias(
            "cin_incorporation_year"
        ),
        ownership_class.alias("cin_ownership_class"),
        roc_serial_digits.cast(pl.Int32, strict=False).alias("cin_roc_serial"),
        pl.concat_str(
            [state_letters, incorporation_year_digits, ownership_class],
            separator=_COHORT_KEY_SEPARATOR,
        ).alias("cin_registrar_year_cohort_key"),
    )


def cin_disagreement_expressions(
    *,
    cin_column: str = "cin",
    registration_date_column: str = "registration_date",
    nic_code_column: str = "nic_code",
    listing_status_column: str = "listing_status",
    state_code_column: str = "state_code",
) -> tuple[pl.Expr, ...]:
    """Polars expressions flagging CIN-versus-record disagreement.

    Emits :data:`CIN_DISAGREEMENT_COLUMNS`. The four flags are tri-state:
    ``True`` disagrees, ``False`` agrees, null undecidable. The count column
    tallies only the ``True`` flags, so it is always a plain integer.

    ``registration_date_column`` must hold a temporal dtype; the other source
    columns are read as strings.
    """

    valid_cin = _valid_cin_expression(cin_column)
    cin_year = _segment_expression(valid_cin, "incorporation_year").cast(
        pl.Int32, strict=False
    )
    cin_nic_division = _segment_expression(valid_cin, "nic_code").str.slice(0, 2)
    cin_is_listed = _segment_expression(valid_cin, "listing_letter") == "L"
    cin_state = _segment_expression(valid_cin, "state_letters").replace_strict(
        dict(CIN_STATE_LETTERS_TO_STATE_NAME),
        default=CIN_UNKNOWN_STATE_NAME,
        return_dtype=pl.String,
    )

    record_year = pl.col(registration_date_column).dt.year().cast(pl.Int32)
    record_nic_division_raw = (
        pl.col(nic_code_column).cast(pl.String).str.strip_chars().str.slice(0, 2)
    )
    record_nic_division = (
        pl.when(record_nic_division_raw.str.contains(_TWO_DIGIT_EXPRESSION))
        .then(record_nic_division_raw)
        .otherwise(None)
    )
    record_listing = (
        pl.col(listing_status_column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
    )
    record_is_listed = (
        pl.when(record_listing == _LISTED_RECORD_VALUE)
        .then(pl.lit(True))
        .when(record_listing == _UNLISTED_RECORD_VALUE)
        .then(pl.lit(False))
        .otherwise(None)
    )
    record_state_raw = (
        pl.col(state_code_column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
    )
    record_state = (
        pl.when(record_state_raw != "").then(record_state_raw).otherwise(None)
    )

    year_disagrees = (
        pl.when(cin_year.is_not_null() & record_year.is_not_null())
        .then(cin_year != record_year)
        .otherwise(None)
        .alias("cin_year_disagrees_with_record")
    )
    nic_disagrees = (
        pl.when(cin_nic_division.is_not_null() & record_nic_division.is_not_null())
        .then(cin_nic_division != record_nic_division)
        .otherwise(None)
        .alias("cin_nic_division_disagrees_with_record")
    )
    listing_disagrees = (
        pl.when(cin_is_listed.is_not_null() & record_is_listed.is_not_null())
        .then(cin_is_listed != record_is_listed)
        .otherwise(None)
        .alias("cin_listing_disagrees_with_record")
    )
    state_disagrees = (
        pl.when(
            cin_state.is_not_null()
            & (cin_state != CIN_UNKNOWN_STATE_NAME)
            & record_state.is_not_null()
        )
        .then(cin_state != record_state)
        .otherwise(None)
        .alias("cin_state_disagrees_with_record")
    )
    disagreement_count = (
        pl.sum_horizontal(
            year_disagrees.fill_null(False).cast(pl.Int32),
            nic_disagrees.fill_null(False).cast(pl.Int32),
            listing_disagrees.fill_null(False).cast(pl.Int32),
            state_disagrees.fill_null(False).cast(pl.Int32),
        )
        .cast(pl.Int32)
        .alias("cin_record_disagreement_count")
    )
    return (
        year_disagrees,
        nic_disagrees,
        listing_disagrees,
        state_disagrees,
        disagreement_count,
    )
