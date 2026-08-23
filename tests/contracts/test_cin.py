from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from assay.contracts.cin import (
    CIN_DECODED_COLUMNS,
    CIN_DISAGREEMENT_COLUMNS,
    CIN_LENGTH,
    CIN_SEGMENT_SPECS,
    CIN_STATE_LETTERS_TO_STATE_NAME,
    CIN_UNKNOWN_STATE_NAME,
    assess_cin_record_disagreement,
    cin_decode_expressions,
    cin_disagreement_expressions,
    decode_cin,
    normalise_cin,
)
from assay.contracts.identifiers import INDIAN_CIN_PATTERN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_GLOB = "data/curated/company_snapshot/schema-1.2.0/*/*.parquet"

# Rows lifted verbatim from the MCA company-master snapshot
# (data/curated/company_snapshot/schema-1.2.0). The snapshot itself is not
# committed, so these real records are embedded to keep the contract tested
# against government data rather than against synthetic identifiers only.
REAL_SNAPSHOT_ROWS: tuple[dict[str, object], ...] = (
    {
        # Real listed public company.
        "legal_entity_identifier": "L17110MH1973PLC019786",
        "company_name": "RELIANCE INDUSTRIES LIMITED",
        "registration_date": date(1973, 5, 8),
        "nic_code": "17110",
        "listing_status": "Listed",
        "state_code": "maharashtra",
    },
    {
        # Real unlisted private company.
        "legal_entity_identifier": "U01100KA2010PTC052043",
        "company_name": "AMARG0L AGRO AND INFRA PROJECTS PRIVATE LIMITED",
        "registration_date": date(2010, 1, 4),
        "nic_code": "01100",
        "listing_status": "Unlisted",
        "state_code": "karnataka",
    },
    {
        # Real unlisted private company in the same registrar-year cohort.
        "legal_entity_identifier": "U01100KA2010PTC053160",
        "company_name": "AGRICRATE PRIVATE LIMITED",
        "registration_date": date(2010, 4, 9),
        "nic_code": "01100",
        "listing_status": "Unlisted",
        "state_code": "karnataka",
    },
    {
        # Real LLPIN: must decode to nulls, never raise.
        "legal_entity_identifier": "AAA-0001",
        "company_name": "HANDOO & HANDOO LEGAL CONSULTANTS LLP",
        "registration_date": date(2009, 4, 2),
        "nic_code": "",
        "listing_status": "",
        "state_code": "delhi",
    },
    {
        # Real FCRN: must decode to nulls.
        "legal_entity_identifier": "F00003",
        "company_name": "CENTRAL PROVINCES MANGANESE ORE CO. LTD. ,",
        "registration_date": date(1903, 3, 31),
        "nic_code": "",
        "listing_status": "",
        "state_code": "delhi",
    },
    {
        # Real quarantined malformed identifier: CIN-shaped but not a CIN.
        "legal_entity_identifier": "U-6021PB1996PTC017865",
        "company_name": "GFC GOODS TRANSPORT PRIVATE LIMITED",
        "registration_date": date(1996, 3, 15),
        "nic_code": "-6021",
        "listing_status": "Unlisted",
        "state_code": "punjab",
    },
    {
        # Real registrar-code disagreement: ROC Delhi, Haryana-registered office.
        "legal_entity_identifier": "U01100DL2023PTC409662",
        "company_name": "ABHISHOOK AGRO PRIVATE LIMITED",
        "registration_date": date(2023, 1, 9),
        "nic_code": "01100",
        "listing_status": "Unlisted",
        "state_code": "haryana",
    },
    {
        # Real year disagreement: CIN says 1997, the record says 1987.
        "legal_entity_identifier": "L15200MH1997PLC107525",
        "company_name": "VADILAL DAIRY INTERNATIONAL LIMITED",
        "registration_date": date(1987, 7, 20),
        "nic_code": "15200",
        "listing_status": "Listed",
        "state_code": "maharashtra",
    },
    {
        # Real listing disagreement: listed letter, unlisted record.
        "legal_entity_identifier": "L01110MH1992PTC069330",
        "company_name": "INDO-FRENCH BIO-TECH ENTERPRISES LTD.",
        "registration_date": date(1992, 11, 3),
        "nic_code": "01110",
        "listing_status": "Unlisted",
        "state_code": "maharashtra",
    },
    {
        # Real unmapped registrar code: 'RH' is not a published code.
        "legal_entity_identifier": "U63010RH2012PLC040995",
        "company_name": "TREKKINGTOES.COM LIMITED",
        "registration_date": date(2012, 12, 13),
        "nic_code": "63010",
        "listing_status": "Unlisted",
        "state_code": "rajasthan",
    },
    {
        # Real post-reorganisation state disagreement: Chhattisgarh code,
        # Madhya Pradesh record, ROC Gwalior.
        "legal_entity_identifier": "U27106CT1989PTC005548",
        "company_name": "INDIAN ISPAT WORKS PRIVATE LIMITED",
        "registration_date": date(1989, 10, 18),
        "nic_code": "27106",
        "listing_status": "Unlisted",
        "state_code": "madhya pradesh",
    },
)

# Every distinct state_code value in the snapshot, so the map's values cannot
# drift out of the record's own vocabulary.
SNAPSHOT_STATE_CODE_VOCABULARY = frozenset(
    {
        "andaman and nicobar islands",
        "andhra pradesh",
        "arunachal pradesh",
        "assam",
        "bihar",
        "chandigarh",
        "chattisgarh",
        "dadra & nagar haveli",
        "daman and diu",
        "delhi",
        "goa",
        "gujarat",
        "haryana",
        "himachal pradesh",
        "jammu & kashmir",
        "jharkhand",
        "karnataka",
        "kerala",
        "ladakh",
        "lakshadweep",
        "madhya pradesh",
        "maharashtra",
        "manipur",
        "meghalaya",
        "mizoram",
        "nagaland",
        "orissa",
        "pondicherry",
        "punjab",
        "rajasthan",
        "sikkim",
        "tamil nadu",
        "telangana",
        "tripura",
        "uttar pradesh",
        "uttarakhand",
        "west bengal",
    }
)

# Every distinct segment-3 registrar code observed across the 3,123,706 CIN
# rows of the snapshot, so the map cannot silently lose coverage.
SNAPSHOT_REGISTRAR_CODES = frozenset(
    {
        "AN", "AP", "AR", "AS", "BR", "CH", "CT", "DC", "DD", "DH",
        "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL",
        "LA", "LD", "LH", "ME", "MH", "ML", "MN", "MP", "MR", "MZ",
        "NL", "OD", "OR", "PB", "PN", "PY", "RH", "RJ", "SK", "TG",
        "TN", "TR", "TS", "TZ", "UP", "UR", "UT", "UW", "WB", "WR",
    }
)

# Registrar codes observed in the snapshot that are deliberately unmapped:
# none has a published meaning consistent with its registrar, and between them
# they account for four rows.
UNMAPPED_SNAPSHOT_REGISTRAR_CODES = frozenset({"DH", "LA", "RH"})

SYNTHETIC_IDENTIFIERS: tuple[str | None, ...] = (
    None,
    "",
    "   ",
    "u01100ka2010ptc052043",
    "  L17110MH1973PLC019786  ",
    "U01100KA2010PTC05204",
    "U01100KA2010PTC0520431",
    "X01100KA2010PTC052043",
    "U0110AKA2010PTC052043",
    "U011001A2010PTC052043",
    "U01100KA20A0PTC052043",
    "U01100KA2010PT1052043",
    "U01100KA2010PTC05204A",
    "U99999ZZ9999ZZZ000000",
    "AAB-1234",
    "F12345",
)


def _record_frame() -> pl.DataFrame:
    return pl.DataFrame(
        list(REAL_SNAPSHOT_ROWS),
        schema={
            "legal_entity_identifier": pl.String,
            "company_name": pl.String,
            "registration_date": pl.Date,
            "nic_code": pl.String,
            "listing_status": pl.String,
            "state_code": pl.String,
        },
    )


def _decoded_frame() -> pl.DataFrame:
    return _record_frame().with_columns(
        *cin_decode_expressions("legal_entity_identifier"),
        *cin_disagreement_expressions(cin_column="legal_entity_identifier"),
    )


def _row_by_identifier(frame: pl.DataFrame, identifier: str) -> dict[str, object]:
    matched = frame.filter(pl.col("legal_entity_identifier") == identifier)
    assert matched.height == 1
    return matched.to_dicts()[0]


def test_segment_specs_tile_the_whole_identifier() -> None:
    assert sum(segment.width for segment in CIN_SEGMENT_SPECS) == CIN_LENGTH
    expected_offset = 0
    for segment in CIN_SEGMENT_SPECS:
        assert segment.offset == expected_offset
        expected_offset += segment.width
    assert expected_offset == CIN_LENGTH
    assert len({segment.name for segment in CIN_SEGMENT_SPECS}) == len(
        CIN_SEGMENT_SPECS
    )


def test_decoded_and_disagreement_column_names_are_unique() -> None:
    all_columns = CIN_DECODED_COLUMNS + CIN_DISAGREEMENT_COLUMNS
    assert len(set(all_columns)) == len(all_columns)
    assert all(column.startswith("cin_") for column in all_columns)


def test_state_map_is_frozen_and_uses_the_record_vocabulary() -> None:
    assert len(CIN_STATE_LETTERS_TO_STATE_NAME) == 47
    assert set(CIN_STATE_LETTERS_TO_STATE_NAME.values()) <= (
        SNAPSHOT_STATE_CODE_VOCABULARY
    )
    for registrar_code in CIN_STATE_LETTERS_TO_STATE_NAME:
        assert re.fullmatch(r"[A-Z]{2}", registrar_code) is not None
    with pytest.raises(TypeError):
        CIN_STATE_LETTERS_TO_STATE_NAME["ZZ"] = "nowhere"  # type: ignore[index]


def test_state_map_covers_every_registrar_code_in_the_snapshot() -> None:
    mapped_codes = set(CIN_STATE_LETTERS_TO_STATE_NAME)
    assert SNAPSHOT_REGISTRAR_CODES - mapped_codes == (
        UNMAPPED_SNAPSHOT_REGISTRAR_CODES
    )
    assert mapped_codes <= SNAPSHOT_REGISTRAR_CODES


def test_deliberately_unmapped_registrar_codes_resolve_to_unknown() -> None:
    for registrar_code in UNMAPPED_SNAPSHOT_REGISTRAR_CODES:
        assert registrar_code not in CIN_STATE_LETTERS_TO_STATE_NAME
    decoded = decode_cin("U63010RH2012PLC040995")
    assert decoded is not None
    assert decoded.state_letters == "RH"
    assert decoded.state_name == CIN_UNKNOWN_STATE_NAME


def test_decode_real_listed_company() -> None:
    decoded = decode_cin("L17110MH1973PLC019786")

    assert decoded is not None
    assert decoded.cin == "L17110MH1973PLC019786"
    assert decoded.listing_letter == "L"
    assert decoded.is_listed is True
    assert decoded.nic_code == "17110"
    assert decoded.nic_division == "17"
    assert decoded.state_letters == "MH"
    assert decoded.state_name == "maharashtra"
    assert decoded.incorporation_year == 1973
    assert decoded.ownership_class == "PLC"
    assert decoded.roc_serial == 19786
    assert decoded.registrar_year_cohort_key == "MH:1973:PLC"


def test_decode_real_unlisted_private_company() -> None:
    decoded = decode_cin("U01100KA2010PTC052043")

    assert decoded is not None
    assert decoded.listing_letter == "U"
    assert decoded.is_listed is False
    assert decoded.nic_code == "01100"
    assert decoded.nic_division == "01"
    assert decoded.state_letters == "KA"
    assert decoded.state_name == "karnataka"
    assert decoded.incorporation_year == 2010
    assert decoded.ownership_class == "PTC"
    assert decoded.roc_serial == 52043
    assert decoded.registrar_year_cohort_key == "KA:2010:PTC"


def test_decode_normalises_case_and_surrounding_whitespace() -> None:
    assert normalise_cin("  u01100ka2010ptc052043 ") == "U01100KA2010PTC052043"
    assert decode_cin("  u01100ka2010ptc052043 ") == decode_cin(
        "U01100KA2010PTC052043"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        None,
        "",
        "AAA-0001",
        "F00003",
        "F123456",
        "U-6021PB1996PTC017865",
        "U01100KA2010PTC05204",
        "U01100KA2010PTC0520431",
        "X01100KA2010PTC052043",
    ],
)
def test_decode_returns_none_for_non_cin_identifiers(
    identifier: str | None,
) -> None:
    assert normalise_cin(identifier) is None
    assert decode_cin(identifier) is None


def test_decoded_cin_always_matches_the_authoritative_pattern() -> None:
    for row in REAL_SNAPSHOT_ROWS:
        identifier = row["legal_entity_identifier"]
        assert isinstance(identifier, str)
        decoded = decode_cin(identifier)
        if decoded is None:
            continue
        assert re.fullmatch(INDIAN_CIN_PATTERN, decoded.cin) is not None


def test_columnar_path_nulls_every_non_cin_row_without_raising() -> None:
    frame = _decoded_frame()

    for identifier in ("AAA-0001", "F00003", "U-6021PB1996PTC017865"):
        row = _row_by_identifier(frame, identifier)
        for column in CIN_DECODED_COLUMNS:
            assert row[column] is None, column
        for column in CIN_DISAGREEMENT_COLUMNS[:-1]:
            assert row[column] is None, column
        assert row["cin_record_disagreement_count"] == 0


def test_columnar_path_survives_every_malformed_shape() -> None:
    frame = pl.DataFrame(
        {"cin": list(SYNTHETIC_IDENTIFIERS)},
        schema={"cin": pl.String},
    ).with_columns(*cin_decode_expressions())

    decoded_count = int(frame["cin_roc_serial"].is_not_null().sum())
    assert decoded_count == 3
    assert frame["cin_state_name"].to_list().count(CIN_UNKNOWN_STATE_NAME) == 1


def test_pure_and_columnar_paths_agree_on_real_and_malformed_input() -> None:
    identifiers: list[str | None] = [
        str(row["legal_entity_identifier"]) for row in REAL_SNAPSHOT_ROWS
    ]
    identifiers.extend(SYNTHETIC_IDENTIFIERS)
    frame = pl.DataFrame(
        {"cin": identifiers}, schema={"cin": pl.String}
    ).with_columns(*cin_decode_expressions())

    for row in frame.to_dicts():
        decoded = decode_cin(row["cin"])
        if decoded is None:
            assert all(row[column] is None for column in CIN_DECODED_COLUMNS)
            continue
        assert row["cin_listing_letter"] == decoded.listing_letter
        assert row["cin_is_listed"] == decoded.is_listed
        assert row["cin_nic_code"] == decoded.nic_code
        assert row["cin_nic_division"] == decoded.nic_division
        assert row["cin_state_letters"] == decoded.state_letters
        assert row["cin_state_name"] == decoded.state_name
        assert row["cin_incorporation_year"] == decoded.incorporation_year
        assert row["cin_ownership_class"] == decoded.ownership_class
        assert row["cin_roc_serial"] == decoded.roc_serial
        assert (
            row["cin_registrar_year_cohort_key"]
            == decoded.registrar_year_cohort_key
        )


def test_pure_and_columnar_disagreement_flags_agree() -> None:
    frame = _decoded_frame()

    for row in frame.to_dicts():
        assessed = assess_cin_record_disagreement(
            row["legal_entity_identifier"],
            registration_date=row["registration_date"],
            nic_code=row["nic_code"],
            listing_status=row["listing_status"],
            state_code=row["state_code"],
        )
        assert row["cin_year_disagrees_with_record"] == assessed.year_disagrees
        assert (
            row["cin_nic_division_disagrees_with_record"]
            == assessed.nic_division_disagrees
        )
        assert (
            row["cin_listing_disagrees_with_record"] == assessed.listing_disagrees
        )
        assert row["cin_state_disagrees_with_record"] == assessed.state_disagrees
        assert (
            row["cin_record_disagreement_count"] == assessed.disagreement_count
        )


def test_agreeing_real_record_raises_no_flag() -> None:
    row = _row_by_identifier(_decoded_frame(), "L17110MH1973PLC019786")

    assert row["cin_year_disagrees_with_record"] is False
    assert row["cin_nic_division_disagrees_with_record"] is False
    assert row["cin_listing_disagrees_with_record"] is False
    assert row["cin_state_disagrees_with_record"] is False
    assert row["cin_record_disagreement_count"] == 0


def test_real_year_disagreement_is_flagged() -> None:
    row = _row_by_identifier(_decoded_frame(), "L15200MH1997PLC107525")

    assert row["cin_incorporation_year"] == 1997
    assert row["cin_year_disagrees_with_record"] is True
    assert row["cin_state_disagrees_with_record"] is False
    assert row["cin_record_disagreement_count"] == 1


def test_real_listing_disagreement_is_flagged() -> None:
    row = _row_by_identifier(_decoded_frame(), "L01110MH1992PTC069330")

    assert row["cin_is_listed"] is True
    assert row["cin_listing_disagrees_with_record"] is True
    assert row["cin_record_disagreement_count"] == 1


def test_real_registrar_state_disagreement_is_flagged() -> None:
    row = _row_by_identifier(_decoded_frame(), "U01100DL2023PTC409662")

    assert row["cin_state_letters"] == "DL"
    assert row["cin_state_name"] == "delhi"
    assert row["cin_state_disagrees_with_record"] is True
    assert row["cin_record_disagreement_count"] == 1


def test_real_post_reorganisation_state_disagreement_is_flagged() -> None:
    row = _row_by_identifier(_decoded_frame(), "U27106CT1989PTC005548")

    assert row["cin_state_name"] == "chattisgarh"
    assert row["cin_state_disagrees_with_record"] is True


def test_unmapped_registrar_code_cannot_disagree() -> None:
    row = _row_by_identifier(_decoded_frame(), "U63010RH2012PLC040995")

    assert row["cin_state_name"] == CIN_UNKNOWN_STATE_NAME
    assert row["cin_state_disagrees_with_record"] is None
    assert row["cin_record_disagreement_count"] == 0


def test_blank_record_columns_make_flags_undecidable_not_false() -> None:
    frame = pl.DataFrame(
        {
            "cin": ["U01100KA2010PTC052043"],
            "registration_date": [None],
            "nic_code": [""],
            "listing_status": [""],
            "state_code": ["  "],
        },
        schema={
            "cin": pl.String,
            "registration_date": pl.Date,
            "nic_code": pl.String,
            "listing_status": pl.String,
            "state_code": pl.String,
        },
    ).with_columns(*cin_disagreement_expressions())
    row = frame.to_dicts()[0]

    for column in CIN_DISAGREEMENT_COLUMNS[:-1]:
        assert row[column] is None, column
    assert row["cin_record_disagreement_count"] == 0

    assessed = assess_cin_record_disagreement(
        "U01100KA2010PTC052043",
        registration_date=None,
        nic_code="",
        listing_status="",
        state_code="  ",
    )
    assert assessed.year_disagrees is None
    assert assessed.nic_division_disagrees is None
    assert assessed.listing_disagrees is None
    assert assessed.state_disagrees is None
    assert assessed.disagreement_count == 0


def test_unrecognised_listing_status_is_undecidable() -> None:
    assessed = assess_cin_record_disagreement(
        "U01100KA2010PTC052043",
        listing_status="Not Available",
    )
    assert assessed.listing_disagrees is None


def test_cohort_key_groups_adjacent_roc_serials() -> None:
    frame = _decoded_frame().filter(
        pl.col("cin_registrar_year_cohort_key") == "KA:2010:PTC"
    )

    assert frame.height == 2
    serials = sorted(int(serial) for serial in frame["cin_roc_serial"].to_list())
    assert serials == [52043, 53160]

    other = decode_cin("U01100KA2011PTC052044")
    assert other is not None
    assert other.registrar_year_cohort_key == "KA:2011:PTC"


def test_cohort_key_separates_registrar_ownership_and_year() -> None:
    base = decode_cin("U01100KA2010PTC052043")
    assert base is not None
    for variant in (
        "U01100TN2010PTC052043",
        "U01100KA2011PTC052043",
        "U01100KA2010PLC052043",
    ):
        decoded = decode_cin(variant)
        assert decoded is not None
        assert decoded.registrar_year_cohort_key != base.registrar_year_cohort_key


def test_decode_is_deterministic_across_repeated_calls() -> None:
    identifiers = [
        str(row["legal_entity_identifier"]) for row in REAL_SNAPSHOT_ROWS
    ]
    first = [decode_cin(identifier) for identifier in identifiers]
    second = [decode_cin(identifier) for identifier in identifiers]
    assert first == second

    frame = pl.DataFrame({"cin": identifiers}).with_columns(
        *cin_decode_expressions()
    )
    assert frame.equals(
        pl.DataFrame({"cin": identifiers}).with_columns(*cin_decode_expressions())
    )


def test_out_of_range_year_digits_are_decoded_not_rejected() -> None:
    decoded = decode_cin("U99999MH9999PTC000001")

    assert decoded is not None
    assert decoded.incorporation_year == 9999
    assert decoded.registrar_year_cohort_key == "MH:9999:PTC"


@pytest.mark.skipif(
    not list(PROJECT_ROOT.glob(SNAPSHOT_GLOB)),
    reason="MCA company snapshot is not present in this working tree",
)
def test_columnar_path_over_a_real_snapshot_part() -> None:
    snapshot_part = min(PROJECT_ROOT.glob(SNAPSHOT_GLOB))
    frame = (
        pl.read_parquet(snapshot_part)
        .select(
            "legal_entity_identifier",
            "legal_entity_identifier_type",
            "registration_date",
            "nic_code",
            "listing_status",
            "state_code",
        )
        .with_columns(
            *cin_decode_expressions("legal_entity_identifier"),
            *cin_disagreement_expressions(cin_column="legal_entity_identifier"),
        )
    )

    assert frame.height > 0
    decoded_mask = frame["cin_roc_serial"].is_not_null()
    is_cin = frame["legal_entity_identifier_type"] == "CIN"
    assert decoded_mask.to_list() == is_cin.to_list()
    assert frame["cin_record_disagreement_count"].null_count() == 0

    for row in frame.head(500).to_dicts():
        decoded = decode_cin(row["legal_entity_identifier"])
        if decoded is None:
            assert row["cin_roc_serial"] is None
            continue
        assert row["cin_roc_serial"] == decoded.roc_serial
        assert row["cin_state_name"] == decoded.state_name
        assert (
            row["cin_registrar_year_cohort_key"]
            == decoded.registrar_year_cohort_key
        )
