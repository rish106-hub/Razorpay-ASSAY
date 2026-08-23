from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from assay.solvency.observations import (
    SolvencyObservationConfig,
    build_solvency_observations,
)

SHAPE_COLUMNS = (
    "address_cluster_registration_span_days",
    "address_cluster_registration_month_entropy",
    "address_cluster_max_month_share",
    "address_cluster_nic_division_distinct",
    "address_cluster_authorised_capital_cv",
    "address_cluster_distinct_name_head_ratio",
)
FEATURE_CUTOFF = date(2023, 11, 3)
SOURCE_SNAPSHOT_ID = "a" * 64


def _cin(sequence_number: int) -> str:
    return f"U{sequence_number:05d}DL2020PTC{sequence_number:06d}"


def _observations(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Build observations from compact row specs, keyed by company snapshot id."""

    row_count = len(rows)
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": [row["id"] for row in rows],
            "source_snapshot_id": [SOURCE_SNAPSHOT_ID] * row_count,
            "snapshot_as_of": [FEATURE_CUTOFF] * row_count,
            "legal_entity_identifier": [row["cin"] for row in rows],
            "legal_entity_identifier_type": ["CIN"] * row_count,
            "legal_entity_identifier_is_valid_format": [True] * row_count,
            "cin": [row["cin"] for row in rows],
            "company_name": [row.get("name") for row in rows],
            "company_status": ["Active"] * row_count,
            "registration_date": [row.get("registered_on") for row in rows],
            "state_code": ["delhi"] * row_count,
            "roc_code": ["roc delhi"] * row_count,
            "company_category": ["company limited by shares"] * row_count,
            "company_subcategory": ["non-government company"] * row_count,
            "company_class": ["private"] * row_count,
            "listing_status": ["unlisted"] * row_count,
            "company_origin": ["indian"] * row_count,
            "authorised_capital_inr": [row.get("capital") for row in rows],
            "paid_up_capital_inr": [row.get("capital") for row in rows],
            "nic_code": [row.get("nic_code") for row in rows],
        },
        schema_overrides={
            "company_name": pl.String,
            "registration_date": pl.Date,
            "authorised_capital_inr": pl.Int64,
            "paid_up_capital_inr": pl.Int64,
            "nic_code": pl.String,
        },
    )
    address_frame = pl.DataFrame(
        {
            "legal_entity_identifier": [row["cin"] for row in rows],
            "source_snapshot_id": [SOURCE_SNAPSHOT_ID] * row_count,
            "snapshot_as_of": [FEATURE_CUTOFF] * row_count,
            "address_group_key": [row.get("address") for row in rows],
        },
        schema_overrides={"address_group_key": pl.String},
    )
    cirp_frame = pl.DataFrame(
        {
            "solvency_event_id": ["unrelated-event"],
            "cin": [_cin(99_998)],
            "event_date": [date(2025, 4, 1)],
        }
    )
    observations = build_solvency_observations(
        company_frame,
        address_frame,
        cirp_frame,
        SolvencyObservationConfig(
            run_id="b" * 64,
            feature_cutoff=FEATURE_CUTOFF,
            outcome_window_end=date(2026, 8, 21),
        ),
    )
    return {row["company_snapshot_id"]: row for row in observations.to_dicts()}


def _shell_factory_rows() -> list[dict[str, object]]:
    """Fifty companies incorporated at one address inside three weeks."""

    return [
        {
            "id": f"shell-{index}",
            "cin": _cin(1_000 + index),
            "address": "1 SHELL LANE",
            "registered_on": date(2020, 1, 2) + timedelta(days=index % 20),
            "capital": 100_000,
            "nic_code": "64990",
            "name": f"Abc Traders {index} Private Limited",
        }
        for index in range(50)
    ]


def _service_provider_rows() -> list[dict[str, object]]:
    """Fifty companies at a registered-office provider across twelve years."""

    industries = ["62010", "10101", "45100", "85100", "41000"]
    return [
        {
            "id": f"service-{index}",
            "cin": _cin(2_000 + index),
            "address": "2 SERVICE ROAD",
            "registered_on": date(2011, 1, 1) + timedelta(days=index * 88),
            "capital": 100_000 * (index + 1),
            "nic_code": industries[index % len(industries)],
            "name": f"Distinct{index} Ventures Private Limited",
        }
        for index in range(50)
    ]


def test_solvency_observations_exclude_prior_cirp_and_freeze_features() -> None:
    snapshot_id = "a" * 64
    company_frame = pl.DataFrame(
        {
            "company_snapshot_id": ["company-1", "company-2", "company-3"],
            "source_snapshot_id": [snapshot_id] * 3,
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
            "legal_entity_identifier": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
                "U99999MH2021PTC999999",
            ],
            "legal_entity_identifier_type": ["CIN"] * 3,
            "legal_entity_identifier_is_valid_format": [True] * 3,
            "cin": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
                "U99999MH2021PTC999999",
            ],
            "company_name": ["A Limited", "B Limited", "C Limited"],
            "company_status": ["Active"] * 3,
            "registration_date": [
                date(2020, 1, 1),
                date(2020, 1, 12),
                date(2021, 5, 1),
            ],
            "state_code": ["delhi", "delhi", "maharashtra"],
            "roc_code": ["roc delhi", "roc delhi", "roc mumbai"],
            "company_category": ["company limited by shares"] * 3,
            "company_subcategory": ["non-government company"] * 3,
            "company_class": ["private", "private", "public"],
            "listing_status": ["unlisted", "unlisted", "listed"],
            "company_origin": ["indian"] * 3,
            "authorised_capital_inr": [200_000, 200_000, 100_000],
            "paid_up_capital_inr": [100_000, 100_000, 50_000],
            "nic_code": ["64990", "64990", "62010"],
        }
    )
    address_frame = pl.DataFrame(
        {
            "legal_entity_identifier": company_frame["legal_entity_identifier"],
            "source_snapshot_id": [snapshot_id] * 3,
            "snapshot_as_of": [date(2023, 11, 3)] * 3,
            "address_group_key": ["12 RISK ROAD", "12 RISK ROAD", "9 SAFE ROAD"],
        }
    )
    cirp_frame = pl.DataFrame(
        {
            "solvency_event_id": ["prior", "window"],
            "cin": [
                "U12345DL2020PTC123456",
                "U54321DL2020PTC654321",
            ],
            "event_date": [date(2023, 10, 1), date(2025, 4, 1)],
        }
    )
    config = SolvencyObservationConfig(
        run_id="b" * 64,
        feature_cutoff=date(2023, 11, 3),
        outcome_window_end=date(2026, 8, 21),
    )

    observations = build_solvency_observations(
        company_frame,
        address_frame,
        cirp_frame,
        config,
    )
    by_company = {
        row["company_snapshot_id"]: row for row in observations.to_dicts()
    }

    assert by_company["company-1"]["has_prior_cirp_announcement"] is True
    assert by_company["company-1"]["evaluation_eligible"] is False
    assert by_company["company-2"]["evaluation_eligible"] is True
    assert by_company["company-2"]["observed_cirp_public_announcement"] is True
    assert by_company["company-2"]["first_cirp_announcement_date"] == date(
        2025,
        4,
        1,
    )
    assert by_company["company-2"]["shared_address_company_count"] == 2
    assert by_company["company-2"][
        "address_registration_month_company_count"
    ] == 2
    assert by_company["company-2"]["paid_to_authorised_capital_ratio"] == 0.5
    assert by_company["company-2"]["nic_division"] == "64"
    assert by_company["company-3"]["observed_cirp_public_announcement"] is False
    assert by_company["company-3"]["target_name"] == (
        "cirp_public_announcement_outcome"
    )



def test_address_cluster_shape_separates_shell_factory_from_service_provider() -> None:
    by_company = _observations(_shell_factory_rows() + _service_provider_rows())
    shell = by_company["shell-0"]
    service = by_company["service-0"]

    assert shell["shared_address_company_count"] == 50
    assert service["shared_address_company_count"] == 50

    assert shell["address_cluster_registration_span_days"] == 19
    assert service["address_cluster_registration_span_days"] == 4312
    assert (
        shell["address_cluster_registration_month_entropy"]
        < service["address_cluster_registration_month_entropy"]
    )
    assert shell["address_cluster_max_month_share"] == 1.0
    assert service["address_cluster_max_month_share"] < 0.2
    assert shell["address_cluster_nic_division_distinct"] == 1
    assert service["address_cluster_nic_division_distinct"] == 5
    assert shell["address_cluster_authorised_capital_cv"] == 0.0
    assert service["address_cluster_authorised_capital_cv"] > 0.5
    assert shell["address_cluster_distinct_name_head_ratio"] == 0.02
    assert service["address_cluster_distinct_name_head_ratio"] == 1.0


def test_address_cluster_shape_defaults_to_single_company_values() -> None:
    by_company = _observations(
        [
            {
                "id": "single",
                "cin": _cin(1),
                "address": "3 SOLO STREET",
                "registered_on": date(2021, 6, 6),
                "capital": 200_000,
                "nic_code": "64990",
                "name": "Solo Ventures Private Limited",
            },
            {
                "id": "blank-address",
                "cin": _cin(2),
                "address": "",
                "registered_on": date(2021, 6, 6),
                "capital": 200_000,
                "nic_code": "64990",
                "name": "Blank Address Private Limited",
            },
            {
                "id": "null-address",
                "cin": _cin(3),
                "address": None,
                "registered_on": date(2021, 6, 6),
                "capital": 200_000,
                "nic_code": None,
                "name": "Null Address Private Limited",
            },
        ]
    )
    single_company_shape = {
        "address_cluster_registration_span_days": 0,
        "address_cluster_registration_month_entropy": 0.0,
        "address_cluster_max_month_share": 1.0,
        "address_cluster_nic_division_distinct": 1,
        "address_cluster_authorised_capital_cv": 0.0,
        "address_cluster_distinct_name_head_ratio": 1.0,
    }
    for column, expected_value in single_company_shape.items():
        assert by_company["single"][column] == expected_value
        assert by_company["blank-address"][column] == expected_value

    assert by_company["single"]["shared_address_company_count"] == 1
    assert by_company["blank-address"]["shared_address_company_count"] == 0
    assert by_company["null-address"]["shared_address_company_count"] == 0
    assert by_company["null-address"]["address_cluster_nic_division_distinct"] == 0


def test_address_cluster_shape_absorbs_missing_registration_capital_and_nic() -> None:
    by_company = _observations(
        [
            {
                "id": "unobserved-a",
                "cin": _cin(11),
                "address": "4 UNOBSERVED WAY",
                "registered_on": None,
                "capital": None,
                "nic_code": None,
                "name": None,
            },
            {
                "id": "unobserved-b",
                "cin": _cin(12),
                "address": "4 UNOBSERVED WAY",
                "registered_on": None,
                "capital": None,
                "nic_code": None,
                "name": None,
            },
            {
                "id": "partial-a",
                "cin": _cin(13),
                "address": "5 PARTIAL WAY",
                "registered_on": date(2020, 5, 1),
                "capital": None,
                "nic_code": "64990",
                "name": "Partial Holdings Private Limited",
            },
            {
                "id": "partial-b",
                "cin": _cin(14),
                "address": "5 PARTIAL WAY",
                "registered_on": None,
                "capital": 500_000,
                "nic_code": None,
                "name": None,
            },
            {
                "id": "zero-capital-a",
                "cin": _cin(15),
                "address": "6 ZERO WAY",
                "registered_on": date(2020, 5, 1),
                "capital": 0,
                "nic_code": "64990",
                "name": "Zero Capital One Private Limited",
            },
            {
                "id": "zero-capital-b",
                "cin": _cin(16),
                "address": "6 ZERO WAY",
                "registered_on": date(2020, 6, 1),
                "capital": 0,
                "nic_code": "64990",
                "name": "Zero Capital Two Private Limited",
            },
        ]
    )

    unobserved = by_company["unobserved-a"]
    assert unobserved["shared_address_company_count"] == 2
    assert unobserved["address_cluster_registration_span_days"] == 0
    assert unobserved["address_cluster_registration_month_entropy"] == 0.0
    assert unobserved["address_cluster_max_month_share"] == 1.0
    assert unobserved["address_cluster_nic_division_distinct"] == 0
    assert unobserved["address_cluster_authorised_capital_cv"] == 0.0
    assert unobserved["address_cluster_distinct_name_head_ratio"] == 1.0

    partial = by_company["partial-a"]
    assert partial["address_cluster_registration_span_days"] == 0
    assert partial["address_cluster_max_month_share"] == 1.0
    assert partial["address_cluster_nic_division_distinct"] == 1
    assert partial["address_cluster_authorised_capital_cv"] == 0.0
    assert partial["address_cluster_distinct_name_head_ratio"] == 1.0

    zero_capital = by_company["zero-capital-a"]
    assert zero_capital["address_cluster_authorised_capital_cv"] == 0.0
    assert zero_capital["address_cluster_registration_span_days"] == 31
    assert zero_capital["address_cluster_max_month_share"] == 0.5
    assert zero_capital["address_cluster_registration_month_entropy"] == 1.0
    assert zero_capital["address_cluster_distinct_name_head_ratio"] == 0.5


def test_address_cluster_shape_is_independent_of_row_order() -> None:
    rows = _shell_factory_rows() + _service_provider_rows()
    forward = _observations(rows)
    reversed_rows = _observations(list(reversed(rows)))

    for company_snapshot_id, observation in forward.items():
        for column in SHAPE_COLUMNS:
            assert (
                reversed_rows[company_snapshot_id][column] == observation[column]
            )


def _cohort_cin(*, state: str, year: int, serial: int) -> str:
    """Build a CIN whose registrar-year cohort and ROC serial are explicit."""

    return f"U72900{state}{year}PTC{serial:06d}"


def test_roc_serial_adjacency_is_measured_inside_a_registrar_year_cohort() -> None:
    """Serials are only comparable within one registrar office and one year."""

    observations = _observations(
        [
            {
                "id": "same-cohort-low",
                "cin": _cohort_cin(state="DL", year=2020, serial=100),
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 1, 2),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Alpha Private Limited",
            },
            {
                "id": "same-cohort-high",
                "cin": _cohort_cin(state="DL", year=2020, serial=101),
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 1, 3),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Beta Private Limited",
            },
            {
                "id": "other-cohort",
                "cin": _cohort_cin(state="MH", year=2020, serial=102),
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 1, 4),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Gamma Private Limited",
            },
        ]
    )

    for company_id in ("same-cohort-low", "same-cohort-high"):
        assert observations[company_id]["registrar_year_cohort_company_count"] == 2
        assert observations[company_id]["address_cluster_cohort_peer_count"] == 1
        assert observations[company_id]["address_cluster_roc_serial_min_gap"] == 1
    other = observations["other-cohort"]
    assert other["registrar_year_cohort_company_count"] == 1
    assert other["address_cluster_cohort_peer_count"] == 0
    assert other["address_cluster_roc_serial_min_gap"] == -1


def test_roc_serial_gap_takes_the_nearest_neighbour_on_either_side() -> None:
    """A company between two peers reports the smaller of the two distances."""

    observations = _observations(
        [
            {
                "id": f"serial-{serial}",
                "cin": _cohort_cin(state="DL", year=2020, serial=serial),
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 1, 2),
                "capital": 100_000,
                "nic_code": "72900",
                "name": f"Company {serial} Private Limited",
            }
            for serial in (100, 105, 106)
        ]
    )

    assert observations["serial-100"]["address_cluster_roc_serial_min_gap"] == 5
    assert observations["serial-105"]["address_cluster_roc_serial_min_gap"] == 1
    assert observations["serial-106"]["address_cluster_roc_serial_min_gap"] == 1
    assert observations["serial-105"]["address_cluster_cohort_peer_count"] == 2


def test_roc_serial_adjacency_is_neutral_without_a_comparable_peer() -> None:
    """No address cluster, or a non-CIN identifier, yields no adjacency claim."""

    observations = _observations(
        [
            {
                "id": "no-address",
                "cin": _cohort_cin(state="DL", year=2020, serial=100),
                "address": None,
                "registered_on": date(2020, 1, 2),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Alpha Private Limited",
            },
            {
                "id": "llpin",
                "cin": "AAB-1234",
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 1, 2),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Beta LLP",
            },
        ]
    )

    no_address = observations["no-address"]
    assert no_address["registrar_year_cohort_company_count"] == 1
    assert no_address["address_cluster_cohort_peer_count"] == 0
    assert no_address["address_cluster_roc_serial_min_gap"] == -1
    llpin = observations["llpin"]
    assert llpin["registrar_year_cohort_company_count"] == 0
    assert llpin["address_cluster_cohort_peer_count"] == 0
    assert llpin["address_cluster_roc_serial_min_gap"] == -1


def test_roc_serial_adjacency_is_independent_of_row_order() -> None:
    """Adjacency is a group property, so input ordering cannot change it."""

    rows = [
        {
            "id": f"serial-{serial}",
            "cin": _cohort_cin(state="DL", year=2020, serial=serial),
            "address": "1 SHELL LANE",
            "registered_on": date(2020, 1, 2),
            "capital": 100_000,
            "nic_code": "72900",
            "name": f"Company {serial} Private Limited",
        }
        for serial in (100, 105, 106)
    ]
    forward = _observations(rows)
    reversed_rows = _observations(list(reversed(rows)))

    for company_id, row in forward.items():
        for column in (
            "registrar_year_cohort_company_count",
            "address_cluster_cohort_peer_count",
            "address_cluster_roc_serial_min_gap",
        ):
            assert row[column] == reversed_rows[company_id][column]


def test_cin_record_disagreement_is_surfaced_per_company() -> None:
    """The identifier's own segments are compared against the record columns."""

    observations = _observations(
        [
            {
                "id": "agrees",
                "cin": _cohort_cin(state="DL", year=2020, serial=100),
                "address": "1 SHELL LANE",
                "registered_on": date(2020, 6, 1),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Alpha Private Limited",
            },
            {
                "id": "year-disagrees",
                "cin": _cohort_cin(state="DL", year=2020, serial=101),
                "address": "1 SHELL LANE",
                "registered_on": date(2019, 6, 1),
                "capital": 100_000,
                "nic_code": "72900",
                "name": "Beta Private Limited",
            },
        ]
    )

    agrees = observations["agrees"]
    assert agrees["cin_year_disagrees_with_record"] is False
    assert agrees["cin_nic_division_disagrees_with_record"] is False
    assert agrees["cin_record_disagreement_count"] == 0
    disagrees = observations["year-disagrees"]
    assert disagrees["cin_year_disagrees_with_record"] is True
    assert disagrees["cin_record_disagreement_count"] == 1
