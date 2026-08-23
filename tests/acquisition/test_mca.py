from __future__ import annotations

import hashlib
import http.client
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

import pytest

from assay.acquisition.mca import (
    McaAcquisitionConfig,
    McaAcquisitionError,
    McaApiClient,
    McaCompanyMasterDownloader,
    _validate_mca_page,
)

MERCHANT_ENTITY_RECORD = {
    "CIN": "U00000DL2020PTC000001",
    "CompanyRegistrationdate_date": "01-01-2020",
    "CompanyName": "Merchant Evidence Private Limited",
    "CompanyStatus": "Active",
    "Registered_Office_Address": "Evidence Street, Delhi",
}


def _mca_page(*, total: int, records: list[dict[str, str]]) -> bytes:
    return json.dumps(
        {
            "status": "ok",
            "total": total,
            "count": len(records),
            "records": records,
        },
        separators=(",", ":"),
    ).encode("utf-8")


class QueuedTransport:
    def __init__(self, outcomes: Iterator[bytes | Exception]) -> None:
        self._outcomes = outcomes
        self.request_urls: list[str] = []

    def __call__(self, request_url: str, timeout_seconds: float) -> bytes:
        assert timeout_seconds > 0
        self.request_urls.append(request_url)
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_downloader_publishes_immutable_pages_and_resumes(tmp_path: Path) -> None:
    first_page = _mca_page(
        total=3,
        records=[MERCHANT_ENTITY_RECORD, MERCHANT_ENTITY_RECORD],
    )
    final_page = _mca_page(total=3, records=[MERCHANT_ENTITY_RECORD])
    transport = QueuedTransport(iter([first_page, final_page]))
    config = McaAcquisitionConfig(
        output_directory=tmp_path,
        page_size=2,
        max_pages=1,
        minimum_free_disk_gb=0.001,
    )
    merchant_api_client = McaApiClient(config, transport=transport, sleep=lambda _: None)
    fixed_time = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)

    first_checkpoint = McaCompanyMasterDownloader(
        config,
        api_client=merchant_api_client,
        now=lambda: fixed_time,
    ).acquire("private-test-key")

    assert first_checkpoint.next_offset == 2
    assert first_checkpoint.total_records == 3
    assert len(first_checkpoint.pages) == 1
    first_artifact = first_checkpoint.pages[0]
    first_raw_path = tmp_path / first_artifact.raw_path
    assert first_raw_path.read_bytes() == first_page
    assert first_artifact.sha256 == hashlib.sha256(first_page).hexdigest()

    final_checkpoint = McaCompanyMasterDownloader(
        config,
        api_client=merchant_api_client,
        now=lambda: fixed_time,
    ).acquire("private-test-key")

    assert final_checkpoint.next_offset == 3
    assert final_checkpoint.total_records == 3
    assert len(final_checkpoint.pages) == 2
    assert len(transport.request_urls) == 2
    assert "offset=0" in transport.request_urls[0]
    assert "offset=2" in transport.request_urls[1]


def test_api_client_honours_rate_limit_before_retry(tmp_path: Path) -> None:
    rate_limit_error = HTTPError(
        url="https://api.data.gov.in/redacted",
        code=429,
        msg="rate limited",
        hdrs={"Retry-After": "3"},
        fp=None,
    )
    transport = QueuedTransport(iter([rate_limit_error, b"merchant-page"]))
    retry_delays: list[float] = []
    config = McaAcquisitionConfig(
        output_directory=tmp_path,
        page_size=2,
        max_attempts=2,
        initial_backoff_seconds=1,
        max_backoff_seconds=10,
        minimum_free_disk_gb=0.001,
    )

    raw_page = McaApiClient(
        config,
        transport=transport,
        sleep=retry_delays.append,
    ).fetch_page("private-test-key", offset=0)

    assert raw_page == b"merchant-page"
    assert retry_delays == [3.0]
    assert len(transport.request_urls) == 2


def test_api_client_never_exposes_api_key_in_failure(tmp_path: Path) -> None:
    transport = QueuedTransport(iter([TimeoutError(), TimeoutError()]))
    config = McaAcquisitionConfig(
        output_directory=tmp_path,
        page_size=2,
        max_attempts=2,
        initial_backoff_seconds=0,
        minimum_free_disk_gb=0.001,
    )

    with pytest.raises(McaAcquisitionError) as captured_error:
        McaApiClient(
            config,
            transport=transport,
            sleep=lambda _: None,
        ).fetch_page("never-print-this-key", offset=0)

    assert "never-print-this-key" not in str(captured_error.value)


def test_api_client_retries_interrupted_response_body(tmp_path: Path) -> None:
    interrupted_response = http.client.IncompleteRead(
        partial=b"partial-merchant-page",
        expected=100,
    )
    transport = QueuedTransport(iter([interrupted_response, b"complete-page"]))
    config = McaAcquisitionConfig(
        output_directory=tmp_path,
        page_size=2,
        max_attempts=2,
        initial_backoff_seconds=0,
        minimum_free_disk_gb=0.001,
    )

    raw_page = McaApiClient(
        config,
        transport=transport,
        sleep=lambda _: None,
    ).fetch_page("private-test-key", offset=0)

    assert raw_page == b"complete-page"
    assert len(transport.request_urls) == 2


def test_page_validation_rejects_missing_risk_evidence_fields() -> None:
    incomplete_merchant_record = {
        "CIN": "U00000DL2020PTC000001",
        "CompanyName": "Incomplete Merchant Private Limited",
    }

    with pytest.raises(McaAcquisitionError, match="missing required fields"):
        _validate_mca_page(
            _mca_page(total=1, records=[incomplete_merchant_record]),
            requested_limit=1,
        )


def test_page_validation_checks_every_merchant_record() -> None:
    incomplete_merchant_record = {
        "CIN": "U00000DL2020PTC000002",
        "CompanyName": "Incomplete Merchant Private Limited",
    }

    with pytest.raises(McaAcquisitionError, match="record 1"):
        _validate_mca_page(
            _mca_page(
                total=2,
                records=[MERCHANT_ENTITY_RECORD, incomplete_merchant_record],
            ),
            requested_limit=2,
        )


def test_downloader_rejects_orphaned_raw_page(tmp_path: Path) -> None:
    (tmp_path / "page-orphan.json").write_text("{}", encoding="utf-8")
    config = McaAcquisitionConfig(
        output_directory=tmp_path,
        page_size=1,
        minimum_free_disk_gb=0.001,
    )

    with pytest.raises(McaAcquisitionError, match="manual audit"):
        McaCompanyMasterDownloader(config).acquire("private-test-key")
