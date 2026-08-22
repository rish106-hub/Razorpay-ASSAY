"""Resumable acquisition for official MCA company-master evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MCA_COMPANY_MASTER_RESOURCE_ID = "4dbe5667-7b6b-41d7-82af-211562424d9a"
DATA_GOV_IN_RESOURCE_BASE_URL = "https://api.data.gov.in/resource"
MCA_SOURCE_ID = "mca_company_master"
MCA_RAW_SCHEMA_VERSION = "1.0.0"
MAX_MCA_PAGE_SIZE = 10_000

REQUIRED_MERCHANT_ENTITY_FIELD_ALIASES = {
    "CORPORATE_IDENTIFICATION_NUMBER": frozenset(
        {"CORPORATE_IDENTIFICATION_NUMBER", "CIN"}
    ),
    "DATE_OF_REGISTRATION": frozenset(
        {"DATE_OF_REGISTRATION", "COMPANY_REGISTRATIONDATE_DATE"}
    ),
    "COMPANY_NAME": frozenset({"COMPANY_NAME"}),
    "COMPANY_STATUS": frozenset({"COMPANY_STATUS"}),
    "REGISTERED_OFFICE_ADDRESS": frozenset({"REGISTERED_OFFICE_ADDRESS"}),
}
RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

HttpTransport = Callable[[str, float], bytes]
Sleep = Callable[[float], None]


class McaAcquisitionError(RuntimeError):
    """A secret-safe failure raised by the MCA acquisition boundary."""


class McaAcquisitionConfig(BaseModel):
    """Controls bounded MCA acquisition without weakening evidence safeguards."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path = Path("data/raw/mca_company_master/pagesize-10000")
    resource_id: str = MCA_COMPANY_MASTER_RESOURCE_ID
    page_size: int = Field(default=MAX_MCA_PAGE_SIZE, ge=1, le=MAX_MCA_PAGE_SIZE)
    max_pages: int | None = Field(default=1, ge=1)
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    max_attempts: int = Field(default=4, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=1.0, ge=0)
    max_backoff_seconds: float = Field(default=30.0, ge=0)
    minimum_free_disk_gb: float = Field(default=25.0, gt=0)

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, resource_id: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", resource_id):
            raise ValueError("resource_id must be a UUID")
        return resource_id.lower()


class McaPageArtifact(BaseModel):
    """Provenance for one immutable MCA API response page."""

    model_config = ConfigDict(frozen=True)

    source_id: str = MCA_SOURCE_ID
    resource_id: str
    schema_version: str = MCA_RAW_SCHEMA_VERSION
    offset: int = Field(ge=0)
    requested_limit: int = Field(ge=1)
    record_count: int = Field(ge=0)
    total_records: int = Field(ge=0)
    retrieved_at: datetime
    raw_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)


class McaAcquisitionCheckpoint(BaseModel):
    """Mutable progress pointer over immutable MCA page artifacts."""

    model_config = ConfigDict(frozen=True)

    source_id: str = MCA_SOURCE_ID
    resource_id: str
    schema_version: str = MCA_RAW_SCHEMA_VERSION
    page_size: int = Field(ge=1, le=MAX_MCA_PAGE_SIZE)
    next_offset: int = Field(ge=0)
    total_records: int | None = Field(default=None, ge=0)
    pages: tuple[McaPageArtifact, ...] = ()


class McaPageEnvelope(BaseModel):
    """Validated fields required to advance an MCA acquisition checkpoint."""

    model_config = ConfigDict(frozen=True)

    total_records: int = Field(ge=0)
    records: tuple[dict[str, Any], ...]


def _default_http_transport(url: str, timeout_seconds: float) -> bytes:
    merchant_risk_request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "ASSAY/0.1 merchant-risk-signal-audit",
        },
    )
    with urlopen(merchant_risk_request, timeout=timeout_seconds) as response:
        return response.read()


def _normalise_source_field(source_field: str) -> str:
    snake_case_field = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", source_field)
    return re.sub(r"[^A-Z0-9]+", "_", snake_case_field.upper()).strip("_")


def _parse_non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise McaAcquisitionError(f"MCA response {field_name} is not an integer.")
    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as error:
        raise McaAcquisitionError(
            f"MCA response {field_name} is not an integer."
        ) from error
    if parsed_value < 0:
        raise McaAcquisitionError(
            f"MCA response {field_name} must be non-negative."
        )
    return parsed_value


def _validate_mca_page(raw_page_bytes: bytes, requested_limit: int) -> McaPageEnvelope:
    try:
        response_payload = json.loads(raw_page_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise McaAcquisitionError("MCA response is not valid JSON.") from error

    if not isinstance(response_payload, dict):
        raise McaAcquisitionError("MCA response root must be an object.")
    if response_payload.get("status") != "ok":
        raise McaAcquisitionError("MCA response status is not ok.")

    merchant_entity_records = response_payload.get("records")
    if not isinstance(merchant_entity_records, list):
        raise McaAcquisitionError("MCA response records must be a list.")
    if any(not isinstance(record, dict) for record in merchant_entity_records):
        raise McaAcquisitionError("MCA response contains a non-object record.")

    total_records = _parse_non_negative_integer(
        response_payload.get("total"), "total"
    )
    declared_count = _parse_non_negative_integer(
        response_payload.get("count"), "count"
    )
    if declared_count != len(merchant_entity_records):
        raise McaAcquisitionError(
            "MCA response count does not match the records returned."
        )
    if declared_count > requested_limit:
        raise McaAcquisitionError("MCA response exceeds the requested page limit.")

    for record_index, merchant_entity_record in enumerate(merchant_entity_records):
        normalised_fields = {
            _normalise_source_field(source_field)
            for source_field in merchant_entity_record
        }
        missing_fields = {
            canonical_field
            for canonical_field, source_aliases in (
                REQUIRED_MERCHANT_ENTITY_FIELD_ALIASES.items()
            )
            if normalised_fields.isdisjoint(source_aliases)
        }
        if missing_fields:
            missing_field_list = ", ".join(sorted(missing_fields))
            raise McaAcquisitionError(
                "MCA response record "
                f"{record_index} is missing required fields: {missing_field_list}."
            )

    try:
        return McaPageEnvelope(
            total_records=total_records,
            records=tuple(merchant_entity_records),
        )
    except ValidationError as error:
        raise McaAcquisitionError("MCA response failed schema validation.") from error


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


class McaApiClient:
    """Secret-safe data.gov.in client with bounded retry behavior."""

    def __init__(
        self,
        config: McaAcquisitionConfig,
        *,
        transport: HttpTransport = _default_http_transport,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._config = config
        self._transport = transport
        self._sleep = sleep

    def fetch_page(self, api_key: str, offset: int) -> bytes:
        if not api_key:
            raise McaAcquisitionError("DATA_GOV_IN_API_KEY is required.")
        if offset < 0:
            raise McaAcquisitionError("MCA page offset must be non-negative.")

        request_url = self._build_request_url(api_key=api_key, offset=offset)
        last_failure = "network failure"

        for attempt_number in range(1, self._config.max_attempts + 1):
            retry_after = None
            try:
                return self._transport(
                    request_url, self._config.request_timeout_seconds
                )
            except HTTPError as error:
                last_failure = f"HTTP {error.code}"
                if error.code not in RETRYABLE_HTTP_STATUS_CODES:
                    raise McaAcquisitionError(
                        f"MCA request rejected with HTTP {error.code}."
                    ) from None
                retry_after = _retry_after_seconds(error.headers)
            except (TimeoutError, URLError):
                last_failure = "network timeout or connection failure"

            if attempt_number == self._config.max_attempts:
                break
            self._sleep(self._retry_delay(attempt_number, retry_after))

        raise McaAcquisitionError(
            "MCA request failed after "
            f"{self._config.max_attempts} attempts ({last_failure})."
        )

    def _build_request_url(self, api_key: str, offset: int) -> str:
        merchant_risk_query = urlencode(
            {
                "api-key": api_key,
                "format": "json",
                "offset": offset,
                "limit": self._config.page_size,
            }
        )
        return (
            f"{DATA_GOV_IN_RESOURCE_BASE_URL}/{self._config.resource_id}"
            f"?{merchant_risk_query}"
        )

    def _retry_delay(
        self, attempt_number: int, retry_after_seconds: float | None
    ) -> float:
        exponential_delay = self._config.initial_backoff_seconds * (
            2 ** (attempt_number - 1)
        )
        requested_delay = (
            exponential_delay
            if retry_after_seconds is None
            else max(exponential_delay, retry_after_seconds)
        )
        return min(requested_delay, self._config.max_backoff_seconds)


class McaCompanyMasterDownloader:
    """Publishes checksum-addressed MCA pages and a resumable checkpoint."""

    def __init__(
        self,
        config: McaAcquisitionConfig,
        *,
        api_client: McaApiClient | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._api_client = api_client or McaApiClient(config)
        self._now = now or (lambda: datetime.now(UTC))

    def acquire(self, api_key: str) -> McaAcquisitionCheckpoint:
        merchant_evidence_directory = self._config.output_directory
        merchant_evidence_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self._load_checkpoint()
        acquired_page_count = 0

        while not self._is_complete(checkpoint):
            if (
                self._config.max_pages is not None
                and acquired_page_count >= self._config.max_pages
            ):
                break

            self._require_disk_headroom()
            raw_page_bytes = self._api_client.fetch_page(
                api_key=api_key,
                offset=checkpoint.next_offset,
            )
            merchant_page = _validate_mca_page(
                raw_page_bytes,
                requested_limit=self._config.page_size,
            )
            self._validate_page_progress(checkpoint, merchant_page)
            merchant_page_artifact = self._publish_page(
                raw_page_bytes=raw_page_bytes,
                merchant_page=merchant_page,
                offset=checkpoint.next_offset,
            )
            checkpoint = McaAcquisitionCheckpoint(
                resource_id=self._config.resource_id,
                page_size=self._config.page_size,
                next_offset=(
                    checkpoint.next_offset + merchant_page_artifact.record_count
                ),
                total_records=merchant_page_artifact.total_records,
                pages=(*checkpoint.pages, merchant_page_artifact),
            )
            self._write_checkpoint(checkpoint)
            acquired_page_count += 1

        return checkpoint

    @property
    def checkpoint_path(self) -> Path:
        return self._config.output_directory / "checkpoint.json"

    def _load_checkpoint(self) -> McaAcquisitionCheckpoint:
        if not self.checkpoint_path.exists():
            orphaned_pages = tuple(self._config.output_directory.glob("page-*.json"))
            if orphaned_pages:
                raise McaAcquisitionError(
                    "MCA raw pages exist without a checkpoint; manual audit is required."
                )
            return McaAcquisitionCheckpoint(
                resource_id=self._config.resource_id,
                page_size=self._config.page_size,
                next_offset=0,
            )

        try:
            checkpoint = McaAcquisitionCheckpoint.model_validate_json(
                self.checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise McaAcquisitionError("MCA checkpoint is invalid.") from error
        if checkpoint.resource_id != self._config.resource_id:
            raise McaAcquisitionError("MCA checkpoint resource_id does not match.")
        if checkpoint.page_size != self._config.page_size:
            raise McaAcquisitionError("MCA checkpoint page_size does not match.")
        self._verify_checkpoint_artifacts(checkpoint)
        return checkpoint

    def _verify_checkpoint_artifacts(
        self, checkpoint: McaAcquisitionCheckpoint
    ) -> None:
        expected_next_offset = 0
        for merchant_page_artifact in checkpoint.pages:
            if merchant_page_artifact.offset != expected_next_offset:
                raise McaAcquisitionError("MCA checkpoint offsets are not contiguous.")
            raw_page_path = self._config.output_directory / Path(
                merchant_page_artifact.raw_path
            ).name
            if not raw_page_path.is_file():
                raise McaAcquisitionError(
                    "MCA checkpoint references a missing raw page."
                )
            actual_sha256 = hashlib.sha256(raw_page_path.read_bytes()).hexdigest()
            if actual_sha256 != merchant_page_artifact.sha256:
                raise McaAcquisitionError("MCA raw page checksum does not match.")
            expected_next_offset += merchant_page_artifact.record_count
        if expected_next_offset != checkpoint.next_offset:
            raise McaAcquisitionError("MCA checkpoint next_offset is inconsistent.")

    def _require_disk_headroom(self) -> None:
        free_disk_gb = shutil.disk_usage(self._config.output_directory).free / 1024**3
        if free_disk_gb < self._config.minimum_free_disk_gb:
            raise McaAcquisitionError(
                "MCA acquisition stopped because free disk is below "
                f"{self._config.minimum_free_disk_gb:.1f} GB."
            )

    def _validate_page_progress(
        self,
        checkpoint: McaAcquisitionCheckpoint,
        merchant_page: McaPageEnvelope,
    ) -> None:
        record_count = len(merchant_page.records)
        if (
            checkpoint.total_records is not None
            and merchant_page.total_records != checkpoint.total_records
        ):
            raise McaAcquisitionError(
                "MCA total record count changed during acquisition."
            )
        if record_count == 0 and checkpoint.next_offset < merchant_page.total_records:
            raise McaAcquisitionError(
                "MCA returned an empty page before the declared final offset."
            )
        remaining_records = merchant_page.total_records - checkpoint.next_offset
        if (
            remaining_records > self._config.page_size
            and record_count != self._config.page_size
        ):
            raise McaAcquisitionError(
                "MCA returned a short page before the final page."
            )
        if record_count > max(0, remaining_records):
            raise McaAcquisitionError(
                "MCA returned more rows than its declared total permits."
            )

    def _publish_page(
        self,
        *,
        raw_page_bytes: bytes,
        merchant_page: McaPageEnvelope,
        offset: int,
    ) -> McaPageArtifact:
        raw_page_sha256 = hashlib.sha256(raw_page_bytes).hexdigest()
        raw_page_name = (
            f"page-offset-{offset:010d}-limit-{self._config.page_size:05d}-"
            f"sha256-{raw_page_sha256[:16]}.json"
        )
        raw_page_path = self._config.output_directory / raw_page_name
        if raw_page_path.exists():
            raise McaAcquisitionError(
                "MCA raw page already exists outside the current checkpoint."
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._config.output_directory,
                prefix=".mca-page-",
                delete=False,
            ) as temporary_page:
                temporary_page.write(raw_page_bytes)
                temporary_page.flush()
                os.fsync(temporary_page.fileno())
                temporary_path = Path(temporary_page.name)
            os.link(temporary_path, raw_page_path)
        except FileExistsError as error:
            raise McaAcquisitionError("MCA raw page publish collision.") from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return McaPageArtifact(
            resource_id=self._config.resource_id,
            offset=offset,
            requested_limit=self._config.page_size,
            record_count=len(merchant_page.records),
            total_records=merchant_page.total_records,
            retrieved_at=self._now(),
            raw_path=raw_page_name,
            sha256=raw_page_sha256,
            byte_count=len(raw_page_bytes),
        )

    def _write_checkpoint(self, checkpoint: McaAcquisitionCheckpoint) -> None:
        checkpoint_payload = json.dumps(
            checkpoint.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        temporary_checkpoint_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._config.output_directory,
                prefix=".mca-checkpoint-",
                delete=False,
            ) as temporary_checkpoint:
                temporary_checkpoint.write(checkpoint_payload)
                temporary_checkpoint.flush()
                os.fsync(temporary_checkpoint.fileno())
                temporary_checkpoint_path = Path(temporary_checkpoint.name)
            os.replace(temporary_checkpoint_path, self.checkpoint_path)
            temporary_checkpoint_path = None
        finally:
            if temporary_checkpoint_path is not None:
                temporary_checkpoint_path.unlink(missing_ok=True)

    @staticmethod
    def _is_complete(checkpoint: McaAcquisitionCheckpoint) -> bool:
        return (
            checkpoint.total_records is not None
            and checkpoint.next_offset >= checkpoint.total_records
        )
