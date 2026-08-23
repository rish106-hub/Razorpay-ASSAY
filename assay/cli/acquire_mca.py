"""Acquire a bounded, resumable slice of MCA company-master data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from assay.acquisition.mca import (
    McaAcquisitionConfig,
    McaAcquisitionError,
    McaCompanyMasterDownloader,
)

DATA_GOV_IN_API_KEY_NAME = "DATA_GOV_IN_API_KEY"


def _load_data_gov_in_api_key(env_file: Path) -> str:
    environment_api_key = os.environ.get(DATA_GOV_IN_API_KEY_NAME, "").strip()
    if environment_api_key:
        return environment_api_key
    if not env_file.is_file():
        raise McaAcquisitionError(
            f"{DATA_GOV_IN_API_KEY_NAME} is missing from the environment."
        )

    for env_line in env_file.read_text(encoding="utf-8").splitlines():
        stripped_line = env_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        env_name, separator, env_value = stripped_line.partition("=")
        if separator and env_name.strip() == DATA_GOV_IN_API_KEY_NAME:
            return env_value.strip().strip("\"'")
    raise McaAcquisitionError(
        f"{DATA_GOV_IN_API_KEY_NAME} is missing from {env_file}."
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    merchant_risk_parser = argparse.ArgumentParser(
        description="Acquire immutable MCA pages for merchant-risk signal audits."
    )
    merchant_risk_parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/raw/mca_company_master/pagesize-10000"),
    )
    merchant_risk_parser.add_argument("--page-size", type=int, default=10_000)
    merchant_risk_parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages for this run. Use 0 only for an intentional full pull.",
    )
    merchant_risk_parser.add_argument(
        "--request-timeout-seconds", type=float, default=45.0
    )
    merchant_risk_parser.add_argument("--max-attempts", type=int, default=4)
    merchant_risk_parser.add_argument(
        "--minimum-free-disk-gb", type=float, default=25.0
    )
    merchant_risk_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return merchant_risk_parser


def _run_acquisition() -> int:
    merchant_risk_arguments = _build_argument_parser().parse_args()
    api_key = _load_data_gov_in_api_key(merchant_risk_arguments.env_file)
    max_pages = (
        None
        if merchant_risk_arguments.max_pages == 0
        else merchant_risk_arguments.max_pages
    )
    acquisition_config = McaAcquisitionConfig(
        output_directory=merchant_risk_arguments.output_directory,
        page_size=merchant_risk_arguments.page_size,
        max_pages=max_pages,
        request_timeout_seconds=merchant_risk_arguments.request_timeout_seconds,
        max_attempts=merchant_risk_arguments.max_attempts,
        minimum_free_disk_gb=merchant_risk_arguments.minimum_free_disk_gb,
    )
    checkpoint = McaCompanyMasterDownloader(acquisition_config).acquire(api_key)
    print(
        json.dumps(
            {
                "source_id": checkpoint.source_id,
                "pages_acquired": len(checkpoint.pages),
                "records_acquired": checkpoint.next_offset,
                "total_records": checkpoint.total_records,
                "complete": (
                    checkpoint.total_records is not None
                    and checkpoint.next_offset >= checkpoint.total_records
                ),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    try:
        return _run_acquisition()
    except (McaAcquisitionError, ValidationError) as error:
        print(f"MCA acquisition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
