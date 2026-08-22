from __future__ import annotations

from pathlib import Path

import pytest

from assay.acquisition.mca import McaAcquisitionError
from assay.cli import acquire_mca
from assay.cli.acquire_mca import _load_data_gov_in_api_key


def test_env_file_loader_reads_only_requested_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UNRELATED_FINTECH_KEY=ignore\nDATA_GOV_IN_API_KEY='merchant-risk-key'\n",
        encoding="utf-8",
    )

    assert _load_data_gov_in_api_key(env_file) == "merchant-risk-key"


def test_environment_credential_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_GOV_IN_API_KEY", "environment-merchant-risk-key")

    assert (
        _load_data_gov_in_api_key(tmp_path / "missing.env")
        == "environment-merchant-risk-key"
    )


def test_missing_credential_has_secret_safe_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)

    with pytest.raises(McaAcquisitionError, match="missing from the environment"):
        _load_data_gov_in_api_key(tmp_path / "missing.env")


def test_cli_returns_secret_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATA_GOV_IN_API_KEY", "never-print-cli-key")
    monkeypatch.setattr(
        acquire_mca.McaCompanyMasterDownloader,
        "acquire",
        lambda _downloader, _api_key: (_ for _ in ()).throw(
            McaAcquisitionError("bounded merchant-risk failure")
        ),
    )
    monkeypatch.setattr("sys.argv", ["acquire_mca"])

    assert acquire_mca.main() == 1
    captured_error = capsys.readouterr().err
    assert "bounded merchant-risk failure" in captured_error
    assert "never-print-cli-key" not in captured_error
