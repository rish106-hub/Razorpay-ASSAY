from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_prerequisites


def test_disk_headroom_is_measured_on_raw_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for required_directory in check_prerequisites.REQUIRED_DIRECTORIES:
        (tmp_path / required_directory).mkdir(parents=True, exist_ok=True)
    measured_paths: list[Path] = []

    def fake_disk_usage(measured_path: Path) -> SimpleNamespace:
        measured_paths.append(measured_path)
        return SimpleNamespace(free=30 * 1024**3)

    monkeypatch.setattr(check_prerequisites.shutil, "disk_usage", fake_disk_usage)

    assert check_prerequisites.main(tmp_path) == 0
    assert measured_paths == [tmp_path / "data/raw"]
    assert "Prerequisites passed" in capsys.readouterr().out


def test_missing_directories_fail_before_disk_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called(_: Path) -> None:
        raise AssertionError("disk usage must not run before directory validation")

    monkeypatch.setattr(check_prerequisites.shutil, "disk_usage", fail_if_called)

    assert check_prerequisites.main(tmp_path) == 1
    assert "Missing directories:" in capsys.readouterr().out


def test_low_raw_store_headroom_fails_with_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for required_directory in check_prerequisites.REQUIRED_DIRECTORIES:
        (tmp_path / required_directory).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        check_prerequisites.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=1 * 1024**3),
    )

    assert check_prerequisites.main(tmp_path) == 1
    assert "Need at least" in capsys.readouterr().out
