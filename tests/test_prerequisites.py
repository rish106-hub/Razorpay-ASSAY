from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import check_prerequisites


def test_required_data_directories_are_declared() -> None:
    assert check_prerequisites.REQUIRED_DIRECTORIES == (
        Path("data/raw"),
        Path("data/staged"),
        Path("data/curated"),
        Path("data/manifests"),
    )


def test_bulk_acquisition_requires_safety_headroom() -> None:
    assert check_prerequisites.MIN_FREE_GB == 25


def test_disk_headroom_is_measured_on_raw_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for required_directory in check_prerequisites.REQUIRED_DIRECTORIES:
        required_directory.mkdir(parents=True, exist_ok=True)
    measured_paths: list[Path] = []

    def fake_disk_usage(measured_path: Path) -> SimpleNamespace:
        measured_paths.append(measured_path)
        return SimpleNamespace(free=30 * 1024**3)

    monkeypatch.setattr(check_prerequisites.shutil, "disk_usage", fake_disk_usage)

    assert check_prerequisites.main() == 0
    assert measured_paths == [Path("data/raw")]


def test_missing_directories_fail_before_disk_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_if_called(_: Path) -> None:
        raise AssertionError("disk usage must not run before directory validation")

    monkeypatch.setattr(check_prerequisites.shutil, "disk_usage", fail_if_called)

    assert check_prerequisites.main() == 1
