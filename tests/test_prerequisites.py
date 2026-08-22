from __future__ import annotations

from pathlib import Path

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
