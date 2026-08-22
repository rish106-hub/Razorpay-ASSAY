"""Validate merchant-risk source and acquisition provenance manifests."""

from __future__ import annotations

from pathlib import Path

from assay.contracts.provenance import load_and_validate_manifests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    source_manifest, acquisition_manifest = load_and_validate_manifests(
        PROJECT_ROOT / "data/manifests/sources.yaml",
        PROJECT_ROOT / "data/manifests/acquired.yaml",
    )
    incomplete_asset_count = sum(
        asset.provenance_status == "incomplete"
        for asset in acquisition_manifest.assets
    )
    print(f"Sources validated: {len(source_manifest.sources)}")
    print(f"Assets validated: {len(acquisition_manifest.assets)}")
    print(f"Assets with incomplete provenance: {incomplete_asset_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
