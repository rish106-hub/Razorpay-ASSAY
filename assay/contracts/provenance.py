"""Validated provenance contracts for merchant-risk data sources and assets."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class MerchantRiskSource(BaseModel):
    """Required provenance declared once for every upstream source."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    canonical_url: HttpUrl
    licence: str = Field(min_length=1)
    access: str = Field(min_length=1)
    constraints: tuple[str, ...] = Field(min_length=1)


class MerchantRiskAsset(BaseModel):
    """Per-file provenance that never relies on implicit source inheritance."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_url: HttpUrl
    publisher: str = Field(min_length=1)
    retrieved_at: datetime
    extract_date: date | Literal["unknown"]
    licence_or_terms: str = Field(min_length=1)
    intended_use: str = Field(min_length=1)
    restrictions: tuple[str, ...] = Field(min_length=1)
    provenance_status: Literal["complete", "incomplete"]
    local_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    format: str = Field(min_length=1)
    status: str = Field(min_length=1)
    sheet: str = Field(min_length=1)
    rows: int = Field(gt=0)
    columns: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_false_complete_status(self) -> MerchantRiskAsset:
        has_unknown_provenance = (
            self.extract_date == "unknown"
            or self.licence_or_terms.lower().startswith("unknown")
        )
        if has_unknown_provenance and self.provenance_status == "complete":
            raise ValueError(
                "an asset with unknown provenance cannot be marked complete"
            )
        return self


class SourceManifest(BaseModel):
    """Collection of unique merchant-risk evidence sources."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[MerchantRiskSource, ...]

    @model_validator(mode="after")
    def require_unique_source_ids(self) -> SourceManifest:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source ids must be unique")
        return self


class AcquisitionManifest(BaseModel):
    """Collection of unique immutable source-file acquisition records."""

    model_config = ConfigDict(frozen=True)

    assets: tuple[MerchantRiskAsset, ...]

    @model_validator(mode="after")
    def require_unique_asset_ids(self) -> AcquisitionManifest:
        asset_ids = [asset.id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset ids must be unique")
        return self


def _read_yaml_mapping(manifest_path: Path) -> dict[str, object]:
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise TypeError(f"{manifest_path} must contain a YAML mapping")
    return manifest_payload


def load_and_validate_manifests(
    source_manifest_path: Path,
    acquisition_manifest_path: Path,
) -> tuple[SourceManifest, AcquisitionManifest]:
    """Load manifests and reject assets that reference undeclared sources."""

    source_manifest = SourceManifest.model_validate(
        _read_yaml_mapping(source_manifest_path)
    )
    acquisition_manifest = AcquisitionManifest.model_validate(
        _read_yaml_mapping(acquisition_manifest_path)
    )
    source_ids = {source.id for source in source_manifest.sources}
    unknown_source_ids = {
        asset.source_id
        for asset in acquisition_manifest.assets
        if asset.source_id not in source_ids
    }
    if unknown_source_ids:
        unknown_sources = ", ".join(sorted(unknown_source_ids))
        raise ValueError(f"assets reference undeclared sources: {unknown_sources}")
    return source_manifest, acquisition_manifest
