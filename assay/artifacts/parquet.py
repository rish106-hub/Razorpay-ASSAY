"""Atomic, checksum-addressable Parquet artifact publishing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class ParquetArtifact(BaseModel):
    """Checksum and row count for one immutable Parquet artifact."""

    model_config = ConfigDict(frozen=True)

    path: str
    rows: int = Field(ge=0)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ImmutableArtifactError(RuntimeError):
    """An immutable artifact could not be published safely."""


def sha256_file(artifact_path: Path) -> str:
    """Hash a file without loading the full artifact into memory."""

    artifact_digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for artifact_chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            artifact_digest.update(artifact_chunk)
    return artifact_digest.hexdigest()


def write_immutable_parquet(
    frame: pl.DataFrame,
    artifact_path: Path,
    compression: str,
) -> ParquetArtifact:
    """Write then atomically hard-link a Parquet file into its final path."""

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        raise ImmutableArtifactError(
            f"Immutable artifact already exists: {artifact_path}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=artifact_path.parent,
            prefix=".assay-parquet-",
            suffix=".parquet",
            delete=False,
        ) as temporary_artifact:
            temporary_path = Path(temporary_artifact.name)
        frame.write_parquet(
            temporary_path,
            compression=compression,
            statistics=True,
        )
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())
        os.link(temporary_path, artifact_path)
    except FileExistsError as error:
        raise ImmutableArtifactError(
            f"Immutable artifact publish collision: {artifact_path}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ParquetArtifact(
        path=str(artifact_path),
        rows=frame.height,
        bytes=artifact_path.stat().st_size,
        sha256=sha256_file(artifact_path),
    )
