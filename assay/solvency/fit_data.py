"""Build the sampled train-and-validation fit payload for a local solvency fit.

The frozen `baseline_with_status` result was fitted in Colab from a payload that
Colab itself sampled, and that payload carries only the fifteen baseline
columns. No feature set wider than the baseline can be fitted from it, so the
sampling step has to exist in this repository before an expanded contract can be
measured at all.

What is sampled and what is not is the whole point. Every positive outcome is
kept. Only negative controls are sampled, and only inside `training` and
`validation`. The two holdouts are never written here: they are scored later, in
full and unsampled, straight from the model-data artifact. A holdout that passed
through a sampler would not be a holdout.

Selection is deterministic without being random. Negatives are ordered by a
seeded hash of `company_snapshot_id` and the first N are taken, so the same
model-data artifact and the same seed always produce the same payload, and no
call order, thread count, or row order can change it. The hash comes from Polars,
so the report records the Polars version that produced the payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import (
    ParquetArtifact,
    verify_parquet_artifact,
    write_immutable_parquet,
)
from assay.solvency.training_data import (
    SOLVENCY_MODEL_DATA_SCHEMA_VERSION,
    SolvencyTrainingDataReport,
)

SOLVENCY_FIT_DATA_SCHEMA_VERSION = "1.0.0"

#: Splits that may be sampled. Anything absent from this tuple is a holdout and
#: is scored in full from the model-data artifact instead.
SAMPLED_SPLITS = ("training", "validation")


class SolvencyFitDataError(RuntimeError):
    """A solvency fit payload broke its sampling or population contract."""


class SolvencyFitDataConfig(BaseModel):
    """Frozen per-split negative-control sample sizes for one fit payload."""

    model_config = ConfigDict(frozen=True)

    model_data_report_path: Path
    #: Matches the negative-control counts of the frozen Colab payload, so a
    #: local fit is comparable to the published baseline rather than merely
    #: similar to it.
    sampled_training_negative_rows: int = Field(default=48_200, gt=0)
    sampled_validation_negative_rows: int = Field(default=46_600, gt=0)
    sample_hash_seed: int = Field(default=106, ge=0)
    project_root: Path = Path(__file__).resolve().parents[2]
    curated_root: Path = Path("data/curated/solvency_fit_data")
    generated_report_root: Path = Path("data/generated/solvency_fit_data")
    parquet_compression: str = "zstd"

    def sampled_negative_rows(self, split_name: str) -> int:
        """Return the frozen negative-control sample size for one split."""

        if split_name == "training":
            return self.sampled_training_negative_rows
        if split_name == "validation":
            return self.sampled_validation_negative_rows
        raise SolvencyFitDataError(f"{split_name} is a holdout and is never sampled.")


class SolvencyFitSplitEvidence(BaseModel):
    """Population and sample mass for one sampled split."""

    model_config = ConfigDict(frozen=True)

    full_rows: int = Field(gt=0)
    full_negative_rows: int = Field(gt=0)
    positive_rows: int = Field(gt=0)
    sampled_negative_rows: int = Field(gt=0)
    negative_restoration_weight: float = Field(gt=0)


class SolvencyFitDataReport(BaseModel):
    """Checksummed evidence for one locally sampled fit payload."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = SOLVENCY_FIT_DATA_SCHEMA_VERSION
    model_data_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_data_schema_version: str
    model_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_hash_seed: int
    polars_version: str
    split_evidence: dict[str, SolvencyFitSplitEvidence]
    fit_data_artifact: ParquetArtifact


def build_solvency_fit_data(
    model_frame: pl.DataFrame,
    config: SolvencyFitDataConfig,
) -> tuple[pl.DataFrame, dict[str, SolvencyFitSplitEvidence]]:
    """Keep every positive and take a seeded slice of the negative controls."""

    required_columns = {"company_snapshot_id", "dataset_split", "target"}
    if missing_columns := required_columns - set(model_frame.columns):
        raise SolvencyFitDataError(
            f"Model data is missing: {', '.join(sorted(missing_columns))}."
        )
    sampled_frames: list[pl.DataFrame] = []
    split_evidence: dict[str, SolvencyFitSplitEvidence] = {}
    for split_name in SAMPLED_SPLITS:
        split_frame = model_frame.filter(pl.col("dataset_split") == split_name)
        if split_frame.is_empty():
            raise SolvencyFitDataError(f"Model data has no {split_name} rows.")
        positives = split_frame.filter(pl.col("target") == 1)
        negatives = split_frame.filter(pl.col("target") == 0)
        requested_negative_rows = config.sampled_negative_rows(split_name)
        if negatives.height < requested_negative_rows:
            raise SolvencyFitDataError(
                f"{split_name} holds {negatives.height} negative controls, "
                f"fewer than the frozen sample of {requested_negative_rows}."
            )
        sampled_negatives = (
            negatives.with_columns(
                pl.col("company_snapshot_id")
                .hash(config.sample_hash_seed)
                .alias("sample_order")
            )
            .sort("sample_order", "company_snapshot_id")
            .head(requested_negative_rows)
            .drop("sample_order")
        )
        sampled_frames.append(pl.concat([positives, sampled_negatives]))
        split_evidence[split_name] = SolvencyFitSplitEvidence(
            full_rows=split_frame.height,
            full_negative_rows=negatives.height,
            positive_rows=positives.height,
            sampled_negative_rows=sampled_negatives.height,
            negative_restoration_weight=negatives.height / sampled_negatives.height,
        )
    return (
        pl.concat(sampled_frames).sort("dataset_split", "company_snapshot_id"),
        split_evidence,
    )


class SolvencyFitDataBuilder:
    """Verify model data, sample controls, and publish a checksummed payload."""

    def __init__(self, config: SolvencyFitDataConfig) -> None:
        self._config = config

    def run(self) -> SolvencyFitDataReport:
        model_data_report = SolvencyTrainingDataReport.model_validate_json(
            self._resolve(self._config.model_data_report_path).read_text(
                encoding="utf-8"
            )
        )
        if model_data_report.split_decision != "READY_FOR_COLAB":
            raise SolvencyFitDataError(
                "Model data has not passed the split-population gate."
            )
        model_data_path = verify_parquet_artifact(
            model_data_report.model_data_artifact,
            self._config.project_root,
        )
        model_frame = pl.read_parquet(model_data_path)
        fit_frame, split_evidence = build_solvency_fit_data(
            model_frame, self._config
        )
        run_payload = {
            "model_data_run_id": model_data_report.run_id,
            "model_data_sha256": model_data_report.model_data_artifact.sha256,
            "sample_hash_seed": self._config.sample_hash_seed,
            "sampled_training_negative_rows": (
                self._config.sampled_training_negative_rows
            ),
            "sampled_validation_negative_rows": (
                self._config.sampled_validation_negative_rows
            ),
            "schema_version": SOLVENCY_FIT_DATA_SCHEMA_VERSION,
        }
        run_id = hashlib.sha256(
            json.dumps(run_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        run_directory = self._resolve(self._config.curated_root) / (
            f"schema-{SOLVENCY_FIT_DATA_SCHEMA_VERSION}/run-{run_id}"
        )
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{SOLVENCY_FIT_DATA_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if run_directory.exists() or report_path.exists():
            raise SolvencyFitDataError("Solvency fit payload already exists.")
        fit_data_artifact = write_immutable_parquet(
            fit_frame,
            run_directory / "solvency_fit_data.parquet",
            self._config.parquet_compression,
        )
        report = SolvencyFitDataReport(
            run_id=run_id,
            model_data_run_id=model_data_report.run_id,
            model_data_schema_version=SOLVENCY_MODEL_DATA_SCHEMA_VERSION,
            model_data_sha256=model_data_report.model_data_artifact.sha256,
            sample_hash_seed=self._config.sample_hash_seed,
            polars_version=pl.__version__,
            split_evidence=split_evidence,
            fit_data_artifact=fit_data_artifact,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    report.model_dump(mode="json"), indent=2, sort_keys=True
                ).encode()
            )
            report_file.flush()
            os.fsync(report_file.fileno())
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path
