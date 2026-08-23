"""Run frozen signal metrics across overall, time, and geography slices."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from assay.artifacts.parquet import verify_parquet_artifact
from assay.evaluation.concentration import (
    GroupConcentrationReport,
    evaluate_group_concentration,
)
from assay.evaluation.metrics import SignalMetricReport, evaluate_signal
from assay.signals.runner import SignalObservationRunReport

EVALUATION_RUN_SCHEMA_VERSION = "1.0.0"


class EvaluationRunError(RuntimeError):
    """An evaluation run cannot satisfy its frozen slice contract."""


class EvaluationRunConfig(BaseModel):
    """Explicit observation artifact and risk-operations evaluation settings."""

    model_config = ConfigDict(frozen=True)

    observation_report_path: Path
    temporal_holdout_start: date
    project_root: Path = Path(__file__).resolve().parents[2]
    generated_report_root: Path = Path("data/generated/evaluation")
    requested_review_capacity: float = Field(default=0.01, gt=0.0, le=1.0)
    review_cost_inr_per_false_positive: float = Field(ge=0.0)
    geography_holdout_modulus: int = Field(default=5, ge=2, le=20)
    geography_holdout_remainder: int = Field(default=0, ge=0)
    null_simulations: int = Field(default=1_000, ge=100, le=100_000)
    random_seed: int = Field(default=106, ge=0)
    minimum_positive_outcomes_per_required_slice: int = Field(default=20, ge=1)


class EvaluationRunReport(BaseModel):
    """All frozen-slice metrics and the gate for downstream model training."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = EVALUATION_RUN_SCHEMA_VERSION
    observation_run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_holdout_start: date
    geography_holdout_states: tuple[str, ...]
    development_states: tuple[str, ...]
    minimum_positive_outcomes_per_required_slice: int = Field(ge=1)
    signal_metrics: tuple[SignalMetricReport, ...]
    concentration_tests: tuple[GroupConcentrationReport, ...]
    required_slice_positive_outcomes: dict[str, int]
    evidence_status: str


def _state_is_holdout(
    state_code: str,
    *,
    modulus: int,
    remainder: int,
) -> bool:
    state_digest = hashlib.sha256(state_code.encode("utf-8")).digest()
    state_bucket = int.from_bytes(state_digest[:8], byteorder="big") % modulus
    return state_bucket == remainder


def build_evaluation_slices(
    observation_frame: pl.DataFrame,
    *,
    temporal_holdout_start: date,
    geography_holdout_modulus: int = 5,
    geography_holdout_remainder: int = 0,
) -> tuple[dict[str, pl.DataFrame], tuple[str, ...], tuple[str, ...]]:
    """Create non-overlapping state partitions and a prospective time risk set."""

    required_columns = {
        "state_code",
        "evaluation_eligible",
        "first_adverse_event_date",
        "observed_adverse_outcome",
        "outcome_window_start",
        "outcome_window_end",
    }
    if missing_columns := required_columns - set(observation_frame.columns):
        raise EvaluationRunError(
            f"Observation artifact is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    if geography_holdout_remainder >= geography_holdout_modulus:
        raise EvaluationRunError(
            "Geography holdout remainder must be smaller than its modulus."
        )
    outcome_window_starts = observation_frame["outcome_window_start"].unique()
    outcome_window_ends = observation_frame["outcome_window_end"].unique()
    if len(outcome_window_starts) != 1 or len(outcome_window_ends) != 1:
        raise EvaluationRunError("Observation rows must share one outcome window.")
    if not (
        outcome_window_starts.item()
        < temporal_holdout_start
        <= outcome_window_ends.item()
    ):
        raise EvaluationRunError(
            "Temporal holdout must fall inside the declared outcome window."
        )

    state_codes = tuple(
        sorted(
            state_code
            for state_code in observation_frame["state_code"]
            .fill_null("")
            .unique()
            .to_list()
            if state_code
        )
    )
    holdout_states = tuple(
        state_code
        for state_code in state_codes
        if _state_is_holdout(
            state_code,
            modulus=geography_holdout_modulus,
            remainder=geography_holdout_remainder,
        )
    )
    development_states = tuple(
        state_code for state_code in state_codes if state_code not in holdout_states
    )
    base_frame = observation_frame.with_columns(
        pl.col("observed_adverse_outcome").alias("slice_target")
    )
    geography_development = base_frame.filter(
        pl.col("state_code").fill_null("").is_in(development_states)
    )
    geography_holdout = base_frame.filter(
        pl.col("state_code").fill_null("").is_in(holdout_states)
    )
    temporal_development = observation_frame.with_columns(
        (
            pl.col("observed_adverse_outcome")
            & (pl.col("first_adverse_event_date") < temporal_holdout_start)
        )
        .fill_null(False)
        .alias("slice_target")
    )
    temporal_holdout = observation_frame.with_columns(
        (
            pl.col("evaluation_eligible")
            & (
                pl.col("first_adverse_event_date").is_null()
                | (pl.col("first_adverse_event_date") >= temporal_holdout_start)
            )
        ).alias("slice_eligible"),
        (
            pl.col("observed_adverse_outcome")
            & (pl.col("first_adverse_event_date") >= temporal_holdout_start)
        )
        .fill_null(False)
        .alias("slice_target"),
    )
    base_frame = base_frame.with_columns(
        pl.col("evaluation_eligible").alias("slice_eligible")
    )
    geography_development = geography_development.with_columns(
        pl.col("evaluation_eligible").alias("slice_eligible")
    )
    geography_holdout = geography_holdout.with_columns(
        pl.col("evaluation_eligible").alias("slice_eligible")
    )
    temporal_development = temporal_development.with_columns(
        pl.col("evaluation_eligible").alias("slice_eligible")
    )
    return (
        {
            "overall": base_frame,
            "geography_development": geography_development,
            "geography_holdout": geography_holdout,
            "temporal_development": temporal_development,
            "temporal_holdout": temporal_holdout,
        },
        holdout_states,
        development_states,
    )


class EvaluationRunner:
    """Evaluate both entity and cohort signals on frozen evidence slices."""

    def __init__(self, config: EvaluationRunConfig) -> None:
        self._config = config

    def run(self) -> EvaluationRunReport:
        observation_report_path = self._resolve(
            self._config.observation_report_path
        )
        if not observation_report_path.is_file():
            raise EvaluationRunError(
                f"Signal observation report is missing: {observation_report_path}."
            )
        observation_report = SignalObservationRunReport.model_validate_json(
            observation_report_path.read_text(encoding="utf-8")
        )
        observation_path = verify_parquet_artifact(
            observation_report.observation_artifact,
            self._config.project_root,
        )
        observation_frame = pl.read_parquet(observation_path)
        slices, holdout_states, development_states = build_evaluation_slices(
            observation_frame,
            temporal_holdout_start=self._config.temporal_holdout_start,
            geography_holdout_modulus=self._config.geography_holdout_modulus,
            geography_holdout_remainder=self._config.geography_holdout_remainder,
        )
        run_payload = json.dumps(
            {
                "observation_run_id": observation_report.run_id,
                "temporal_holdout_start": (
                    self._config.temporal_holdout_start.isoformat()
                ),
                "requested_review_capacity": (
                    self._config.requested_review_capacity
                ),
                "review_cost_inr_per_false_positive": (
                    self._config.review_cost_inr_per_false_positive
                ),
                "geography_holdout_modulus": (
                    self._config.geography_holdout_modulus
                ),
                "geography_holdout_remainder": (
                    self._config.geography_holdout_remainder
                ),
                "null_simulations": self._config.null_simulations,
                "random_seed": self._config.random_seed,
                "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        run_id = hashlib.sha256(run_payload).hexdigest()
        report_path = self._resolve(self._config.generated_report_root) / (
            f"schema-{EVALUATION_RUN_SCHEMA_VERSION}/run-{run_id}.report.json"
        )
        if report_path.exists():
            raise EvaluationRunError("Evaluation run report already exists.")

        signal_definitions = (
            (
                "shared_address",
                "shared_address_company_count",
                "shared_address_signal",
                "address_group_key",
            ),
            (
                "address_registration_month_cohort",
                "address_registration_month_cohort_company_count",
                "address_registration_month_cohort_signal",
                "address_registration_month_cohort_key",
            ),
        )
        signal_metrics: list[SignalMetricReport] = []
        concentration_tests: list[GroupConcentrationReport] = []
        for slice_name, slice_frame in slices.items():
            for signal_name, score_column, signal_column, group_column in (
                signal_definitions
            ):
                signal_metrics.append(
                    evaluate_signal(
                        slice_frame,
                        slice_name=slice_name,
                        signal_name=signal_name,
                        score_column=score_column,
                        signal_column=signal_column,
                        target_column="slice_target",
                        eligibility_column="slice_eligible",
                        requested_review_capacity=(
                            self._config.requested_review_capacity
                        ),
                        review_cost_inr_per_false_positive=(
                            self._config.review_cost_inr_per_false_positive
                        ),
                    )
                )
                concentration_tests.append(
                    evaluate_group_concentration(
                        slice_frame,
                        slice_name=slice_name,
                        group_column=group_column,
                        target_column="slice_target",
                        eligibility_column="slice_eligible",
                        null_simulations=self._config.null_simulations,
                        random_seed=self._config.random_seed,
                    )
                )

        required_slice_positive_outcomes = {
            slice_name: int(
                slice_frame.filter(pl.col("slice_eligible"))["slice_target"].sum()
            )
            for slice_name, slice_frame in slices.items()
            if slice_name in {"overall", "geography_holdout", "temporal_holdout"}
        }
        evidence_status = (
            "evaluation_ready"
            if holdout_states
            and all(
                positive_count
                >= self._config.minimum_positive_outcomes_per_required_slice
                for positive_count in required_slice_positive_outcomes.values()
            )
            else "insufficient_held_out_positive_outcomes"
        )
        report = EvaluationRunReport(
            run_id=run_id,
            observation_run_id=observation_report.run_id,
            temporal_holdout_start=self._config.temporal_holdout_start,
            geography_holdout_states=holdout_states,
            development_states=development_states,
            minimum_positive_outcomes_per_required_slice=(
                self._config.minimum_positive_outcomes_per_required_slice
            ),
            signal_metrics=tuple(signal_metrics),
            concentration_tests=tuple(concentration_tests),
            required_slice_positive_outcomes=required_slice_positive_outcomes,
            evidence_status=evidence_status,
        )
        self._write_report(report_path, report)
        return report

    def _resolve(self, configured_path: Path) -> Path:
        if configured_path.is_absolute():
            return configured_path
        return self._config.project_root / configured_path

    @staticmethod
    def _write_report(report_path: Path, report: EvaluationRunReport) -> None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("xb") as report_file:
            report_file.write(
                json.dumps(
                    report.model_dump(mode="json"),
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
            )
            report_file.flush()
            os.fsync(report_file.fileno())
