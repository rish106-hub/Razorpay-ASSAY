"""Score frozen, unsampled solvency holdouts and persist the evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from assay.solvency.artifact import load_solvency_model_package
from assay.solvency.evaluation import evaluate_solvency_holdouts


def main() -> None:
    """Parse paths, evaluate once, and write the holdout report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--model-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    package = load_solvency_model_package(args.run_directory)
    report = evaluate_solvency_holdouts(package, args.model_data)
    if args.output.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
