"""Publish the immutable merchant lookup index used by the serving API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.artifacts.parquet import ImmutableArtifactError
from assay.serving.index import (
    MerchantIndexConfig,
    MerchantIndexError,
    MerchantRiskIndexBuilder,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-report", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = MerchantRiskIndexBuilder(
            MerchantIndexConfig(
                observation_report_path=arguments.observation_report
            )
        ).run()
    except (ImmutableArtifactError, MerchantIndexError, ValueError) as error:
        print(f"Merchant index build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
