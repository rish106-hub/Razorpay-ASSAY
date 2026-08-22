"""Canonicalise official NSE adverse-regulatory source files."""

from __future__ import annotations

import json
import sys

from assay.artifacts.parquet import ImmutableArtifactError
from assay.canonicalisation.nse import (
    NseCanonicalisationConfig,
    NseCanonicalisationError,
    NseCanonicaliser,
)


def main() -> int:
    try:
        report = NseCanonicaliser(NseCanonicalisationConfig()).run()
    except (NseCanonicalisationError, ImmutableArtifactError) as error:
        print(f"NSE canonicalisation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "source_snapshot_id": report.source_snapshot_id,
                "adverse_event_rows": report.adverse_event_rows,
                "cin_candidate_rows": report.cin_candidate_rows,
                "pan_candidate_rows": report.pan_candidate_rows,
                "missing_event_date_rows": report.missing_event_date_rows,
                "provenance_complete_assets": report.provenance_complete_assets,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
