"""Canonicalise official IBBI CIRP public announcements."""

from __future__ import annotations

import json
import sys

from assay.artifacts.parquet import ImmutableArtifactError
from assay.canonicalisation.ibbi import (
    IbbiCanonicalisationConfig,
    IbbiCanonicalisationError,
    IbbiCanonicaliser,
)


def main() -> int:
    try:
        report = IbbiCanonicaliser(IbbiCanonicalisationConfig()).run()
    except (IbbiCanonicalisationError, ImmutableArtifactError) as error:
        print(f"IBBI canonicalisation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "source_snapshot_id": report.source_snapshot_id,
                "solvency_event_rows": report.solvency_event_rows,
                "unique_cin_count": report.unique_cin_count,
                "post_mca_cutoff_unique_cin_count": (
                    report.post_mca_cutoff_unique_cin_count
                ),
                "malformed_tsv_rows": report.malformed_tsv_rows,
                "provenance_complete": report.provenance_complete,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
