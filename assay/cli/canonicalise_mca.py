"""Canonicalise a completed MCA company-master acquisition."""

from __future__ import annotations

import json
import sys

from assay.artifacts.parquet import ImmutableArtifactError
from assay.canonicalisation.mca import (
    McaCanonicalisationConfig,
    McaCanonicalisationError,
    McaCanonicaliser,
)


def main() -> int:
    try:
        report = McaCanonicaliser(McaCanonicalisationConfig()).run()
    except (McaCanonicalisationError, ImmutableArtifactError) as error:
        print(f"MCA canonicalisation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "source_snapshot_id": report.source_snapshot_id,
                "company_snapshot_rows": report.company_snapshot_rows,
                "address_snapshot_rows": report.address_snapshot_rows,
                "duplicate_legal_entity_identifier_rows": (
                    report.duplicate_legal_entity_identifier_rows
                ),
                "invalid_legal_entity_identifier_rows": (
                    report.invalid_legal_entity_identifier_rows
                ),
                "cin_rows": report.cin_rows,
                "llpin_rows": report.llpin_rows,
                "fcrn_rows": report.fcrn_rows,
                "quality_status": report.quality_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
