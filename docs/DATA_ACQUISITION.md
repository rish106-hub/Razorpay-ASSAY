# Data acquisition and verification plan

## Order of work

1. Acquire and checksum MCA Company Master Data.
2. Acquire official NSE and SEBI `adverse_regulatory_outcome` files.
3. Run schema, null, duplicate, and date-coverage audits.
4. Resolve entities with conservative matching rules and record match evidence.
5. Reproduce candidate-signal tests, starting with the shared-address claim.
6. Optionally generate the Fraud Detection Handbook controlled dataset after
   the core milestone; it does not block milestone one.
7. Acquire GLEIF files if the MCA/NSE join requires identity or ownership
   enrichment.

## Source rules

| Source | Use | Status | Rule |
| --- | --- | --- | --- |
| MCA Company Master Data | Core legal-entity records | Full 3,674,314-record API snapshot acquired and checksum-addressed | Record source URL, download date, extract date and checksum. |
| NSE / SEBI | Official adverse validation | 16,165 official rows acquired, schema-validated, and canonicalised | Retain original source file and source URL. Do not call all listings fraud. |
| GLEIF | Entity identity and ownership enrichment | Not acquired | Check applicable reuse terms before staging. |
| Fraud Detection Handbook | Optional controlled benchmark | Pinned Colab T4 benchmark complete; isolated from entity risk | Label it synthetic in all results and demos. |
| IEEE-CIS | Optional transaction benchmark | Not acquired | Require Kaggle access; do not block milestone one. |

## Required manifest fields

Each acquired asset needs: source URL, publisher, access date, licence or terms,
extract date, SHA-256, file size, intended use, and any restrictions.
Assets repeat or override required source fields rather than relying on implicit
inheritance. Unknown licence or extract dates must be explicit and force
`provenance_status: incomplete`. Run
`uv run python -m scripts.validate_manifests` before downstream evidence work.

## Initial evaluations

- How many legal entities share an address?
- Which shared-address groups have independent adverse evidence?
- Does adding director, ownership, time, and `adverse_regulatory_outcome` evidence materially
  reduce false positives?
- What fraction of MCA records join to NSE/SEBI with strong, reviewable match
  evidence?
- Is each signal `SHIP`, `DO_NOT_SHIP`, or `RE_UNIT`?
