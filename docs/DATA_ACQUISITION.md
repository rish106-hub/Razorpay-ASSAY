# Data acquisition and verification plan

## Order of work

1. Acquire and checksum MCA Company Master Data.
2. Acquire official NSE and SEBI adverse-action files.
3. Generate the Fraud Detection Handbook controlled dataset locally.
4. Acquire GLEIF files if the MCA/NSE join requires identity or ownership
   enrichment.
5. Run schema, null, duplicate, and date-coverage audits.
6. Resolve entities with conservative matching rules and record match evidence.
7. Reproduce candidate-signal tests, starting with the shared-address claim.

## Source rules

| Source | Use | Status | Rule |
| --- | --- | --- | --- |
| MCA Company Master Data | Core legal-entity records | API key present; one-record schema verified; full extract not acquired | Record source URL, download date, extract date and checksum. |
| NSE / SEBI | Official adverse validation | Raw official files acquired and schema-validated | Retain original source file and source URL. Do not call all listings fraud. |
| GLEIF | Entity identity and ownership enrichment | Not acquired | Check applicable reuse terms before staging. |
| Fraud Detection Handbook | Controlled benchmark | Not generated | Label it synthetic in all results and demos. |
| IEEE-CIS | Optional transaction benchmark | Not acquired | Require Kaggle access; do not block milestone one. |

## Required manifest fields

Each acquired asset needs: source URL, publisher, access date, licence or terms,
extract date, SHA-256, file size, intended use, and any restrictions.

## Initial evaluations

- How many legal entities share an address?
- Which shared-address groups have independent adverse evidence?
- Does adding director, ownership, time, and adverse-action evidence materially
  reduce false positives?
- What fraction of MCA records join to NSE/SEBI with strong, reviewable match
  evidence?
- Is each signal SHIP, DO NOT SHIP, or RE-UNIT?
