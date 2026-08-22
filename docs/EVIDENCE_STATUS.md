# Current evidence status

Checked on 22 August 2026. This file records the current real-data result, not
the earlier exploratory hypothesis.

## Data and linkage

- MCA acquisition: 3,674,314 records in 368 immutable API pages.
- MCA legal identifiers: 3,123,706 CINs, 545,206 LLPINs, 5,343 FCRNs, and 59
  malformed identifiers quarantined as `UNKNOWN`.
- Duplicate MCA legal identifiers: zero.
- NSE adverse regulatory events: 16,165. These are not generic fraud labels.
- Unique exact-CIN matches: 40.
- Unique name candidates awaiting review: 2,134.
- Ambiguous name events awaiting review: 58.
- Unmatched NSE events: 13,933.

## Separate merchant-solvency outcome

- Official IBBI CIRP export rows: 9,067.
- Canonical dated valid-CIN announcements: 8,523 across 7,979 companies.
- Rows quarantined: 535 invalid CINs and 9 malformed TSV rows.
- Unique companies announced after the MCA cutoff: 1,963.
- Exact post-cutoff CIN matches to the MCA snapshot: 1,931; unmatched: 32.
- Year coverage after the cutoff: 140 in 2023, 782 in 2024, 615 in 2025,
  and 426 in 2026.

This target is `cirp_public_announcement_outcome`. It measures a public
merchant-solvency event, not fraud or regulatory debarment. The event date is
the announcement date, not the CIRP commencement date. It has enough positives
to build and evaluate a separate held-out model, but no metric is claimed until
the observation, leakage, geography, and temporal-split gates run.

## Leakage-safe signal population

- Feature cutoff: 3 November 2023.
- Outcome window: 4 November 2023 through 22 August 2026.
- Evaluation-eligible companies: 2,920,186.
- Exact-CIN positive outcomes after the cutoff: 8.
- Shared-address signal rows: 466,414, or 15.97% of the eligible population.
- Address-plus-registration-month cohort rows: 174,025, or 5.96% of the
  eligible population.

These are signal fire rates, not precision or fraud rates. Precision, recall,
PR-AUC, lift, confidence intervals, and held-out stability cannot be reported
until the reviewed outcome set passes the frozen evidence threshold.

## Model decision

The NSE regulatory entity-risk decision is `DO_NOT_TRAIN` for now. Eight positive outcomes are
below the frozen minimum of 20 in each required slice even before the data is
partitioned into unseen-geography and temporal holdouts. The review workbook
contains 557 candidate rows across 498 sampled events. A real reviewer must
complete it. Model output cannot be used as fake human ground truth.

The generated frontend contract has report ID
`910fab616718664ed7f8927b6a1a7109b31591cabe1ec2dc63acf088d8be4207`.
The read-only API serves this contract and cannot change the training decision.

## Controlled benchmark

The isolated Fraud Detection Handbook benchmark trained successfully on a
Colab T4. A six-candidate validation-only search selected a 26-feature model.
Its temporal test PR-AUC is 0.75930, precision at 1% review capacity is 0.67242,
recall is 0.75328, lift is 75.32x, and Brier score is 0.00313. The exported
XGBoost 3.4.1 model, raw-to-model feature contract, and checksums were verified
and exercised locally on CPU.

This benchmark validates transaction-fraud training mechanics only. It is not
a Razorpay production model and cannot be transferred to MCA/NSE entity risk.

## Remaining external inputs

1. A completed human linkage-review CSV with reviewer identity and timezone
   timestamps.
2. Measured analyst handling cost per false-positive review.
3. Approved false-positive review-spend budget per evaluation slice.
4. Written clearance if NSE-derived artifacts or row-level data will be
   redistributed. The source extract date remains unknown, so provenance stays
   incomplete.
5. Written reuse clearance before redistributing IBBI raw or row-level data.
