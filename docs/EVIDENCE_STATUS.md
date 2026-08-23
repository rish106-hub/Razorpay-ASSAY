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
the announcement date, not the CIRP commencement date.

The leakage-safe observation run contains 2,574,352 eligible exact-CIN
companies after excluding 5,777 companies with a prior CIRP announcement. It
has 1,823 observed outcomes, including 298 in the frozen unseen-geography slice
and 408 in the prospective 2026 temporal slice. All three counts clear the
predeclared minimum of 20, so the separate solvency target is `READY_TO_TRAIN`.

The compute-bounded Colab run selected a depth-4 XGBoost model at iteration
265 using weighted validation PR-AUC. That fit is the `baseline_with_status`
feature set and its published numbers are a **contaminated baseline**, not a
discovered signal. The unseen-geography holdout contains 593,478 unsampled
companies and 298 outcomes; its PR-AUC is 0.49491, precision at 0.1% review
capacity is 0.28620, recall is 0.57047, and lift is 569.97x. The prospective
temporal holdout contains 296,955 unsampled companies and 328 outcomes; its
PR-AUC is 0.34280, precision at 0.1% review capacity is 0.37710, recall is
0.34146, and lift is 341.41x.

`company_status` supplies 71.28% of that model's total XGBoost gain, and
`company_status == "Under CIRP"` — the company already inside insolvency
resolution at the cutoff — alone supplies 112 of the 296 temporal review slots,
which is the entire reported temporal precision.

The `baseline_no_status` refit drops that one column, keeps the same pinned fit
artifact, seed, and validation-only selection rule, and was scored once on the
same full unsampled holdouts. Geography PR-AUC falls to 0.00815 with 1.18%
precision and 2.35% recall at 0.1% capacity. Temporal PR-AUC falls to 0.01602
with 3.03% precision and 2.74% recall. ROC-AUC stays at 0.89384 and 0.90867 and
lift stays between 18.71x and 30.87x, so the remaining fourteen public features
carry real but weak signal.

On companies that were `Active` at the cutoff — 101 of 298 geography positives
and 215 of 328 temporal positives — both models score PR-AUC below 0.013. That
slice is the actual open problem.

The `expanded_no_status` feature set was built to attack it: six address-cluster
shape statistics, three ROC-serial adjacency columns, and the CIN-versus-record
disagreement structure, 28 features in total. It is **worse than the honest
baseline on every measured number**. Geography PR-AUC 0.00502 against 0.00823,
temporal 0.00934 against 0.01591, and the `Active` slice 0.00239 and 0.00772
against 0.00475 and 0.01216.

The columns are not empty and the model did not ignore them: they carry 9.01% of
its total gain, and positives do sit in measurably tighter address clusters
(mean registration span 1,566 days against 6,711). Nine extra dimensions on 964
positive training rows did not generalise, and validation-only selection said so
before the holdouts were touched. Describing a batch-incorporation cluster is
not the same as predicting insolvency from it. The next attempt needs a
different input, MCA director/DIN data, not a rearrangement of this one.

All three fits share one payload, one set of splits, and one scoring pass each.
See `docs/SOLVENCY_MODEL.md`.

The model decision is `EVALUATED_NOT_PRODUCTION_READY`. It remains blocked on
IBBI reuse clearance, real merchant/payment telemetry, measured false-positive
cost, and an approved risk-operations review budget.

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
