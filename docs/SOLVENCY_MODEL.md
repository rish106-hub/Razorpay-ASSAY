# Merchant-solvency model evidence

## Claim boundary

This model estimates the probability of a future public IBBI CIRP announcement
from an as-of MCA company snapshot. Its output is named
`cirp_public_announcement_score`. It is not a fraud score, payment-risk score,
regulatory-debarment score, or adverse action.

The model is a public-data baseline. It is `EVALUATED_NOT_PRODUCTION_READY`.
Razorpay merchant/payment telemetry, an approved review budget, measured false
positive cost, and legal clearance for IBBI-derived deployment artifacts are
still missing.

## Frozen evidence chain

| Evidence | Value |
| --- | --- |
| Feature cutoff | 3 November 2023 |
| Outcome | First public CIRP announcement after the cutoff |
| Eligible exact-CIN companies | 2,574,352 |
| Observed outcomes | 1,823 |
| Model-data run | `67e0cc4f177e35f42c2eff017555ec5ffc25be7d60e192e6f3ec811a0c2ede2f` |
| Model SHA-256 | `ca884db278f501bcaeec0a432a4d5be5a784c6efaa8c839f7e6ce30109cf5de4` |
| Colab runtime | T4, XGBoost 3.4.1, seed 106 |
| Selected candidate | depth 4, minimum child weight 5, best iteration 265 |

The fit artifact retained every positive in training and validation. It sampled
negative controls at 50:1 and 200:1 respectively. Training and validation
weights restore the unsampled negative population mass. Geography and temporal
tests were never uploaded to Colab and were scored once locally in full.

## Feature sets

Every fit names a frozen feature contract. The name is recorded in
`portable_preprocessor.json`, `selection_metrics.json`, and the evaluation
artifact. A missing key resolves to `baseline_with_status`, so the original
Colab artifact keeps loading unchanged.

| Feature set | Features | `company_status` |
| --- | ---: | --- |
| `baseline_with_status` | 15 | included |
| `baseline_no_status` | 14 | dropped |

`company_status` is declared outcome-adjacent. It is a pre-cutoff snapshot
field, so it is temporally legal, but one of its levels is the outcome observed
early. Measured on the frozen `baseline_with_status` artifact, grouped XGBoost
total gain by source feature:

| source feature | share of total gain |
| --- | ---: |
| `company_status` (all levels) | 71.28% |
| ... of which `company_status_Under CIRP` | 36.82% |
| `log_authorised_capital_inr` | 17.97% |
| `log_paid_up_capital_inr` | 4.42% |
| `company_age_years` | 3.28% |
| `paid_to_authorised_capital_ratio` | 0.76% |
| `log_shared_address_company_count` | 0.25% |
| `log_address_registration_month_company_count` | 0.05% |

`company_status == "Under CIRP"` means the company is already inside the
insolvency resolution process at the cutoff. In the holdouts:

| split | rows | positives | positive rate |
| --- | ---: | ---: | ---: |
| geography_test | 144 | 131 | 90.97% |
| temporal_test | 120 | 112 | 93.33% |

The reported `baseline_with_status` temporal precision at 0.1% review capacity
is 0.37710. Those 120 rows alone supply 112 of the 296 review slots, which is
0.37838. The headline number is that one status value and nothing else.

Evaluation schema 1.2.0 therefore always emits `status_slices`, whether or not
`company_status` is a model feature, so this contamination stays visible in
every future report.

## Contaminated baseline holdout results

| Metric | Unseen geography | Prospective time |
| --- | ---: | ---: |
| Rows | 593,478 | 296,955 |
| Positives | 298 | 328 |
| Base rate | 0.0502% | 0.1105% |
| PR-AUC | 0.49491 | 0.34280 |
| ROC-AUC | 0.96916 | 0.93782 |
| Brier score | 0.000294 | 0.000759 |
| Precision at 0.1% review capacity | 28.62% | 37.71% |
| Recall at 0.1% review capacity | 57.05% | 34.15% |
| Lift at 0.1% review capacity | 569.97x | 341.41x |
| Precision at 0.5% review capacity | 6.50% | 8.89% |
| Recall at 0.5% review capacity | 64.77% | 40.24% |
| Lift at 0.5% review capacity | 129.50x | 80.48x |
| Precision at 2% review capacity | 1.92% | 2.95% |
| Recall at 2% review capacity | 76.51% | 53.35% |
| Lift at 2% review capacity | 38.25x | 26.67x |

The 2% capacity was added after the 0.1% and 0.5% queues so that a serving
triage scale could be derived from measured queues rather than from invented
constants. Regenerating the artifact moved no previously published number:
every metric already recorded at 0.1% and 0.5%, and both PR-AUC values, are
byte-identical.

These are `baseline_with_status` numbers. They are the contaminated baseline and
must not be quoted as a discovered signal.

## Honest baseline holdout results

`baseline_no_status`, refit locally on CPU from the same checksum-pinned fit
artifact `45179f4ab560b93878cff5ea8a5eb525a7c15a76f344e66d2e7eacfce3b4f18a`,
same seed 106, same validation-only selection rule. The selected candidate is
again `depth_4_regularised`, at iteration 205. Model SHA-256 is
`f9d15023c025a205858a37acfae3fd8787bee3bea1d4d99c9feab2514e2c903b`. The full
unsampled holdouts were scored once.

| Metric | Unseen geography | Prospective time |
| --- | ---: | ---: |
| Rows | 593,478 | 296,955 |
| Positives | 298 | 328 |
| PR-AUC | 0.00815 | 0.01602 |
| ROC-AUC | 0.89384 | 0.90867 |
| Brier score | 0.000500 | 0.001095 |
| Precision at 0.1% review capacity | 1.18% | 3.03% |
| Recall at 0.1% review capacity | 2.35% | 2.74% |
| Lift at 0.1% review capacity | 23.47x | 27.43x |
| Precision at 0.5% review capacity | 1.55% | 2.49% |
| Recall at 0.5% review capacity | 15.44% | 11.28% |
| Lift at 0.5% review capacity | 30.87x | 22.63x |
| Precision at 2% review capacity | 0.94% | 2.07% |
| Recall at 2% review capacity | 37.58% | 37.50% |
| Lift at 2% review capacity | 18.79x | 18.71x |

Dropping one column moves geography PR-AUC from 0.49491 to 0.00815 and temporal
PR-AUC from 0.34280 to 0.01602. That is a 61x and a 21x collapse. The published
`baseline_with_status` result was almost entirely the label observed early.

The honest baseline still beats the base rate by 19x to 31x at every measured
review capacity, and ROC-AUC stays near 0.90, so the remaining fourteen
features are not noise. They are also nowhere near a deployable queue: a 1% to
3% precision review queue is not an operational product.

## The remaining target

Positives that were `Active` at the cutoff, and the PR-AUC each model achieves
inside that slice:

| split | `Active` rows | `Active` positives | with-status PR-AUC | no-status PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| geography_test | 352,656 | 101 | 0.00455 | 0.00457 |
| temporal_test | 180,919 | 215 | 0.01074 | 0.01224 |

Both models are equally weak here, which is the point: `company_status` bought
the headline number and bought nothing at all on the companies that were still
trading at the cutoff. Both address-cluster count features carry 0.30% of
combined gain. Counts cannot separate a registered-office service provider from
a batch-incorporation cluster, and that separation is the founding research
question of this project.

Diagnostic accuracy at a 0.5 threshold is above 99.9% on both holdouts. It is
not an acceptance metric. The class imbalance makes a high accuracy number easy
to obtain while missing nearly every outcome.

Unseen-state PR-AUC is 0.46489 for Karnataka, 0.60370 for Kerala, 0.44540 for
Tamil Nadu, and 0.58696 for Telangana. These slices contain between 35 and 119
positive outcomes. They are evidence of geographic generalisation, not a
deployment guarantee.

## Artifact portability

The Colab runtime used scikit-learn 1.6.1 while the repository uses a newer
version. The scorer therefore never loads the exported sklearn pickle. Colab
also exports `portable_preprocessor.json`, which freezes numeric medians,
categorical imputers, category order, infrequent-category buckets, and all 163
transformed feature names. Package loading verifies its checksum and the model
feature width before scoring.

The CPU scorer also passes `iteration_range=(0, 266)` to XGBoost. Loading a raw
early-stopped booster without this range would incorrectly score with trees
that the selected classifier did not use.

## Reproduce locally

Install the optional XGBoost dependency:

```bash
uv sync --extra gpu --locked
```

Verify the downloaded Colab package:

```bash
PYTHONPATH=. uv run python -m assay.cli.verify_solvency_model \
  --run-directory data/generated/solvency_model/run-67e0cc4f177e35f4 \
  --load-model
```

Evaluate the frozen full holdouts:

```bash
PYTHONPATH=. uv run python -m assay.cli.evaluate_solvency_model \
  --run-directory data/generated/solvency_model/run-67e0cc4f177e35f4 \
  --model-data data/curated/solvency_model_data/schema-1.0.0/run-67e0cc4f177e35f42c2eff017555ec5ffc25be7d60e192e6f3ec811a0c2ede2f/solvency_model_data.parquet \
  --output data/generated/solvency_model/run-67e0cc4f177e35f4/holdout_metrics.schema-1.2.0.json
```

Refit and score the honest baseline:

```bash
PYTHONPATH=. uv run python -m assay.cli.train_solvency_model \
  --fit-artifact data/generated/solvency_model_data/colab-fit-67e0cc4f177e35f4.parquet \
  --fit-sha256 45179f4ab560b93878cff5ea8a5eb525a7c15a76f344e66d2e7eacfce3b4f18a \
  --feature-set baseline_no_status \
  --device cpu \
  --output-directory data/generated/solvency_model/run-67e0cc4f177e35f4-baseline-no-status

PYTHONPATH=. uv run python -m assay.cli.evaluate_solvency_model \
  --run-directory data/generated/solvency_model/run-67e0cc4f177e35f4-baseline-no-status \
  --model-data data/curated/solvency_model_data/schema-1.0.0/run-67e0cc4f177e35f42c2eff017555ec5ffc25be7d60e192e6f3ec811a0c2ede2f/solvency_model_data.parquet \
  --output data/generated/solvency_model/run-67e0cc4f177e35f4-baseline-no-status/holdout_metrics.schema-1.2.0.json
```

The command refuses to overwrite an existing output, so every earlier artifact
was left in place and each regenerated evidence file was written beside it.
Schema 1.1.0 evaluates three review capacities instead of two and records
`score_threshold`, the minimum score that entered each queue. Schema 1.2.0 adds
`feature_set_name` and `status_slices`. Only a 1.1.0-or-later artifact loads
under the current contract, and only it can band a served score. Serving still
consumes the `baseline_with_status` artifact, because that is the model whose
bands were measured; the honest baseline is evidence, not a serving swap. The serving surface that consumes it is documented in
[`docs/MERCHANT_RISK_API.md`](MERCHANT_RISK_API.md).

Generated model bytes and row-level data remain ignored. Only code, contracts,
tests, and evidence summaries belong in Git.
