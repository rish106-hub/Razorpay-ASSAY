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

## Untouched holdout results

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
  --output data/generated/solvency_model/run-67e0cc4f177e35f4/holdout_metrics.json
```

Generated model bytes and row-level data remain ignored. Only code, contracts,
tests, and evidence summaries belong in Git.
