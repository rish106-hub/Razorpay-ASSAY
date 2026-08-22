# Controlled transaction-fraud benchmark

## Claim boundary

This run validates ASSAY's model-training and evaluation mechanics on the
Fraud Detection Handbook simulated transaction dataset. It does not validate
MCA/NSE entity-risk signals, real Razorpay payments, underwriting decisions, or
a production fraud model. The targets and artifacts must remain separate from
the MCA/NSE pipeline.

## Provenance and split

- Upstream: `Fraud-Detection-Handbook/simulated-data-transformed`
- Pinned revision: `6e3ca5849b4681430388056d3f1dcfb41d4e8269`
- Seed: `106`
- Development train: 863,006 rows, 6,761 fraud rows
- Seven-day embargo
- Development validation: 286,826 rows, 2,559 fraud rows
- Seven-day embargo
- Temporal test: 469,445 rows, 4,191 fraud rows
- Raw transaction, customer, terminal, absolute time, label, and fraud-scenario
  identifiers are excluded from model features.

## Result

| Metric | GPU model | Amount-only baseline |
| --- | ---: | ---: |
| Temporal PR-AUC | 0.7158 | 0.2516 |
| Precision at 1% review capacity | 0.6445 | 0.2245 |
| Recall at 1% review capacity | 0.7220 | 0.2515 |
| Lift at 1% review capacity | 72.19x | 25.15x |
| Brier score | 0.00552 | 0.33070 |

The model's temporal-test accuracy is 99.36%, but that number is not used for
selection because the fraud base rate is 0.89%. PR-AUC, precision, recall, lift,
calibration, and review capacity are the decision metrics.

## Training runtime and artifacts

- Runtime: Google Colab, Tesla T4
- Python 3.13.15, XGBoost 3.4.1, scikit-learn 1.6.1
- Best iteration: 320
- GPU fit time: 8.64 seconds
- Model SHA-256:
  `fc79b3263dc192abe3458d0ef0371858d7fc72461708fb5db56a6ff7cb660624`
- Metrics SHA-256:
  `f8df02cfd4350e6661cf2b124c0647f59939d2413975362507001c39f5e63a6b`
- Downloaded archive SHA-256:
  `fa9cd8fab8978ae66a355ae89fc723e58cba822c93b39756302e4fbfd5449309`

The notebook is [`notebooks/ASSAY.ipynb`](../notebooks/ASSAY.ipynb). Large
model artifacts stay under ignored `data/generated/` and are not committed.
