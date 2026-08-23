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

The v2 search compared six fixed XGBoost candidates on validation data only.
The winning `probability_depth_8` candidate was then evaluated once on the
temporal test set. It uses 15 raw fields and 11 deterministic transaction,
customer-velocity, and terminal-risk interaction features.

| Metric | v2 GPU model | v1 GPU model | Amount-only baseline |
| --- | ---: | ---: | ---: |
| Temporal PR-AUC | 0.7593 | 0.7158 | 0.2516 |
| Precision at 1% review capacity | 0.6724 | 0.6445 | 0.2245 |
| Recall at 1% review capacity | 0.7533 | 0.7220 | 0.2515 |
| Lift at 1% review capacity | 75.32x | 72.19x | 25.15x |
| Brier score | 0.00313 | 0.00552 | 0.33070 |

The model's temporal-test accuracy is 99.64%, but that number is not used for
selection because the fraud base rate is 0.89%. PR-AUC, precision, recall, lift,
calibration, and review capacity are the decision metrics.

## Training runtime and artifacts

- Runtime: Google Colab, Tesla T4
- Python 3.13.15, XGBoost 3.4.1, scikit-learn 1.6.1
- Feature contract: `transaction_velocity_v2`, 15 raw and 26 model features
- Candidate count: 6
- Selection policy: validation PR-AUC, then precision at 1% review capacity
- Winner: depth 8, learning rate 0.05, minimum child weight 3
- Best iteration: 513
- Winning GPU fit time: 12.82 seconds
- Total bounded search time: 86.75 seconds
- Model SHA-256:
  `3c57b090d0e2043b9833b074d4b66c046b0a10df4d0f3324385d7828c653c3ee`
- Metrics SHA-256:
  `fad62cd02cb5336bae5b23bc5427e760d3b063dd1428469631de2cf84de415ed`
- Downloaded archive SHA-256:
  `34fd62a7b98869d788a2ec9879ef0681bdca57577112a66ae36c907c9fa315bd`

The notebook is [`notebooks/ASSAY.ipynb`](../notebooks/ASSAY.ipynb). Large
model artifacts stay under ignored `data/generated/` and are not committed.

Verify the downloaded package, checksum chain, XGBoost feature order, and CPU
model loading with:

```bash
uv sync --extra gpu --locked
uv run python -m assay.cli.verify_controlled_benchmark \
  --run-directory data/generated/controlled_benchmark/run-v2-6e3ca5849b46
```

The reusable scorer returns only
`controlled_transaction_fraud_score`. It does not choose an operating
threshold, create a Razorpay action, or enter the MCA/NSE entity-risk API.
