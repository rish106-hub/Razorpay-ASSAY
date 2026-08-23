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

## Local three-way comparison

Three feature sets, one fit payload, one set of splits, each holdout scored
once. This is the comparison that matters, because everything except the
feature list is held fixed.

| Evidence | Value |
| --- | --- |
| Observation run (schema 1.2.0) | `5c0e6fd26d1fa757829d8be0f0af6be0c0ced8fd12c8b838433b44ac7dbc2e85` |
| Model-data run (schema 1.1.0) | `993b7c044d4daec5f0deca9259c8dd571d31ba23c387f67f4692f556bc6b0f80` |
| Model-data SHA-256 | `915349ce0f6f218ad595fe45f213e08d66f9d712a89b422568ebbe9f1a21e7df` |
| Fit payload run | `f679b48695e8796b0541f00f176d72d567a395135038d71de2e0e6663b2a12de` |
| Fit payload SHA-256 | `977b956cd74d6f019239774796cdea1e186fbd960c500f281d458454b9dcdea2` |
| Population | 2,574,352 companies, 1,823 outcomes |
| Splits | training 1,386,170/964, validation 297,749/233, geography 593,478/298, temporal 296,955/328 |

The rebuilt population and every split count are identical to the frozen run,
so nothing here moved because the data moved.

Unseen-geography holdout, 593,478 rows and 298 positives:

| Metric | `baseline_with_status` | `baseline_no_status` | `expanded_no_status` |
| --- | ---: | ---: | ---: |
| PR-AUC | 0.50156 | 0.00823 | 0.00502 |
| ROC-AUC | 0.96771 | 0.89229 | 0.84236 |
| Precision at 0.1% capacity | 28.96% | 1.01% | 0.84% |
| Recall at 0.1% capacity | 57.72% | 2.01% | 1.68% |
| Lift at 0.1% capacity | 576.72x | 20.05x | 16.71x |
| Precision at 2% capacity | 1.96% | 0.91% | 0.75% |
| Recall at 2% capacity | 78.19% | 36.24% | 29.87% |
| `Active`-at-cutoff PR-AUC | 0.00408 | 0.00475 | 0.00239 |

Prospective-time holdout, 296,955 rows and 328 positives:

| Metric | `baseline_with_status` | `baseline_no_status` | `expanded_no_status` |
| --- | ---: | ---: | ---: |
| PR-AUC | 0.33668 | 0.01591 | 0.00934 |
| ROC-AUC | 0.93439 | 0.90865 | 0.83910 |
| Precision at 0.1% capacity | 37.71% | 3.37% | 0.67% |
| Recall at 0.1% capacity | 34.15% | 3.05% | 0.61% |
| Lift at 0.1% capacity | 341.41x | 30.49x | 6.13x |
| Precision at 2% capacity | 2.81% | 2.00% | 1.20% |
| Recall at 2% capacity | 50.92% | 36.28% | 21.65% |
| `Active`-at-cutoff PR-AUC | 0.01025 | 0.01216 | 0.00772 |

Model SHA-256: `648765759b3bd7a942399eff14b480d167cc8de157fdba691cf1c628f7563c90`,
`b2f31d033862a74b0b88a56f13a118d8d5c7848df7921cebd9739916dccec17c`, and
`34cbb3bab0a70135d89ef9fc8bda0eaefb5722d646e607e1ecfc94fd58e62228`.

Two things this establishes and one it refutes.

The local pipeline reproduces the published baseline. Refitting
`baseline_with_status` locally on CPU from a locally sampled payload lands at
geography PR-AUC 0.50156 against Colab's 0.49491 and temporal 0.33668 against
0.34280 - inside 0.007 on both, from a different negative sample, a different
sklearn, and a different platform. The earlier `baseline_no_status` refit taken
from Colab's own pinned payload landed at 0.00815 and 0.01602, against 0.00823
and 0.01591 here. Neither number depends on the sampler.

Dropping `company_status` collapses the result, and that is confirmed twice
over on two independently sampled payloads.

**`expanded_no_status` is worse than the honest baseline on every measured
number.** Cluster shape, CIN structure, and ROC-serial adjacency were the
intended answer to the founding research question, and they did not work.

## Why the expanded set failed

Not because the columns are empty. Over the 2,574,352-row population they are
populated, varied, and univariately separating in the expected direction:
positives sit in address clusters with a mean registration span of 1,566 days
against 6,711 for negatives, month entropy 0.257 against 0.123, and a largest-
month share of 0.818 against 0.913. A batch-incorporated cluster does look
different from a service provider's.

Not because the model ignored them either. Grouped XGBoost total gain for
`expanded_no_status` spends 9.01% on the new columns -
`log_registrar_year_cohort_company_count` 2.83%,
`address_cluster_authorised_capital_cv` 2.40%,
`address_cluster_registration_span_days` 1.34%,
`address_cluster_roc_serial_min_gap` 0.62%,
`log_address_cluster_cohort_peer_count` 0.51%, and the four shape remainders
1.31%. The four CIN-disagreement flags contribute nothing measurable, which is
unsurprising: only 2,702 companies disagree on year and 871 on registrar state,
out of 2.57 million.

The likely mechanism is that the cluster-span features re-encode company age,
which the baseline already has, while adding nine dimensions of variance to a
model with 964 positive training rows. Gain moved off `company_age_years`
(10.30% to 8.02%) and `paid_to_authorised_capital_ratio` (4.33% to 2.55%) onto
the new columns without buying generalisation. Validation said so before the
holdouts were touched: weighted validation PR-AUC was 0.01997 for
`baseline_no_status` and 0.01544 for `expanded_no_status`. Selection is
validation-only, so this was a predicted loss, not a surprise.

What this does not license is a second search over the same holdouts for a
configuration that wins. The holdouts have been scored once per candidate and
that budget is spent.

## The remaining target

Positives that were `Active` at the cutoff, and the PR-AUC each model achieves
inside that slice:

| split | `Active` rows | `Active` positives | with status | no status | expanded |
| --- | ---: | ---: | ---: | ---: | ---: |
| geography_test | 352,656 | 101 | 0.00408 | 0.00475 | 0.00239 |
| temporal_test | 180,919 | 215 | 0.01025 | 0.01216 | 0.00772 |

This slice is the actual open problem, and no feature set built so far touches
it. `company_status` bought the headline number and bought nothing at all on
the companies that were still trading at the cutoff - the with-status model is
if anything slightly *worse* here than the honest baseline. The expanded set is
worse again.

Separating a registered-office service provider from a batch-incorporation
cluster is the founding research question of this project. The counts could not
do it. The shape statistics, the CIN structure, and ROC-serial adjacency can
describe the difference but did not predict this outcome from it. The next
honest attempt needs a different input, not a different arrangement of this
one; MCA director/DIN data is the obvious candidate and is deliberately still
out of scope.

Diagnostic accuracy at a 0.5 threshold is above 99.9% on both holdouts. It is
not an acceptance metric. The class imbalance makes a high accuracy number easy
to obtain while missing nearly every outcome.

Unseen-state PR-AUC for the frozen Colab `baseline_with_status` artifact is
0.46489 for Karnataka, 0.60370 for Kerala, 0.44540 for Tamil Nadu, and 0.58696
for Telangana, across 35 to 119 positive outcomes per state. Those figures
inherit the same `company_status` contamination as the headline, so they are
evidence that the contamination generalises geographically, not that a signal
does. Under `baseline_no_status` the same four states fall to 0.00854, 0.00933,
0.00918, and 0.00893.

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

Rebuild the population, the model data, and the sampled fit payload:

```bash
PYTHONPATH=. uv run python -m assay.cli.build_solvency_observations \
  --mca-report data/generated/mca_canonicalisation/schema-1.2.0/snapshot-64181add1929b03d992205d7f9bfc7b0d6d7ab3a8f51e03118d7a7d5044c5623.report.json \
  --ibbi-report data/generated/ibbi_canonicalisation/schema-1.0.0/snapshot-cae5bdcbe537f3c7f358f9f492a034986c64ba1aee0b4c1c4d46aceb4b7c8c85.report.json \
  --outcome-window-end 2026-08-21 \
  --temporal-holdout-start 2026-01-01

PYTHONPATH=. uv run python -m assay.cli.build_solvency_model_data \
  --observation-report data/generated/solvency_observation/schema-1.2.0/run-5c0e6fd26d1fa757829d8be0f0af6be0c0ced8fd12c8b838433b44ac7dbc2e85.report.json

PYTHONPATH=. uv run python -m assay.cli.build_solvency_fit_data \
  --model-data-report data/generated/solvency_model_data/schema-1.1.0/run-993b7c044d4daec5f0deca9259c8dd571d31ba23c387f67f4692f556bc6b0f80.report.json
```

Then fit and score each feature set, once per set:

```bash
FIT=data/curated/solvency_fit_data/schema-1.0.0/run-f679b48695e8796b0541f00f176d72d567a395135038d71de2e0e6663b2a12de/solvency_fit_data.parquet
MODEL_DATA=data/curated/solvency_model_data/schema-1.1.0/run-993b7c044d4daec5f0deca9259c8dd571d31ba23c387f67f4692f556bc6b0f80/solvency_model_data.parquet

for SET in baseline_with_status baseline_no_status expanded_no_status; do
  RUN=data/generated/solvency_model/run-993b7c044d4daec5-$SET
  PYTHONPATH=. uv run python -m assay.cli.train_solvency_model \
    --fit-data "$FIT" \
    --fit-sha256 977b956cd74d6f019239774796cdea1e186fbd960c500f281d458454b9dcdea2 \
    --feature-set "$SET" --device cpu --output-directory "$RUN"
  PYTHONPATH=. uv run python -m assay.cli.evaluate_solvency_model \
    --run-directory "$RUN" --model-data "$MODEL_DATA" \
    --output "$RUN/holdout_metrics.schema-1.2.0.json"
done
```

Refit and score the honest baseline against Colab's own pinned payload, which
is the cross-check that the sampler is not the story:

```bash
PYTHONPATH=. uv run python -m assay.cli.train_solvency_model \
  --fit-data data/generated/solvency_model_data/colab-fit-67e0cc4f177e35f4.parquet \
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
