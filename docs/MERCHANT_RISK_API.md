# Public merchant-risk API

## Claim boundary

This service returns `cirp_public_announcement_score`: the modelled probability
that a public Indian company has a future IBBI CIRP public announcement, given
an as-of MCA company snapshot. It is not a fraud score, a payment-abuse score,
a chargeback score, or a regulatory-debarment score. No Razorpay merchant,
payment, or settlement data is used, joined, or required.

The underlying model is `EVALUATED_NOT_PRODUCTION_READY`. Every response says
so, carries the frozen holdout evidence behind its band, and states its own
limitations. The service is read-only. It never actions a merchant.

| Permitted | Prohibited |
| --- | --- |
| KYB enrichment during merchant onboarding | Automated onboarding rejection |
| Analyst triage ordering of an enhanced-due-diligence queue | Automated settlement hold or account termination |
| Evidence-backed context for a manual underwriting review | Presenting the score as fraud or payment risk |
| Risk-operations monitoring of a public-company portfolio | Any merchant-facing decision without a human reviewer |

A low score is not evidence of solvency. It means the company did not enter any
measured review-capacity queue.

## Frozen serving artifacts

| Artifact | Value |
| --- | --- |
| Merchant index run | `878d2b581a40c9148383800f6c6939d6427fe18141977964d8511ee7be524ec1` |
| Source observation run | `dd0fd88618ec0fb1f6159d7c14ab10faf4fc56bff90af08fbafdb10f35cd7479` |
| Indexed companies | 2,574,352 |
| Index Parquet size | 115,578,294 bytes |
| Companies sharing a strict normalised name | 14,578 |
| Feature cutoff | 3 November 2023 |
| Name normalisation | `india_company_name_v1` |
| Model SHA-256 | `ca884db278f501bcaeec0a432a4d5be5a784c6efaa8c839f7e6ce30109cf5de4` |
| Holdout evidence | `holdout_metrics.schema-1.1.0.json` |

Train and serve share one feature derivation. `solvency_feature_expressions()`
in `assay/solvency/training_data.py` builds the fifteen model features for both
the training dataset and the serving index, so parity is structural rather than
asserted.

## Build the serving index

The index is an immutable, CIN-sorted projection of one frozen solvency
observation run. It adds no feature, recomputes no outcome, and never stores
the training label.

```bash
PYTHONPATH=. uv run python -m assay.cli.build_merchant_index \
  --observation-report data/generated/solvency_observation/schema-1.0.0/run-dd0fd88618ec0fb1f6159d7c14ab10faf4fc56bff90af08fbafdb10f35cd7479.report.json
```

The build refuses to overwrite an existing run. It writes the Parquet under
`data/curated/merchant_risk_index/schema-1.0.0/run-<id>/` and the checksummed
report under `data/generated/merchant_risk_index/schema-1.0.0/`.

## Configure and run the service

| Variable | Required for | Purpose |
| --- | --- | --- |
| `ASSAY_EVIDENCE_REPORT_PATH` | Process start | Evidence-readiness artifact served on `/v1/evidence/readiness` |
| `ASSAY_SOLVENCY_EVALUATION_PATH` | Scoring | Frozen holdout evidence served on `/v1/models/solvency/evaluation`; also the source of the risk bands |
| `ASSAY_MERCHANT_INDEX_REPORT_PATH` | Scoring | Checksum-verified merchant index report |
| `ASSAY_SOLVENCY_MODEL_DIRECTORY` | Scoring | Verified solvency model package directory |
| `ASSAY_ALLOWED_ORIGINS` | Browser callers | Comma-separated CORS origins; defaults to `http://localhost:3000,http://127.0.0.1:3000` |

```bash
export ASSAY_EVIDENCE_REPORT_PATH=data/generated/evidence_readiness/<report>.json
export ASSAY_SOLVENCY_EVALUATION_PATH=data/generated/solvency_model/run-67e0cc4f177e35f4/holdout_metrics.schema-1.1.0.json
export ASSAY_MERCHANT_INDEX_REPORT_PATH=data/generated/merchant_risk_index/schema-1.0.0/run-878d2b581a40c9148383800f6c6939d6427fe18141977964d8511ee7be524ec1.report.json
export ASSAY_SOLVENCY_MODEL_DIRECTORY=data/generated/solvency_model/run-67e0cc4f177e35f4
uv run uvicorn assay.api.app:app --host 127.0.0.1 --port 8000
```

Startup rules, all enforced in `assay/api/app.py`:

- `ASSAY_MERCHANT_INDEX_REPORT_PATH` and `ASSAY_SOLVENCY_MODEL_DIRECTORY` must
  be set together. One without the other fails startup.
- Either of them without `ASSAY_SOLVENCY_EVALUATION_PATH` fails startup. A
  score is never served without the frozen holdout evidence that bands it.
- The evaluation artifact must be schema 1.1.0. The older
  `holdout_metrics.json` records no `score_threshold` and no longer parses.
- The index Parquet checksum is verified before the first request.
- Set none of the three and the service still starts; scoring endpoints then
  return 503 and `/healthz` reports `merchant_assessment_available: false`.

CORS allows `GET` and `POST` with `Accept` and `Content-Type` headers, and
sends no credentials.

## `POST /v1/risk/assess`

The body requires at least one of `cin` or `company_name`. When both are
supplied the CIN is used and the name is ignored. Unknown fields are rejected.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/risk/assess \
  -H 'Content-Type: application/json' \
  -d '{"cin": "L00000CH1983PLC031318"}'
```

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-08-23T04:30:37.581844Z",
  "score_name": "cirp_public_announcement_score",
  "label_boundary": "cirp_public_announcement_outcome_not_fraud",
  "model_status": "EVALUATED_NOT_PRODUCTION_READY",
  "identity": {
    "cin": "L00000CH1983PLC031318",
    "company_name": "SAB INDUSTRIES LIMITED",
    "company_status": "Active",
    "registration_date": "1983-02-16",
    "state_code": "chandigarh",
    "roc_code": "ROC Chandigarh",
    "company_class": "Public",
    "listing_status": "Listed",
    "company_origin": "India",
    "nic_code": "00000",
    "nic_division": "00",
    "authorised_capital_inr": 300000000.0,
    "paid_up_capital_inr": 152100780.0
  },
  "identity_match_method": "exact_cin",
  "cirp_public_announcement_score": 0.005218474194407463,
  "risk_band": "STANDARD",
  "risk_band_evidence": {
    "band": "STANDARD",
    "minimum_score": 0.003305471735075116,
    "review_capacity_fraction": 0.02,
    "holdout_precision": 0.029461279461279462,
    "holdout_recall": 0.5335365853658537,
    "holdout_lift": 26.672787324464156,
    "interpretation": "Top 2% of the prospective-time holdout queue; 2.9% of the companies in this queue had a public IBBI CIRP announcement in the outcome window, covering 53.4% of the announcements observed in that holdout."
  },
  "explanation_factors": [
    {
      "feature": "log_authorised_capital_inr",
      "label": "Authorised capital",
      "value": "INR 300,000,000",
      "direction": "increases_risk",
      "contribution": 2.0693113803863525,
      "evidence": "Authorised capital of INR 300,000,000 raises the modelled CIRP announcement probability for this merchant."
    }
  ],
  "evidence_sources": [
    {
      "source_id": "mca_company_master",
      "description": "MCA company master register published on data.gov.in, canonicalised into an as-of company snapshot.",
      "reference": "64181add1929b03d992205d7f9bfc7b0d6d7ab3a8f51e03118d7a7d5044c5623",
      "as_of": "2023-11-03"
    }
  ],
  "confidence": {
    "level": "low",
    "identity_match_method": "exact_cin",
    "feature_snapshot_date": "2023-11-03",
    "feature_snapshot_age_days": 1024,
    "band_holdout_precision": 0.029461279461279462,
    "reasons": [
      "The MCA feature snapshot is 1024 days old, so later filings, status changes, or capital changes are absent."
    ]
  },
  "permitted_use": ["KYB enrichment during merchant onboarding."],
  "limitations": [
    "The score estimates a future public IBBI CIRP announcement. It is not a fraud, payment-abuse, chargeback, or regulatory-debarment score."
  ]
}
```

The response above is real output for a real public company. Only
`explanation_factors`, `evidence_sources`, `permitted_use`, and `limitations`
are trimmed; each returns more entries. `explanation_factors` returns at most
six. `evidence_sources` always returns four: the MCA snapshot, the IBBI
outcome export, the merchant index run, and the model package. `limitations`
returns five standing limitations followed by the deployment blockers recorded
in the evaluation artifact.

## Identity resolution and failure outcomes

Resolution is ordered. Each form stops the search as soon as it matches any
row: one row is a match, several rows is ambiguity that the caller must
resolve, and only a form matching nothing falls through to the next.

1. Exact CIN.
2. Exact strict normalised name. Uppercase, non-alphanumeric characters
   collapsed to single spaces, trimmed.
3. Strict normalised name with Indian legal-form suffixes removed
   (`PRIVATE`, `PVT`, `LIMITED`, `LTD`, `LLP`, `COMPANY`, `CO`).

A malformed CIN never falls back to a name. A caller who mistypes an
identifier gets a rejection, not a guess at a different company.

| Status | Outcome | Cause | Analyst next step |
| --- | --- | --- | --- |
| 200 | `matched` | Exactly one public company resolved | Read the band, factors, and confidence; a name match still needs identity confirmation |
| 422 | `invalid_identifier` | `cin` is not a 21-character MCA CIN | Resubmit a valid CIN, or submit the registered company name instead |
| 404 | `no_match` | Well-formed identity outside the frozen eligible population | Route to manual KYB review; the company may post-date the cutoff, be an LLPIN or FCRN, or already have been in CIRP before the cutoff |
| 409 | `ambiguous_name` | Several public companies share the normalised name | Resubmit with the exact CIN from the incorporation documents; up to ten candidates are listed with CIN, name, state, status, and registration date |
| 422 | request validation | Body has neither identifier, both empty, or an unknown field | Fix the request body; this returns the FastAPI validation shape, not the lookup problem shape |
| 503 | not configured | Index, model package, or holdout evidence is absent | Configure all three environment variables and restart |

Every non-200 lookup returns `outcome`, `detail`, `guidance`, and `candidates`.
A 409 example, with the candidate list intact:

```json
{
  "outcome": "ambiguous_name",
  "detail": "2 public company records in the frozen eligible population share the normalised name 'GREAT WESTERN INDUSTRIES LIMITED'. This response lists 2 of them, sorted by CIN. Resubmit the request with a CIN; the analyst must confirm identity before relying on any of these records.",
  "guidance": "Several public companies normalise to this name. Resubmit with the exact CIN from the merchant's incorporation documents.",
  "candidates": [
    {
      "cin": "L00219KA1991PLC012579",
      "company_name": "GREAT WESTERN INDUSTRIES LIMITED",
      "state_code": "karnataka",
      "company_status": "Inactive for e-filing",
      "registration_date": "1991-12-04"
    },
    {
      "cin": "U93090TN1997PLC037593",
      "company_name": "GREAT WESTERN INDUSTRIES LIMITED",
      "state_code": "tamil nadu",
      "company_status": "Active",
      "registration_date": "1997-02-25"
    }
  ]
}
```

Lookups scan the single sorted Parquet lazily with the predicate pushed down.
A cold first CIN lookup against the full index measured 21.6 ms and a cold name
lookup 37.8 ms; warm repeats measured 5 to 7 ms and 22 to 24 ms.

## `GET /v1/risk/bands`

Returns the scale and the evidence behind it, with no request body. Bands are
not invented constants. Each boundary is the score a company had to reach to
enter a fixed-size review queue on the frozen prospective-time holdout, and
each band carries the retrieval metrics that queue actually measured.

The source split is `temporal_test`: 296,955 rows, 328 positives, base rate
0.0011045.

| Band | `minimum_score` | Review capacity | Holdout precision | Holdout recall | Lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ELEVATED_REVIEW` | 0.03875885 | 0.1% | 37.71% | 34.15% | 341.41x |
| `WATCH` | 0.00723891 | 0.5% | 8.89% | 40.24% | 80.48x |
| `STANDARD` | 0.00330547 | 2.0% | 2.95% | 53.35% | 26.67x |
| `LOW_SIGNAL` | 0.0 | none | none | none | none |

A band names a queue position measured on public data. It does not state that a
merchant is insolvent, and it is not an action. `LOW_SIGNAL` returns null
precision, recall, and lift because no measured queue covers it.

The scale is rebuilt from the configured evaluation artifact at startup and
rejected unless its thresholds are strictly descending.

## `GET /healthz`

```json
{
  "status": "ok",
  "service": "assay-evidence-api",
  "report_id": "<evidence report id>",
  "schema_version": "<evidence schema version>",
  "solvency_evaluation_available": true,
  "merchant_assessment_available": true,
  "merchant_index_run_id": "878d2b581a40c9148383800f6c6939d6427fe18141977964d8511ee7be524ec1"
}
```

`merchant_index_run_id` is null when assessment is not configured.

## Explanation factors

Contributions are exact tree SHAP values taken from the same iteration range
the selected classifier used, `(0, 266)`. The 163 transformed columns are
summed back to the fifteen raw model features, ranked by absolute contribution,
and the top six non-zero factors are returned. Units are log-odds.

The contributions plus the model bias reconcile to the served probability
through the inverse logit. On the worked example above the reconstruction
differed from the served score by 1.4e-08.

What a factor means: this feature moved this model's output in this direction
for this row. What it does not mean: a cause of insolvency, a fact about the
merchant's conduct, or a reason an analyst may cite as a finding. A factor is a
model explanation, not evidence about the company beyond the MCA field value it
reports.

## Confidence

Confidence is deterministic and has two levels. It can never be `high`.

| Condition | Effect |
| --- | --- |
| Identity resolved from a normalised name rather than an exact CIN | `low` |
| MCA feature snapshot older than 400 days | `low` |
| Score falls below every measured review-capacity queue | `low` |
| None of the above | `moderate` |

Each triggered condition appends its own reason string, so the response always
states why. The feature cutoff is 3 November 2023, which is 1,024 days old as
of 23 August 2026, so every response served from the current index is `low` and
names the snapshot age as the reason.

## Reproducible demo scenarios

`assay.cli.build_demo_scenarios` scores a deterministic sample of the frozen
index and curates one real public company per band plus the three failure
paths. Every scenario is a real MCA company; none is synthetic and none is a
Razorpay account.

```bash
PYTHONPATH=. uv run python -m assay.cli.build_demo_scenarios \
  --index-report data/generated/merchant_risk_index/schema-1.0.0/run-878d2b581a40c9148383800f6c6939d6427fe18141977964d8511ee7be524ec1.report.json \
  --model-directory data/generated/solvency_model/run-67e0cc4f177e35f4 \
  --evaluation-report data/generated/solvency_model/run-67e0cc4f177e35f4/holdout_metrics.schema-1.1.0.json \
  --output data/generated/merchant_risk_index/demo_scenarios.json
```

## Limitations

1. The score estimates a public IBBI CIRP announcement. It is not fraud,
   payment abuse, chargeback risk, or regulatory debarment.
2. The feature snapshot is frozen at 3 November 2023. Later filings, status
   changes, capital changes, and strike-offs are absent from every response.
3. Coverage is limited to the 2,574,352 companies in the frozen eligible
   population. LLPs, foreign companies, companies registered after the cutoff,
   and companies already in CIRP before the cutoff return 404.
4. Name resolution is deterministic normalisation only. There is no fuzzy
   matching, so a genuine company can return 404 on a name that differs from
   its registered form.
5. Bands are measured on one prospective-time holdout of a single public
   population. They are not calibrated to a Razorpay merchant mix, and no
   review budget or false-positive cost has been supplied by risk operations.
6. IBBI row-level reuse and deployment terms still require legal approval.
7. The model is `EVALUATED_NOT_PRODUCTION_READY`. Every output requires a human
   risk-operations reviewer before any merchant-facing decision.

The evidence behind the model itself is in
[`docs/SOLVENCY_MODEL.md`](SOLVENCY_MODEL.md).
