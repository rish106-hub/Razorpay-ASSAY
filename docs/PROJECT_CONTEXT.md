# Transferred project context

## Objective

Build a Razorpay AI Builder Hackathon submission that verifies merchant-risk
signals at cohort and network level, rather than treating each merchant as an
isolated record.

The product must avoid flooding analysts with false positives from simplistic
signals such as a shared address. It should turn validated risk evidence into
graded, expiring, reviewable actions.

## Non-negotiables

- Scoring and decisioning are deterministic and auditable.
- An LLM may explain already-computed evidence to an analyst. It must not
  calculate the risk score or make the decision.
- A shared address is not itself an adverse signal.
- A company being struck off is not proof of fraud.
- Every key claim must be reproducible from acquired data.

## Current evidence hypothesis

Earlier exploration estimated that shared addresses can touch about 18.24% of
companies, while stricter suspicious cohorts may account for about 0.54%.
Those figures are hypotheses until this repository reproduces them against the
acquired data.

## Dataset decisions

| Role | Dataset / source | Decision |
| --- | --- | --- |
| Core company records | MCA Company Master Data | Required |
| Adverse validation | Official NSE / SEBI debarment and adverse-action data | Required |
| Identity and ownership enrichment | GLEIF | Conditional |
| Controlled benchmark | Fraud Detection Handbook simulator | Optional; does not block milestone one |
| Transaction fraud benchmark | IEEE-CIS | Optional, requires Kaggle |
| Credit-card fraud data | ULB dataset | Excluded |

MCA catalogue metadata may be recent, but the underlying extract previously
observed ended on 3 November 2023. Treat data recency as a measured limitation.

## Competitive position

Graph and rule-based risk detection already exist in offerings associated with
Sardine, Unit21, Coris, and Mastercard. The differentiator is a
falsification-first method:

- **SHIP**: evidence supports using the signal.
- **DO NOT SHIP**: evidence shows the signal is too broad or unreliable.
- **RE-UNIT**: the signal becomes useful only after changing its unit of
  analysis, for example from company to cohort, ownership group, or network.

## Gate before architecture

Do not settle the product architecture or build the main UI until we have:

1. Acquired the raw MCA and official NSE/SEBI files.
2. Audited their schema, completeness, and licensing terms.
3. Measured linkage coverage.
4. Reproduced the shared-address inversion and cohort result.
5. Logged the outcome for each candidate signal as SHIP, DO NOT SHIP, or
   RE-UNIT.

## Credentials

No AI key is needed for ingestion, entity resolution, signal testing, scoring,
or evaluation. Keep any OpenAI, Gemini, or DeepSeek key for a later optional
analyst-explanation layer. Never add a key to Git or paste it into project
files.
