# Razorpay AI Risk Manager: Research and Data Handoff

**Research date:** 22 August 2026  
**Buildathon application deadline:** 5 September 2026  
**Status:** Dataset selection and empirical thesis validation, before architecture

## 1. Executive decision

Do not build a generic fraud detector, generic merchant graph, or LLM risk-scoring wrapper.

The strongest current direction is a **merchant-risk signal verifier** demonstrated on Indian corporate data. Its distinctive function is not merely scoring a signal. It tests whether a signal that appears flat or inverted at the company level becomes useful at the correct unit of analysis, such as a cohort, address cluster, director cluster, or merchant network.

The key operation is **RE-UNIT**:

1. Evaluate a proposed risk signal at the individual-entity level.
2. Measure its precision, recall, lift, calibration, stability, false-positive cost, and coverage.
3. If the aggregate result is weak, test whether outcomes are overdispersed across size-matched groups relative to a binomial null.
4. If meaningful group-level concentration exists, change the unit of analysis and retest.
5. Return `SHIP`, `DO NOT SHIP`, or `RE-UNIT`, with evidence.

This is narrower and more defensible than claiming that merchant rings, graph analysis, or rule backtesting are novel. Existing vendors already market those capabilities.

## 2. What the prior conversation actually established

The earlier discussion went through several materially different ideas. These are the decisions that must survive into implementation.

### 2.1 Initial thesis

- Track: Razorpay AI Risk Manager.
- Constraints: public data only, a short build window, and a working system with held-out metrics.
- Initial proposal: merchant-side ring detector plus a graded decision layer.
- Correct AI boundary: no LLM in the scoring path. An LLM may explain already-computed evidence, but it must not create the fraud score.

### 2.2 First empirical correction

The initial shared-address heuristic did not work at the individual-company level.

Earlier exploratory figures in the shared chat were:

- 18.24% of companies shared a registered address.
- Only 0.54% were ring candidates under the proposed rule.
- A naive shared-address policy would overflag by roughly 33.9 times.
- Shared-address lift was about 0.76 times, worse than random.
- Tightly timed incorporation cohorts were stronger than raw address reuse.

These are historical experiment results, not current facts. They must be reproduced from the current raw extract before being used in a pitch.

### 2.3 Metric correction

An early model result appeared strong because age-related information dominated it.

- All features, including age: AUC about 0.8788 and top-1% precision about 0.878.
- Leakage-reduced features: AUC about 0.7969 and top-1% precision about 0.390.
- Leakage-reduced, unseen geography: AUC about 0.7672 and top-1% precision about 0.290.
- Recall at the narrow operating point was only about 1.53%.

The lesson is not that the detector is excellent. The lesson is that time, geography, and label construction can create deceptively strong metrics. The product should expose such failures.

### 2.4 Pivot to ASSAY

The earlier discussion renamed the concept **ASSAY**, a signal audit harness. It proposed nine gates and three verdicts:

- `SHIP`
- `DO NOT SHIP`
- `RE-UNIT`

The most distinctive gate was the group-concentration test: compare observed outcome dispersion across size-matched groups against a binomial null before killing a signal whose aggregate lift is flat or inverted.

Five earlier candidate signals were considered. Only a cohort-style signal appeared promising. Shared address moved to `RE-UNIT`; round capital, free-email, and missing-activity-style signals were candidates for rejection. All of these outcomes require reproduction.

### 2.5 Later product framing

The concept was later reframed as a **Merchant Risk Context Engine**, with a **Merchant Ring Verifier** module. This framing is useful only if the system remains a verifier of external entity context rather than another transaction fraud engine.

Boundary with Razorpay's internal systems:

- Razorpay's transaction systems model payment behavior across merchants.
- This project should model external corporate, regulatory, ownership, and entity-linkage context.
- The two can meet at a risk-policy interface, but public data cannot reproduce Razorpay's private payment network.

## 3. Official buildathon bar

The [official Buildathon page](https://razorpay.com/buildathon/) asks the AI Risk Manager track to build a working detector, verifier, or auto-responder for one class of loss, with measured precision and recall on a held-out test set. It explicitly asks for honest metrics, including false-positive cost, and limits the track to defensive work.

Judging emphasizes:

1. Problem taste.
2. Build quality.
3. AI judgment, including where not to use AI.
4. Failure recovery.

The application form closes on **5 September 2026**. It asks for a public GitHub repository, a five-minute pitch video, and an explanation of what broke and how it was recovered. Therefore the failed shared-address thesis is potentially useful if reproduced honestly and turned into the verifier's core demonstration.

## 4. Competitive landscape

### 4.1 Razorpay's internal baseline

Razorpay already has substantial fraud and merchant-risk infrastructure:

- [Vulcan](https://razorpay.com/blog/one-foundation-model-built-for-indias-payments-ecosystem/) is presented as a payments foundation model operating over Razorpay's network.
- [Bumblebee](https://engineering.razorpay.com/meet-bumblebee-the-multi-agent-ai-architecture-that-changed-fraud-detection-at-razorpay-c2b6d5704f51) describes a multi-agent fraud architecture.
- Razorpay has described [duplicate and fraudulent merchant detection](https://engineering.razorpay.com/how-does-razorpay-capital-detect-duplicate-or-fraud-merchants-5ddc67e1535a).
- It has also described its [merchant risk-review workflow](https://engineering.razorpay.com/our-obsession-with-merchant-experience-breaking-the-risk-review-black-box-7fa38d699ef1).
- Its [real-time anomaly-detection system](https://aws.amazon.com/blogs/big-data/how-razorpay-built-real-time-anomaly-detection-with-amazon-msk/) already covers transaction anomaly alerts.

This makes a generic anomaly detector, risk dashboard, case queue, or multi-agent wrapper a weak submission.

### 4.2 External market baseline

The earlier statement that nobody ships merchant-ring or relationship-aware risk products is false.

- [Sardine Merchant Risk](https://www.sardine.ai/merchant-risk) treats a merchant as an evolving entity and markets hidden-relationship analysis.
- [Sardine's rules engine](https://www.sardine.ai/rules-engine) includes backtesting, shadow mode, A/B testing, precision, recall, and rule fire rate.
- [Unit21 transaction monitoring](https://www.unit21.ai/products/aml-transaction-monitoring) markets linked-entity analysis, historical backtesting, and shadow mode.
- [Coris](https://www.coris.ai/) combines merchant underwriting, monitoring, transaction signals, and merchant intelligence.
- [LegitScript](https://www.legitscript.com/solutions/merchant-risk-solutions/), [G2 Risk Solutions](https://g2risksolutions.com/), and [Mastercard Merchant Monitoring](https://www.mastercard.com/global/en/business/cybersecurity-fraud-prevention/risk-decisioning/merchant-monitoring-with-ai.html) also occupy merchant-risk territory.

### 4.3 Defensible wedge

The defensible wedge is not graph detection by itself, and it is not backtesting by itself. It is:

> A falsification-first verifier that detects when a candidate merchant-risk signal is being evaluated at the wrong unit of analysis, then proves or rejects the alternative unit with held-out evidence and explicit false-positive cost.

The Indian corporate-data case study makes this concrete, but the verifier should be designed as a general evaluation workflow rather than a claim that public MCA data can identify fraud.

No public submission leaderboard was found as of 22 August 2026, and applicant counts are not public. Current competition density cannot be stated as fact.

## 5. Dataset decision table

| Dataset | Role | Current status | Strength | Main limitation | Decision |
|---|---|---|---|---|---|
| MCA Company Master Data | Core entity and cohort table | Official source verified; Delhi preview inspected; bulk file not yet acquired | CIN, status, registration date, capital, NIC, address, geography | Underlying data only through 3 Nov 2023; state-by-state delivery; company status is not fraud truth | **Core, conditional on successful bulk acquisition and schema audit** |
| Official NSE/SEBI debarred entities | Adverse-outcome validation | Official download page verified; file not yet locally acquired | Direct regulatory provenance | Debarment is not fraud; name/CIN coverage must be measured | **Primary adverse label source** |
| OpenSanctions NSE debarred mirror | Convenient normalized adverse data | Source and licence verified | Daily normalization and searchable entities | CC BY-NC 4.0; unsuitable for a commercial product without a licence | **Student-prototype convenience only** |
| GLEIF Level 1 LEI | Corporate identity enrichment | Official file endpoints and CC0 terms verified; local file not acquired | Global identifiers; some deterministic registration-authority joins | India coverage is mandate-biased toward larger borrowers | **Conditional enrichment** |
| GLEIF Level 2 relationships and exceptions | Parent-child graph enrichment | Official endpoints verified; local file not acquired | Reported ownership relationships and exceptions | Only useful where relevant entities have LEIs; usable Indian edges unknown | **Include only if coverage gate passes** |
| Fraud Detection Handbook simulator | Controlled temporal benchmark | Official generator and licence verified | Reproducible class imbalance, temporal splits, controlled scenarios | Customer/terminal fraud is not merchant corporate risk | **Optional verifier benchmark, kept separate from MCA story** |
| IEEE-CIS Fraud Detection | Secondary transaction benchmark | Kaggle source verified; not acquired | Large, familiar transaction/identity benchmark | Kaggle access and rules; unrelated to corporate-risk unit selection | **Secondary or cut** |
| ULB credit-card fraud | Generic benchmark | Source known; not needed | Small and common | PCA features are opaque; only 492 fraud cases; weak narrative fit | **Cut** |
| RBI/NPCl material | Regulatory and product context | Official regulatory source verified | Explains merchant monitoring obligation | Not a model-training dataset | **Context only** |
| 2017 suspected-shell-company lists | Exploratory adverse reference | Historical official material exists | Potential corporate adverse examples | Old, small, legally sensitive, not fraud ground truth | **Exploratory only, never primary labels** |

## 6. Dataset facts and acquisition notes

### 6.1 MCA Company Master Data

Official sources:

- [Company Master Data catalogue](https://www.data.gov.in/catalog/company-master-data)
- [ROC-wise resource](https://www.data.gov.in/resource/registrars-companies-roc-wise-company-master-data)

Verified observations:

- The catalogue page was updated on 22 July 2026.
- Its resource note says the underlying company data is only current through 3 November 2023.
- Distribution is currently state/ROC based rather than a single verified national archive.
- The Delhi preview reported 507,637 records and 16 fields.
- Fields include CIN, company name, ROC, category, subcategory, class, authorized and paid-up capital, registration date, registered-office address, listing status, company status, state, Indian/foreign flag, NIC code, and industrial classification.
- Previewed statuses included Active, Strike Off, and Converted to LLP.
- A browser-initiated Delhi download timed out. No local file should be treated as acquired.

Critical interpretation:

- `Strike Off` is not a fraud or shell-company label.
- Status strongly depends on age, geography, compliance, and administrative processes.
- Earlier record counts and rates from the shared conversation may have come from a stale or incomplete extract. Recompute every quoted statistic.

### 6.2 NSE and OpenSanctions

Official source:

- [NSE SEBI-debarred entities page](https://www.nseindia.com/static/regulations/member-sebi-debarred-entities)

Convenience source:

- [OpenSanctions NSE-debarred dataset](https://www.opensanctions.org/datasets/in_nse_debarred/)
- [OpenSanctions noncommercial exemption](https://www.opensanctions.org/docs/commercial/exemption/)

Verified observations as of the research date:

- The official NSE page exposes spreadsheets for SEBI-debarred entities and actions by other competent authorities.
- The OpenSanctions mirror reported 31,631 total entities, 15,468 searchable entities, and 15,349 targets, with daily updates.
- OpenSanctions offers simplified CSV and JSON, but the open licence is noncommercial.

Join plan:

1. Use CIN where present.
2. Normalize company names and legal suffixes.
3. Use conservative probabilistic matching only for unresolved records.
4. Manually inspect a stratified match sample.
5. Publish exact-match, probable-match, ambiguous, and unmatched rates.

Do not call these fraud labels. Name the target `adverse_regulatory_outcome` or a source-specific equivalent.

### 6.3 GLEIF

Official sources:

- [GLEIF Golden Copy downloads](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy)
- [GLEIF Level 2 ownership data](https://www.gleif.org/en/lei-data/access-and-use-lei-data/level-2-data-who-owns-whom)

Verified observations:

- GLEIF data is available under CC0 and is commercially reusable.
- Current files include Level 1 identity data, Level 2 relationship data, and reporting-exception data.
- India has substantial LEI growth, but mandates make the population nonrepresentative of the full MCA universe.
- Download actions were identified but did not produce local files in the current browser workspace.

Coverage gate:

- Measure how many MCA records join deterministically to LEIs through registration-authority identifiers such as CIN.
- Measure the number of Indian parent-child edges after filtering to usable reporting status.
- Measure sector, company-age, and capital bias of matched versus unmatched MCA records.
- Exclude GLEIF from the MVP if it adds little held-out coverage or only decorates already-obvious large entities.

### 6.4 Fraud Detection Handbook

Official sources:

- [Simulator documentation](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html)
- [Source repository](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)

The common generated dataset has roughly 1.75 million transactions and about 0.8% fraud. It includes transaction time, customer, terminal, amount, and fraud scenario. Its value is experimental control, not realism for Indian merchant due diligence.

Use it only for a separate claim: the verifier can evaluate signals and operating policies under time-based splits and severe class imbalance. Never blend its labels with MCA corporate statuses.

### 6.5 IEEE-CIS

Source:

- [IEEE-CIS Fraud Detection data](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The competition provides roughly 590,000 training transactions with transaction and identity/device features. Acquisition requires Kaggle access and compliance with the competition rules. It should not be on the critical path because it validates transaction fraud, while the proposed product wedge concerns entity-context signals and unit selection.

## 7. Regulatory relevance

The [RBI Master Direction on Payment Aggregators](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12896), dated 15 September 2025, requires payment aggregators to monitor subsequent merchant transactions so they remain consistent with the merchant's business profile. It also gave an adaptation period for merchants onboarded through 31 December 2025.

This supports the importance of continuing merchant monitoring. It does not make the project unique. Competitors already market continuous merchant monitoring, so regulation is context, not the core differentiation.

## 8. Required experiments before architecture

Architecture should begin only after the following gates are run on locally acquired data.

### Gate A: data integrity

- Record counts by source and state.
- Schema drift and type failures.
- Registration-date parsing success.
- Duplicate CIN rate.
- Missingness by field and geography.
- Address normalization collision rate.
- Status frequencies by registration cohort.

### Gate B: label audit

- Define every target in plain legal and statistical language.
- Quantify how age and geography predict each target.
- Separate current status from later outcomes using an as-of date where possible.
- Never use `Strike Off` as shorthand for fraud.
- Report label coverage and possible selection bias.

### Gate C: reproduce the failed heuristic

Reproduce the earlier shared-address and incorporation-cohort analysis from raw data. Confirm or reject:

- address-sharing prevalence;
- candidate-ring prevalence;
- shared-address lift;
- overflag ratio;
- cohort-level concentration;
- stability across states and registration years.

If the direction of the result does not reproduce, stop using the prior narrative.

### Gate D: RE-UNIT test

For each candidate signal:

1. Evaluate individual-entity lift and calibration.
2. Build size-matched groups without using the target.
3. Estimate the expected outcome-count distribution under a binomial null.
4. Measure observed overdispersion and concentration.
5. Use bootstrap or permutation confidence intervals.
6. Retest on held-out time and geography.
7. Compare the individual policy with the group-aware policy at equal review capacity.

### Gate E: adverse-label linkage

- Join NSE/SEBI adverse entities to MCA.
- Publish exact and fuzzy match coverage.
- Audit false matches manually.
- Test whether the proposed signals precede or merely restate the adverse label.
- Report precision, recall, lift, and false-positive cost with confidence intervals.

### Gate F: GLEIF inclusion

Include GLEIF only if:

- deterministic India join coverage is material;
- usable ownership edges are numerous enough for held-out evaluation;
- the matched population is not so biased that results are misleading;
- at least one measurable decision improves beyond MCA-only baselines.

## 9. Evaluation contract

Every model or rule must specify:

- prediction target;
- unit of analysis;
- observation window;
- outcome window;
- split strategy;
- review capacity or operating threshold;
- precision, recall, PR-AUC, lift, and calibration;
- confidence intervals;
- false-positive cost;
- abstention and missing-data behavior;
- performance by state, age cohort, industry, and match-quality band.

Minimum baselines:

1. Random selection at equal review capacity.
2. Base-rate-only model.
3. Simple age and geography model.
4. Individual signal alone.
5. Group-aware version of the same signal.

No result should be presented using a random row split if the deployment claim is temporal or geographic generalization.

## 10. Product claims we can and cannot make

### Defensible after successful experiments

- The system audits candidate merchant-risk signals before deployment.
- It exposes leakage, base-rate mistakes, instability, and false-positive cost.
- It can recommend changing the unit of analysis when group-level concentration survives a null test and held-out validation.
- It keeps explanation separate from scoring.
- It provides a reproducible failure-and-recovery story.

### Not defensible today

- “Strike Off companies are fraudulent or shell companies.”
- “Shared addresses identify fraud rings.”
- “Nobody else detects merchant rings.”
- “Nobody else backtests fraud rules.”
- “The earlier AUC proves deployment readiness.”
- “GLEIF covers the Indian merchant population.”
- “All required datasets have been downloaded.”
- “Public data can reproduce Razorpay's transaction network.”

## 11. Recommended dataset stack

### MVP stack

1. **MCA Company Master Data** for entities, cohorts, addresses, capital, industry, and company status.
2. **Official NSE/SEBI adverse lists** for a narrow external validation target.
3. **Fraud Detection Handbook simulator**, only as an isolated controlled benchmark for the verifier workflow.

### Conditional extension

4. **GLEIF Level 1 and Level 2** if deterministic India coverage and usable ownership edges pass the coverage gate.
5. **OpenSanctions normalized data** for student-prototype convenience, with its noncommercial limitation made explicit.

### Defer or cut

- IEEE-CIS: defer unless a transaction-layer benchmark is needed and Kaggle access is available.
- ULB credit-card dataset: cut.
- RBI and NPCI documents: context only.
- Historical suspected-shell lists: exploratory only.

## 12. Immediate acquisition checklist

1. Download all required MCA state/ROC resources and record source URL, retrieval time, checksum, byte size, and declared data date.
2. Download the official NSE/SEBI spreadsheets and preserve the raw files unchanged.
3. Generate a fixed-version Fraud Detection Handbook dataset with a recorded seed and simulator commit.
4. Download GLEIF Level 1, relationship, and reporting-exception files only if time allows, then run the coverage gate immediately.
5. Store raw files as immutable inputs and convert working copies to Parquet.
6. Create a machine-readable source manifest containing licence, provenance, schema, time coverage, and allowed use.
7. Run the six research gates before UI or production architecture.

## 13. Build boundary for the next phase

Once the gates pass, the architecture should support:

- immutable raw-data ingestion;
- deterministic normalization and entity resolution;
- signal definitions as versioned code or configuration;
- individual and grouped evaluation;
- time/geography held-out experiments;
- reproducible policy simulation;
- decision reports with `SHIP`, `DO NOT SHIP`, or `RE-UNIT`;
- a thin API and demo interface;
- optional LLM narrative generated only from structured evidence.

Do not choose the final stack or divide work between Codex and Claude Code until raw-data access, target definitions, and the RE-UNIT experiment are validated. That division should be made from repository boundaries and verification responsibilities, not by asking both agents to edit the same files.

## 14. Current blockers and honest status

- The full MCA bulk extract is not yet local.
- Official NSE spreadsheets are not yet local.
- GLEIF endpoints were verified, but files are not yet local.
- IEEE-CIS is behind Kaggle access and rules.
- No present-day applicant leaderboard or submission count is public.
- Historical experiment statistics have not yet been reproduced on the current MCA distribution.

The research phase has therefore selected and prioritized the sources, but it has not yet earned the right to claim a production dataset or final architecture.
