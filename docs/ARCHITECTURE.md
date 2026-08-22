# ASSAY architecture

## Decision

Build a reproducible **signal-audit pipeline** first. It tests a candidate merchant-risk signal at entity and group levels, then returns `SHIP`, `DO_NOT_SHIP`, or `RE_UNIT` with saved evidence. This is not a fraud classifier. NSE/SEBI data is called `adverse_regulatory_outcome` and retains its source and match quality.

## Scope boundary

| Build now | Wait for evidence gate |
| --- | --- |
| Immutable source ingestion and manifests | Analyst-facing UI |
| Schema validation and canonical tables | Production-ready risk-score claims |
| Reviewable entity resolution | GLEIF data and ownership graphs |
| Deterministic signal/evaluation engine | LLM explanation layer |
| Reproducible reports | Deployment or Razorpay integration |

The first vertical slice is raw NSE files plus a bounded MCA API sample, ending in canonical tables and a match-quality report. It proves the data contract before a national MCA pull uses local disk.

## System shape

```text
Official source files/API
        |
        v
immutable raw store + checksum manifest
        |
        v
schema validation and canonicalisation
        |
        +--> MCA entity tables
        +--> NSE adverse-event tables
        |
        v
entity resolution + reviewed match decisions
        |
        v
feature and group builder, all as-of dated
        |
        v
signal audit engine --> metrics, null tests, held-out slices
        |                         |
        v                         v
verdict engine              immutable run artifacts
        |
        v
static report first; analyst UI only after the evidence gate
```

## Storage and data contracts

Use Parquet for canonical and derived data. Use DuckDB only as the local query engine. Raw files are never mutated or overwritten.

| Layer | Location | Purpose | Rule |
| --- | --- | --- | --- |
| Raw | `data/raw/` | Exact downloaded bytes | Checksum and retrieval metadata required |
| Staged | `data/staged/` | Source-shaped parsed rows | Preserve source row number and raw values |
| Curated | `data/curated/` | Canonical Parquet tables | Deterministic schemas and stable IDs |
| Generated | `data/generated/` | Run reports and metrics | Never a hidden source of truth |
| Manifest | `data/manifests/` | Source/acquisition provenance | Append versioned acquisition entries |

Canonical tables carry `source_id`, `source_record_id`, `source_file_sha256`, `source_snapshot_id`, `snapshot_as_of`, `observed_at`, and `schema_version`. Mutable company facts also carry source-backed `valid_from` and `valid_to` when those dates are known. Never infer an effective date from retrieval time. Derived tables carry `run_id`, `code_revision`, and `as_of_date`.

| Table | Grain | Decision it supports |
| --- | --- | --- |
| `company_snapshot` | one MCA legal entity per CIN, LLPIN, or FCRN and source snapshot | Eligible entity population and leakage-safe temporal cohorts |
| `company_address_snapshot` | one normalised address per company and source snapshot | As-of address groups without treating reuse as a label |
| `adverse_event` | one regulatory row from NSE | Narrow, sourced outcome target |
| `entity_match` | one MCA-to-NSE candidate match | Label coverage and false-match audit |
| `signal_definition` | one versioned signal specification | Exact audit reproducibility |
| `signal_observation` | entity or group per signal/run | Signal value before outcome window |
| `evaluation_slice` | one run and holdout slice | Stability and generalisation checks |
| `verdict` | one signal/run | Final result with evidence pointers |

Company and address snapshots are append-only. As-of features may use only a snapshot whose `snapshot_as_of` precedes the feature cutoff. When a source does not provide effective validity, `valid_from` and `valid_to` remain null and the pipeline must not claim that the fact was valid between snapshots.

No table stores a generic `fraud_label`. Features cannot use data after their outcome window begins, and mutable overwrites never hide a previous source or evaluation run.

## Matching policy

1. Match on CIN or DIN when an authoritative identifier exists.
2. Otherwise apply deterministic name normalisation.
3. Fuzzy matching creates candidates, never automatic labels.
4. Persist method, score, candidate set, and review decision.
5. Report exact, reviewed, ambiguous, and unmatched rates separately.

Only identifier and high-confidence reviewed matches can feed outcome-based evaluation until a reviewed sample proves match precision.

## Evaluation flow

Each signal definition declares its unit, population, observation window, outcome window, grouping rule, target, review capacity, and holdout strategy.

1. Validate data and target coverage.
2. Compute the entity-level baseline at fixed review capacity.
3. Construct groups without outcome data.
4. Test outcome concentration against a size-matched binomial null.
5. Bootstrap confidence intervals and compare with random, base-rate, and age-plus-geography baselines.
6. Hold out later registration periods and unseen geographies.
7. Check performance by state, age cohort, industry, and match-quality band.
8. Apply deterministic verdict thresholds and save all artifacts.

An aggregate lift alone cannot produce `SHIP`. A result that works only on a random row split cannot support a deployment claim.

## Interfaces

The initial interface is a command-line workflow and static HTML/JSON report. A browser UI comes later and reads versioned report artifacts only. It must never calculate or alter verdicts.

The optional LLM explanation adapter receives a redacted, structured verdict artifact and returns prose with references to computed fields. It has no raw-data access and cannot score or change a verdict.

## Technology choices

- Python 3.12 and `uv` for the environment.
- Polars and PyArrow for streaming typed transforms.
- DuckDB for local analytical queries and report inputs.
- Pydantic for file and run contracts.
- RapidFuzz for candidate generation only.
- scikit-learn and SciPy for baselines, bootstrap, and null tests.
- Pytest and Ruff for verification.

No database service, vector store, agent framework, or external AI API is required for the evidence pipeline.

## Preconditions for implementation

1. Confirm MCA page-size limits and estimate storage from representative pages.
2. Keep at least 25 GB free before a full raw MCA pull, or explicitly approve a different immutable-storage plan.
3. Implement a downloader with resumable checkpoints, bounded retries, rate-limit handling, per-page checksums, and no secret logging.
4. Validate a small vertical slice before requesting all 3.67 million records.
5. Freeze first signal definitions before adding an interface.

## Explicitly out of the first build

- Live Razorpay data or actioning.
- Fraud-detection claims.
- Automatic fuzzy-match labels.
- GLEIF ingestion before its coverage gate.
- Transaction benchmarks blended with MCA outcomes.
- A graph database. A relational edge table is enough until measured queries prove otherwise.
- A polished dashboard before the first reproducible report.
