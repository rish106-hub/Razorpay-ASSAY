# Razorpay AI Builder Hackathon: Merchant Risk Signal Verifier

This project tests whether merchant-risk signals hold up at the right level:
individual company, address cluster, ownership group, or network.

The first milestone is data evidence, not an application UI. We must reproduce
the shared-address false-positive result and validate suspicious cohorts before
choosing a production architecture.

See [the transferred project context](docs/PROJECT_CONTEXT.md) and the
[data acquisition plan](docs/DATA_ACQUISITION.md).

Install the locked Python 3.12 environment with `uv sync --locked`. Verify the
backend with `uv run python -m ruff check .` and `uv run python -m pytest`.

Acquire one bounded MCA page with:

```bash
uv run python -m assay.cli.acquire_mca --page-size 100 --max-pages 1
```

The command reads `DATA_GOV_IN_API_KEY`, publishes checksum-named immutable raw
pages, and advances `checkpoint.json` only after a valid page is stored. Keep
the default bounded run until schema and storage artifacts have been audited.

After the acquisition checkpoint is complete, canonicalise it with:

```bash
uv run python -m assay.cli.canonicalise_mca
```

This writes versioned staged Parquet, append-only `company_snapshot` and
`company_address_snapshot` artifacts, and a deterministic quality report. The
source cutoff remains 3 November 2023; retrieval time is not treated as an
effective date.

Canonicalise official NSE regulatory sources separately with:

```bash
uv run python -m assay.cli.canonicalise_nse
```

Link explicit, quality-passed MCA and NSE snapshots. Only a unique exact CIN
match is outcome-eligible. Exact-name matches stay pending for human review.

```bash
uv run python -m assay.cli.link_entities \
  --mca-report data/generated/mca_canonicalisation/<snapshot>.report.json \
  --nse-report data/generated/nse_canonicalisation/<snapshot>.report.json
```

Build a stratified linkage review workbook. Reviewers must use `MATCH`,
`NO_MATCH`, or `UNSURE`; exact-name candidates never become labels by default.

```bash
uv run python -m assay.cli.build_linkage_review \
  --linkage-report data/generated/entity_linkage/<run>.report.json \
  --nse-report data/generated/nse_canonicalisation/<snapshot>.report.json
```

Build the as-of shared-address and address-plus-registration-month signals. The
outcome window starts the day after the MCA source cutoff, and entities with a
known earlier regulatory outcome are excluded from evaluation.

```bash
uv run python -m assay.cli.build_signal_observations \
  --mca-report data/generated/mca_canonicalisation/<snapshot>.report.json \
  --nse-report data/generated/nse_canonicalisation/<snapshot>.report.json \
  --linkage-report data/generated/entity_linkage/<run>.report.json \
  --outcome-window-end 2026-08-22
```

Evaluate both signal units on the full eligible population, unseen-state
holdout, and prospective temporal holdout. Review-capacity ties are retained
and reported instead of silently broken.

```bash
uv run python -m assay.cli.evaluate_signals \
  --observation-report data/generated/signal_observation/<run>.report.json \
  --temporal-holdout-start 2026-01-01 \
  --review-capacity 0.01 \
  --false-positive-review-cost-inr 100
```

The resulting `adverse_event` artifacts keep SEBI and other-authority actions
separate and name the target `adverse_regulatory_outcome`. They do not create a
generic fraud label.
