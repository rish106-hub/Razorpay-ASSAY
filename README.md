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
