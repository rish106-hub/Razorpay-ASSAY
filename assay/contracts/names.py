"""Deterministic company-name normalisation shared by linkage and serving."""

from __future__ import annotations

import polars as pl

COMPANY_NAME_NORMALISATION_VERSION = "india_company_name_v1"
_LEGAL_SUFFIX_PATTERN = r"\b(PRIVATE|PVT|LIMITED|LTD|LLP|COMPANY|CO)\b"


def strict_name_expression(source_column: str) -> pl.Expr:
    """Uppercase and collapse a name to alphanumeric words only."""

    return (
        pl.col(source_column)
        .fill_null("")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def legal_name_expression(strict_name: pl.Expr) -> pl.Expr:
    """Drop Indian legal-form suffixes from an already strict-normalised name."""

    return (
        strict_name.str.replace_all(_LEGAL_SUFFIX_PATTERN, " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalise_query_name(company_name: str) -> tuple[str, str]:
    """Normalise one caller-supplied name with the pipeline's exact rules."""

    frame = pl.DataFrame({"query_name": [company_name]}).select(
        strict_name_expression("query_name").alias("strict"),
    )
    strict_name = str(frame["strict"][0])
    legal_frame = pl.DataFrame({"strict": [strict_name]}).select(
        legal_name_expression(pl.col("strict")).alias("legal"),
    )
    return strict_name, str(legal_frame["legal"][0])
