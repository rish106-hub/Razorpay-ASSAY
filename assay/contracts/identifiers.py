"""Authoritative identifier formats used by merchant-risk data contracts."""

from __future__ import annotations

# MCA Corporate Identity Number: listing status, industry code, state, year,
# ownership type, and six-digit registration number.
INDIAN_CIN_PATTERN = r"^[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$"

# MCA Limited Liability Partnership Identification Number.
INDIAN_LLPIN_PATTERN = r"^[A-Z]{3}-[0-9]{4}$"

# MCA Foreign Company Registration Number.
FOREIGN_COMPANY_REGISTRATION_PATTERN = r"^F[0-9]{5}$"
