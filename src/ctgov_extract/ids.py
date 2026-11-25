"""Parsing of NCT IDs and ClinicalTrials.gov URLs."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from ctgov_extract.errors import InvalidTrialIdError

_NCT_BARE = re.compile(r"^\s*(NCT\d{8})\s*$", re.IGNORECASE)
_NCT_ANYWHERE = re.compile(r"(?<![A-Za-z0-9])(NCT\d{8})(?![0-9])", re.IGNORECASE)
_ALLOWED_HOSTS = {"clinicaltrials.gov", "www.clinicaltrials.gov", "classic.clinicaltrials.gov"}


def parse_nct_id(value: str) -> str:
    """Return the canonical (upper-case) NCT ID contained in ``value``.

    Accepts a bare ID (``nct01234567``), or a ClinicalTrials.gov URL such as
    ``https://clinicaltrials.gov/study/NCT01234567?tab=results`` or the legacy
    ``https://clinicaltrials.gov/ct2/show/NCT01234567``. URLs without a scheme
    (``clinicaltrials.gov/study/...``) are accepted too.

    Raises:
        InvalidTrialIdError: if no NCT ID can be found, or the URL points to
            another host.
    """
    if not value or not value.strip():
        raise InvalidTrialIdError("Empty trial identifier.")

    bare = _NCT_BARE.match(value)
    if bare:
        return bare.group(1).upper()

    text = value.strip()
    candidate = text if "://" in text else f"https://{text}"
    parts = urlsplit(candidate)
    host = (parts.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise InvalidTrialIdError(
            f"Not an NCT ID or ClinicalTrials.gov URL: {value!r}. "
            "Expected something like NCT01234567 or https://clinicaltrials.gov/study/NCT01234567."
        )

    # Prefer the path; fall back to the query string (e.g. ?id=NCT01234567).
    for part in (parts.path, parts.query):
        match = _NCT_ANYWHERE.search(part)
        if match:
            return match.group(1).upper()
    raise InvalidTrialIdError(f"No NCT ID found in URL: {value!r}.")


def is_nct_id(value: str) -> bool:
    """True if ``value`` is exactly a well-formed NCT ID (case-insensitive)."""
    return bool(_NCT_BARE.match(value))
