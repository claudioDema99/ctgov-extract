"""Validation of model-quoted evidence against the original text.

Models are asked to copy a verbatim span. In practice they often change
whitespace, drop markdown escapes (``\\>=`` becomes ``>=``), turn ``≥`` into
``>=`` or change case. The matcher therefore tries, in order:

1. an exact substring match (status ``exact``);
2. a match after normalization (status ``normalized``): markdown escapes
   removed, runs of whitespace collapsed, typographic quotes, dashes and the
   ``≥``/``≤``/``≠`` signs mapped to ASCII, then the same with case folding.

Offsets always point into the original, un-normalized text. If nothing
matches, the status is ``not_found`` and the offsets are ``None``: the item is
kept but flagged, never silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ctgov_extract.models import Evidence, EvidenceStatus

_ESCAPABLE = set("\\`*_{}[]()#+-.!<>=~|^")
_REPLACEMENTS = {
    "\u2265": ">=",  # greater-than or equal to
    "\u2264": "<=",  # less-than or equal to
    "\u2260": "!=",  # not equal to
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # no-break space
}


@dataclass(frozen=True)
class _Normalized:
    text: str
    index: tuple[int, ...]  # index[i] = offset in the original of normalized char i


def normalize_with_map(text: str, *, fold_case: bool = False) -> _Normalized:
    chars: list[str] = []
    index: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] in _ESCAPABLE:
            i += 1
            ch = text[i]
            chars.append(ch)
            index.append(i)
            i += 1
            continue
        ch = _REPLACEMENTS.get(ch, ch)
        if ch.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                index.append(i)
            i += 1
            continue
        for c in ch:
            lowered = c.lower()
            chars.append(lowered if fold_case and len(lowered) == 1 else c)
            index.append(i)
        i += 1
    while chars and chars[-1] == " ":
        chars.pop()
        index.pop()
    return _Normalized("".join(chars), tuple(index))


def normalize(text: str, *, fold_case: bool = False) -> str:
    return normalize_with_map(text, fold_case=fold_case).text


def locate(source_text: str, haystack: str) -> tuple[EvidenceStatus, int | None, int | None]:
    """Find ``source_text`` in ``haystack``; return status and offsets within ``haystack``."""
    if not source_text or not source_text.strip():
        return "not_found", None, None

    pos = haystack.find(source_text)
    if pos >= 0:
        return "exact", pos, pos + len(source_text)

    for fold in (False, True):
        hay = normalize_with_map(haystack, fold_case=fold)
        needle = normalize(source_text, fold_case=fold)
        if not needle:
            break
        npos = hay.text.find(needle)
        if npos >= 0:
            start = hay.index[npos]
            end = hay.index[npos + len(needle) - 1] + 1
            return "normalized", start, end
    return "not_found", None, None


def validate_evidence(source_text: str, criterion_text: str, base_offset: int = 0) -> Evidence:
    """Check that ``source_text`` occurs in ``criterion_text``.

    ``base_offset`` is the offset of ``criterion_text`` within the full eligibility
    text, so the returned offsets are absolute.
    """
    status, start, end = locate(source_text, criterion_text)
    if start is None or end is None:
        return Evidence(source_text=source_text, status=status)
    return Evidence(
        source_text=source_text, status=status, start=base_offset + start, end=base_offset + end
    )


EVIDENCE_FIELD_NAMES = frozenset({"evidence", "source_text", "quote"})


class EvidenceCheck(BaseModel):
    path: str = Field(description="JSONPath-like location of the field in the result")
    source_text: str
    status: EvidenceStatus
    location: Literal["eligibility", "other_fields"] | None = None
    start: int | None = Field(default=None, description="Offset into the eligibility text")
    end: int | None = None


def check_evidence_fields(
    data: Any, eligibility_text: str, other_text: str = "", path: str = "$"
) -> list[EvidenceCheck]:
    """Validate every string field named like an evidence field inside ``data``.

    Used for user-defined schemas: each value of a field called ``evidence``,
    ``source_text`` or ``quote`` is searched in the eligibility text first (with
    offsets), then in the other trial fields (titles, conditions, outcomes...).
    """
    checks: list[EvidenceCheck] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{path}.{key}"
            if key in EVIDENCE_FIELD_NAMES and isinstance(value, str):
                checks.append(_check_one(child, value, eligibility_text, other_text))
            else:
                checks.extend(check_evidence_fields(value, eligibility_text, other_text, child))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            checks.extend(
                check_evidence_fields(value, eligibility_text, other_text, f"{path}[{i}]")
            )
    return checks


def _check_one(path: str, value: str, eligibility_text: str, other_text: str) -> EvidenceCheck:
    status, start, end = locate(value, eligibility_text)
    if status != "not_found":
        return EvidenceCheck(
            path=path,
            source_text=value,
            status=status,
            location="eligibility",
            start=start,
            end=end,
        )
    status, _, _ = locate(value, other_text)
    if status != "not_found":
        return EvidenceCheck(path=path, source_text=value, status=status, location="other_fields")
    return EvidenceCheck(path=path, source_text=value, status="not_found")
