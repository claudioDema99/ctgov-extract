"""Prompt templates. Kept in one module so they are easy to review and version."""

from __future__ import annotations

import json
from typing import Any

from ctgov_extract.models import Category, Criterion, Operator, TrialRecord

ATOMIC_SYSTEM = """\
You convert clinical trial eligibility criteria into atomic, machine-readable criteria.

An atomic criterion tests exactly one concept (one diagnosis, one lab value, one score,
one drug class, one procedure...). Split a criterion into several atomic criteria when it
combines several tests with "and", "or", commas or ranges. Every input criterion must
produce at least one atomic criterion.

Fields:
- category: one of {categories}.
  Biomarker requirements (HER2, PD-L1, amyloid PET positivity...) use "laboratory/biomarker";
  "imaging" is for other imaging findings. ECOG and similar scales use
  "cognitive/functional score".
- concept: short normalized name, e.g. "MMSE score", "LVEF", "HER2 status",
  "prior anti-PD-1 therapy". No values or units in the concept.
- operator: one of {operators}, or null if no test applies.
  Use present/absent for things that are or are not there (a diagnosis, a prior
  treatment). Use between with value (lower) and value_max (upper) for ranges, and say
  in the concept if bounds are exclusive.
- value: a number when the text gives one, otherwise a short string (e.g. "positive",
  "0-1" is a range: use between) or null.
- unit: as written ("mg/dL", "%", "ms", "years"), or null.
- time_window: when the criterion has a time constraint, an object with text (as written,
  e.g. "within 6 months before screening") and, if possible, amount, unit and direction
  (before / after / during). Otherwise null.
- negated: true only if the text uses negation for this concept ("no", "not", "without",
  "absence of", "never"). The section (inclusion/exclusion) is NOT negation: an exclusion
  "History of stroke" has operator "present" and negated false.
- is_biomarker: true for molecular, genetic, imaging or fluid biomarkers (HER2, PD-L1,
  EGFR mutation, amyloid PET, CSF tau, APOE4...).
- source_text: copy the shortest span of the criterion text that supports this atomic
  criterion, character for character. Do not paraphrase, do not fix typos, do not join
  separate pieces. It must be a substring of the criterion text.

Answer with a single JSON object matching this JSON schema, and nothing else:
{schema}
"""

ATOMIC_USER = """\
Trial: {title}
Conditions: {conditions}
Section: {kind}{section}{group}{parent}

Criterion text:
<<<
{text}
>>>
"""


def atomic_system_prompt(schema: dict[str, Any]) -> str:
    return ATOMIC_SYSTEM.format(
        categories=", ".join(f'"{c.value}"' for c in Category),
        operators=", ".join(f'"{o.value}"' for o in Operator),
        schema=json.dumps(schema, ensure_ascii=False),
    )


def atomic_user_prompt(
    criterion: Criterion,
    *,
    title: str | None,
    conditions: list[str],
    parent: Criterion | None,
    has_children: bool,
) -> str:
    section = f" (header: {criterion.section})" if criterion.section else ""
    group = f"\nSub-heading: {criterion.group}" if criterion.group else ""
    parent_text = ""
    if parent is not None:
        parent_text = (
            f'\nThis is a sub-item of: "{parent.clean_text}" '
            "(use it as context; quote source_text only from the criterion text below)."
        )
    if has_children:
        parent_text += (
            "\nThis criterion has sub-items that are processed separately; extract only "
            "what this text itself states."
        )
    return ATOMIC_USER.format(
        title=title or "(unknown)",
        conditions=", ".join(conditions) or "(not given)",
        kind=criterion.kind,
        section=section,
        group=group,
        parent=parent_text,
        text=criterion.clean_text,
    )


CUSTOM_SYSTEM = """\
You extract structured information from a clinical trial record from ClinicalTrials.gov.
Use only information present in the record. If something is not stated, use null (or an
empty list) instead of guessing. When a field asks for evidence or a quote, copy the text
character for character from the record.
{instructions}
Answer with a single JSON object matching this JSON schema, and nothing else:
{schema}
"""


def custom_system_prompt(schema: dict[str, Any], instructions: str | None) -> str:
    extra = f"\nTask-specific instructions:\n{instructions.strip()}\n" if instructions else ""
    return CUSTOM_SYSTEM.format(instructions=extra, schema=json.dumps(schema, ensure_ascii=False))


def trial_context(record: TrialRecord) -> str:
    """Compact, readable rendering of a trial for the custom-schema prompt."""
    summary = record.model_dump(
        mode="json",
        exclude={"eligibility_text", "arms", "secondary_outcomes", "last_update_posted"},
        exclude_none=True,
    )
    for outcome in summary.get("primary_outcomes", []):
        outcome.pop("description", None)
    return (
        "Structured fields (JSON):\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=1)}\n\n"
        "Eligibility criteria (verbatim):\n<<<\n"
        f"{record.eligibility_text}\n>>>\n"
    )
