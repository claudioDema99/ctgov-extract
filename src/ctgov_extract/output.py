"""Serialisation of results to JSON, JSONL and CSV."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Literal

from pydantic import BaseModel

from ctgov_extract.custom import CustomExtractionResult
from ctgov_extract.errors import CtgovExtractError
from ctgov_extract.models import AtomicCriterion, CriteriaResult
from ctgov_extract.user_schema import SchemaSpec

OutputFormat = Literal["json", "jsonl", "csv"]

ATOMIC_COLUMNS = [
    "id",
    "criterion_id",
    "kind",
    "category",
    "concept",
    "operator",
    "value",
    "value_max",
    "unit",
    "time_window",
    "negated",
    "is_biomarker",
    "evidence_status",
    "evidence_start",
    "evidence_end",
    "source_text",
]


def to_json(model: BaseModel) -> str:
    return model.model_dump_json(indent=2) + "\n"


def atomic_row(item: AtomicCriterion) -> dict[str, Any]:
    return {
        "id": item.id,
        "criterion_id": item.criterion_id,
        "kind": item.kind,
        "category": item.category.value,
        "concept": item.concept,
        "operator": item.operator.value if item.operator else None,
        "value": item.value,
        "value_max": item.value_max,
        "unit": item.unit,
        "time_window": item.time_window.text if item.time_window else None,
        "negated": item.negated,
        "is_biomarker": item.is_biomarker,
        "evidence_status": item.evidence.status,
        "evidence_start": item.evidence.start,
        "evidence_end": item.evidence.end,
        "source_text": item.evidence.source_text,
    }


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict | list) else v
                for k, v in row.items()
            }
        )
    return buf.getvalue()


def render_criteria(result: CriteriaResult, fmt: OutputFormat) -> str:
    if fmt == "json":
        return to_json(result)
    rows = [{"nct_id": result.nct_id, **atomic_row(a)} for a in result.atomic]
    if fmt == "jsonl":
        return _jsonl(rows)
    return _csv(rows, ["nct_id", *ATOMIC_COLUMNS])


def render_custom(result: CustomExtractionResult, spec: SchemaSpec, fmt: OutputFormat) -> str:
    """JSON always works; JSONL/CSV need exactly one top-level list[object] field to tabulate."""
    if fmt == "json":
        return to_json(result)
    list_fields = spec.list_fields
    if len(list_fields) != 1:
        raise CtgovExtractError(
            f"--format {fmt} needs a schema with exactly one top-level list[object] field "
            f"(found {len(list_fields)}); use --format json."
        )
    list_field = list_fields[0]
    assert list_field.fields is not None
    rows = [{"nct_id": result.nct_id, **item} for item in result.data.get(list_field.name) or []]
    if fmt == "jsonl":
        return _jsonl(rows)
    return _csv(rows, ["nct_id", *(f.name for f in list_field.fields)])
