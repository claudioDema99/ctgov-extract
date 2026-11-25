"""Evaluation of atomic-criteria extraction against a human-annotated gold set.

Gold format: JSONL, one :class:`GoldTrial` per line (see ``eval/ANNOTATION.md``).

Matching rule (one-to-one, greedy):

* A predicted item can only match a gold item of the same ``kind``.
* If both have character offsets, the score is the span IoU
  (intersection over union of the ``[start, end)`` intervals). Pairs with
  IoU >= ``SPAN_IOU_THRESHOLD`` are candidates.
* Otherwise (gold or prediction without offsets, e.g. evidence not found), the
  score is the fuzzy similarity of the normalized ``source_text`` strings
  (difflib ratio). Pairs with similarity >= ``TEXT_SIM_THRESHOLD`` are candidates.
* Candidates are matched greedily by descending score, then by concept
  similarity as a tie-breaker. Each gold and predicted item is used at most once.

Matched pairs are true positives; unmatched predictions are false positives and
unmatched gold items false negatives. Field accuracy is computed on matched pairs only.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from ctgov_extract.evidence import normalize
from ctgov_extract.models import (
    AtomicCriterion,
    Category,
    CriteriaResult,
    CriterionKind,
    Operator,
)

SPAN_IOU_THRESHOLD = 0.5
TEXT_SIM_THRESHOLD = 0.8
FIELDS = ("category", "operator", "value", "value_max", "unit", "negated")


class GoldItem(BaseModel):
    kind: CriterionKind
    category: Category
    concept: str
    operator: Operator | None = None
    value: float | str | None = None
    value_max: float | None = None
    unit: str | None = None
    negated: bool = False
    is_biomarker: bool | None = None
    time_window: str | None = None
    source_text: str
    start: int | None = None
    end: int | None = None
    criterion_id: str | None = None
    note: str | None = None


class GoldTrial(BaseModel):
    nct_id: str
    eligibility_text: str
    title: str | None = None
    conditions: list[str] = Field(default_factory=list)
    annotator: str | None = None
    annotated_at: str | None = None
    draft: bool = False
    synthetic: bool = False
    notes: str | None = None
    items: list[GoldItem]

    @model_validator(mode="after")
    def _check_offsets(self) -> GoldTrial:
        n = len(self.eligibility_text)
        for i, item in enumerate(self.items):
            if (item.start is None) != (item.end is None):
                raise ValueError(f"item {i}: give both start and end, or neither")
            if item.start is None or item.end is None:
                continue
            if not 0 <= item.start < item.end <= n:
                raise ValueError(f"item {i}: offsets [{item.start}, {item.end}) out of range")
            span = self.eligibility_text[item.start : item.end]
            if normalize(span) != normalize(item.source_text):
                raise ValueError(
                    f"item {i}: source_text does not match eligibility_text[start:end] "
                    f"({item.source_text!r} vs {span!r})"
                )
        return self


def load_gold(paths: Iterable[Path]) -> list[GoldTrial]:
    trials: list[GoldTrial] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    trials.append(GoldTrial.model_validate_json(line))
                except ValidationError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid gold record: {exc}") from exc
    return trials


def load_predictions(path: Path) -> dict[str, CriteriaResult]:
    preds: dict[str, CriteriaResult] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                result = CriteriaResult.model_validate_json(line)
                if result.nct_id:
                    preds[result.nct_id] = result
    return preds


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def text_similarity(a: str, b: str) -> float:
    na, nb = normalize(a, fold_case=True), normalize(b, fold_case=True)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb, autojunk=False).ratio()


def _pair_score(gold: GoldItem, pred: AtomicCriterion) -> float | None:
    if gold.kind != pred.kind:
        return None
    ev = pred.evidence
    if (
        gold.start is not None
        and gold.end is not None
        and ev.start is not None
        and ev.end is not None
    ):
        iou = span_iou((gold.start, gold.end), (ev.start, ev.end))
        return iou if iou >= SPAN_IOU_THRESHOLD else None
    sim = text_similarity(gold.source_text, ev.source_text)
    return sim if sim >= TEXT_SIM_THRESHOLD else None


def match_items(gold: list[GoldItem], pred: list[AtomicCriterion]) -> list[tuple[int, int, float]]:
    """Return ``(gold_index, pred_index, score)`` for each matched pair."""
    candidates = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            score = _pair_score(g, p)
            if score is not None:
                candidates.append((score, text_similarity(g.concept, p.concept), gi, pi))
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    pairs = []
    for score, _, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, score))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Field comparison
# ---------------------------------------------------------------------------


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace(",", ""))
        except ValueError:
            return None
    return None


def values_equal(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is None and b is None
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-6, abs_tol=1e-9)
    return normalize(str(a), fold_case=True) == normalize(str(b), fold_case=True)


def units_equal(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return not a and not b
    return a.replace(" ", "").lower() == b.replace(" ", "").lower()


def field_equal(name: str, gold: GoldItem, pred: AtomicCriterion) -> bool:
    if name == "unit":
        return units_equal(gold.unit, pred.unit)
    if name in ("value", "value_max"):
        return values_equal(getattr(gold, name), getattr(pred, name))
    return bool(getattr(gold, name) == getattr(pred, name))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class TrialScore:
    nct_id: str
    n_gold: int
    n_pred: int
    tp: int
    not_found: int
    failures: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    calls: int
    field_hits: dict[str, int] = field(default_factory=dict)

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gold - self.tp


def prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_trial(gold: GoldTrial, result: CriteriaResult) -> TrialScore:
    pairs = match_items(gold.items, result.atomic)
    hits = dict.fromkeys(FIELDS, 0)
    for gi, pi, _ in pairs:
        for name in FIELDS:
            if field_equal(name, gold.items[gi], result.atomic[pi]):
                hits[name] += 1
    return TrialScore(
        nct_id=gold.nct_id,
        n_gold=len(gold.items),
        n_pred=len(result.atomic),
        tp=len(pairs),
        not_found=len(result.evidence_not_found),
        failures=len(result.failures),
        latency_s=result.elapsed_s,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        calls=result.usage.calls,
        field_hits=hits,
    )


class EvalSummary(BaseModel):
    n_trials: int
    n_gold: int
    n_pred: int
    tp: int
    precision: float
    recall: float
    f1: float
    field_accuracy: dict[str, float | None]
    evidence_not_found_rate: float | None
    failed_criteria: int
    latency_mean_s: float | None
    latency_median_s: float | None
    latency_max_s: float | None
    prompt_tokens: int
    completion_tokens: int
    calls: int
    per_trial: list[dict[str, Any]]


def summarize(scores: list[TrialScore]) -> EvalSummary:
    n_gold = sum(s.n_gold for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    tp = sum(s.tp for s in scores)
    p, r, f = prf(tp, n_pred, n_gold)
    accuracy: dict[str, float | None] = {
        name: (sum(s.field_hits.get(name, 0) for s in scores) / tp if tp else None)
        for name in FIELDS
    }
    latencies = [s.latency_s for s in scores]
    per_trial = []
    for s in scores:
        tp_, tr, tf = prf(s.tp, s.n_pred, s.n_gold)
        per_trial.append(
            {
                "nct_id": s.nct_id,
                "gold": s.n_gold,
                "pred": s.n_pred,
                "tp": s.tp,
                "precision": tp_,
                "recall": tr,
                "f1": tf,
                "not_found": s.not_found,
                "failures": s.failures,
                "latency_s": s.latency_s,
            }
        )
    return EvalSummary(
        n_trials=len(scores),
        n_gold=n_gold,
        n_pred=n_pred,
        tp=tp,
        precision=p,
        recall=r,
        f1=f,
        field_accuracy=accuracy,
        evidence_not_found_rate=(sum(s.not_found for s in scores) / n_pred) if n_pred else None,
        failed_criteria=sum(s.failures for s in scores),
        latency_mean_s=statistics.fmean(latencies) if latencies else None,
        latency_median_s=statistics.median(latencies) if latencies else None,
        latency_max_s=max(latencies) if latencies else None,
        prompt_tokens=sum(s.prompt_tokens for s in scores),
        completion_tokens=sum(s.completion_tokens for s in scores),
        calls=sum(s.calls for s in scores),
        per_trial=per_trial,
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    summary: EvalSummary,
    *,
    model: str | None,
    gold_files: list[str],
    skipped: list[str],
) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: {model or 'n/a'}",
        f"- Gold files: {', '.join(gold_files)}",
        f"- Trials scored: {summary.n_trials}",
    ]
    if skipped:
        lines.append(f"- Skipped (draft or synthetic, not human-verified): {', '.join(skipped)}")
    lines += [
        "",
        "## Criterion-level detection",
        "",
        "| Gold items | Predicted | Matched | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {summary.n_gold} | {summary.n_pred} | {summary.tp} | {_fmt(summary.precision)} | "
        f"{_fmt(summary.recall)} | {_fmt(summary.f1)} |",
        "",
        "## Field accuracy on matched pairs",
        "",
        "| Field | Accuracy |",
        "|---|---:|",
    ]
    lines += [f"| {name} | {_fmt(acc)} |" for name, acc in summary.field_accuracy.items()]
    lines += [
        "",
        "## Evidence, failures, cost",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evidence not found rate | {_fmt(summary.evidence_not_found_rate)} |",
        f"| Criteria with failed LLM output | {summary.failed_criteria} |",
        f"| LLM calls | {summary.calls} |",
        f"| Prompt tokens | {summary.prompt_tokens} |",
        f"| Completion tokens | {summary.completion_tokens} |",
        f"| Latency per trial, mean (s) | {_fmt(summary.latency_mean_s, 2)} |",
        f"| Latency per trial, median (s) | {_fmt(summary.latency_median_s, 2)} |",
        f"| Latency per trial, max (s) | {_fmt(summary.latency_max_s, 2)} |",
        "",
        "## Per trial",
        "",
        "| NCT ID | Gold | Pred | Matched | P | R | F1 | Not found | Failed | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in summary.per_trial:
        lines.append(
            f"| {t['nct_id']} | {t['gold']} | {t['pred']} | {t['tp']} | {_fmt(t['precision'])} | "
            f"{_fmt(t['recall'])} | {_fmt(t['f1'])} | {t['not_found']} | {t['failures']} | "
            f"{_fmt(t['latency_s'], 2)} |"
        )
    lines += [
        "",
        "## Matching rule",
        "",
        f"Same kind (inclusion/exclusion) required. Span IoU >= {SPAN_IOU_THRESHOLD} when both "
        f"sides have offsets, otherwise normalized source_text similarity >= {TEXT_SIM_THRESHOLD}. "
        "Greedy one-to-one assignment by score. Field accuracy is computed on matched pairs.",
        "",
    ]
    return "\n".join(lines)


def result_to_draft(
    result: CriteriaResult, *, title: str | None, conditions: list[str]
) -> GoldTrial:
    """Turn a model output into a *draft* gold record for a human to correct."""
    items = []
    for a in result.atomic:
        found = a.evidence.status != "not_found"
        items.append(
            GoldItem(
                kind=a.kind,
                category=a.category,
                concept=a.concept,
                operator=a.operator,
                value=a.value,
                value_max=a.value_max,
                unit=a.unit,
                negated=a.negated,
                is_biomarker=a.is_biomarker,
                time_window=a.time_window.text if a.time_window else None,
                source_text=(
                    result.eligibility_text[a.evidence.start : a.evidence.end]
                    if found and a.evidence.start is not None and a.evidence.end is not None
                    else a.evidence.source_text
                ),
                start=a.evidence.start if found else None,
                end=a.evidence.end if found else None,
                criterion_id=a.criterion_id,
                note=None
                if found
                else "Model quote not found in the text: fix source_text and add offsets.",
            )
        )
    return GoldTrial(
        nct_id=result.nct_id or "UNKNOWN",
        eligibility_text=result.eligibility_text,
        title=title,
        conditions=conditions,
        annotator=None,
        annotated_at=None,
        draft=True,
        notes=f"Draft produced by model {result.model or 'unknown'}. Review every item, "
        "add missing items, then set draft to false and fill annotator.",
        items=items,
    )


def dump_jsonl(records: Iterable[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
