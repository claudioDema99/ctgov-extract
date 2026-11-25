"""Metric code tests on a tiny SYNTHETIC gold record (tests/fixtures/synthetic_gold.jsonl).

The synthetic record is hand-written test data, not an annotation of a real trial.
"""

from pathlib import Path

import pytest

from ctgov_extract.evaluation import (
    GoldTrial,
    load_gold,
    match_items,
    render_report,
    result_to_draft,
    score_trial,
    span_iou,
    summarize,
    text_similarity,
    units_equal,
    values_equal,
)
from ctgov_extract.evidence import validate_evidence
from ctgov_extract.models import AtomicCriterion, Category, CriteriaResult, Operator, Usage
from tests.conftest import FIXTURES

GOLD_PATH = FIXTURES / "synthetic_gold.jsonl"


@pytest.fixture
def gold() -> GoldTrial:
    (trial,) = load_gold([GOLD_PATH])
    assert trial.synthetic
    return trial


def _pred(gold: GoldTrial, quote: str, **kwargs: object) -> AtomicCriterion:
    fields: dict[str, object] = {
        "id": f"p{len(quote)}",
        "criterion_id": "c",
        "kind": "inclusion",
        "category": Category.OTHER,
        "concept": "x",
    } | kwargs
    return AtomicCriterion(evidence=validate_evidence(quote, gold.eligibility_text), **fields)  # type: ignore[arg-type]


def _predictions(gold: GoldTrial) -> CriteriaResult:
    atomic = [
        # exact match of gold item 0, all fields right
        _pred(
            gold,
            "Age 18 years or older",
            category=Category.DEMOGRAPHICS,
            concept="age",
            operator=Operator.GE,
            value=18.0,
            unit="Years",
        ),
        # overlaps gold item 1 (IoU >= 0.5), operator wrong
        _pred(
            gold,
            "MMSE score between 20 and 26 at screening",
            category=Category.SCORE,
            concept="MMSE score",
            operator=Operator.GE,
            value="20",
            value_max=26,
        ),
        # gold item 3 found through normalization (escape dropped)
        _pred(
            gold,
            "LVEF < 50%",
            kind="exclusion",
            category=Category.LABORATORY,
            concept="LVEF",
            operator=Operator.LT,
            value=50,
            unit="%",
        ),
        # spurious prediction, quote not in the text
        _pred(gold, "ECOG 0-1", kind="exclusion", concept="ECOG"),
    ]
    return CriteriaResult(
        nct_id=gold.nct_id,
        model="fake",
        eligibility_text=gold.eligibility_text,
        criteria=[],
        atomic=atomic,
        usage=Usage(calls=4, prompt_tokens=100, completion_tokens=40, total_tokens=140),
        elapsed_s=1.5,
    )


def test_gold_offsets_are_consistent(gold: GoldTrial) -> None:
    for item in gold.items:
        assert item.start is not None and item.end is not None
        assert gold.eligibility_text[item.start : item.end] == item.source_text


def test_gold_validation_rejects_wrong_offsets(gold: GoldTrial) -> None:
    data = gold.model_dump()
    data["items"][0]["start"] += 1
    with pytest.raises(ValueError, match="does not match"):
        GoldTrial.model_validate(data)
    data = gold.model_dump()
    data["items"][0]["end"] = None
    with pytest.raises(ValueError, match="both start and end"):
        GoldTrial.model_validate(data)


def test_matching_and_scores(gold: GoldTrial) -> None:
    result = _predictions(gold)
    pairs = match_items(gold.items, result.atomic)
    assert [(g, p) for g, p, _ in pairs] == [(0, 0), (1, 1), (3, 2)]

    score = score_trial(gold, result)
    assert (score.tp, score.fp, score.fn) == (3, 1, 1)
    assert score.not_found == 1
    assert score.field_hits["category"] == 3
    assert score.field_hits["operator"] == 2
    assert score.field_hits["value"] == 3  # "20" == 20, 18.0 == 18
    assert score.field_hits["value_max"] == 3
    assert score.field_hits["unit"] == 3  # "Years" == "years"
    assert score.field_hits["negated"] == 3

    summary = summarize([score])
    assert summary.precision == pytest.approx(0.75)
    assert summary.recall == pytest.approx(0.75)
    assert summary.f1 == pytest.approx(0.75)
    assert summary.field_accuracy["operator"] == pytest.approx(2 / 3)
    assert summary.evidence_not_found_rate == pytest.approx(0.25)
    assert summary.latency_mean_s == pytest.approx(1.5)
    assert summary.prompt_tokens == 100


def test_kind_must_match(gold: GoldTrial) -> None:
    wrong_kind = _pred(gold, "Age 18 years or older", kind="exclusion")
    assert match_items(gold.items, [wrong_kind]) == []


def test_fuzzy_text_fallback_when_offsets_missing(gold: GoldTrial) -> None:
    paraphrase = _pred(gold, "age 18 years or older.", concept="age")  # case + period
    assert paraphrase.evidence.status == "not_found"
    pairs = match_items(gold.items, [paraphrase])
    assert [(g, p) for g, p, _ in pairs] == [(0, 0)]


def test_one_to_one(gold: GoldTrial) -> None:
    a = _pred(gold, "Age 18 years or older", concept="age")
    b = _pred(gold, "Age 18 years", concept="age")
    pairs = match_items(gold.items, [b, a])
    assert len(pairs) == 1 and pairs[0][1] == 1  # the better (exact) span wins


def test_empty_summary() -> None:
    summary = summarize([])
    assert summary.f1 == 0.0
    assert summary.field_accuracy["category"] is None
    assert summary.latency_mean_s is None


def test_helpers() -> None:
    assert span_iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert span_iou((0, 5), (5, 10)) == 0.0
    assert text_similarity("A  b", "a b") == 1.0
    assert text_similarity("", "x") == 0.0
    assert values_equal(None, None) and not values_equal(None, 0)
    assert values_equal("1,500", 1500)
    assert values_equal("Positive", "positive")
    assert units_equal("mg/dL", "mg / dl") and units_equal(None, "") and not units_equal("%", None)


def test_report(gold: GoldTrial) -> None:
    summary = summarize([score_trial(gold, _predictions(gold))])
    report = render_report(summary, model="fake", gold_files=[str(GOLD_PATH)], skipped=["X"])
    assert report.startswith("# Evaluation report")
    assert "| 4 | 4 | 3 | 0.750 | 0.750 | 0.750 |" in report
    assert "| operator | 0.667 |" in report
    assert "SYNTH-0001" in report
    assert "Skipped" in report


def test_draft_from_result(gold: GoldTrial, tmp_path: Path) -> None:
    result = _predictions(gold)
    draft = result_to_draft(result, title="t", conditions=[])
    assert draft.draft and draft.annotator is None
    assert len(draft.items) == 4
    assert draft.items[2].source_text == "LVEF \\< 50%"  # raw span, not the model's quote
    assert draft.items[3].start is None and draft.items[3].note
    GoldTrial.model_validate(draft.model_dump())  # drafts are valid gold-format records
