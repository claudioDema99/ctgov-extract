import asyncio
import json
from typing import Any

from ctgov_extract.atomize import CriteriaExtractor
from ctgov_extract.llm.base import ChatMessage
from ctgov_extract.llm.fake import FakeLLMClient
from ctgov_extract.models import Category, Operator
from tests.conftest import criterion_text_from, eligibility


def _item(**kwargs: Any) -> dict[str, Any]:
    base = {"category": "other", "concept": "x", "negated": False, "is_biomarker": False}
    return base | kwargs


def lecanemab_responder(messages: list[ChatMessage], _: dict[str, Any] | None) -> str:
    """Hand-written replies for a few criteria; generic echo for the rest."""
    text = criterion_text_from(messages)
    if text.startswith("Mini mental state"):
        items = [
            _item(
                category="cognitive/functional score",
                concept="MMSE score",
                operator=">=",
                value=22,
                time_window={"text": "at Screening and Baseline"},
                # the model drops the markdown escape: should still be located
                source_text="MMSE) score >=22 at Screening and Baseline",
            ),
            _item(
                category="cognitive/functional score",
                concept="MMSE score",
                operator="<=",
                value=30,
                # paraphrase, not a quote: must be flagged
                source_text="MMSE no higher than 30",
            ),
        ]
    elif text.startswith("Positive biomarker"):
        items = [
            _item(
                category="laboratory/biomarker",
                concept="brain amyloid pathology",
                operator="present",
                is_biomarker=True,
                source_text="Positive biomarker for brain amyloid pathology",
            )
        ]
    elif text.startswith("Geriatric Depression Scale"):
        return "this is not JSON at all"  # both attempts fail -> local failure
    else:
        quote = " ".join(text.split()[:4])
        items = [_item(concept=quote.lower(), operator="present", source_text=quote)]
    return json.dumps({"items": items})


def test_atomic_extraction_with_evidence_validation() -> None:
    text = eligibility("NCT03887455")
    client = FakeLLMClient(responder=lecanemab_responder)
    result = asyncio.run(
        CriteriaExtractor(client).extract_text(
            text, nct_id="NCT03887455", title="Lecanemab", conditions=["Early Alzheimer's Disease"]
        )
    )
    assert result.nct_id == "NCT03887455"
    assert result.model == "fake-model"
    mmse = [a for a in result.atomic if a.concept == "MMSE score"]
    assert len(mmse) == 2
    first, second = mmse
    assert first.id.endswith("-a1") and second.id.endswith("-a2")
    assert first.kind == "inclusion"
    assert first.category is Category.SCORE
    assert first.operator is Operator.GE and first.value == 22
    assert first.evidence.status == "normalized"
    assert first.evidence.start is not None and first.evidence.end is not None
    assert text[first.evidence.start : first.evidence.end] == (
        "MMSE) score \\>=22 at Screening and Baseline"
    )
    assert second.evidence.status == "not_found"
    assert second.evidence.start is None

    amyloid = next(a for a in result.atomic if a.concept == "brain amyloid pathology")
    assert amyloid.is_biomarker and amyloid.evidence.status == "exact"

    # the GDS criterion failed twice (initial + repair), others are unaffected
    assert len(result.failures) == 1
    gds = next(c for c in result.criteria if c.text.startswith("Geriatric Depression Scale"))
    assert result.failures[0].criterion_id == gds.id
    assert not any(a.criterion_id == gds.id for a in result.atomic)
    assert len({a.criterion_id for a in result.atomic}) == len(result.criteria) - 1
    # one call per criterion plus one repair call for the failing criterion
    assert len(client.calls) == len(result.criteria) + 1
    assert result.usage.calls == len(client.calls)  # failed attempts are counted too
    assert result.evidence_not_found == [second]


def test_prompts_carry_context(echo_llm: FakeLLMClient) -> None:
    text = eligibility("NCT03529110")
    asyncio.run(
        CriteriaExtractor(echo_llm).extract_text(
            text, nct_id="NCT03529110", title="DESTINY-Breast03", conditions=["Breast Cancer"]
        )
    )
    prompts = {criterion_text_from(c.messages): c.messages for c in echo_llm.calls}
    lvef = prompts["LVEF < 50% within 28 d prior to randomization"]
    assert "Section: exclusion" in lvef[-1].content
    assert "sub-item of" in lvef[-1].content
    assert "Uncontrolled or significant cardiovascular disease" in lvef[-1].content
    parent = prompts[
        "Uncontrolled or significant cardiovascular disease, including any of the following:"
    ]
    assert "sub-items that are processed separately" in parent[-1].content
    system = lvef[0].content
    assert "laboratory/biomarker" in system and '"between"' in system
    assert '"source_text"' in system  # JSON schema is in the prompt
    assert "DESTINY-Breast03" in lvef[-1].content and "Breast Cancer" in lvef[-1].content


def test_echo_quotes_are_all_found(echo_llm: FakeLLMClient) -> None:
    text = eligibility("NCT03529110")
    result = asyncio.run(CriteriaExtractor(echo_llm).extract_text(text))
    assert result.atomic
    assert result.evidence_not_found == []
    for item in result.atomic:
        assert item.evidence.start is not None and item.evidence.end is not None
        criterion = next(c for c in result.criteria if c.id == item.criterion_id)
        assert criterion.start <= item.evidence.start < item.evidence.end <= criterion.end


def test_empty_text_makes_no_calls(echo_llm: FakeLLMClient) -> None:
    result = asyncio.run(CriteriaExtractor(echo_llm).extract_text(""))
    assert result.criteria == [] and result.atomic == []
    assert echo_llm.calls == []
