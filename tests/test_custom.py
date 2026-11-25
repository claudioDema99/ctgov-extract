import asyncio
import json

import pytest

from ctgov_extract.custom import CustomExtractor
from ctgov_extract.errors import StructuredOutputError
from ctgov_extract.llm.fake import FakeLLMClient
from ctgov_extract.mapping import study_to_record
from ctgov_extract.user_schema import load_schema_file
from tests.conftest import REPO_ROOT, load_study

SCHEMA = REPO_ROOT / "schemas" / "oncology_biomarkers.yaml"

GOOD = {
    "biomarkers": [
        {
            "name": "HER2",
            "kind": "inclusion",
            "biomarker_type": "protein expression",
            "requirement": "positive",
            "evidence": "Subjects must be HER2-positive as confirmed by central laboratory "
            "assessment of most recent tumor tissue sample available.",
        },
        {
            "name": "LVEF",
            "kind": "exclusion",
            "operator": "<",
            "threshold": 50,
            "unit": "%",
            "evidence": "LVEF < 50% within 28 d prior to randomization",
        },
    ]
}


def test_custom_extraction_with_repair() -> None:
    record = study_to_record(load_study("NCT03529110"))
    spec = load_schema_file(SCHEMA)
    bad = {"biomarkers": [{"name": "HER2", "kind": "maybe", "evidence": "x"}]}
    client = FakeLLMClient(responses=[json.dumps(bad), "```json\n" + json.dumps(GOOD) + "\n```"])

    result = asyncio.run(CustomExtractor(client).extract(record, spec))

    assert result.nct_id == "NCT03529110"
    assert result.schema_name == "oncology_biomarkers"
    assert result.repaired
    assert result.output_mode == "json_schema"
    assert result.data["biomarkers"][1]["threshold"] == 50.0
    assert result.data["biomarkers"][1]["assay_or_method"] is None
    statuses = [c.status for c in result.evidence_checks]
    assert statuses == ["exact", "normalized"]  # LVEF quote dropped the "\<" escape
    lvef = result.evidence_checks[1]
    assert lvef.start is not None and lvef.end is not None
    assert (
        record.eligibility_text[lvef.start : lvef.end]
        == "LVEF \\< 50% within 28 d prior to randomization"
    )

    system, user = client.calls[0].messages
    assert "Task-specific instructions" in system.content
    assert '"biomarkers"' in system.content
    assert "Eligibility criteria (verbatim)" in user.content
    assert "Trastuzumab deruxtecan" in user.content
    repair = client.calls[1].messages[-1].content
    assert "biomarkers.0.kind" in repair


def test_custom_extraction_gives_up_after_one_repair() -> None:
    record = study_to_record(load_study("NCT03529110"))
    spec = load_schema_file(SCHEMA)
    client = FakeLLMClient(responses=['{"biomarkers": "none"}', '{"biomarkers": "still none"}'])
    with pytest.raises(StructuredOutputError):
        asyncio.run(CustomExtractor(client).extract(record, spec))
    assert len(client.calls) == 2
