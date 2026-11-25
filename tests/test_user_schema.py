import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ctgov_extract.errors import SchemaDefinitionError
from ctgov_extract.user_schema import (
    build_model,
    list_example_schemas,
    load_schema_file,
    parse_schema_spec,
    parse_schema_text,
)
from tests.conftest import REPO_ROOT

SCHEMA_YAML = """
name: cognitive scales
description: Scales with cut-offs.
fields:
  - name: scales
    type: list[object]
    required: true
    fields:
      - {name: scale, type: string, required: true}
      - {name: operator, type: enum, enum: ["<", ">=", "between"]}
      - {name: min_value, type: number}
      - {name: count, type: integer}
      - {name: evidence, type: string, required: true}
  - name: requires_study_partner
    type: boolean
  - name: stages
    type: list[string]
"""


def test_build_model_and_validate() -> None:
    model = build_model(parse_schema_text(SCHEMA_YAML))
    assert model.__name__ == "CognitiveScales"
    value = model.model_validate(
        {
            "scales": [
                {"scale": "MMSE", "operator": ">=", "min_value": 22, "evidence": "MMSE \\>=22"},
            ],
            "stages": ["MCI"],
        }
    )
    dumped = value.model_dump()
    assert dumped["scales"][0]["min_value"] == 22.0
    assert dumped["scales"][0]["count"] is None
    assert dumped["requires_study_partner"] is None


def test_generated_json_schema() -> None:
    schema = build_model(parse_schema_text(SCHEMA_YAML)).model_json_schema()
    assert schema["required"] == ["scales"]
    assert schema["additionalProperties"] is False
    item_ref = schema["properties"]["scales"]["items"]["$ref"].split("/")[-1]
    item = schema["$defs"][item_ref]
    assert set(item["required"]) == {"scale", "evidence"}
    operator = item["properties"]["operator"]["anyOf"][0]
    assert operator["enum"] == ["<", ">=", "between"]


@pytest.mark.parametrize(
    "bad",
    [
        {"scales": [{"scale": "MMSE"}]},  # missing required evidence
        {"scales": [{"scale": "MMSE", "evidence": "x", "operator": "approx"}]},  # not in enum
        {"scales": [], "extra_field": 1},  # extra keys are rejected
        {"requires_study_partner": True},  # required list missing
    ],
)
def test_validation_errors(bad: dict[str, object]) -> None:
    model = build_model(parse_schema_text(SCHEMA_YAML))
    with pytest.raises(ValidationError):
        model.model_validate(bad)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "x", "fields": []}, "at least 1"),
        ({"name": "x", "fields": [{"name": "a", "type": "enum"}]}, "non-empty 'enum'"),
        ({"name": "x", "fields": [{"name": "a", "type": "string", "enum": ["b"]}]}, "only allowed"),
        ({"name": "x", "fields": [{"name": "a", "type": "list[object]"}]}, "nested 'fields'"),
        (
            {"name": "x", "fields": [{"name": "a", "type": "string", "fields": []}]},
            r"need type list\[object\]",
        ),
        ({"name": "x", "fields": [{"name": "1abc", "type": "string"}]}, "invalid field name"),
        ({"name": "x", "fields": [{"name": "model_dump", "type": "string"}]}, "reserved"),
        ({"name": "x", "fields": [{"name": "a", "type": "date"}]}, "type"),
        (
            {
                "name": "x",
                "fields": [{"name": "a", "type": "string"}, {"name": "a", "type": "string"}],
            },
            "duplicate field names",
        ),
        (
            {"name": "x", "fields": [{"name": "a", "type": "enum", "enum": ["b", "b"]}]},
            "duplicate enum",
        ),
        ({"name": "x", "fields": [{"name": "a", "type": "string", "typo": 1}]}, "Extra inputs"),
    ],
)
def test_bad_schema_definitions(spec: dict[str, object], message: str) -> None:
    with pytest.raises(SchemaDefinitionError, match=message):
        parse_schema_spec(spec)


def test_parse_errors() -> None:
    with pytest.raises(SchemaDefinitionError, match="YAML"):
        parse_schema_text("name: [unclosed")
    with pytest.raises(SchemaDefinitionError, match="mapping"):
        parse_schema_text("- just a list")


def test_json_schema_file(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"name": "s", "fields": [{"name": "a", "type": "integer"}]}))
    assert load_schema_file(path).fields[0].type == "integer"
    path.write_text("{oops")
    with pytest.raises(SchemaDefinitionError, match="not valid JSON"):
        load_schema_file(path)
    with pytest.raises(SchemaDefinitionError, match="not found"):
        load_schema_file(tmp_path / "missing.yaml")


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "schemas").glob("*.yaml")))
def test_example_schemas_build(path: Path) -> None:
    spec = load_schema_file(path)
    schema = build_model(spec).model_json_schema()
    assert schema["type"] == "object"


def test_list_example_schemas() -> None:
    examples = list_example_schemas()
    assert {
        "oncology_biomarkers.yaml",
        "neuro_cognitive_scores.yaml",
        "simple_overview.yaml",
    } <= set(examples)
