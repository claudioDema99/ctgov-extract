"""User-defined extraction schemas (YAML or JSON) turned into Pydantic models at runtime.

Example (YAML)::

    name: cognitive_scales
    description: Cognitive and functional scales used for eligibility.
    instructions: Only include scales that appear in the eligibility criteria.
    fields:
      - name: scales
        type: list[object]
        required: true
        fields:
          - {name: scale, type: string, required: true}
          - {name: operator, type: enum, enum: ["<", "<=", ">", ">=", "=", "between"]}
          - {name: min_value, type: number}
          - {name: max_value, type: number}
          - {name: evidence, type: string, description: Verbatim quote}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator
from pydantic import model_validator as pyd_model_validator

from ctgov_extract.errors import SchemaDefinitionError

FieldType = Literal[
    "string", "number", "integer", "boolean", "enum", "list[string]", "list[object]"
]

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_RESERVED = frozenset(name for name in dir(BaseModel) if not name.startswith("_"))


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: FieldType
    description: str | None = None
    required: bool = False
    enum: list[str] | None = None
    fields: list[FieldSpec] | None = None

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(
                f"invalid field name {value!r}: use letters, digits and underscores, "
                "starting with a letter"
            )
        if value in _RESERVED or value.startswith("model_"):
            raise ValueError(f"field name {value!r} is reserved; please rename it")
        return value

    @pyd_model_validator(mode="after")
    def _consistent(self) -> FieldSpec:
        if self.type == "enum":
            if not self.enum:
                raise ValueError(f"field {self.name!r}: type enum needs a non-empty 'enum' list")
            if len(set(self.enum)) != len(self.enum):
                raise ValueError(f"field {self.name!r}: duplicate enum values")
        elif self.enum is not None:
            raise ValueError(f"field {self.name!r}: 'enum' is only allowed with type enum")
        if self.type == "list[object]":
            if not self.fields:
                raise ValueError(f"field {self.name!r}: type list[object] needs nested 'fields'")
            _check_unique(self.fields, self.name)
        elif self.fields is not None:
            raise ValueError(f"field {self.name!r}: nested 'fields' need type list[object]")
        return self


class SchemaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    description: str | None = None
    instructions: str | None = None
    fields: list[FieldSpec] = Field(min_length=1)

    @pyd_model_validator(mode="after")
    def _unique(self) -> SchemaSpec:
        _check_unique(self.fields, self.name)
        return self

    @property
    def list_fields(self) -> list[FieldSpec]:
        return [f for f in self.fields if f.type == "list[object]"]


def _check_unique(fields: list[FieldSpec], owner: str) -> None:
    names = [f.name for f in fields]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{owner!r}: duplicate field names {dupes}")


def _class_name(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name)
    camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not camel or not camel[0].isalpha():
        camel = f"Schema{camel}"
    return camel


def _python_type(spec: FieldSpec, owner: str) -> Any:
    match spec.type:
        case "string":
            return str
        case "number":
            return float
        case "integer":
            return int
        case "boolean":
            return bool
        case "enum":
            assert spec.enum is not None
            return Literal[tuple(spec.enum)]
        case "list[string]":
            return list[str]
        case "list[object]":
            assert spec.fields is not None
            item_model = _build(f"{owner}{_class_name(spec.name)}Item", spec.fields)
            return list[item_model]  # type: ignore[valid-type]
    raise SchemaDefinitionError(f"Unsupported field type {spec.type!r}")  # pragma: no cover


def _build(class_name: str, fields: list[FieldSpec], doc: str | None = None) -> type[BaseModel]:
    definitions: dict[str, Any] = {}
    for spec in fields:
        py_type = _python_type(spec, class_name)
        if spec.required:
            definitions[spec.name] = (py_type, Field(..., description=spec.description))
        else:
            definitions[spec.name] = (py_type | None, Field(None, description=spec.description))
    model: type[BaseModel] = create_model(
        class_name,
        __config__=ConfigDict(extra="forbid"),
        __doc__=doc,
        **definitions,
    )
    return model


def build_model(spec: SchemaSpec) -> type[BaseModel]:
    """Create a Pydantic model class for ``spec`` (nested list[object] fields become submodels)."""
    return _build(_class_name(spec.name), spec.fields, spec.description)


def parse_schema_spec(data: Any) -> SchemaSpec:
    try:
        return SchemaSpec.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
            for e in exc.errors(include_url=False)
        )
        raise SchemaDefinitionError(f"Invalid extraction schema: {problems}") from exc


def parse_schema_text(text: str) -> SchemaSpec:
    """Parse a schema given as YAML or JSON text (JSON is valid YAML)."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SchemaDefinitionError(f"Schema is not valid YAML/JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaDefinitionError("Schema must be a mapping with 'name' and 'fields'.")
    return parse_schema_spec(data)


def load_schema_file(path: Path) -> SchemaSpec:
    if not path.is_file():
        raise SchemaDefinitionError(f"Schema file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            return parse_schema_spec(json.loads(text))
        except json.JSONDecodeError as exc:
            raise SchemaDefinitionError(f"{path} is not valid JSON: {exc}") from exc
    return parse_schema_text(text)


def example_schema_dir() -> Path | None:
    """Location of the bundled example schemas (installed wheel or source checkout)."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "example_schemas", here.parents[1] / "schemas"):
        if candidate.is_dir():
            return candidate
    return None


def list_example_schemas() -> dict[str, str]:
    """Map of example file name to its raw text."""
    directory = example_schema_dir()
    if directory is None:
        return {}
    return {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(directory.iterdir())
        if p.suffix.lower() in {".yaml", ".yml", ".json"}
    }
