"""User-defined schema extraction over a whole trial (structured fields + eligibility text)."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from ctgov_extract.evidence import EvidenceCheck, check_evidence_fields
from ctgov_extract.llm.base import ChatMessage, LLMClient
from ctgov_extract.llm.structured import StructuredGenerator
from ctgov_extract.models import TrialRecord, Usage
from ctgov_extract.prompts import custom_system_prompt, trial_context
from ctgov_extract.user_schema import SchemaSpec, build_model


class CustomExtractionResult(BaseModel):
    nct_id: str
    schema_name: str
    model: str | None = None
    data: dict[str, Any]
    evidence_checks: list[EvidenceCheck] = Field(default_factory=list)
    output_mode: str
    repaired: bool = False
    usage: Usage = Field(default_factory=Usage)
    elapsed_s: float = 0.0


def _other_fields_text(record: TrialRecord) -> str:
    parts: list[str] = [record.brief_title or "", record.official_title or ""]
    parts += record.conditions + record.keywords
    parts += [i.name for i in record.interventions]
    parts += [o.measure for o in record.primary_outcomes + record.secondary_outcomes]
    parts += [o.time_frame or "" for o in record.primary_outcomes + record.secondary_outcomes]
    return "\n".join(p for p in parts if p)


class CustomExtractor:
    def __init__(self, client: LLMClient, *, generator: StructuredGenerator | None = None) -> None:
        self.client = client
        self.generator = generator or StructuredGenerator(client)

    async def extract(self, record: TrialRecord, spec: SchemaSpec) -> CustomExtractionResult:
        started = time.perf_counter()
        output_model = build_model(spec)
        schema = output_model.model_json_schema()
        messages = [
            ChatMessage("system", custom_system_prompt(schema, spec.instructions)),
            ChatMessage("user", trial_context(record)),
        ]
        result = await self.generator.generate(
            messages, output_model, name=spec.name, schema=schema
        )
        data = result.value.model_dump(mode="json")
        return CustomExtractionResult(
            nct_id=record.nct_id,
            schema_name=spec.name,
            model=result.model or self.client.model_name,
            data=data,
            evidence_checks=check_evidence_fields(
                data, record.eligibility_text, _other_fields_text(record)
            ),
            output_mode=result.mode,
            repaired=result.repaired,
            usage=result.usage,
            elapsed_s=round(time.perf_counter() - started, 3),
        )
