"""Default schema: turn each split criterion into atomic criteria with validated evidence."""

from __future__ import annotations

import asyncio
import logging
import time

from ctgov_extract.config import OutputMode
from ctgov_extract.criteria import split_criteria
from ctgov_extract.errors import LLMError
from ctgov_extract.evidence import validate_evidence
from ctgov_extract.llm.base import ChatMessage, LLMClient
from ctgov_extract.llm.structured import StructuredGenerator
from ctgov_extract.models import (
    AtomicCriterion,
    AtomicDraftList,
    CriteriaResult,
    Criterion,
    CriterionFailure,
    Usage,
)
from ctgov_extract.prompts import atomic_system_prompt, atomic_user_prompt

logger = logging.getLogger(__name__)


class CriteriaExtractor:
    """Split eligibility text and atomize every criterion with an LLM.

    One model call per criterion keeps the evidence scope small (the quote must
    come from that criterion) and makes failures local: a criterion whose output
    cannot be validated is reported in ``failures`` and the rest still succeed.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        generator: StructuredGenerator | None = None,
        output_mode: OutputMode = "text",
    ) -> None:
        self.client = client
        self.generator = generator or StructuredGenerator(client, start_mode=output_mode)
        self._schema = AtomicDraftList.model_json_schema()
        self._system = atomic_system_prompt(self._schema)

    async def extract_text(
        self,
        eligibility_text: str,
        *,
        nct_id: str | None = None,
        title: str | None = None,
        conditions: list[str] | None = None,
    ) -> CriteriaResult:
        started = time.perf_counter()
        split = split_criteria(eligibility_text, nct_id)
        by_id = {c.id: c for c in split.criteria}
        parents = {c.parent_id for c in split.criteria if c.parent_id}

        tasks = [
            self._atomize(
                c,
                title=title,
                conditions=conditions or [],
                parent=by_id.get(c.parent_id) if c.parent_id else None,
                has_children=c.id in parents,
            )
            for c in split.criteria
        ]
        outcomes = await asyncio.gather(*tasks)

        usage = Usage()
        atomic: list[AtomicCriterion] = []
        failures: list[CriterionFailure] = []
        model_name: str | None = self.client.model_name
        for criterion, (items, call_usage, error) in zip(split.criteria, outcomes, strict=True):
            usage.add(call_usage)
            atomic.extend(items)
            if error is not None:
                failures.append(CriterionFailure(criterion_id=criterion.id, error=error))

        return CriteriaResult(
            nct_id=nct_id,
            model=model_name,
            eligibility_text=eligibility_text,
            criteria=split.criteria,
            atomic=atomic,
            failures=failures,
            warnings=split.warnings,
            usage=usage,
            elapsed_s=round(time.perf_counter() - started, 3),
        )

    async def _atomize(
        self,
        criterion: Criterion,
        *,
        title: str | None,
        conditions: list[str],
        parent: Criterion | None,
        has_children: bool,
    ) -> tuple[list[AtomicCriterion], Usage, str | None]:
        messages = [
            ChatMessage("system", self._system),
            ChatMessage(
                "user",
                atomic_user_prompt(
                    criterion,
                    title=title,
                    conditions=conditions,
                    parent=parent,
                    has_children=has_children,
                ),
            ),
        ]
        try:
            result = await self.generator.generate(
                messages, AtomicDraftList, name="atomic_criteria", schema=self._schema
            )
        except LLMError as exc:
            logger.warning("Criterion %s failed: %s", criterion.id, exc)
            spent = getattr(exc, "usage", None)
            return [], spent if isinstance(spent, Usage) else Usage(), str(exc)
        return to_atomic(criterion, result.value), result.usage, None


def to_atomic(criterion: Criterion, drafts: AtomicDraftList) -> list[AtomicCriterion]:
    """Attach ids, kind and validated evidence to the model's drafts."""
    items = []
    for n, draft in enumerate(drafts.items, start=1):
        evidence = validate_evidence(draft.source_text, criterion.text, criterion.start)
        items.append(
            AtomicCriterion(
                id=f"{criterion.id}-a{n}",
                criterion_id=criterion.id,
                kind=criterion.kind,
                category=draft.category,
                concept=draft.concept,
                operator=draft.operator,
                value=draft.value,
                value_max=draft.value_max,
                unit=draft.unit,
                time_window=draft.time_window,
                negated=draft.negated,
                is_biomarker=draft.is_biomarker,
                evidence=evidence,
            )
        )
    return items
