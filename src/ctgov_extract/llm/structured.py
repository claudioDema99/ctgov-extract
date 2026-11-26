"""Structured output: JSON-schema mode with fallbacks, Pydantic validation and one repair round."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from ctgov_extract.config import OutputMode
from ctgov_extract.errors import StructuredOutputError, UnsupportedResponseFormatError
from ctgov_extract.llm.base import ChatMessage, LLMClient, LLMResponse
from ctgov_extract.models import Usage

logger = logging.getLogger(__name__)

# Fallback order, from the most constrained mode to plain text.
MODES: tuple[OutputMode, ...] = ("json_schema", "json_object", "text")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Reasoning models (Qwen3, DeepSeek-R1...) may prefix the answer with a thinking block
# when the server has no reasoning parser. An unclosed block means a truncated answer.
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse the first JSON value found in ``text``.

    Handles bare JSON, JSON inside markdown code fences, and JSON surrounded by prose.
    Raises ``ValueError`` if nothing parseable is found.
    """
    text = _THINK.sub("", text)
    stripped = text.strip()
    candidates = [stripped]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            return value
    raise ValueError("No JSON object found in model output.")


def format_validation_errors(exc: ValidationError) -> list[str]:
    errors = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        errors.append(f"{loc}: {err['msg']}")
    return errors


def _response_format(mode: OutputMode, name: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    if mode == "json_schema":
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64] or "output"
        return {
            "type": "json_schema",
            "json_schema": {"name": safe_name, "schema": schema, "strict": False},
        }
    if mode == "json_object":
        return {"type": "json_object"}
    return None


@dataclass
class StructuredResult[T: BaseModel]:
    value: T
    mode: OutputMode
    attempts: int
    repaired: bool
    usage: Usage
    raw_output: str
    model: str | None = None
    errors: list[str] = field(default_factory=list)


class StructuredGenerator:
    """Ask a model for JSON matching a Pydantic model.

    The schema is always in the prompt. By default (``text``) nothing else is sent,
    which works with any chat endpoint. ``json_object`` and ``json_schema`` also send
    ``response_format`` so the server can constrain decoding; if the backend
    rejects it, the generator falls back towards ``text`` and remembers the
    working mode for later calls.

    If the output does not parse or validate, the errors are sent back to the
    model once ("repair round"). A second failure raises
    :class:`StructuredOutputError`.
    """

    def __init__(self, client: LLMClient, *, start_mode: OutputMode = "text") -> None:
        self.client = client
        self._mode_index = MODES.index(start_mode)

    @property
    def mode(self) -> OutputMode:
        return MODES[self._mode_index]

    async def _chat(
        self, messages: list[ChatMessage], name: str, schema: dict[str, Any]
    ) -> tuple[LLMResponse, OutputMode]:
        while True:
            mode = MODES[self._mode_index]
            try:
                response = await self.client.chat(
                    messages, response_format=_response_format(mode, name, schema)
                )
            except UnsupportedResponseFormatError as exc:
                if self._mode_index == len(MODES) - 1:
                    raise
                logger.info("Structured output mode %s unsupported (%s); falling back", mode, exc)
                self._mode_index += 1
                continue
            return response, mode

    async def generate[T: BaseModel](
        self,
        messages: list[ChatMessage],
        output_model: type[T],
        *,
        name: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> StructuredResult[T]:
        schema = schema or output_model.model_json_schema()
        name = name or output_model.__name__
        usage = Usage()
        conversation = list(messages)
        errors: list[str] = []
        raw = ""
        mode: OutputMode = self.mode
        model_name: str | None = None

        for attempt in (1, 2):
            response, mode = await self._chat(conversation, name, schema)
            _add_usage(usage, response)
            model_name = response.model
            raw = response.content
            try:
                value = output_model.model_validate(extract_json(raw))
            except ValueError as exc:  # ValidationError is a ValueError subclass
                errors = (
                    format_validation_errors(exc)
                    if isinstance(exc, ValidationError)
                    else [str(exc)]
                )
                logger.info("Attempt %d produced invalid output: %s", attempt, "; ".join(errors))
                if attempt == 1:
                    conversation = [
                        *conversation,
                        ChatMessage("assistant", raw),
                        ChatMessage("user", repair_prompt(errors, schema)),
                    ]
                continue
            return StructuredResult(
                value=value,
                mode=mode,
                attempts=attempt,
                repaired=attempt > 1,
                usage=usage,
                raw_output=raw,
                model=model_name,
                errors=errors,
            )

        raise StructuredOutputError(
            f"Model output did not match the schema after a repair round: {'; '.join(errors)}",
            raw_output=raw,
            errors=errors,
            usage=usage,
        )


def repair_prompt(errors: list[str], schema: dict[str, Any]) -> str:
    bullet_list = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "Your previous answer could not be used. Problems found:\n"
        f"{bullet_list}\n\n"
        "Return the corrected answer as a single JSON object that validates against this "
        "JSON schema. Output only JSON, no explanations.\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def _add_usage(usage: Usage, response: LLMResponse) -> None:
    usage.calls += 1
    usage.latency_s += response.latency_s
    if response.usage is not None:
        usage.prompt_tokens += response.usage.prompt_tokens
        usage.completion_tokens += response.usage.completion_tokens
        usage.total_tokens += response.usage.total_tokens
