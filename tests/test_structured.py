import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from ctgov_extract.errors import StructuredOutputError, UnsupportedResponseFormatError
from ctgov_extract.llm.base import ChatMessage
from ctgov_extract.llm.fake import FakeLLMClient
from ctgov_extract.llm.structured import StructuredGenerator, extract_json


class Answer(BaseModel):
    name: str
    score: int


MESSAGES = [ChatMessage("system", "s"), ChatMessage("user", "u")]


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here you go:\n```\n{"a": 1}\n```\nAnything else?',
        'The answer is {"a": 1} as requested.',
        'noise {broken then {"a": 1}',
        '<think>maybe {"a": 2}? no.</think>\n{"a": 1}',
    ],
)
def test_extract_json(text: str) -> None:
    assert extract_json(text) == {"a": 1}


def test_extract_json_failure() -> None:
    with pytest.raises(ValueError, match="No JSON"):
        extract_json("no json here")


def test_extract_json_ignores_unclosed_thinking_block() -> None:
    with pytest.raises(ValueError, match="No JSON"):
        extract_json('<think>the answer could be {"a": 1} but')


def test_default_mode_sends_schema_in_prompt_only() -> None:
    client = FakeLLMClient(responses=['{"name": "MMSE", "score": 22}'])
    result = asyncio.run(StructuredGenerator(client).generate(MESSAGES, Answer))
    assert result.value == Answer(name="MMSE", score=22)
    assert result.mode == "text"
    assert client.calls[0].response_format is None


def test_json_schema_mode_first() -> None:
    client = FakeLLMClient(responses=['{"name": "MMSE", "score": 22}'])
    result = asyncio.run(
        StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
    )
    assert result.value == Answer(name="MMSE", score=22)
    assert result.mode == "json_schema"
    assert result.attempts == 1 and not result.repaired
    fmt = client.calls[0].response_format
    assert fmt is not None and fmt["type"] == "json_schema"
    assert fmt["json_schema"]["schema"]["properties"]["score"]["type"] == "integer"
    assert result.usage.calls == 1 and result.usage.total_tokens > 0


def test_falls_back_to_json_object_then_text_and_remembers() -> None:
    client = FakeLLMClient(
        responses=['{"name": "a", "score": 1}', '{"name": "b", "score": 2}'],
        unsupported_formats=["json_schema", "json_object"],
    )
    gen = StructuredGenerator(client, start_mode="json_schema")
    first = asyncio.run(gen.generate(MESSAGES, Answer))
    assert first.mode == "text"
    assert client.calls[0].response_format is None
    second = asyncio.run(gen.generate(MESSAGES, Answer))
    assert second.mode == "text" and second.value.name == "b"
    assert len(client.calls) == 2  # no repeated failed attempts after the fallback


def test_fallback_stops_at_json_object_when_supported() -> None:
    client = FakeLLMClient(
        responses=['{"name": "a", "score": 1}'], unsupported_formats=["json_schema"]
    )
    result = asyncio.run(
        StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
    )
    assert result.mode == "json_object"
    assert client.calls[0].response_format == {"type": "json_object"}


def test_repair_round_sends_errors_back() -> None:
    client = FakeLLMClient(
        responses=['{"name": "MMSE", "score": "twenty-two"}', '{"name": "MMSE", "score": 22}']
    )
    result = asyncio.run(
        StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
    )
    assert result.repaired and result.attempts == 2
    assert result.value.score == 22
    assert result.usage.calls == 2
    repair_messages = client.calls[1].messages
    assert repair_messages[-2].role == "assistant"
    assert "twenty-two" in repair_messages[-2].content
    assert repair_messages[-1].role == "user"
    assert "score" in repair_messages[-1].content
    assert "valid integer" in repair_messages[-1].content


def test_repair_round_handles_non_json() -> None:
    client = FakeLLMClient(responses=["I cannot do that.", '{"name": "x", "score": 0}'])
    result = asyncio.run(
        StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
    )
    assert result.repaired
    assert "No JSON object found" in client.calls[1].messages[-1].content


def test_second_failure_raises_with_details() -> None:
    client = FakeLLMClient(responses=['{"name": 1}', '{"name": "x"}'])
    with pytest.raises(StructuredOutputError) as info:
        asyncio.run(
            StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
        )
    assert info.value.raw_output == '{"name": "x"}'
    assert any("score" in e for e in info.value.errors)


def test_all_modes_unsupported_raises() -> None:
    def responder(_: list[ChatMessage], fmt: dict[str, Any] | None) -> str:
        raise UnsupportedResponseFormatError("text mode rejected too")

    client = FakeLLMClient(responder=responder, unsupported_formats=["json_schema", "json_object"])
    with pytest.raises(UnsupportedResponseFormatError):
        asyncio.run(
            StructuredGenerator(client, start_mode="json_schema").generate(MESSAGES, Answer)
        )


def test_custom_schema_and_name_are_used() -> None:
    client = FakeLLMClient(responses=[json.dumps({"name": "n", "score": 1})])
    schema = {"type": "object", "title": "custom"}
    asyncio.run(
        StructuredGenerator(client, start_mode="json_schema").generate(
            MESSAGES, Answer, name="my schema!", schema=schema
        )
    )
    fmt = client.calls[0].response_format
    assert fmt is not None
    assert fmt["json_schema"]["name"] == "my_schema_"
    assert fmt["json_schema"]["schema"] == schema
