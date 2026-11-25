"""A scripted LLM client for tests and offline demos.

It never calls a network. Responses are produced by a callable (or popped from
a list) so tests can assert on the exact prompts that were sent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from ctgov_extract.errors import UnsupportedResponseFormatError
from ctgov_extract.llm.base import ChatMessage, LLMResponse, TokenUsage


@dataclass
class FakeCall:
    messages: list[ChatMessage]
    response_format: dict[str, Any] | None


Responder = Callable[[list[ChatMessage], dict[str, Any] | None], str]


@dataclass
class FakeLLMClient:
    """Return canned responses.

    Args:
        responder: a function ``(messages, response_format) -> content``, or
        responses: a list of contents returned in order.
        unsupported_formats: ``response_format`` types to reject, to exercise fallbacks.
    """

    responder: Responder | None = None
    responses: list[str] = field(default_factory=list)
    unsupported_formats: Iterable[str] = ()
    model: str = "fake-model"
    calls: list[FakeCall] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if response_format is not None and response_format.get("type") in set(
            self.unsupported_formats
        ):
            raise UnsupportedResponseFormatError(f"fake: {response_format.get('type')} unsupported")
        self.calls.append(FakeCall(list(messages), response_format))
        if self.responder is not None:
            content = self.responder(messages, response_format)
        elif self.responses:
            content = self.responses.pop(0)
        else:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(content.split())
        return LLMResponse(
            content=content,
            model=self.model,
            usage=TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens),
            latency_s=0.0,
        )
