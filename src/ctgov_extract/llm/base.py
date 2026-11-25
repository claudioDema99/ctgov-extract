"""LLM client interface. Everything else depends on this, never on a concrete backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str | None = None
    usage: TokenUsage | None = None
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat-completions interface.

    ``response_format`` follows the OpenAI convention, e.g.
    ``{"type": "json_object"}`` or
    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}``.
    Implementations raise :class:`~ctgov_extract.errors.UnsupportedResponseFormatError`
    when the backend rejects the format, so callers can fall back.
    """

    @property
    def model_name(self) -> str | None: ...

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
