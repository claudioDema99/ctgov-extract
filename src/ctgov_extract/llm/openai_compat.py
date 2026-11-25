"""Client for any OpenAI-compatible ``/chat/completions`` endpoint (vLLM, Ollama, OpenAI...)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Self

import httpx

from ctgov_extract.config import Settings
from ctgov_extract.errors import LLMError, LLMNotConfiguredError, UnsupportedResponseFormatError
from ctgov_extract.http_retry import RETRYABLE_STATUS, Sleep, default_sleep, retry_delay
from ctgov_extract.llm.base import ChatMessage, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OpenAICompatClient:
    """Async chat client with a concurrency limit, timeouts and retries on 429/5xx."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
        max_concurrency: int = 4,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = default_sleep,
    ) -> None:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        if not settings.llm_base_url or not settings.llm_model:
            raise LLMNotConfiguredError(
                "No LLM backend configured. Set LLM_BASE_URL and LLM_MODEL (see .env.example)."
            )
        return cls(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            extra_body=settings.llm_extra_body,
            timeout_s=settings.llm_timeout_s,
            max_concurrency=settings.llm_max_concurrency,
            max_retries=settings.llm_max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Backend-specific fields first, so they can't override the core ones.
        payload: dict[str, Any] = {
            **self.extra_body,
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        async with self._semaphore:
            started = time.perf_counter()
            data = await self._post_with_retries(payload, response_format is not None)
            latency = time.perf_counter() - started

        return self._parse(data, latency)

    async def _post_with_retries(self, payload: dict[str, Any], has_format: bool) -> dict[str, Any]:
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await self._http.post("/chat/completions", json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status < 400:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise LLMError("LLM backend returned invalid JSON.") from exc
                    if not isinstance(body, dict):
                        raise LLMError("LLM backend returned an unexpected payload.")
                    return body
                if has_format and status in (400, 422):
                    fmt_type = payload["response_format"].get("type")
                    raise UnsupportedResponseFormatError(
                        f"Backend rejected response_format {fmt_type!r} "
                        f"(HTTP {status}): {response.text[:300]}"
                    )
                if status not in RETRYABLE_STATUS:
                    raise LLMError(f"LLM backend returned HTTP {status}: {response.text[:500]}")
                last_error = f"HTTP {status}"
            if attempt < self.max_retries:
                delay = retry_delay(attempt, response)
                logger.info("LLM call failed (%s); retrying in %.1fs", last_error, delay)
                await self._sleep(delay)
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_error}")

    def _parse(self, data: dict[str, Any], latency: float) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response has no choices[0].message.content.") from exc
        if not isinstance(content, str):
            raise LLMError("LLM response content is not a string.")

        usage = None
        raw_usage = data.get("usage")
        if isinstance(raw_usage, dict):
            usage = TokenUsage(
                prompt_tokens=int(raw_usage.get("prompt_tokens") or 0),
                completion_tokens=int(raw_usage.get("completion_tokens") or 0),
                total_tokens=int(raw_usage.get("total_tokens") or 0),
            )
            logger.debug(
                "LLM usage: prompt=%d completion=%d latency=%.2fs",
                usage.prompt_tokens,
                usage.completion_tokens,
                latency,
            )
        model = data.get("model")
        return LLMResponse(
            content=content,
            model=model if isinstance(model, str) else self._model,
            usage=usage,
            latency_s=latency,
            raw=data,
        )
