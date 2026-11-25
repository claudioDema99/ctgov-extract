import asyncio
import json

import httpx
import pytest

from ctgov_extract.config import Settings
from ctgov_extract.errors import LLMError, LLMNotConfiguredError, UnsupportedResponseFormatError
from ctgov_extract.llm.base import ChatMessage
from ctgov_extract.llm.openai_compat import OpenAICompatClient


async def _no_sleep(_: float) -> None:
    return None


def _completion(content: str, usage: bool = True) -> dict[str, object]:
    body: dict[str, object] = {
        "model": "served-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return body


def _client(handler: httpx.MockTransport, **kwargs: object) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://llm.test/v1/",
        model="test-model",
        transport=handler,
        sleep=_no_sleep,
        **kwargs,  # type: ignore[arg-type]
    )


def test_request_payload_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion('{"ok": true}'))

    async def go() -> None:
        async with _client(httpx.MockTransport(handler), api_key="secret", max_tokens=256) as c:
            response = await c.chat(
                [ChatMessage("user", "hi")], response_format={"type": "json_object"}
            )
            assert response.content == '{"ok": true}'
            assert response.model == "served-model"
            assert response.usage is not None and response.usage.total_tokens == 15

    asyncio.run(go())
    request = seen[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    payload = json.loads(request.content)
    assert payload == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }


def test_no_api_key_means_no_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("x", usage=False))

    async def go() -> None:
        async with _client(httpx.MockTransport(handler)) as c:
            response = await c.chat([ChatMessage("user", "hi")])
            assert response.usage is None

    asyncio.run(go())
    assert "authorization" not in seen[0].headers
    assert "response_format" not in json.loads(seen[0].content)


def test_retries_then_succeeds() -> None:
    statuses = [429, 502]

    def handler(request: httpx.Request) -> httpx.Response:
        if statuses:
            return httpx.Response(statuses.pop(0))
        return httpx.Response(200, json=_completion("done"))

    async def go() -> str:
        async with _client(httpx.MockTransport(handler)) as c:
            return (await c.chat([ChatMessage("user", "hi")])).content

    assert asyncio.run(go()) == "done"


def test_rejected_response_format_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "response_format not supported"})

    async def go() -> None:
        async with _client(httpx.MockTransport(handler)) as c:
            await c.chat([ChatMessage("user", "hi")], response_format={"type": "json_schema"})

    with pytest.raises(UnsupportedResponseFormatError, match="json_schema"):
        asyncio.run(go())


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, text="bad key"), "HTTP 401"),
        (httpx.Response(200, json={"choices": []}), "no choices"),
        (httpx.Response(200, text="not json"), "invalid JSON"),
        (httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), "not a string"),
    ],
)
def test_errors(response: httpx.Response, message: str) -> None:
    async def go() -> None:
        async with _client(httpx.MockTransport(lambda r: response), max_retries=0) as c:
            await c.chat([ChatMessage("user", "hi")])

    with pytest.raises(LLMError, match=message):
        asyncio.run(go())


def test_timeouts_exhaust_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async def go() -> None:
        async with _client(httpx.MockTransport(handler), max_retries=1) as c:
            await c.chat([ChatMessage("user", "hi")])

    with pytest.raises(LLMError, match="after 2 attempts"):
        asyncio.run(go())


def test_concurrency_limit() -> None:
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=_completion("x"))

    async def go() -> None:
        async with _client(httpx.MockTransport(handler), max_concurrency=2) as c:
            await asyncio.gather(*(c.chat([ChatMessage("user", str(i))]) for i in range(6)))

    asyncio.run(go())
    assert peak == 2


def test_from_settings() -> None:
    with pytest.raises(LLMNotConfiguredError):
        OpenAICompatClient.from_settings(Settings())
    settings = Settings.from_env(
        {
            "LLM_BASE_URL": "http://x/v1",
            "LLM_MODEL": "m",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "64",
        }
    )
    client = OpenAICompatClient.from_settings(settings)
    assert client.model_name == "m"
    assert client.temperature == 0.2
    assert client.max_tokens == 64
    asyncio.run(client.aclose())


def test_extra_body_is_merged_without_overriding_core_fields() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("{}"))

    settings = Settings.from_env(
        {
            "LLM_BASE_URL": "http://llm.test/v1",
            "LLM_MODEL": "m",
            "LLM_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}, "model": "x"}',
        }
    )
    assert settings.llm_extra_body is not None

    async def go() -> None:
        async with _client(httpx.MockTransport(handler), extra_body=settings.llm_extra_body) as c:
            await c.chat([ChatMessage("user", "hi")])

    asyncio.run(go())
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen[0]["model"] == "test-model"


def test_extra_body_must_be_a_json_object() -> None:
    assert Settings.from_env({"LLM_EXTRA_BODY": ""}).llm_extra_body is None
    with pytest.raises(ValueError, match="JSON object"):
        Settings.from_env({"LLM_EXTRA_BODY": "[1, 2]"})


def test_settings_cache_dir(tmp_path: object) -> None:
    assert Settings.from_env({"CTGOV_CACHE_DIR": "off"}).cache_dir is None
    assert str(Settings.from_env({"CTGOV_CACHE_DIR": "/tmp/x"}).cache_dir) == "/tmp/x"
    default = Settings.from_env({"XDG_CACHE_HOME": "/c"}).cache_dir
    assert str(default) == "/c/ctgov-extract"
    assert not Settings.from_env({}).llm_configured
