"""Shared fixtures. Everything here is offline."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ctgov_extract.ctgov import CtgovClient
from ctgov_extract.llm.base import ChatMessage
from ctgov_extract.llm.fake import FakeLLMClient

FIXTURES = Path(__file__).parent / "fixtures"
STUDIES = FIXTURES / "studies"
FIXTURE_IDS = ["NCT03529110", "NCT03887455", "NCT01844505", "NCT00000620"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_study(nct_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((STUDIES / f"{nct_id}.json").read_text(encoding="utf-8"))
    return data


def eligibility(nct_id: str) -> str:
    text: str = load_study(nct_id)["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    return text


@pytest.fixture
def study() -> Callable[[str], dict[str, Any]]:
    return load_study


def fixture_transport(calls: list[str] | None = None) -> httpx.MockTransport:
    """Serve the saved fixtures as if they came from the API; unknown IDs get a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        match = re.search(r"/studies/(NCT\d{8})$", request.url.path)
        if match and (STUDIES / f"{match.group(1)}.json").is_file():
            return httpx.Response(200, json=load_study(match.group(1)))
        return httpx.Response(404, text=f"NCT number {request.url.path} not found")

    return httpx.MockTransport(handler)


async def _no_sleep(_: float) -> None:
    return None


def make_ctgov_client(calls: list[str] | None = None, **kwargs: Any) -> CtgovClient:
    return CtgovClient(transport=fixture_transport(calls), sleep=_no_sleep, **kwargs)


def criterion_text_from(messages: list[ChatMessage]) -> str:
    """Pull the criterion text out of the atomization prompt."""
    first_user = next(m for m in messages if m.role == "user")  # repair rounds append more
    match = re.search(r"<<<\n(.*)\n>>>", first_user.content, re.DOTALL)
    assert match, "prompt has no <<< >>> block"
    return match.group(1)


def echo_responder(messages: list[ChatMessage], _: dict[str, Any] | None) -> str:
    """One atomic item per criterion, quoting the first five words verbatim."""
    text = criterion_text_from(messages)
    quote = " ".join(text.split()[:5])
    return json.dumps(
        {
            "items": [
                {
                    "category": "other",
                    "concept": quote.lower(),
                    "operator": "present",
                    "negated": False,
                    "is_biomarker": False,
                    "source_text": quote,
                }
            ]
        }
    )


@pytest.fixture
def echo_llm() -> FakeLLMClient:
    return FakeLLMClient(responder=echo_responder)
