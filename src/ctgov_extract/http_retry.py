"""Small retry policy shared by the ClinicalTrials.gov and LLM clients."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

Sleep = Callable[[float], Awaitable[None]]


def retry_delay(
    attempt: int,
    response: httpx.Response | None = None,
    *,
    base: float = 0.5,
    cap: float = 30.0,
) -> float:
    """Exponential backoff with jitter; honours a numeric ``Retry-After`` header."""
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(cap, max(0.0, float(retry_after)))
            except ValueError:
                pass
    delay: float = base * (2**attempt)
    return float(min(cap, delay + random.uniform(0, base)))


async def default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
