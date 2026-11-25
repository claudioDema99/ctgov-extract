import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ctgov_extract.ctgov import CtgovClient, StudyCache, load_study_file, resolve_trial
from ctgov_extract.errors import CtgovApiError, InvalidTrialIdError, StudyNotFoundError
from tests.conftest import STUDIES, load_study, make_ctgov_client


async def _no_sleep(_: float) -> None:
    return None


def test_fetch_record_by_url() -> None:
    calls: list[str] = []

    async def go() -> None:
        async with make_ctgov_client(calls) as client:
            record = await client.fetch_record(
                "https://clinicaltrials.gov/study/nct03887455?tab=table"
            )
            assert record.nct_id == "NCT03887455"

    asyncio.run(go())
    assert calls == ["/api/v2/studies/NCT03887455"]


def test_unknown_id_gives_clear_error() -> None:
    async def go() -> None:
        async with make_ctgov_client() as client:
            await client.fetch_study("NCT99999999")

    with pytest.raises(StudyNotFoundError, match="NCT99999999"):
        asyncio.run(go())


def test_invalid_id_is_rejected_before_any_request() -> None:
    calls: list[str] = []

    async def go() -> None:
        async with make_ctgov_client(calls) as client:
            await client.fetch_study("not-a-trial")

    with pytest.raises(InvalidTrialIdError):
        asyncio.run(go())
    assert calls == []


def test_cache_avoids_second_request(tmp_path: Path) -> None:
    calls: list[str] = []

    async def go() -> None:
        async with make_ctgov_client(calls, cache=StudyCache(tmp_path)) as client:
            first = await client.fetch_study("NCT00000620")
            second = await client.fetch_study("nct00000620")
            assert first == second

    asyncio.run(go())
    assert len(calls) == 1
    assert (tmp_path / "studies" / "NCT00000620.json").is_file()


def test_cache_ttl_and_corrupt_entries(tmp_path: Path) -> None:
    cache = StudyCache(tmp_path, ttl_s=-1)  # everything is expired
    cache.put("NCT00000620", {"protocolSection": {}})
    assert cache.get("NCT00000620") is None
    fresh = StudyCache(tmp_path)
    assert fresh.get("NCT00000620") == {"protocolSection": {}}
    (tmp_path / "studies" / "NCT00000620.json").write_text("{not json", encoding="utf-8")
    assert fresh.get("NCT00000620") is None


def _flaky_transport(statuses: list[int], calls: list[int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        status = statuses.pop(0) if statuses else 200
        if status == 200:
            return httpx.Response(200, json=load_study("NCT00000620"))
        if status == -1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(status, headers={"Retry-After": "0"})

    return httpx.MockTransport(handler)


def test_retries_on_429_5xx_and_timeouts() -> None:
    calls: list[int] = []
    delays: list[float] = []

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    async def go() -> None:
        client = CtgovClient(
            transport=_flaky_transport([429, 503, -1], calls), sleep=record_sleep, max_retries=3
        )
        async with client:
            study = await client.fetch_study("NCT00000620")
            assert study["protocolSection"]["identificationModule"]["nctId"] == "NCT00000620"

    asyncio.run(go())
    assert len(calls) == 4
    assert len(delays) == 3
    assert delays[0] == 0.0  # Retry-After honoured


def test_gives_up_after_max_retries() -> None:
    calls: list[int] = []

    async def go() -> None:
        client = CtgovClient(
            transport=_flaky_transport([500] * 10, calls), sleep=_no_sleep, max_retries=2
        )
        async with client:
            await client.fetch_study("NCT00000620")

    with pytest.raises(CtgovApiError, match="after 3 attempts"):
        asyncio.run(go())
    assert len(calls) == 3


def test_unexpected_payload() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"hello": "world"}))

    async def go() -> None:
        async with CtgovClient(transport=transport, sleep=_no_sleep) as client:
            await client.fetch_study("NCT00000620")

    with pytest.raises(CtgovApiError, match="protocolSection"):
        asyncio.run(go())


def test_resolve_trial_from_file_and_search_page(tmp_path: Path) -> None:
    page = tmp_path / "page.json"
    page.write_text(json.dumps({"studies": [load_study("NCT01844505")]}), encoding="utf-8")

    async def go() -> None:
        async with make_ctgov_client() as client:
            direct = await resolve_trial(str(STUDIES / "NCT01844505.json"), client)
            wrapped = await resolve_trial(str(page), client)
            assert direct == wrapped
            assert direct.nct_id == "NCT01844505"

    asyncio.run(go())


def test_resolve_trial_rejects_files_when_disallowed() -> None:
    async def go() -> None:
        async with make_ctgov_client() as client:
            await resolve_trial(str(STUDIES / "NCT01844505.json"), client, allow_files=False)

    with pytest.raises(InvalidTrialIdError):
        asyncio.run(go())


def test_load_study_file_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2", encoding="utf-8")
    with pytest.raises(InvalidTrialIdError, match="not valid JSON"):
        load_study_file(bad)
    many = tmp_path / "many.json"
    many.write_text(json.dumps({"studies": [{}, {}]}), encoding="utf-8")
    with pytest.raises(InvalidTrialIdError, match="2 studies"):
        load_study_file(many)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    with pytest.raises(InvalidTrialIdError, match="API v2"):
        load_study_file(other)
