"""ClinicalTrials.gov API v2 client with an on-disk cache and retries."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Self

import httpx

from ctgov_extract import __version__
from ctgov_extract.config import Settings
from ctgov_extract.errors import CtgovApiError, InvalidTrialIdError, StudyNotFoundError
from ctgov_extract.http_retry import RETRYABLE_STATUS, Sleep, default_sleep, retry_delay
from ctgov_extract.ids import parse_nct_id
from ctgov_extract.mapping import study_to_record
from ctgov_extract.models import TrialRecord

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]


class StudyCache:
    """One JSON file per study, keyed by NCT ID, with an optional TTL."""

    def __init__(self, directory: Path, ttl_s: float | None = None) -> None:
        self.directory = directory
        self.ttl_s = ttl_s

    def _path(self, nct_id: str) -> Path:
        return self.directory / "studies" / f"{nct_id}.json"

    def get(self, nct_id: str) -> JsonDict | None:
        path = self._path(nct_id)
        if not path.is_file():
            return None
        if self.ttl_s is not None and time.time() - path.stat().st_mtime > self.ttl_s:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable cache entry %s", path)
            return None
        return data if isinstance(data, dict) else None

    def put(self, nct_id: str, study: JsonDict) -> None:
        path = self._path(nct_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(study, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class CtgovClient:
    """Async client for ``GET /api/v2/studies/{nct_id}``.

    Retries on 429/5xx and transport errors with exponential backoff, and caches
    successful responses on disk when a cache is configured.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        cache: StudyCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = default_sleep,
    ) -> None:
        self.cache = cache
        self.max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            headers={"Accept": "application/json", "User-Agent": f"ctgov-extract/{__version__}"},
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings, *, use_cache: bool = True) -> Self:
        cache = (
            StudyCache(settings.cache_dir, settings.cache_ttl_s)
            if use_cache and settings.cache_dir is not None
            else None
        )
        return cls(
            base_url=settings.ctgov_base_url, timeout_s=settings.ctgov_timeout_s, cache=cache
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def fetch_study(self, nct_id: str) -> JsonDict:
        """Return the raw API v2 JSON for ``nct_id`` (an ID or URL)."""
        nct_id = parse_nct_id(nct_id)
        if self.cache is not None:
            cached = self.cache.get(nct_id)
            if cached is not None:
                logger.debug("Cache hit for %s", nct_id)
                return cached

        study = await self._get_with_retries(nct_id)
        if self.cache is not None:
            self.cache.put(nct_id, study)
        return study

    async def fetch_record(self, nct_id: str) -> TrialRecord:
        return study_to_record(await self.fetch_study(nct_id))

    async def _get_with_retries(self, nct_id: str) -> JsonDict:
        url = f"/studies/{nct_id}"
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await self._http.get(url, params={"format": "json"})
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 404:
                    raise StudyNotFoundError(f"ClinicalTrials.gov has no study {nct_id}.")
                if response.status_code == 400:
                    raise InvalidTrialIdError(f"ClinicalTrials.gov rejected the ID {nct_id!r}.")
                if response.status_code not in RETRYABLE_STATUS:
                    return self._parse(response, nct_id)
                last_error = f"HTTP {response.status_code}"

            if attempt < self.max_retries:
                delay = retry_delay(attempt, response)
                logger.info("Retrying %s in %.1fs (%s)", nct_id, delay, last_error)
                await self._sleep(delay)
        raise CtgovApiError(
            f"Failed to fetch {nct_id} after {self.max_retries + 1} attempts: {last_error}"
        )

    @staticmethod
    def _parse(response: httpx.Response, nct_id: str) -> JsonDict:
        if response.status_code >= 400:
            raise CtgovApiError(
                f"ClinicalTrials.gov returned HTTP {response.status_code} for {nct_id}."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise CtgovApiError(f"Invalid JSON from ClinicalTrials.gov for {nct_id}.") from exc
        if not isinstance(data, dict) or "protocolSection" not in data:
            raise CtgovApiError(f"Unexpected payload for {nct_id}: no protocolSection.")
        return data


def load_study_file(path: Path) -> JsonDict:
    """Load an API v2 study JSON from disk (a single study, not a search page)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidTrialIdError(f"{path} is not valid JSON: {exc}") from exc
    if isinstance(data, dict) and "studies" in data and isinstance(data["studies"], list):
        studies = data["studies"]
        if len(studies) != 1:
            raise InvalidTrialIdError(f"{path} contains {len(studies)} studies; expected one.")
        data = studies[0]
    if not isinstance(data, dict) or "protocolSection" not in data:
        raise InvalidTrialIdError(f"{path} does not look like an API v2 study record.")
    return data


async def resolve_trial(ref: str, client: CtgovClient, *, allow_files: bool = True) -> TrialRecord:
    """Turn a user reference (NCT ID, URL or path to a study JSON) into a :class:`TrialRecord`.

    ``allow_files=False`` restricts input to IDs and URLs (used by the web API, so that
    remote callers cannot make the server read local files).
    """
    if not allow_files:
        return await client.fetch_record(ref)
    path = Path(ref).expanduser()
    if ref.lower().endswith(".json") or path.is_file():
        if not path.is_file():
            raise InvalidTrialIdError(f"File not found: {ref}")
        return study_to_record(load_study_file(path))
    return await client.fetch_record(ref)
