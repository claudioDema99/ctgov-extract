"""FastAPI app: JSON endpoints plus a single static page."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ctgov_extract import __version__
from ctgov_extract.atomize import CriteriaExtractor
from ctgov_extract.config import Settings
from ctgov_extract.ctgov import CtgovClient, resolve_trial
from ctgov_extract.custom import CustomExtractionResult, CustomExtractor
from ctgov_extract.errors import (
    CtgovApiError,
    InvalidTrialIdError,
    LLMError,
    LLMNotConfiguredError,
    SchemaDefinitionError,
    StructuredOutputError,
    StudyNotFoundError,
)
from ctgov_extract.llm.base import LLMClient
from ctgov_extract.llm.openai_compat import OpenAICompatClient
from ctgov_extract.models import CriteriaResult, TrialRecord
from ctgov_extract.user_schema import list_example_schemas, parse_schema_text

STATIC_DIR = Path(__file__).parent / "static"
MAX_SCHEMA_CHARS = 50_000


class CriteriaRequest(BaseModel):
    trial: str = Field(description="NCT ID or ClinicalTrials.gov URL", max_length=500)


class ExtractRequest(BaseModel):
    trial: str = Field(description="NCT ID or ClinicalTrials.gov URL", max_length=500)
    schema_text: str = Field(
        description="Extraction schema as YAML or JSON", max_length=MAX_SCHEMA_CHARS
    )


class CriteriaResponse(BaseModel):
    trial: TrialRecord
    result: CriteriaResult


class ExtractResponse(BaseModel):
    trial: TrialRecord
    result: CustomExtractionResult


class _Services:
    """Holds the clients for one app instance; LLM client is created on first use."""

    def __init__(
        self, settings: Settings, ctgov: CtgovClient | None, llm: LLMClient | None
    ) -> None:
        self.settings = settings
        self._owns_ctgov = ctgov is None
        self.ctgov = ctgov or CtgovClient.from_settings(settings)
        self._llm = llm
        self._owns_llm = llm is None

    @property
    def llm_available(self) -> bool:
        return self._llm is not None or self.settings.llm_configured

    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatClient.from_settings(self.settings)
        return self._llm

    async def aclose(self) -> None:
        if self._owns_ctgov:
            await self.ctgov.aclose()
        if self._owns_llm and isinstance(self._llm, OpenAICompatClient):
            await self._llm.aclose()


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message, **extra})


def create_app(
    settings: Settings | None = None,
    *,
    ctgov_client: CtgovClient | None = None,
    llm_client: LLMClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    services = _Services(settings, ctgov_client, llm_client)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await services.aclose()

    app = FastAPI(
        title="ctgov-extract",
        version=__version__,
        description="Structured eligibility criteria from ClinicalTrials.gov records.",
        lifespan=lifespan,
    )
    app.state.services = services

    @app.exception_handler(InvalidTrialIdError)
    async def _invalid(_: Request, exc: InvalidTrialIdError) -> JSONResponse:
        return _error(422, str(exc))

    @app.exception_handler(SchemaDefinitionError)
    async def _bad_schema(_: Request, exc: SchemaDefinitionError) -> JSONResponse:
        return _error(422, str(exc))

    @app.exception_handler(StudyNotFoundError)
    async def _not_found(_: Request, exc: StudyNotFoundError) -> JSONResponse:
        return _error(404, str(exc))

    @app.exception_handler(LLMNotConfiguredError)
    async def _no_llm(_: Request, exc: LLMNotConfiguredError) -> JSONResponse:
        return _error(503, str(exc))

    @app.exception_handler(StructuredOutputError)
    async def _bad_output(_: Request, exc: StructuredOutputError) -> JSONResponse:
        return _error(502, str(exc), details=exc.errors)

    @app.exception_handler(LLMError)
    async def _llm_failed(_: Request, exc: LLMError) -> JSONResponse:
        return _error(502, str(exc))

    @app.exception_handler(CtgovApiError)
    async def _ctgov_failed(_: Request, exc: CtgovApiError) -> JSONResponse:
        return _error(502, str(exc))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "llm_configured": services.llm_available,
            "llm_model": settings.llm_model if llm_client is None else llm_client.model_name,
        }

    @app.get("/api/trial/{trial_id:path}")
    async def trial(trial_id: str) -> TrialRecord:
        return await resolve_trial(trial_id, services.ctgov, allow_files=False)

    @app.get("/api/schemas")
    async def schemas() -> dict[str, dict[str, str]]:
        return {"examples": list_example_schemas()}

    @app.post("/api/criteria")
    async def criteria(body: CriteriaRequest) -> CriteriaResponse:
        record = await resolve_trial(body.trial, services.ctgov, allow_files=False)
        result = await CriteriaExtractor(
            services.llm(), output_mode=settings.llm_output_mode
        ).extract_text(
            record.eligibility_text,
            nct_id=record.nct_id,
            title=record.brief_title,
            conditions=record.conditions,
        )
        return CriteriaResponse(trial=record, result=result)

    @app.post("/api/extract")
    async def extract(body: ExtractRequest) -> ExtractResponse:
        spec = parse_schema_text(body.schema_text)
        record = await resolve_trial(body.trial, services.ctgov, allow_files=False)
        result = await CustomExtractor(
            services.llm(), output_mode=settings.llm_output_mode
        ).extract(record, spec)
        return ExtractResponse(trial=record, result=result)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
