"""Exception hierarchy shared by the library, the CLI and the web API."""

from typing import Any


class CtgovExtractError(Exception):
    """Base class for all errors raised by this package."""


class InvalidTrialIdError(CtgovExtractError, ValueError):
    """The input is not a recognisable NCT ID, ClinicalTrials.gov URL or study file."""


class StudyNotFoundError(CtgovExtractError):
    """ClinicalTrials.gov has no study with the requested NCT ID."""


class CtgovApiError(CtgovExtractError):
    """The ClinicalTrials.gov API failed after retries or returned an unexpected payload."""


class SchemaDefinitionError(CtgovExtractError, ValueError):
    """A user-defined extraction schema is malformed."""


class LLMError(CtgovExtractError):
    """The LLM backend failed (transport, HTTP error or unusable response)."""


class LLMNotConfiguredError(LLMError):
    """No LLM backend is configured (LLM_BASE_URL / LLM_MODEL missing)."""


class UnsupportedResponseFormatError(LLMError):
    """The backend rejected the requested ``response_format``."""


class StructuredOutputError(LLMError):
    """The model output could not be parsed or validated, even after a repair round."""

    def __init__(
        self, message: str, *, raw_output: str, errors: list[str], usage: Any = None
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.errors = errors
        self.usage = usage  # token usage of the failed attempts, if known
