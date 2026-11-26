"""Runtime settings, read from environment variables."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OutputMode = Literal["text", "json_object", "json_schema"]
OUTPUT_MODES: tuple[OutputMode, ...] = ("text", "json_object", "json_schema")


def _default_cache_dir(env: Mapping[str, str]) -> Path:
    base = env.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ctgov-extract"


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return float(raw) if raw not in (None, "") else default


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return int(raw) if raw not in (None, "") else default


def _json_object(env: Mapping[str, str], key: str) -> dict[str, Any] | None:
    raw = env.get(key)
    if raw in (None, ""):
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object, got {type(value).__name__}.")
    return value


def _output_mode(env: Mapping[str, str]) -> OutputMode:
    raw = (env.get("LLM_OUTPUT_MODE") or "text").strip().lower()
    if raw not in OUTPUT_MODES:
        raise ValueError(f"LLM_OUTPUT_MODE must be one of {', '.join(OUTPUT_MODES)}, got {raw!r}.")
    return raw


@dataclass(frozen=True)
class Settings:
    """All tunables in one place. Use :meth:`from_env` in applications."""

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_timeout_s: float = 120.0
    llm_max_concurrency: int = 4
    llm_max_retries: int = 3
    llm_max_tokens: int | None = None
    llm_extra_body: dict[str, Any] | None = None
    llm_output_mode: OutputMode = "text"
    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_timeout_s: float = 30.0
    cache_dir: Path | None = None
    cache_ttl_s: float = 7 * 24 * 3600

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        cache_raw = env.get("CTGOV_CACHE_DIR")
        if cache_raw is not None and cache_raw.strip().lower() in {"", "none", "off", "0"}:
            cache_dir: Path | None = None
        else:
            cache_dir = Path(cache_raw) if cache_raw else _default_cache_dir(env)
        max_tokens = env.get("LLM_MAX_TOKENS")
        return cls(
            llm_base_url=env.get("LLM_BASE_URL") or None,
            llm_api_key=env.get("LLM_API_KEY") or None,
            llm_model=env.get("LLM_MODEL") or None,
            llm_temperature=_float(env, "LLM_TEMPERATURE", 0.0),
            llm_timeout_s=_float(env, "LLM_TIMEOUT", 120.0),
            llm_max_concurrency=_int(env, "LLM_MAX_CONCURRENCY", 4),
            llm_max_retries=_int(env, "LLM_MAX_RETRIES", 3),
            llm_max_tokens=int(max_tokens) if max_tokens else None,
            llm_extra_body=_json_object(env, "LLM_EXTRA_BODY"),
            llm_output_mode=_output_mode(env),
            ctgov_timeout_s=_float(env, "CTGOV_TIMEOUT", 30.0),
            cache_dir=cache_dir,
            cache_ttl_s=_float(env, "CTGOV_CACHE_TTL", 7 * 24 * 3600),
        )
