"""Typed application settings with explicit, side-effect-free environment loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse


class SettingsError(ValueError):
    """Raised when required application configuration is absent or inconsistent."""


class ExecutionMode(StrEnum):
    """Supported online execution modes."""

    FIXTURE = "fixture"
    LIVE = "live"


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise SettingsError(f"{name} must be set.")
    return value


def _csv(value: str, *, name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise SettingsError(f"{name} must contain at least one value.")
    if len(values) != len(set(values)):
        raise SettingsError(f"{name} must not contain duplicate values.")
    return values


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive.")
    return value


@dataclass(frozen=True, slots=True)
class OnlineSettings:
    """Validated configuration for the online service.

    Environment variables are read only when :meth:`from_environment` is called.
    Importing this module never reads files, environment variables, or secrets.
    """

    mode: ExecutionMode
    asset_path: Path
    allowed_models: tuple[str, ...]
    default_model: str
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    auth_token: str | None = field(default=None, repr=False)
    llm_base_url: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    request_timeout_seconds: int = 300
    retrieve_k: int = 10
    rerank_n: int = 10
    top_select: int = 5
    max_query_chars: int = 12_000
    max_history_messages: int = 6
    max_history_chars: int = 3_000

    def __post_init__(self) -> None:
        """Reject unsafe or ambiguous configuration."""
        if not self.allowed_models:
            raise SettingsError("allowed_models must not be empty.")
        if len(self.allowed_models) != len(set(self.allowed_models)):
            raise SettingsError("allowed_models must not contain duplicates.")
        if any(not model.strip() or len(model) > 128 for model in self.allowed_models):
            raise SettingsError("Each allowed model must contain between 1 and 128 characters.")
        if self.default_model not in self.allowed_models:
            raise SettingsError("default_model must be present in allowed_models.")
        if not self.asset_path:
            raise SettingsError("asset_path must be configured explicitly.")
        if not self.cors_origins:
            raise SettingsError("At least one CORS origin must be configured.")
        for origin in self.cors_origins:
            if origin == "*":
                raise SettingsError("Wildcard CORS origins are not permitted.")
            parsed = urlparse(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.params
                or parsed.query
                or parsed.fragment
            ):
                raise SettingsError(f"Invalid CORS origin: {origin!r}.")
        if self.auth_token is not None and len(self.auth_token) < 16:
            raise SettingsError("Configured bearer tokens must contain at least 16 characters.")
        for name in (
            "request_timeout_seconds",
            "retrieve_k",
            "rerank_n",
            "top_select",
            "max_query_chars",
            "max_history_messages",
            "max_history_chars",
        ):
            if getattr(self, name) <= 0:
                raise SettingsError(f"{name} must be positive.")
        if self.top_select > self.rerank_n:
            raise SettingsError("top_select cannot exceed rerank_n.")
        if self.rerank_n > self.retrieve_k:
            raise SettingsError("rerank_n cannot exceed retrieve_k.")
        if self.max_query_chars > 12_000:
            raise SettingsError("max_query_chars cannot exceed the API limit of 12000.")
        if self.max_history_messages > 6:
            raise SettingsError("max_history_messages cannot exceed the API limit of 6.")
        if self.max_history_chars > 3_000:
            raise SettingsError("max_history_chars cannot exceed the API limit of 3000.")
        if self.mode is ExecutionMode.LIVE:
            if not self.llm_base_url:
                raise SettingsError("llm_base_url is required in live mode.")
            parsed = urlparse(self.llm_base_url)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise SettingsError("Live LLM endpoints must use a valid HTTPS URL.")
            if not self.llm_api_key:
                raise SettingsError("llm_api_key is required in live mode.")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> OnlineSettings:
        """Build settings from an explicitly supplied environment mapping."""
        mode_raw = _required(environment, "BCA_ONLINE_MODE")
        try:
            mode = ExecutionMode(mode_raw)
        except ValueError as exc:
            raise SettingsError("BCA_ONLINE_MODE must be either 'fixture' or 'live'.") from exc

        auth_token = environment.get("BCA_AUTH_TOKEN", "").strip() or None
        llm_base_url = environment.get("BCA_LLM_BASE_URL", "").strip() or None
        llm_api_key = environment.get("BCA_LLM_API_KEY", "").strip() or None
        return cls(
            mode=mode,
            asset_path=Path(_required(environment, "BCA_ASSET_PATH")).expanduser(),
            allowed_models=_csv(
                _required(environment, "BCA_ALLOWED_MODELS"),
                name="BCA_ALLOWED_MODELS",
            ),
            default_model=_required(environment, "BCA_DEFAULT_MODEL"),
            cors_origins=_csv(
                environment.get("BCA_CORS_ORIGINS", "http://localhost:5173"),
                name="BCA_CORS_ORIGINS",
            ),
            auth_token=auth_token,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            request_timeout_seconds=_positive_int(environment, "BCA_REQUEST_TIMEOUT_SECONDS", 300),
            retrieve_k=_positive_int(environment, "BCA_RETRIEVE_K", 10),
            rerank_n=_positive_int(environment, "BCA_RERANK_N", 10),
            top_select=_positive_int(environment, "BCA_TOP_SELECT", 5),
            max_query_chars=_positive_int(environment, "BCA_MAX_QUERY_CHARS", 12_000),
            max_history_messages=_positive_int(environment, "BCA_MAX_HISTORY_MESSAGES", 6),
            max_history_chars=_positive_int(environment, "BCA_MAX_HISTORY_CHARS", 3_000),
        )
