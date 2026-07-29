"""Typed HTTP request models for the online API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HistoryMessage(BaseModel):
    """One bounded user-visible conversation message."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=3_000)


class RunRequest(BaseModel):
    """Validated input for a single pipeline run."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=12_000)
    query_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
    )
    model: str = Field(min_length=1, max_length=128)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)
