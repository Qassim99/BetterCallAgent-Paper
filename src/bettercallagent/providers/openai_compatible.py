"""Minimal asynchronous OpenAI-compatible client with mandatory TLS verification."""

from __future__ import annotations

import asyncio
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from bettercallagent.schemas import ChatMessage, LLMResponse


class ProviderError(RuntimeError):
    """Raised when a configured model provider cannot return a valid response."""


class ChatProvider(Protocol):
    """Interface consumed by all LLM-backed online stages."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        purpose: str,
        metadata: Mapping[str, str] | None = None,
        json_response: bool = False,
        max_tokens: int = 1_024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Return one normalized chat completion."""
        ...

    async def close(self) -> None:
        """Release provider resources."""
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse an exact JSON object and reject prose, fences, and arrays."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError("Model response is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise ProviderError("Model response must be one JSON object.")
    return value


def parse_completion_envelope(
    body: str,
    *,
    requested_model: str,
) -> LLMResponse:
    """Normalize a chat-completion envelope and its explicit content fields.

    Some compatible reasoning models place their final text in
    ``reasoning_content`` while returning an empty ``content`` field. The
    fallback is restricted to those two documented fields; both-empty responses
    still fail.
    """
    try:
        parsed = json.loads(body)
        message = parsed["choices"][0]["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Provider returned an invalid completion envelope.") from exc
    if not isinstance(message, dict):
        raise ProviderError("Provider returned an invalid completion message.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        content = message.get("reasoning_content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("Provider returned empty completion content.")
    usage = parsed.get("usage") or {}
    total_tokens = usage.get("total_tokens", 0)
    if not isinstance(total_tokens, int) or total_tokens < 0:
        raise ProviderError("Provider returned an invalid token count.")
    return LLMResponse(
        content=content.strip(),
        model=str(parsed.get("model") or requested_model),
        usage_total_tokens=total_tokens,
        raw=parsed,
    )


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """OpenAI-compatible `/chat/completions` client using the system trust store."""

    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OpenAI-compatible providers must use a valid HTTPS URL.")
        if not self.api_key:
            raise ValueError("OpenAI-compatible providers require an API key.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.base_url = self.base_url.rstrip("/")

    def _complete_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        json_response: bool,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_response:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        context = ssl.create_default_context()
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=context,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Provider returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderError("Provider request failed.") from exc
        return parse_completion_envelope(
            body,
            requested_model=model,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        purpose: str,
        metadata: Mapping[str, str] | None = None,
        json_response: bool = False,
        max_tokens: int = 1_024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Call the provider without blocking the asynchronous API worker."""
        del purpose, metadata
        return await asyncio.to_thread(
            self._complete_sync,
            messages,
            model=model,
            json_response=json_response,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def close(self) -> None:
        """No persistent connections are retained by the urllib client."""
        return None
