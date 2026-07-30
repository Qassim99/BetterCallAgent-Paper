"""Minimal asynchronous OpenAI-compatible client with mandatory TLS verification."""

from __future__ import annotations

import asyncio
import json
import math
import ssl
import time
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
    latency_seconds: float = 0.0,
) -> LLMResponse:
    """Normalize an OpenAI-compatible completion without promoting reasoning.

    Reasoning models may return a separate ``reasoning_content`` field or put a
    leading ``<think>...</think>`` block in ``content``. Reasoning is retained
    for audit, but only the non-empty final answer is exposed as ``content``.
    """
    try:
        parsed = json.loads(body)
        choice = parsed["choices"][0]
        message = choice["message"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Provider returned an invalid completion envelope.") from exc
    if not isinstance(message, dict):
        raise ProviderError("Provider returned an invalid completion message.")
    if choice.get("finish_reason") == "length":
        raise ProviderError("Provider truncated the completion at the token limit.")

    raw_content = message.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        raise ProviderError("Provider returned no final completion content.")
    content, inline_reasoning = _separate_inline_reasoning(raw_content)
    provider_reasoning = message.get("reasoning_content")
    if provider_reasoning is not None and not isinstance(provider_reasoning, str):
        raise ProviderError("Provider returned invalid reasoning_content.")
    reasoning_parts = [
        value.strip()
        for value in (provider_reasoning, inline_reasoning)
        if isinstance(value, str) and value.strip()
    ]
    reasoning_content = "\n\n".join(reasoning_parts) or None

    usage = parsed.get("usage") or {}
    if not isinstance(usage, dict):
        raise ProviderError("Provider returned invalid token usage.")
    total_tokens = _nonnegative_token_count(usage, "total_tokens")
    prompt_tokens = _nonnegative_token_count(usage, "prompt_tokens")
    completion_tokens = _nonnegative_token_count(usage, "completion_tokens")
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        raise ProviderError("Provider returned invalid completion token details.")
    reasoning_tokens = _nonnegative_token_count(completion_details, "reasoning_tokens")
    if not isinstance(latency_seconds, (int, float)) or not math.isfinite(latency_seconds):
        raise ProviderError("Provider request latency must be finite.")
    if latency_seconds < 0:
        raise ProviderError("Provider request latency cannot be negative.")
    return LLMResponse(
        content=content,
        model=str(parsed.get("model") or requested_model),
        usage_total_tokens=total_tokens,
        usage_prompt_tokens=prompt_tokens,
        usage_completion_tokens=completion_tokens,
        usage_reasoning_tokens=reasoning_tokens,
        latency_seconds=float(latency_seconds),
        reasoning_content=reasoning_content,
        raw=parsed,
    )


def _nonnegative_token_count(container: Mapping[str, Any], field_name: str) -> int:
    value = container.get(field_name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError(f"Provider returned invalid {field_name}.")
    return value


def _separate_inline_reasoning(content: str) -> tuple[str, str | None]:
    """Return final text and an optional leading ``<think>`` trace."""

    stripped = content.strip()
    if not stripped.startswith("<think>"):
        if "</think>" in stripped:
            raise ProviderError("Provider returned a malformed inline reasoning block.")
        return stripped, None
    closing = stripped.find("</think>", len("<think>"))
    if closing < 0:
        raise ProviderError("Provider returned a truncated inline reasoning block.")
    reasoning = stripped[len("<think>") : closing].strip()
    final_content = stripped[closing + len("</think>") :].strip()
    if not final_content:
        raise ProviderError("Provider returned reasoning without a final answer.")
    if "<think>" in final_content or "</think>" in final_content:
        raise ProviderError("Provider returned nested or repeated inline reasoning blocks.")
    return final_content, reasoning or None


@dataclass(slots=True)
class OpenAICompatibleProvider:
    """OpenAI-compatible `/chat/completions` client using the system trust store."""

    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: int = 300
    chat_template_kwargs: Mapping[str, Any] | None = None

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
        if self.chat_template_kwargs is not None:
            self.chat_template_kwargs = dict(self.chat_template_kwargs)
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
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
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
        started = time.perf_counter()
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
            latency_seconds=time.perf_counter() - started,
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
