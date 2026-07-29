"""Deterministic provider used exclusively by the explicit fixture mode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bettercallagent.providers.openai_compatible import ProviderError
from bettercallagent.schemas import ChatMessage, LLMResponse
from online.repository import FixtureScript


@dataclass(slots=True)
class FixtureChatProvider:
    """Return versioned scripted outputs without making network requests."""

    script: FixtureScript

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
        """Return the exact fixture response associated with a stage purpose."""
        del messages, max_tokens, temperature
        if purpose == "understanding":
            value: object = self.script.understanding
        elif purpose == "query_generation":
            value = self.script.query_plan
        elif purpose == "rerank":
            doc_ref = (metadata or {}).get("doc_ref")
            if not doc_ref or doc_ref not in self.script.rerank:
                raise ProviderError("Fixture rerank request has no known doc_ref.")
            value = self.script.rerank[doc_ref]
        elif purpose == "answer":
            if json_response:
                raise ProviderError("Fixture answer must not request JSON mode.")
            return LLMResponse(content=self.script.answer, model=model)
        else:
            raise ProviderError(f"Unsupported fixture purpose: {purpose!r}.")
        if not json_response:
            raise ProviderError(f"Fixture purpose {purpose!r} requires JSON mode.")
        return LLMResponse(
            content=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            model=model,
        )

    async def close(self) -> None:
        """Fixture mode retains no external resources."""
        return None
