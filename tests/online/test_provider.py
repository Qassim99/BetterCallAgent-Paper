"""Regression tests for OpenAI-compatible completion normalization."""

from __future__ import annotations

import json
import unittest

from bettercallagent.providers.openai_compatible import (
    ProviderError,
    parse_completion_envelope,
)


def envelope(*, content: object, reasoning_content: object) -> str:
    return json.dumps(
        {
            "model": "returned-model",
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": reasoning_content,
                    }
                }
            ],
            "usage": {"total_tokens": 17},
        }
    )


class CompletionEnvelopeTests(unittest.TestCase):
    def test_prefers_standard_content_when_both_fields_are_present(self) -> None:
        response = parse_completion_envelope(
            envelope(content=" final ", reasoning_content="reasoning"),
            requested_model="requested-model",
        )
        self.assertEqual(response.content, "final")
        self.assertEqual(response.model, "returned-model")
        self.assertEqual(response.usage_total_tokens, 17)

    def test_uses_reasoning_content_when_standard_content_is_empty(self) -> None:
        response = parse_completion_envelope(
            envelope(content=" ", reasoning_content=" grounded answer "),
            requested_model="requested-model",
        )
        self.assertEqual(response.content, "grounded answer")

    def test_rejects_response_when_both_content_fields_are_empty(self) -> None:
        with self.assertRaisesRegex(ProviderError, "empty completion"):
            parse_completion_envelope(
                envelope(content=None, reasoning_content=" "),
                requested_model="requested-model",
            )
