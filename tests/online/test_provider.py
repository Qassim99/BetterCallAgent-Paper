"""Regression tests for OpenAI-compatible completion normalization."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from bettercallagent.providers.openai_compatible import (
    OpenAICompatibleProvider,
    ProviderError,
    parse_completion_envelope,
)
from bettercallagent.schemas import ChatMessage


def envelope(
    *,
    content: object,
    reasoning_content: object,
    finish_reason: object = "stop",
) -> str:
    return json.dumps(
        {
            "model": "returned-model",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": content,
                        "reasoning_content": reasoning_content,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 6,
                "total_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }
    )


class CompletionEnvelopeTests(unittest.TestCase):
    def test_keeps_final_content_separate_from_provider_reasoning(self) -> None:
        response = parse_completion_envelope(
            envelope(content=" final ", reasoning_content="reasoning"),
            requested_model="requested-model",
            latency_seconds=0.25,
        )
        self.assertEqual(response.content, "final")
        self.assertEqual(response.reasoning_content, "reasoning")
        self.assertEqual(response.model, "returned-model")
        self.assertEqual(response.usage_prompt_tokens, 11)
        self.assertEqual(response.usage_completion_tokens, 6)
        self.assertEqual(response.usage_reasoning_tokens, 4)
        self.assertEqual(response.usage_total_tokens, 17)
        self.assertEqual(response.latency_seconds, 0.25)

    def test_separates_leading_inline_reasoning_from_final_content(self) -> None:
        response = parse_completion_envelope(
            envelope(
                content="<think>inline trace</think>\nfinal answer",
                reasoning_content="provider trace",
            ),
            requested_model="requested-model",
        )
        self.assertEqual(response.content, "final answer")
        self.assertEqual(response.reasoning_content, "provider trace\n\ninline trace")

    def test_does_not_promote_reasoning_to_final_content(self) -> None:
        with self.assertRaisesRegex(ProviderError, "no final completion content"):
            parse_completion_envelope(
                envelope(content=" ", reasoning_content="grounded answer"),
                requested_model="requested-model",
            )

    def test_rejects_token_limit_truncation(self) -> None:
        with self.assertRaisesRegex(ProviderError, "truncated.*token limit"):
            parse_completion_envelope(
                envelope(
                    content="partial final answer",
                    reasoning_content="trace",
                    finish_reason="length",
                ),
                requested_model="requested-model",
            )

    def test_rejects_truncated_or_reasoning_only_inline_output(self) -> None:
        cases = (
            ("<think>unfinished trace", "truncated inline reasoning"),
            ("<think>complete trace</think>", "reasoning without a final answer"),
        )
        for content, expected_error in cases:
            with (
                self.subTest(content=content),
                self.assertRaisesRegex(ProviderError, expected_error),
            ):
                parse_completion_envelope(
                    envelope(content=content, reasoning_content=None),
                    requested_model="requested-model",
                )


class ProviderPayloadTests(unittest.TestCase):
    def test_forwards_chat_template_kwargs_in_request_payload(self) -> None:
        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return envelope(content="final", reasoning_content="trace").encode("utf-8")

        provider = OpenAICompatibleProvider(
            base_url="https://provider.example/api",
            api_key="test-key",
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch(
            "bettercallagent.providers.openai_compatible.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            provider._complete_sync(
                (ChatMessage(role="user", content="Hallo."),),
                model="hosted-reasoning-model",
                json_response=True,
                max_tokens=128,
                temperature=0.7,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload,
            {
                "model": "hosted-reasoning-model",
                "messages": [{"role": "user", "content": "Hallo."}],
                "temperature": 0.7,
                "max_tokens": 128,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )

    def test_can_match_hosted_payload_without_response_format(self) -> None:
        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return envelope(content="final", reasoning_content=None).encode("utf-8")

        provider = OpenAICompatibleProvider(
            base_url="https://provider.example/api",
            api_key="test-key",
            chat_template_kwargs={"enable_thinking": False},
        )
        with patch(
            "bettercallagent.providers.openai_compatible.urllib.request.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            provider._complete_sync(
                (ChatMessage(role="user", content="Hallo."),),
                model="hosted-model",
                json_response=False,
                max_tokens=128,
                temperature=0.7,
            )

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload,
            {
                "model": "hosted-model",
                "messages": [{"role": "user", "content": "Hallo."}],
                "temperature": 0.7,
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.assertNotIn("response_format", payload)
