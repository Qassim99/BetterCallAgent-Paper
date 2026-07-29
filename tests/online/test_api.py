"""HTTP contract, authentication, and SSE integration tests."""

from __future__ import annotations

import json
import unittest
from collections.abc import Mapping, Sequence

from bettercallagent.providers.openai_compatible import ChatProvider
from bettercallagent.schemas import ChatMessage, LLMResponse
from online.app import create_app
from tests.online.helpers import (
    asgi_request,
    fixture_dependencies,
    fixture_request_body,
    parse_sse,
)

_TOKEN = "fixture-token-at-least-16-characters"


class InventingAnswerProvider:
    """Delegate every stage except the final deliberately invalid answer."""

    def __init__(self, delegate: ChatProvider) -> None:
        self.delegate = delegate

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
        if purpose == "answer":
            return LLMResponse(
                content=("SECRET-RAW-OUTPUT: Eine erfundene Grundlage wäre Art. 999 Abs. 1 OR."),
                model=model,
            )
        return await self.delegate.complete(
            messages,
            model=model,
            purpose=purpose,
            metadata=metadata,
            json_response=json_response,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def close(self) -> None:
        await self.delegate.close()


class OnlineApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.dependencies = fixture_dependencies(auth_token=_TOKEN)
        self.app = create_app(
            self.dependencies.settings,
            dependencies=self.dependencies,
        )
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()

    async def asyncTearDown(self) -> None:
        await self.lifespan.__aexit__(None, None, None)

    async def test_health_is_public_but_models_require_authentication(self) -> None:
        health_status, _, health_body = await asgi_request(
            self.app,
            method="GET",
            path="/api/health",
        )
        self.assertEqual(health_status, 200)
        health = json.loads(health_body)
        self.assertTrue(health["ready"])
        self.assertEqual(health["artifact_documents"], 5)
        self.assertNotIn("corpus_documents", health)

        missing_status, _, _ = await asgi_request(
            self.app,
            method="GET",
            path="/api/models",
        )
        self.assertEqual(missing_status, 401)

        model_status, _, model_body = await asgi_request(
            self.app,
            method="GET",
            path="/api/models",
            headers={"authorization": f"Bearer {_TOKEN}"},
        )
        self.assertEqual(model_status, 200)
        self.assertEqual(json.loads(model_body)["default"], "fixture-reviewer")

        query_status, _, query_body = await asgi_request(
            self.app,
            method="GET",
            path="/api/queries",
            headers={"authorization": f"Bearer {_TOKEN}"},
        )
        self.assertEqual(query_status, 200)
        self.assertEqual(
            json.loads(query_body)["queries"][0]["split"],
            "fixture",
        )

    async def test_unknown_model_is_rejected_before_streaming(self) -> None:
        body = fixture_request_body(self.dependencies)
        body["model"] = "unapproved-model"
        response_status, _, response_body = await asgi_request(
            self.app,
            method="POST",
            path="/api/runs/stream",
            body=body,
            headers={"authorization": f"Bearer {_TOKEN}"},
        )
        self.assertEqual(response_status, 422)
        self.assertIn("allowlist", json.loads(response_body)["detail"])

    async def test_fixture_sse_completes_with_sanitized_contract(self) -> None:
        response_status, response_headers, response_body = await asgi_request(
            self.app,
            method="POST",
            path="/api/runs/stream",
            body=fixture_request_body(self.dependencies),
            headers={"authorization": f"Bearer {_TOKEN}"},
        )
        self.assertEqual(response_status, 200)
        self.assertTrue(response_headers["content-type"].startswith("text/event-stream"))
        events = parse_sse(response_body)
        self.assertEqual(events[-1]["type"], "stream_end")
        self.assertEqual(
            len([event for event in events if event["type"] == "step_complete"]),
            6,
        )
        self.assertNotIn("error", [event["type"] for event in events])

    async def test_stream_error_does_not_expose_invalid_model_output(self) -> None:
        self.dependencies.provider = InventingAnswerProvider(self.dependencies.provider)
        with self.assertLogs("online.app", level="ERROR"):
            response_status, _, response_body = await asgi_request(
                self.app,
                method="POST",
                path="/api/runs/stream",
                body=fixture_request_body(self.dependencies),
                headers={"authorization": f"Bearer {_TOKEN}"},
            )
        self.assertEqual(response_status, 200)
        events = parse_sse(response_body)
        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "pipeline_error")
        self.assertEqual(error["step"], 6)
        self.assertNotIn("SECRET-RAW-OUTPUT", error["message"])
        self.assertEqual(events[-1]["type"], "stream_end")
