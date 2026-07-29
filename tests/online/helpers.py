"""Test helpers that require no third-party HTTP client."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bettercallagent.settings import ExecutionMode, OnlineSettings
from online.dependencies import OnlineDependencies

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPOSITORY_ROOT / "online" / "fixtures" / "demo.json"
FIXTURE_QUERY_ID = "demo-contract-delay"
FIXTURE_MODEL = "fixture-reviewer"


def fixture_settings(*, auth_token: str | None = None) -> OnlineSettings:
    """Return explicit network-free settings for tests."""
    return OnlineSettings(
        mode=ExecutionMode.FIXTURE,
        asset_path=FIXTURE_PATH,
        allowed_models=(FIXTURE_MODEL,),
        default_model=FIXTURE_MODEL,
        cors_origins=("http://localhost:5173",),
        auth_token=auth_token,
    )


def fixture_dependencies(
    *,
    auth_token: str | None = None,
) -> OnlineDependencies:
    """Build all fixture-mode dependencies from the committed asset."""
    return OnlineDependencies.build(fixture_settings(auth_token=auth_token))


def fixture_request_body(dependencies: OnlineDependencies) -> dict[str, Any]:
    """Build the exact request advertised by the query endpoint."""
    record = dependencies.repository.queries[FIXTURE_QUERY_ID]
    return {
        "query_id": record.query_id,
        "query": record.query,
        "model": FIXTURE_MODEL,
        "history": [],
    }


async def asgi_request(
    app: Any,
    *,
    method: str,
    path: str,
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """Issue one HTTP request directly against an ASGI app."""
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else b""
    request_headers = {
        "host": "testserver",
        "content-type": "application/json",
        **(headers or {}),
    }
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in request_headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, Any]] = []
    first_receive = True
    never_disconnect = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal first_receive
        if first_receive:
            first_receive = False
            return {"type": "http.request", "body": raw_body, "more_body": False}
        await never_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1") for key, value in start["headers"]
    }
    return start["status"], response_headers, response_body


def parse_sse(payload: bytes) -> list[dict[str, Any]]:
    """Parse the service's one-data-line SSE representation."""
    events: list[dict[str, Any]] = []
    for block in payload.decode("utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        data_line = next(line for line in lines if line.startswith("data: "))
        events.append(json.loads(data_line.removeprefix("data: ")))
    return events
