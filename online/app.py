"""FastAPI application factory for the BetterCallAgent online pipeline."""

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from bettercallagent.schemas import ChatMessage
from bettercallagent.settings import OnlineSettings
from online.context import RunContext
from online.contracts import RunRequest
from online.dependencies import OnlineDependencies
from online.pipeline import SCHEMA_VERSION, PipelineStageError, stream_pipeline
from online.repository import AssetError

LOGGER = logging.getLogger(__name__)


def _sse(event: dict[str, Any]) -> bytes:
    """Encode one event using one compact, newline-safe JSON data field."""
    event_name = str(event["type"])
    data = json.dumps(
        event,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"event: {event_name}\ndata: {data}\n\n".encode()


def create_app(
    settings: OnlineSettings | None = None,
    *,
    dependencies: OnlineDependencies | None = None,
) -> FastAPI:
    """Create an application from explicit settings or the supplied environment.

    Uvicorn calls this function via ``--factory``. Environment variables are
    therefore read at application construction, never during module import.
    """
    resolved_settings = settings
    if resolved_settings is None:
        resolved_settings = (
            dependencies.settings
            if dependencies is not None
            else OnlineSettings.from_environment(os.environ)
        )
    if dependencies is not None and dependencies.settings != resolved_settings:
        raise ValueError("Injected dependencies and settings must match exactly.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        initialized = dependencies or OnlineDependencies.build(resolved_settings)
        app.state.dependencies = initialized
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await initialized.close()

    app = FastAPI(
        title="BetterCallAgent Online API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.ready = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    def get_dependencies(request: Request) -> OnlineDependencies:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is not ready.",
            )
        return request.app.state.dependencies

    async def authorize(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = resolved_settings.auth_token
        if expected is None:
            return
        if authorization is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer authentication is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not hmac.compare_digest(token, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/api/health")
    async def health(
        initialized: Annotated[
            OnlineDependencies,
            Depends(get_dependencies),
        ],
    ) -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "mode": initialized.settings.mode.value,
            "default_model": initialized.settings.default_model,
            "artifact_documents": len(initialized.repository.documents),
        }

    @app.get("/api/models", dependencies=[Depends(authorize)])
    async def models(
        initialized: Annotated[
            OnlineDependencies,
            Depends(get_dependencies),
        ],
    ) -> dict[str, Any]:
        return {
            "default": initialized.settings.default_model,
            "models": list(initialized.settings.allowed_models),
        }

    @app.get("/api/queries", dependencies=[Depends(authorize)])
    async def queries(
        initialized: Annotated[
            OnlineDependencies,
            Depends(get_dependencies),
        ],
    ) -> dict[str, Any]:
        return {
            "queries": initialized.repository.public_queries(
                public_split=initialized.settings.mode.value,
            )
        }

    @app.post("/api/runs/stream", dependencies=[Depends(authorize)])
    async def run_stream(
        body: RunRequest,
        initialized: Annotated[
            OnlineDependencies,
            Depends(get_dependencies),
        ],
    ) -> StreamingResponse:
        settings_for_run = initialized.settings
        if len(body.query) > settings_for_run.max_query_chars:
            raise HTTPException(
                status_code=422,
                detail="query exceeds the configured length limit.",
            )
        if len(body.history) > settings_for_run.max_history_messages:
            raise HTTPException(
                status_code=422,
                detail="history exceeds the configured message limit.",
            )
        if any(
            len(message.content) > settings_for_run.max_history_chars for message in body.history
        ):
            raise HTTPException(
                status_code=422,
                detail="history message exceeds the configured length limit.",
            )
        if body.model not in settings_for_run.allowed_models:
            raise HTTPException(
                status_code=422,
                detail="model is not in the configured allowlist.",
            )
        try:
            record = initialized.repository.resolve_query(
                query=body.query,
                query_id=body.query_id,
            )
        except AssetError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        context = RunContext(
            run_id=uuid.uuid4().hex,
            record=record,
            model=body.model,
            history=tuple(
                ChatMessage(role=message.role, content=message.content) for message in body.history
            ),
        )

        async def event_source() -> AsyncIterator[bytes]:
            try:
                async for event in stream_pipeline(context, initialized):
                    yield _sse(event)
            except asyncio.CancelledError:
                raise
            except PipelineStageError as exc:
                LOGGER.exception(
                    "Pipeline run %s failed at stage %d (%s).",
                    context.run_id,
                    exc.step,
                    exc.stage_name,
                )
                yield _sse(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "type": "error",
                        "ts": time.time(),
                        "run_id": context.run_id,
                        "code": "pipeline_error",
                        "step": exc.step,
                        "message": "The pipeline could not complete this stage.",
                    }
                )
                yield _sse(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "type": "stream_end",
                        "ts": time.time(),
                        "run_id": context.run_id,
                    }
                )
            except Exception:
                LOGGER.exception("Pipeline run %s failed.", context.run_id)
                yield _sse(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "type": "error",
                        "ts": time.time(),
                        "run_id": context.run_id,
                        "code": "pipeline_error",
                        "message": "The pipeline could not complete the request.",
                    }
                )
                yield _sse(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "type": "stream_end",
                        "ts": time.time(),
                        "run_id": context.run_id,
                    }
                )

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app
