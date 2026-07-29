"""Six-stage online pipeline with a stable, sanitized event contract."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from online import stages
from online.context import RunContext
from online.dependencies import OnlineDependencies

SCHEMA_VERSION = 1
StageCallable = Callable[
    [RunContext, OnlineDependencies],
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One numbered pipeline stage and its public name."""

    number: int
    name: str
    summary: str
    runner: StageCallable


STAGES = (
    StageSpec(
        1,
        "Question understanding",
        "Legal topic and key concepts identified.",
        stages.understand,
    ),
    StageSpec(
        2,
        "Query generation",
        "Five deterministic retrieval views prepared.",
        stages.generate_queries,
    ),
    StageSpec(
        3,
        "Artifact-backed ranking replay",
        "Versioned dense and BM25 rankings replayed and fused with weighted RRF.",
        stages.retrieve,
    ),
    StageSpec(
        4,
        "Candidate reranking",
        "Dense candidates independently scored and ordered.",
        stages.rerank,
    ),
    StageSpec(
        5,
        "Citation validation",
        "Fixed evidence gates accepted and rejected citations.",
        stages.control_citations,
    ),
    StageSpec(
        6,
        "Grounded answer",
        "Final answer generated and citations exactly post-validated.",
        stages.answer,
    ),
)


class PipelineStageError(RuntimeError):
    """Wrap an internal failure with only safe stage metadata."""

    def __init__(self, stage: StageSpec) -> None:
        super().__init__(f"Pipeline stage {stage.number} failed.")
        self.step = stage.number
        self.stage_name = stage.name


def _event(event_type: str, run_id: str, **payload: Any) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "type": event_type,
        "ts": time.time(),
        "run_id": run_id,
        **payload,
    }


async def stream_pipeline(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> AsyncIterator[dict[str, Any]]:
    """Run all stages and yield their stable public event sequence."""
    started_at = time.monotonic()
    yield _event(
        "run_start",
        context.run_id,
        query=context.record.query,
        query_id=context.record.query_id,
        model=context.model,
        mode=dependencies.settings.mode.value,
    )
    for stage in STAGES:
        yield _event(
            "step_start",
            context.run_id,
            step=stage.number,
            name=stage.name,
        )
        try:
            data = await stage.runner(context, dependencies)
        except Exception as exc:
            raise PipelineStageError(stage) from exc
        yield _event(
            "step_complete",
            context.run_id,
            step=stage.number,
            name=stage.name,
            summary=stage.summary,
            data=data,
        )
    if context.answer is None or context.answer_validation is None:
        raise RuntimeError("The completed pipeline has no validated answer.")
    yield _event(
        "final_answer",
        context.run_id,
        markdown=context.answer,
        grounded_on=list(context.answer_validation.exact_mentioned),
    )
    yield _event(
        "run_complete",
        context.run_id,
        elapsed_s=max(0.0, time.monotonic() - started_at),
        usage_total_tokens=context.usage_total_tokens,
    )
    yield _event("stream_end", context.run_id)
