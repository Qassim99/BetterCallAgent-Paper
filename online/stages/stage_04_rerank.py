"""Stage 4: independently verify and rerank dense candidates."""

from __future__ import annotations

import asyncio
from typing import Any

from bettercallagent.providers.openai_compatible import parse_json_object
from bettercallagent.retrieval.query_views import QueryViews
from bettercallagent.schemas import LLMResponse, RankedCandidate, RetrievalHit
from online.context import RunContext
from online.dependencies import OnlineDependencies
from online.parsing import (
    require_exact_keys,
    require_number,
    require_string,
)
from online.prompts import rerank_messages

_FIELDS = ("score", "confidence", "rationale_de")
_MAX_CONCURRENCY = 4


async def _score_one(
    hit: RetrievalHit,
    *,
    context: RunContext,
    views: QueryViews,
    dependencies: OnlineDependencies,
    semaphore: asyncio.Semaphore,
) -> tuple[RankedCandidate, LLMResponse]:
    if hit.document is None:
        raise RuntimeError(f"Dense hit {hit.doc_ref} has no document payload.")
    async with semaphore:
        response = await dependencies.provider.complete(
            rerank_messages(
                query=context.record.query,
                views=views,
                document=hit.document,
            ),
            model=context.model,
            purpose="rerank",
            metadata={"doc_ref": hit.doc_ref},
            json_response=True,
            max_tokens=350,
            temperature=0.0,
        )
    parsed = parse_json_object(response.content)
    require_exact_keys(parsed, _FIELDS, purpose="rerank")
    citations = dependencies.vocabulary.require_all(hit.document.citations)
    return (
        RankedCandidate(
            doc_ref=hit.doc_ref,
            rank=hit.rank,
            score=require_number(
                parsed["score"],
                field="score",
                minimum=0.0,
                maximum=10.0,
            ),
            confidence=require_number(
                parsed["confidence"],
                field="confidence",
                minimum=0.0,
                maximum=1.0,
            ),
            rationale=require_string(
                parsed["rationale_de"],
                field="rationale_de",
            ),
            citations=citations,
            text=hit.document.text,
            court=hit.document.court,
            decision_id=hit.document.decision_id,
            docket_number=hit.document.docket_number,
            decision_date=hit.document.decision_date,
        ),
        response,
    )


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Score configured dense candidates and apply stable tie-breaking."""
    if context.query_views is None or not context.dense_hits:
        raise RuntimeError("Stage 3 must complete before reranking.")
    views = context.query_views
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    scored = await asyncio.gather(
        *(
            _score_one(
                hit,
                context=context,
                views=views,
                dependencies=dependencies,
                semaphore=semaphore,
            )
            for hit in context.dense_hits[: dependencies.settings.rerank_n]
        )
    )
    candidates = [candidate for candidate, _ in scored]
    candidates.sort(key=lambda item: (-item.score, item.rank, item.doc_ref))
    context.reranked = [
        RankedCandidate(
            doc_ref=item.doc_ref,
            rank=rank,
            score=item.score,
            confidence=item.confidence,
            rationale=item.rationale,
            citations=item.citations,
            text=item.text,
            court=item.court,
            decision_id=item.decision_id,
            docket_number=item.docket_number,
            decision_date=item.decision_date,
        )
        for rank, item in enumerate(candidates, start=1)
    ]
    context.usage_total_tokens += sum(response.usage_total_tokens for _, response in scored)
    return {
        "kind": "reranking",
        "model": context.model,
        "before": [
            {"rank": hit.rank, "doc_ref": hit.doc_ref}
            for hit in context.dense_hits[: dependencies.settings.rerank_n]
        ],
        "after": [
            {
                "doc_ref": candidate.doc_ref,
                "court": candidate.court,
                "decision_id": candidate.decision_id,
                "docket_number": candidate.docket_number,
                "decision_date": candidate.decision_date,
                "snippet": candidate.text[:700],
                "rank": candidate.rank,
                "score": candidate.score,
                "score_kind": "reranker",
                "confidence": candidate.confidence,
                "rationale_de": candidate.rationale,
            }
            for candidate in context.reranked
        ],
        "top_select": dependencies.settings.top_select,
    }
