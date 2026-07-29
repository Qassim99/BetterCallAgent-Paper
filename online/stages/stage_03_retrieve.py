"""Stage 3: replay versioned dense/BM25 rankings and fuse them with weighted RRF."""

from __future__ import annotations

from typing import Any

from bettercallagent.retrieval.rrf import weighted_rrf
from bettercallagent.schemas import RetrievalHit
from online.context import RunContext
from online.dependencies import OnlineDependencies


def _public_hit(hit: RetrievalHit) -> dict[str, Any]:
    """Serialize only reviewer-safe ranking-replay metadata."""
    document = hit.document
    return {
        "doc_ref": hit.doc_ref,
        "court": document.court if document else None,
        "decision_id": document.decision_id if document else None,
        "docket_number": document.docket_number if document else None,
        "decision_date": document.decision_date if document else None,
        "snippet": document.text[:700] if document else "",
        "rank": hit.rank,
        "score": round(hit.score, 8),
        "score_kind": hit.score_kind,
    }


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Replay artifact-backed rankings without querying an index or gold labels."""
    if context.query_views is None:
        raise RuntimeError("Stage 2 must complete before retrieval.")
    limit = dependencies.settings.retrieve_k
    context.dense_hits = dependencies.repository.ranked_hits(
        query_id=context.record.query_id,
        source="dense",
        limit=limit,
    )
    context.bm25_hits = dependencies.repository.ranked_hits(
        query_id=context.record.query_id,
        source="bm25",
        limit=limit,
    )
    context.fused_hits = weighted_rrf(
        {"dense": context.dense_hits, "bm25": context.bm25_hits},
        {"dense": 0.7, "bm25": 0.3},
        k=60,
        limit=limit,
    )
    return {
        "kind": "retrieval",
        "counts": {
            "dense": len(context.dense_hits),
            "bm25": len(context.bm25_hits),
            "hybrid_unique": len(context.fused_hits),
        },
        "dense_available": bool(context.dense_hits),
        "dense": [_public_hit(hit) for hit in context.dense_hits],
        "bm25": [_public_hit(hit) for hit in context.bm25_hits],
        "hybrid": [_public_hit(hit) for hit in context.fused_hits],
    }
