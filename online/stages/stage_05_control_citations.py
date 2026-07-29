"""Stage 5: apply the fixed, deterministic citation evidence gate."""

from __future__ import annotations

from typing import Any

from bettercallagent.citations.policy import select_citations
from bettercallagent.schemas import CitationDecision
from online.context import RunContext
from online.dependencies import OnlineDependencies


def _public_decision(decision: CitationDecision) -> dict[str, str]:
    return {
        "citation": decision.citation,
        "type": decision.kind.value,
        "reason": decision.reason,
        "votes": decision.votes,
    }


async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, Any]:
    """Select citations using only reranked evidence and BM25 top-k counts."""
    if not context.reranked:
        raise RuntimeError("Stage 4 must complete before citation control.")
    bm25_counts = dependencies.repository.citation_counts(
        query_id=context.record.query_id,
        source="bm25",
        top_k=dependencies.policy.bm25_top_k,
    )
    selection = select_citations(
        context.reranked,
        bm25_counts,
        dependencies.policy,
    )
    dependencies.vocabulary.require_all(selection.predicted_citations)
    context.selection = selection
    return {
        "kind": "citation_validation",
        "rule": dependencies.policy.description,
        "qwen_rule": (
            f"dense vote >= {dependencies.policy.minimum_dense_votes} OR "
            f"top-{dependencies.policy.anchor_top_k} score >= "
            f"{dependencies.policy.anchor_minimum_score:g}"
        ),
        "bm25_rule": (
            f"BM25 support >= {dependencies.policy.minimum_bm25_votes} "
            f"within top-{dependencies.policy.bm25_top_k}"
        ),
        "accepted": [_public_decision(item) for item in selection.accepted],
        "rejected": [_public_decision(item) for item in selection.rejected],
        "predicted_citations": list(selection.predicted_citations),
        "bm25_support": {
            "top_k": dependencies.policy.bm25_top_k,
            "min_votes": dependencies.policy.minimum_bm25_votes,
            "counts": dict(selection.bm25_counts),
        },
    }
