"""Deterministic weighted reciprocal-rank fusion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from bettercallagent.schemas import RetrievalHit


def weighted_rrf(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    weights: Mapping[str, float],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[RetrievalHit]:
    """Fuse rankings with weighted reciprocal rank and stable tie-breaking."""
    if k <= 0:
        raise ValueError("RRF k must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("RRF limit must be positive when provided.")
    missing_weights = set(rankings) - set(weights)
    extra_weights = set(weights) - set(rankings)
    if missing_weights or extra_weights:
        raise ValueError(
            "Ranking and weight sources must match exactly; "
            f"missing={sorted(missing_weights)}, extra={sorted(extra_weights)}."
        )
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("RRF source weights must be positive.")

    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, set[str]] = defaultdict(set)
    exemplar: dict[str, RetrievalHit] = {}
    encounter_order: dict[str, int] = {}
    for source, hits in rankings.items():
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            if hit.doc_ref in seen:
                continue
            seen.add(hit.doc_ref)
            encounter_order.setdefault(hit.doc_ref, len(encounter_order))
            scores[hit.doc_ref] += weights[source] / (k + position)
            sources[hit.doc_ref].add(source)
            exemplar.setdefault(hit.doc_ref, hit)

    ordered = sorted(
        scores,
        key=lambda doc_ref: (-scores[doc_ref], encounter_order[doc_ref]),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [
        RetrievalHit(
            doc_ref=doc_ref,
            score=scores[doc_ref],
            score_kind="weighted_rrf",
            rank=position,
            sources=tuple(sorted(sources[doc_ref])),
            document=exemplar[doc_ref].document,
        )
        for position, doc_ref in enumerate(ordered, start=1)
    ]
