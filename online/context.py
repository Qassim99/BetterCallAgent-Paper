"""Mutable state passed through the six online pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bettercallagent.citations.output_validation import AnswerCitationValidation
from bettercallagent.retrieval.query_views import QueryViews
from bettercallagent.schemas import (
    ChatMessage,
    CitationSelection,
    QueryRecord,
    RankedCandidate,
    RetrievalHit,
)


@dataclass(slots=True)
class RunContext:
    """All request-local state for one pipeline run."""

    run_id: str
    record: QueryRecord
    model: str
    history: tuple[ChatMessage, ...] = ()
    understanding: dict[str, Any] | None = None
    query_views: QueryViews | None = None
    dense_hits: list[RetrievalHit] = field(default_factory=list)
    bm25_hits: list[RetrievalHit] = field(default_factory=list)
    fused_hits: list[RetrievalHit] = field(default_factory=list)
    reranked: list[RankedCandidate] = field(default_factory=list)
    selection: CitationSelection | None = None
    answer: str | None = None
    answer_validation: AnswerCitationValidation | None = None
    usage_total_tokens: int = 0
