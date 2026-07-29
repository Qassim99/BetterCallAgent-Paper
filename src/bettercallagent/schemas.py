"""Framework-independent typed records shared by online and offline pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CitationKind(StrEnum):
    """Closed set of supported citation families."""

    LAW = "law"
    COURT = "court"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One sanitized conversation message."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported chat role: {self.role!r}.")
        if not self.content.strip():
            raise ValueError("Chat message content must not be empty.")


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """A query that is backed by explicitly configured retrieval artifacts."""

    query_id: str
    query: str
    split: str

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.query.strip() or not self.split.strip():
            raise ValueError("Query identifiers, text, and split must be non-empty.")


@dataclass(frozen=True, slots=True)
class Document:
    """One retrievable legal document and its closed-vocabulary citations."""

    doc_ref: str
    text: str
    citations: tuple[str, ...]
    court: str | None = None
    decision_id: str | None = None
    docket_number: str | None = None
    decision_date: str | None = None

    def __post_init__(self) -> None:
        if not self.doc_ref.strip():
            raise ValueError("doc_ref must not be empty.")
        if not self.text.strip():
            raise ValueError(f"Document {self.doc_ref!r} has no text.")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError(f"Document {self.doc_ref!r} contains duplicate citations.")
        if any(not citation.strip() for citation in self.citations):
            raise ValueError(f"Document {self.doc_ref!r} contains an empty citation.")


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A ranked retrieval result."""

    doc_ref: str
    score: float
    score_kind: str
    rank: int
    sources: tuple[str, ...] = ()
    document: Document | None = None

    def __post_init__(self) -> None:
        if not self.doc_ref.strip():
            raise ValueError("Retrieval hits require a document reference.")
        if self.rank <= 0:
            raise ValueError("Retrieval ranks are one-based and must be positive.")


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """A retrieved document after verifier reranking."""

    doc_ref: str
    rank: int
    score: float
    citations: tuple[str, ...]
    text: str = ""
    confidence: float = 0.0
    rationale: str = ""
    court: str | None = None
    decision_id: str | None = None
    docket_number: str | None = None
    decision_date: str | None = None

    def __post_init__(self) -> None:
        if not self.doc_ref.strip():
            raise ValueError("Ranked candidates require a document reference.")
        if self.rank <= 0:
            raise ValueError("Candidate rank must be positive.")
        if not 0.0 <= self.score <= 10.0:
            raise ValueError("Candidate scores must be in the inclusive range [0, 10].")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Candidate confidence must be in the inclusive range [0, 1].")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError(f"Candidate {self.doc_ref!r} contains duplicate citations.")


@dataclass(frozen=True, slots=True)
class CitationDecision:
    """An auditable accept or reject decision for one citation."""

    citation: str
    kind: CitationKind
    reason: str
    votes: str = ""


@dataclass(frozen=True, slots=True)
class CitationSelection:
    """Deterministic output of the evidence-gated citation policy."""

    accepted: tuple[CitationDecision, ...]
    rejected: tuple[CitationDecision, ...]
    dense_vote_counts: Mapping[str, int]
    bm25_counts: Mapping[str, int]

    @property
    def predicted_citations(self) -> tuple[str, ...]:
        """Return accepted canonical citations in deterministic lexical order."""
        return tuple(sorted(decision.citation for decision in self.accepted))


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response from a chat-completion provider."""

    content: str
    model: str
    usage_total_tokens: int = 0
    raw: Mapping[str, Any] | None = None
