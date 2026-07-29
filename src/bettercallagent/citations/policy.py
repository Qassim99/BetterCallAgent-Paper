"""Canonical deterministic citation policy used by online and offline runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bettercallagent.schemas import (
    CitationDecision,
    CitationKind,
    CitationSelection,
    RankedCandidate,
)


def _citation_kind(citation: str) -> CitationKind:
    if citation.startswith("Art. "):
        return CitationKind.LAW
    return CitationKind.COURT


@dataclass(frozen=True, slots=True)
class FixedVotePolicy:
    """Configured Step-H rule with auxiliary BM25 citation support."""

    candidate_top_k: int = 10
    minimum_dense_votes: int = 4
    anchor_top_k: int = 3
    anchor_minimum_score: float = 8.5
    bm25_top_k: int = 5
    minimum_bm25_votes: int = 2

    def __post_init__(self) -> None:
        if (
            min(
                self.candidate_top_k,
                self.minimum_dense_votes,
                self.anchor_top_k,
                self.bm25_top_k,
                self.minimum_bm25_votes,
            )
            <= 0
        ):
            raise ValueError("All vote-policy count parameters must be positive.")
        if not 0.0 <= self.anchor_minimum_score <= 10.0:
            raise ValueError("anchor_minimum_score must be in [0, 10].")

    @property
    def description(self) -> str:
        """Human-readable, stable description for traces and documentation."""
        return (
            f"dense vote >= {self.minimum_dense_votes} OR "
            f"top-{self.anchor_top_k} score >= {self.anchor_minimum_score:g}; "
            f"add BM25 support >= {self.minimum_bm25_votes}/{self.bm25_top_k}"
        )


def select_citations(
    ranked_candidates: Sequence[RankedCandidate],
    bm25_counts: Mapping[str, int],
    policy: FixedVotePolicy,
) -> CitationSelection:
    """Apply the canonical fixed-vote and BM25-support rule deterministically.

    Candidate position follows the configured verifier ordering: score
    descending, citation count descending, then retrieval rank ascending.
    """
    for citation, count in bm25_counts.items():
        if not citation.strip():
            raise ValueError("BM25 support contains an empty citation.")
        if count < 0:
            raise ValueError(f"BM25 support count cannot be negative: {citation}")

    ordered = order_candidates(ranked_candidates)[: policy.candidate_top_k]

    dense_counts: Counter[str] = Counter()
    best_evidence: dict[str, tuple[int, float]] = {}
    for position, candidate in enumerate(ordered, start=1):
        for citation in set(candidate.citations):
            dense_counts[citation] += 1
            current = best_evidence.get(citation)
            evidence = (position, candidate.score)
            if current is None or candidate.score > current[1]:
                best_evidence[citation] = evidence

    accepted_reasons: dict[str, list[str]] = {}
    all_citations = set(dense_counts) | set(bm25_counts)
    for citation in sorted(all_citations):
        dense_votes = dense_counts.get(citation, 0)
        position, best_score = best_evidence.get(citation, (10**9, -1.0))
        if dense_votes >= policy.minimum_dense_votes:
            accepted_reasons.setdefault(citation, []).append(
                f"dense vote {dense_votes}/{policy.candidate_top_k}"
            )
        if position <= policy.anchor_top_k and best_score >= policy.anchor_minimum_score:
            accepted_reasons.setdefault(citation, []).append(
                f"top-{position} score {best_score:.1f}"
            )
        bm25_votes = bm25_counts.get(citation, 0)
        if bm25_votes >= policy.minimum_bm25_votes:
            accepted_reasons.setdefault(citation, []).append(
                f"BM25 support {bm25_votes}/{policy.bm25_top_k}"
            )

    accepted = tuple(
        CitationDecision(
            citation=citation,
            kind=_citation_kind(citation),
            reason="selected by the configured evidence gate",
            votes=" + ".join(reasons),
        )
        for citation, reasons in sorted(accepted_reasons.items())
    )
    rejected = tuple(
        CitationDecision(
            citation=citation,
            kind=_citation_kind(citation),
            reason="extracted but below every evidence threshold",
            votes=(
                f"dense vote {dense_counts.get(citation, 0)}/{policy.candidate_top_k}; "
                f"BM25 support {bm25_counts.get(citation, 0)}/{policy.bm25_top_k}"
            ),
        )
        for citation in sorted(all_citations - set(accepted_reasons))
    )
    return CitationSelection(
        accepted=accepted,
        rejected=rejected,
        dense_vote_counts=dict(sorted(dense_counts.items())),
        bm25_counts=dict(sorted(bm25_counts.items())),
    )


def order_candidates(
    ranked_candidates: Sequence[RankedCandidate],
) -> list[RankedCandidate]:
    """Return the exact deterministic verifier order used by the gate."""

    return sorted(
        ranked_candidates,
        key=lambda candidate: (
            -candidate.score,
            -len(candidate.citations),
            candidate.rank,
            candidate.doc_ref,
        ),
    )
