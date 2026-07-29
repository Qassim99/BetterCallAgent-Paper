"""Exact post-validation for citations emitted by answer-generation models."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.vocabulary import CitationVocabulary


class OutputCitationError(ValueError):
    """Raised when a generated answer contains an unsupported citation."""


@dataclass(frozen=True, slots=True)
class AnswerCitationValidation:
    """Auditable result of final-answer citation validation."""

    mentioned: tuple[str, ...]
    exact_mentioned: tuple[str, ...]
    accepted: tuple[str, ...]


_BROAD_ARTICLE_SURFACE = re.compile(
    r"\bArt\.\s*[0-9]+[a-z]?(?:/[0-9]+)?"
    r"(?:\s+(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)"
    r"\s*[0-9a-z]+)*"
    r"\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü.]{1,15}\b"
)
_PARENT_LAW = re.compile(r"^(?P<prefix>Art\.\s+\S+\s+Abs\.\s+\S+)\s+(?P<code>\S+)$")


def _is_derived_parent(citation: str, exact_mentions: Collection[str]) -> bool:
    """Allow the extractor's documented parent for an exact ``lit.`` citation."""

    match = _PARENT_LAW.fullmatch(citation)
    if match is None:
        return False
    prefix = re.escape(match.group("prefix"))
    code = re.escape(match.group("code"))
    child = re.compile(rf"^{prefix}\s+lit\.\s+\S+\s+{code}$")
    return any(child.fullmatch(exact) for exact in exact_mentions)


def validate_answer_citations(
    answer: str,
    *,
    accepted_citations: Collection[str],
    vocabulary: CitationVocabulary,
    extractor: CitationExtractor | None = None,
) -> AnswerCitationValidation:
    """Require every generated citation to be both corpus-valid and selected.

    The function never deletes, rewrites, or silently ignores an unsupported
    citation. A bad model response fails loudly so it cannot reach the user.
    """
    extractor = extractor or CitationExtractor()
    accepted = vocabulary.require_all(accepted_citations)
    mentioned = extractor.extract(answer)
    exact_mentions = tuple(citation for citation in mentioned if citation in answer)
    outside_vocabulary = tuple(citation for citation in mentioned if citation not in vocabulary)
    if outside_vocabulary:
        raise OutputCitationError(
            "The generated answer contains citations outside the configured vocabulary: "
            + ", ".join(outside_vocabulary)
        )
    accepted_set = set(accepted)
    unsupported = tuple(citation for citation in mentioned if citation not in accepted_set)
    if unsupported:
        raise OutputCitationError(
            "The generated answer contains citations not accepted by the evidence gate: "
            + ", ".join(unsupported)
        )
    non_exact = tuple(
        citation
        for citation in mentioned
        if citation not in exact_mentions and not _is_derived_parent(citation, exact_mentions)
    )
    if non_exact:
        raise OutputCitationError(
            "The generated answer did not use the exact configured citation "
            "string for: " + ", ".join(non_exact)
        )
    broad_surfaces = tuple(
        " ".join(match.group(0).split()) for match in _BROAD_ARTICLE_SURFACE.finditer(answer)
    )
    unrecognized = tuple(surface for surface in broad_surfaces if surface not in exact_mentions)
    if unrecognized:
        raise OutputCitationError(
            "The generated answer contains a citation surface outside the "
            "configured vocabulary: " + ", ".join(unrecognized)
        )
    return AnswerCitationValidation(
        mentioned=mentioned,
        exact_mentioned=exact_mentions,
        accepted=accepted,
    )
