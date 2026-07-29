"""Deterministic query views used by the audited retrieval configuration."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_CITATION_FRAGMENT_PATTERN = re.compile(
    r"\b(?:"
    r"Art\.\s*[^,;.()]{1,80}"
    r"|BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.a-z]+)?"
    r"|(?:\d[A-Z]?|[A-Z])_[0-9]+/[0-9]{4}"
    r"(?:\s+E\.\s*[\d.a-z]+)?"
    r")"
)
_TOKEN_PATTERN = re.compile(r"[A-Za-zÄÖÜäöüÉéÈèÀàÇç]{4,}|\d[A-Z]?_[0-9]+/[0-9]{4}")
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"(?<=[?;.])\s+")

# Order is significant: the configured metadata view preserves this order.
LEGAL_WORDS = (
    "vorsorgliche Massnahmen",
    "superprovisorische Massnahmen",
    "Beweissicherung",
    "Zuständigkeit",
    "internationale Zuständigkeit",
    "Urheberrecht",
    "UWG",
    "unlauterer Wettbewerb",
    "Geschäftsgeheimnis",
    "Software",
    "Quellcode",
    "Mietrecht",
    "Leasing",
    "Verjährung",
    "Schuldanerkennung",
    "Kontokorrent",
    "Scheidung",
    "Kindesschutz",
    "Unterhalt",
    "Arbeitsfähigkeit",
    "Invalidenversicherung",
    "Unfall",
    "Haftpflicht",
    "SVG",
    "StPO",
    "Untersuchungshaft",
    "Kollusionsgefahr",
    "Verhältnismässigkeit",
    "Beschwerde",
    "Rechtsöffnung",
    "SchKG",
    "Erwägung",
    "Berufung",
    "Revision",
    "Ausstand",
    "Kündigung",
    "Arbeitsvertrag",
    "einfache Gesellschaft",
    "Liquidation",
    "Erbrecht",
    "Nachlass",
    "Markenschutz",
    "Datenschutz",
)
_KEYWORD_STOPWORDS = frozenset(
    [
        "that",
        "with",
        "from",
        "into",
        "under",
        "when",
        "while",
        "have",
        "been",
        "were",
        "this",
        "does",
        "should",
        "could",
        "would",
        "their",
        "there",
        "where",
        "after",
        "before",
        "alleged",
        "alleges",
        "claimant",
        "defendant",
        "plaintiff",
        "court",
        "legal",
        "risk",
        "claim",
        "claims",
        "order",
        "decision",
        "dated",
        "later",
        "whether",
        "meaning",
        "considering",
        "because",
        "between",
        "against",
        "upon",
        "within",
        "which",
        "those",
        "these",
    ]
)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _citation_fragments(query: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(0) for match in _CITATION_FRAGMENT_PATTERN.finditer(query))
    )


def make_meta_searchterm(query: str) -> str:
    """Reproduce the configured citation, legal-word, head, and tail view."""
    cleaned = _clean(query)
    if not cleaned:
        raise ValueError("query must not be empty.")
    citations = "; ".join(_citation_fragments(cleaned))
    lowered = cleaned.lower()
    legal_words = [word for word in LEGAL_WORDS if word.lower() in lowered]
    parts = _CLAUSE_BOUNDARY_PATTERN.split(cleaned)
    legal_tail = " ".join(parts[-3:]) if len(parts) > 3 else cleaned
    head = " ".join(parts[:2])
    components = (
        citations,
        ", ".join(legal_words),
        head[:700],
        legal_tail[:900],
    )
    return "; ".join(component for component in components if component)[:2_200]


def make_keyword_view(query: str) -> str:
    """Reproduce citation-prefixed frequency keywords with stable ties."""
    cleaned = _clean(query)
    if not cleaned:
        raise ValueError("query must not be empty.")
    frequencies: dict[str, int] = {}
    for token in _TOKEN_PATTERN.findall(cleaned):
        normalized = token.lower()
        if normalized in _KEYWORD_STOPWORDS or len(normalized) < 4:
            continue
        frequencies[normalized] = frequencies.get(normalized, 0) + 1
    top = sorted(
        frequencies,
        key=lambda token: (-frequencies[token], token),
    )[:70]
    return _clean(" ".join((*_citation_fragments(cleaned), *top)))[:1_600]


def make_citation_view(query: str) -> str:
    """Return configured raw citation fragments, or the query when none exist."""
    if not _clean(query):
        raise ValueError("query must not be empty.")
    citations = _clean(" ".join(_citation_fragments(query)))
    return citations or query


@dataclass(frozen=True, slots=True)
class QueryViews:
    """Five stable textual views expected by the weighted retrieval index."""

    normal_query: str
    meta_searchterm: str
    keywords: str
    fulltext: str
    citations: str

    def as_mapping(self) -> dict[str, str]:
        """Return views with stable field names used by index manifests."""
        return {
            "normal_query": self.normal_query,
            "meta_searchterm": self.meta_searchterm,
            "keywords": self.keywords,
            "fulltext": self.fulltext,
            "citations": self.citations,
        }


def build_query_views(
    query: str,
    *,
    meta_searchterm: str | None = None,
    keywords: Iterable[str] | None = None,
) -> QueryViews:
    """Build configured views, with explicit online overrides when supplied."""
    if not _clean(query):
        raise ValueError("query must not be empty.")
    if meta_searchterm is None:
        meta_view = make_meta_searchterm(query)
    else:
        meta_view = _clean(meta_searchterm)
        if not meta_view:
            raise ValueError("meta_searchterm override must not be empty.")
        meta_view = meta_view[:2_200]
    if keywords is None:
        keyword_view = make_keyword_view(query)
    else:
        cleaned_keywords = (_clean(keyword) for keyword in keywords)
        keyword_values = tuple(dict.fromkeys(keyword for keyword in cleaned_keywords if keyword))
        if not keyword_values:
            raise ValueError("keywords override must not be empty.")
        keyword_view = _clean(" ".join(keyword_values))[:1_600]
    return QueryViews(
        normal_query=query,
        meta_searchterm=meta_view,
        keywords=keyword_view,
        fulltext=query,
        citations=make_citation_view(query),
    )
