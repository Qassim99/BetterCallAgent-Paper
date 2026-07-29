"""Strict reader for reproducible online retrieval assets."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.vocabulary import InMemoryCitationVocabulary
from bettercallagent.retrieval.query_views import QueryViews
from bettercallagent.schemas import (
    CitationKind,
    Document,
    QueryRecord,
    RetrievalHit,
)

_QUERY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
ASSET_SCHEMA_VERSION = 2
_RETRIEVAL_VIEW_FIELDS = (
    "normal_query",
    "meta_searchterm",
    "keywords",
    "fulltext",
    "citations",
)


class AssetError(ValueError):
    """Raised when an online asset is absent, inconsistent, or unsupported."""


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AssetError(f"{field} must be a JSON object with string keys.")
    return dict(value)


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AssetError(f"{field} must be a JSON array.")
    return list(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssetError(f"{field} must be a non-empty string.")
    return value.strip()


def _verbatim_string(value: object, *, field: str) -> str:
    """Validate text while preserving every code point for exact comparison."""
    if not isinstance(value, str) or not value.strip():
        raise AssetError(f"{field} must be a non-empty string.")
    return value


def _string_list(value: object, *, field: str) -> tuple[str, ...]:
    items = _list(value, field=field)
    result = tuple(_string(item, field=f"{field}[]") for item in items)
    if len(result) != len(set(result)):
        raise AssetError(f"{field} must not contain duplicates.")
    return result


def _normalized_query(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True, slots=True)
class FixtureScript:
    """Explicit model outputs used only in fixture mode."""

    understanding: Mapping[str, Any]
    query_plan: Mapping[str, Any]
    rerank: Mapping[str, Mapping[str, Any]]
    answer: str


@dataclass(frozen=True, slots=True)
class OnlineAssetRepository:
    """Validated queries, documents, views, and precomputed ranking artifacts."""

    queries: Mapping[str, QueryRecord]
    documents: Mapping[str, Document]
    retrieval_views: Mapping[str, QueryViews]
    dense_rankings: Mapping[str, tuple[str, ...]]
    bm25_rankings: Mapping[str, tuple[str, ...]]
    fixture_script: FixtureScript | None

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        extractor: CitationExtractor | None = None,
    ) -> OnlineAssetRepository:
        """Load one versioned asset file and validate every cross-reference."""
        if not path.is_file():
            raise FileNotFoundError(f"Online asset file does not exist: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssetError(f"Online asset is not valid JSON: {path}") from exc
        root = _object(raw, field="root")
        if root.get("version") != ASSET_SCHEMA_VERSION:
            raise AssetError(f"Online asset version must be exactly {ASSET_SCHEMA_VERSION}.")

        extractor = extractor or CitationExtractor()
        queries: dict[str, QueryRecord] = {}
        for index, item in enumerate(_list(root.get("queries"), field="queries")):
            row = _object(item, field=f"queries[{index}]")
            query_id = _string(row.get("query_id"), field=f"queries[{index}].query_id")
            if not _QUERY_ID_PATTERN.fullmatch(query_id):
                raise AssetError(f"Invalid query_id: {query_id!r}.")
            if query_id in queries:
                raise AssetError(f"Duplicate query_id: {query_id}.")
            queries[query_id] = QueryRecord(
                query_id=query_id,
                query=_string(row.get("query"), field=f"queries[{index}].query"),
                split=_string(row.get("split"), field=f"queries[{index}].split"),
            )
        if not queries:
            raise AssetError("Online assets must contain at least one query.")

        documents: dict[str, Document] = {}
        for index, item in enumerate(_list(root.get("documents"), field="documents")):
            row = _object(item, field=f"documents[{index}]")
            doc_ref = _string(row.get("doc_ref"), field=f"documents[{index}].doc_ref")
            if doc_ref in documents:
                raise AssetError(f"Duplicate doc_ref: {doc_ref}.")
            text = _string(row.get("text"), field=f"documents[{index}].text")
            extracted = extractor.extract(text)
            declared_raw = row.get("citations")
            if declared_raw is not None:
                declared = tuple(
                    sorted(
                        _string_list(
                            declared_raw,
                            field=f"documents[{index}].citations",
                        )
                    )
                )
                if declared != extracted:
                    raise AssetError(
                        f"Declared citations for {doc_ref} do not exactly match "
                        "deterministic extraction from its text."
                    )
            documents[doc_ref] = Document(
                doc_ref=doc_ref,
                text=text,
                citations=extracted,
                court=_optional_string(row.get("court"), field=f"documents[{index}].court"),
                decision_id=_optional_string(
                    row.get("decision_id"),
                    field=f"documents[{index}].decision_id",
                ),
                docket_number=_optional_string(
                    row.get("docket_number"),
                    field=f"documents[{index}].docket_number",
                ),
                decision_date=_optional_string(
                    row.get("decision_date"),
                    field=f"documents[{index}].decision_date",
                ),
            )
        if not documents:
            raise AssetError("Online assets must contain at least one document.")

        retrieval_views = _parse_retrieval_views(
            root.get("retrieval_views"),
            queries=queries,
        )
        dense = _parse_rankings(
            root.get("dense_rankings"),
            field="dense_rankings",
            queries=queries,
            documents=documents,
        )
        bm25 = _parse_rankings(
            root.get("bm25_rankings"),
            field="bm25_rankings",
            queries=queries,
            documents=documents,
        )
        script_raw = root.get("fixture_script")
        fixture_script = (
            _parse_fixture_script(script_raw, documents=documents)
            if script_raw is not None
            else None
        )
        return cls(
            queries=queries,
            documents=documents,
            retrieval_views=retrieval_views,
            dense_rankings=dense,
            bm25_rankings=bm25,
            fixture_script=fixture_script,
        )

    def resolve_query(self, *, query: str, query_id: str | None) -> QueryRecord:
        """Resolve a request to one exact artifact-backed query."""
        normalized = _normalized_query(query)
        if not normalized:
            raise AssetError("query must not be empty.")
        if query_id is not None:
            record = self.queries.get(query_id)
            if record is None:
                raise AssetError("query_id is not present in the configured asset.")
            if _normalized_query(record.query) != normalized:
                raise AssetError("query does not match the configured query_id.")
            return record
        matches = [
            record
            for record in self.queries.values()
            if _normalized_query(record.query) == normalized
        ]
        if len(matches) != 1:
            raise AssetError(
                "query must exactly match one configured query when query_id is omitted."
            )
        return matches[0]

    def ranked_hits(
        self,
        *,
        query_id: str,
        source: str,
        limit: int,
    ) -> list[RetrievalHit]:
        """Replay deterministic one-based hits from a configured ranking artifact."""
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if source == "dense":
            ranking = self.dense_rankings.get(query_id)
        elif source == "bm25":
            ranking = self.bm25_rankings.get(query_id)
        else:
            raise ValueError(f"Unsupported retrieval source: {source!r}.")
        if ranking is None:
            raise AssetError(f"No {source} ranking exists for query_id {query_id}.")
        return [
            RetrievalHit(
                doc_ref=doc_ref,
                score=1.0 / rank,
                score_kind=f"{source}_reciprocal_rank",
                rank=rank,
                sources=(source,),
                document=self.documents[doc_ref],
            )
            for rank, doc_ref in enumerate(ranking[:limit], start=1)
        ]

    def citation_counts(
        self,
        *,
        query_id: str,
        source: str,
        top_k: int,
    ) -> dict[str, int]:
        """Count document citations in a retriever's top-k evidence."""
        counts: Counter[str] = Counter()
        for hit in self.ranked_hits(query_id=query_id, source=source, limit=top_k):
            if hit.document is None:
                raise AssetError(f"Hit {hit.doc_ref} has no document payload.")
            counts.update(set(hit.document.citations))
        return dict(sorted(counts.items()))

    def vocabulary(self) -> InMemoryCitationVocabulary:
        """Build a closed vocabulary from citations extracted from corpus text."""
        entries: dict[str, CitationKind] = {}
        for document in self.documents.values():
            for citation in document.citations:
                kind = CitationKind.LAW if citation.startswith("Art. ") else CitationKind.COURT
                previous = entries.setdefault(citation, kind)
                if previous is not kind:
                    raise AssetError(f"Inconsistent citation kind for {citation}.")
        if not entries:
            raise AssetError("No citations could be extracted from the online corpus.")
        return InMemoryCitationVocabulary(entries=entries)

    def public_queries(self, *, public_split: str | None = None) -> list[dict[str, object]]:
        """Return stable non-sensitive metadata for the query picker."""
        return [
            {
                "query_id": record.query_id,
                "query": record.query,
                "split": public_split or record.split,
                "has_dense": record.query_id in self.dense_rankings,
            }
            for record in sorted(self.queries.values(), key=lambda item: item.query_id)
        ]


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _parse_retrieval_views(
    value: object,
    *,
    queries: Mapping[str, QueryRecord],
) -> dict[str, QueryViews]:
    """Parse exact query views that bind each saved ranking to its inputs."""
    raw = _object(value, field="retrieval_views")
    if set(raw) != set(queries):
        raise AssetError("retrieval_views keys must exactly match configured query IDs.")
    parsed: dict[str, QueryViews] = {}
    expected_fields = set(_RETRIEVAL_VIEW_FIELDS)
    for query_id, views_raw in raw.items():
        views = _object(views_raw, field=f"retrieval_views.{query_id}")
        if set(views) != expected_fields:
            missing = sorted(expected_fields - set(views))
            unknown = sorted(set(views) - expected_fields)
            raise AssetError(
                f"retrieval_views.{query_id} fields differ; missing={missing}, unknown={unknown}."
            )
        parsed[query_id] = QueryViews(
            normal_query=_verbatim_string(
                views["normal_query"],
                field=f"retrieval_views.{query_id}.normal_query",
            ),
            meta_searchterm=_verbatim_string(
                views["meta_searchterm"],
                field=f"retrieval_views.{query_id}.meta_searchterm",
            ),
            keywords=_verbatim_string(
                views["keywords"],
                field=f"retrieval_views.{query_id}.keywords",
            ),
            fulltext=_verbatim_string(
                views["fulltext"],
                field=f"retrieval_views.{query_id}.fulltext",
            ),
            citations=_verbatim_string(
                views["citations"],
                field=f"retrieval_views.{query_id}.citations",
            ),
        )
    return parsed


def _parse_rankings(
    value: object,
    *,
    field: str,
    queries: Mapping[str, QueryRecord],
    documents: Mapping[str, Document],
) -> dict[str, tuple[str, ...]]:
    raw = _object(value, field=field)
    if set(raw) != set(queries):
        raise AssetError(f"{field} keys must exactly match configured query IDs.")
    parsed: dict[str, tuple[str, ...]] = {}
    for query_id, ranking_raw in raw.items():
        ranking = _string_list(ranking_raw, field=f"{field}.{query_id}")
        missing = set(ranking) - set(documents)
        if missing:
            raise AssetError(f"{field}.{query_id} references unknown documents: {sorted(missing)}.")
        if not ranking:
            raise AssetError(f"{field}.{query_id} must not be empty.")
        parsed[query_id] = ranking
    return parsed


def _parse_fixture_script(
    value: object,
    *,
    documents: Mapping[str, Document],
) -> FixtureScript:
    raw = _object(value, field="fixture_script")
    understanding = _object(
        raw.get("understanding"),
        field="fixture_script.understanding",
    )
    query_plan = _object(raw.get("query_plan"), field="fixture_script.query_plan")
    rerank_raw = _object(raw.get("rerank"), field="fixture_script.rerank")
    if set(rerank_raw) != set(documents):
        raise AssetError("fixture_script.rerank keys must exactly match configured document refs.")
    rerank = {
        doc_ref: _object(details, field=f"fixture_script.rerank.{doc_ref}")
        for doc_ref, details in rerank_raw.items()
    }
    return FixtureScript(
        understanding=understanding,
        query_plan=query_plan,
        rerank=rerank,
        answer=_string(raw.get("answer"), field="fixture_script.answer"),
    )
