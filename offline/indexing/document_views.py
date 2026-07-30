"""Construct the five document text views embedded by the paper pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

FIELDS = ("normal_query", "meta_searchterm", "keywords", "fulltext", "citations")
FULLTEXT_FIELDS = ("regeste", "abstract_de", "abstract_fr", "abstract_it", "full_text")
CITATION_FIELDS = (
    "valid_law_citations",
    "valid_court_citations",
    "law_citations",
    "court_citations",
    "citations",
    "citation",
    "rootCitation",
    "gold_citations",
)
CITATION_PATTERN = re.compile(
    r"\b(?:Art\.\s*\d+[a-zA-Z]*(?:\s*Abs\.\s*\d+[a-zA-Z]*)?"
    r"(?:\s*(?:lit\.|Ziff\.)\s*[a-zA-Z0-9]+)?\s*[A-ZÄÖÜ]{2,}[A-ZÄÖÜ0-9-]*|"
    r"(?:BGE|ATF)\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.a-zA-Z/]+)?|"
    r"\d+[A-Z]_[0-9]+/\d{4})\b"
)


def clean(value: object) -> str:
    """Collapse whitespace without changing letters or punctuation."""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _recover_string(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return str(json.loads(f'"{match.group(1)}"'))
    except json.JSONDecodeError:
        return match.group(1).replace('\\"', '"')


def _recover_list(text: str, key: str) -> list[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text, flags=re.DOTALL)
    if match is None:
        return []
    block = text[match.end() :]
    if (closing := block.find("]")) >= 0:
        block = block[:closing]
    if next_key := re.search(r'\n\s*"[a-zA-Z_]+"\s*:', block):
        block = block[: next_key.start()]
    values = []
    for item in re.finditer(r'"((?:\\.|[^"\\])*)"', block):
        try:
            values.append(str(json.loads(f'"{item.group(1)}"')))
        except json.JSONDecodeError:
            values.append(item.group(1).replace('\\"', '"'))
    return values


def parse_generated(content: object) -> dict[str, Any]:
    """Parse JSON and recover known fields when model output was truncated."""

    if isinstance(content, Mapping):
        return dict(content)
    text = str(content or "").strip()
    if not text:
        return {}
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    recovered: dict[str, Any] = {}
    for key in ("normal_query", "meta_searchterm_de", "notes"):
        if value := _recover_string(text, key):
            recovered[key] = value
    for key in ("keywords_de", "keywords_en"):
        if values := _recover_list(text, key):
            recovered[key] = values
    return recovered


def join_list(value: object) -> str:
    """Join generated list values with the historical separator."""

    if isinstance(value, list):
        return " ; ".join(text for item in value if (text := clean(item)))
    return clean(value)


def fulltext_from_row(row: Mapping[str, Any]) -> str:
    """Render the exact labeled source fields used for full-text embeddings."""

    return "\n".join(
        f"{field}: {value}" for field in FULLTEXT_FIELDS if (value := clean(row.get(field)))
    )


def citation_text(row: Mapping[str, Any], generated: Mapping[str, Any]) -> str:
    """Collect explicit and regex-derived citations in stable first-seen order."""

    values = []
    for field in CITATION_FIELDS:
        if row.get(field):
            values.append(join_list(row[field]))
        if generated.get(field):
            values.append(join_list(generated[field]))
    source = "\n".join(clean(row.get(field)) for field in FULLTEXT_FIELDS if row.get(field))
    values.extend(CITATION_PATTERN.findall(source[:120_000]))
    return " ; ".join(dict.fromkeys(value for item in values if (value := clean(item))))


def build_texts(generated_record: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, str]:
    """Return all five views in the index manifest's canonical order."""

    generated = parse_generated(generated_record.get("content"))
    keyword_parts = (
        join_list(generated.get("keywords_de")),
        join_list(generated.get("keywords_en")),
    )
    return {
        "normal_query": clean(generated.get("normal_query")),
        "meta_searchterm": clean(generated.get("meta_searchterm_de")),
        "keywords": " ; ".join(part for part in keyword_parts if part),
        "fulltext": fulltext_from_row(row),
        "citations": citation_text(row, generated),
    }
