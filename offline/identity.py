"""Stable candidate identities shared by offline reranking and selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def candidate_id(row: Mapping[str, Any]) -> str:
    """Identify one retrieved row without conflating duplicate documents."""

    query_id = str(row.get("query_id") or "").strip()
    doc_id = str(row.get("doc_id") or "").strip()
    if not query_id or not doc_id:
        raise ValueError("Candidate records require non-empty query_id and doc_id.")

    explicit = str(row.get("candidate_id") or "").strip()
    if explicit:
        if not explicit.startswith(f"{query_id}::"):
            raise ValueError("candidate_id does not belong to its query_id.")
        return explicit

    global_index = row.get("global_idx")
    if global_index is None or str(global_index).strip() == "":
        return f"{query_id}::{doc_id}"
    return f"{query_id}::gi{int(global_index)}"
