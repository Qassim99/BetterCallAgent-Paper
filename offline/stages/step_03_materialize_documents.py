"""Stage 3: materialize full document text for retrieved Parquet rows."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from offline.io import JsonObject, atomic_write_jsonl, read_jsonl, require_file

DOCUMENT_REFERENCE = re.compile(
    r"^(?P<file>[^:]+[.]parquet):rg(?P<row_group>[0-9]+):row(?P<row>[0-9]+)$"
)
PREFIX_FIELDS = (
    "decision_id",
    "docket_number",
    "docket_number_2",
    "decision_date",
    "publication_date",
    "court",
    "canton",
    "chamber",
    "language",
    "title",
    "legal_area",
    "regeste",
    "abstract_de",
    "abstract_fr",
    "abstract_it",
    "outcome",
    "decision_type",
    "appeal_info",
    "source_url",
    "pdf_url",
    "bge_reference",
    "cited_decisions",
)


def document_reference(candidate: JsonObject) -> str:
    """Create the stable Parquet row reference used throughout the pipeline."""

    direct = str(candidate.get("doc_id") or "").strip()
    if direct:
        return direct
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Candidate has neither a valid doc_id nor metadata locator.")
    source = metadata.get("source_parquet")
    row_group = metadata.get("row_group")
    row = metadata.get("row_index_in_group")
    if source is None or row_group is None or row is None:
        raise ValueError("Candidate metadata lacks source_parquet/row_group/row_index_in_group.")
    return f"{Path(str(source)).name}:rg{int(row_group)}:row{int(row)}"


def render_document(row: dict[str, Any]) -> str:
    """Render metadata and full text with explicit field labels."""

    parts = [
        f"[{field}] {value}"
        for field in PREFIX_FIELDS
        if (value := row.get(field)) is not None and str(value).strip()
    ]
    if (full_text := row.get("full_text")) is not None and str(full_text).strip():
        parts.append(f"[full_text] {full_text}")
    return " ".join(parts)


def collect_locations(
    candidates_path: Path,
    *,
    top_n_per_query: int,
) -> dict[str, tuple[str, int, int]]:
    """Collect only reranker-bound references and reject conflicting locators."""

    if top_n_per_query <= 0:
        raise ValueError("top_n_per_query must be positive.")
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for candidate in read_jsonl(candidates_path):
        query_id = str(candidate.get("query_id") or "").strip()
        if not query_id:
            raise ValueError("Candidate is missing query_id.")
        grouped[query_id].append(candidate)
    if not grouped:
        raise ValueError("Candidate file contains no query records.")
    locations: dict[str, tuple[str, int, int]] = {}
    for query_id, candidates in grouped.items():
        candidates.sort(key=lambda row: int(row.get("rank") or 10**9))
        for candidate in candidates[:top_n_per_query]:
            reference = document_reference(candidate)
            match = DOCUMENT_REFERENCE.fullmatch(reference)
            if match is None:
                raise ValueError(f"Invalid document reference: {reference}")
            location = (
                match.group("file"),
                int(match.group("row_group")),
                int(match.group("row")),
            )
            previous = locations.setdefault(reference, location)
            if previous != location:
                raise ValueError(f"Conflicting locators for {reference}")
        if not candidates:
            raise ValueError(f"No candidates for query {query_id!r}.")
    return locations


def materialize(
    candidates_path: Path,
    parquet_root: Path,
    *,
    top_n_per_query: int,
) -> list[JsonObject]:
    """Read every requested row exactly once and fail on missing data."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Stage 3 requires PyArrow. Install the `offline-gpu` dependency group."
        ) from exc

    locations = collect_locations(
        candidates_path,
        top_n_per_query=top_n_per_query,
    )
    by_file: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for reference, (filename, row_group, row) in locations.items():
        by_file[filename].append((row_group, row, reference))

    output: list[JsonObject] = []
    for filename, items in sorted(by_file.items()):
        path = require_file(parquet_root / filename, description="source Parquet file")
        parquet_file = parquet.ParquetFile(path)
        by_row_group: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for row_group, row, reference in items:
            by_row_group[row_group].append((row, reference))
        for row_group, requested_rows in sorted(by_row_group.items()):
            if row_group >= parquet_file.num_row_groups:
                raise IndexError(f"{filename}: missing row group {row_group}")
            rows = parquet_file.read_row_group(row_group).to_pylist()
            for row_index, reference in sorted(requested_rows):
                if row_index >= len(rows):
                    raise IndexError(f"{filename}: row group {row_group} has no row {row_index}")
                row = rows[row_index]
                text = render_document(row)
                if not text:
                    raise ValueError(f"{reference}: materialized document is empty")
                output.append(
                    {
                        "doc_id": reference,
                        "full_document_text": text,
                        "full_text_length": len(str(row.get("full_text") or "")),
                        "document_length": len(text),
                        "decision_id": row.get("decision_id"),
                        "docket_number": row.get("docket_number"),
                        "decision_date": row.get("decision_date"),
                        "court": row.get("court"),
                        "source_url": row.get("source_url"),
                        "pdf_url": row.get("pdf_url"),
                    }
                )
    if len(output) != len(locations):
        raise RuntimeError(f"Materialized {len(output)} of {len(locations)} requested documents.")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = materialize(
        args.candidates,
        args.parquet_root.expanduser().resolve(),
        top_n_per_query=args.top_n,
    )
    atomic_write_jsonl(args.output, rows)
    print(f"Materialized {len(rows)} documents -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
