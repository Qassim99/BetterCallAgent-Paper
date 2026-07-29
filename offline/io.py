"""Small, strict I/O helpers shared by offline stages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def require_file(path: Path, *, description: str) -> Path:
    """Return a resolved file path or raise a diagnostic error."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def read_jsonl(path: Path) -> Iterator[JsonObject]:
    """Yield JSON objects with line-aware errors."""

    resolved = require_file(path, description="JSONL input")
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: expected a JSON object")
            yield value


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write newline-delimited JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_queries(path: Path, *, require_gold: bool = False) -> dict[str, JsonObject]:
    """Load a query CSV and enforce unique non-empty identifiers."""

    resolved = require_file(path, description="query CSV")
    with resolved.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"query_id", "query"}
        if require_gold:
            required.add("gold_citations")
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{resolved} is missing columns: {', '.join(missing)}")
        output: dict[str, JsonObject] = {}
        for row_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            query = (row.get("query") or "").strip()
            if not query_id or not query:
                raise ValueError(f"{resolved}:{row_number}: query_id and query must be non-empty")
            if query_id in output:
                raise ValueError(f"{resolved}:{row_number}: duplicate query_id {query_id!r}")
            output[query_id] = dict(row)
    if not output:
        raise ValueError(f"{resolved} contains no queries")
    return output


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with require_file(path, description="file to hash").open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> tuple[str, int, int]:
    """Hash a directory as ordered relative paths plus file-content digests.

    The returned tuple contains ``(digest, file_count, total_bytes)``. Empty
    directories are rejected because they cannot represent a usable artifact.
    """

    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {root}")

    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
        total_bytes += item.stat().st_size
    return digest.hexdigest(), len(files), total_bytes


def normalize_text(value: object) -> str:
    """Normalize whitespace without changing letter case."""

    return " ".join(str(value or "").split())


def split_citations(value: object) -> set[str]:
    """Parse the competition's semicolon-separated citation set."""

    return {
        normalized for item in str(value or "").split(";") if (normalized := normalize_text(item))
    }
