"""Memory-bounded citation-vocabulary loading for offline runs."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from bettercallagent.citations.vocabulary import InMemoryCitationVocabulary
from bettercallagent.schemas import CitationKind


def _open(path: Path) -> TextIO:
    if not path.is_file():
        raise FileNotFoundError(f"Missing citation vocabulary: {path}")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _citations(path: Path) -> Iterator[str]:
    with _open(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "citation" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'citation' column.")
        for row in reader:
            citation = (row.get("citation") or "").strip()
            if citation:
                yield citation


def load_targeted_vocabulary(
    *,
    laws_path: Path,
    courts_path: Path,
    candidates: set[str],
) -> InMemoryCitationVocabulary:
    """Scan large CSV vocabularies but retain only extracted candidate strings."""

    if not candidates:
        raise ValueError("At least one extracted citation candidate is required.")
    entries: dict[str, CitationKind] = {}
    remaining = set(candidates)
    for citation in _citations(laws_path):
        if citation in remaining:
            entries[citation] = CitationKind.LAW
            remaining.remove(citation)
            if not remaining:
                break
    if remaining:
        for citation in _citations(courts_path):
            if citation in remaining:
                entries[citation] = CitationKind.COURT
                remaining.remove(citation)
                if not remaining:
                    break
    if not entries:
        raise ValueError("None of the extracted citations occur in the closed vocabulary.")
    return InMemoryCitationVocabulary(entries=entries)
