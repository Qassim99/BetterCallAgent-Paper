"""Closed-vocabulary abstractions for exact citation validation."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from bettercallagent.schemas import CitationKind


class CitationVocabulary(Protocol):
    """Interface required by online and offline citation validation."""

    def __contains__(self, citation: object) -> bool:
        """Return whether a citation exists exactly in the configured corpus."""
        ...

    def kind_of(self, citation: str) -> CitationKind:
        """Return the corpus family for an exact citation."""
        ...

    def require_all(self, citations: Iterable[str]) -> tuple[str, ...]:
        """Validate citations and return them in deterministic order."""
        ...


@dataclass(frozen=True, slots=True)
class InMemoryCitationVocabulary:
    """Small exact vocabulary suitable for fixtures and unit tests."""

    entries: Mapping[str, CitationKind]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("Citation vocabularies must contain at least one entry.")
        if any(not citation.strip() for citation in self.entries):
            raise ValueError("Citation vocabularies cannot contain empty entries.")

    def __contains__(self, citation: object) -> bool:
        return isinstance(citation, str) and citation in self.entries

    def kind_of(self, citation: str) -> CitationKind:
        try:
            return self.entries[citation]
        except KeyError as exc:
            raise KeyError(
                f"Citation is not present in the configured vocabulary: {citation}"
            ) from exc

    def require_all(self, citations: Iterable[str]) -> tuple[str, ...]:
        values = tuple(sorted(set(citations)))
        missing = tuple(citation for citation in values if citation not in self)
        if missing:
            raise KeyError(f"Unknown citations: {', '.join(missing)}")
        return values


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _read_citations(path: Path) -> Iterator[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Citation vocabulary file does not exist: {path}")
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "citation" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'citation' column.")
        for row in reader:
            citation = (row.get("citation") or "").strip()
            if citation:
                yield citation


@dataclass(frozen=True, slots=True)
class CsvCitationVocabulary(InMemoryCitationVocabulary):
    """Exact vocabulary loaded explicitly from law and court CSV files."""

    @classmethod
    def from_paths(
        cls,
        *,
        laws_path: Path,
        courts_path: Path,
    ) -> CsvCitationVocabulary:
        """Load corpus citation identifiers without retaining source text."""
        entries: dict[str, CitationKind] = {}
        for citation in _read_citations(laws_path):
            entries[citation] = CitationKind.LAW
        for citation in _read_citations(courts_path):
            previous = entries.setdefault(citation, CitationKind.COURT)
            if previous is not CitationKind.COURT:
                raise ValueError(f"Citation appears in both law and court vocabularies: {citation}")
        return cls(entries=entries)
