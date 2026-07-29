"""Canonical citation extraction used by the configured evidence gate."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_MAP = {
    "CPP": "StPO",
    "CPC": "ZPO",
    "CP": "StGB",
    "CC": "ZGB",
    "CO": "OR",
    "LCR": "SVG",
    "LTF": "BGG",
    "LOAP": "StBOG",
    "LPGA": "ATSG",
    "LAI": "IVG",
    "LDIP": "IPRG",
    "Cst.": "BV",
    "Cst": "BV",
}
_LAW_PATTERN = re.compile(
    r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)"
    r"((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)"
    r"\s*[0-9a-z]+)*)"
    r"\s*(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV"
    r"|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r"\b(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV"
    r"|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b",
    re.IGNORECASE,
)
_BARE_ARTICLE_PATTERN = re.compile(
    r"(?:\bArt\.\s*|[,;]|\bet\b|\bund\b|\band\b)"
    r"\s*([0-9]+[a-z]?(?:/[0-9]+)?)"
    r"((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)"
    r"\s*[0-9a-z]+)*)",
    re.IGNORECASE,
)
_PARAGRAPH_RANGE_PATTERN = re.compile(
    r"(Abs\.|al\.|alinéa)\s*([0-9]+)\s*(?:et|und|and)\s*([0-9]+)",
    re.IGNORECASE,
)
_ARTICLE_PARAGRAPH_RANGE_PATTERN = re.compile(
    r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)\s*"
    r"(Abs\.|al\.|alinéa)\s*([0-9]+)\s*(?:et|und|and)\s*([0-9]+)"
    r"(?!\s*(?:Abs\.|al\.|alinéa))",
    re.IGNORECASE,
)
_COURT_PATTERNS = (
    re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+\s+E\.\s*[\dA-Za-z_.-]+"),
    re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+"),
    re.compile(r"\b\d+[A-Z]_\d+/\d{4}\s+E\.\s*[\dA-Za-z_.-]+"),
    re.compile(r"\b\d+[A-Z]_\d+/\d{4}"),
)
_CONSIDERATION_LIST = (
    r"[\dA-Za-z_.-]+"
    r"(?:\s*(?:,|et|und|and)\s*[\dA-Za-z_.-]+)*"
)
_ATF_CONSIDERATION_PATTERN = re.compile(
    r"\bATF\s+(\d+\s+[IVX]+\s+\d+)\s+(?:consid\.|c\.)\s*(" + _CONSIDERATION_LIST + ")",
    re.IGNORECASE,
)
_BGE_CONSIDERATION_PATTERN = re.compile(
    r"\bBGE\s+(\d+\s+[IVX]+\s+\d+)\s+(?:consid\.|c\.|E\.)\s*(" + _CONSIDERATION_LIST + ")",
    re.IGNORECASE,
)
_CASE_CONSIDERATION_PATTERN = re.compile(
    r"\b([1-9][A-Z]_\d+/\d{4})\b"
    r"(?:.{0,120}?)(?:consid\.|c\.|E\.)\s*(" + _CONSIDERATION_LIST + ")",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_law_parts(number: str, tail: str, code: str) -> str:
    normalized_code = _CODE_MAP.get(code.strip(), code.strip())
    normalized_tail = tail or ""
    normalized_tail = re.sub(
        r"\bal\.\s*",
        "Abs. ",
        normalized_tail,
        flags=re.IGNORECASE,
    )
    normalized_tail = re.sub(
        r"\balinéa\s*",
        "Abs. ",
        normalized_tail,
        flags=re.IGNORECASE,
    )
    normalized_tail = re.sub(
        r"\blet\.\s*",
        "lit. ",
        normalized_tail,
        flags=re.IGNORECASE,
    )
    normalized_tail = re.sub(
        r"\bch\.\s*",
        "Ziff. ",
        normalized_tail,
        flags=re.IGNORECASE,
    )
    normalized_tail = re.sub(
        r"\bcifra\s*",
        "Ziff. ",
        normalized_tail,
        flags=re.IGNORECASE,
    )
    normalized_tail = re.sub(r"\s+", " ", normalized_tail).strip(" ,;.")
    citation = f"Art. {number.strip()} {normalized_tail} {normalized_code}"
    return citation.replace("  ", " ").strip()


def _law_variants(citation: str) -> set[str]:
    variants = {citation}
    match = re.match(
        r"^(Art\.\s+\S+(?:\s+Abs\.\s+\S+)?)"
        r"\s+lit\.\s+\S+\s+(\S+)$",
        citation,
    )
    if match and "Abs." in match.group(1):
        variants.add(f"{match.group(1)} {match.group(2)}")
    return variants


def _split_considerations(raw: str) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in re.split(
            r"\s*(?:,|et|und|and)\s*",
            raw or "",
        )
        if value.strip()
    )


def _extract_laws(text: str) -> set[str]:
    laws: set[str] = set()
    for match in _LAW_PATTERN.finditer(text):
        normalized = _normalize_law_parts(
            match.group(1),
            match.group(2),
            match.group(3),
        )
        laws.update(_law_variants(normalized))
        normalized_code = _CODE_MAP.get(
            match.group(3).strip(),
            match.group(3).strip(),
        )
        if normalized_code == "StGB" and re.search(
            r"\b(ch\.|Ziff\.)\s*1\b",
            match.group(2),
            re.IGNORECASE,
        ):
            laws.add(f"Art. {match.group(1).strip()} Abs. 1 StGB")
        for range_match in _PARAGRAPH_RANGE_PATTERN.finditer(match.group(2) or ""):
            second_tail = re.sub(
                _PARAGRAPH_RANGE_PATTERN,
                f"{range_match.group(1)} {range_match.group(3)}",
                match.group(2),
                count=1,
            )
            laws.update(
                _law_variants(
                    _normalize_law_parts(
                        match.group(1),
                        second_tail,
                        match.group(3),
                    )
                )
            )

    for code_match in _CODE_PATTERN.finditer(text):
        code = code_match.group(1)
        segment_start = max(
            text.rfind(";", 0, code_match.start()),
            text.rfind("(", 0, code_match.start()),
        )
        segment = text[segment_start + 1 : code_match.start()]
        inner_codes = list(_CODE_PATTERN.finditer(segment))
        if inner_codes:
            segment = segment[inner_codes[-1].end() :]
        if not re.search(r"\bart\.", segment, re.IGNORECASE) or len(segment) > 600:
            continue
        for range_match in _ARTICLE_PARAGRAPH_RANGE_PATTERN.finditer(segment):
            laws.update(
                _law_variants(
                    _normalize_law_parts(
                        range_match.group(1),
                        f"{range_match.group(2)} {range_match.group(3)}",
                        code,
                    )
                )
            )
            laws.update(
                _law_variants(
                    _normalize_law_parts(
                        range_match.group(1),
                        f"{range_match.group(2)} {range_match.group(4)}",
                        code,
                    )
                )
            )
        for article_match in _BARE_ARTICLE_PATTERN.finditer(segment):
            normalized = _normalize_law_parts(
                article_match.group(1),
                article_match.group(2),
                code,
            )
            laws.update(_law_variants(normalized))
            for range_match in _PARAGRAPH_RANGE_PATTERN.finditer(article_match.group(2) or ""):
                second_tail = re.sub(
                    _PARAGRAPH_RANGE_PATTERN,
                    f"{range_match.group(1)} {range_match.group(3)}",
                    article_match.group(2),
                    count=1,
                )
                laws.update(
                    _law_variants(
                        _normalize_law_parts(
                            article_match.group(1),
                            second_tail,
                            code,
                        )
                    )
                )
    return laws


def _extract_courts(text: str) -> set[str]:
    courts: set[str] = set()
    for pattern in _COURT_PATTERNS:
        for match in pattern.finditer(text):
            courts.add(re.sub(r"\s+", " ", match.group(0)).strip(" .;,:"))
    for match in _ATF_CONSIDERATION_PATTERN.finditer(text):
        reference = re.sub(r"\s+", " ", match.group(1)).strip()
        for consideration in _split_considerations(match.group(2)):
            courts.add(f"BGE {reference} E. {consideration}")
    for match in _BGE_CONSIDERATION_PATTERN.finditer(text):
        reference = re.sub(r"\s+", " ", match.group(1)).strip()
        for consideration in _split_considerations(match.group(2)):
            courts.add(f"BGE {reference} E. {consideration}")
    for match in _CASE_CONSIDERATION_PATTERN.finditer(text):
        for consideration in _split_considerations(match.group(2)):
            courts.add(f"{match.group(1).strip()} E. {consideration}")
    return courts


@dataclass(frozen=True, slots=True)
class CitationMatches:
    """Canonical law and court citations extracted from one text."""

    laws: tuple[str, ...]
    courts: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        """Return both families in stable lexical order."""
        return tuple(sorted(set(self.laws) | set(self.courts)))


class CitationExtractor:
    """Reproduce the configured law alias, range, and court extraction."""

    def extract_by_kind(self, text: str) -> CitationMatches:
        """Return law and court citations as separate stable tuples."""
        normalized_text = text or ""
        return CitationMatches(
            laws=tuple(sorted(_extract_laws(normalized_text))),
            courts=tuple(sorted(_extract_courts(normalized_text))),
        )

    def extract(self, text: str) -> tuple[str, ...]:
        """Return all unique canonical citations in stable lexical order."""
        return self.extract_by_kind(text).all


_DEFAULT_EXTRACTOR = CitationExtractor()


def extract_citations_by_kind(text: str) -> CitationMatches:
    """Convenience API for canonical citation extraction by family."""
    return _DEFAULT_EXTRACTOR.extract_by_kind(text)


def extract_citations(text: str) -> tuple[str, ...]:
    """Convenience API for canonical Swiss citation extraction."""
    return _DEFAULT_EXTRACTOR.extract(text)
