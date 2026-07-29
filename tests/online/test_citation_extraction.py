"""Golden tests for the configured citation extraction and normalization."""

from __future__ import annotations

import unittest

from bettercallagent.citations.extract import (
    extract_citations,
    extract_citations_by_kind,
)


class CitationExtractionTests(unittest.TestCase):
    def test_normalizes_french_code_and_paragraph_aliases(self) -> None:
        self.assertEqual(
            extract_citations("Selon Art. 97 al. 1 CO, Art. 221 al. 1 CPP et Art. 42 al. 2 LTF."),
            (
                "Art. 221 Abs. 1 StPO",
                "Art. 42 Abs. 2 BGG",
                "Art. 97 Abs. 1 OR",
            ),
        )

    def test_extracts_bare_article_lists_with_the_trailing_code(self) -> None:
        self.assertEqual(
            extract_citations("Nach Art. 12, 13 und 14 OR gelten die Regeln."),
            ("Art. 12 OR", "Art. 13 OR", "Art. 14 OR"),
        )

    def test_preserves_configured_paragraph_range_expansion(self) -> None:
        self.assertEqual(
            extract_citations("Gemäss Art. 64 Abs. 1 und 2 StGB gilt dies."),
            (
                "Art. 2 StGB",
                "Art. 64 Abs. 1 StGB",
                "Art. 64 Abs. 2 StGB",
            ),
        )

    def test_adds_configured_parent_variant_for_letter_citations(self) -> None:
        self.assertEqual(
            extract_citations("Art. 12 Abs. 1 lit. a OR ist anwendbar."),
            ("Art. 12 Abs. 1 OR", "Art. 12 Abs. 1 lit. a OR"),
        )

    def test_normalizes_atf_and_expands_court_consideration_lists(self) -> None:
        matches = extract_citations_by_kind(
            "ATF 145 III 42 consid. 2.1 et 2.3; "
            "BGE 140 II 10 E. 4.2 und 4.3; "
            "Urteil 1B_42/2025, E. 3.1 und 3.2."
        )
        self.assertEqual(matches.laws, ())
        self.assertEqual(
            matches.courts,
            (
                "1B_42/2025",
                "1B_42/2025 E. 3.1",
                "1B_42/2025 E. 3.2.",
                "BGE 140 II 10",
                "BGE 140 II 10 E. 4.2",
                "BGE 140 II 10 E. 4.3",
                "BGE 145 III 42 E. 2.1",
                "BGE 145 III 42 E. 2.3",
            ),
        )

    def test_stgb_number_one_variant_matches_configured_gate(self) -> None:
        self.assertEqual(
            extract_citations("Art. 123 Ziff. 1 StGB."),
            ("Art. 123 Abs. 1 StGB", "Art. 123 Ziff. 1 StGB"),
        )
