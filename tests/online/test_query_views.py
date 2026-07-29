"""Regression tests for the configured offline query-view transformation."""

from __future__ import annotations

import unittest

from bettercallagent.retrieval.query_views import (
    build_query_views,
    make_citation_view,
    make_keyword_view,
    make_meta_searchterm,
)

_QUERY = "Software Art. 97 OR. Kopf zwei. MUSS_WEG. Ende vier. Ende fünf. Verjährung Ende sechs?"


class QueryViewTests(unittest.TestCase):
    def test_meta_view_preserves_configured_component_order_and_clauses(
        self,
    ) -> None:
        self.assertEqual(
            make_meta_searchterm(_QUERY),
            (
                "Art. 97 OR; Software, Verjährung; "
                "Software Art. 97 OR.; "
                "Ende vier. Ende fünf. Verjährung Ende sechs?"
            ),
        )
        self.assertNotIn("MUSS_WEG", make_meta_searchterm(_QUERY))

    def test_keyword_view_prepends_citations_and_uses_frequency_ties(self) -> None:
        self.assertEqual(
            make_keyword_view(_QUERY),
            ("Art. 97 OR ende fünf kopf muss sechs software verjährung vier zwei"),
        )

    def test_citation_view_falls_back_to_cleaned_query(self) -> None:
        self.assertEqual(make_citation_view(_QUERY), "Art. 97 OR")
        self.assertEqual(
            make_citation_view("  Keine   Norm genannt. "),
            "  Keine   Norm genannt. ",
        )

    def test_build_query_views_matches_all_configured_defaults(self) -> None:
        views = build_query_views(_QUERY)
        self.assertEqual(views.normal_query, _QUERY)
        self.assertEqual(views.fulltext, _QUERY)
        self.assertEqual(views.meta_searchterm, make_meta_searchterm(_QUERY))
        self.assertEqual(views.keywords, make_keyword_view(_QUERY))
        self.assertEqual(views.citations, make_citation_view(_QUERY))
