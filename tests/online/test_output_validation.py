"""Tests for exact final-answer citation post-validation."""

from __future__ import annotations

import unittest

from bettercallagent.citations.output_validation import (
    OutputCitationError,
    validate_answer_citations,
)
from bettercallagent.citations.vocabulary import InMemoryCitationVocabulary
from bettercallagent.schemas import CitationKind


class AnswerCitationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = InMemoryCitationVocabulary(
            entries={
                "Art. 97 Abs. 1 OR": CitationKind.LAW,
                "Art. 102 Abs. 1 OR": CitationKind.LAW,
            }
        )

    def test_accepts_exact_selected_citation(self) -> None:
        result = validate_answer_citations(
            "Ein Anspruch kann aus Art. 97 Abs. 1 OR folgen.",
            accepted_citations=("Art. 97 Abs. 1 OR",),
            vocabulary=self.vocabulary,
        )
        self.assertEqual(result.mentioned, ("Art. 97 Abs. 1 OR",))

    def test_rejects_known_but_unselected_citation(self) -> None:
        with self.assertRaisesRegex(OutputCitationError, "not accepted"):
            validate_answer_citations(
                "Verzug richtet sich nach Art. 102 Abs. 1 OR.",
                accepted_citations=("Art. 97 Abs. 1 OR",),
                vocabulary=self.vocabulary,
            )

    def test_rejects_citation_outside_vocabulary(self) -> None:
        with self.assertRaisesRegex(OutputCitationError, "outside"):
            validate_answer_citations(
                "Unbelegt wäre Art. 41 Abs. 1 OR.",
                accepted_citations=("Art. 97 Abs. 1 OR",),
                vocabulary=self.vocabulary,
            )

    def test_rejects_alias_when_exact_accepted_string_is_not_emitted(self) -> None:
        with self.assertRaisesRegex(OutputCitationError, "exact configured"):
            validate_answer_citations(
                "La responsabilité suit Art. 97 al. 1 CO.",
                accepted_citations=("Art. 97 Abs. 1 OR",),
                vocabulary=self.vocabulary,
            )

    def test_accepts_exact_letter_citation_with_derived_parent_variant(self) -> None:
        vocabulary = InMemoryCitationVocabulary(
            entries={
                "Art. 12 Abs. 1 OR": CitationKind.LAW,
                "Art. 12 Abs. 1 lit. a OR": CitationKind.LAW,
            }
        )
        result = validate_answer_citations(
            "Massgeblich ist Art. 12 Abs. 1 lit. a OR.",
            accepted_citations=(
                "Art. 12 Abs. 1 OR",
                "Art. 12 Abs. 1 lit. a OR",
            ),
            vocabulary=vocabulary,
        )
        self.assertEqual(
            result.mentioned,
            ("Art. 12 Abs. 1 OR", "Art. 12 Abs. 1 lit. a OR"),
        )
        self.assertEqual(
            result.exact_mentioned,
            ("Art. 12 Abs. 1 lit. a OR",),
        )

    def test_rejects_alias_for_each_individual_citation(self) -> None:
        vocabulary = InMemoryCitationVocabulary(
            entries={
                "Art. 41 OR": CitationKind.LAW,
                "Art. 212 StPO": CitationKind.LAW,
            }
        )
        with self.assertRaisesRegex(OutputCitationError, "exact configured"):
            validate_answer_citations(
                "Art. 41 OR und Art. 212 CPP",
                accepted_citations=("Art. 41 OR", "Art. 212 StPO"),
                vocabulary=vocabulary,
            )

    def test_rejects_unrecognized_article_surface(self) -> None:
        vocabulary = InMemoryCitationVocabulary(entries={"Art. 41 OR": CitationKind.LAW})
        with self.assertRaisesRegex(OutputCitationError, "outside"):
            validate_answer_citations(
                "Art. 41 OR gilt; Art. 6 EMRK ist kein freigegebener Beleg.",
                accepted_citations=("Art. 41 OR",),
                vocabulary=vocabulary,
            )
