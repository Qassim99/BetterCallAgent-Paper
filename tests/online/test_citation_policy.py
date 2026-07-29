"""Golden tests for the canonical fixed citation-vote policy."""

from __future__ import annotations

import unittest

from bettercallagent.citations.policy import FixedVotePolicy, select_citations
from bettercallagent.schemas import RankedCandidate


def candidate(
    doc_ref: str,
    rank: int,
    score: float,
    *citations: str,
) -> RankedCandidate:
    return RankedCandidate(
        doc_ref=doc_ref,
        rank=rank,
        score=score,
        citations=tuple(citations),
    )


class FixedVotePolicyTests(unittest.TestCase):
    """Pin every configured acceptance branch and the no-fallback behavior."""

    def test_dense_anchor_and_bm25_rules_are_combined_deterministically(self) -> None:
        candidates = [
            candidate("d1", 1, 9.7, "Art. 97 Abs. 1 OR", "Art. 102 Abs. 1 OR"),
            candidate("d2", 2, 9.1, "Art. 97 Abs. 1 OR", "Art. 102 Abs. 1 OR"),
            candidate("d3", 3, 8.2, "Art. 97 Abs. 1 OR", "Art. 107 Abs. 2 OR"),
            candidate("d4", 4, 7.8, "Art. 97 Abs. 1 OR"),
            candidate("d5", 5, 4.0, "Art. 41 Abs. 1 OR", "Art. 107 Abs. 2 OR"),
        ]
        selection = select_citations(
            candidates,
            {
                "Art. 102 Abs. 1 OR": 2,
                "Art. 107 Abs. 2 OR": 2,
                "Art. 41 Abs. 1 OR": 1,
                "Art. 97 Abs. 1 OR": 4,
            },
            FixedVotePolicy(),
        )

        self.assertEqual(
            selection.predicted_citations,
            (
                "Art. 102 Abs. 1 OR",
                "Art. 107 Abs. 2 OR",
                "Art. 97 Abs. 1 OR",
            ),
        )
        self.assertEqual(
            tuple(item.citation for item in selection.rejected),
            ("Art. 41 Abs. 1 OR",),
        )
        reasons = {item.citation: item.votes for item in selection.accepted}
        self.assertIn("top-1 score 9.7", reasons["Art. 102 Abs. 1 OR"])
        self.assertEqual(
            reasons["Art. 107 Abs. 2 OR"],
            "BM25 support 2/5",
        )
        self.assertIn("dense vote 4/10", reasons["Art. 97 Abs. 1 OR"])

    def test_policy_returns_empty_selection_when_no_threshold_is_met(self) -> None:
        selection = select_citations(
            [candidate("d1", 1, 2.0, "Art. 41 Abs. 1 OR")],
            {"Art. 41 Abs. 1 OR": 1},
            FixedVotePolicy(),
        )
        self.assertEqual(selection.predicted_citations, ())
        self.assertEqual(len(selection.rejected), 1)

    def test_anchor_position_is_ordered_by_verifier_score_not_retrieval_rank(
        self,
    ) -> None:
        selection = select_citations(
            [
                candidate("retrieval-first", 1, 5.0, "Art. 41 Abs. 1 OR"),
                candidate("verifier-first", 10, 9.5, "Art. 97 Abs. 1 OR"),
            ],
            {},
            FixedVotePolicy(),
        )
        self.assertEqual(
            selection.predicted_citations,
            ("Art. 97 Abs. 1 OR",),
        )
        self.assertEqual(selection.accepted[0].votes, "top-1 score 9.5")
