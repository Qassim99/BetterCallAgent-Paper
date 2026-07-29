"""Tests for shared online/offline evaluation and retrieval utilities."""

from __future__ import annotations

import unittest

from bettercallagent.evaluation.metrics import macro_f1
from bettercallagent.retrieval.rrf import weighted_rrf
from bettercallagent.schemas import RetrievalHit


def hit(doc_ref: str, rank: int, source: str) -> RetrievalHit:
    return RetrievalHit(
        doc_ref=doc_ref,
        score=1.0 / rank,
        score_kind=source,
        rank=rank,
        sources=(source,),
    )


class SharedUtilityTests(unittest.TestCase):
    def test_weighted_rrf_preserves_first_encounter_for_exact_ties(self) -> None:
        fused = weighted_rrf(
            {
                "dense": [hit("doc-b", 1, "dense"), hit("doc-a", 2, "dense")],
                "bm25": [hit("doc-a", 1, "bm25"), hit("doc-b", 2, "bm25")],
            },
            {"dense": 1.0, "bm25": 1.0},
            k=60,
        )
        self.assertEqual([item.doc_ref for item in fused], ["doc-b", "doc-a"])
        self.assertEqual(fused[0].sources, ("bm25", "dense"))

    def test_macro_f1_averages_query_scores_not_global_counts(self) -> None:
        metrics = macro_f1(
            {
                "q1": ("A",),
                "q2": ("X", "Y", "Z"),
            },
            {
                "q1": ("A",),
                "q2": ("B",),
            },
        )
        self.assertEqual(metrics.macro_f1, 0.5)
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 3)
        self.assertEqual(metrics.false_negatives, 1)
