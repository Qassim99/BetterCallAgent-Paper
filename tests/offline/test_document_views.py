"""Tests for the versioned document-view text contract."""

from __future__ import annotations

from offline.indexing.document_views import FIELDS, build_texts, parse_generated


def test_builds_all_views_in_canonical_order() -> None:
    generated = {
        "content": {
            "normal_query": "  legal   question ",
            "meta_searchterm_de": " abstrakter  Kern ",
            "keywords_de": ["Verjährung", " Vertrag "],
            "keywords_en": ["limitation period"],
            "citations": ["Art. 1 ZGB"],
        }
    }
    source = {
        "regeste": "Leitsatz",
        "abstract_de": "Zusammenfassung",
        "full_text": "Siehe Art. 1 ZGB und BGE 123 IV 4.",
        "cited_decisions": ["BGE 123 IV 4"],
    }

    texts = build_texts(generated, source)

    assert tuple(texts) == FIELDS
    assert texts["normal_query"] == "legal question"
    assert texts["meta_searchterm"] == "abstrakter Kern"
    assert texts["keywords"] == "Verjährung ; Vertrag ; limitation period"
    assert texts["fulltext"] == (
        "regeste: Leitsatz\n"
        "abstract_de: Zusammenfassung\n"
        "full_text: Siehe Art. 1 ZGB und BGE 123 IV 4."
    )
    assert texts["citations"] == "Art. 1 ZGB ; BGE 123 IV 4"


def test_recovers_fields_from_truncated_generated_json() -> None:
    truncated = (
        '{"normal_query":"Frage","meta_searchterm_de":"Kern",'
        '"keywords_de":["eins","zwei"],"keywords_en":["one"'
    )

    recovered = parse_generated(truncated)

    assert recovered["normal_query"] == "Frage"
    assert recovered["meta_searchterm_de"] == "Kern"
    assert recovered["keywords_de"] == ["eins", "zwei"]
    assert recovered["keywords_en"] == ["one"]


def test_blank_generated_fields_remain_explicitly_blank() -> None:
    texts = build_texts({"content": ""}, {"full_text": "No citation"})

    assert texts == {
        "normal_query": "",
        "meta_searchterm": "",
        "keywords": "",
        "fulltext": "full_text: No citation",
        "citations": "",
    }
