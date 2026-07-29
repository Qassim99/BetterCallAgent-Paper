"""Regression tests for the paper's dense-encoder lifecycle."""

from __future__ import annotations

import sys
import unittest
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from bettercallagent.retrieval.query_views import QueryViews
from offline.stages.step_01_retrieve_dense import (
    FIELDS,
    QwenEmbeddingEncoder,
    _encode_query_views,
)


class _RecordingEncoder:
    def __init__(
        self,
        *,
        identifier: int,
        events: list[tuple[str, int, tuple[str, ...] | None]],
        dimensions: int = 3,
        fail: bool = False,
    ) -> None:
        self.identifier = identifier
        self.events = events
        self.dimensions = dimensions
        self.fail = fail

    def encode(self, texts: Sequence[str]) -> Any:
        values = tuple(texts)
        self.events.append(("encode", self.identifier, values))
        if self.fail:
            raise RuntimeError("synthetic encoding failure")
        return values

    def close(self) -> None:
        self.events.append(("close", self.identifier, None))


class DenseEncoderLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.views = [
            QueryViews(
                normal_query="normal one",
                meta_searchterm="meta one",
                keywords="keywords one",
                fulltext="fulltext one",
                citations="citations one",
            ),
            QueryViews(
                normal_query="normal two",
                meta_searchterm="meta two",
                keywords="keywords two",
                fulltext="fulltext two",
                citations="citations two",
            ),
        ]

    def test_uses_one_fresh_encoder_per_field_in_canonical_order(self) -> None:
        events: list[tuple[str, int, tuple[str, ...] | None]] = []
        created: list[_RecordingEncoder] = []

        def factory() -> _RecordingEncoder:
            encoder = _RecordingEncoder(identifier=len(created), events=events)
            created.append(encoder)
            return encoder

        matrices = _encode_query_views(
            views=self.views,
            encoder_factory=factory,
            expected_dimensions=3,
        )

        self.assertEqual(len(created), len(FIELDS))
        self.assertEqual(list(matrices), list(FIELDS))
        for index, field in enumerate(FIELDS):
            expected_texts = tuple(getattr(view, field) for view in self.views)
            self.assertEqual(events[index * 2], ("encode", index, expected_texts))
            self.assertEqual(events[index * 2 + 1], ("close", index, None))
            self.assertEqual(matrices[field], expected_texts)

    def test_closes_encoder_when_encoding_fails(self) -> None:
        events: list[tuple[str, int, tuple[str, ...] | None]] = []

        with self.assertRaisesRegex(RuntimeError, "synthetic encoding failure"):
            _encode_query_views(
                views=self.views,
                encoder_factory=lambda: _RecordingEncoder(
                    identifier=0,
                    events=events,
                    fail=True,
                ),
                expected_dimensions=3,
            )

        self.assertEqual(events[-1], ("close", 0, None))

    def test_closes_encoder_when_dimensions_do_not_match(self) -> None:
        events: list[tuple[str, int, tuple[str, ...] | None]] = []

        with self.assertRaisesRegex(ValueError, "normal_query encoder dimension 2"):
            _encode_query_views(
                views=self.views,
                encoder_factory=lambda: _RecordingEncoder(
                    identifier=0,
                    events=events,
                    dimensions=2,
                ),
                expected_dimensions=3,
            )

        self.assertEqual(events, [("close", 0, None)])

    def test_loads_model_directly_on_configured_device(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.float16 = object()
        fake_torch.float32 = object()

        tokenizer = object()
        tokenizer_loader = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4_096),
            eval=Mock(),
        )
        model_loader = SimpleNamespace(from_pretrained=Mock(return_value=model))
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = model_loader
        fake_transformers.AutoTokenizer = tokenizer_loader

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "transformers": fake_transformers,
            },
        ):
            encoder = QwenEmbeddingEncoder(
                model="Qwen/example",
                revision="immutable-revision",
                device="cuda:0",
                batch_size=16,
                max_length=1_024,
                local_files_only=True,
            )

        model_loader.from_pretrained.assert_called_once_with(
            "Qwen/example",
            revision="immutable-revision",
            dtype=fake_torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        self.assertEqual(encoder.dimensions, 4_096)
        model.eval.assert_called_once_with()
        encoder.close()


if __name__ == "__main__":
    unittest.main()
