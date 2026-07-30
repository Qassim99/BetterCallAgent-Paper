"""End-to-end tests for Stage 0 index construction and Stage 1 consumption."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from offline.indexing.build_dense_index import (
    FIELDS,
    STATE_FILE,
    EncodingConfig,
    ViewRecord,
    build_dense_index,
    create_build_plan,
)
from offline.stages.step_01_retrieve_dense import (
    DenseIndexManifest,
    retrieve_from_index,
)


class _Rows:
    def __init__(self) -> None:
        self.values: dict[tuple[str, int, int], Mapping[str, Any]] = {
            ("documents.parquet", 0, 0): {
                "regeste": "alpha",
                "full_text": "alpha Art. 1 ZGB",
            },
            ("documents.parquet", 0, 1): {
                "regeste": "beta",
                "full_text": "beta Art. 2 OR",
            },
        }

    def load(self, record: ViewRecord) -> Mapping[str, Any]:
        return self.values[record.locator]


class _DocumentEncoder:
    dimensions = 3

    def __init__(self, *, fail_after_calls: int | None = None) -> None:
        self.fail_after_calls = fail_after_calls
        self.calls = 0
        self.closed = False

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Any:
        assert batch_size > 0
        assert max_length > 0
        self.calls += 1
        if self.fail_after_calls is not None and self.calls > self.fail_after_calls:
            raise RuntimeError("synthetic interruption")
        rows = []
        for text in texts:
            if "beta" in text.lower() or "Art. 2 OR" in text:
                rows.append((0.0, 1.0, 0.0))
            else:
                rows.append((1.0, 0.0, 0.0))
        return np.asarray(rows, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class _QueryEncoder:
    dimensions = 3

    def __init__(self) -> None:
        self.closed = False

    def encode(self, texts: Sequence[str]) -> Any:
        return np.repeat(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            len(texts),
            axis=0,
        )

    def close(self) -> None:
        self.closed = True


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    views = tmp_path / "views"
    corpus.mkdir()
    views.mkdir()
    parquet.write_table(pa.table({"row": [0, 1]}), corpus / "documents.parquet")
    records = [
        {
            "doc_id": "alpha-source",
            "source_parquet": "/old/machine/documents.parquet",
            "row_group": 0,
            "row_index_in_group": 0,
            "content": {
                "normal_query": "alpha",
                "meta_searchterm_de": "alpha",
                "keywords_de": ["alpha"],
            },
        },
        {
            "doc_id": "beta-source",
            "source_parquet": "/old/machine/documents.parquet",
            "row_group": 0,
            "row_index_in_group": "1",
            "content": {
                "normal_query": "beta",
                "meta_searchterm_de": "beta",
                "keywords_de": ["beta"],
            },
        },
    ]
    (views / "shard_000000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return corpus, views


def _config() -> EncodingConfig:
    return EncodingConfig(
        model="Qwen/test",
        model_revision="immutable-test-revision",
        device="cpu",
        batch_size_short=2,
        batch_size_fulltext=1,
        max_length_short=64,
        max_length_fulltext=128,
        checkpoint_rows=1,
        local_files_only=True,
    )


def test_builds_reader_compatible_index_and_stage_1_retrieves_it(tmp_path: Path) -> None:
    corpus, views = _inputs(tmp_path)
    plan = create_build_plan(corpus_dir=corpus, view_dir=views)
    output = tmp_path / "index"
    encoder = _DocumentEncoder()

    manifest = build_dense_index(
        plan=plan,
        output_dir=output,
        config=_config(),
        encoder=encoder,
        row_loader=_Rows(),
        expected_rows=2,
    )

    assert encoder.closed
    assert manifest["fields"] == list(FIELDS)
    assert manifest["model_revision"] == "immutable-test-revision"
    assert len(list(output.iterdir())) == 8
    loaded = DenseIndexManifest.load(
        output,
        expected_model="Qwen/test",
        expected_revision="immutable-test-revision",
    )
    assert (loaded.rows, loaded.dimensions, loaded.dtype) == (2, 3, "float16")
    for field in FIELDS:
        matrix = np.memmap(
            output / loaded.matrix_files[field],
            dtype=np.float16,
            mode="r",
            shape=(2, 3),
        )
        np.testing.assert_array_equal(matrix[0], np.asarray([1, 0, 0]))
        np.testing.assert_array_equal(matrix[1], np.asarray([0, 1, 0]))

    queries = tmp_path / "queries.csv"
    queries.write_text("query_id,query\nq1,alpha legal question\n", encoding="utf-8")
    query_encoder = _QueryEncoder()
    candidates, _summaries = retrieve_from_index(
        queries_path=queries,
        index_dir=output,
        encoder=query_encoder,
        model="Qwen/test",
        model_revision="immutable-test-revision",
        top_k=2,
        fusion_top_k=2,
        rrf_k=60,
        chunk_size=2,
        device="cpu",
    )
    assert query_encoder.closed
    assert candidates[0]["doc_id"] == "documents.parquet:rg0:row0"


def test_resumes_only_from_the_last_complete_checkpoint(tmp_path: Path) -> None:
    corpus, views = _inputs(tmp_path)
    plan = create_build_plan(corpus_dir=corpus, view_dir=views)
    output = tmp_path / "resumed-index"

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_dense_index(
            plan=plan,
            output_dir=output,
            config=_config(),
            encoder=_DocumentEncoder(fail_after_calls=5),
            row_loader=_Rows(),
        )

    state = json.loads((output / STATE_FILE).read_text(encoding="utf-8"))
    assert state["rows_complete"] == 1
    build_dense_index(
        plan=plan,
        output_dir=output,
        config=_config(),
        encoder=_DocumentEncoder(),
        row_loader=_Rows(),
        resume=True,
    )
    assert not (output / STATE_FILE).exists()
    assert (output / "manifest.json").is_file()


def test_resumes_publication_after_partial_file_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus, views = _inputs(tmp_path)
    plan = create_build_plan(corpus_dir=corpus, view_dir=views)
    output = tmp_path / "publication-resume"

    original_replace = Path.replace

    def interrupting_replace(path: Path, target: Path) -> Path:
        if path.name == "meta_searchterm.f16.memmap.partial":
            raise RuntimeError("synthetic publication interruption")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupting_replace)
        with pytest.raises(RuntimeError, match="synthetic publication interruption"):
            build_dense_index(
                plan=plan,
                output_dir=output,
                config=_config(),
                encoder=_DocumentEncoder(),
                row_loader=_Rows(),
            )

    assert (output / STATE_FILE).is_file()
    assert (output / "normal_query.f16.memmap").is_file()
    assert (output / "meta_searchterm.f16.memmap.partial").is_file()
    build_dense_index(
        plan=plan,
        output_dir=output,
        config=_config(),
        encoder=_DocumentEncoder(),
        row_loader=_Rows(),
        resume=True,
    )
    assert not (output / STATE_FILE).exists()
    assert (output / "manifest.json").is_file()


def test_plan_rejects_duplicate_source_rows(tmp_path: Path) -> None:
    corpus, views = _inputs(tmp_path)
    path = views / "shard_000000.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(f"{first}\n{first}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate source row"):
        create_build_plan(corpus_dir=corpus, view_dir=views)


def test_rejects_inputs_changed_after_planning(tmp_path: Path) -> None:
    corpus, views = _inputs(tmp_path)
    plan = create_build_plan(corpus_dir=corpus, view_dir=views)
    with (corpus / "documents.parquet").open("ab") as handle:
        handle.write(b"changed")

    output = tmp_path / "changed-input-index"
    encoder = _DocumentEncoder()
    with pytest.raises(RuntimeError, match="inputs changed"):
        build_dense_index(
            plan=plan,
            output_dir=output,
            config=_config(),
            encoder=encoder,
            row_loader=_Rows(),
        )

    assert encoder.closed
    assert not (output / "manifest.json").exists()


def test_plan_rejects_out_of_range_parquet_locator(tmp_path: Path) -> None:
    corpus, views = _inputs(tmp_path)
    path = views / "shard_000000.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[-1]["row_index_in_group"] = 2
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(IndexError, match="document views reference row 2"):
        create_build_plan(corpus_dir=corpus, view_dir=views)


@pytest.mark.parametrize("value", [True, 1.5])
def test_plan_rejects_non_integer_locators(tmp_path: Path, value: object) -> None:
    corpus, views = _inputs(tmp_path)
    path = views / "shard_000000.jsonl"
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    first["row_group"] = value
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be an integer"):
        create_build_plan(corpus_dir=corpus, view_dir=views)
