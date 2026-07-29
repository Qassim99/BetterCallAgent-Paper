"""Stage 1: retrieve candidates from five dense views and fuse them with RRF.

The full mode reads a versioned Qwen embedding index.  The fixture mode consumes
saved per-view rankings and exercises exactly the same fusion and serialization
code without loading a model or a multi-gigabyte index.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from bettercallagent.retrieval.query_views import QueryViews, build_query_views
from bettercallagent.retrieval.rrf import weighted_rrf
from bettercallagent.schemas import RetrievalHit
from offline.io import (
    JsonObject,
    atomic_write_json,
    atomic_write_jsonl,
    load_queries,
    read_jsonl,
    require_file,
)

FIELDS = (
    "normal_query",
    "meta_searchterm",
    "keywords",
    "fulltext",
    "citations",
)
DEFAULT_WEIGHTS = {
    "normal_query": 0.40,
    "meta_searchterm": 1.25,
    "keywords": 0.85,
    "fulltext": 1.35,
    "citations": 0.15,
}


@dataclass(frozen=True, slots=True)
class DenseIndexManifest:
    """Validated metadata required to interpret the raw embedding matrices."""

    rows: int
    dimensions: int
    dtype: str
    fields: tuple[str, ...]
    model: str
    model_revision: str
    model_binding: str
    metadata_file: str
    matrix_files: Mapping[str, str]

    @classmethod
    def load(
        cls,
        index_dir: Path,
        *,
        expected_model: str | None = None,
        expected_revision: str | None = None,
    ) -> DenseIndexManifest:
        """Load and validate metadata needed to interpret the index.

        The original index manifest predates model-provenance fields. Its hashed
        release configuration therefore supplies those two values explicitly.
        Newly generated manifests should store both fields directly.
        """

        path = require_file(index_dir / "manifest.json", description="index manifest")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        fields = tuple(payload.get("fields") or FIELDS)
        if fields != FIELDS:
            raise ValueError(
                f"{path}: expected fields {FIELDS}, received {fields}. "
                "Field order is part of the paper configuration."
            )
        matrices = payload.get("matrix_files") or {field: f"{field}.f16.memmap" for field in fields}
        if not isinstance(matrices, dict) or set(matrices) != set(fields):
            raise ValueError(f"{path}: matrix_files must map every retrieval field")
        declared_model = str(payload.get("model") or "")
        declared_revision = str(payload.get("model_revision") or "")
        if declared_model and expected_model and declared_model != expected_model:
            raise ValueError(f"{path}: embedding model does not match the configuration")
        if declared_revision and expected_revision and declared_revision != expected_revision:
            raise ValueError(f"{path}: embedding model revision does not match the configuration")
        model = declared_model or str(expected_model or "")
        model_revision = declared_revision or str(expected_revision or "")
        manifest = cls(
            rows=int(payload["rows"]),
            dimensions=int(payload.get("dimensions") or payload.get("dim")),
            dtype=str(payload.get("dtype") or "float16"),
            fields=fields,
            model=model,
            model_revision=model_revision,
            model_binding=(
                "index_manifest"
                if declared_model and declared_revision
                else "hashed_release_config"
            ),
            metadata_file=str(payload.get("metadata_file") or "metadata.jsonl"),
            matrix_files={str(key): str(value) for key, value in matrices.items()},
        )
        if manifest.rows <= 0 or manifest.dimensions <= 0:
            raise ValueError(f"{path}: rows and dimensions must be positive")
        if manifest.dtype != "float16":
            raise ValueError(f"{path}: only normalized float16 matrices are supported")
        if not manifest.model or not manifest.model_revision:
            raise ValueError(
                f"{path}: model and immutable model_revision are required either "
                "in the index manifest or as explicit expected values"
            )
        return manifest


class TextEncoder(Protocol):
    """Minimal interface used by the retrieval algorithm."""

    dimensions: int

    def encode(self, texts: Sequence[str]) -> Any:
        """Return an L2-normalized float32 matrix in input order."""

    def close(self) -> None:
        """Release model resources after encoding one retrieval view."""


class QwenEmbeddingEncoder:
    """One isolated Qwen encoder instance for a single retrieval view."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        device: str,
        batch_size: int,
        max_length: int,
        local_files_only: bool,
    ) -> None:
        if not revision or revision == "SET_IMMUTABLE_REVISION":
            raise ValueError("An immutable embedding model revision is required.")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive.")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Full retrieval requires the `offline-gpu` dependency group."
            ) from exc

        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            revision=revision,
            padding_side="left",
            local_files_only=local_files_only,
        )
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._model = AutoModel.from_pretrained(
            model,
            revision=revision,
            dtype=dtype,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        self._model.eval()
        self.dimensions = int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> Any:
        """Encode texts with last-token pooling, as used in the original run."""

        import numpy as np

        torch = self._torch
        batches: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self._batch_size):
                values = [text.strip() or " " for text in texts[start : start + self._batch_size]]
                inputs = self._tokenizer(
                    values,
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                hidden = self._model(**inputs).last_hidden_state
                attention = inputs["attention_mask"]
                if bool((attention[:, -1].sum() == attention.shape[0]).item()):
                    pooled = hidden[:, -1]
                else:
                    positions = attention.sum(dim=1) - 1
                    rows = torch.arange(hidden.shape[0], device=hidden.device)
                    pooled = hidden[rows, positions]
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(normalized.detach().cpu().float().numpy())
        if not batches:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.vstack(batches).astype(np.float32, copy=False)

    def close(self) -> None:
        """Release this view's model before constructing the next encoder."""

        del self._model
        del self._tokenizer
        gc.collect()


def _metadata_reference(metadata: Mapping[str, Any]) -> str:
    source = metadata.get("source_parquet")
    row_group = metadata.get("row_group")
    row = metadata.get("row_index_in_group")
    if source is not None and row_group is not None and row is not None:
        return f"{Path(str(source)).name}:rg{int(row_group)}:row{int(row)}"
    direct = str(metadata.get("doc_id") or "").strip()
    if direct:
        return direct
    raise ValueError("Index metadata lacks both a document ID and a Parquet locator.")


def _read_metadata(
    index_dir: Path,
    manifest: DenseIndexManifest,
    *,
    wanted_indices: set[int],
) -> dict[int, JsonObject]:
    """Stream index metadata and retain only the fused rows."""

    if not wanted_indices:
        raise ValueError("At least one fused metadata row is required.")
    if min(wanted_indices) < 0 or max(wanted_indices) >= manifest.rows:
        raise IndexError("A fused index row falls outside the manifest dimensions.")
    metadata: dict[int, JsonObject] = {}
    path = require_file(index_dir / manifest.metadata_file, description="index metadata")
    for fallback_index, record in enumerate(read_jsonl(path)):
        global_index = int(record.get("global_idx", fallback_index))
        if global_index not in wanted_indices:
            continue
        if global_index in metadata:
            raise ValueError(f"{path}: duplicate wanted global_idx {global_index}")
        metadata[global_index] = record
        if len(metadata) == len(wanted_indices):
            break
    missing = sorted(wanted_indices - set(metadata))
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} fused metadata rows; first={missing[:5]}")
    return metadata


def _search_matrix(
    *,
    matrix_path: Path,
    rows: int,
    dimensions: int,
    query_matrix: Any,
    top_k: int,
    chunk_size: int,
    device: str,
) -> list[list[tuple[float, int]]]:
    """Search a normalized float16 matrix in bounded-memory chunks."""

    import numpy as np

    if top_k <= 0 or chunk_size <= 0:
        raise ValueError("top_k and chunk_size must be positive.")
    if query_matrix.shape[1] != dimensions:
        raise ValueError(
            f"Query dimension {query_matrix.shape[1]} does not match index {dimensions}."
        )
    expected_bytes = rows * dimensions * np.dtype(np.float16).itemsize
    path = require_file(matrix_path, description="embedding matrix")
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, received {path.stat().st_size}")
    matrix = np.memmap(path, dtype=np.float16, mode="r", shape=(rows, dimensions))
    heaps: list[list[tuple[float, int]]] = [[] for _ in range(len(query_matrix))]

    use_cuda = device.startswith("cuda")
    if use_cuda:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA matrix search requires PyTorch.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        query_device = torch.from_numpy(query_matrix).to(device).T.contiguous()

    for start in range(0, rows, chunk_size):
        end = min(rows, start + chunk_size)
        chunk = np.asarray(matrix[start:end], dtype=np.float32)
        if use_cuda:
            scores = torch.from_numpy(chunk).to(device).matmul(query_device).detach().cpu().numpy()
        else:
            scores = chunk @ query_matrix.T
        take = min(top_k, end - start)
        for query_index in range(scores.shape[1]):
            column = scores[:, query_index]
            local_indices = np.argpartition(column, -take)[-take:]
            heap = heaps[query_index]
            for local_index in local_indices:
                item = (float(column[local_index]), start + int(local_index))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    return [sorted(heap, reverse=True) for heap in heaps]


def fuse_rankings(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    rrf_k: int = 60,
    limit: int = 1_000,
) -> list[RetrievalHit]:
    """Fuse one query's five per-view rankings with the canonical implementation."""

    if set(rankings) != set(FIELDS):
        raise ValueError(f"Rankings must contain exactly the fields {FIELDS}.")
    ordered = {field: rankings[field] for field in FIELDS}
    return weighted_rrf(ordered, weights, k=rrf_k, limit=limit)


def _serialize_query(
    *,
    query_id: str,
    query: str,
    views: QueryViews,
    rankings: Mapping[str, Sequence[RetrievalHit]],
    fused: Sequence[RetrievalHit],
    metadata: Mapping[str, Mapping[str, Any]],
    document_ids: Mapping[str, str] | None = None,
) -> tuple[list[JsonObject], JsonObject]:
    details: dict[tuple[str, str], tuple[int, float]] = {}
    for field, hits in rankings.items():
        for hit in hits:
            details[(field, hit.doc_ref)] = (hit.rank, hit.score)

    candidates: list[JsonObject] = []
    summary_top: list[JsonObject] = []
    for hit in fused:
        item_metadata = dict(metadata.get(hit.doc_ref) or {})
        document_id = (
            document_ids.get(hit.doc_ref, hit.doc_ref) if document_ids is not None else hit.doc_ref
        )
        hit_details = [
            {
                "field": field,
                "rank": details[(field, hit.doc_ref)][0],
                "score": details[(field, hit.doc_ref)][1],
            }
            for field in FIELDS
            if (field, hit.doc_ref) in details
        ]
        candidate = {
            "query_id": query_id,
            "rank": hit.rank,
            "doc_id": document_id,
            "global_idx": item_metadata.get("global_idx"),
            "fusion_score": hit.score,
            "metadata": item_metadata,
            "hits": hit_details,
        }
        candidates.append(candidate)
        if len(summary_top) < 100:
            summary_top.append(candidate)
    summary = {
        "query_id": query_id,
        "query": query,
        "meta_query": views.meta_searchterm,
        "keywords_query": views.keywords,
        "query_views": views.as_mapping(),
        "top": summary_top,
    }
    return candidates, summary


def _encode_query_views(
    *,
    views: Sequence[QueryViews],
    encoder_factory: Callable[[], TextEncoder],
    expected_dimensions: int,
) -> dict[str, Any]:
    """Encode each retrieval view with a fresh, promptly released model.

    Model lifecycle is part of the historical experiment configuration. Sharing
    one Qwen instance across the five left-padded batches changes the saved
    ``meta_searchterm`` ranking.
    """

    matrices: dict[str, Any] = {}
    for field in FIELDS:
        encoder = encoder_factory()
        try:
            if encoder.dimensions != expected_dimensions:
                raise ValueError(
                    f"{field} encoder dimension {encoder.dimensions} "
                    f"!= index dimension {expected_dimensions}."
                )
            matrices[field] = encoder.encode([getattr(view, field) for view in views])
        finally:
            encoder.close()
    return matrices


def retrieve_from_index(
    *,
    queries_path: Path,
    index_dir: Path,
    encoder_factory: Callable[[], TextEncoder],
    model: str,
    model_revision: str,
    top_k: int,
    fusion_top_k: int,
    rrf_k: int,
    chunk_size: int,
    device: str,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Run full five-view retrieval against a validated local index."""

    manifest = DenseIndexManifest.load(
        index_dir,
        expected_model=model,
        expected_revision=model_revision,
    )
    query_rows = load_queries(queries_path)
    query_ids = list(query_rows)
    views = [build_query_views(str(query_rows[query_id]["query"])) for query_id in query_ids]
    query_matrices = _encode_query_views(
        views=views,
        encoder_factory=encoder_factory,
        expected_dimensions=manifest.dimensions,
    )

    per_field: dict[str, list[list[tuple[float, int]]]] = {}
    for field in FIELDS:
        per_field[field] = _search_matrix(
            matrix_path=index_dir / manifest.matrix_files[field],
            rows=manifest.rows,
            dimensions=manifest.dimensions,
            query_matrix=query_matrices[field],
            top_k=top_k,
            chunk_size=chunk_size,
            device=device,
        )

    indexed_states: list[tuple[dict[str, list[RetrievalHit]], list[RetrievalHit]]] = []
    wanted_indices: set[int] = set()
    for query_index, _query_id in enumerate(query_ids):
        rankings: dict[str, list[RetrievalHit]] = {}
        for field in FIELDS:
            hits: list[RetrievalHit] = []
            for rank, (score, global_index) in enumerate(per_field[field][query_index], start=1):
                hits.append(
                    RetrievalHit(
                        doc_ref=str(global_index),
                        score=score,
                        score_kind="cosine",
                        rank=rank,
                        sources=(field,),
                    )
                )
            rankings[field] = hits
        fused = fuse_rankings(
            rankings,
            weights=weights,
            rrf_k=rrf_k,
            limit=fusion_top_k,
        )
        indexed_states.append((rankings, fused))
        wanted_indices.update(int(hit.doc_ref) for hit in fused)

    metadata_by_index = _read_metadata(
        index_dir,
        manifest,
        wanted_indices=wanted_indices,
    )
    reference_by_index: dict[str, str] = {}
    metadata_by_index_identity: dict[str, JsonObject] = {}
    for global_index, item_metadata in metadata_by_index.items():
        reference = _metadata_reference(item_metadata)
        enriched = dict(item_metadata)
        enriched["global_idx"] = global_index
        identity = str(global_index)
        metadata_by_index_identity[identity] = enriched
        reference_by_index[identity] = reference

    output: list[JsonObject] = []
    summaries: list[JsonObject] = []
    for query_index, query_id in enumerate(query_ids):
        indexed_rankings, indexed_fused = indexed_states[query_index]
        candidates, summary = _serialize_query(
            query_id=query_id,
            query=str(query_rows[query_id]["query"]),
            views=views[query_index],
            rankings=indexed_rankings,
            fused=indexed_fused,
            metadata=metadata_by_index_identity,
            document_ids=reference_by_index,
        )
        output.extend(candidates)
        summaries.append(summary)
    return output, summaries


def retrieve_from_saved_rankings(
    *,
    queries_path: Path,
    rankings_path: Path,
    fusion_top_k: int,
    rrf_k: int,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Fuse explicit fixture/replay rankings without model or index access."""

    queries = load_queries(queries_path)
    grouped: dict[str, dict[str, list[RetrievalHit]]] = {
        query_id: {field: [] for field in FIELDS} for query_id in queries
    }
    metadata: dict[str, JsonObject] = {}
    seen: set[tuple[str, str, int]] = set()
    for row in read_jsonl(rankings_path):
        query_id = str(row.get("query_id") or "")
        field = str(row.get("field") or "")
        if query_id not in grouped or field not in FIELDS:
            raise ValueError(f"Unknown query or retrieval field: {query_id!r}/{field!r}")
        rank = int(row.get("rank") or 0)
        key = (query_id, field, rank)
        if rank <= 0 or key in seen:
            raise ValueError(f"Duplicate or invalid saved rank: {key}")
        seen.add(key)
        doc_ref = str(row.get("doc_id") or "").strip()
        item_metadata = row.get("metadata") or {}
        if not isinstance(item_metadata, dict):
            raise ValueError(f"{rankings_path}: metadata must be an object")
        previous = metadata.setdefault(doc_ref, dict(item_metadata))
        if previous != item_metadata:
            raise ValueError(f"Conflicting saved metadata for {doc_ref!r}")
        grouped[query_id][field].append(
            RetrievalHit(
                doc_ref=doc_ref,
                score=float(row["score"]),
                score_kind="saved_cosine",
                rank=rank,
                sources=(field,),
            )
        )

    output: list[JsonObject] = []
    summaries: list[JsonObject] = []
    for query_id, query_row in queries.items():
        rankings = grouped[query_id]
        for field, hits in rankings.items():
            hits.sort(key=lambda hit: hit.rank)
            if not hits:
                raise ValueError(f"No saved {field} rankings for query {query_id!r}")
            if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)):
                raise ValueError(f"Saved {field} ranks are not contiguous for {query_id!r}")
        query = str(query_row["query"])
        query_views = build_query_views(query)
        fused = fuse_rankings(
            rankings,
            weights=weights,
            rrf_k=rrf_k,
            limit=fusion_top_k,
        )
        candidates, summary = _serialize_query(
            query_id=query_id,
            query=query,
            views=query_views,
            rankings=rankings,
            fused=fused,
            metadata=metadata,
        )
        output.extend(candidates)
        summaries.append(summary)
    return output, summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index-dir", type=Path)
    source.add_argument(
        "--saved-rankings",
        type=Path,
        help="Explicit no-model fixture/replay input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--top-k", type=int, default=1_000)
    parser.add_argument("--fusion-top-k", type=int, default=1_000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=30_000)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Permit Hugging Face downloads; default is local-files-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.saved_rankings is not None:
        candidates, summary = retrieve_from_saved_rankings(
            queries_path=args.queries,
            rankings_path=args.saved_rankings,
            fusion_top_k=args.fusion_top_k,
            rrf_k=args.rrf_k,
        )
    else:
        index_dir = args.index_dir.expanduser().resolve()
        manifest = DenseIndexManifest.load(
            index_dir,
            expected_model=args.model,
            expected_revision=args.model_revision,
        )
        if args.model != manifest.model:
            raise ValueError(
                f"Configured model {args.model!r} does not match index {manifest.model!r}."
            )
        revision = args.model_revision or manifest.model_revision
        if revision != manifest.model_revision:
            raise ValueError("Embedding revision does not match the index manifest.")
        encoder_factory = partial(
            QwenEmbeddingEncoder,
            model=args.model,
            revision=revision,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=not args.allow_model_download,
        )
        candidates, summary = retrieve_from_index(
            queries_path=args.queries,
            index_dir=index_dir,
            encoder_factory=encoder_factory,
            model=args.model,
            model_revision=revision,
            top_k=args.top_k,
            fusion_top_k=args.fusion_top_k,
            rrf_k=args.rrf_k,
            chunk_size=args.chunk_size,
            device=args.device,
        )
    atomic_write_jsonl(args.output, candidates)
    atomic_write_json(args.summary, summary)
    print(f"Retrieved {len(candidates)} fused candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
