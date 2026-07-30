"""Stage 0: build the five-view dense index consumed by Stage 1.

The builder intentionally separates document-view generation from embedding.  Its
inputs are the versioned case-law Parquet files and JSONL document views.  It writes
the five normalized float16 matrices, row metadata, row offsets, and a manifest only
after every row has been committed successfully.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from offline.indexing.document_views import FIELDS, build_texts, clean
from offline.io import sha256_file

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
INDEX_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
MATRIX_DTYPE = "float16"
METADATA_FILE = "metadata.jsonl"
ROW_OFFSETS_FILE = "row_offsets.json"
STATE_FILE = ".build-state.json"


@dataclass(frozen=True, slots=True)
class ViewRecord:
    """One generated document view and its stable source-row locator."""

    payload: Mapping[str, Any]
    source_file: str
    row_group: int
    row_index: int
    doc_id: str | None
    input_file: str
    input_line: int

    @property
    def locator(self) -> tuple[str, int, int]:
        """Return the identity used to reject duplicate index rows."""

        return (self.source_file, self.row_group, self.row_index)


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Validated, deterministic input plan created before model loading."""

    corpus_dir: Path
    view_dir: Path
    corpus_files: tuple[Path, ...]
    view_files: tuple[Path, ...]
    rows: int
    corpus_tree_sha256: str
    view_tree_sha256: str
    row_offsets: Mapping[str, Mapping[str, int]]
    max_records: int | None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable plan summary."""

        return {
            "rows": self.rows,
            "corpus_files": len(self.corpus_files),
            "view_files": len(self.view_files),
            "corpus_tree_sha256": self.corpus_tree_sha256,
            "view_tree_sha256": self.view_tree_sha256,
            "order": "sorted view filename, then JSONL line order",
            "max_records": self.max_records,
        }


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """Versioned model and batching settings that define an index."""

    model: str
    model_revision: str
    device: str
    batch_size_short: int
    batch_size_fulltext: int
    max_length_short: int
    max_length_fulltext: int
    checkpoint_rows: int
    local_files_only: bool

    def validate(self) -> None:
        """Reject settings that would create an ambiguous or unusable index."""

        if not self.model.strip():
            raise ValueError("model must not be blank.")
        if not self.model_revision.strip() or self.model_revision == "SET_IMMUTABLE_REVISION":
            raise ValueError("An immutable model revision is required.")
        positive = {
            "batch_size_short": self.batch_size_short,
            "batch_size_fulltext": self.batch_size_fulltext,
            "max_length_short": self.max_length_short,
            "max_length_fulltext": self.max_length_fulltext,
            "checkpoint_rows": self.checkpoint_rows,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"These settings must be positive: {', '.join(invalid)}")


@dataclass(frozen=True, slots=True)
class BuildState:
    """Small checkpoint committed after a complete row batch."""

    schema_version: int
    fingerprint: str
    rows: int
    dimensions: int
    rows_complete: int
    metadata_bytes: int


class DocumentEncoder(Protocol):
    """Minimal encoder interface used by the storage algorithm."""

    dimensions: int

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Any:
        """Return an L2-normalized float matrix in input order."""

    def close(self) -> None:
        """Release model resources."""


class SourceRowLoader(Protocol):
    """Source-row interface kept injectable for deterministic unit tests."""

    def load(self, record: ViewRecord) -> Mapping[str, Any]:
        """Load the Parquet row identified by ``record``."""


def _require_directory(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _discover_files(root: Path, pattern: str, *, description: str) -> tuple[Path, ...]:
    if "/" in pattern or "\\" in pattern or pattern in {"", ".", ".."}:
        raise ValueError(f"{description} pattern must be a filename glob.")
    files = tuple(sorted(path for path in root.glob(pattern) if path.is_file()))
    if not files:
        raise FileNotFoundError(f"No {description} matched {pattern!r} below {root}")
    return files


def _tree_hash(root: Path, files: Sequence[Path]) -> str:
    """Hash sorted relative paths and file hashes like artifact verification."""

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _as_nonnegative_int(value: object, *, location: str, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{location}: {field} must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise ValueError(f"{location}: {field} must be an integer")
    if parsed < 0:
        raise ValueError(f"{location}: {field} must not be negative")
    return parsed


def _parse_view_record(payload: object, *, path: Path, line_number: int) -> ViewRecord:
    location = f"{path}:{line_number}"
    if not isinstance(payload, dict):
        raise ValueError(f"{location}: expected a JSON object")
    source = str(payload.get("source_parquet") or "").strip()
    if not source:
        raise ValueError(f"{location}: source_parquet must not be blank")
    source_file = Path(source).name
    if not source_file.endswith(".parquet"):
        raise ValueError(f"{location}: source_parquet must identify a Parquet file")
    row_group = _as_nonnegative_int(
        payload.get("row_group"),
        location=location,
        field="row_group",
    )
    row_index = _as_nonnegative_int(
        payload.get("row_index_in_group"),
        location=location,
        field="row_index_in_group",
    )
    doc_id = str(payload.get("doc_id") or "").strip() or None
    return ViewRecord(
        payload=payload,
        source_file=source_file,
        row_group=row_group,
        row_index=row_index,
        doc_id=doc_id,
        input_file=path.name,
        input_line=line_number,
    )


def iter_view_records(
    files: Sequence[Path],
    *,
    max_records: int | None = None,
) -> Iterator[ViewRecord]:
    """Stream document-view JSONL in canonical filename and line order."""

    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided.")
    emitted = 0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                yield _parse_view_record(
                    payload,
                    path=path,
                    line_number=line_number,
                )
                emitted += 1
                if max_records is not None and emitted >= max_records:
                    return


def _validate_source_locators(
    corpus_files: Sequence[Path],
    locators: set[tuple[str, int, int]],
) -> None:
    """Verify every referenced row against Parquet metadata without loading row data."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Index planning requires the `offline-index` dependency group.") from exc

    paths = {path.name: path for path in corpus_files}
    referenced_groups: dict[str, dict[int, int]] = {}
    for source, row_group, row_index in locators:
        groups = referenced_groups.setdefault(source, {})
        groups[row_group] = max(groups.get(row_group, -1), row_index)

    for source in sorted(referenced_groups):
        parquet_file = parquet.ParquetFile(paths[source])
        for row_group, largest_row in sorted(referenced_groups[source].items()):
            if row_group >= parquet_file.num_row_groups:
                raise IndexError(f"{source}: missing row group {row_group}")
            row_count = parquet_file.metadata.row_group(row_group).num_rows
            if largest_row >= row_count:
                raise IndexError(
                    f"{source}: row group {row_group} has {row_count} rows, "
                    f"but document views reference row {largest_row}"
                )


def _verify_plan_inputs(plan: BuildPlan) -> None:
    """Reject input changes between planning and index publication."""

    corpus_hash = _tree_hash(plan.corpus_dir, plan.corpus_files)
    view_hash = _tree_hash(plan.view_dir, plan.view_files)
    if corpus_hash != plan.corpus_tree_sha256 or view_hash != plan.view_tree_sha256:
        raise RuntimeError("Corpus or document-view inputs changed during index construction.")


def create_build_plan(
    *,
    corpus_dir: Path,
    view_dir: Path,
    corpus_pattern: str = "*.parquet",
    view_pattern: str = "shard_*.jsonl",
    max_records: int | None = None,
) -> BuildPlan:
    """Validate all row identities and hash every input before GPU work."""

    corpus_root = _require_directory(corpus_dir, description="corpus directory")
    view_root = _require_directory(view_dir, description="document-view directory")
    corpus_files = _discover_files(
        corpus_root,
        corpus_pattern,
        description="corpus files",
    )
    view_files = _discover_files(
        view_root,
        view_pattern,
        description="document-view files",
    )
    corpus_names = [path.name for path in corpus_files]
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("Corpus filenames must be unique for stable row materialization.")
    known_sources = set(corpus_names)

    seen: set[tuple[str, int, int]] = set()
    rows = 0
    offset_counts: dict[str, dict[str, int]] = {}
    for record in iter_view_records(view_files, max_records=max_records):
        if record.source_file not in known_sources:
            raise ValueError(
                f"{record.input_file}:{record.input_line}: source {record.source_file!r} "
                "is absent from the corpus directory"
            )
        if record.locator in seen:
            source, row_group, row_index = record.locator
            raise ValueError(
                f"Duplicate source row in document views: {source}:rg{row_group}:row{row_index}"
            )
        seen.add(record.locator)
        entry = offset_counts.setdefault(
            record.input_file,
            {"start": rows, "rows": 0},
        )
        entry["rows"] += 1
        rows += 1
    if rows == 0:
        raise ValueError("Document-view inputs contain no records.")
    _validate_source_locators(corpus_files, seen)

    selected_view_files = tuple(path for path in view_files if path.name in offset_counts)
    return BuildPlan(
        corpus_dir=corpus_root,
        view_dir=view_root,
        corpus_files=corpus_files,
        view_files=selected_view_files,
        rows=rows,
        corpus_tree_sha256=_tree_hash(corpus_root, corpus_files),
        view_tree_sha256=_tree_hash(view_root, selected_view_files),
        row_offsets=offset_counts,
        max_records=max_records,
    )


class ParquetRowLoader:
    """Load source rows while caching only the active Parquet row group."""

    def __init__(self, plan: BuildPlan) -> None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Index construction requires the `offline-index` dependency group."
            ) from exc
        self._parquet = parquet
        self._paths = {path.name: path for path in plan.corpus_files}
        self._active_key: tuple[str, int] | None = None
        self._active_rows: list[Mapping[str, Any]] = []

    def load(self, record: ViewRecord) -> Mapping[str, Any]:
        key = (record.source_file, record.row_group)
        if key != self._active_key:
            path = self._paths[record.source_file]
            parquet_file = self._parquet.ParquetFile(path)
            if record.row_group >= parquet_file.num_row_groups:
                raise IndexError(f"{record.source_file}: missing row group {record.row_group}")
            self._active_rows = parquet_file.read_row_group(record.row_group).to_pylist()
            self._active_key = key
        if record.row_index >= len(self._active_rows):
            raise IndexError(
                f"{record.source_file}: row group {record.row_group} has no row {record.row_index}"
            )
        return self._active_rows[record.row_index]


class QwenDocumentEncoder:
    """Pinned Qwen last-token encoder shared across all five document views."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        device: str,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen index encoding requires the `offline-gpu` dependency group."
            ) from exc
        self._torch = torch
        self._device = device
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
            device_map={"": device},
            attn_implementation="sdpa",
        )
        self._model.eval()
        self.dimensions = int(self._model.config.hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> Any:
        """Encode with last-token pooling and return normalized float32 rows."""

        import numpy as np

        torch = self._torch
        batches: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                values = [
                    value if clean(value) else " " for value in texts[start : start + batch_size]
                ]
                inputs = self._tokenizer(
                    values,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
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
        """Release model memory after the index has been finalized."""

        torch = self._torch
        del self._model
        del self._tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON file and its directory entry."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _matrix_name(field: str) -> str:
    return f"{field}.f16.memmap"


def _partial_matrix_name(field: str) -> str:
    return f"{_matrix_name(field)}.partial"


def _fingerprint(
    plan: BuildPlan,
    config: EncodingConfig,
    *,
    dimensions: int,
) -> str:
    payload = {
        "plan": plan.summary(),
        "encoding": asdict(config),
        "dimensions": dimensions,
        "fields": FIELDS,
        "dtype": MATRIX_DTYPE,
        "text_contract": "document_views_v1",
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_state(path: Path) -> BuildState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = BuildState(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid build checkpoint: {path}") from exc
    if state.schema_version != STATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported build checkpoint schema: {state.schema_version}")
    return state


def _validate_vectors(
    values: Any,
    *,
    rows: int,
    dimensions: int,
    field: str,
) -> Any:
    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (rows, dimensions):
        raise ValueError(
            f"{field}: encoder returned shape {matrix.shape}, expected {(rows, dimensions)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{field}: encoder returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=5e-3, atol=5e-3):
        raise ValueError(f"{field}: encoder rows are not L2-normalized")
    return matrix


def _open_matrices(
    output_dir: Path,
    *,
    rows: int,
    dimensions: int,
    resume: bool,
) -> dict[str, Any]:
    import numpy as np

    mode = "r+" if resume else "w+"
    expected_bytes = rows * dimensions * np.dtype(np.float16).itemsize
    matrices: dict[str, Any] = {}
    for field in FIELDS:
        partial_path = output_dir / _partial_matrix_name(field)
        final_path = output_dir / _matrix_name(field)
        path = partial_path
        if resume:
            existing = [
                candidate for candidate in (partial_path, final_path) if candidate.is_file()
            ]
            if len(existing) != 1:
                raise ValueError(
                    f"{field}: expected exactly one partial or finalized matrix, "
                    f"found {len(existing)}"
                )
            path = existing[0]
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(f"{path}: matrix size {actual_bytes}, expected {expected_bytes}")
        matrices[field] = np.memmap(
            path,
            dtype=np.float16,
            mode=mode,
            shape=(rows, dimensions),
        )
    return matrices


def _metadata_record(
    *,
    record: ViewRecord,
    global_index: int,
    texts: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "global_idx": global_index,
        "doc_id": record.doc_id,
        "source_parquet": record.source_file,
        "row_group": record.row_group,
        "row_index_in_group": record.row_index,
        "nonempty": {field: bool(clean(texts[field])) for field in FIELDS},
    }


def _batched_records(
    records: Iterator[ViewRecord],
    *,
    batch_size: int,
) -> Iterator[list[ViewRecord]]:
    batch: list[ViewRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _prepare_output(
    output_dir: Path,
    *,
    resume: bool,
) -> Path:
    output = output_dir.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError(f"Index output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries = list(output.iterdir())
    if resume:
        if not (output / STATE_FILE).is_file():
            raise ValueError(f"--resume requires {output / STATE_FILE}")
        if (output / "manifest.json").exists():
            raise ValueError("The index already has a final manifest and is complete.")
    elif entries:
        raise ValueError(
            f"Index output directory must be new or empty: {output}. "
            "Use --resume only for a matching checkpoint."
        )
    return output


def build_dense_index(
    *,
    plan: BuildPlan,
    output_dir: Path,
    config: EncodingConfig,
    encoder: DocumentEncoder,
    row_loader: SourceRowLoader,
    resume: bool = False,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Build or resume an index and publish ``manifest.json`` last."""

    import numpy as np

    config.validate()
    if expected_rows is not None and plan.rows != expected_rows:
        raise ValueError(f"Input rows {plan.rows} do not match expected_rows {expected_rows}.")
    if encoder.dimensions <= 0:
        raise ValueError("Encoder dimensions must be positive.")
    output = _prepare_output(output_dir, resume=resume)
    fingerprint = _fingerprint(plan, config, dimensions=encoder.dimensions)
    state_path = output / STATE_FILE
    metadata_partial = output / f"{METADATA_FILE}.partial"

    if resume:
        state = _load_state(state_path)
        if (
            state.fingerprint != fingerprint
            or state.rows != plan.rows
            or state.dimensions != encoder.dimensions
        ):
            raise ValueError("Build checkpoint does not match the requested inputs/configuration.")
        if not 0 <= state.rows_complete <= plan.rows:
            raise ValueError("Build checkpoint rows_complete is outside the index.")
        finalized_matrices = [
            output / _matrix_name(field)
            for field in FIELDS
            if (output / _matrix_name(field)).is_file()
        ]
        if finalized_matrices and state.rows_complete != plan.rows:
            raise ValueError("Finalized matrices are only valid after all rows are committed.")
        metadata_final = output / METADATA_FILE
        existing_metadata = [
            candidate for candidate in (metadata_partial, metadata_final) if candidate.is_file()
        ]
        if len(existing_metadata) != 1:
            raise ValueError(
                "Expected exactly one partial or finalized metadata file, "
                f"found {len(existing_metadata)}"
            )
        metadata_working = existing_metadata[0]
        if metadata_working == metadata_final and state.rows_complete != plan.rows:
            raise ValueError("Finalized metadata is only valid after all rows are committed.")
        if metadata_working.stat().st_size < state.metadata_bytes:
            raise ValueError(f"{metadata_working}: file is shorter than the checkpoint")
        with metadata_working.open("r+b") as handle:
            handle.truncate(state.metadata_bytes)
        rows_complete = state.rows_complete
        metadata_bytes = state.metadata_bytes
    else:
        rows_complete = 0
        metadata_bytes = 0
        metadata_working = metadata_partial
        metadata_partial.touch(exist_ok=False)
        _atomic_write_json(
            state_path,
            asdict(
                BuildState(
                    schema_version=STATE_SCHEMA_VERSION,
                    fingerprint=fingerprint,
                    rows=plan.rows,
                    dimensions=encoder.dimensions,
                    rows_complete=0,
                    metadata_bytes=0,
                )
            ),
        )

    matrices = _open_matrices(
        output,
        rows=plan.rows,
        dimensions=encoder.dimensions,
        resume=resume,
    )
    records = iter_view_records(plan.view_files, max_records=plan.max_records)
    for _ in range(rows_complete):
        next(records)

    try:
        with metadata_working.open("ab", buffering=0) as metadata_handle:
            for batch in _batched_records(records, batch_size=config.checkpoint_rows):
                text_rows = [
                    build_texts(record.payload, row_loader.load(record)) for record in batch
                ]
                start = rows_complete
                end = start + len(batch)
                for field in FIELDS:
                    batch_size = (
                        config.batch_size_fulltext
                        if field == "fulltext"
                        else config.batch_size_short
                    )
                    max_length = (
                        config.max_length_fulltext
                        if field == "fulltext"
                        else config.max_length_short
                    )
                    vectors = encoder.encode(
                        [texts[field] for texts in text_rows],
                        batch_size=batch_size,
                        max_length=max_length,
                    )
                    matrices[field][start:end] = _validate_vectors(
                        vectors,
                        rows=len(batch),
                        dimensions=encoder.dimensions,
                        field=field,
                    ).astype(np.float16, copy=False)

                for position, (record, texts) in enumerate(
                    zip(batch, text_rows, strict=True),
                    start=start,
                ):
                    line = (
                        json.dumps(
                            _metadata_record(
                                record=record,
                                global_index=position,
                                texts=texts,
                            ),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        + b"\n"
                    )
                    metadata_handle.write(line)
                    metadata_bytes += len(line)

                for matrix in matrices.values():
                    matrix.flush()
                os.fsync(metadata_handle.fileno())
                rows_complete = end
                _atomic_write_json(
                    state_path,
                    asdict(
                        BuildState(
                            schema_version=STATE_SCHEMA_VERSION,
                            fingerprint=fingerprint,
                            rows=plan.rows,
                            dimensions=encoder.dimensions,
                            rows_complete=rows_complete,
                            metadata_bytes=metadata_bytes,
                        )
                    ),
                )
                print(
                    json.dumps(
                        {
                            "event": "index_progress",
                            "rows_complete": rows_complete,
                            "rows_total": plan.rows,
                        }
                    ),
                    flush=True,
                )
    finally:
        encoder.close()

    if rows_complete != plan.rows:
        raise RuntimeError(f"Encoded {rows_complete} of {plan.rows} planned rows.")
    for matrix in matrices.values():
        matrix.flush()
    del matrices
    _verify_plan_inputs(plan)

    for field in FIELDS:
        partial_path = output / _partial_matrix_name(field)
        final_path = output / _matrix_name(field)
        if partial_path.is_file():
            partial_path.replace(final_path)
        elif not final_path.is_file():
            raise FileNotFoundError(f"Missing finalized matrix: {final_path}")
    metadata_final = output / METADATA_FILE
    if metadata_partial.is_file():
        metadata_partial.replace(metadata_final)
    elif not metadata_final.is_file():
        raise FileNotFoundError(f"Missing finalized metadata: {metadata_final}")
    _atomic_write_json(output / ROW_OFFSETS_FILE, plan.row_offsets)

    manifest: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "rows": plan.rows,
        "dimensions": encoder.dimensions,
        "dtype": MATRIX_DTYPE,
        "fields": list(FIELDS),
        "model": config.model,
        "model_revision": config.model_revision,
        "pooling": "last_token",
        "normalization": "l2",
        "text_contract": "document_views_v1",
        "metadata_file": METADATA_FILE,
        "matrix_files": {field: _matrix_name(field) for field in FIELDS},
        "row_offsets_file": ROW_OFFSETS_FILE,
        "source": {
            **plan.summary(),
            "corpus_path_recording": "relative filenames only",
        },
        "encoding": {
            "batch_size_short": config.batch_size_short,
            "batch_size_fulltext": config.batch_size_fulltext,
            "max_length_short": config.max_length_short,
            "max_length_fulltext": config.max_length_fulltext,
            "blank_text": "single_space",
            "shared_model_across_fields": True,
        },
    }
    _atomic_write_json(output / "manifest.json", manifest)
    state_path.unlink()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-pattern", default="*.parquet")
    parser.add_argument("--view-pattern", default="shard_*.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size-short", type=int, default=256)
    parser.add_argument("--batch-size-fulltext", type=int, default=32)
    parser.add_argument("--max-length-short", type=int, default=1_024)
    parser.add_argument("--max-length-fulltext", type=int, default=8_192)
    parser.add_argument("--checkpoint-rows", type=int, default=256)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--max-records",
        type=int,
        help="Build a deterministic prefix for smoke tests; omit for paper-scale runs.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face downloads instead of requiring the pinned local snapshot.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted build with exactly matching inputs and settings.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and hash inputs without loading the embedding model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EncodingConfig(
        model=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size_short=args.batch_size_short,
        batch_size_fulltext=args.batch_size_fulltext,
        max_length_short=args.max_length_short,
        max_length_fulltext=args.max_length_fulltext,
        checkpoint_rows=args.checkpoint_rows,
        local_files_only=not args.allow_model_download,
    )
    config.validate()
    plan = create_build_plan(
        corpus_dir=args.corpus_dir,
        view_dir=args.views_dir,
        corpus_pattern=args.corpus_pattern,
        view_pattern=args.view_pattern,
        max_records=args.max_records,
    )
    if args.expected_rows is not None and plan.rows != args.expected_rows:
        raise ValueError(f"Input rows {plan.rows} do not match expected_rows {args.expected_rows}.")
    print(json.dumps({"event": "index_plan", **plan.summary()}, sort_keys=True))
    if args.plan_only:
        return 0

    encoder = QwenDocumentEncoder(
        model=config.model,
        revision=config.model_revision,
        device=config.device,
        local_files_only=config.local_files_only,
    )
    manifest = build_dense_index(
        plan=plan,
        output_dir=args.output_dir,
        config=config,
        encoder=encoder,
        row_loader=ParquetRowLoader(plan),
        resume=args.resume,
        expected_rows=args.expected_rows,
    )
    print(
        json.dumps(
            {
                "event": "index_complete",
                "rows": manifest["rows"],
                "dimensions": manifest["dimensions"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
