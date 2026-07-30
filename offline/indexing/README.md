# Dense index construction and contract

Stage 0 builds the complete five-view vector database consumed by Stage 1. The
reviewer-facing implementation is
[`offline/stages/step_00_build_dense_index.py`](../stages/step_00_build_dense_index.py),
and the installed command is `bca-build-index`.

## Output layout

```text
qwen3_embedding_8b/
├── manifest.json
├── metadata.jsonl
├── row_offsets.json
├── normal_query.f16.memmap
├── meta_searchterm.f16.memmap
├── keywords.f16.memmap
├── fulltext.f16.memmap
└── citations.f16.memmap
```

The five matrices contain L2-normalized `float16` vectors in identical row order.
`metadata.jsonl` maps each row to a stable document identifier or Parquet locator.
`manifest.json` records row count, dimensions, data type, field order, metadata
filename, and matrix filenames. The complete audited tree has eight files; the reader
requires the seven entries named above, while verification binds every file.

The audited predecessor manifest did not contain embedding `model` or
`model_revision` fields. This release therefore binds the exact index through both:

1. the complete eight-file tree checksum
   `e5b12c92060fc44e9ee04b57e683fb753d842d146f3a768b0cad8265e6131a23`;
2. the hashed run configuration pinning
   `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

Newly produced indexes should include both fields directly in `manifest.json`.

## Build inputs

Stage 0 accepts two versioned inputs:

1. `--corpus-dir`: a flat directory of Parquet files with unique basenames.
2. `--views-dir`: sorted `shard_*.jsonl` files containing one unique source-row
   locator and the generated search views per line.

Each JSONL object must contain `source_parquet`, `row_group`,
`row_index_in_group`, optional `doc_id`, and `content`. `content` may be an object
or JSON string with `normal_query`, `meta_searchterm_de`, `keywords_de`, and
`keywords_en`. Full-text and citation views are derived from the located Parquet
row by versioned functions in `offline/indexing/document_views.py`.
Locators should be JSON integers; canonical unsigned decimal strings are also
accepted for compatibility with the audited legacy view shards.

## Copy-paste build

```bash
uv sync --locked --extra offline-gpu

export BCA_CORPUS_DIR="$PWD/artifacts/downloads/corpus"
export BCA_VIEWS_DIR="$PWD/artifacts/downloads/document_views"
export BCA_INDEX_DIR="$PWD/artifacts/generated/qwen3_embedding_8b"

# Validate every locator and hash both input trees without loading a model.
uv run --locked bca-build-index \
  --corpus-dir "$BCA_CORPUS_DIR" \
  --views-dir "$BCA_VIEWS_DIR" \
  --output-dir "$BCA_INDEX_DIR" \
  --plan-only

# Create the complete five-view vector database.
uv run --locked bca-build-index \
  --corpus-dir "$BCA_CORPUS_DIR" \
  --views-dir "$BCA_VIEWS_DIR" \
  --output-dir "$BCA_INDEX_DIR" \
  --model Qwen/Qwen3-Embedding-8B \
  --model-revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
  --device cuda:0
```

The pinned model must already be cached. `--allow-model-download` makes network
acquisition explicit. For a fast integration test, add `--max-records 32`.

An interrupted run is resumed by repeating the exact build command with
`--resume`. The builder rejects changed inputs or settings, truncates metadata to
the last committed checkpoint, overwrites the next matrix range, and publishes
`manifest.json` last.

## Scale and execution

The verified tree contains eight files and 40,519,716,885 bytes. Stage 1 streams
metadata, scans matrix chunks with exact inner products, and keeps only fused rows in
memory. A compatible GPU is recommended; CPU execution is supported by configuration
but is substantially slower.

Query encoding preserves non-blank view strings exactly and maps blank values to one
space. A single pinned model encodes the fields in canonical order and is released
before matrix search. The run manifest records the text-preparation contract,
`shared_across_fields` lifecycle, and single-device placement.

The index is not stored in Git and currently has no DOI or stable download URL.
Obtain the authorized artifact from the authors, place it at
`artifacts/downloads/index/qwen3_embedding_8b`, and verify it:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

## Historical boundary

The audited paper index remains identified by its tree checksum. Its original
generated document-view snapshot was not versioned as a release artifact. Stage 0
therefore closes the source-code gap and creates provenance-bound **new** indexes;
it does not retroactively make a new index byte-identical to the historical tree.

Do not substitute a newly encoded index, mix matrix row orders, or combine index and
query embeddings from different revisions and call the output a reproduction.
