# Dense index contract

Stage 1 consumes the exact five-view dense index used by the authors. This release
contains an index **reader**, not an index builder.

## Reader-required layout

```text
qwen3_embedding_8b/
├── manifest.json
├── metadata.jsonl
├── normal_query.f16.memmap
├── meta_searchterm.f16.memmap
├── keywords.f16.memmap
├── fulltext.f16.memmap
├── citations.f16.memmap
└── …                       # any additional file remains part of the tree hash
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

## Scale and execution

The verified tree contains eight files and 40,519,716,885 bytes. Stage 1 streams
metadata, scans matrix chunks with exact inner products, and keeps only fused rows in
memory. A compatible GPU is recommended; CPU execution is supported by configuration
but is substantially slower.

Query encoding constructs and releases one pinned model instance per retrieval
field and loads it directly through a single-device map. This `fresh_per_field`
lifecycle and placement match the audited run and are part of the retrieval
provenance rather than interchangeable performance optimizations.

The index is not stored in Git and currently has no DOI or stable download URL.
Obtain the authorized artifact from the authors, place it at
`artifacts/downloads/index/qwen3_embedding_8b`, and verify it:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

Do not substitute a newly encoded index, mix matrix row orders, or combine index and
query embeddings from different revisions and call the output a reproduction.
