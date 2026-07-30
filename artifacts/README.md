# External artifacts

The Git repository contains source and tiny synthetic fixtures only. The validation
CSV files, closed vocabularies, case-law snapshot, dense index, dense candidates,
verifier inputs and outputs, saved sparse evidence, and reference results are
external because they are large or have separate redistribution terms.

`manifest.toml` is the machine-readable artifact contract. It records:

- a stable artifact identifier and destination below `artifacts/downloads`;
- file or directory-tree kind;
- exact byte size and, for trees, file count;
- SHA-256 file or tree checksum;
- provenance; and
- the current redistribution boundary.

After obtaining files through an authorized channel:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

Verification never downloads, modifies, or substitutes an artifact. The tree checksum
is computed from sorted relative paths and each file's SHA-256.

## Availability status

The authors' artifact set currently has **no DOI or stable URL**. This release
therefore provides source and a checksum manifest; full reproduction is blocked until
the authors publish an authorized archive. In particular:

- a Stage 0 builder can produce a new dense index, while the generated document-view
  snapshot required for byte-identical historical rebuilding was not versioned;
- the historical verifier file intentionally has only 19 of 20 batches and 95 of 100
  scores;
- the exact index is bound by its tree hash and the hashed model configuration because
  its original manifest omitted model and revision fields; and
- a newly generated complete verifier run is a new result, not a replacement for the
  historical snapshot.

Missing files should be reported as unavailable, not replaced by a convenient
alternative.
