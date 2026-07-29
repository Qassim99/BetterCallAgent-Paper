# Configuration

Configuration files contain versioned, non-secret experiment parameters.

- `models.toml` records the historical model identities and clearly marks optional
  recommendations as unvalidated.
- `offline.example.toml` pins the full seven-stage run, artifact paths, five retrieval
  fields and weights, exact embedding revision, verifier parameters, citation gate,
  and input checksums.

Create a run-specific copy:

```bash
mkdir -p runs
cp configs/offline.example.toml runs/paper_validation.toml
```

Review paths, device selection, and provider environment-variable names in the copy.
Do not weaken the integrity table for a reproduction.

Secrets never belong in TOML. The example refers only to environment variable names:

```bash
export BCA_RERANK_BASE_URL="https://your-provider.example/v1"
export BCA_RERANK_API_KEY="read-at-runtime-from-your-secret-manager"
```

The runner stores the resolved configuration without secret values, hashes the inputs
and configuration, records source and dependency identity, and rejects reuse of a
non-empty output directory.

The exact predecessor index manifest omitted embedding model and revision metadata.
For that artifact, the tree checksum in `[integrity]` and the pinned
`[retrieval]` model/revision are a joint identity.
