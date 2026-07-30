# Reproducibility contract

BetterCallAgent distinguishes software verification, historical reconstruction, and a
new experiment.

## 1. Synthetic software verification

```bash
BCA_FIXTURE_OUT="$(mktemp -d)/bettercallagent-fixture"
uv run --locked python -m offline.run \
  --config offline/fixtures/config.toml \
  --output "$BCA_FIXTURE_OUT"
```

This explicit fixture makes no network call. Its invented questions, documents,
rankings, scores, and labels test program behavior only. A passing fixture does not
reproduce either reported Macro-F1 value.

## 2. Historical July 2026 reconstruction

The checksum-bound snapshot contains ten validation queries and a verifier artifact
with 19 of 20 batches:

- 100 reranker inputs were prepared;
- 95 candidates were scored; and
- `val_002` ranks 6–10 have no verifier scores.

The fixed-vote reference submission evaluates to `0.4709962618` Macro-F1. The
historical adapter applies the saved balanced sparse support to the same incomplete
score set and asserts `0.4806246255438345`. Together they reconstruct the two
reported exploratory values from the same artifact lineage. The adapter lists every
missing identity in its output manifest rather than silently treating it as scored.

Follow [the historical replay guide](../experiments/historical_validation/README.md).
Both thresholds were selected on this same ten-query split; neither value is a
held-out estimate.

## 3. Fresh strict experiment

After the authors publish the authorized external artifact archive:

```bash
uv sync --locked --extra dev --extra online --extra offline-gpu
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads

mkdir -p runs
cp configs/offline.example.toml runs/paper_validation.toml
export BCA_RERANK_BASE_URL="https://your-provider.example/v1"
export BCA_RERANK_API_KEY="read-at-runtime-from-your-secret-manager"
uv run --locked python -m offline.run --config runs/paper_validation.toml
```

The full runner:

- rejects missing, malformed, duplicate, or hash-mismatched inputs;
- records SHA-256 values for file and directory inputs;
- records the resolved non-secret configuration, command, seed, platform, dependency
  versions, Git state, and per-stage status;
- records the dense text-preparation, shared-model lifecycle, and single-device
  placement settings;
- binds replay scores to query, candidate, rank, prompt, batch, model, and input
  fingerprints;
- requires complete candidate-score joins; and
- rejects a non-empty output directory.

The generated run manifest, intermediate artifacts, selection audit, predictions, and
metrics stay together in the unique output directory.

A complete new verifier run has 100 scores and is a **new result**. It must not be
reported as a reproduction of the incomplete historical run, even if its metric is
similar.

## External artifact boundary

`artifacts/manifest.toml` lists the exact filenames, byte sizes, file counts, and
SHA-256 values for the authors' snapshot. Verification is local and never downloads
or substitutes data.

The external archive currently has no DOI or stable URL. This release therefore
provides source and a checksum manifest; full reproduction is blocked until the
authors publish files they are authorized to redistribute.

The exact historical dense index is supplied as a tree artifact. Stage 0 builds new
indexes from validated Parquet and document-view inputs. Its predecessor manifest
omitted model and revision fields, so the clean release
binds it through:

1. the complete eight-file tree hash; and
2. the hashed run configuration pinning
   `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

Do not combine a different index, model revision, provider label, prompt, or cached
score file and call it the same run.

## Dependency and source identity

- `uv.lock` is the Python dependency lock.
- `frontend/package-lock.json` is the JavaScript dependency lock.
- Clean runs record the Git commit. Dirty runs are marked and include a working-tree
  digest.
- Secrets are read at runtime and are excluded from resolved configuration.
