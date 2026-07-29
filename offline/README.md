# Offline paper pipeline

This folder contains the seven-stage citation-prediction experiment. It is separate
from the interactive online application and is the only pipeline that reads gold
labels for evaluation.

| Stage | File | Purpose |
|---|---|---|
| 1 | `stages/step_01_retrieve_dense.py` | Build five exact query views, retrieve from five normalized Qwen matrices, and apply weighted reciprocal-rank fusion |
| 2 | `stages/step_02_retrieve_sparse_support.py` | Aggregate citation counts from saved balanced sparse evidence |
| 3 | `stages/step_03_materialize_documents.py` | Join candidates to full documents in the case-law Parquet snapshot |
| 4 | `stages/step_04_prepare_rerank_input.py` | Create ordered, identity-bound top-N verifier batches |
| 5 | `stages/step_05_rerank.py` | Run the configured verifier or consume an exact fingerprint-bound replay |
| 6 | `stages/step_06_select_citations.py` | Apply the fixed vote and sparse-support policy |
| 7 | `stages/step_07_evaluate.py` | Compute query-level citation Macro-F1 offline |

`run.py` validates configuration, input integrity, joins, stage coverage, and output
isolation. It has two explicit modes.

## Fixture mode

```bash
BCA_FIXTURE_OUT="$(mktemp -d)/bettercallagent-fixture"
uv run --locked python -m offline.run \
  --config offline/fixtures/config.toml \
  --output "$BCA_FIXTURE_OUT"
cat "$BCA_FIXTURE_OUT/07_metrics.json"
```

The fixture contains invented records and makes no network request. It verifies
software behavior only; its score is not a paper result.

## Full mode

Full mode needs the exact external files in `artifacts/manifest.toml`, local model
weights for the pinned embedding revision, a compatible accelerator, and a live
verifier endpoint:

```bash
uv sync --locked --extra dev --extra offline-gpu
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads

mkdir -p runs
cp configs/offline.example.toml runs/paper_validation.toml
export BCA_RERANK_BASE_URL="https://your-provider.example/v1"
export BCA_RERANK_API_KEY="read-at-runtime-from-your-secret-manager"
uv run --locked python -m offline.run --config runs/paper_validation.toml
```

Edit only the copied configuration. Do not put secrets in TOML. The output directory
must be new or empty.

The exact dense index is an external, hash-verified input. No index builder is
included in this release; see [indexing/README.md](indexing/README.md).

## Historical result versus a new run

The reported July 2026 experiment used ten validation queries and only 95 of 100
verifier scores. One five-document batch (`val_002` ranks 6–10) is missing. It records:

- fixed-vote Macro-F1 `0.4709962618`; and
- fixed vote plus saved sparse support Macro-F1 `0.4806246255438345`.

Thresholds were selected on the same ten-query split. Use the dedicated
[historical adapter](../experiments/historical_validation/README.md) to reconstruct
that incomplete snapshot.

The strict full runner requires complete score coverage. Its result is therefore a
new experiment and must not be labeled as the historical result.

## Run outputs

A successful run keeps these items together:

- the resolved non-secret configuration and command;
- Git, platform, dependency, seed, and input-hash provenance;
- machine-readable stage status;
- intermediate candidate, document, verifier, support, and audit artifacts; and
- the final submission and metrics.

The external archive currently has no DOI or stable URL. This release provides source
and a checksum manifest; full execution is blocked until the authors publish an
authorized artifact archive.
