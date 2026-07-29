# Historical ten-query validation reconstruction

This adapter reconstructs the two reported July 2026 validation measurements from the
exact saved artifact set:

| Configuration | Macro citation F1 |
|---|---:|
| Dense fixed-vote reference submission | `0.4709962618` |
| Same incomplete scores plus saved balanced sparse support | `0.4806246255438345` |

This is not a fresh run. The verifier completed only 19 of 20 batches: 100 candidates
were prepared, 95 were scored, and `val_002` ranks 6–10 are absent. Gate and support
thresholds were selected on these same ten queries.

The adapter makes no model or network request. It:

- verifies the exact file and directory-tree hashes;
- joins scores to immutable document identities;
- reports every missing score identity;
- uses the shared audited extractor, vocabulary, and evidence gate; and
- asserts the saved support result exactly.

## Run

First obtain the authorized files and verify the complete artifact set:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

Then evaluate the fixed-vote reference and reconstruct the support variant:

```bash
BCA_HIST_OUT="$(mktemp -d)"

uv run --locked python -m offline.stages.step_07_evaluate \
  --submission artifacts/downloads/reference/submission_fixed_vote_rule.csv \
  --gold artifacts/downloads/eval/val.csv \
  --output "$BCA_HIST_OUT/fixed_vote_metrics.json"

uv run --locked python \
  -m experiments.historical_validation.replay_reported_result \
  --queries artifacts/downloads/eval/val.csv \
  --reranker-input artifacts/downloads/derived/val_qwen_top10_fulltext.jsonl \
  --legacy-scores artifacts/downloads/derived/val_qwen_top10_rerank_results.jsonl \
  --sparse-traces artifacts/downloads/derived/sira_bm25_traces \
  --laws artifacts/downloads/eval/laws_de.csv \
  --courts artifacts/downloads/eval/court_considerations.csv.gz \
  --output "$BCA_HIST_OUT/support_reconstruction"

cat "$BCA_HIST_OUT/fixed_vote_metrics.json"
cat "$BCA_HIST_OUT/support_reconstruction/metrics.json"
cat "$BCA_HIST_OUT/support_reconstruction/historical_replay_manifest.json"
```

The command-line name `--legacy-scores` describes the historical file schema; it does
not permit a different score artifact. The adapter rejects every hash or shape
mismatch.

## Interpretation

The reconstruction is provenance for exploratory measurements on `n=10`. It does not:

- fill or infer the five missing scores;
- establish a held-out improvement;
- show that thresholds generalize; or
- imply that a complete 100-score rerun should match either value.

A fresh strict run is a new result and must be labeled with its own model, prompt,
input hashes, and metric.
