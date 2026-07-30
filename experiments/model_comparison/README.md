# Hosted SOOFI verifier comparison

This experiment produces a fresh, complete verifier run on the exact 100-candidate
validation artifact used by BetterCallAgent. It reuses the production offline
pipeline instead of maintaining a second implementation:

1. Stage 5 applies the historical German legal reranking rubric in ordered batches
   of five candidates.
2. Stage 6 applies the existing fixed-vote citation gate and saved sparse support.
3. Stage 7 computes citation-level macro F1 against the validation labels.

This is a new model comparison. It must not be presented as a reconstruction of the
historical 19-of-20-batch Qwen result.

## Hosted model configuration

The endpoint, served model name, and credential are runtime configuration. For the
currently available SOOFI deployment when running from Neumann, set:

```bash
export SOOFI_BASE_URL='https://soofi-owu.l3s.de/api'
export SOOFI_MODEL='Soofi-S-RLVR-Isar'
```

Store the deployment settings in the repository-root `.env`, which is ignored by Git:

```bash
SOOFI_API_KEY='replace-with-your-issued-key'
SOOFI_BASE_URL='https://soofi-owu.l3s.de/api'
SOOFI_MODEL='Soofi-S-RLVR-Isar'
```

The client uses the operating system trust store and requires an HTTPS URL. It never
disables certificate validation. Do not add `curl -k` or an unverified HTTP client.

The hosted request contains `model`, `messages`, `temperature`, `max_tokens`, and:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

Provider-side `response_format` is disabled by default because the supplied SOOFI
example omits it. The final response is still parsed as strict JSON. Pass
`--json-response` only after confirming that the deployment supports OpenAI JSON
mode.

## Neumann setup

From the repository root:

```bash
uv sync --locked --extra dev
cp .env.example .env
chmod 600 .env
set -a
source .env
set +a
test -n "${SOOFI_API_KEY:-}"
```

The following audited inputs must exist below `artifacts/downloads/`:

```text
derived/val_qwen_top10_fulltext.jsonl
derived/sira_bm25_traces/
eval/val.csv
eval/laws_de.csv
eval/court_considerations.csv.gz
```

Their expected checksums are already recorded in `artifacts/manifest.toml`. The
runner rejects substituted or modified inputs.

## Validate, smoke-test, and resume

Validate all files and configuration without reading the API key or making a network
request:

```bash
uv run --locked python -m experiments.model_comparison.run_soofi --dry-run
```

Score the first five-candidate batch as a smoke test:

```bash
uv run --locked python -m experiments.model_comparison.run_soofi \
  --limit-batches 1 \
  --output-dir runs/model_comparison/soofi
```

Resume the same directory and finish all 20 sequential batches:

```bash
uv run --locked python -m experiments.model_comparison.run_soofi \
  --output-dir runs/model_comparison/soofi
```

Completed checkpoint batches are fingerprint-bound and skipped during a restart.
Changing the input, prompt-affecting generation settings, model, or endpoint requires
a new output directory.

To submit the CPU-only API client through Slurm:

```bash
sbatch scripts/run_soofi_hosted_api.sbatch
```

For a one-batch Slurm smoke test:

```bash
SOOFI_LIMIT_BATCHES=1 sbatch scripts/run_soofi_hosted_api.sbatch
```

No GPU is requested: inference occurs at the hosted endpoint.

## Outputs and audit rules

All mutable files are replaced atomically:

| File | Purpose |
| --- | --- |
| `batches.jsonl` | restart checkpoint with separate reasoning/final content, token counts, and latency |
| `scores.jsonl` | exactly validated candidate scores used by the gate |
| `manifest.json` | input, prompt, and semantic-configuration hashes plus output hashes |
| `status.json` | compact `dry_run`, `running`, `partial`, `scores_complete`, `complete`, or `failed` state |
| `sparse_support.jsonl` | deterministic replay of saved sparse traces |
| `submission.csv` | citation predictions in validation-query order |
| `selection_audit.json` | complete fixed-gate decision audit |
| `metrics.json` | fresh validation macro F1 |

The parser requires exactly one score for every requested candidate ID, numeric
scores in `[0, 10]`, numeric confidence values in `[0, 1]`, and a non-empty German
rationale. A partial run is never evaluated. A complete evaluation requires all 100
scores.

`reasoning_content` is never treated as the model's final answer. If a provider emits
a leading `<think>...</think>` block, it is stored separately and only the non-empty
suffix is parsed. Reasoning-only, malformed, or token-truncated replies are rejected
and retried.
