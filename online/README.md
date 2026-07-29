# Online pipeline

This folder contains a reviewable six-stage API for BetterCallAgent:

1. understand the question;
2. generate five retrieval views and byte-validate them against the asset;
3. replay versioned dense and BM25 rankings and combine them with weighted RRF;
4. independently rerank dense candidates;
5. apply the fixed citation-vote policy;
6. generate an answer and reject unsupported citations exactly.

The service exposes intermediate stage data as Server-Sent Events (SSE). It
never reads gold answers, never disables TLS verification, and never falls back
from a failed live provider to fixture output.

Stage 3 is an artifact-backed ranking replay for curated paper queries. It does
not claim to search a live vector or BM25 index. Each version-2 asset stores the
five exact retrieval views that produced its saved rankings. Stage 2 must
reproduce all five UTF-8 strings byte for byte before the rankings can be
replayed, so a changed model output cannot be presented as the cause of an
unrelated saved ranking.

## Quick start: fixture mode

The committed fixture is synthetic and runs end to end without a network
connection or secret.

```bash
uv sync --locked --extra online

cp online/.env.example online/.env
set -a
source online/.env
set +a

uv run --locked python -m uvicorn online.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/api/docs`, or list the exact reproducible input:

```bash
curl --fail http://127.0.0.1:8000/api/queries
```

Stream a run by copying the `query` and `query_id` returned above:

```bash
curl --no-buffer --fail \
  -H 'Content-Type: application/json' \
  --data '{
    "query_id": "demo-contract-delay",
    "query": "Ein Käufer hat einen verbindlichen Liefertermin vereinbart. Der Verkäufer liefert trotz Mahnung nicht. Unter welchen Voraussetzungen kann der Käufer vom Vertrag zurücktreten und Schadenersatz verlangen?",
    "model": "fixture-reviewer",
    "history": []
  }' \
  http://127.0.0.1:8000/api/runs/stream
```

## Live mode

Use a separate uncommitted environment file and a version-2 retrieval asset
with the same schema as `fixtures/demo.json`. A live run still uses only the
configured artifact-backed queries and saved rankings, making the evidence
presented to reviewers reproducible. If the live model does not regenerate the
asset's exact retrieval views, the run fails at stage 2 instead of replaying
mismatched evidence.

```bash
export BCA_ONLINE_MODE=live
export BCA_ASSET_PATH="$PWD/artifacts/downloads/online/versioned_asset.json"
export BCA_ALLOWED_MODELS=your-provider-model-id
export BCA_DEFAULT_MODEL=your-provider-model-id
export BCA_CORS_ORIGINS=http://localhost:5173
export BCA_LLM_BASE_URL=https://your-provider.example/v1
export BCA_LLM_API_KEY='read-from-your-secret-manager'

uv run --locked python -m uvicorn online.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

`BCA_ALLOWED_MODELS` is an explicit comma-separated allowlist. If bearer
authentication is required, set `BCA_AUTH_TOKEN` to a random token of at least
16 characters and send `Authorization: Bearer <token>`.

Recommended new-experiment candidates are DeepSeek V4 Flash (official local ID
`deepseek-ai/DeepSeek-V4-Flash`, provider example
`deepseek/deepseek-v4-flash`) and `Qwen/Qwen3.6-27B`. The German/English Soofi S
model is an early-access, deployment-specific option; `Soofi-S-Instruct` is only
a provider example, and no generally downloadable checkpoint was recorded as of
2026-07-29. Use the exact identifier advertised by your endpoint in
`BCA_ALLOWED_MODELS`. None of these alternatives produced the historical reported
values.

## API contract

- `GET /api/health` reports readiness, execution mode, and the number of
  documents materialized in the configured artifact as `artifact_documents`.
- `GET /api/models` returns the model allowlist.
- `GET /api/queries` lists reproducible, artifact-backed queries.
- `POST /api/runs/stream` emits `run_start`, six pairs of `step_start` and
  `step_complete`, `final_answer`, `run_complete`, and `stream_end`.

Errors inside a stream contain only a stable code and stage metadata. Detailed
exceptions remain in server logs and are never sent to the browser.
