# BetterCallAgent frontend

This folder contains the paper-facing user interface for BetterCallAgent. It is
a purpose-built React application that visualizes the pipeline's observable
operations:

1. question understanding;
2. multilingual query generation;
3. dense and sparse retrieval;
4. candidate reranking;
5. deterministic citation validation; and
6. grounded answer generation.

The interface displays queries, retrieved documents, scores, and citation
decisions. It does **not** request or reveal hidden chain-of-thought.

There is one frontend implementation and two explicit data sources:

- `live` connects to the FastAPI service;
- `fixture` replays deterministic, bundled demonstration data.

The application never switches from live mode to fixture mode automatically.
Configuration and protocol errors are shown clearly. The paper interface is
curated-only: it selects the first available query automatically, displays the
question read-only, and always sends the corresponding `query_id`.

## Requirements

- Node.js `20.19+`, `22.13+`, or `24+`
- npm `11+`

The exact dependency graph is recorded in `package-lock.json`.

## Quick start: deterministic fixture

Fixture mode is useful for reviewing the interface without models, credentials,
corpora, or a running backend.

```bash
cd frontend
cp .env.example .env
sed -i 's/VITE_DATA_SOURCE=live/VITE_DATA_SOURCE=fixture/' .env
npm ci
npm run dev
```

Open the URL printed by Vite, normally <http://localhost:5173>.

## Quick start: live pipeline

Start the BetterCallAgent API first. It must expose the endpoints documented
below. Then run:

```bash
cd frontend
cp .env.example .env
npm ci
npm run dev
```

The default example connects to `http://localhost:8000`. Change
`VITE_API_BASE_URL` in `.env` when the API uses another origin.

## Validate and build

These commands are safe to copy and paste:

```bash
cd frontend
npm ci
npm run check
npm run build
```

`npm run check` runs ESLint, strict TypeScript checking, and all Vitest tests.
The production assets are written to `dist/`.

To inspect the production build locally:

```bash
npm run preview
```

## Environment variables

| Variable | Required | Description |
|---|---:|---|
| `VITE_DATA_SOURCE` | yes | Exactly `live` or `fixture`. |
| `VITE_API_BASE_URL` | live mode | Absolute HTTP(S) base URL of the FastAPI service. |
| `VITE_ENABLE_DEBUG` | no | `true` shows normalized raw events; otherwise `false`. |
| `VITE_API_BEARER_TOKEN` | no | Development-only token for a protected local API. |

Every `VITE_*` value is embedded into the browser bundle. Do not place valuable
secrets in these variables. For a public protected deployment, put the frontend
and backend behind a same-origin authentication proxy instead.

## Live API contract

### Health

`GET /api/health`

```json
{
  "status": "ok",
  "ready": true,
  "mode": "fixture",
  "default_model": "fixture-deterministic",
  "artifact_documents": 2
}
```

### Models

`GET /api/models`

```json
{
  "default": "fixture-deterministic",
  "models": ["fixture-deterministic"]
}
```

The frontend does not hardcode live-provider availability. Recommended research
models such as DeepSeek V4 Flash, Qwen3.6-27B, and a configured German Soofi model
appear only when the backend reports their exact deployment identifiers.

### Curated queries

`GET /api/queries`

```json
{
  "queries": [
{
  "query_id": "fixture_detention",
  "query": "When may Swiss criminal authorities extend pre-trial detention because of a concrete risk of collusion?",
  "has_dense": true,
  "split": "fixture"
}
  ]
}
```

### Stream a run

`POST /api/runs/stream`

```json
{
  "query": "When may Swiss criminal authorities extend pre-trial detention because of a concrete risk of collusion?",
  "query_id": "fixture_detention",
  "model": "fixture-deterministic",
  "history": [
    {
      "role": "user",
      "content": "Optional prior question."
    },
    {
      "role": "assistant",
      "content": "Optional prior answer."
    }
  ]
}
```

The JSON examples above describe the frontend's built-in deterministic fixture.
Backend fixture mode has its own curated query and advertises it through
`GET /api/queries`; clients must always use the exact returned pair.

`query` and `query_id` must be the exact pair returned by `GET /api/queries`.
`history` is optional. The response media type must be `text/event-stream`. The
expected event order is:

```text
run_start
(step_start, step_complete) × 6
final_answer
run_complete
stream_end
```

Errors use a sanitized `error` event followed by `stream_end`. The parser
supports LF and CRLF delimiters, multiline `data:` fields, fragmented network
chunks, fragmented UTF-8 code points, and a final event without a trailing
blank line.

The complete TypeScript contract is in `src/domain/models.ts`; runtime
validation is in `src/domain/decoders.ts`.

## Evaluation metrics

The standard online contract does not return gold labels or evaluation metrics.
If an evaluation-only backend explicitly includes `metrics` on the
`final_answer` event, the interface labels and displays them. Metrics are never
invented, inferred, or loaded from bundled validation answers.

## Project map

```text
src/
├── api/
│   ├── config.ts                 # validated Vite environment
│   ├── data-source.ts            # shared live/fixture interface
│   ├── live-data-source.ts       # FastAPI + SSE client
│   ├── fixture-data-source.ts    # explicit offline demonstration
│   └── sse.ts                    # incremental event-stream parser
├── components/                   # accessible, domain-specific UI
├── domain/
│   ├── models.ts                 # typed API and pipeline models
│   └── decoders.ts               # runtime protocol validation
├── state/
│   └── run-reducer.ts            # deterministic event reducer
├── App.tsx
├── main.tsx
└── styles.css
```

Parser and reducer tests live beside the code they verify.

## Design decisions

- No Next.js, assistant-ui, Tailwind, shadcn, attachment system, or hidden
  runtime framework.
- One React state owner for the selected curated `query_id` and model; the
  question is derived from that selection and rendered read-only.
- One abort controller per run and unique client/server run identifiers.
- A typed reducer converts the streamed event sequence into visible state.
- Raw events are hidden unless `VITE_ENABLE_DEBUG=true`.
- Gold metrics are optional and evaluation-only.
- Responsive layouts, visible focus indicators, text status labels,
  `aria-live` updates, and correctly connected disclosure controls support
  keyboard and screen-reader review.
